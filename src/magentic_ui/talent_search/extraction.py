"""
Extraction and parsing functions for Talent Search System
Handles author extraction, candidate filtering, and profile validation
"""

from typing import List, Dict, Any
from . import config
from . import utils
from . import schemas
from . import llm
from . import search

def extract_paper_name_from_sources(source: Any, spec: schemas.QuerySpec, api_key: str = None) -> Dict[str, str]:
    """Extract a single paper name from source texts using LLM"""
    llm_instance = llm.get_llm("paper_name", temperature=0.4, api_key=api_key)

    # Build topic + venue hint
    topic_hint = ", ".join(spec.keywords) if spec.keywords else "the relevant topics"
    venue_hint = f" Restrict to papers from {', '.join(spec.venues)} if possible." if spec.venues else ""

    blobs = []
    url, content_text = source
    if len(content_text) < 50:  # skip very short snippets
        return {"have_paper_name": False, "paper_name": ""}
    seg = f"[Source] {url}\n{content_text[:8000]}\n"
    blobs.append(seg)

    if not blobs:
        return {"have_paper_name": False, "paper_name": ""}

    dump = "\n\n".join(blobs)

    prompt = (
        "You extract EXACTLY ONE academic paper title if present; otherwise return false.\n\n"
        "SCOPE:\n"
        "- 'Academic paper title' = title of a research article (conference/journal/arXiv/workshop). "
        "Not a page header, profile, CFP, submission guide, program/schedule, or an entire proceedings volume.\n"
        "- Upstream sections may appear: 'SNIPPET:' (search-engine preview text), 'TITLE:' (page main title), "
        "'BODY:' (main content), 'SOURCE:' (URL).\n\n"
        f"TASK: Choose one title most relevant to {topic_hint}. {venue_hint}\n"
        "If none is high-confidence: {{\"have_paper_name\": false, \"paper_name\": \"\"}}.\n"
        "If found: copy the title EXACTLY as written (preserve casing/punctuation; trim surrounding quotes) as "
        "{{\"have_paper_name\": true, \"paper_name\": \"<title>\"}}.\n\n"
        "PRIORITIES:\n"
        "1) A plausible line under 'TITLE:'.\n"
        "2) A heading in BODY near cues like 'Abstract', 'Authors', 'PDF', 'DOI', 'arXiv'; length 5–300 chars; not a section label.\n"
        "3) Candidate repeated verbatim or matching the URL slug.\n"
        "4) Hosts typical of article pages: arxiv.org/abs, dl.acm.org/doi, openreview.net/forum, "
        "ieeexplore.ieee.org/document, nature.com/articles, sciencedirect.com/science/article, springer.com/article.\n\n"
        "NEGATIVE FILTERS:\n"
        "- 'Call for Papers/CFP', 'Submission Guidelines', 'Program/Schedule', 'Proceedings of …' (whole volume), "
        "'Home/About/Profile/CV'.\n"
        "- Section names only: 'Abstract', 'Introduction', 'Related Work', 'Methods', 'Results', 'Conclusion', "
        "'Acknowledgments', 'References', 'Appendix'.\n"
        "- Pure venue/year like 'NeurIPS 2024', 'ACL 2023 Main Conference' without a distinct article title.\n\n"
        "CONSERVATIVE RULES:\n"
        "- Do not invent/paraphrase/complete truncated strings. Keep original casing/punctuation.\n"
        "- Reject ALL-CAPS unless clearly an acronym-based title (e.g., 'BERT: …').\n"
        "- If the page is mainly a list of many references or is ambiguous which item is the page's article, return false.\n\n"
        "OUTPUT (STRICT): Return only one JSON object with exactly keys 'have_paper_name' (boolean) and 'paper_name' (string). "
        "No extra text or markdown.\n\n"
        "Mini-examples:\n"
        "  TITLE: Learning with Noisy Labels -> {\"have_paper_name\": true, \"paper_name\": \"Learning with Noisy Labels\"}\n"
        "  BODY: Call for Papers … -> {\"have_paper_name\": false, \"paper_name\": \"\"}\n\n"
        "Analyze the sources below and respond in STRICT JSON: {\"have_paper_name\": true|false, \"paper_name\": \"<title>\"}\n\n"
        f"{dump}\n==== END ===="
    )



    alist = llm.safe_structured(llm_instance, prompt, schemas.LLMPaperNameSpec)
    if config.VERBOSE:
        print(f"[extract_paper_name] Have paper name: {alist.have_paper_name}, extracted paper: {alist.paper_name}")

    return alist
    
    


# ============================ AUTHOR EXTRACTION FUNCTIONS ============================

def extract_authors_from_sources(sources: Dict[str, str], spec: schemas.QuerySpec,
                                max_chars: int = 140_000) -> List[str]:
    """Extract authors from source texts using LLM"""
    llm_instance = llm.get_llm("authors", temperature=0.4)
    topic_hint = ", ".join(spec.keywords) if spec.keywords else "the target topics mentioned in the request"

    blobs = []
    total = 0
    for u, t in sources.items():
        # Skip profile pages
        if _looks_like_profile_url(u):
            continue
        if len(t) < 600:
            continue
        seg = f"[Source] {u}\n{t[:14000]}\n"
        if total + len(seg) > max_chars:
            break
        total += len(seg)
        blobs.append(seg)

    if not blobs:
        return []

    dump = "\n\n".join(blobs)
    prompt = (
        f"From the sources below, extract up to 25 author names working on {topic_hint}. "
        f"Prioritize {', '.join(spec.author_priority) if spec.author_priority else 'first, last'} authors. "
        "Return STRICT JSON {\"authors\": [\"Name1\", ...]}.\n\n"
        f"{dump}\n==== END ===="
    )

    alist = llm.safe_structured(llm_instance, prompt, schemas.AuthorListSpec)
    if config.VERBOSE:
        print(f"[extract_authors] extracted authors: {alist.authors}")

    return alist.authors

# ============================ CANDIDATE FILTERING FUNCTIONS ============================

def postfilter_candidates(cands: List[Dict[str, Any]], must_be_student: bool = True) -> List[Dict[str, Any]]:
    """Filter candidates based on criteria"""
    out = []
    for c in cands:
        role = (c.get("Current Role & Affiliation") or "")
        notes = (c.get("Evidence Notes") or "")
        profs = c.get("Profiles") or {}

        if must_be_student and not (utils.looks_like_student(role) or utils.looks_like_student(notes)):
            continue
        if not any(utils.is_valid_profile_url(v) for v in profs.values() if v):
            continue
        out.append(c)
    return out

# ============================ CANDIDATE SYNTHESIS FUNCTIONS ============================

def synthesize_candidates(sources: Dict[str, str], spec: schemas.QuerySpec, max_sources: int = 20) -> Dict[str, Any]:
    """Synthesize candidate information from sources"""
    llm_instance = llm.get_llm("synthesize", temperature=0.6)

    # Choose top sources by content length
    items = list(sources.items())
    items.sort(key=lambda kv: len(kv[1]), reverse=True)
    src = dict(items[:max_sources])

    if not src:
        if VERBOSE:
            print("[synthesize] no sources yet")
        return {
            "candidates": [],
            "citations": [],
            "need_more": True,
            "followups": ["Need more accepted/program/proceedings pages and author profiles."]
        }

    parts = []
    total = 0
    limit = 240_000
    for u, t in src.items():
        t = t[:16_000]
        blob = f"[Source] {u}\n{t}\n"
        if total + len(blob) > limit:
            break
        total += len(blob)
        parts.append(blob)

    research_dump = "\n\n".join(parts)

    # Build filtering rules
    degree_hint = ", ".join(spec.degree_levels) if spec.degree_levels else "PhD/Master students"
    must_line = "They MUST be current students." if spec.must_be_current_student else "They CAN be current students or recent graduates."
    topic_hint = ", ".join(spec.keywords) if spec.keywords else "the target topics mentioned in the request"

    prompt = (
        "You are an HR talent scout. Using ONLY the sources below, extract up to "
        f"{spec.top_n} candidates who are {degree_hint} working on {topic_hint}. {must_line}\n"
        "For each candidate, return a JSON object with EXACT keys:\n"
        ' - "Name"\n'
        ' - "Current Role & Affiliation" (e.g., "PhD Student, UC Berkeley EECS")\n'
        ' - "Research Focus" (array of short keywords)\n'
        ' - "Profiles" (object with any of: "Homepage", "Google Scholar", "Twitter", "OpenReview", "GitHub", "LinkedIn")\n'
        ' - "Notable" (1 short line, optional)\n'
        ' - "Evidence Notes" (where you saw the info)\n\n'
        "STRICT rules:\n"
        "- Only include candidates if the student-status is stated or strongly implied when required.\n"
        "- Do NOT fabricate links; only include profile links that appear in sources.\n"
        "- Deduplicate the same person across multiple links.\n\n"
        "Return STRICT JSON:\n"
        "{\n"
        '  "candidates": [ { ... }, ... ],\n'
        '  "citations": ["<used source url>", ...],\n'
        '  "need_more": false,\n'
        '  "followups": []\n'
        "}\n\n"
        "==== SOURCES (url -> text) ====\n"
        f"{research_dump}\n==== END SOURCES ====\n"
    )

    syn = llm.safe_structured(llm_instance, prompt, schemas.CandidatesSpec)

    candidates_json = [cc.model_dump(by_alias=True) for cc in syn.candidates]
    candidates_json = postfilter_candidates(candidates_json, must_be_student=spec.must_be_current_student)

    if config.VERBOSE:
        print(f"[synthesize] candidates(after filter)={len(candidates_json)} need_more={syn.need_more}")

    return {
        "candidates": candidates_json,
        "citations": syn.citations or [],
        "need_more": bool(syn.need_more),
        "followups": syn.followups or [],
    }

# ============================ HELPER FUNCTIONS ============================

def _looks_like_profile_url(u: str) -> bool:
    """Check if URL looks like a profile page (internal helper)"""
    return utils.looks_like_profile_url(u)
