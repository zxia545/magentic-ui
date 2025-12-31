from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set
from urllib.parse import urlparse

from . import config, search
from .arxiv_fallback import ArxivFallbackClient
from .semantic_paper_search import SemanticScholarClient
from .utils import sanitize_author_ids

_TITLE_SPLIT_RE = re.compile(r"\s+[-|–|—]\s+")
_TITLE_SUFFIXES = {
    "arxiv",
    "openreview",
    "semantic scholar",
    "proceedings",
    "acm digital library",
    "ieee xplore",
    "springer",
    "pdf",
}

_ACADEMIC_DOMAINS = {
    "arxiv.org",
    "openreview.net",
    "aclanthology.org",
    "proceedings.mlr.press",
    "papers.nips.cc",
    "openaccess.thecvf.com",
    "ieee.org",
    "dl.acm.org",
    "acm.org",
    "springer.com",
    "link.springer.com",
    "sciencedirect.com",
    "nature.com",
    "science.org",
    "jmlr.org",
    "usenix.org",
    "aaai.org",
    "ijcai.org",
    "aclweb.org",
}

_BLOCKED_DOMAINS = {
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "vimeo.com",
    "wps.com",
    "wps.cn",
}

_STOPWORD_TOKENS = {
    "paper",
    "papers",
    "study",
    "research",
    "analysis",
    "system",
    "method",
    "approach",
    "model",
    "models",
    "learning",
    "deep",
    "neural",
    "framework",
    "survey",
    "review",
    "arxiv",
    "conference",
    "journal",
    "proceedings",
}


def _domain_from_url(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except Exception:
        return ""


def _build_keyword_tokens(keywords: Optional[List[str]]) -> List[str]:
    tokens: List[str] = []
    for kw in keywords or []:
        for token in re.findall(r"[a-z0-9]+", kw.lower()):
            if token in _STOPWORD_TOKENS:
                continue
            if len(token) < 3:
                continue
            tokens.append(token)
    return tokens


def _matches_keywords(text: str, keywords: Optional[List[str]]) -> bool:
    if not keywords:
        return True
    text_lc = text.lower()
    for kw in keywords:
        kw_lc = kw.lower().strip()
        if kw_lc and kw_lc in text_lc:
            return True
    for token in _build_keyword_tokens(keywords):
        if token in text_lc:
            return True
    return False


def _should_consider_result(item: Dict[str, Any], keywords: Optional[List[str]]) -> bool:
    title = (item.get("title") or "").strip()
    url = (item.get("url") or "").strip()
    snippet = (item.get("snippet") or "").strip()
    if not title or not url:
        return False
    domain = _domain_from_url(url)
    if domain and any(domain.endswith(b) for b in _BLOCKED_DOMAINS):
        return False
    text = f"{title} {snippet}".strip().lower()
    if domain and any(domain.endswith(d) for d in _ACADEMIC_DOMAINS):
        return True
    if _matches_keywords(text, keywords):
        return True
    title_tokens = re.findall(r"[a-z0-9]+", title.lower())
    if len(title_tokens) <= 1:
        return False
    return False


def _normalize_candidate_title(raw_title: str) -> str:
    title = raw_title.strip()
    title = re.sub(r"\s*\[pdf\]\s*", "", title, flags=re.IGNORECASE)
    if not title:
        return ""
    parts = _TITLE_SPLIT_RE.split(title)
    if parts:
        title = parts[0].strip()
    lower = title.lower()
    for suffix in _TITLE_SUFFIXES:
        if lower.endswith(f" {suffix}"):
            title = title[: -len(suffix)].strip()
            break
        if lower == suffix:
            title = ""
            break
    return title.strip()


def build_bing_query(
    keywords: List[str],
    years: Optional[List[int]] = None,
    venues: Optional[List[str]] = None,
) -> str:
    parts: List[str] = []
    if keywords:
        parts.extend(keywords)
    parts.append("paper")
    if venues:
        parts.extend(venues[:2])
    if years:
        parts.extend(str(y) for y in years[:2])
    return " ".join(parts).strip()


def _paper_from_details(
    details: Dict[str, Any],
    query: str,
    fallback_url: str = "",
) -> Dict[str, Any]:
    author_ids = sanitize_author_ids(details.get("author_ids", []))
    url = details.get("url") or fallback_url
    return {
        "title": details.get("title", ""),
        "authors": details.get("authors", []),
        "author_ids": author_ids,
        "venue": details.get("venue", "") or "",
        "year": details.get("year"),
        "paper_id": details.get("paper_id", "") or details.get("paperId", ""),
        "topic": query,
        "source": details.get("data_source", "bing"),
        "url": url,
        "abstract": details.get("abstract", "") or "",
        "introduction": details.get("tldr", "") or "",
        "pdf_url": details.get("pdf_url", "") or "",
    }


def _paper_from_s2_search_item(item: Dict[str, Any], query: str) -> Dict[str, Any]:
    authors_data = item.get("authors", []) or []
    authors = [a.get("name", "") for a in authors_data if a.get("name")]
    author_ids = sanitize_author_ids(
        [a.get("authorId") if isinstance(a, dict) else None for a in authors_data]
    )
    tldr_obj = item.get("tldr") or {}
    tldr_text = ""
    if isinstance(tldr_obj, dict):
        tldr_text = tldr_obj.get("text") or ""
    elif isinstance(tldr_obj, str):
        tldr_text = tldr_obj or ""
    external_ids = item.get("externalIds", {}) or {}
    arxiv_id = external_ids.get("ArXiv", "")
    open_access_pdf = item.get("openAccessPdf", {}) or {}
    pdf_url = ""
    if isinstance(open_access_pdf, dict):
        pdf_url = open_access_pdf.get("url", "") or ""
    if not pdf_url and arxiv_id:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    return {
        "title": item.get("title", ""),
        "authors": authors,
        "author_ids": author_ids,
        "venue": item.get("venue", "") or "",
        "year": item.get("year"),
        "paper_id": item.get("paperId", "") or "",
        "topic": query,
        "source": "semantic_scholar_search",
        "url": item.get("url", "") or "",
        "abstract": item.get("abstract", "") or "",
        "introduction": tldr_text,
        "pdf_url": pdf_url,
    }


def _passes_year_venue(
    item: Dict[str, Any],
    years: Optional[List[int]],
    venues: Optional[List[str]],
) -> bool:
    if years:
        year = item.get("year")
        try:
            if year is not None and int(year) not in years:
                return False
        except Exception:
            return False
    if venues:
        venue = (item.get("venue") or "").lower()
        if venue:
            if not any(v.lower() in venue for v in venues if v):
                return False
    return True


def search_papers_bing(
    query: str,
    k_per_query: int = 10,
    *,
    keywords: Optional[List[str]] = None,
    years: Optional[List[int]] = None,
    venues: Optional[List[str]] = None,
    seen_titles: Optional[Set[str]] = None,
    min_match_score: Optional[float] = None,
) -> List[Dict[str, Any]]:
    if not query:
        return []
    seen_titles = {t.lower().strip() for t in (seen_titles or set()) if t}
    min_match = min_match_score or config.BING_PAPER_MIN_MATCH_SCORE
    max_results = max(k_per_query * 3, config.BING_PAPER_MAX_RESULTS)
    results = search.searxng_search(query, pages=1, k_per_query=max_results)
    if not results and config.VERBOSE:
        print(f"[Bing Search] No Bing results for '{query}', fallback to Semantic Scholar search")

    s2_client = SemanticScholarClient(
        api_key=config.SEMANTIC_SCHOLAR_API_KEY or None,
        timeout=12.0,
        requests_per_second=1.0,
    )
    arxiv_client = ArxivFallbackClient()
    papers: List[Dict[str, Any]] = []

    if results:
        for item in results:
            if not _should_consider_result(item, keywords):
                continue
            raw_title = (item.get("title") or "").strip()
            candidate_title = _normalize_candidate_title(raw_title)
            if not candidate_title:
                continue
            title_key = candidate_title.lower().strip()
            if title_key in seen_titles:
                continue

            details = s2_client.get_paper_full_details(
                title=candidate_title,
                year=None,
                venue=None,
                min_match_score=min_match,
            )
            if not details:
                details = arxiv_client.search_by_title(candidate_title, max_results=3)
            if not details:
                continue

            paper = _paper_from_details(details, query=query, fallback_url=item.get("url", ""))
            if not _passes_year_venue(paper, years, venues):
                continue
            papers.append(paper)
            seen_titles.add(title_key)
            if len(papers) >= k_per_query:
                break

    if papers:
        return papers

    if config.VERBOSE:
        print(f"[Bing Search] No valid papers from Bing for '{query}', fallback to Semantic Scholar search")

    fallback_query = " ".join(keywords or []) or query
    s2_items = s2_client.search_papers(
        fallback_query,
        limit=max_results,
    )
    for item in s2_items:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        title_key = title.lower().strip()
        if title_key in seen_titles:
            continue
        if not _passes_year_venue(item, years, venues):
            continue
        paper = _paper_from_s2_search_item(item, query=fallback_query)
        if keywords and not _matches_keywords(
            f"{paper.get('title', '')} {paper.get('abstract', '')}", keywords
        ):
            continue
        papers.append(paper)
        seen_titles.add(title_key)
        if len(papers) >= k_per_query:
            break

    return papers
