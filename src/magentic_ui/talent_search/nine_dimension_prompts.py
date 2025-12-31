"""
Nine-Dimension Profile Extraction Prompts
每个维度独立提取，支持并行调用，零幻觉模式
"""
# ============================ 1. BACKGROUND  ============================

PROMPT_EXTRACT_BACKGROUND = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

Extract ONLY the **background/bio** of **{author_name}** from the **HOMEPAGE FULL TEXT** part of TEXT.
Ignore OpenReview/Scholar unless needed to confirm current position/affiliation.

TEXT:
{text}
==== END ====

### What to extract
Find the author's self-introduction/bio near the top of the homepage
(e.g., opening “I am…” paragraph, About/Bio/Introduction section).
Extract the FULL narrative bio (all relevant paragraphs).

Include:
- Self-intro, current role & affiliation
- Educational/professional background in narrative form
- Research focus/goals described in sentences
- Key achievements or highlights mentioned within the bio

Exclude:
- Pure bullet/list-style “research interests” enumerations
- Standalone advisor-name lines (only keep if embedded in a sentence)

### Output JSON (English only)
{{
  "name": "<full name or ''>",
  "position": "<current position or ''>",
  "affiliation": "<institution or ''>",
  "bio_text": "<complete bio/introduction with paragraphs separated by \\n\\n, or ''>",
  "career_stage": "<explicit stage or ''>"
}}

Rules:
- Extract bio_text completely if found (keep paragraph structure)
- Typical length: ~100–500 words, but do not truncate
- If a field is not explicitly stated, output ""
- If no bio found: bio_text = ""
- **CRITICAL: All output must be in English. If the source text is in Chinese or other languages, translate it to English.**
Return ONLY valid JSON.
"""

# ============================ 2. RESEARCH INTERESTS ============================

PROMPT_EXTRACT_RESEARCH_INTERESTS = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

Extract ONLY **Research Interests** of **{author_name}** that are **explicitly listed**
in dedicated interest/focus sections of TEXT (e.g., “Research Interests/Areas/Focus/Interests”).
Do NOT infer from publications, titles, abstracts, or keywords.

TEXT:
{text}
==== END ====

### What to extract
For each explicitly listed interest:
- Extract a **short topic name** (2–8 words, noun-phrase).  
  If the text is a long sentence, keep only the core field/topic.
- If there is a nearby explanation that clearly describes what/how/why the author studies
  this topic (often after a colon, parentheses, or the same/next line), put it in description.
  Otherwise set description="" (don’t guess).

### Output JSON (English only)
{{
  "interests": [
    {{
      "name": "",
      "description": ""
    }}
  ]
}}

Rules:
- Max 10 interests
- **CRITICAL: All output must be in English. Translate Chinese/other language content to English.**
- Keep wording verbatim (translate to English if needed)
- If no explicit interests section, return "interests": []
Return ONLY valid JSON. All fields must be in English.
"""


# ============================ 3. SELECTED RESEARCH (Publications) ============================

PROMPT_EXTRACT_PUBLICATIONS = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode. Extract ONLY the **Publications** from TEXT.

TARGET AUTHOR: {author_name}

DATA SOURCES:
The text contains FUSED DATA from 3 VERIFIED SOURCES:
1. Homepage - Publication lists with descriptions
2. OpenReview - Verified publication records with abstracts
3. Scholar - Publications with citation counts

Cross-validate papers that appear in multiple sources to ensure accuracy.
For citation counts: use Scholar data if available, otherwise use 0.
For venues: prioritize Homepage/OpenReview explicit statements over Scholar.

TEXT CONTENT:
{text}
==== END ====

EXTRACT publications from sections like "Publications", "Selected Papers", "Research", "Papers".

Return JSON:
{{
  "publications": [
    {{
      "title": "<paper title>",
      "venue": "<conference/journal name or ''>",
      "year": <int or null>,
      "authors": "<author list or ''>",
      "url": "<paper URL or ''>",
      "abstract": "<abstract if present, else ''>",
      "type": "Conference Paper|Journal Article|Preprint|Workshop"
    }}
  ]
}}

RULES:
1. Extract up to 20 most prominent papers
2. **Type mapping**:
   - If venue contains "arXiv" → "Preprint"
   - If venue is Nature/Science/PAMI/JMLR → "Journal Article"
   - If venue has "Workshop" → "Workshop"
   - Else → "Conference Paper"
3. Year must be an integer (e.g., 2024) or null
4. URL: copy exact URL from text, do NOT construct
5. If no publications section, return "publications": []
6. **CRITICAL - OUTPUT IN ENGLISH**: All titles, venues, abstracts, and other fields must be in English. Translate Chinese/other language content to English.

Return ONLY valid JSON. All output fields must be in ENGLISH.
"""

# ============================ 4. AWARDS & HONORS ============================

PROMPT_EXTRACT_AWARDS = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

Extract ONLY **Awards & Honors** of **{author_name}** from TEXT.
Ignore all other people.

TEXT:
{text}
==== END ====

### What counts as an award/honor
Include ONLY competitive recognitions explicitly indicating selection or honor, e.g.:
awards, prizes, scholarships, fellowships, grants, honors, distinctions,
outstanding/excellence titles, winner/runner-up/honorable mention.

Do NOT extract items that are ONLY about:
poster/oral presentation, invited talk, publication, acceptance, preprint,
camera-ready, "paper accepted", etc., unless paired with an award phrase
(e.g., "Best Poster Award").

### Award Types and Required Fields

1. **Paper Awards** (tied to specific paper):
   - MUST include: name, paper_title, venue, year
   - Example: Best Paper Award at ICLR 2024 for "Paper Title"
   - Fields: name="Best Paper Award", paper_title="...", venue="ICLR 2024", year="2024"

2. **Project Awards** (tied to specific project):
   - MUST include: name, project_name, organization, year
   - Example: Best Project Award for "Project Name"
   - Fields: name="Best Project Award", project_name="...", organization="...", year="..."

3. **Personal Awards** (not tied to paper/project):
   - MUST include: name, organization (if applicable), year
   - Example: NSF Fellowship, Outstanding Reviewer Award
   - Fields: name="NSF Fellowship", organization="NSF", year="2024"

### Output JSON (English only)
{{
  "awards": [
    {{
      "name": "<award name>",
      "year": "<year or ''>",
      "organization": "<organization or ''>",
      "paper_title": "<paper title if paper award, else ''>",
      "venue": "<venue if paper award, else ''>",
      "project_name": "<project name if project award, else ''>",
      "description": "<additional details or ''>",
      "award_type": "<'paper' | 'project' | 'personal'>"
    }}
  ]
}}

Rules:
- Order: most recent first
- For paper awards: paper_title and venue are REQUIRED
- For project awards: project_name is REQUIRED
- For personal awards: name is REQUIRED
- Max 50 awards total
- If none found: {{"awards": []}}
- **CRITICAL: All output must be in English. Translate Chinese/other language content to English.**
Return ONLY valid JSON. All fields must be in English.
"""


# ============================ 5. PROFESSIONAL SERVICES ============================

PROMPT_EXTRACT_SERVICES = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode. Extract ONLY **Professional Services** from TEXT.

TARGET AUTHOR: {author_name}

DATA SOURCES:
The text contains fused data from multiple sources (Homepage, OpenReview, Scholar).

**CRITICAL**: ONLY extract services from the **HOMEPAGE** section (SOURCE 1).
- DO NOT extract from OpenReview section
- DO NOT extract from Scholar section
- OpenReview reviewer records are NOT reliable service records

TEXT CONTENT:
{text}
==== END ====

EXTRACT from sections like "Service", "Professional Service", "Academic Activities", "Committee" **ONLY in the HOMEPAGE section**.

Return JSON:
{{
  "services": [
    {{
      "role": "<role, e.g., 'Area Chair', 'Reviewer', 'Program Committee'>",
      "venue": "<conference/journal/organization>",
      "year": "<year or ''>",
      "description": "<additional details or ''>"
    }}
  ],
  "invited_talks": [
    {{
      "title": "<talk title or topic>",
      "venue": "<venue/event>",
      "year": "<year or ''>",
      "description": "<additional context or ''>"
    }}
  ]
}}

RULES:
1. Separate service roles from invited talks
2. Service roles: PC, AC, OC, Reviewer, Editor, Organizer, Chair
3. Invited talks: marked as "invited", "keynote", "talk", "seminar"
4. Extract verbatim information
5. Maximum 30 services + 20 talks
6. If no service section, return empty lists
7. **CRITICAL - OUTPUT IN ENGLISH**: All roles, venues, titles, descriptions must be in English. Translate Chinese/other language content to English.

Return ONLY valid JSON. All output fields must be in ENGLISH.
"""

# ============================ 6. CAREER & EDUCATION HISTORY ============================
PROMPT_EXTRACT_CAREER_EDUCATION = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

TASK:
Extract ONLY the academic career & education history of **{author_name}** from TEXT.
Ignore all other people.

TEXT:
{text}
==== END ====

### Allowed Scope (academic only)
Include ONLY clearly academic / degree-training entries:
1. Student phases: Bachelor, Master, PhD/Doctoral, Postdoc
   - Postdoc must be explicitly stated as “Postdoc” or “Postdoctoral” at a university.
2. University faculty: Professor, Assistant Professor, Associate Professor, Lecturer (faculty role).
3. Teaching/training roles: Instructor, Teaching Assistant (TA), teaching-only Lecturer.

### Strict Exclusions
❌ Exclude ANY non-academic roles, especially:
- Any industry/company/corporate-lab position.
- Any title containing “Researcher/Scientist” (e.g., Research Scientist, Senior Scientist), even if university-affiliated.
- Research Fellow / Research Associate unless explicitly described as a university faculty-like role (not a lab / not industry).
- Anything not clearly academic or degree-training.

### Critical Anti-Hallucination Rules
- Extract ONLY what is explicitly stated in the TEXT.
- Do NOT invent, infer, merge, or construct degrees/positions not written.
- Do NOT assume someone pursued a higher degree unless explicitly stated.
- Do NOT copy/paraphrase this prompt in the output.

### Output Format (MUST follow exactly)
Return JSON only:
{{
  "career_education": [
    {{
      "degree_or_position": "",
      "institution": "",
      "department": "",
      "duration": "",
      "field": "",
      "advisor": "",
      "description": ""
    }}
  ]
}}

Rules:
- Output in **English** (translate non-English content).
- Max 30 entries.
- Order: most recent first. If duration missing, place those entries at the end.
- Duration: extract ONLY if explicitly stated; otherwise "" (no guessing).
- Still extract degree/position even if duration is missing.
- Institution: copy COMPLETE name exactly as shown (no shortening/modification).
- Department / field / advisor / description: fill ONLY if explicitly stated.
  * Advisor ONLY for student phases; otherwise "".
- If no valid entries, return `"career_education": []`
"""


# ============================ 7. INDUSTRIAL EXPERIENCE ============================

PROMPT_EXTRACT_EXPERIENCE = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

Extract ONLY **Industrial / Industry Experience** of **{author_name}** from TEXT.
Ignore all other people.

TEXT:
{text}
==== END ====

### Include (Industry)
Add any position held by {author_name} with titles such as:
- Researcher / Research Scientist / Senior/Principal/Staff Scientist
  *If title contains “Researcher” or “Scientist”, treat as industry even at research institutes.*
- Engineer / ML Engineer / Software Engineer / Data Scientist
- Intern / Research Intern (at companies)
- Consultant / Advisor (industry)

### Exclude (Academic / Training)
Do NOT include:
- Professor / Associate/Assistant Professor / Lecturer / Faculty roles
- Bachelor/Master/PhD student roles
- Postdoc / Postdoctoral roles

### Output JSON (English only)
{{
  "experiences": [
    {{
      "position": "",
      "company": "",
      "duration": "",
      "description": ""
    }}
  ]
}}

Rules:
- **CRITICAL: All output must be in English. Translate Chinese/other language content to English.**
- Order: most recent first
- Extract description verbatim if present, else ""
- Max 15 entries
- If none, return "experiences": []
Return ONLY valid JSON. All fields must be in English.
"""

# ============================ 8. CONTACT ============================
PROMPT_EXTRACT_CONTACT = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode.

Extract ONLY **verbatim contact information** of **{author_name}** from TEXT.
Do NOT infer or construct anything. Use ONLY what appears literally in TEXT.

TEXT:
{text}
==== END ====

### What to extract
Extract contact info exactly as written:
- Email (must be a personal/academic email; exclude generic info/admin/support)
- Phone number
- Office location
- Social/profile links (URLs must appear verbatim), including:
  homepage, google scholar, github, linkedin, twitter/x, orcid, dblp, openreview

### Output JSON (English only)
{{
  "email": "",
  "phone": "",
  "office": "",
  "social_links": {{
    "homepage": "",
    "google_scholar": "",
    "github": "",
    "linkedin": "",
    "twitter": "",
    "orcid": "",
    "dblp": "",
    "openreview": ""
  }}
}}

Rules:
- Copy URLs exactly as shown (preserve http/https)
- If a platform is mentioned without a visible URL, leave it as ""
- Do NOT guess or derive URLs, IDs, or usernames
- If no contact info exists, return empty strings
Return ONLY valid JSON.
"""

# ============================ FALLBACK: COMPREHENSIVE EXTRACTION ============================

PROMPT_COMPREHENSIVE_FALLBACK = lambda author_name, text: f"""
You are in ZERO-HALLUCINATION mode. Extract ALL available information from TEXT for {author_name}.

TEXT CONTENT:
{text}
==== END ====

This is a FALLBACK extraction when section-specific prompts fail.
Extract ANY available information and organize into relevant fields.

Return JSON with ALL fields (use "" or [] if not found):
{{
  "position": "",
  "affiliation": "",
  "bio": "",
  "research_interests": [],
  "publications": [],
  "awards": [],
  "education": [],
  "experience": [],
  "teaching": [],
  "email": "",
  "social_links": {{}}
}}

Extract everything you can find, but still follow ZERO-HALLUCINATION rule.

**OUTPUT IN ENGLISH**: All extracted content must be in English. Translate Chinese/other languages.

Return ONLY valid JSON. All output fields must be in ENGLISH.
"""

