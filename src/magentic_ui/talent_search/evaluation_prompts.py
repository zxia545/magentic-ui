"""
Four-Dimensional Candidate Evaluation Prompts
Each dimension is evaluated independently on a 1-5 scale.
Total score: sum of 4 dimensions (max 20 points)
"""

# ============================ DIMENSION 1: TOPIC MATCH ============================

PROMPT_TOPIC_MATCH = """
EVALUATION TASK: Assess Topic Alignment

OBJECTIVE:
Evaluate how well the candidate's research topics align with the user's query/research area.

INPUTS:
User Query: {user_query}

Candidate Profile:
- Name: {candidate_name}
- Research Interests: {research_interests}
- Research Keywords: {research_keywords}
- Research Focus: {research_focus}
- Representative Papers: {representative_papers}

SCORING CRITERIA (1-5):

**Score 5 - Perfect Match (Core Focus)**
- The candidate's PRIMARY research direction is EXACTLY what the user is looking for
- Multiple research interests/keywords directly match the query topic
- Representative papers are centered on this topic
- This is clearly their main expertise area
- Example: Query "graph neural networks" → Candidate's main focus is "graph neural networks for molecular property prediction"

**Score 4 - Strong Match (Major Area)**
- The query topic is ONE OF the candidate's main research areas
- Several keywords/interests align well with the query
- Has multiple papers in this area
- Shows deep expertise in related subfields
- Example: Query "graph neural networks" → Candidate works on "graph learning and geometric deep learning"

**Score 3 - Moderate Match (Related Work)**
- The candidate has done SOME work in the query area
- Shows familiarity but not primary focus
- May have 1-2 papers touching this topic
- Research interests show partial overlap
- Example: Query "graph neural networks" → Candidate works on "deep learning for structured data" with some graph work

**Score 2 - Weak Match (Peripheral)**
- The candidate's work is in a BROADLY RELATED field
- Very few specific overlaps with query topic
- Might have used related methods as tools but not core focus
- Tangential connection only
- Example: Query "graph neural networks" → Candidate works on "computer vision" but used GNNs once

**Score 1 - No Match (Unrelated)**
- The candidate's research is in a COMPLETELY DIFFERENT area
- No meaningful overlap with query topic
- Different problem domains, methods, and goals
- Example: Query "graph neural networks" → Candidate works purely on "natural language processing" with no graph work

CRITICAL RULES:
1. Focus on ACTUAL research content, not just similar-sounding terms
2. Consider research interests, keywords, and actual papers together
3. Distinguish between "main focus" vs "occasionally used"
4. Be strict: Topic alignment is the foundation of relevance
5. **IMPORTANT: Use the EXACT user query as provided. DO NOT add, modify, or expand keywords from the user query.**
   - If user query is "culture llm alignment", use exactly that - do NOT add "Artificial Intelligence" or other terms
   - Quote the user query exactly as shown in INPUTS when referencing it in your explanation
   - Do not infer or append related terms even if they seem relevant

OUTPUT FORMAT (JSON only):
{{
  "score": <1-5>,
  "explanation": "<Detailed analysis: What topics match? How central are they to the candidate's work? Evidence from papers/interests? Why this score? When quoting the user query, use it EXACTLY as provided without modifications.>"
}}
"""

# ============================ DIMENSION 2: VENUE FIT & EVIDENCE STRENGTH ============================

PROMPT_VENUE_FIT = """
EVALUATION TASK: Assess Publication Quality and Venue Fit

OBJECTIVE:
Evaluate the candidate's publication quality (top-tier venues) and whether they publish in the target community's premier conferences/journals.

INPUTS:
User Query: {user_query}
Target Research Field: {research_field}
Query Venues: {query_venues}

Candidate Profile:
- Name: {candidate_name}
- Representative Papers: {representative_papers}
- Publication Overview: {publication_overview}
- Top-tier Hits (Last 24 Months): {top_tier_hits}

VENUE TIER REFERENCE:
**Machine Learning:** NeurIPS, ICML, ICLR
**Computer Vision:** CVPR, ICCV, ECCV
**Natural Language Processing:** ACL, EMNLP, NAACL
**Artificial Intelligence:** AAAI, IJCAI
**Data Mining:** KDD, WWW, WSDM
**Robotics:** ICRA, IROS, RSS, CoRL
**High-Impact Journals:** Nature, Science, PAMI, JMLR

SCORING CRITERIA (1-5):

**Score 5 - Exceptional Quality & Perfect Fit**
- Has MULTIPLE papers at TOP-TIER venues in the target field (e.g., multiple NeurIPS/ICLR papers for ML)
- Papers published at the EXACT venues user cares about (mentioned in query)
- Recent publications (last 1-2 years) at these premier venues
- Demonstrates consistent high-quality output
- Example: Query about ML → Candidate has 3+ NeurIPS/ICML/ICLR papers in last 2 years

**Score 4 - Strong Quality & Good Fit**
- Has SEVERAL papers at top-tier or strong second-tier venues
- At least 1-2 papers at premier venues in the target community
- Mostly publishes in relevant conferences
- Good recent track record
- Example: Query about CV → Has 2 CVPR papers + several ICCV/ECCV papers

**Score 3 - Moderate Quality & Reasonable Fit**
- Has SOME papers at recognized venues in the field
- Mix of top-tier and second-tier venues
- May have 1 top-tier paper but not consistent
- Publications are relevant but not all in premier venues
- Example: Query about NLP → Has 1 ACL paper + several workshop/smaller conference papers

**Score 2 - Limited Quality & Weak Fit**
- MOSTLY publishes at lower-tier venues or workshops
- Very few (or no) papers at recognized conferences in the target field
- May have arxiv-only papers
- Publication venues don't match the target community
- Example: Query about ML → Mostly regional conferences, no top-tier ML venues

**Score 1 - Poor Quality & No Fit**
- NO publications at recognized venues
- Only arxiv preprints or very low-tier venues
- Publishes in completely different communities
- No evidence of academic rigor
- Example: Query about top-tier ML → No conference papers, only arxiv or blogs

CRITICAL RULES:
1. TOP-TIER venues > quantity: 1 NeurIPS paper > 5 workshop papers
2. RECENCY matters: Recent top-tier > old top-tier
3. COMMUNITY FIT: Publishing in the RIGHT venues shows deep integration
4. Consider both paper count AND venue prestige

OUTPUT FORMAT (JSON only):
{{
  "score": <1-5>,
  "explanation": "<Detailed analysis: What venues? How many top-tier papers? How recent? Do venues match the target community? Why this score?>"
}}
"""

# ============================ DIMENSION 3: RECENCY & MOMENTUM ============================

PROMPT_RECENCY_MOMENTUM = """
EVALUATION TASK: Assess Research Recency and Momentum IN THE QUERY DOMAIN

OBJECTIVE:
Evaluate whether the candidate's work in THE QUERY-SPECIFIED AREA is recent and shows sustained momentum, not just any recent work.

INPUTS:
User Query: {user_query}

Candidate Profile:
- Name: {candidate_name}
- Representative Papers: {representative_papers}
- Top-tier Hits (Last 24 Months): {top_tier_hits}
- Research Keywords: {research_keywords}
- Highlights: {highlights}

SCORING CRITERIA (1-5):

**Score 5 - Active Leader (High Momentum in Query Area)**
- Has published MULTIPLE papers in the query topic in the LAST 12 MONTHS
- Shows ACCELERATING output: more recent papers than older ones
- Latest work (2024-2025) is directly on the query topic
- Clear upward trajectory and sustained focus on this area
- Example: Query "diffusion models" → 3+ diffusion papers in 2024-2025, including very recent ones

**Score 4 - Active Contributor (Good Momentum in Query Area)**
- Has published SEVERAL papers in the query topic in the LAST 18-24 MONTHS
- Consistent output: regular publications in this area
- Most recent work (2023-2024) includes query-relevant papers
- Demonstrates sustained interest
- Example: Query "graph learning" → 2-3 graph papers per year in 2023-2024

**Score 3 - Moderate Activity (Some Recent Work in Query Area)**
- Has published SOME papers in the query topic in the LAST 24-36 MONTHS
- Not very frequent but shows continued interest
- At least 1 recent paper (2023+) related to query topic
- May show declining momentum
- Example: Query "transformers" → 1 transformer paper in 2023, 1-2 in 2022

**Score 2 - Low Activity (Older Work in Query Area)**
- Most work in the query topic is FROM 3-5 YEARS AGO
- Very few or no recent papers in this area
- May have SHIFTED to other topics recently
- Example: Query "GANs" → Last GAN paper was in 2021, now working on other things

**Score 1 - No Recent Activity (Stale or Inactive in Query Area)**
- NO work in the query topic in the LAST 3+ YEARS
- All relevant work is OLD (pre-2021)
- Has clearly MOVED ON from this research area
- Or has very few publications overall
- Example: Query "reinforcement learning" → Last RL paper was in 2019

CRITICAL RULES:
1. DOMAIN-SPECIFIC: Only count papers/work RELATED TO THE QUERY TOPIC
   - If query is "diffusion models", ignore recent "transformer" papers
   - If query is "graph learning", ignore recent "vision" papers
2. RECENCY is key: 2024-2025 work > 2022-2023 work > pre-2022 work
3. MOMENTUM means: consistent/increasing output over time in THIS area
4. Recent shift AWAY from query topic = lower score
5. Be STRICT: General recent activity doesn't count if not in query domain
6. **IMPORTANT: Use the EXACT user query as provided. DO NOT add, modify, or expand keywords from the user query when referencing it.**

OUTPUT FORMAT (JSON only):
{{
  "score": <1-5>,
  "explanation": "<Detailed analysis: When were query-relevant papers published? How many in last 1/2/3 years? Is there sustained momentum or declining interest in THIS topic? Evidence from papers/highlights? Why this score? When quoting the user query, use it EXACTLY as provided without modifications.>"
}}
"""

# ============================ DIMENSION 4: ROLE LEADERSHIP FIT ============================

PROMPT_ROLE_LEADERSHIP = """
EVALUATION TASK: Assess Leadership and Independent Research Capability

OBJECTIVE:
Evaluate whether the candidate plays a LEADING role (first author, corresponding author, organizer) vs. just participating in research.

INPUTS:
User Query: {user_query}
Author Priority from Query: {author_priority}

Candidate Profile:
- Name: {candidate_name}
- Representative Papers: {representative_papers}
- Publication Overview: {publication_overview}
- Academic Service / Invited Talks: {service_talks}
- Honors/Grants: {honors_grants}

SCORING CRITERIA (1-5):

**Score 5 - Clear Leader (Proven Independence)**
- MAJORITY of papers are as FIRST AUTHOR or CORRESPONDING AUTHOR
- Has papers where they are the SOLE or PRIMARY contributor
- Shows INDEPENDENT research direction and vision
- May have leadership roles: organizer, area chair, invited speaker
- Example: PhD student with 5+ first-author papers at top venues

**Score 4 - Strong Contributor (Significant Leadership)**
- MANY papers as first or corresponding author
- Mix of leading and collaborative roles
- Clear evidence of driving research projects
- Some service/organization roles
- Example: Researcher with 50%+ first-author papers + some area chair roles

**Score 3 - Active Participant (Some Leadership)**
- SOME papers as first author (but not majority)
- Often appears as middle author
- Shows capability but less independent leadership
- Limited evidence of driving research agenda
- Example: Postdoc with 30-40% first-author papers

**Score 2 - Supporting Role (Limited Leadership)**
- RARELY appears as first author
- Mostly middle or last author positions
- Appears to support others' research rather than lead
- No clear independent research direction
- Example: Junior researcher mostly as 3rd-5th author

**Score 1 - Follower (No Leadership Evidence)**
- NEVER or almost never as first author
- Always in supporting positions (middle author)
- No evidence of independent research capability
- No service/leadership activities
- Example: Student always appearing as 4th+ author

ADDITIONAL LEADERSHIP INDICATORS (Boost score if present):
- Program Committee / Area Chair roles: +0.5
- Invited talks / keynotes: +0.5
- Best paper awards as first author: +0.5
- Research grants as PI: +0.5
- Tutorial/workshop organizer: +0.5

CRITICAL RULES:
1. FIRST AUTHOR matters most for assessing research capability
2. CORRESPONDING AUTHOR shows leadership in collaborative work
3. Consider PROPORTION: 3/5 first-author > 5/20 first-author
4. Service roles (PC/AC) indicate community recognition
5. Be STRICT for junior candidates: expect clear first-authorship

OUTPUT FORMAT (JSON only):
{{
  "score": <1-5>,
  "explanation": "<Detailed analysis: How many first/corresponding author papers? What proportion? Evidence of independent research? Leadership activities? Why this score?>"
}}
"""

# ============================ HELPER FUNCTION ============================

def format_candidate_info_for_eval(candidate_overview, user_query, research_field="", query_venues=None, author_priority=None):
    """
    Helper function to format candidate information for evaluation prompts
    
    Args:
        candidate_overview: CandidateOverview object
        user_query: User's search query string
        research_field: Research field from query spec
        query_venues: List of venues from query spec
        author_priority: Author priority from query spec
        
    Returns:
        Dict with formatted strings for each prompt
    """
    # Format representative papers with enhanced details (Title + Venue + Summary)
    papers_list = []
    representative_papers = getattr(candidate_overview, 'representative_papers', [])
    
    # Limit to 8 papers to control token usage
    for paper in representative_papers[:8]:
        title = paper.get('Title', 'N/A')
        year = paper.get('Year', 'N/A')
        venue = paper.get('Venue', 'N/A')
        
        paper_str = f"- {title} ({year}, {venue})"
        
        # Add paper summary (priority: tldr > introduction > abstract)
        summary = (
            paper.get('tldr', '') or 
            paper.get('introduction', '') or 
            paper.get('abstract', '')
        )
        
        if summary:
            # Trim summary to 250 characters to control token usage
            summary_trimmed = summary[:250] + "..." if len(summary) > 250 else summary
            paper_str += f"\n  {summary_trimmed}"
        
        papers_list.append(paper_str)
    
    papers_text = "\n\n".join(papers_list) if papers_list else "No papers listed"
    
    return {
        "candidate_name": getattr(candidate_overview, 'name', 'Unknown'),
        "user_query": user_query,
        "research_field": research_field,
        "query_venues": ", ".join(query_venues) if query_venues else "Not specified",
        "author_priority": ", ".join(author_priority) if author_priority else "Not specified",
        "research_interests": ", ".join(getattr(candidate_overview, 'research_keywords', [])),
        "research_keywords": ", ".join(getattr(candidate_overview, 'research_keywords', [])),
        "research_focus": ", ".join(getattr(candidate_overview, 'research_focus', [])),
        "representative_papers": papers_text,
        "publication_overview": "\n".join(getattr(candidate_overview, 'publication_overview', [])),
        "top_tier_hits": "\n".join(getattr(candidate_overview, 'top_tier_hits', [])),
        "highlights": "\n".join(getattr(candidate_overview, 'highlights', [])),
        "service_talks": "\n".join(getattr(candidate_overview, 'service_talks', [])),
        "honors_grants": "\n".join(getattr(candidate_overview, 'honors_grants', [])),
    }

