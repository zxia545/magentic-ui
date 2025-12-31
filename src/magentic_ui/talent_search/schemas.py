"""
Pydantic schemas for Talent Search System
Defines all data models used in the system
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict, field_validator

from . import config
from . import utils
from .utils import normalize_whitespace, normalize_url

# ============================ QUERY AND PLANNING SCHEMAS ============================

class QuerySpec(BaseModel):
    """Structured intent parsed from user query"""
    top_n: int = config.DEFAULT_TOP_N
    years: List[int] = Field(default_factory=lambda: config.DEFAULT_YEARS)
    venues: List[str] = Field(default_factory=lambda: ["ICLR","ICML","NeurIPS"])     # e.g., ["ICLR","ICML","NeurIPS",...]
    keywords: List[str] = Field(default_factory=lambda: ["social simulation","multi-agent"])   # e.g., ["social simulation","multi-agent",...]
    research_field: str = "Machine Learning"  # Primary research field identified from query
    must_be_current_student: bool = True
    degree_levels: List[str] = Field(
        default_factory=lambda: ["PhD","Master"],
        description="Candidate categories. Valid: Master, PhD, Professor, Postdoc, Industrial Researcher, Institution Researcher"
    )
    author_priority: List[str] = Field(
        default_factory=lambda: ["first"],
        description="[DEPRECATED] Author position filter - now extracts ALL authors from papers"
    )
    extra_constraints: List[str] = Field(default_factory=list)  # Other constraints (region/domain etc.)

    @field_validator("years")
    @classmethod
    def keep_ints(cls, v):
        out = []
        for x in v:
            try:
                out.append(int(x))
            except:
                pass
        return out[:5]

    @field_validator("keywords", "venues", "degree_levels", "author_priority", "extra_constraints")
    @classmethod
    def trim_list(cls, v):
        # Deduplicate while preserving order + limit length
        seen = set()
        out = []
        for s in v:
            s = utils.normalize_whitespace(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:32]


# ============================ CLASSIFICATION SCHEMAS ============================

class UserAdjustmentClassification(BaseModel):
    """LLM classification result for user adjustment detection"""
    is_adjustment: bool = Field(..., description="Whether the user input is requesting a change to search parameters")
    help_instruction: str = Field(..., description="Help instruction for the user if not an adjustment")

class SearchValidationResult(BaseModel):
    """LLM validation result for search request"""
    is_valid_search: bool = Field(..., description="Whether the user input contains valid searchable content")
    search_terms_found: List[str] = Field(default_factory=list, description="Search terms extracted from the input")
    missing_elements: List[str] = Field(default_factory=list, description="Missing elements needed for a good search")
    suggestion: str = Field(..., description="Suggestion for improving the search request")


class QuerySpecDiff(BaseModel):
    """Partial update to QuerySpec. Omit fields that are unchanged."""
    top_n: Optional[int] = None
    years: Optional[List[int]] = None
    venues: Optional[List[str]] = None
    keywords: Optional[List[str]] = None
    research_field: Optional[str] = None
    must_be_current_student: Optional[bool] = None
    degree_levels: Optional[List[str]] = None
    author_priority: Optional[List[str]] = None
    extra_constraints: Optional[List[str]] = None

    @field_validator("years")
    @classmethod
    def keep_ints(cls, v):
        if v is None:
            return v
        out = []
        for x in v:
            try:
                out.append(int(x))
            except:
                pass
        return out[:5]

    @field_validator("keywords", "venues", "degree_levels", "author_priority", "extra_constraints")
    @classmethod
    def trim_list(cls, v):
        if v is None:
            return v
        seen = set()
        out = []
        for s in v:
            s = utils.normalize_whitespace(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:32]

class PlanSpec(BaseModel):
    """Search plan specification"""
    search_terms: List[str] = Field(..., description="Initial search queries.")
    selection_hint: str = Field(..., description="Preferred sources to select.")

    @field_validator("search_terms")
    @classmethod
    def non_empty(cls, v):
        if not v:
            raise ValueError("search_terms cannot be empty")
        return v[:config.MAX_SEARCH_TERMS]

# ============================ SEARCH AND SELECTION SCHEMAS ============================

class LLMSelectSpec(BaseModel):
    """LLM decision for single URL selection"""
    should_fetch: bool = Field(..., description="Whether this URL should be fetched")

class LLMSelectSpecWithValue(BaseModel):
    """LLM decision for single URL selection with value score"""
    should_fetch: bool = Field(..., description="Whether this URL should be fetched")
    value_score: float = Field(..., description="Value score for this URL")
    reason: str = Field(..., description="Reason for the value score")

class LLMSelectSpecHasAuthorInfo(BaseModel):
    """LLM decision for single URL selection with author info"""
    has_author_info: bool = Field(..., description="Whether this URL contains author info")
    confidence: float = Field(..., description="Confidence score for the author info")
    reason: str = Field(..., description="Reason for the author info")

class LLMSelectSpecVerifyIdentity(BaseModel):
    """LLM decision for profile identity verification"""
    is_target_author: bool = Field(..., description="Whether this profile belongs to the target author")
    confidence: float = Field(..., description="Confidence score for identity verification")
    reason: str = Field(..., description="Specific reason for the decision")

class LLMHomepageIdentitySpecSimple(BaseModel):
    """LLM decision for homepage identity verification before content extraction"""
    is_personal_homepage: bool = Field(..., description="Whether this homepage belongs to the target author")
    confidence: float = Field(..., description="Confidence score for homepage identity verification")
    reason: str = Field(..., description="Detailed reason for the verification decision")


class LLMHomepageIdentitySpec(BaseModel):
    """LLM decision for homepage identity verification before content extraction"""
    is_target_author_homepage: bool = Field(..., description="Whether this homepage belongs to the target author")
    confidence: float = Field(..., description="Confidence score for homepage identity verification")
    author_name_found: str = Field(default="", description="Author name found on the homepage")
    research_area_match: bool = Field(default=False, description="Whether research areas match expectations")
    reason: str = Field(..., description="Detailed reason for the verification decision")

class SelectSpec(BaseModel):
    """URL selection specification - keeping existing structure"""
    urls: List[str] = Field(..., description="Up to N URLs worth fetching (http/https).")

    @field_validator("urls")
    @classmethod
    def limit_len(cls, v):
        seen = set()
        out = []
        for u in v:
            nu = utils.normalize_whitespace(u)
            if nu.startswith("http") and nu not in seen:
                seen.add(nu)
                out.append(nu)
        return out[:config.MAX_URLS]
    
class LLMPaperNameSpec(BaseModel):
    """Specification for paper name extraction"""
    have_paper_name: bool = Field(..., description="Whether the paper name is extracted")
    paper_name: str = Field(..., description="The name of the paper")
    
    
# ============================ CONTENT ANALYSIS SCHEMAS ============================

# Removed complex content analysis - keeping it simple for now

# ============================ AUTHOR AND CANDIDATE SCHEMAS ============================

class CandidateCard(BaseModel):
    """Individual candidate information card"""
    name: str = Field(..., alias="Name")
    current_role_affiliation: str = Field(..., alias="Current Role & Affiliation")
    research_focus: List[str] = Field(default_factory=list, alias="Research Focus")
    profiles: Dict[str, str] = Field(default_factory=dict, alias="Profiles")
    notable: Optional[str] = Field(default=None, alias="Notable")
    evidence_notes: Optional[str] = Field(default=None, alias="Evidence Notes")
    model_config = ConfigDict(populate_by_name=True)

class CandidatesSpec(BaseModel):
    """Specification for candidate extraction results"""
    candidates: List[CandidateCard] = Field(default_factory=list)
    citations: List[str] = Field(default_factory=list)
    need_more: bool = False
    followups: List[str] = Field(default_factory=list)

class AuthorListSpec(BaseModel):
    """Specification for author list extraction"""
    authors: List[str] = Field(default_factory=list)

    @field_validator("authors")
    @classmethod
    def limit_authors(cls, v):
        seen = set()
        out = []
        for name in v:
            name = normalize_whitespace(name)
            if config.MIN_AUTHOR_NAME_LENGTH <= len(name) <= config.MAX_AUTHOR_NAME_LENGTH and name not in seen:
                seen.add(name)
                out.append(name)
        return out[:config.MAX_AUTHORS]

# ============================ PAPER AND AUTHOR SCHEMAS ============================

class PaperInfo(BaseModel):
    """Information about a paper with deduplication support"""
    paper_name: str = Field(..., description="The extracted paper name")
    urls: List[str] = Field(default_factory=list, description="List of URLs where this paper was found")
    primary_url: Optional[str] = Field(default=None, description="The primary/best URL for this paper")

    @field_validator("paper_name")
    @classmethod
    def normalize_paper_name(cls, v):
        return normalize_whitespace(v)

    @field_validator("urls")
    @classmethod
    def deduplicate_urls(cls, v):
        seen = set()
        out = []
        for url in v:
            if url and url not in seen:
                seen.add(url)
                out.append(url)
        return out

class CorrespondingAuthorEvidence(BaseModel):
    """Evidence for corresponding author detection from PDF"""
    author_index: int = Field(..., description="1-based author index in the author list")
    name: str = Field(..., description="Author name as found in evidence")
    email: Optional[str] = Field(default=None, description="Email address if found")
    page: int = Field(default=1, description="PDF page number where evidence was found")
    confidence: float = Field(default=0.0, ge=0.0, le=1.0, description="Confidence score (0.0-1.0)")
    snippet: str = Field(default="", description="Evidence text snippet for audit")
    method: str = Field(default="", description="Detection method: rule_type_I, rule_type_II, rule_type_III, llm_fallback")


class AuthorWithId(BaseModel):
    """Author information with Semantic Scholar ID and affiliations"""
    name: str = Field(..., description="Author name")
    author_id: Optional[str] = Field(default=None, description="Semantic Scholar author ID")
    affiliations: Optional[List[str]] = Field(default=None, description="List of author affiliations (institutions) from paper")
    is_corresponding: Optional[bool] = Field(default=None, description="Whether this author is a corresponding author (from PDF detection)")

    @field_validator("name")
    @classmethod
    def normalize_name(cls, v):
        return normalize_whitespace(v)
    
    @field_validator("affiliations")
    @classmethod
    def normalize_affiliations(cls, v):
        if v:
            # Normalize and deduplicate
            normalized = [normalize_whitespace(aff) for aff in v if aff and aff.strip()]
            return normalized if normalized else None
        return v

class PaperAuthorsResult(BaseModel):
    """Result from Semantic Scholar paper search with authors"""
    url: str = Field(..., description="Original URL where paper was found")
    paper_name: str = Field(..., description="Paper title used for search")
    paper_id: Optional[str] = Field(default=None, description="Semantic Scholar paper ID")
    match_score: Optional[float] = Field(default=None, description="Semantic Scholar match score")
    year: Optional[int] = Field(default=None, description="Publication year")
    venue: Optional[str] = Field(default=None, description="Publication venue")
    paper_url: Optional[str] = Field(default=None, description="Semantic Scholar paper URL")
    authors: List[AuthorWithId] = Field(default_factory=list, description="List of authors with IDs")
    found: bool = Field(default=False, description="Whether the paper was found in Semantic Scholar")
    # Corresponding author detection results
    corresponding_author_evidences: Optional[List[CorrespondingAuthorEvidence]] = Field(
        default=None, 
        description="Evidence for corresponding authors detected from PDF"
    )

class PaperCollection(BaseModel):
    """Collection of unique papers with deduplication"""
    papers: Dict[str, PaperInfo] = Field(default_factory=dict, description="Paper name -> PaperInfo mapping")

    def add_paper(self, paper_name: str, url: str) -> bool:
        """
        Add a paper URL. Returns True if added, False if paper already exists.
        If paper exists, adds URL to existing list if not already present.
        """
        paper_name = normalize_whitespace(paper_name)

        if not paper_name or not url:
            return False

        if paper_name in self.papers:
            # Paper already exists, add URL if not present
            if url not in self.papers[paper_name].urls:
                self.papers[paper_name].urls.append(url)
            return False  # Not newly added

        # New paper
        self.papers[paper_name] = PaperInfo(
            paper_name=paper_name,
            urls=[url],
            primary_url=url
        )
        return True

    def get_all_papers(self) -> List[PaperInfo]:
        """Get all papers as a list"""
        return list(self.papers.values())

    def get_paper_names(self) -> List[str]:
        """Get all paper names"""
        return list(self.papers.keys())

    def get_urls_for_paper(self, paper_name: str) -> List[str]:
        """Get all URLs for a specific paper"""
        paper_name = normalize_whitespace(paper_name)
        if paper_name in self.papers:
            return self.papers[paper_name].urls
        return []

# ============================ AUTHOR PROFILE SCHEMAS ============================

class LLMAuthorProfileSpec(BaseModel):
    """LLM specification for author profile extraction"""
    name: str = Field(default="", description="Author name as written")
    aliases: List[str] = Field(default_factory=list, description="Name variants/aliases of THIS AUTHOR ONLY")
    affiliation_current: str = Field(default="", description="Current affiliation")
    emails: List[str] = Field(default_factory=list, description="Professional emails")
    personal_homepage: str = Field(default="", description="Personal website URL (not current page)")
    homepage_url: str = Field(default="", description="Personal or lab/university page (legacy field)")
    interests: List[str] = Field(default_factory=list, description="Research interests")
    selected_publications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Selected publications with title/year/venue/url"
    )
    notable_achievements: List[str] = Field(
        default_factory=list,
        description="Awards, honors, fellowships, recognitions"
    )
    social_impact: str = Field(default="", description="H-index, citations, influence metrics")
    career_stage: str = Field(default="", description="Career stage: student/postdoc/assistant_prof/etc")
    social_links: Dict[str, str] = Field(
        default_factory=dict,
        description="Social media and platform links extracted from page"
    )

class HomepageInsightsSpec(BaseModel):
    """Insights extracted specifically from personal website/homepage"""
    current_status: str = Field(default="", description="Concise current status/position as written on homepage")
    role_affiliation_detailed: str = Field(default="", description="Detailed current role and affiliation from homepage")
    research_focus: List[str] = Field(default_factory=list, description="Research focus/themes bullets")
    research_keywords: List[str] = Field(default_factory=list, description="Research keywords/tags")
    highlights: List[str] = Field(default_factory=list, description="News/highlights/awards as listed on homepage")

    @field_validator("research_focus", "research_keywords", "highlights")
    @classmethod
    def _dedup_trim(cls, v):
        seen = set()
        out = []
        for s in v or []:
            s = normalize_whitespace(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:16]


class HomepageHighlightsSpec(BaseModel):
    """LLM-curated homepage highlights with brief summary"""
    curated_highlights: List[str] = Field(default_factory=list, description="Curated and summarized highlights from homepage")
    summary: str = Field(default="", description="1–2 sentence summary of highlights")

    @field_validator("curated_highlights")
    @classmethod
    def _dedup_trim_highlights(cls, v):
        seen = set()
        out = []
        for s in v or []:
            s = normalize_whitespace(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:16]


class OpenSourceItem(BaseModel):
    """Single open-source project/dataset entry"""
    name: str = Field(default="", description="Project or dataset name")
    type: str = Field(default="", description="project | dataset | library | code")
    url: str = Field(default="", description="URL if present on homepage")
    description: str = Field(default="", description="1-line description as written")


class OpenSourceProjectsSpec(BaseModel):
    """LLM output for open-source projects/datasets"""
    items: List[OpenSourceItem] = Field(default_factory=list)


class AcademicServiceSpec(BaseModel):
    """LLM output for academic service and invited talks"""
    service_roles: List[str] = Field(default_factory=list, description="Committees, editorial roles, organizing")
    invited_talks: List[str] = Field(default_factory=list, description="Invited/keynote talks as listed")

    @field_validator("service_roles", "invited_talks")
    @classmethod
    def _dedup_trim_lists(cls, v):
        seen = set()
        out = []
        for s in v or []:
            s = normalize_whitespace(s)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out[:24]

# ============================ STATE AND RESEARCH SCHEMAS ============================

class ResearchState(BaseModel):
    """Main state object for the research process"""
    query: str
    round: int = 0
    query_spec: QuerySpec = Field(default_factory=QuerySpec)
    plan: Dict[str, Any] = Field(default_factory=dict)
    serp: List[Dict[str, str]] = Field(default_factory=list)
    selected_urls: List[str] = Field(default_factory=list)
    selected_serp: List[Dict[str, str]] = Field(default_factory=list)
    sources: Dict[str, str] = Field(default_factory=dict)   # url -> text
    report: Optional[str] = None
    candidates: List[Dict[str, Any]] = Field(default_factory=list)
    need_more: bool = False
    followups: List[str] = Field(default_factory=list)
    expanded_authors: bool = False

# ============================ UTILITY FUNCTIONS ============================

# Removed create_selection_prompt - now handled directly in graph.py

# ============================ PAPER SCORING SCHEMAS ============================

class PaperScoreSpec(BaseModel):
    """LLM output for paper relevance scoring"""
    score: int = Field(..., ge=1, le=10, description="Relevance score from 1-10, where 10 is most relevant")


class PDFScoringEvidence(BaseModel):
    """Single piece of evidence from PDF for scoring"""
    orig_page: int = Field(default=0, description="Original PDF page number (1-based)")
    snippet: str = Field(default="", description="Verbatim quote from PDF (<=25 words)")
    supports: str = Field(default="", description="What this evidence supports for relevance")


class FullPDFScoreResult(BaseModel):
    """
    Complete scoring result from Full PDF evaluation.
    This is the new standard for paper relevance scoring using complete PDF content.
    Replaces the old abstract+introduction based scoring approach.
    """
    score: int = Field(default=4, ge=1, le=8, description="Relevance score from 1-8")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence in the score (0.0-1.0)")
    explanation: str = Field(default="", description="3-5 sentence explanation grounded in the paper")
    evidence: List[PDFScoringEvidence] = Field(default_factory=list, description="Evidence items with page numbers and snippets")
    missing_aspects: List[str] = Field(default_factory=list, description="Aspects that couldn't be verified from the PDF")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Metadata about the scoring process (method, timing, etc.)")
    
    @field_validator("score")
    @classmethod
    def clamp_score(cls, v):
        """Ensure score is within valid range"""
        return max(1, min(8, v))

class PaperWithScore(BaseModel):
    """Paper with its relevance score"""
    url: str = Field(..., description="URL of the paper")
    title: str = Field(..., description="Title of the paper")
    abstract: str = Field(default="", description="Abstract or snippet of the paper")
    introduction: str = Field(default="", description="Introduction section of the paper (Section 1)")
    venue: str = Field(default="", description="Conference or journal where the paper was published (e.g., 'ACL 2025', 'ICLR 2024')")
    score: int = Field(..., ge=1, le=8, description="Relevance score from 1-8 (papers with score < 6 won't be used for candidate extraction but kept in reference list)")
    explanation: str = Field(default="", description="LLM explanation for the relevance score")
    relevant_tier: int = Field(default=0, description="Relevance tier: 1 for highly relevant (6-8), 0 for normal (< 6)")
    authors: List[str] = Field(default_factory=list, description="List of author names for this paper")
    associated_candidates: List[str] = Field(default_factory=list, description="Names of candidates associated with this paper")
    # Multi-dimensional scoring fields (10 dimensions: 5 relevance + 5 quality)
    relevance_score: float = Field(default=0.0, description="Average of 5 relevance dimensions (1-8), used for candidate filtering threshold")
    quality_score: float = Field(default=0.0, description="Average of 5 quality dimensions (1-8), for reference only")
    relevance_dimensions: Dict[str, Any] = Field(default_factory=dict, description="5 relevance dimension scores with rationale")
    quality_dimensions: Dict[str, Any] = Field(default_factory=dict, description="5 quality dimension scores with rationale")
    relevance_explanation: str = Field(default="", description="Brief explanation of relevance score")
    
    @field_validator("url")
    @classmethod
    def normalize_url_field(cls, v):
        return normalize_url(v)

class SearchResults(BaseModel):
    """Complete search results including candidates and reference papers"""
    recommended_candidates: List['CandidateOverview'] = Field(default_factory=list, description="Top recommended candidates based on score")
    additional_candidates: List['CandidateOverview'] = Field(default_factory=list, description="Additional candidates that may be of interest")
    candidates_by_category: Dict[str, List['CandidateOverview']] = Field(default_factory=dict, description="All candidates grouped by category (PhD Student, Master Student, Professor, etc.)")
    reference_papers: List[PaperWithScore] = Field(default_factory=list, description="All scored papers with associated candidates")
    total_candidates_found: int = Field(default=0, description="Total number of candidates found (before Topic Match filtering)")
    matching_candidates_count: int = Field(default=0, description="Number of candidates matching degree requirements and Topic Match >= 3")
    search_query: str = Field(default="", description="Original search query")
    search_logs: str = Field(default="", description="Detailed search execution logs for debugging")
    log_file_path: str = Field(default="", description="Path to the log file for this search")
    enhanced_profiles: Dict[str, 'EnhancedAuthorProfile'] = Field(default_factory=dict, description="Enhanced 9-field profiles by candidate name")
    
    model_config = ConfigDict(arbitrary_types_allowed=True, from_attributes=True, validate_assignment=False)

# ============================ EVALUATION & OVERVIEW SCHEMAS ============================

class EvaluationItem(BaseModel):
    """Single evaluation item for one dimension with score and justification"""
    dimension: str = Field(..., description="Dimension name")
    score: int = Field(..., ge=1, le=5, description="Score 1–5")
    justification: str = Field(..., description="Detailed justification")

class EvaluationResult(BaseModel):
    """Structured evaluation across four dimensions"""
    items: List[EvaluationItem] = Field(default_factory=list)
    radar: Dict[str, int] = Field(default_factory=dict, description="Dimension -> score mapping for radar plot")
    total_score: int = Field(default=0, description="Sum of four dimension scores (max 20)")
    details: Dict[str, str] = Field(default_factory=dict, description="Dimension -> 'x/5 - justification'")

class LLMEvaluationItemSpec(BaseModel):
    """LLM output spec: one evaluation entry"""
    dimension: str = Field(...)
    score: int = Field(..., ge=1, le=5)
    justification: str = Field(...)

class LLMEvaluationResultSpec(BaseModel):
    """LLM output spec: list of four evaluation entries"""
    items: List[LLMEvaluationItemSpec] = Field(default_factory=list)

class LLMSingleDimensionEval(BaseModel):
    """LLM output spec: single dimension evaluation with score and explanation"""
    score: int = Field(..., ge=1, le=5, description="Score from 1 to 5")
    explanation: str = Field(..., description="Detailed explanation for the score")

class RepresentativePaper(BaseModel):
    """Simplified representative paper entry for demo output"""
    title: str = Field(..., alias="Title")
    venue: str = Field(default="", alias="Venue")
    year: Optional[int] = Field(default=None, alias="Year")
    type: str = Field(default="", alias="Type")
    links: str = Field(default="", alias="Links")
    model_config = ConfigDict(populate_by_name=True)


class LLMRepresentativePapersSpec(BaseModel):
    """LLM output: list of representative papers extracted from homepage"""
    papers: List[RepresentativePaper] = Field(default_factory=list)

# ============================ ENHANCED PROFILE SCHEMAS (9 FIELDS) ============================

class IntroductionInfo(BaseModel):
    """Introduction section of enhanced profile"""
    model_config = ConfigDict(from_attributes=True)
    name: str = Field(default="")
    affiliation: str = Field(default="")
    position: str = Field(default="")
    text: str = Field(default="", description="Brief introduction text (1-2 sentences)")

class CurrentRoleInfo(BaseModel):
    """LLM-determined current role metadata"""
    model_config = ConfigDict(from_attributes=True)
    category: str = Field(default="Unknown", description="One of the 7 role categories")
    role_text: str = Field(default="", description="Natural role description returned by LLM")
    affiliation: str = Field(default="", description="Affiliation selected alongside the role")
    explanation: str = Field(default="", description="Short reasoning for the role selection")

class ResearchInterest(BaseModel):
    """Single research interest with name and description"""
    model_config = ConfigDict(from_attributes=True)
    name: str = Field(..., description="Research interest name")
    description: str = Field(default="", description="1-2 sentence description")

class PublicationInfo(BaseModel):
    """Publication entry with classification and author position tracking"""
    model_config = ConfigDict(from_attributes=True)
    title: str = Field(...)
    authors: List[str] = Field(default_factory=list)
    venue: str = Field(default="")
    year: Optional[int] = None
    url: str = Field(default="")
    citations: int = Field(default=0)
    abstract: str = Field(default="")
    keywords: List[str] = Field(default_factory=list)
    category: str = Field(default="Uncategorized", description="Research interest category")
    note: str = Field(default="", description="Special note (e.g., Best Paper Award)")
    # Relevance scoring (used for candidate selection, >= 7 threshold)
    relevance_score: float = Field(default=0.0, description="LLM relevance score (1-8) based on user query, average of 5 relevance dimensions")
    relevance_explanation: str = Field(default="", description="LLM explanation for relevance score")
    # Quality scoring (for reference only, not used in selection)
    quality_score: float = Field(default=0.0, description="LLM quality score (1-8) average of 5 quality dimensions")
    # Detailed dimension scores (10 dimensions total: 5 relevance + 5 quality)
    relevance_dimensions: Dict[str, Any] = Field(default_factory=dict, description="5 relevance dimension scores with rationale")
    quality_dimensions: Dict[str, Any] = Field(default_factory=dict, description="5 quality dimension scores with rationale")
    # Author position tracking for the candidate
    candidate_author_position: Optional[int] = Field(default=None, description="1-based position of candidate in author list (1=first author)")
    candidate_author_position_label: Optional[str] = Field(default=None, description="Human-readable position label (e.g., '1st author', 'last author')")
    total_authors: Optional[int] = Field(default=None, description="Total number of authors in this paper")
    # Corresponding author tracking
    is_corresponding_author: Optional[bool] = Field(default=None, description="Whether the candidate is a corresponding author (from PDF detection)")
    # Co-first author tracking
    is_cofirst_author: Optional[bool] = Field(default=None, description="Whether the candidate is a co-first author (from PDF detection)")

class SelectedResearch(BaseModel):
    """Selected research with publications grouped by category"""
    model_config = ConfigDict(from_attributes=True)
    by_category: Dict[str, List[PublicationInfo]] = Field(default_factory=dict)
    all_publications: List[PublicationInfo] = Field(default_factory=list)

class AwardInfo(BaseModel):
    """Unified Award/Honor entry supporting various binding types"""
    model_config = ConfigDict(from_attributes=True)
    # 核心字段（必填）
    name: str = Field(..., description="Award name (e.g., 'Best Paper Award', 'NSF Fellowship')")
    year: str = Field(default="", description="Year as string (e.g., '2024')")
    # 组织/机构信息
    organization: str = Field(default="", description="Awarding organization (e.g., 'ICLR', 'NSF', 'ACM')")
    # 论文相关（用于 paper awards）
    paper_title: str = Field(default="", description="Paper title if award is tied to a specific paper")
    venue: str = Field(default="", description="Conference/journal venue (e.g., 'ICLR 2024', 'NeurIPS')")
    # 项目相关（用于 project awards）
    project_name: str = Field(default="", description="Project name if award is tied to a specific project")
    # 其他信息
    description: str = Field(default="", description="Additional details or context")
    # Award 类型标识（可选，用于内部处理）
    award_type: str = Field(default="personal", description="Type: 'paper', 'project', 'personal'")

class ServiceInfo(BaseModel):
    """Professional service entry"""
    model_config = ConfigDict(from_attributes=True)
    role: str = Field(..., description="Role (e.g., PC, AC, Reviewer)")
    venue: str = Field(..., description="Conference/Journal/Workshop name")
    year: str = Field(default="")

class EducationInfo(BaseModel):
    """Education entry (legacy - kept for backward compatibility)"""
    model_config = ConfigDict(from_attributes=True)
    degree: str = Field(..., description="Degree/Position (e.g., Ph.D. in CS)")
    institution: str = Field(...)
    duration: str = Field(default="", description="Time period (e.g., 2020-2024)")
    field: str = Field(default="", description="Field of study (e.g., Computer Science, Electrical Engineering)")

class CareerEducationInfo(BaseModel):
    """Career & Education History entry (all non-industry career stages)"""
    model_config = ConfigDict(from_attributes=True)
    degree_or_position: str = Field(default="", description="Degree (PhD/Master/Bachelor/Postdoc) OR Academic position (Professor/Lecturer/TA)")
    institution: str = Field(default="", description="University/Research institute name")
    department: str = Field(default="", description="Department/School name")
    duration: str = Field(default="", description="Time period (e.g., '2020-Present', 'Fall 2023')")
    field: str = Field(default="", description="Field of study or course name")
    advisor: str = Field(default="", description="Advisor name (for student phases)")
    description: str = Field(default="", description="Additional details")

class ExperienceInfo(BaseModel):
    """Industrial experience entry"""
    model_config = ConfigDict(from_attributes=True)
    position: str = Field(...)
    organization: Optional[str] = Field(default="")
    duration: str = Field(default="")
    description: str = Field(default="")

class ContactInfo(BaseModel):
    """Contact information"""
    model_config = ConfigDict(from_attributes=True)
    email: str = Field(default="")
    phone: str = Field(default="")
    homepage: str = Field(default="")
    google_scholar: str = Field(default="")
    github: str = Field(default="")
    twitter: str = Field(default="")
    linkedin: str = Field(default="")
    orcid: str = Field(default="")
    dblp: str = Field(default="")
    openreview: str = Field(default="", description="OpenReview profile URL")

class EnhancedAuthorProfile(BaseModel):
    """Enhanced 9-field structured author profile"""
    model_config = ConfigDict(from_attributes=True)
    # 1. Introduction (general background)
    introduction: IntroductionInfo = Field(default_factory=IntroductionInfo)
    current_role: CurrentRoleInfo = Field(default_factory=CurrentRoleInfo, description="LLM-determined current role metadata")
    
    # 2. Research Interests (with descriptions)
    research_interests: List[ResearchInterest] = Field(default_factory=list)
    
    # 3. Selected Research (publications grouped by category)
    selected_research: SelectedResearch = Field(default_factory=SelectedResearch)
    
    # 4. Awards & Honors
    awards: List[AwardInfo] = Field(default_factory=list)
    
    # 5. Professional Services
    professional_services: List[ServiceInfo] = Field(default_factory=list)
    
    # 6. Career & Education History (all non-industry career stages)
    career_education_history: List[CareerEducationInfo] = Field(default_factory=list)
    
    # 7. Industrial Experience
    industrial_experience: List[ExperienceInfo] = Field(default_factory=list)
    
    # 8. Contact
    contact: ContactInfo = Field(default_factory=ContactInfo)
    
    # Metadata
    confidence: float = Field(default=0.7)
    data_sources: List[str] = Field(default_factory=list, description="List of data sources used")
    # External IDs for paper retrieval
    semantic_scholar_author_id: Optional[str] = Field(default=None, description="Semantic Scholar author ID for fetching publications")
    openreview_author_id: Optional[str] = Field(default=None, description="OpenReview author ID (e.g., ~John_Doe1)")
    
    high_quality_publications: List[Dict[str, Any]] = Field(default_factory=list, description="High-quality publications (score ≥ 7) from homepage/scholar")

# ============================ LLM RESPONSE SCHEMAS FOR ENHANCED EXTRACTION ============================

class LLMStringSpec(BaseModel):
    """LLM response with single string value"""
    value: str = Field(...)

class LLMDescriptionQualitySpec(BaseModel):
    """LLM response for evaluating description quality"""
    needs_refinement: bool = Field(..., description="Whether the description needs refinement")
    reason: str = Field(default="", description="Brief explanation for the decision")

class LLMTrustworthyBackgroundSpec(BaseModel):
    """LLM response for evaluating if background text is trustworthy"""
    is_trustworthy: bool = Field(..., description="Whether the background text is trustworthy and real")
    reason: str = Field(default="", description="Brief explanation for the decision")

class LLMStringListSpec(BaseModel):
    """LLM response with list of strings"""
    items: List[str] = Field(default_factory=list)

class LLMResearchInterestItem(BaseModel):
    """Single research interest for LLM extraction - compatible with OpenAI Structured Output"""
    name: str = Field(default="", description="Research interest name")
    description: str = Field(default="", description="1-2 sentence description of this research area")

class LLMResearchInterestsSpec(BaseModel):
    """LLM response for research interests synthesis - compatible with OpenAI Structured Output"""
    research_interests: List[LLMResearchInterestItem] = Field(default_factory=list)

# ============================ NINE-DIMENSION EXTRACTION SCHEMAS ============================

class LLMBackgroundSpec(BaseModel):
    """Background/Introduction extraction"""
    name: str = Field(default="")
    position: str = Field(default="")
    affiliation: str = Field(default="")
    bio_text: str = Field(default="")
    career_stage: str = Field(default="")

class ResearchInterestRaw(BaseModel):
    """Single research interest with optional description"""
    name: str = Field(..., description="Research interest name")
    description: str = Field(default="", description="Description of this research interest")

class LLMResearchInterestsRawSpec(BaseModel):
    """Raw research interests extraction (new format with name + description)"""
    interests: List[ResearchInterestRaw] = Field(default_factory=list)

# ============================ LLM STRUCTURED OUTPUT COMPATIBLE SCHEMAS ============================
# These schemas are designed to be compatible with OpenAI's Structured Output API requirements:
# 1. All properties must be in 'required' array
# 2. No additionalProperties: true (no Dict[str, Any])
# 3. Nested objects must have explicit schema

class LLMPublicationItem(BaseModel):
    """Single publication item for LLM extraction"""
    title: str = Field(default="", description="Publication title")
    venue: str = Field(default="", description="Conference/journal name")
    year: str = Field(default="", description="Publication year")
    url: str = Field(default="", description="URL to the publication")
    authors: str = Field(default="", description="Author list as string")

    @field_validator("year", mode="before")
    @classmethod
    def convert_year_to_string(cls, v):
        """Allow int or string for year, but convert to string"""
        if v is None:
            return ""
        return str(v)

class LLMPublicationsSpec(BaseModel):
    """Publications extraction - compatible with OpenAI Structured Output"""
    publications: List[LLMPublicationItem] = Field(default_factory=list)

class LLMAwardItem(BaseModel):
    """Single award item for LLM extraction"""
    name: str = Field(default="", description="Award name")
    year: str = Field(default="", description="Year received")
    organization: str = Field(default="", description="Awarding organization")
    paper_title: str = Field(default="", description="Associated paper title if any")
    venue: str = Field(default="", description="Conference/journal if paper award")
    description: str = Field(default="", description="Additional details")

class LLMAwardsSpec(BaseModel):
    """Awards extraction - compatible with OpenAI Structured Output"""
    awards: List[LLMAwardItem] = Field(default_factory=list)

class LLMServiceItem(BaseModel):
    """Single service item for LLM extraction"""
    role: str = Field(default="", description="Role (PC, AC, Reviewer, etc.)")
    venue: str = Field(default="", description="Conference/journal name")
    year: str = Field(default="", description="Year")

class LLMTalkItem(BaseModel):
    """Single invited talk for LLM extraction"""
    title: str = Field(default="", description="Talk title")
    venue: str = Field(default="", description="Event/institution name")
    year: str = Field(default="", description="Year")

class LLMServicesSpec(BaseModel):
    """Professional services extraction - compatible with OpenAI Structured Output"""
    services: List[LLMServiceItem] = Field(default_factory=list)
    invited_talks: List[LLMTalkItem] = Field(default_factory=list)

class LLMEducationItem(BaseModel):
    """Single education item for LLM extraction"""
    degree: str = Field(default="", description="Degree (PhD, Master, Bachelor, etc.)")
    institution: str = Field(default="", description="University/institution name")
    department: str = Field(default="", description="Department/school name")
    duration: str = Field(default="", description="Time period (e.g., 2020-2024)")
    field: str = Field(default="", description="Field of study")
    advisor: str = Field(default="", description="Advisor name if applicable")

class LLMEducationSpec(BaseModel):
    """Education extraction - compatible with OpenAI Structured Output"""
    education: List[LLMEducationItem] = Field(default_factory=list)

class LLMExperienceItem(BaseModel):
    """Single experience item for LLM extraction"""
    position: str = Field(default="", description="Job title/position")
    organization: str = Field(default="", description="Company/organization name")
    duration: str = Field(default="", description="Time period")
    description: str = Field(default="", description="Brief description of role")

class LLMExperienceSpec(BaseModel):
    """Industrial experience extraction - compatible with OpenAI Structured Output"""
    experiences: List[LLMExperienceItem] = Field(default_factory=list)

class LLMCareerEducationItem(BaseModel):
    """Single career/education item for LLM extraction"""
    degree_or_position: str = Field(default="", description="Degree or academic position")
    institution: str = Field(default="", description="University/research institute")
    department: str = Field(default="", description="Department/school name")
    duration: str = Field(default="", description="Time period")
    field: str = Field(default="", description="Field of study or research area")
    advisor: str = Field(default="", description="Advisor name")
    description: str = Field(default="", description="Additional details")

class LLMCareerEducationSpec(BaseModel):
    """Career & Education History extraction - compatible with OpenAI Structured Output"""
    career_education: List[LLMCareerEducationItem] = Field(default_factory=list)

class LLMSocialLinks(BaseModel):
    """Social links for LLM extraction - explicit schema instead of Dict[str, str]"""
    homepage: str = Field(default="", description="Personal homepage URL")
    google_scholar: str = Field(default="", description="Google Scholar URL")
    github: str = Field(default="", description="GitHub URL")
    twitter: str = Field(default="", description="Twitter/X URL")
    linkedin: str = Field(default="", description="LinkedIn URL")
    dblp: str = Field(default="", description="DBLP URL")
    orcid: str = Field(default="", description="ORCID URL")
    semantic_scholar: str = Field(default="", description="Semantic Scholar URL")

class LLMContactSpec(BaseModel):
    """Contact information extraction - compatible with OpenAI Structured Output"""
    email: str = Field(default="")
    phone: str = Field(default="")
    office: str = Field(default="")
    social_links: LLMSocialLinks = Field(default_factory=LLMSocialLinks)

class LLMRoleDeterminationSpec(BaseModel):
    """Role determination from career history"""
    role_category: str = Field(..., description="One of: PhD Student, Master Student, Undergraduate Student, Professor, Postdoc, Industrial Researcher, Institution Researcher")
    role_text: str = Field(..., description="Natural language role description (e.g., 'PhD student', 'Undergraduate student', 'Research Scientist')")
    affiliation: str = Field(..., description="Current institution or company name")
    explanation: str = Field(default="", description="Brief explanation of why this was chosen as current role")

class TriggerPaperInfo(BaseModel):
    """Trigger paper information for diversity rerank display"""
    paper_url: str = Field(default="", description="Paper URL (unique identifier)")
    title: str = Field(default="", description="Paper title")
    venue: str = Field(default="", description="Paper venue")
    score: float = Field(default=0.0, description="Relevance score (1-8)")
    contribution: float = Field(default=0.0, description="Contribution score (LLM_score × author_weight × venue_weight)")
    is_shared: bool = Field(default=False, description="Whether this paper is shared evidence (used by previous candidates)")
    author_position: Optional[str] = Field(default=None, description="Author position (e.g., 'first', 'second', 'last')")
    author_position_index: Optional[int] = Field(default=None, description="Author position index (1-based)")
    total_authors: Optional[int] = Field(default=None, description="Total number of authors")
    is_corresponding_author: Optional[bool] = Field(default=None, description="Whether the candidate is a corresponding author")

class CandidateOverview(BaseModel):
    """Candidate overview with 9-field structured profile + evaluation results"""
    # Core identification
    name: str = Field(..., alias="Name")
    # 1. General Background (Introduction)
    introduction: IntroductionInfo = Field(default_factory=IntroductionInfo, alias="Introduction")
    # 2. Research Interests (with descriptions)
    research_interests: List[ResearchInterest] = Field(default_factory=list, alias="Research Interests")
    # 3. Selected Research (publications grouped by category)
    selected_research: SelectedResearch = Field(default_factory=SelectedResearch, alias="Selected Research")
    # 4. Awards & Honors
    awards: List[AwardInfo] = Field(default_factory=list, alias="Awards")
    # 5. Professional Services
    professional_services: List[ServiceInfo] = Field(default_factory=list, alias="Professional Services")
    # 6. Career & Education History (all non-industry career stages)
    career_education_history: List[CareerEducationInfo] = Field(default_factory=list, alias="Career & Education History")
    # 7. Industrial Experience
    industrial_experience: List[ExperienceInfo] = Field(default_factory=list, alias="Industrial Experience")
    # 8. Contact
    contact: ContactInfo = Field(default_factory=ContactInfo, alias="Contact")
    # Metadata & Evaluation (kept for backward compatibility and UI display)
    trigger_paper_title: str = Field(default="", alias="Trigger Paper Title")
    trigger_paper_url: Optional[str] = Field(default="", alias="Trigger Paper URL")
    trigger_paper_venue: Optional[str] = Field(default="", alias="Trigger Paper Venue")
    trigger_paper_score: float = Field(default=0.0, alias="Trigger Paper Score")
    trigger_paper_explanation: str = Field(default="", alias="Trigger Paper Explanation")
    # Added: trigger paper authorship position information
    trigger_paper_position: Optional[str] = Field(
        default=None,
        alias="Trigger Paper Position",
        description="Author position in the trigger paper"
    )
    trigger_paper_position_index: Optional[int] = Field(
        default=None,
        alias="Trigger Paper Position Index",
        description="Author position index (1-based) in the trigger paper"
    )
    trigger_paper_total_authors: Optional[int] = Field(
        default=None,
        alias="Trigger Paper Total Authors",
        description="Total number of authors in the trigger paper"
    )
    radar: Dict[str, int] = Field(default_factory=dict, alias="Radar")
    total_score: int = Field(default=0, alias="Total Score")
    detailed_scores: Dict[str, str] = Field(default_factory=dict, alias="Detailed Scores")
    
    #候选人类别（6类）
    candidate_category: str = Field(default="Unknown", description="Candidate category: PhD Student, Master Student, Professor, Postdoc, Industrial Researcher, Institution Researcher")
    
    #论文作者位置评分
    score_paper: float = Field(default=0.0, alias="Score Paper", description="Paper authorship score based on author positions")
    final_score: float = Field(default=0.0, alias="Final Score", description="Final ranking score combining paper score and radar dimensions")
    
    #发现轮次（用于异步搜索增量显示）
    discovered_in_round: int = Field(default=1, alias="Discovered In Round", description="Which round this candidate was discovered in")
    # External IDs for paper retrieval
    semantic_scholar_author_id: Optional[str] = Field(default=None, description="Semantic Scholar author ID for fetching publications")
    openreview_author_id: Optional[str] = Field(default=None, description="OpenReview author ID (e.g., ~John_Doe1)")
    # Diversity rerank: trigger paper for display (1 paper, prioritizing independent evidence)
    trigger_papers: List['TriggerPaperInfo'] = Field(
        default_factory=list, 
        alias="Trigger Papers",
        description="Trigger paper for display (1 paper, prioritizing independent evidence)"
    )
    # Diversity rerank score (adjusted score after diversity penalty)
    diversity_adjusted_score: float = Field(
        default=0.0, 
        alias="Diversity Adjusted Score",
        description="Score after diversity rerank penalty (indep_sum + shared_sum with decay)"
    )
    # Independent evidence ratio (for debugging/display)
    independent_ratio: float = Field(
        default=1.0,
        alias="Independent Ratio",
        description="Ratio of independent evidence (indep_sum / total_paper_contrib)"
    )
    
    model_config = ConfigDict(populate_by_name=True)

# ============================ TASK STATE SCHEMAS FOR INCREMENTAL SEARCH ============================

class SearchTaskState(BaseModel):
    """State snapshot for resumable search tasks"""
    task_id: str = Field(..., description="Unique task ID")
    spec: QuerySpec = Field(..., description="Original query specification")
    
    # Search progress state
    pos: int = Field(default=0, description="Current position in search terms")
    terms: List[Any] = Field(
        default_factory=list, 
        description="Structured search parameters [{'keywords': [...], 'venue': '...', 'year': ...}]"
    )
    rounds_completed: int = Field(default=0, description="Number of candidate pool processing rounds completed")
    
    # Accumulated results
    candidates_accum: Dict[str, 'CandidateOverview'] = Field(
        default_factory=dict, 
        description="Accumulated candidates {name -> CandidateOverview}"
    )
    enhanced_profiles_accum: Dict[str, 'EnhancedAuthorProfile'] = Field(
        default_factory=dict,
        description="Accumulated enhanced profiles {name -> EnhancedAuthorProfile}"
    )
    all_serp: List[Dict[str, Any]] = Field(default_factory=list, description="All SERP results (flexible schema)")
    sources: Dict[str, str] = Field(default_factory=dict, description="Fetched sources {url -> text}")
    all_scored_papers: Dict[str, PaperWithScore] = Field(
        default_factory=dict,
        description="All scored papers {url -> PaperWithScore}"
    )
    search_candidate_set: List[tuple] = Field(
        default_factory=list,
        description="Set of candidates to process: [(author_name, author_id, paper_title, paper_url), ...]. Note: author_name can be from ANY position (1st, 2nd, ..., last author)"
    )
    
    # Tracking sets (for deduplication)
    selected_urls_set: set = Field(default_factory=set, description="Set of selected URLs")
    selected_serp_url_set: set = Field(default_factory=set, description="Set of SERP URLs")
    
    # Accumulated logs
    accumulated_logs: str = Field(default="", description="Accumulated search logs from all rounds")
    
    # Metadata
    created_at: float = Field(default_factory=lambda: __import__('time').time())
    updated_at: float = Field(default_factory=lambda: __import__('time').time())
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class PartialSearchResults(BaseModel):
    """Partial search results returned during incremental search"""
    task_id: str = Field(..., description="Task ID for resuming")
    need_user_decision: bool = Field(default=False, description="Whether user decision is needed")
    rounds_completed: int = Field(default=0, description="Number of rounds completed")
    total_candidates_found: int = Field(default=0, description="Total candidates found so far")
    matching_degree_candidates: int = Field(default=0, description="Candidates matching degree requirements")
    current_candidates: List['CandidateOverview'] = Field(
        default_factory=list, 
        description="Current candidates found"
    )
    message: str = Field(default="", description="Status message for user")
    log_file_path: str = Field(default="", description="Path to the log file for this search")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)


class SearchTaskAction(BaseModel):
    """Action to perform on a search task"""
    action: str = Field(..., description="Action: 'start', 'resume', or 'finish'")
    task_id: Optional[str] = Field(default=None, description="Task ID (required for resume/finish)")
    spec: Optional[QuerySpec] = Field(default=None, description="Query spec (required for start)")
# Rebuild models to resolve forward references
# IMPORTANT: Order matters! Rebuild inner/nested types first, then outer types
# Step 1: Rebuild all basic field types (innermost layer)
IntroductionInfo.model_rebuild()
CurrentRoleInfo.model_rebuild()
ResearchInterest.model_rebuild()
PublicationInfo.model_rebuild()
SelectedResearch.model_rebuild()
AwardInfo.model_rebuild()
ServiceInfo.model_rebuild()
CareerEducationInfo.model_rebuild()
ExperienceInfo.model_rebuild()
ContactInfo.model_rebuild()
# Step 2: Rebuild profiles that use above types
EnhancedAuthorProfile.model_rebuild()
# Step 3: Rebuild trigger/candidate types
TriggerPaperInfo.model_rebuild()
CandidateOverview.model_rebuild()
# Step 4: Rebuild paper/result types
PaperWithScore.model_rebuild()
# Step 5: Rebuild final result types (outermost layer)
SearchResults.model_rebuild()
PartialSearchResults.model_rebuild()
SearchTaskState.model_rebuild()
