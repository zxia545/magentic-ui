"""
Corresponding Author Detector

Three-stage pipeline for detecting corresponding authors from PDF:
- Stage A: Lightweight evidence locating (no judgment, pure regex)
- Stage B: Rule engine for strong evidence judgment (core logic)
- Stage C: LLM fallback (only when evidence exists but rules fail)

Design Principles:
- Precision first (>= 0.95): Better to miss than to be wrong
- Evidence-driven: Every judgment must have page + text snippet
- No guessing: If no correspond* keyword evidence, never mark

Red Lines (MUST NOT mark as corresponding author if):
- No correspond*/通讯作者 keyword evidence found
- Mapping to author list is not unique
- Only email list / only asterisk / only equal contribution present
"""
import re
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass

try:
    from . import llm
    from . import config
except ImportError:
    llm = None
    config = None


# ==================== Configuration ====================

# Corresponding author keywords (English)
CORRESPOND_KEYWORDS_EN = [
    r'corresponding\s+author',
    r'correspondence\s+to',
    r'for\s+correspondence',
    r'\*\s*correspondence',
    r'correspondence:',
    r'✉',
    r'📧',
]

# Corresponding author keywords (Chinese)
CORRESPOND_KEYWORDS_ZH = [
    r'通讯作者',
    r'通信作者',
    r'通讯',
    r'联系人',
    r'通讯邮箱',
]

# All keywords combined
ALL_CORRESPOND_KEYWORDS = CORRESPOND_KEYWORDS_EN + CORRESPOND_KEYWORDS_ZH

# ==================== Co-First Author Keywords ====================
# IMPORTANT: Without these keywords, NEVER mark as co-first author
# even if asterisks (*) or emails are present

# Co-first author keywords (English) - MUST match to consider marking
COFIRST_KEYWORDS_EN = [
    r'contributed\s+equally',
    r'equal\s+contribution',
    r'co-first\s+author',
    r'co-first\s+authors',
    r'joint\s+first\s+author',
    r'joint\s+first\s+authors', 
    r'shared\s+first\s+authorship',
    r'these\s+authors\s+contributed\s+equally',
    r'authors\s+contributed\s+equally',
]

# Co-first author keywords (Chinese) - MUST match to consider marking
COFIRST_KEYWORDS_ZH = [
    r'共同第一作者',
    r'并列第一作者',
    r'等贡献',
    r'贡献相同',
    r'共一',
    r'一作同等贡献',
    r'同等贡献',
    r'共同贡献',
]

# All co-first keywords combined
ALL_COFIRST_KEYWORDS = COFIRST_KEYWORDS_EN + COFIRST_KEYWORDS_ZH

# Negative patterns (filter out equal contribution etc.) - for corresponding author only
NEGATIVE_PATTERNS = [
    r'contributed\s+equally',
    r'equal\s+contribution',
    r'co-first\s+author',
    r'共同一作',
    r'等贡献',
    r'共同贡献',
]

# Markers used to link authors to correspondence
MARKERS = {'*', '†', '‡', '§', '¶', '✉', '✱', '★', '✝', '†'}

# Email regex pattern
EMAIL_PATTERN = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'

# Character window size for evidence extraction
WINDOW_SIZE = 400


# ==================== Data Classes ====================

@dataclass
class EvidenceBlock:
    """Single evidence block from PDF"""
    page: int  # 1-based page number
    text: str  # Evidence text window
    keyword_matched: str  # Which keyword triggered this block
    start_pos: int  # Start position in page text
    end_pos: int  # End position in page text


@dataclass
class CorrespondingAuthorDecision:
    """Decision for a single corresponding author"""
    author_index: int  # 1-based index in author list
    name: str  # Matched author name
    email: Optional[str]  # Email if found
    page: int  # Evidence page
    confidence: float  # 0.0 - 1.0
    snippet: str  # Evidence text for audit
    method: str  # Detection method


# ==================== Stage A: Evidence Locating ====================

def _get_pages_to_scan(doc) -> List[int]:
    """
    Get page indices to scan (0-based).
    Strategy: pages 1, 2, and last (for footnotes).
    """
    total = doc.page_count
    pages = [0]  # First page always
    if total > 1:
        pages.append(1)  # Second page
    if total > 2:
        pages.append(total - 1)  # Last page (footnotes often here)
    return pages


def _extract_evidence_blocks_from_text(
    page_text: str,
    page_num: int,
    window_size: int = WINDOW_SIZE
) -> List[EvidenceBlock]:
    """
    Find all evidence blocks containing correspondence keywords in a page.
    
    Args:
        page_text: Text content of the page
        page_num: 1-based page number
        window_size: Character window around keyword match
        
    Returns:
        List of EvidenceBlock objects
    """
    blocks = []
    page_lower = page_text.lower()
    
    for pattern in ALL_CORRESPOND_KEYWORDS:
        for match in re.finditer(pattern, page_lower, re.IGNORECASE):
            start = max(0, match.start() - window_size // 2)
            end = min(len(page_text), match.end() + window_size // 2)
            
            # Extract window from original text (preserve case)
            window_text = page_text[start:end]
            
            blocks.append(EvidenceBlock(
                page=page_num,
                text=window_text,
                keyword_matched=pattern,
                start_pos=start,
                end_pos=end
            ))
    
    return blocks


def locate_evidence_blocks(pdf_bytes: bytes) -> List[EvidenceBlock]:
    """
    Stage A: Locate all evidence blocks containing correspondence keywords.
    
    This stage ONLY locates text blocks, does NOT make any judgment.
    Pure regex matching, no LLM, very fast.
    
    Args:
        pdf_bytes: PDF file content
        
    Returns:
        List of EvidenceBlock objects (may be empty)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[Correspond Detector] PyMuPDF not installed")
        return []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            doc.close()
            return []
        
        all_blocks = []
        pages_to_scan = _get_pages_to_scan(doc)
        
        for page_idx in pages_to_scan:
            if page_idx >= doc.page_count:
                continue
            page = doc[page_idx]
            page_text = page.get_text("text")
            
            if not page_text:
                continue
            
            # Find evidence blocks in this page
            blocks = _extract_evidence_blocks_from_text(
                page_text, 
                page_num=page_idx + 1  # Convert to 1-based
            )
            all_blocks.extend(blocks)
        
        doc.close()
        
        print(f"[Correspond Detector] Stage A: Found {len(all_blocks)} evidence blocks")
        return all_blocks
        
    except Exception as e:
        print(f"[Correspond Detector] Stage A error: {e}")
        return []


# ==================== Stage B: Rule Engine ====================

def _normalize_name(name: str) -> str:
    """
    Normalize author name for matching.
    - Lowercase
    - Remove dots (J. Smith -> J Smith)
    - Normalize whitespace
    - Handle common name patterns
    """
    if not name:
        return ""
    name = name.lower()
    name = re.sub(r'\.', ' ', name)  # J. Doe -> J Doe
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def _check_negative_patterns(text: str) -> bool:
    """
    Check if evidence window contains negative patterns.
    Returns True if negative pattern found (should discard).
    """
    text_lower = text.lower()
    for pattern in NEGATIVE_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    return False


def _extract_emails(text: str) -> List[str]:
    """Extract all email addresses from text."""
    return re.findall(EMAIL_PATTERN, text, re.IGNORECASE)


def _extract_name_patterns(text: str) -> List[str]:
    """
    Extract potential name patterns from correspondence text.
    Common patterns:
    - "Correspondence to: Yi Lu"
    - "Corresponding author: John Doe (john@...)"
    - "✉ John Doe"
    """
    names = []
    
    # Pattern 1: "Correspondence/Corresponding to/author: NAME"
    patterns = [
        r'correspond(?:ing|ence)?\s*(?:to|author)?[:\s]+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)',
        r'通讯作者[：:\s]+([^\s\(\)（）,，]+)',
        r'通信作者[：:\s]+([^\s\(\)（）,，]+)',
        r'✉\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        names.extend(matches)
    
    return [n.strip() for n in names if n and len(n) >= 2]


def _find_unique_author_match(
    evidence_name: str,
    authors: List[str]
) -> Optional[int]:
    """
    Find unique author match in author list.
    Returns 1-based index if unique match found, None otherwise.
    
    CRITICAL: Must return unique match only. Multiple candidates -> None.
    """
    if not evidence_name or not authors:
        return None
    
    evidence_norm = _normalize_name(evidence_name)
    candidates = []
    
    for idx, author in enumerate(authors):
        author_norm = _normalize_name(author)
        
        # Strategy 1: Exact match
        if evidence_norm == author_norm:
            candidates.append(idx + 1)
            continue
        
        # Strategy 2: Evidence name contains author name (or vice versa)
        if evidence_norm in author_norm or author_norm in evidence_norm:
            # Check if it's a substantial match (> 70%)
            shorter = min(len(evidence_norm), len(author_norm))
            longer = max(len(evidence_norm), len(author_norm))
            if shorter / longer > 0.5:
                candidates.append(idx + 1)
                continue
        
        # Strategy 3: Last name match (must be unique among authors)
        evidence_parts = evidence_norm.split()
        author_parts = author_norm.split()
        if evidence_parts and author_parts:
            # Check last name (usually last part)
            if evidence_parts[-1] == author_parts[-1]:
                candidates.append(idx + 1)
                continue
    
    # Only return if UNIQUE match
    if len(candidates) == 1:
        return candidates[0]
    
    return None


def _find_marker_mapping(
    evidence_text: str,
    authors: List[str],
    header_text: str = ""
) -> Optional[Tuple[int, str]]:
    """
    Find author via marker (*, †, ‡, ✉, etc.) mapping.
    
    Strategy:
    1. Find marker in evidence text near correspondence keyword
    2. Find same marker in header/author list area
    3. Map to author name
    
    Returns:
        (1-based author index, marker) if unique match, None otherwise
    """
    # Find markers in evidence text
    evidence_markers = set()
    for marker in MARKERS:
        if marker in evidence_text:
            evidence_markers.add(marker)
    
    if not evidence_markers:
        return None
    
    # Try to find author with same marker in header
    # (This is a simplified version - full implementation would need header text)
    # For now, we look for patterns like "Author Name*" or "Author Name†"
    
    candidates = []
    for marker in evidence_markers:
        for idx, author in enumerate(authors):
            # Check if author name appears with this marker in evidence
            escaped_marker = re.escape(marker)
            pattern = rf'{re.escape(author)}\s*{escaped_marker}'
            if re.search(pattern, evidence_text, re.IGNORECASE):
                candidates.append((idx + 1, marker))
    
    # Only return if unique
    if len(candidates) == 1:
        return candidates[0]
    
    return None


def apply_rule_engine(
    evidence_blocks: List[EvidenceBlock],
    authors: List[str]
) -> List[CorrespondingAuthorDecision]:
    """
    Stage B: Apply rule engine to determine corresponding authors.
    
    Three types of strong evidence:
    - Type I: Keyword + explicit name (+ optional email)
    - Type II: Keyword + email + marker closed loop
    - Type III: Keyword + email + adjacent author name (weaker)
    
    Args:
        evidence_blocks: Evidence blocks from Stage A
        authors: Author list from Semantic Scholar
        
    Returns:
        List of CorrespondingAuthorDecision objects
    """
    if not evidence_blocks or not authors:
        return []
    
    decisions = []
    seen_indices = set()  # Avoid duplicates
    
    for block in evidence_blocks:
        # Skip if contains negative patterns
        if _check_negative_patterns(block.text):
            print(f"[Correspond Detector] Skipping block with negative pattern: {block.text[:50]}...")
            continue
        
        # === Type I: Keyword + explicit name ===
        names = _extract_name_patterns(block.text)
        emails = _extract_emails(block.text)
        
        for name in names:
            author_idx = _find_unique_author_match(name, authors)
            if author_idx and author_idx not in seen_indices:
                # Found unique match via explicit name
                email = emails[0] if emails else None
                confidence = 0.95
                
                decisions.append(CorrespondingAuthorDecision(
                    author_index=author_idx,
                    name=authors[author_idx - 1],
                    email=email,
                    page=block.page,
                    confidence=confidence,
                    snippet=block.text[:100],
                    method="rule_type_I"
                ))
                seen_indices.add(author_idx)
                print(f"[Correspond Detector] Type I match: {authors[author_idx - 1]} (confidence: {confidence})")
        
        # === Type II: Keyword + email + marker closed loop ===
        if emails:
            marker_result = _find_marker_mapping(block.text, authors)
            if marker_result:
                author_idx, marker = marker_result
                if author_idx not in seen_indices:
                    decisions.append(CorrespondingAuthorDecision(
                        author_index=author_idx,
                        name=authors[author_idx - 1],
                        email=emails[0],
                        page=block.page,
                        confidence=0.90,
                        snippet=block.text[:100],
                        method="rule_type_II"
                    ))
                    seen_indices.add(author_idx)
                    print(f"[Correspond Detector] Type II match: {authors[author_idx - 1]} via marker '{marker}'")
        
        # === Type III: Keyword + email + adjacent name (weaker) ===
        # Try matching email domain/username to author name
        for email in emails:
            # Extract potential name from email (e.g., john.doe@... -> john doe)
            local_part = email.split('@')[0]
            # Common patterns: john.doe, johndoe, j.doe
            name_from_email = re.sub(r'[._-]', ' ', local_part)
            
            author_idx = _find_unique_author_match(name_from_email, authors)
            if author_idx and author_idx not in seen_indices:
                # Check if this author name also appears near the keyword
                author_name_lower = authors[author_idx - 1].lower()
                if author_name_lower in block.text.lower():
                    decisions.append(CorrespondingAuthorDecision(
                        author_index=author_idx,
                        name=authors[author_idx - 1],
                        email=email,
                        page=block.page,
                        confidence=0.85,
                        snippet=block.text[:100],
                        method="rule_type_III"
                    ))
                    seen_indices.add(author_idx)
                    print(f"[Correspond Detector] Type III match: {authors[author_idx - 1]} via email pattern")
    
    # Filter by confidence threshold
    decisions = [d for d in decisions if d.confidence >= 0.85]
    
    print(f"[Correspond Detector] Stage B: {len(decisions)} decisions passed threshold")
    return decisions


# ==================== Stage C: LLM Fallback ====================

# LLM prompt for parsing correspondence info
LLM_PARSE_PROMPT = """You are an academic paper metadata parser. Given a small text window from a paper's first page that contains corresponding author information, extract the corresponding author(s).

**STRICT RULES:**
1. ONLY output if you see explicit keywords: "Corresponding author", "Correspondence to", "For correspondence", "通讯作者", "通信作者", or the ✉ symbol
2. You MUST provide an "evidence_quote" - a verbatim 10-30 character snippet from the input text that proves your finding
3. If information is unclear or no correspondence keyword is present, return an empty array []
4. DO NOT guess or infer - only extract what is explicitly stated

**Output Format (pure JSON, no markdown):**
[
  {{"name": "Author Full Name", "email": "email@example.com", "evidence_quote": "verbatim quote from text"}}
]

If no corresponding author can be determined, output: []

--- TEXT WINDOW ---
{window_text}
--- END ---

JSON:"""


def apply_llm_fallback(
    evidence_blocks: List[EvidenceBlock],
    authors: List[str],
    api_key: str = None
) -> List[CorrespondingAuthorDecision]:
    """
    Stage C: LLM fallback for evidence blocks that rules couldn't parse.
    
    STRICT trigger conditions:
    - Evidence blocks exist (already located correspond* keyword)
    - Rule engine couldn't extract name/marker mapping
    
    LLM is used as a PARSER, not an INFERRER.
    
    Args:
        evidence_blocks: Evidence blocks from Stage A
        authors: Author list from Semantic Scholar
        api_key: Optional API key
        
    Returns:
        List of CorrespondingAuthorDecision objects
    """
    if not evidence_blocks or not authors:
        return []
    
    if llm is None:
        print("[Correspond Detector] LLM module not available, skipping Stage C")
        return []
    
    decisions = []
    seen_indices = set()
    
    # Only process first 2 evidence blocks to limit LLM calls
    for block in evidence_blocks[:2]:
        try:
            prompt = LLM_PARSE_PROMPT.format(window_text=block.text)
            
            llm_instance = llm.get_llm("extract", temperature=0.0, api_key=api_key)
            response = llm_instance.invoke(prompt)
            
            if hasattr(response, 'content'):
                result = response.content.strip()
            else:
                result = str(response).strip()
            
            # Parse JSON response
            import json
            
            # Clean up markdown code blocks
            result = re.sub(r'^```json\s*', '', result)
            result = re.sub(r'^```\s*', '', result)
            result = re.sub(r'\s*```$', '', result)
            result = result.strip()
            
            # Extract JSON array
            if not result.startswith('['):
                start = result.find('[')
                end = result.rfind(']')
                if start >= 0 and end > start:
                    result = result[start:end+1]
                else:
                    continue
            
            parsed = json.loads(result)
            
            if not isinstance(parsed, list):
                continue
            
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                
                name = item.get('name', '')
                email = item.get('email', '')
                evidence_quote = item.get('evidence_quote', '')
                
                # CRITICAL: Must have evidence_quote
                if not evidence_quote or len(evidence_quote) < 5:
                    print(f"[Correspond Detector] LLM result rejected: no evidence_quote")
                    continue
                
                # Verify evidence_quote exists in original text
                if evidence_quote.lower() not in block.text.lower():
                    print(f"[Correspond Detector] LLM result rejected: evidence_quote not in original text")
                    continue
                
                # Try to match to author list (must be unique)
                author_idx = _find_unique_author_match(name, authors)
                if author_idx and author_idx not in seen_indices:
                    decisions.append(CorrespondingAuthorDecision(
                        author_index=author_idx,
                        name=authors[author_idx - 1],
                        email=email if email else None,
                        page=block.page,
                        confidence=0.80,  # Lower confidence for LLM
                        snippet=evidence_quote,
                        method="llm_fallback"
                    ))
                    seen_indices.add(author_idx)
                    print(f"[Correspond Detector] LLM fallback match: {authors[author_idx - 1]}")
                    
        except Exception as e:
            print(f"[Correspond Detector] Stage C error: {e}")
            continue
    
    return decisions


# ==================== Main Detection Function ====================

def detect_corresponding_authors(
    pdf_bytes: bytes,
    authors: List[str],
    api_key: str = None,
    use_llm_fallback: bool = True
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Main function: Detect corresponding authors from PDF.
    
    Three-stage pipeline:
    1. Stage A: Locate evidence blocks (pure regex, fast)
    2. Stage B: Apply rule engine (core logic)
    3. Stage C: LLM fallback (only if rules fail but evidence exists)
    
    Args:
        pdf_bytes: PDF file content
        authors: Author list from Semantic Scholar (for matching)
        api_key: Optional API key for LLM
        use_llm_fallback: Whether to use LLM fallback (default True)
        
    Returns:
        (corresponding_indices, evidences)
        - corresponding_indices: List of 1-based author indices
        - evidences: List of evidence dicts for storage/audit
    """
    if not pdf_bytes or not authors:
        return [], []
    
    print(f"[Correspond Detector] Starting detection for {len(authors)} authors")
    
    # Stage A: Locate evidence blocks
    evidence_blocks = locate_evidence_blocks(pdf_bytes)
    
    if not evidence_blocks:
        print("[Correspond Detector] No evidence blocks found, exiting")
        return [], []
    
    # Stage B: Apply rule engine
    decisions = apply_rule_engine(evidence_blocks, authors)
    
    # Stage C: LLM fallback if rules didn't find anything but evidence exists
    if not decisions and use_llm_fallback:
        print("[Correspond Detector] Rules found nothing, trying LLM fallback")
        decisions = apply_llm_fallback(evidence_blocks, authors, api_key)
    
    if not decisions:
        print("[Correspond Detector] No corresponding authors detected")
        return [], []
    
    # Convert to output format
    indices = [d.author_index for d in decisions]
    evidences = [
        {
            "author_index": d.author_index,
            "name": d.name,
            "email": d.email,
            "page": d.page,
            "confidence": d.confidence,
            "snippet": d.snippet,
            "method": d.method
        }
        for d in decisions
    ]
    
    print(f"[Correspond Detector] Detected {len(indices)} corresponding author(s): {[d.name for d in decisions]}")
    return indices, evidences


def detect_corresponding_authors_from_url(
    pdf_url: str,
    authors: List[str],
    api_key: str = None,
    use_llm_fallback: bool = True,
    max_bytes: int = 20 * 1024 * 1024,
    timeout: float = 15.0
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Convenience function: Download PDF and detect corresponding authors.
    
    Args:
        pdf_url: URL to PDF file
        authors: Author list from Semantic Scholar
        api_key: Optional API key for LLM
        use_llm_fallback: Whether to use LLM fallback
        max_bytes: Maximum PDF size to download
        timeout: Download timeout
        
    Returns:
        (corresponding_indices, evidences)
    """
    if not pdf_url or not authors:
        return [], []
    
    # Import PDF download function
    try:
        from .pdf_introduction_extractor import download_pdf_safely
    except ImportError:
        print("[Correspond Detector] Cannot import download_pdf_safely")
        return [], []
    
    # Download PDF
    pdf_bytes = download_pdf_safely(pdf_url, max_bytes=max_bytes, timeout=timeout)
    
    if not pdf_bytes:
        print(f"[Correspond Detector] Failed to download PDF: {pdf_url[:50]}...")
        return [], []
    
    return detect_corresponding_authors(pdf_bytes, authors, api_key, use_llm_fallback)


# ==================== Co-First Author Detection ====================

@dataclass
class CoFirstAuthorDecision:
    """Decision for a co-first author"""
    author_index: int  # 1-based index in author list
    name: str  # Matched author name
    page: int  # Evidence page
    confidence: float  # 0.0 - 1.0
    snippet: str  # Evidence text for audit
    method: str  # Detection method


def _extract_cofirst_evidence_blocks_from_text(
    page_text: str,
    page_num: int,
    window_size: int = WINDOW_SIZE
) -> List[EvidenceBlock]:
    """
    Find all evidence blocks containing co-first author keywords in a page.
    
    CRITICAL: Only match co-first keywords. No keyword = no co-first detection.
    """
    blocks = []
    page_lower = page_text.lower()
    
    for pattern in ALL_COFIRST_KEYWORDS:
        for match in re.finditer(pattern, page_lower, re.IGNORECASE):
            start = max(0, match.start() - window_size // 2)
            end = min(len(page_text), match.end() + window_size // 2)
            
            # Extract window from original text (preserve case)
            window_text = page_text[start:end]
            
            blocks.append(EvidenceBlock(
                page=page_num,
                text=window_text,
                keyword_matched=pattern,
                start_pos=start,
                end_pos=end
            ))
    
    return blocks


def locate_cofirst_evidence_blocks(pdf_bytes: bytes) -> List[EvidenceBlock]:
    """
    Stage A for Co-First: Locate all evidence blocks containing co-first keywords.
    
    CRITICAL: Without matching keywords, this returns empty list.
    No asterisks (*) or other markers alone can trigger co-first detection.
    
    Args:
        pdf_bytes: PDF file content
        
    Returns:
        List of EvidenceBlock objects (may be empty)
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[CoFirst Detector] PyMuPDF not installed")
        return []
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            doc.close()
            return []
        
        all_blocks = []
        pages_to_scan = _get_pages_to_scan(doc)
        
        for page_idx in pages_to_scan:
            if page_idx >= doc.page_count:
                continue
            page = doc[page_idx]
            page_text = page.get_text("text")
            
            if not page_text:
                continue
            
            # Find co-first evidence blocks in this page
            blocks = _extract_cofirst_evidence_blocks_from_text(
                page_text, 
                page_num=page_idx + 1  # Convert to 1-based
            )
            all_blocks.extend(blocks)
        
        doc.close()
        
        print(f"[CoFirst Detector] Stage A: Found {len(all_blocks)} evidence blocks")
        return all_blocks
        
    except Exception as e:
        print(f"[CoFirst Detector] Stage A error: {e}")
        return []


def _find_cofirst_authors_from_markers(
    evidence_text: str,
    authors: List[str]
) -> List[int]:
    """
    Find co-first authors by marker mapping (* or †).
    
    Strategy:
    1. Find markers (*, †, etc.) in evidence text near co-first keyword
    2. Find authors with same marker
    3. Return all matching author indices (1-based)
    
    Returns:
        List of 1-based author indices
    """
    # Common markers for co-first authors
    cofirst_markers = {'*', '†', '‡', '¹', '²', '³'}
    
    # Find markers in evidence text
    evidence_markers = set()
    for marker in cofirst_markers:
        if marker in evidence_text:
            evidence_markers.add(marker)
    
    if not evidence_markers:
        return []
    
    # Find authors with matching markers
    matched_indices = []
    for marker in evidence_markers:
        for idx, author in enumerate(authors):
            # Check if author name appears with this marker in evidence
            escaped_marker = re.escape(marker)
            # Pattern: Author Name* or Author Name†
            pattern = rf'{re.escape(author)}\s*{escaped_marker}'
            if re.search(pattern, evidence_text, re.IGNORECASE):
                matched_indices.append(idx + 1)  # 1-based
    
    return list(set(matched_indices))  # Remove duplicates


def _infer_cofirst_from_position(
    evidence_text: str,
    authors: List[str],
    max_cofirst: int = 3
) -> List[int]:
    """
    Infer co-first authors from position when markers don't work.
    
    Common patterns:
    - "A and B contributed equally" -> first 2 authors
    - "A, B, and C contributed equally" -> first 3 authors
    - "These authors contributed equally: A, B" -> named authors
    
    Strategy:
    1. Try to find explicit author names in evidence
    2. Fallback to first N authors if keyword matches but no specific names
    
    Returns:
        List of 1-based author indices
    """
    if not authors:
        return []
    
    # First try to find specific author names in evidence
    matched_indices = []
    for idx, author in enumerate(authors):
        if _normalize_name(author) in _normalize_name(evidence_text):
            matched_indices.append(idx + 1)
    
    # If found specific names, use them (but cap at max_cofirst from front)
    if matched_indices:
        # Sort and take only the first few (co-first are usually at front)
        matched_indices.sort()
        front_authors = [i for i in matched_indices if i <= max_cofirst]
        if front_authors:
            return front_authors
    
    # Fallback: If evidence has keywords but no specific mapping,
    # assume first 2 authors are co-first (most common pattern)
    # Only do this if we have the keyword evidence
    if len(authors) >= 2:
        return [1, 2]  # First and second authors
    elif len(authors) == 1:
        return [1]
    
    return []


def apply_cofirst_rule_engine(
    evidence_blocks: List[EvidenceBlock],
    authors: List[str]
) -> List[CoFirstAuthorDecision]:
    """
    Stage B for Co-First: Apply rule engine to determine co-first authors.
    
    CRITICAL: Must have keyword evidence to trigger. No keyword = no co-first.
    
    Args:
        evidence_blocks: Evidence blocks from Stage A
        authors: Author list from Semantic Scholar
        
    Returns:
        List of CoFirstAuthorDecision objects
    """
    if not evidence_blocks or not authors:
        return []
    
    decisions = []
    seen_indices = set()  # Avoid duplicates
    
    for block in evidence_blocks:
        # Try marker-based detection first (more reliable)
        marker_indices = _find_cofirst_authors_from_markers(block.text, authors)
        
        if marker_indices:
            for author_idx in marker_indices:
                if author_idx not in seen_indices and author_idx <= len(authors):
                    decisions.append(CoFirstAuthorDecision(
                        author_index=author_idx,
                        name=authors[author_idx - 1],
                        page=block.page,
                        confidence=0.95,
                        snippet=block.text[:100],
                        method="cofirst_marker"
                    ))
                    seen_indices.add(author_idx)
                    print(f"[CoFirst Detector] Marker match: {authors[author_idx - 1]}")
        else:
            # Fallback: infer from position/context
            inferred_indices = _infer_cofirst_from_position(block.text, authors)
            
            for author_idx in inferred_indices:
                if author_idx not in seen_indices and author_idx <= len(authors):
                    decisions.append(CoFirstAuthorDecision(
                        author_index=author_idx,
                        name=authors[author_idx - 1],
                        page=block.page,
                        confidence=0.85,  # Lower confidence for inferred
                        snippet=block.text[:100],
                        method="cofirst_inferred"
                    ))
                    seen_indices.add(author_idx)
                    print(f"[CoFirst Detector] Inferred match: {authors[author_idx - 1]}")
    
    # Filter by confidence threshold
    decisions = [d for d in decisions if d.confidence >= 0.80]
    
    print(f"[CoFirst Detector] Stage B: {len(decisions)} co-first authors detected")
    return decisions


def detect_cofirst_authors(
    pdf_bytes: bytes,
    authors: List[str]
) -> Tuple[List[int], List[Dict[str, Any]]]:
    """
    Main function: Detect co-first authors from PDF.
    
    CRITICAL RULE: Without co-first keywords, NEVER mark as co-first.
    Even if asterisks (*) or other markers are present, without the
    explicit keyword evidence, this function returns empty.
    
    Two-stage pipeline (no LLM needed for co-first):
    1. Stage A: Locate evidence blocks (pure regex, fast)
    2. Stage B: Apply rule engine (marker/position-based)
    
    Args:
        pdf_bytes: PDF file content
        authors: Author list from Semantic Scholar (for matching)
        
    Returns:
        (cofirst_indices, evidences)
        - cofirst_indices: List of 1-based author indices
        - evidences: List of evidence dicts for storage/audit
    """
    if not pdf_bytes or not authors:
        return [], []
    
    print(f"[CoFirst Detector] Starting detection for {len(authors)} authors")
    
    # Stage A: Locate evidence blocks (MUST find keyword to proceed)
    evidence_blocks = locate_cofirst_evidence_blocks(pdf_bytes)
    
    if not evidence_blocks:
        print("[CoFirst Detector] No co-first keyword evidence found, exiting")
        return [], []
    
    # Stage B: Apply rule engine
    decisions = apply_cofirst_rule_engine(evidence_blocks, authors)
    
    if not decisions:
        print("[CoFirst Detector] No co-first authors detected")
        return [], []
    
    # Convert to output format
    indices = [d.author_index for d in decisions]
    evidences = [
        {
            "author_index": d.author_index,
            "name": d.name,
            "page": d.page,
            "confidence": d.confidence,
            "snippet": d.snippet,
            "method": d.method
        }
        for d in decisions
    ]
    
    print(f"[CoFirst Detector] Detected {len(indices)} co-first author(s): {[d.name for d in decisions]}")
    return indices, evidences
