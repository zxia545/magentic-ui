"""
PDF Introduction Extractor

Extracts introduction section from PDF using layered approach.

Pipeline (4 layers):
1. Rule-based extraction (fast, free, ~85% success)
2. LLM locate + rule slice (10 output tokens, preserves original text)
3. LLM summarize (500 output tokens, when boundaries unclear)
4. Fallback to TLDR

Design Principle (规则与LLM职责分离):
- Rules: Handle precise pattern matching and text extraction (fast, free)
- LLM Locate: Find boundaries only, output ~10 tokens (efficient)
- LLM Summarize: Content understanding, output ~500 tokens (last resort)
- Never use LLM for simple copy-paste (waste of tokens)
"""
import io
import re
import time
import requests
from typing import Optional, Tuple, List
from . import config
from . import llm

# ============ CONFIGURATION ============
MAX_INTRO_CHARS = 3000  # Max introduction length to return
MIN_INTRO_CHARS = 200   # Min valid introduction length
PAGE_EXPANSION = [2, 4, 6]  # Progressive page expansion

# Introduction title patterns (case-insensitive)
# Enhanced with more format variations to reduce LLM fallback
INTRO_PATTERNS = [
    # Standard formats with optional punctuation
    r'^\s*1\.?\s+Introduction\s*$',
    r'^\s*I\.?\s+Introduction\s*$',
    r'^\s*Introduction\s*$',
    r'^\s*INTRODUCTION\s*$',
    r'^\s*1\.?\s+INTRODUCTION\s*$',
    r'^\s*I\.?\s+INTRODUCTION\s*$',
    
    # Section prefix variants
    r'^\s*Section\s+1:?\s+Introduction\s*$',
    r'^\s*Section\s+I:?\s+Introduction\s*$',
    r'^\s*Sec\.?\s+1\.?\s+Introduction\s*$',
    
    # Number without punctuation (common in PDFs)
    r'^\s*1\s+Introduction\s*$',
    r'^\s*I\s+Introduction\s*$',
    
    # Alternative intro-like titles
    r'^\s*1\.?\s+Overview\s*$',
    r'^\s*1\.?\s+Motivation\s*$',
    r'^\s*1\.?\s+Background\s*$',
    r'^\s*1\.?\s+Preliminaries\s*$',
    
    # Multi-word variants
    r'^\s*1\.?\s+Introduction\s+and\s+Motivation\s*$',
    r'^\s*1\.?\s+Background\s+and\s+Introduction\s*$',
]

# Next section patterns (to find intro end)
# Enhanced with more flexible matching to reduce LLM fallback
NEXT_SECTION_PATTERNS = [
    # Section 2 with various punctuation
    r'^\s*2\.\s+[A-Z]',          # "2. Related Work" - must have dot and capital
    r'^\s*2\s+[A-Z][a-z]+',       # "2 Related" - number space Capital
    r'^\s*II\.?\s+[A-Z]',         # Roman numeral
    r'^\s*Section\s+2:?\s+',      # "Section 2: Related Work"
    r'^\s*Sec\.?\s+2\.?\s+',    # "Sec. 2. Related Work"
    
    # Common next section titles (case-insensitive via flag)
    r'^\s*RELATED\s+WORK\s*$',    # All caps, standalone
    r'^\s*Related\s+Work\s*$',    # Title case, standalone
    r'^\s*BACKGROUND\s*$',
    r'^\s*Background\s*$',
    r'^\s*PRELIMINARIES\s*$',
    r'^\s*Preliminaries\s*$',
    r'^\s*METHODOLOGY\s*$',
    r'^\s*Methodology\s*$',
    r'^\s*METHOD\s*$',
    r'^\s*Method\s*$',
    r'^\s*METHODS\s*$',
    r'^\s*Methods\s*$',
    r'^\s*APPROACH\s*$',
    r'^\s*Approach\s*$',
    r'^\s*OUR\s+APPROACH\s*$',
    r'^\s*Our\s+Approach\s*$',
    r'^\s*PROBLEM\s+FORMULATION\s*$',
    r'^\s*Problem\s+Formulation\s*$',
    r'^\s*EXPERIMENTAL\s+SETUP\s*$',
    r'^\s*Experimental\s+Setup\s*$',
    
    # Multi-word variants
    r'^\s*2\.?\s+Background\s+and\s+Related\s+Work\s*$',
    r'^\s*2\.?\s+Related\s+Work\s+and\s+Background\s*$',
    r'^\s*2\.?\s+Prior\s+(Work|Art|Research)\s*$',
    r'^\s*2\.?\s+Literature\s+Review\s*$',
]

# Abstract patterns
ABSTRACT_PATTERNS = [
    r'^\s*Abstract\s*$',
    r'^\s*ABSTRACT\s*$',
]
# PDF Size limit for download
MAX_PDF_SIZE = 40 * 1024 * 1024  # 40MB

def download_pdf_partial(pdf_url: str, max_bytes: int = 20 * 1024 * 1024, timeout: float = 30.0) -> Optional[bytes]:
    """
    Download PDF file, attempting complete download for smaller files.
    Strategy:
    1. Check Content-Length header first
    2. If file <= max_bytes: download complete file
    3. If file > max_bytes: skip (truncated PDF is useless)
    Args:
        pdf_url: URL to PDF file
        max_bytes: Maximum file size to download (20MB default)
        timeout: Request timeout
        
    Returns:
        PDF bytes or None if failed/too large
    """
    if not pdf_url:
        return None
    # Normalize arXiv URL to ensure .pdf extension
    if 'arxiv.org/pdf/' in pdf_url and not pdf_url.endswith('.pdf'):
        # Handle URLs like https://arxiv.org/pdf/2310.01361 -> add .pdf
        if not pdf_url.endswith('.pdf'):
            pdf_url = pdf_url.rstrip('/') + '.pdf'

    # Retry configuration
    max_retries = 3
    retry_delay = 2  # seconds
    for attempt in range(max_retries):
        try:
            # First, make HEAD request to check Content-Length
            head_response = requests.head(
                pdf_url,
                timeout=10,
                headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')},
                allow_redirects=True
            )
            content_length = head_response.headers.get('Content-Length')
            if content_length:
                file_size = int(content_length)
                if file_size > max_bytes:
                    print(f"[PDF Download] File too large ({file_size / 1024 / 1024:.1f}MB > {max_bytes / 1024 / 1024:.0f}MB limit), skipping: {pdf_url[:50]}...")
                    return None
                print(f"[PDF Download] File size: {file_size / 1024:.0f}KB, downloading complete file...")
            
            # Download complete file with stream=True to handle large files
            response = requests.get(
                pdf_url,
                timeout=timeout,
                headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')},
                stream=True  # Stream download to avoid connection reset
            )
            
            if response.status_code != 200:
                print(f"[PDF Download] HTTP {response.status_code}: {pdf_url[:60]}...")
                return None
            
            # Check content type
            content_type = response.headers.get('Content-Type', '')
            if 'pdf' not in content_type.lower() and 'octet-stream' not in content_type.lower():
                print(f"[PDF Download] Not a PDF: {content_type}")
                return None
            
            # Download in chunks to avoid connection reset
            pdf_bytes = b''
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_bytes += chunk
                    if len(pdf_bytes) > max_bytes:
                        print(f"[PDF Download] Downloaded file exceeds limit, stopping")
                        return None
            
            print(f"[PDF Download] Downloaded {len(pdf_bytes) / 1024:.0f}KB from {pdf_url[:50]}...")
            return pdf_bytes
            
        except (requests.exceptions.ConnectionError, ConnectionResetError) as e:
            if attempt < max_retries - 1:
                print(f"[PDF Download] Connection reset (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                print(f"[PDF Download] Connection failed after {max_retries} attempts: {e}")
                return None
        
        except requests.RequestException as e:
            print(f"[PDF Download] Error: {e}")
            return None
    
    return None

download_pdf_safely = download_pdf_partial
def extract_first_pages_text(pdf_bytes: bytes, max_pages: int = 2) -> Optional[str]:
    """
    Extract text from first N pages of PDF.
    Args:
        pdf_bytes: PDF file content
        max_pages: Maximum pages to extract (default 2)
    Returns:
        Extracted text or None if failed
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[PDF Extract] PyMuPDF not installed. Run: pip install pymupdf")
        return None
    try:
        # Open PDF from bytes
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            print("[PDF Extract] Empty PDF")
            doc.close()
            return None
        # Extract text from first N pages
        pages_to_extract = min(max_pages, doc.page_count)
        text_parts = []
        for page_num in range(pages_to_extract):
            page = doc[page_num]
            text = page.get_text("text")
            if text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
        doc.close()
        full_text = "\n\n".join(text_parts)
        print(f"[PDF Extract] Extracted {len(full_text)} chars from {pages_to_extract} pages")
        return full_text
    except Exception as e:
        print(f"[PDF Extract] Error: {e}")
        return None

def _find_line_matching(lines: List[str], patterns: List[str], start_idx: int = 0) -> int:
    """
    Find first line matching any pattern.
    Returns line index or -1 if not found.
    """
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        for pattern in patterns:
            if re.match(pattern, line, re.IGNORECASE):
                return i
    return -1
def extract_authors_with_affiliations(pdf_url: str, paper_title: str, api_key: str = None) -> List[dict]:
    """
    Extract authors and their affiliations from PDF header area (top 25% of first page).
    This is a FALLBACK method when Semantic Scholar API doesn't provide affiliations.
    Uses layout-aware extraction to focus on the paper header where author info lives.
    Args:
        pdf_url: URL to the PDF file
        paper_title: Paper title for context
        api_key: Optional API key for LLM
    Returns:
        List of dicts with 'name' and 'affiliations' keys
        Example: [{'name': 'John Doe', 'affiliations': ['MIT', 'Google']}, ...]
    """
    try:
        # Step 1: Download PDF (with size limit)
        pdf_bytes = download_pdf_safely(pdf_url, max_bytes=MAX_PDF_SIZE, timeout=15)
        
        if not pdf_bytes:
            print(f"[PDF Authors] Failed to download PDF")
            return []
        
        # Step 2: Extract ONLY top 25% of first page (header area)
        header_text = _extract_pdf_header_area(pdf_bytes)
        
        if not header_text or len(header_text) < 50:
            print(f"[PDF Authors] Header text too short: {len(header_text) if header_text else 0} chars")
            return []
        
        print(f"[PDF Authors] Extracted {len(header_text)} chars from PDF header")
        
        # Step 3: Use LLM to parse authors and affiliations
        # Truncate if still too long
        if len(header_text) > 2000:
            header_text = header_text[:2000] + "\n... [truncated]"
        
        prompt = f"""Extract ALL authors and their affiliations from this academic paper header.

        Paper title: "{paper_title}"

        **CRITICAL RULES**:
        1. Extract EXACT affiliations as written (DO NOT expand abbreviations)
        2. Each author may have multiple affiliations
        3. Output ONLY valid JSON array, nothing else
        4. If you cannot find clear author/affiliation mapping, return []

        **Output Format** (pure JSON, no markdown):
        [
        {{"name": "John Doe", "affiliations": ["MIT", "Google Research"]}},
        {{"name": "Jane Smith", "affiliations": ["Stanford"]}}
        ]

        --- PAPER HEADER ---
        {header_text}
        --- END ---

        JSON:"""
        llm_instance = llm.get_llm("extract", temperature=0.0, api_key=api_key)
        response = llm_instance.invoke(prompt)
        if hasattr(response, 'content'):
            result = response.content.strip()
        else:
            result = str(response).strip()
        # Parse JSON with robust bracket extraction
        import json
        # Clean up markdown code blocks
        result = re.sub(r'^```json\s*', '', result)
        result = re.sub(r'^```\s*', '', result)
        result = re.sub(r'\s*```$', '', result)
        result = result.strip()
        # Extract JSON array using bracket matching (fallback)
        if not result.startswith('['):
            # Find first [ and last ]
            start = result.find('[')
            end = result.rfind(']')
            if start >= 0 and end > start:
                result = result[start:end+1]
            else:
                print(f"[PDF Authors] No JSON array found in response")
                return []
        try:
            authors_list = json.loads(result)
            # Validate structure
            if not isinstance(authors_list, list):
                print(f"[PDF Authors] Invalid JSON structure: {type(authors_list)}")
                return []
            # Normalize and filter
            valid_authors = []
            for author in authors_list:
                if isinstance(author, dict) and 'name' in author:
                    name = author.get('name', '').strip()
                    affiliations_raw = author.get('affiliations', [])
                    
                    # Handle both string and list formats
                    if isinstance(affiliations_raw, str):
                        affiliations = [affiliations_raw.strip()] if affiliations_raw.strip() else []
                    elif isinstance(affiliations_raw, list):
                        affiliations = [aff.strip() for aff in affiliations_raw if aff and aff.strip()]
                    else:
                        affiliations = []
                    
                    if name and len(name) >= config.MIN_AUTHOR_NAME_LENGTH:
                        valid_authors.append({
                            'name': name,
                            'affiliations': affiliations if affiliations else None
                        })
            print(f"[PDF Authors] Extracted {len(valid_authors)} authors from PDF header")
            return valid_authors[:config.MAX_AUTHORS]
        except json.JSONDecodeError as e:
            print(f"[PDF Authors] Failed to parse JSON: {e}")
            print(f"[PDF Authors] Raw response: {result[:200]}")
            return []
        
    except Exception as e:
        print(f"[PDF Authors] Extraction error: {e}")
        return []


def _extract_pdf_header_area(pdf_bytes: bytes, top_percentage: float = 0.25) -> Optional[str]:
    """
    Extract text from the top portion of the first PDF page (layout-aware).
    Author/affiliation information is typically in the top 20-30% of the first page.
    This focuses extraction on that region to reduce noise.
    Args:
        pdf_bytes: PDF file content
        top_percentage: Fraction of page height to extract (default 0.25 = top 25%)
    Returns:
        Extracted text from header area or None
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        print("[PDF Header] PyMuPDF not installed")
        return None
    
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        
        if doc.page_count == 0:
            print("[PDF Header] Empty PDF")
            doc.close()
            return None
        
        page = doc[0]  # First page only
        page_rect = page.rect  # Get page dimensions
        page_height = page_rect.height
        
        # Define header area (top 25% of page)
        header_height = page_height * top_percentage
        header_rect = fitz.Rect(0, 0, page_rect.width, header_height)
        
        # Extract text from header area only
        header_text = page.get_text(clip=header_rect)
        
        doc.close()
        
        if not header_text or len(header_text) < 50:
            print(f"[PDF Header] Header text too short: {len(header_text) if header_text else 0} chars")
            return None
        
        print(f"[PDF Header] Extracted {len(header_text)} chars from top {top_percentage*100:.0f}% of page")
        return header_text
        
    except Exception as e:
        print(f"[PDF Header] Error: {e}")
        return None

def get_paper_authors_with_affiliations(
    paper_id: str,
    paper_title: str,
    pdf_url: Optional[str],
    s2_client,
    api_key: str = None
) -> List[dict]:
    """
    Get authors with affiliations using layered strategy.
    Strategy:
    1. Try Semantic Scholar API (fast, structured, ~60-70% has affiliations)
    2. Fallback to PDF extraction (slower, LLM-based, for missing affiliations)
    Args:
        paper_id: Semantic Scholar paper ID
        paper_title: Paper title
        pdf_url: URL to PDF (optional)
        s2_client: SemanticScholarSearchClient instance
        api_key: Optional API key for LLM
    Returns:
        List of dicts: [{'name': 'X', 'author_id': 'Y', 'affiliations': ['Z', ...]}, ...]
    """
    # Layer 1: Try Semantic Scholar API first
    print(f"[Authors] Layer 1: Trying Semantic Scholar API for paper {paper_id}")
    
    s2_authors = s2_client.get_paper_authors_with_affiliations(paper_id)
    
    if not s2_authors:
        print(f"[Authors] S2 API returned no authors, trying PDF fallback")
        # Fallback to PDF extraction
        if pdf_url:
            pdf_authors = extract_authors_with_affiliations(pdf_url, paper_title, api_key)
            return pdf_authors
        else:
            print(f"[Authors] No PDF URL available")
            return []
    
    # Check which authors are missing affiliations
    authors_missing_affiliations = [
        (i, author) for i, author in enumerate(s2_authors)
        if not author.affiliations
    ]
    
    if not authors_missing_affiliations:
        # All authors have affiliations from S2
        print(f"[Authors] All {len(s2_authors)} authors have affiliations from S2")
        return [
            {
                'name': author.name,
                'author_id': author.authorId,
                'affiliations': author.affiliations
            }
            for author in s2_authors
        ]
    
    # Layer 2: Some authors missing affiliations, try PDF extraction
    print(f"[Authors] {len(authors_missing_affiliations)}/{len(s2_authors)} authors missing affiliations")
    
    if not pdf_url:
        print(f"[Authors] No PDF URL for fallback, returning partial data")
        return [
            {
                'name': author.name,
                'author_id': author.authorId,
                'affiliations': author.affiliations
            }
            for author in s2_authors
        ]
    
    print(f"[Authors] Layer 2: Trying PDF extraction as fallback")
    pdf_authors = extract_authors_with_affiliations(pdf_url, paper_title, api_key)
    
    if not pdf_authors:
        print(f"[Authors] PDF extraction failed, returning S2 data only")
        return [
            {
                'name': author.name,
                'author_id': author.authorId,
                'affiliations': author.affiliations
            }
            for author in s2_authors
        ]
    
    # Merge: Fill in missing affiliations from PDF
    print(f"[Authors] Merging S2 data with PDF affiliations")
    
    result = []
    for i, s2_author in enumerate(s2_authors):
        if s2_author.affiliations:
            # Already has affiliations from S2
            result.append({
                'name': s2_author.name,
                'author_id': s2_author.authorId,
                'affiliations': s2_author.affiliations
            })
        else:
            # Try to match with PDF author by name
            matched_pdf_author = None
            for pdf_author in pdf_authors:
                # Simple name matching (could be improved)
                if _normalize_author_name(s2_author.name) == _normalize_author_name(pdf_author['name']):
                    matched_pdf_author = pdf_author
                    break
            
            if matched_pdf_author and matched_pdf_author.get('affiliations'):
                result.append({
                    'name': s2_author.name,
                    'author_id': s2_author.authorId,
                    'affiliations': matched_pdf_author['affiliations']
                })
                print(f"[Authors] Filled affiliations for {s2_author.name} from PDF")
            else:
                # No match, keep S2 data without affiliations
                result.append({
                    'name': s2_author.name,
                    'author_id': s2_author.authorId,
                    'affiliations': None
                })
    filled_count = sum(1 for a in result if a['affiliations'])
    print(f"[Authors] Final: {filled_count}/{len(result)} authors with affiliations")
    return result

def _normalize_author_name(name: str) -> str:
    """Normalize author name for matching (lowercase, remove dots, extra spaces)."""
    if not name:
        return ""
    # Remove dots (J. Doe -> J Doe)
    name = name.replace('.', ' ')
    # Lowercase and normalize whitespace
    name = ' '.join(name.lower().split())
    return name
def _is_likely_section_title(line: str) -> bool:
    """
    Check if a line looks like a section title (not regular content).
    Section titles typically:
    - Are short (< 80 chars)
    - Don't end with period/comma (sentence endings)
    - Are mostly capitalized or title case
    - Don't contain long phrases
    """
    line = line.strip()
    if not line:
        return False
    # Too long for a title
    if len(line) > 80:
        return False
    # Ends with sentence punctuation - likely content
    if line.endswith('.') or line.endswith(',') or line.endswith(':'):
        return False
    # Contains too many words - likely content
    words = line.split()
    if len(words) > 10:
        return False
    return True
def _find_next_section(lines: List[str], start_idx: int) -> int:
    """
    Find the next section title after start_idx.
    Uses both pattern matching and heuristics.
    Returns line index or -1 if not found.
    """
    for i in range(start_idx, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        
        # First check: must look like a title (short, no ending punctuation)
        if not _is_likely_section_title(line):
            continue
        
        # Then check: matches our strict patterns
        for pattern in NEXT_SECTION_PATTERNS:
            if re.match(pattern, line, re.IGNORECASE):
                return i
    
    return -1
def extract_intro_span_rule_based(pdf_text: str) -> Optional[str]:
    """
    Extract introduction verbatim using rule-based matching.
    Strategy:
    1. Find Abstract end position
    2. Find Introduction title after Abstract
    3. Find next section title (intro end)
    4. Return verbatim text between intro start and end
    Args:
        pdf_text: Raw text from PDF pages 
    Returns:
        Introduction verbatim span or None if not found
    """
    if not pdf_text or len(pdf_text) < 100:
        return None
    lines = pdf_text.split('\n')
    # Step 1: Find Abstract position (to skip it)
    abstract_end_idx = 0
    abstract_idx = _find_line_matching(lines, ABSTRACT_PATTERNS)
    if abstract_idx >= 0:
        # Skip abstract content: find next section-like title after abstract
        # Usually intro is within 30-50 lines after abstract
        abstract_end_idx = abstract_idx + 1
    # Step 2: Find Introduction title
    intro_start_idx = _find_line_matching(lines, INTRO_PATTERNS, abstract_end_idx)
    if intro_start_idx < 0:
        print("[Rule Extract] No Introduction title found")
        return None
    print(f"[Rule Extract] Found Introduction at line {intro_start_idx}: '{lines[intro_start_idx].strip()}'")
    
    # Step 3: Find next section title (intro end) using strict matching
    intro_end_idx = _find_next_section(lines, intro_start_idx + 1)
    
    if intro_end_idx < 0:
        # No next section found, use fixed window (take ~3000 chars worth)
        # Average line is ~60 chars, so ~50 lines for 3000 chars
        intro_end_idx = min(len(lines), intro_start_idx + 60)
        print(f"[Rule Extract] No next section found, using fixed window: {intro_end_idx - intro_start_idx} lines")
    else:
        print(f"[Rule Extract] Found next section at line {intro_end_idx}: '{lines[intro_end_idx].strip()}'")    
    # Step 4: Extract verbatim text (skip the title line itself)
    intro_lines = lines[intro_start_idx + 1 : intro_end_idx]
    intro_text = '\n'.join(intro_lines).strip()
    
    # Truncate if too long
    if len(intro_text) > MAX_INTRO_CHARS:
        intro_text = intro_text[:MAX_INTRO_CHARS] + "..."
    
    # Validate minimum length
    if len(intro_text) < MIN_INTRO_CHARS:
        print(f"[Rule Extract] Intro too short: {len(intro_text)} chars")
        return None
    
    print(f"[Rule Extract] Extracted {len(intro_text)} chars verbatim intro")
    return intro_text


def _remove_abstract_and_before(pdf_text: str) -> str:
    """
    Remove Abstract section and everything before it (title, authors, affiliations).
    This ensures LLM only sees the main body content for summarization.
    Args:
        pdf_text: Raw text from PDF pages
        
    Returns:
        Text with Abstract and preceding content removed
    """
    if not pdf_text:
        return pdf_text
    
    lines = pdf_text.split('\n')
    
    # Find Abstract position
    abstract_end_idx = -1
    for i, line in enumerate(lines):
        line_stripped = line.strip().lower()
        # Find Abstract title
        if line_stripped in ['abstract', 'abstract.']:
            # Skip until we find the next section (Introduction, 1., etc.)
            for j in range(i + 1, min(i + 50, len(lines))):  # Look ahead max 50 lines
                next_line = lines[j].strip()
                if not next_line:
                    continue
                # Check if this looks like Introduction or Section 1
                if re.match(r'^\s*(1\.?\s+|I\.?\s+)?Introduction', next_line, re.IGNORECASE):
                    abstract_end_idx = j
                    break
                if re.match(r'^\s*1\.?\s+[A-Z]', next_line):
                    abstract_end_idx = j
                    break
            if abstract_end_idx == -1:
                # Didn't find intro, just skip ~30 lines after abstract
                abstract_end_idx = min(i + 30, len(lines))
            break
    
    if abstract_end_idx > 0:
        # Return content after Abstract
        remaining_text = '\n'.join(lines[abstract_end_idx:])
        print(f"[LLM Summarize] Removed {abstract_end_idx} lines (Abstract + before), {len(remaining_text)} chars remaining")
        return remaining_text
    
    # Abstract not found, try to remove obvious header content
    # Skip first 20 lines which typically contain title/authors
    skip_lines = 20
    remaining_text = '\n'.join(lines[skip_lines:])
    print(f"[LLM Summarize] No Abstract found, skipped first {skip_lines} lines")
    return remaining_text


def extract_intro_with_llm_summarize(pdf_text: str, paper_title: str, api_key: str = None) -> Optional[str]:
    """
    Use LLM to SUMMARIZE the introduction content (not verbatim copy).
    This is a fallback when rule-based extraction fails.
    Leverages LLM's summarization capability (its strength) rather than
    wasting tokens on copy-paste tasks.
    Args:
        pdf_text: Raw text from PDF (will be preprocessed to remove Abstract)
        paper_title: Paper title for context
        api_key: Optional API key
    Returns:
        Summarized introduction or None
    """
    if not pdf_text or len(pdf_text) < 100:
        return None
    
    # Step 1: Remove Abstract and content before it
    # We already have Abstract from Semantic Scholar, no need to include it
    cleaned_text = _remove_abstract_and_before(pdf_text)
    
    if not cleaned_text or len(cleaned_text) < 100:
        print(f"[LLM Summarize] Not enough content after removing Abstract")
        return None
    
    # Step 2: Truncate to fit context window
    if len(cleaned_text) > 8000:
        cleaned_text = cleaned_text[:8000] + "\n... [truncated]"
    
    # Step 3: Use LLM to summarize (not copy)
    prompt = f"""You are analyzing the INTRODUCTION section of an academic paper.
    Paper title: "{paper_title}"
    Based on the text below (which starts AFTER the Abstract), summarize the key information from the Introduction section.
    Your task: Generate a concise summary covering:
    1. **Research Problem**: What problem does this paper address? (1-2 sentences)
    2. **Motivation**: Why is this problem important? What gaps exist in current solutions? (2-3 sentences)
    3. **Key Contributions**: What are the main contributions? (3-5 bullet points)
    4. **Approach Overview**: What is the proposed method/approach? (1-2 sentences)
    IMPORTANT:
    - Focus on the Introduction section content
    - Be concise and factual
    - Do NOT include Abstract content (we already have it separately)
    - Output ~400-600 words total
    - If Introduction content is not clear, output "[NOT_FOUND]"
    --- PAPER CONTENT (after Abstract) ---
    {cleaned_text}
    --- END ---
    Introduction Summary:"""

    try:
        llm_instance = llm.get_llm("extract", temperature=0.1, api_key=api_key)
        response = llm_instance.invoke(prompt)
        
        if hasattr(response, 'content'):
            result = response.content.strip()
        else:
            result = str(response).strip()
        
        # Check for failure marker
        if "[NOT_FOUND]" in result or len(result) < 100:
            print(f"[LLM Summarize] Could not summarize introduction")
            return None
        
        # Reasonable length check (summary should be 400-1500 chars)
        if len(result) > 2000:
            result = result[:2000] + "..."
        
        print(f"[LLM Summarize] Generated {len(result)} chars introduction summary")
        return result
        
    except Exception as e:
        print(f"[LLM Summarize] Error: {e}")
        return None


def extract_intro_with_llm_locate(pdf_text: str, paper_title: str, api_key: str = None) -> Optional[str]:
    """
    Use LLM to LOCATE introduction boundaries, then extract with rules.
    This is the most token-efficient approach:
    - LLM does: semantic understanding to find start/end lines (~10 output tokens)
    - Rules do: precise text extraction (slicing)
    Fully follows "规则与LLM职责分离" principle.
    Args:
        pdf_text: Raw text from PDF
        paper_title: Paper title for context
        api_key: Optional API key
        
    Returns:
        Extracted introduction text or None
    """
    if not pdf_text or len(pdf_text) < 100:
        return None
    
    lines = pdf_text.split('\n')
    total_lines = len(lines)
    # Create a numbered version for LLM (helps with line identification)
    # Only include non-empty lines with their original indices
    numbered_preview = []
    for i, line in enumerate(lines):
        if line.strip():  # Skip empty lines in preview
            numbered_preview.append(f"L{i}: {line.strip()[:100]}")
    # Truncate if too long
    preview_text = '\n'.join(numbered_preview[:150])  # First 150 non-empty lines
    prompt = f"""Analyze this academic paper text and locate the INTRODUCTION section.
    Paper: "{paper_title}"
    Task: Find the exact line numbers where Introduction STARTS and ENDS.
    Rules:
    1. Introduction typically starts after Abstract, with title like "1. Introduction" or "Introduction"
    2. Introduction ends when next section begins (e.g., "2. Related Work", "Background", "Method")
    3. Output ONLY the line numbers in this exact format: START:XX END:YY
    4. If you cannot find clear boundaries, output: NOT_FOUND
    --- TEXT (L=line number) ---
    {preview_text}
    --- END ---
    Line numbers (format: START:XX END:YY):"""

    try:
        llm_instance = llm.get_llm("extract", temperature=0.0, api_key=api_key)
        response = llm_instance.invoke(prompt)
        
        if hasattr(response, 'content'):
            result = response.content.strip()
        else:
            result = str(response).strip()
        
        # Parse result
        if "NOT_FOUND" in result:
            print(f"[LLM Locate] Could not locate introduction boundaries")
            return None
        
        # Extract line numbers using regex
        import re
        match = re.search(r'START:?(\d+).*?END:?(\d+)', result, re.IGNORECASE)
        if not match:
            print(f"[LLM Locate] Could not parse line numbers from: {result}")
            return None
        
        start_line = int(match.group(1))
        end_line = int(match.group(2))
        
        # Validate boundaries
        if start_line < 0 or end_line > total_lines or start_line >= end_line:
            print(f"[LLM Locate] Invalid boundaries: {start_line}-{end_line} (total: {total_lines})")
            return None
        
        # Extract text using rules (the actual extraction)
        intro_lines = lines[start_line:end_line]
        intro_text = '\n'.join(intro_lines).strip()
        
        # Skip the title line if it matches Introduction pattern
        first_line = intro_lines[0].strip() if intro_lines else ""
        if re.match(r'^\s*(1\.?\s+)?Introduction', first_line, re.IGNORECASE):
            intro_text = '\n'.join(intro_lines[1:]).strip()
        
        # Validate length
        if len(intro_text) < MIN_INTRO_CHARS:
            print(f"[LLM Locate] Extracted intro too short: {len(intro_text)} chars")
            return None
        
        if len(intro_text) > MAX_INTRO_CHARS:
            intro_text = intro_text[:MAX_INTRO_CHARS] + "..."
        
        print(f"[LLM Locate] Located intro at lines {start_line}-{end_line}, extracted {len(intro_text)} chars")
        return intro_text
        
    except Exception as e:
        print(f"[LLM Locate] Error: {e}")
        return None


def get_paper_introduction(
    pdf_url: str,
    paper_title: str,
    api_key: str = None,
    fallback_tldr: str = ""
) -> Tuple[str, str]:
    """
    Main function: Get paper introduction from PDF using 3-layer fallback.
    
    Pipeline:
    - Layer 1: Rule-based extraction on first 2 pages (fast, cheap)
    - Layer 2: Expand to 4/6 pages if layer 1 fails
    - Layer 3: LLM summarize introduction (leverages LLM's strength)
    - Fallback: Use TLDR/abstract
    
    Design principle: Rules handle precise extraction, LLM handles
    semantic understanding and summarization (not copy-paste).
    
    Args:
        pdf_url: URL to PDF file
        paper_title: Paper title
        api_key: Optional API key for LLM
        fallback_tldr: Fallback text if PDF extraction fails
        
    Returns:
        (introduction_text, source) where source is:
        - 'pdf_rule': Rule-based extraction (fastest, free)
        - 'pdf_llm': LLM located boundaries + rule extraction (efficient)
        - 'pdf_llm_summary': LLM summarized content (fallback)
        - 'tldr': Semantic Scholar TLDR
        - 'none': No content available
    """
    start_time = time.time()
    
    if not pdf_url:
        print(f"[Introduction] No PDF URL for: {paper_title[:50]}...")
        return fallback_tldr, 'tldr' if fallback_tldr else 'none'
    
    # Step 1: Download PDF
    pdf_bytes = download_pdf_partial(pdf_url)
    if not pdf_bytes:
        print(f"[Introduction] PDF download failed, using fallback")
        return fallback_tldr, 'tldr' if fallback_tldr else 'none'
    
    # Step 2: Progressive page expansion with rule-based extraction
    pdf_text_cache = {}  # Cache extracted text by page count
    
    for max_pages in PAGE_EXPANSION:
        # Extract text for this page count
        if max_pages not in pdf_text_cache:
            pdf_text = extract_first_pages_text(pdf_bytes, max_pages=max_pages)
            pdf_text_cache[max_pages] = pdf_text
        else:
            pdf_text = pdf_text_cache[max_pages]
        
        if not pdf_text:
            print(f"[Introduction] PDF text extraction failed for {max_pages} pages")
            continue
        
        # Try rule-based extraction
        print(f"[Introduction] Trying rule-based extraction on {max_pages} pages...")
        intro = extract_intro_span_rule_based(pdf_text)
        
        if intro:
            elapsed = time.time() - start_time
            print(f"[Introduction] Rule-based extraction success in {elapsed:.1f}s: {len(intro)} chars")
            return intro, 'pdf_rule'
    
    # Step 3: LLM locate fallback (most token-efficient)
    # LLM finds line numbers, rules do extraction
    # Output: ~10 tokens vs 3000 tokens for verbatim copy
    max_text = pdf_text_cache.get(PAGE_EXPANSION[-1]) or pdf_text_cache.get(PAGE_EXPANSION[0])
    if max_text:
        print(f"[Introduction] Rule-based failed, trying LLM locate...")
        intro = extract_intro_with_llm_locate(max_text, paper_title, api_key)
        
        if intro:
            elapsed = time.time() - start_time
            print(f"[Introduction] LLM locate success in {elapsed:.1f}s: {len(intro)} chars")
            return intro, 'pdf_llm'
        
        # Step 4: LLM summarize as last resort
        # Only if locate fails (can't find boundaries)
        print(f"[Introduction] LLM locate failed, trying LLM summarize...")
        intro = extract_intro_with_llm_summarize(max_text, paper_title, api_key)
        
        if intro:
            elapsed = time.time() - start_time
            print(f"[Introduction] LLM summarize success in {elapsed:.1f}s: {len(intro)} chars")
            return intro, 'pdf_llm_summary'
    
    # Step 4: All extraction failed, use fallback
    elapsed = time.time() - start_time
    print(f"[Introduction] All extraction methods failed in {elapsed:.1f}s, using TLDR fallback")
    return fallback_tldr, 'tldr' if fallback_tldr else 'none'


def get_papers_introductions_batch(
    papers: list,
    api_key: str = None,
    max_concurrent: int = 3
) -> list:
    """
    Batch process multiple papers to get introductions.
    
    Args:
        papers: List of dicts with 'pdf_url', 'title', 'tldr' keys
        api_key: Optional API key
        max_concurrent: Max concurrent downloads (keep low to avoid rate limits)
        
    Returns:
        List of (introduction, source) tuples
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    
    results = [None] * len(papers)
    
    def process_one(idx: int, paper: dict):
        pdf_url = paper.get('pdf_url', '')
        title = paper.get('title', '')
        tldr = paper.get('tldr', '') or paper.get('introduction', '')
        
        intro, source = get_paper_introduction(pdf_url, title, api_key, tldr)
        return idx, intro, source
    
    print(f"[Introduction Batch] Processing {len(papers)} papers...")
    
    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for idx, paper in enumerate(papers):
            future = executor.submit(process_one, idx, paper)
            futures[future] = idx
        
        for future in as_completed(futures):
            try:
                idx, intro, source = future.result(timeout=30)
                results[idx] = (intro, source)
                print(f"[Introduction Batch] [{idx+1}/{len(papers)}] {source}: {len(intro) if intro else 0} chars")
            except Exception as e:
                idx = futures[future]
                results[idx] = (papers[idx].get('tldr', ''), 'error')
                print(f"[Introduction Batch] [{idx+1}/{len(papers)}] Error: {e}")
    
    # Fill any None results with fallback
    for idx, result in enumerate(results):
        if result is None:
            results[idx] = (papers[idx].get('tldr', ''), 'none')
    # Stats
    sources = [r[1] for r in results]
    pdf_rule = sources.count('pdf_rule')
    pdf_llm = sources.count('pdf_llm')
    pdf_llm_summary = sources.count('pdf_llm_summary')
    tldr = sources.count('tldr')
    error = sources.count('error')
    print(f"[Introduction Batch] Complete: pdf_rule={pdf_rule}, pdf_llm_locate={pdf_llm}, pdf_llm_summary={pdf_llm_summary}, tldr={tldr}, error={error}")
    return results
# ==================== Corresponding Author Detection ====================
def detect_corresponding_authors_for_paper(
    pdf_url: str,
    authors: list,
    api_key: str = None,
    use_llm_fallback: bool = True
) -> Tuple[list, list, list]:
    """
    Detect corresponding authors for a paper and update author list.
    
    This is a convenience function that:
    1. Downloads PDF
    2. Runs 3-stage corresponding author detection
    3. Returns updated data structures
    
    Args:
        pdf_url: URL to PDF file
        authors: List of author dicts or AuthorWithId objects
        api_key: Optional API key for LLM fallback
        use_llm_fallback: Whether to use LLM fallback (default True)
        
    Returns:
        (corresponding_indices, evidences, author_names)
        - corresponding_indices: List of 1-based author indices that are corresponding authors
        - evidences: List of evidence dicts for audit/storage
        - author_names: List of author name strings (for detection input)
    """
    if not pdf_url or not authors:
        return [], [], []
    
    try:
        from .corresponding_author_detector import (
            detect_corresponding_authors_from_url
        )
    except ImportError:
        try:
            from corresponding_author_detector import detect_corresponding_authors_from_url
        except ImportError:
            print("[PDF Correspond] Cannot import corresponding_author_detector")
            return [], [], []
    
    # Extract author names for detection
    author_names = []
    for author in authors:
        if hasattr(author, 'name'):
            author_names.append(author.name)
        elif isinstance(author, dict) and 'name' in author:
            author_names.append(author['name'])
        elif isinstance(author, str):
            author_names.append(author)
    
    if not author_names:
        return [], [], []
    
    # Run detection
    indices, evidences = detect_corresponding_authors_from_url(
        pdf_url=pdf_url,
        authors=author_names,
        api_key=api_key,
        use_llm_fallback=use_llm_fallback
    )
    
    return indices, evidences, author_names


def update_authors_with_corresponding_info(
    authors: list,
    corresponding_indices: list
) -> list:
    """
    Update author objects/dicts with is_corresponding flag.
    
    Args:
        authors: List of author dicts or AuthorWithId objects
        corresponding_indices: List of 1-based indices that are corresponding authors
        
    Returns:
        Updated authors list (same objects, modified in place)
    """
    if not authors or not corresponding_indices:
        return authors
    
    corresponding_set = set(corresponding_indices)
    
    for idx, author in enumerate(authors):
        author_idx = idx + 1  # Convert to 1-based
        is_corresponding = author_idx in corresponding_set
        
        if hasattr(author, 'is_corresponding'):
            author.is_corresponding = is_corresponding if is_corresponding else None
        elif isinstance(author, dict):
            author['is_corresponding'] = is_corresponding if is_corresponding else None
    
    return authors