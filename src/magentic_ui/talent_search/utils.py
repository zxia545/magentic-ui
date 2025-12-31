"""
Utility functions for talent search module
"""
import re
import time
from typing import List, Tuple, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode
# ==================== Text Processing ====================

def strip_thinking(text: str) -> str:
    """
    Strip thinking tags and their content from LLM responses.
    Some LLMs wrap their reasoning process in <thinking>...</thinking> tags.
    This function removes these tags and their content to extract clean output.
    Args:
        text: Input text that may contain thinking tags  
    Returns:
        Text with thinking tags removed
    """
    if not text:
        return ""
    # Remove <thinking>...</thinking> blocks (case-insensitive, multiline)
    import re
    cleaned = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.IGNORECASE | re.DOTALL)
    # Also remove self-closing thinking tags
    cleaned = re.sub(r'<thinking\s*/>', '', cleaned, flags=re.IGNORECASE)
    # Strip extra whitespace that may be left
    cleaned = cleaned.strip()
    return cleaned
def normalize_whitespace(s: str) -> str:
    """
    Normalize whitespace in a string:
    - Replace multiple spaces/tabs/newlines with single space
    - Strip leading/trailing whitespace
    """
    if not s:
        return ""
    return re.sub(r'\s+', ' ', s).strip()
def clean_text(text: str, max_chars: int = 0) -> str:
    """
    Clean text by removing extra whitespace and normalizing.
    Args:
        text: Input text
        max_chars: Maximum characters to return (0 = no limit)
    """
    if not text:
        return ""
    # Remove control characters
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    # Normalize whitespace
    result = normalize_whitespace(text)
    # Truncate if needed
    if max_chars > 0 and len(result) > max_chars:
        result = result[:max_chars]
    return result
# ==================== URL Processing ====================
_TRACKING_KEYS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "fbclid", "mc_cid", "mc_eid", "oly_anon_id", "oly_enc_id",
    "ref", "source", "ref_src", "ref_url"
}
def normalize_url(url: str) -> str:
    """
    Normalize a URL by:
    - Removing tracking parameters
    - Removing fragments
    - Lowercasing the domain
    - Ensuring consistent formatting
    """
    if not url:
        return ""
    try:
        p = urlparse(url)
        # Remove tracking query params
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) 
             if k.lower() not in _TRACKING_KEYS]
        # Rebuild URL without fragment
        p2 = p._replace(
            netloc=p.netloc.lower(),
            query=urlencode(q, doseq=True),
            fragment=""
        )
        return urlunparse(p2)
    except Exception:
        return url
def domain_of(url: str) -> str:
    """
    Extract the domain (netloc) from a URL.
    """
    if not url:
        return ""
    try:
        p = urlparse(url)
        return p.netloc.lower()
    except Exception:
        return ""
def looks_like_profile_url(url: str) -> bool:
    """
    Check if a URL looks like a personal/academic profile page.
    Matches patterns like:
    - /~username
    - /people/username
    - /staff/username
    - /faculty/username
    - /author/username
    - /profile/username
    """
    if not url:
        return False 
    profile_patterns = [
        r'/~[\w-]+',           # Unix-style home pages
        r'/people/[\w-]+',     # People directories
        r'/staff/[\w-]+',      # Staff pages
        r'/faculty/[\w-]+',    # Faculty pages
        r'/author/[\w-]+',     # Author pages
        r'/profile/[\w-]+',    # Profile pages
        r'/user/[\w-]+',       # User pages
        r'/u/[\w-]+',          # Short user paths
        r'/members/[\w-]+',    # Members
    ]
    url_lower = url.lower()
    for pattern in profile_patterns:
        if re.search(pattern, url_lower):
            return True
    
    # Check for academic domains with personal pages
    academic_domains = ['.edu', '.ac.uk', '.edu.cn', '.edu.au']
    if any(d in url_lower for d in academic_domains):
        # If it's an academic URL with a name-like path
        if re.search(r'/[a-z]{2,}[a-z0-9_-]*/?$', url_lower):
            return True
    
    return False
# ==================== Timing ====================
def safe_sleep(seconds: float):
    """
    Sleep for specified seconds, handling interrupts gracefully.
    """
    if seconds <= 0:
        return
    try:
        time.sleep(seconds)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        pass
# ==================== Author Position ====================

def compute_author_position(candidate_name: str, authors: List[str]) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """
    计算候选人在论文作者列表中的位置
    
    Args:
        candidate_name: 候选人名字
        authors: 作者列表
        
    Returns:
        (position_index, position_label, total_authors)
        - position_index: 1-based 位置 (1=一作)，找不到则为 None
        - position_label: 人类可读的位置标签 (如 "1st author", "last author")
        - total_authors: 作者总数
    """
    if not authors or not candidate_name:
        return None, None, None
    
    total_authors = len(authors)
    candidate_lower = candidate_name.lower().strip()
    authors_lower = [author.lower().strip() for author in authors]
    
    # 策略：优先完全匹配，其次最长子串匹配
    position_index = None
    best_match_idx = -1
    best_match_score = 0
    
    for idx, author in enumerate(authors_lower):
        # 策略1: 完全匹配（最高优先级）
        if candidate_lower == author:
            position_index = idx + 1
            break
        
        # 策略2: 候选人名字是作者全名的子集
        if candidate_lower in author:
            match_score = len(candidate_lower) / len(author)
            if match_score > 0.7 and match_score > best_match_score:
                best_match_idx = idx
                best_match_score = match_score
        
        # 策略3: 作者全名是候选人名字的子集
        elif author in candidate_lower:
            match_score = len(author) / len(candidate_lower)
            if match_score > 0.7 and match_score > best_match_score:
                best_match_idx = idx
                best_match_score = match_score
    
    # 如果没有完全匹配，使用最佳子串匹配
    if position_index is None and best_match_idx != -1:
        position_index = best_match_idx + 1
    
    if position_index is None:
        return None, None, total_authors
    
    # 生成人类可读的位置标签
    if position_index == 1:
        position_label = "1st author"
    elif position_index == total_authors and total_authors > 1:
        position_label = f"last author ({position_index}/{total_authors})"
    elif position_index == 2:
        position_label = "2nd author"
    elif position_index == 3:
        position_label = "3rd author"
    else:
        if total_authors:
            position_label = f"author #{position_index}/{total_authors}"
        else:
            position_label = f"author #{position_index}"
    
    return position_index, position_label, total_authors


def get_author_position_coef(
    position_index: Optional[int], 
    total_authors: Optional[int],
    is_corresponding: Optional[bool] = None,
    is_cofirst: Optional[bool] = None
) -> float:
    """
    根据作者位置计算系数
    
    Args:
        position_index: 1-based 作者位置
        total_authors: 作者总数
        is_corresponding: 是否为通讯作者（从 PDF 检测得到）
        is_cofirst: 是否为共同一作（从 PDF 检测得到）
        
    Returns:
        位置系数 (0.0 - 1.0)
        
    权重规则:
        - 一作：1.0
        - 共同一作：1.0（与一作相同权重）
        - 通讯作者：0.9（如果同时是一作/共一则仍为 1.0）
        - 尾作：0.8
        - 二作：0.4
        - 三作及之后：0.4/n
        
    注意：如果无法确定通讯作者/共一（is_corresponding=None/is_cofirst=None），则不考虑这些权重
    """
    if position_index is None or position_index <= 0:
        return 0.0
    
    if position_index == 1:
        # 一作：系数 1.0（即使同时是通讯作者/共一也是 1.0）
        return 1.0
    # 共同一作（非一作但标记为共一）：系数 1.0（与一作相同权重）
    if is_cofirst is True:
        return 1.0
    
    # 通讯作者（非一作/非共一）：系数 0.9
    if is_corresponding is True:
        return 0.9
    if total_authors and position_index == total_authors and total_authors > 1:
        # 尾作：系数 0.8
        return 0.8
    elif position_index == 2:
        # 二作：系数 0.4
        return 0.4
    else:
        # 三作及之后：系数 0.4/n
        return 0.4 / position_index
# ==================== Author ID Validation ====================
def is_valid_openreview_id(author_id: str) -> bool:
    """
    检查是否是有效的 OpenReview profile ID（以 ~ 开头）
    Args:
        author_id: 待验证的 ID
    Returns:
        True 如果是有效的 OpenReview ID
    """
    if not author_id or not isinstance(author_id, str):
        return False
    return author_id.strip().startswith('~')
def is_valid_semantic_scholar_id(author_id: str) -> bool:
    """
    检查是否是有效的 Semantic Scholar author ID（纯数字）
    Args:
        author_id: 待验证的 ID
    Returns:
        True 如果是有效的 Semantic Scholar ID
    """
    if not author_id or not isinstance(author_id, str):
        return False
    # Semantic Scholar ID 是纯数字
    return author_id.strip().isdigit()
def validate_author_id(author_id) -> Tuple[Optional[str], str]:
    """
    验证 author_id 并返回清理后的值和类型
    过滤掉无效的 ID 格式：
    - DBLP URLs (https://dblp.org/...)
    - 其他非法格式
    Args:
        author_id: 任意格式的 author ID
    Returns:
        (cleaned_id, id_type)
        - cleaned_id: 清理后的 ID，如果无效则返回 None
        - id_type: 'openreview' / 'semantic_scholar' / 'invalid'
    """
    if author_id is None:
        return None, 'invalid'
    # 处理非字符串类型
    if isinstance(author_id, (int, float)):
        author_id = str(int(author_id))
    elif not isinstance(author_id, str):
        return None, 'invalid'
    author_id = author_id.strip()
    if not author_id:
        return None, 'invalid'
    # 检查是否是 URL（无效）
    if author_id.startswith('http://') or author_id.startswith('https://'):
        # 特别检查 DBLP URLs
        if 'dblp.org' in author_id:
            return None, 'invalid'  # DBLP URL 不是有效的 author_id
        # 其他 URL 也不是有效的 author_id
        return None, 'invalid'
    # 检查 OpenReview ID
    if author_id.startswith('~'):
        return author_id, 'openreview'
    # 检查 Semantic Scholar ID（纯数字）
    if author_id.isdigit():
        return author_id, 'semantic_scholar'
    # 其他格式（可能是无效的）
    return None, 'invalid'
def sanitize_author_ids(author_ids: List) -> List[Optional[str]]:
    """
    清理 author_ids 列表，过滤掉无效的 ID
    Args:
        author_ids: 原始 author_ids 列表
        
    Returns:
        清理后的列表，无效 ID 被替换为 None
    """
    if not author_ids:
        return []
    result = []
    for aid in author_ids:
        cleaned_id, id_type = validate_author_id(aid)
        if id_type == 'invalid':
            result.append(None)
        else:
            result.append(cleaned_id)
    return result
