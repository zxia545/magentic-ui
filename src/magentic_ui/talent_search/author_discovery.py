"""
Author Discovery Module for Talent Search System
Implements comprehensive author profile discovery and integration
"""
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Any, Tuple, Optional
import difflib
import re
import time
import traceback
import threading
from dataclasses import dataclass, field
from urllib.parse import urlparse, urljoin
import json
import requests
from bs4 import BeautifulSoup
import warnings

# Suppress Streamlit ScriptRunContext warnings in ThreadPoolExecutor
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing script run context.*')

# ============================ GLOBAL SAFE PRINT UTILITY ============================

def safe_print(msg):
    """
    Safe print that won't crash on I/O errors.
    Critical for long-running tasks where log files might disconnect.
    Prevents cascading failures when stdout/log_file becomes unavailable.
    """
    try:
        print(msg)
    except (OSError, IOError, BrokenPipeError):
        pass  # Silently ignore I/O errors

from . import config
from . import search as search
from . import llm
from . import schemas
from .utils import normalize_url, domain_of
from .search import searxng_search, fetch_text, extract_title_unified, extract_main_text
from .dynamic_concurrency import get_optimal_workers, get_llm_workers, get_extraction_workers
from .nine_dimension_extractor import NineDimensionExtractor
from .semanticscholar_client import (
    SemanticScholarSearchClient,
    SemanticScholarPaper,
)
# Import author position utility from shared utils
from .utils import compute_author_position
# Import new 4-dimension evaluation prompts
from .evaluation_prompts import (
    PROMPT_TOPIC_MATCH,
    PROMPT_VENUE_FIT,
    PROMPT_RECENCY_MOMENTUM,
    PROMPT_ROLE_LEADERSHIP,
    format_candidate_info_for_eval
)
# ============================ DATA CLASSES ============================
@dataclass
class AuthorProfile:
    """Complete author profile with all discovered information"""
    name: str
    aliases: List[str]
    platforms: Dict[str, str]
    ids: Dict[str, str]
    homepage_url: Optional[str]
    affiliation_current: Optional[str]
    emails: List[str]
    interests: List[str]
    selected_publications: List[Dict[str, Any]]
    confidence: float
    
    # 新增字段
    notable_achievements: List[str] = field(default_factory=list)  # Awards, honors, recognitions
    social_impact: Optional[str] = None      # H-index, citations, influence metrics
    career_stage: Optional[str] = None       # student/postdoc/assistant_prof/etc
    overall_score: float = 0.0               # 综合评分 0~100

@dataclass
class ProfileCandidate:
    """Candidate profile URL with scoring and LLM decision"""
    url: str
    title: str
    snippet: str
    score: float
    should_fetch: Optional[bool] = None
    reason: Optional[str] = None
    trusted_source: bool = False  # True if from OpenReview profile (skip validation)
# ============================ GLOBAL DEDUPLICATION SET ============================
# 全局集合：维护已处理的 OpenReview ID，用于去重
_processed_openreview_ids = set()

def reset_processed_openreview_ids():
    """重置已处理的 OpenReview ID 集合（用于测试或新会话）"""
    global _processed_openreview_ids
    _processed_openreview_ids.clear()
    print(f"[Author Discovery] 🔄 Reset processed OpenReview IDs set")
# ============================ REGEX PATTERNS FOR ID EXTRACTION ============================

ID_PATTERNS = {
    'orcid': re.compile(r'orcid\.org/(\d{4}-\d{4}-\d{4}-\d{4})'),
    'openreview': re.compile(r'openreview\.net/profile\?id=([A-Za-z0-9_\-\.%]+)'),
    'scholar': re.compile(r'scholar\.google\.com/citations\?user=([A-Za-z0-9_\-]+)'),
    'semanticscholar': re.compile(r'semanticscholar\.org/author/([^/\s]+)'),
    'dblp1': re.compile(r'dblp\.org/pid/([0-9a-z/]+)'),
    'dblp2': re.compile(r'dblp\.org/pers/([0-9a-z/]+)'),
    'twitter': re.compile(r'(?:x\.com|twitter\.com)/([A-Za-z0-9_]{1,15})(?:/|$)'),
    'github': re.compile(r'github\.com/([A-Za-z0-9\-]+)(?:/|$)'),
}

_INTERACTIVE_HTML_PATTERNS = [
    r'role=["\']tab["\']',
    r'data-bs-toggle=["\']tab["\']',
    r'data-bs-toggle=["\']collapse["\']',
    r'class=["\'][^"\']*(nav-tabs|tab-pane|accordion|collapse|tabs-container|MuiTabs-root)[^"\']*["\']',
    r'aria-expanded=["\']false["\']',
    r'onclick\s*='
]
# ============================ SORTING UTILITIES ============================

def _parse_year_from_duration(duration: str) -> Tuple[Optional[int], Optional[int]]:
    """
    从 duration 字符串中解析开始年份和结束年份
    
    Returns:
        (start_year, end_year) 元组，如果无法解析则返回 (None, None)
    """
    if not duration:
        return (None, None)
    
    # 标准化：统一各种分隔符和格式
    duration = duration.strip()
    duration_lower = duration.lower()
    
    # 处理 Present/Current/Now
    if 'present' in duration_lower or 'current' in duration_lower or 'now' in duration_lower:
        # 提取开始年份
        match = re.search(r'(\d{4})', duration)
        if match:
            year = int(match.group(1))
            # 验证年份合理性（1900-2100）
            if 1900 <= year <= 2100:
                return (year, 9999)  # 9999 表示当前
    
    # 处理各种分隔符：–, -, to, ~, / 等
    # 先尝试匹配 "YYYY–YYYY" 或 "YYYY-YYYY" 或 "YYYY to YYYY" 等格式
    patterns = [
        r'(\d{4})\s*[–\-~]\s*(\d{4})',  # 2020-2024, 2020–2024, 2020~2024
        r'(\d{4})\s+to\s+(\d{4})',      # 2020 to 2024
        r'(\d{4})\s*/\s*(\d{4})',       # 2020/2024
    ]
    
    for pattern in patterns:
        match = re.search(pattern, duration)
        if match:
            start = int(match.group(1))
            end = int(match.group(2))
            # 验证年份合理性（1900-2100）
            if 1900 <= start <= 2100 and 1900 <= end <= 2100:
                return (start, end)
    
    # 提取所有4位数年份
    years = re.findall(r'\d{4}', duration)
    if len(years) >= 2:
        start = int(years[0])
        end = int(years[1])
        # 验证年份合理性
        if 1900 <= start <= 2100 and 1900 <= end <= 2100:
            return (start, end)
    elif len(years) == 1:
        year = int(years[0])
        # 验证年份合理性
        if 1900 <= year <= 2100:
            return (year, year)
    
    return (None, None)
def _sort_career_education_items(items: List[schemas.CareerEducationInfo]) -> List[schemas.CareerEducationInfo]:
    """
    按时间从新到旧排序 Career & Education History
    排序规则：
    1. Present/Current 的项目排在最前面（按开始年份从新到旧）
    2. 已结束的项目按结束年份从新到旧排序
    3. 无法解析时间的项目排在最后
    """
    def sort_key(item: schemas.CareerEducationInfo) -> Tuple[int, int, int]:
        start_year, end_year = _parse_year_from_duration(item.duration or "")
        if start_year is None and end_year is None:
            # 无法解析时间，排在最后（优先级最低=1）
            return (1, 0, 0)
        if end_year == 9999:
            # Present/Current，按开始年份从新到旧（优先级最高=3）
            # 使用负数让年份大的排前面
            return (3, -start_year, 0)
        # 已结束的项目，按结束年份从新到旧（优先级=2）
        return (2, -end_year, -start_year)
    return sorted(items, key=sort_key, reverse=True)

def _sort_experience_items(items: List[schemas.ExperienceInfo]) -> List[schemas.ExperienceInfo]:
    """
    按时间从新到旧排序 Industrial Experience
    排序规则与 Career & Education History 相同
    """
    def sort_key(item: schemas.ExperienceInfo) -> Tuple[int, int, int]:
        start_year, end_year = _parse_year_from_duration(item.duration or "")
        if start_year is None and end_year is None:
            # 无法解析时间，排在最后（优先级最低=1）
            return (1, 0, 0)
        
        if end_year == 9999:
            # Present/Current，按开始年份从新到旧（优先级最高=3）
            return (3, -start_year, 0)
        
        # 已结束的项目，按结束年份从新到旧（优先级=2）
        return (2, -end_year, -start_year)
    
    # 不需要 reverse=True，因为我们已经用优先级数字(3>2>1)和负数年份实现了正确排序
    return sorted(items, key=sort_key, reverse=True)

def _normalize_text_for_key(text: Optional[str]) -> str:
    """标准化字符串用于去重 key：小写 + 去掉首尾空格。"""
    return (text or "").lower().strip()

def _normalize_title_for_key(title: Optional[str]) -> str:
    """标准化职位标题：小写并压缩多余空格。"""
    t = _normalize_text_for_key(title)
    return re.sub(r"\s+", " ", t)

def _normalize_duration_for_key(duration: Optional[str]) -> str:
    """将 duration 归一化为 "start-end" 形式，仅用于去重 key。"""
    start_year, end_year = _parse_year_from_duration(duration or "")
    if start_year is None and end_year is None:
        return ""
    return f"{start_year}-{end_year}"

def _normalize_duration_format(duration: Optional[str]) -> str:
    """统一时间段格式为 "YYYY–YYYY" 或 "YYYY–Present" 形式。
    Args:
        duration: 原始时间段字符串
    Returns:
        统一格式的时间段字符串，如果无法解析则返回原始字符串
    """
    if not duration:
        return ""
    start_year, end_year = _parse_year_from_duration(duration)
    if start_year is None:
        return duration  # 无法解析，返回原始字符串
    
    if end_year == 9999:
        return f"{start_year}–Present"
    elif end_year == start_year:
        return str(start_year)
    else:
        return f"{start_year}–{end_year}"

_PLACEHOLDER_VALUES = {
    "unknown",
    "n/a",
    "na",
    "not specified",
    "none",
    "tbd",
    "to be determined",
    "待定",
    "未指定",
}

def _is_placeholder_text(value: Optional[str]) -> bool:
    """
    Treat empty strings or common placeholder tokens as missing content.

    This helps us avoid tagging candidates as PhD/Master/etc. when their
    displayed position & affiliation are still "Unknown".
    """
    if value is None:
        return True
    normalized = value.strip().lower()
    if not normalized:
        return True
    return normalized in _PLACEHOLDER_VALUES

def _validate_career_education_item(item_dict: Dict[str, Any]) -> bool:
    """验证 Career & Education 记录是否合理。
    
    Args:
        item_dict: 包含 degree_or_position, institution, duration 等字段的字典
    
    Returns:
        True 如果记录合理，False 否则
    """
    degree_or_pos = (item_dict.get('degree_or_position') or item_dict.get('degree') or item_dict.get('position') or '').strip()
    institution = (item_dict.get('institution', '') or '').strip()
    duration = item_dict.get('duration', '')
    
    # 至少需要有学位/职位或机构名称
    if not degree_or_pos and not institution:
        return False
    # 拒绝 institution 为占位符值的情况（如 "Not specified", "N/A", "Unknown" 等）
    # 这些值表示数据缺失，不应该作为有效的职业经历记录
    invalid_institution_values = [
        "not specified", "n/a", "na", "unknown", 
        "none", "tbd", "to be determined", "待定", "未指定"
    ]
    if institution and institution.lower() in invalid_institution_values:
        return False
    
    # 验证时间段格式（如果提供）
    if duration:
        start_year, end_year = _parse_year_from_duration(duration)
        # 如果时间段无法解析，但提供了时间段字符串，可能是格式问题，仍然接受
        # 但如果解析出的年份不合理，则拒绝
        if start_year is not None and not (1900 <= start_year <= 2100):
            return False
        if end_year is not None and end_year != 9999 and not (1900 <= end_year <= 2100):
            return False
    
    return True

def _durations_overlap(dur1: str, dur2: str, max_year_diff: int = 2) -> bool:
    """检查两个时间段是否重叠或接近（允许最多 max_year_diff 年的差异）。
    
    处理 Present/Current 的情况（end_year = 9999）。
    
    Args:
        dur1: 第一个时间段字符串
        dur2: 第二个时间段字符串
        max_year_diff: 允许的最大年份差异（默认2年）
    
    Returns:
        True 如果时间段重叠或接近，False 否则
    """
    start1, end1 = _parse_year_from_duration(dur1)
    start2, end2 = _parse_year_from_duration(dur2)
    
    if start1 is None or start2 is None:
        return False
    
    # 处理 Present/Current（end_year = 9999 表示当前）
    is_present1 = (end1 == 9999)
    is_present2 = (end2 == 9999)
    
    # 如果两个时间段都有明确的结束年份（都不是 Present）
    if not is_present1 and not is_present2:
        # 检查是否有重叠
        if not (end1 < start2 or end2 < start1):
            return True
        # 检查是否接近（允许 max_year_diff 年的差异）
        if abs(start1 - start2) <= max_year_diff:
            return True
        if abs((end1 or start1) - (end2 or start2)) <= max_year_diff:
            return True
    # 如果一个是 Present，另一个不是
    elif is_present1 and not is_present2:
        # Present 的开始年份应该接近另一个时间段的开始或结束年份
        if abs(start1 - start2) <= max_year_diff or abs(start1 - (end2 or start2)) <= max_year_diff:
            return True
    elif is_present2 and not is_present1:
        if abs(start2 - start1) <= max_year_diff or abs(start2 - (end1 or start1)) <= max_year_diff:
            return True
    # 如果两个都是 Present
    else:
        # 开始年份接近即可
        if abs(start1 - start2) <= max_year_diff:
            return True
    
    return False


def _is_research_job_title(title: str) -> bool:
    """判断职位标题是否为 research 类岗位（统一归入 Industrial Experience）。"""
    if not title:
        return False
    t = title.lower()
    keywords = [
        # 明确的研究岗
        "research scientist", "staff scientist", "senior scientist", "principal scientist",
        "researcher", "research engineer", "research assistant", "research associate",
        "research fellow", "research staff", "research intern",
        "early career fellow", "postdoctoral fellow", "postdoc fellow",
        "research fellowship", "fellowship", "research scholar",
        # 相关技术岗
        "data scientist", "machine learning engineer", "ml engineer",
    ]
    if any(kw in t for kw in keywords):
        return True
    # 兜底：只要包含 research 或 scientist 关键字，也视为 research 岗
    return ("research" in t) or ("scientist" in t)


def _is_academic_or_training_title(title: str) -> bool:
    """判断职位标题是否为明显的学术 / 教学 / 学位阶段岗位。"""
    if not title:
        return False
    t = title.lower()
    research_fellowship_keywords = [
        "early career fellow", "postdoctoral fellow", "postdoc fellow",
        "research fellow", "research fellowship", "fellowship",
    ]
    if any(kw in t for kw in research_fellowship_keywords):
        return False
    
    academic_keywords = [
        # 教授 / 教学岗位
        "professor", "lecturer", "instructor", "faculty", "chair", "dean",
        "adjunct", "visiting professor",
        # 学位 / 学生阶段
        "bachelor", "master", "phd", "doctor", "student", "postdoc", "postdoctoral",
        "teaching assistant", "teaching fellow", "ta", "course assistant",
        # 明确的学术 fellowship（非研究类）
        "academic fellow", "university fellow",
    ]
    return any(kw in t for kw in academic_keywords)


def _reconcile_career_and_industrial_lists(
    career_items: List[schemas.CareerEducationInfo],
    industrial_items: List[schemas.ExperienceInfo],
) -> Tuple[List[schemas.CareerEducationInfo], List[schemas.ExperienceInfo]]:
    """根据职位标题对 Career & Education 与 Industrial Experience 做最后一次划分与去重。

    规则：
    - 所有 research 相关职位统一归入 industrial_experience
    - 纯教育 / 学术 / 教学职位归入 career_education_history
    - 确保同一条经历不会同时出现在两个列表中
    - 使用归一化学位标签进行去重
    """
    new_career: List[schemas.CareerEducationInfo] = []
    new_industrial: List[schemas.ExperienceInfo] = []

    # 使用更智能的 key：归一化学位标签 + 学校 + 时间段
    seen_career: Dict[Tuple[str, str, str], int] = {}
    seen_industrial: Dict[Tuple[str, str, str], int] = {}

    def career_key(item: schemas.CareerEducationInfo) -> Tuple[str, str, str]:
        """生成去重 key：使用归一化的标准标签（学位/学术职位/教学角色）。
        如果无法映射到标准标签，返回 None 作为第一个元素，表示应该被过滤。
        """
        normalized_label = _normalize_career_education_label(item.degree_or_position or "")
        # 如果无法映射到标准标签，返回 None（表示应该被过滤）
        if not normalized_label:
            return (None, "", "")
        return (
            normalized_label,
            _normalize_text_for_key(item.institution),
            _normalize_duration_for_key(item.duration),
        )

    def industrial_key(item: schemas.ExperienceInfo) -> Tuple[str, str, str]:
        return (
            _normalize_title_for_key(item.position),
            _normalize_text_for_key(item.organization),
            _normalize_duration_for_key(item.duration),
        )

    def career_score(item: schemas.CareerEducationInfo) -> Tuple[int, int, int, int, int]:
        return (
            int(bool(item.duration)),
            int(bool(item.field)),
            int(bool(item.advisor)),
            int(bool(item.department)),
            len(item.description or ""),
        )

    def industrial_score(item: schemas.ExperienceInfo) -> Tuple[int, int]:
        return (
            int(bool(item.duration)),
            len(item.description or ""),
        )

    def add_career(item: schemas.CareerEducationInfo) -> None:
        key = career_key(item)
        # 如果无法映射到标准标签，直接跳过
        if key[0] is None:
            return
        idx = seen_career.get(key)
        if idx is None:
            # 检查是否有归一化标签相同、学校相同、时间段重叠的项
            normalized_label = _normalize_career_education_label(item.degree_or_position or "")
            institution = _normalize_text_for_key(item.institution)
            duration = item.duration or ""
            
            # 如果归一化标签是标准学位标签（phd/master/bachelor/postdoc），进行更宽松的匹配
            if normalized_label in ["phd", "master", "bachelor", "postdoc"]:
                for i, existing in enumerate(new_career):
                    existing_label = _normalize_career_education_label(existing.degree_or_position or "")
                    existing_institution = _normalize_text_for_key(existing.institution)
                    existing_duration = existing.duration or ""
                    
                    # 如果归一化标签相同、学校相同、时间段重叠或接近
                    if (existing_label == normalized_label and 
                        existing_institution == institution and
                        _durations_overlap(duration, existing_duration)):
                        # 选择信息更丰富的版本
                        if career_score(item) > career_score(existing):
                            new_career[i] = item
                        return
            
            new_career.append(item)
            seen_career[key] = len(new_career) - 1
        else:
            existing = new_career[idx]
            if career_score(item) > career_score(existing):
                new_career[idx] = item

    def add_industrial(item: schemas.ExperienceInfo) -> None:
        key = industrial_key(item)
        idx = seen_industrial.get(key)
        if idx is None:
            # 检查是否有相同职位+机构、时间段重叠的项（用于合并重叠时间段）
            position = _normalize_title_for_key(item.position)
            organization = _normalize_text_for_key(item.organization)
            duration = item.duration or ""
            
            for i, existing in enumerate(new_industrial):
                existing_position = _normalize_title_for_key(existing.position or "")
                existing_organization = _normalize_text_for_key(existing.organization or "")
                existing_duration = existing.duration or ""
                
                # 如果职位和机构相同，且时间段重叠或接近，则合并
                if (existing_position == position and 
                    existing_organization == organization and
                    _durations_overlap(duration, existing_duration)):
                    # 选择信息更丰富的版本
                    if industrial_score(item) > industrial_score(existing):
                        new_industrial[i] = item
                        # 更新 seen_industrial 中的 key 映射
                        seen_industrial[key] = i
                    return
            
            new_industrial.append(item)
            seen_industrial[key] = len(new_industrial) - 1
        else:
            existing = new_industrial[idx]
            if industrial_score(item) > industrial_score(existing):
                new_industrial[idx] = item

    # 1) 处理 career_items：如遇 research 职位则移动到 industrial
    for ce in career_items or []:
        title = ce.degree_or_position or ""
        if _is_research_job_title(title):
            exp = schemas.ExperienceInfo(
                position=ce.degree_or_position,
                organization=ce.institution,
                duration=ce.duration,
                description=ce.description or ce.field or "",
            )
            add_industrial(exp)
        else:
            add_career(ce)

    # 2) 处理 industrial_items：如遇明显 academic 职位则移动到 career
    # 但只保留映射表中的内容
    for exp in industrial_items or []:
        title = exp.position or ""
        if _is_research_job_title(title):
            add_industrial(exp)
        elif _is_academic_or_training_title(title):
            # 检查是否可以映射到标准标签，如果不能则跳过
            normalized_label = _normalize_career_education_label(title)
            if normalized_label:
                ce = schemas.CareerEducationInfo(
                    degree_or_position=exp.position,
                    institution=exp.organization or "",
                    department="",
                    duration=exp.duration or "",
                    field="",
                    advisor="",
                    description=exp.description or "",
                )
                add_career(ce)
            # 如果无法映射，保持为 industrial（可能是其他类型的职位）
            else:
                add_industrial(exp)
        else:
            add_industrial(exp)

    return new_career, new_industrial


def page_requires_interaction(html: Optional[str]) -> bool:
    if not html:
        return False
    return any(re.search(pattern, html, re.IGNORECASE) for pattern in _INTERACTIVE_HTML_PATTERNS)

def _run_homepage_agent_upgrade(
    result: Dict[str, Any],
    url: str,
    author_name: str,
    max_chars: Optional[int],
    log_prefix: str = "[Homepage Fetcher]",
    api_key: str = None,
) -> Dict[str, Any]:
    """
    强制运行九维度Agent以探测交互元素，并在必要时升级抓取结果。
    """
    try:
        html_snapshot = result.get('full_html', '') or ''
        text_snapshot = result.get('text_content', '') or ''
        text_snapshot_len = len(text_snapshot)

        needs_agent = page_requires_interaction(html_snapshot)

        print(
            f"{log_prefix} 🤖 Running Nine-Dimension Agent for interactive detection "
            f"(regex_flag={needs_agent}, baseline_text_len={text_snapshot_len})"
        )

        from .nine_dimension_aware_agent import fetch_with_nine_dimension_awareness

        agent_text, agent_metadata = fetch_with_nine_dimension_awareness(
            url=url,
            author_name=author_name or "",
            api_key=api_key,
            required_dimensions=None,
            headless=True
        )

        interactive_candidates = 0
        interactive_queue_length = 0
        total_interactions = 0
        has_interactive_elements = False

        if agent_metadata:
            diagnostics = agent_metadata.get("interactive_diagnostics", {}) or {}
            interactive_candidates = diagnostics.get("total_candidates") or agent_metadata.get("interactive_candidates") or 0
            interactive_queue_length = diagnostics.get("queue_length") or agent_metadata.get("interactive_queue_length") or 0
            total_interactions = agent_metadata.get("total_interactions", 0) or 0
            has_interactive_elements = any([
                interactive_candidates > 0,
                interactive_queue_length > 0,
                total_interactions > 0
            ])
        else:
            diagnostics = {}

        if has_interactive_elements:
            print(
                f"{log_prefix} Agent detected interactive elements "
                f"(candidates={interactive_candidates}, queue={interactive_queue_length}, interactions={total_interactions})"
            )
        else:
            print(
                f"{log_prefix} Agent detected no interactive elements "
                f"(regex_flag={needs_agent})"
            )

        should_use_agent = bool(agent_text and agent_text.strip() and (needs_agent or has_interactive_elements))

        if agent_metadata:
            result['agent_metadata'] = agent_metadata
            if agent_metadata.get('extracted_dimensions'):
                result['extracted_dimensions'] = agent_metadata.get('extracted_dimensions')
                # 🆕 确保 metadata 中也包含 extracted_dimensions，以便 MultiSourceNineDimensionCollector 可以读取
                if not result.get('metadata'):
                    result['metadata'] = {}
                result['metadata']['extracted_dimensions'] = agent_metadata.get('extracted_dimensions')
                if agent_metadata.get('extracted_dimensions_meta'):
                    result['metadata']['extracted_dimensions_meta'] = agent_metadata.get('extracted_dimensions_meta')
                print(f"{log_prefix} ✅ Stored extracted_dimensions in result['metadata'] for MultiSourceNineDimensionCollector")

        result['interactive_diagnostics'] = diagnostics
        result['interactive_candidates'] = interactive_candidates
        result['interactive_queue_length'] = interactive_queue_length
        result['total_interactions'] = total_interactions
        result['has_interactive_elements'] = has_interactive_elements

        if should_use_agent:
            trimmed_text = agent_text[:max_chars] if max_chars else agent_text
            result['text_content'] = trimmed_text
            result['method'] = 'nine_dimension_agent'
            result['fetched_via'] = 'nine_dimension_agent'
            result['agent_text_length'] = len(agent_text or "")
            print(f"{log_prefix} ✅ Nine-Dimension Agent fetched {len(agent_text or '')} chars (trimmed to {len(trimmed_text)})")
        else:
            print(
                f"{log_prefix} ℹ️ Retaining traditional parser output "
                f"(should_use_agent={should_use_agent}, has_interactive={has_interactive_elements})"
            )
    except Exception as agent_err:
        print(f"{log_prefix} ⚠️ Nine-Dimension Agent invocation failed: {agent_err}")

    return result

# ============================ LLM PROMPTS ============================

# Profile身份验证prompt（用于LinkedIn/Twitter等社交平台）
PROMPT_VERIFY_PROFILE_IDENTITY = lambda author_name, platform, url, content_preview: f"""
Verify if this {platform} profile belongs to the target researcher.

TARGET AUTHOR: {author_name}

PROFILE TO VERIFY:
Platform: {platform}
URL: {url}
Content Preview: {content_preview[:1000]}

Is this profile definitely for the target author {author_name}?

VERIFICATION CRITERIA:
✓ Name match (exact or reasonable variations)
✓ Research area consistency 
✓ Institution/affiliation match
✓ Publication overlap
✓ Profile completeness and authenticity

REJECT IF:
✗ Different person with similar name
✗ Generic/incomplete profile
✗ Conflicting information (different field, institution)
✗ Suspicious/fake profile indicators

Return JSON: {{"is_target_author": true/false, "confidence": 0.0-1.0, "reason": "<specific reason>"}}
"""

# 通用字段抽取prompt
PROFILE_EXTRACT_PROMPT = lambda author_name, dump, platform_type="generic": f"""
Extract author profile fields from {platform_type} page content.

TARGET AUTHOR: {author_name}
PLATFORM TYPE: {platform_type}

TEXT CONTENT:
{dump}
==== END ====

Return STRICT JSON with these keys:
{{
  "name": "<full name as written>",
  "aliases": ["ONLY alternative names/nicknames of THIS AUTHOR"],
  "affiliation_current": "<FULL role + institution, e.g. 'PhD student at MIT' or 'Postdoc at Stanford' or 'Professor at CMU'>",
  "emails": ["professional emails only"],
  "personal_homepage": "<personal website URL if different from current page>",
  "interests": ["research areas/topics"],
  "selected_publications": [{{"title":"...", "year":2024, "venue":"...", "url":"..."}}],
  "notable_achievements": ["awards/honors/recognitions"],
  "social_impact": "<h-index, citations, influence metrics>",
  "career_stage": "<student/postdoc/assistant_prof/associate_prof/full_prof/industry>",
  "social_links": {{"platform": "url"}}
}}

CRITICAL EXTRACTION RULES:
1. **Name**: Use the most complete/formal version found for THIS AUTHOR ONLY
2. **Aliases**: ONLY include alternative names/nicknames of THE TARGET AUTHOR
3. **Affiliation**: MUST include both role AND institution (e.g., "PhD student at XYZ University", "Associate Professor at ABC Institute", "Research Scientist at Company"). Never just the institution name alone.
4. **Emails**: Only work/institutional emails visible on page
5. **Homepage**: Personal website URL (NOT the current page URL)
6. **Interests**: Specific research areas, not generic terms
7. **Publications**: Max 3 most recent/important papers by THIS AUTHOR
8. **Notable**: Awards, fellowships, best papers of THIS AUTHOR only
9. **Social Impact**: Citation counts, h-index of THIS AUTHOR
10. **Career Stage**: Current career stage of THIS AUTHOR
11. **Social Links**: Extract social media/platform links from the page

PLATFORM-SPECIFIC HINTS:
- OpenReview: Focus on reviews, paper submissions, expertise areas  
- Google Scholar: Emphasize citation metrics, publication trends
- ORCID: Look for comprehensive work history, affiliations
- University pages: Focus on teaching, research groups, lab info
- GitHub: Technical projects, code contributions, collaboration

If any field is unclear/absent, return empty string "" or empty list [].
DO NOT invent information not present in the text.
NEVER include names of other people in aliases field.
"""

# ============================ ENHANCED PROFILE LLM PROMPTS ============================

PROMPT_SYNTHESIZE_INTERESTS_FROM_PAPERS = """
Based on the researcher's publication titles (and abstracts if available), synthesize **BROAD and DISTINCT** research areas.

Publications:
{papers_list}

Background (optional context):
- Bio: {bio}
- Google Scholar tags: {scholar_tags}

Requirements:
1. Generate **3-4 BROAD** research areas (prefer 3, max 4).
2. Think **UMBRELLA TERMS** that group multiple related papers together.
3. **Grouping Threshold**: Each area should be BROAD enough to cover 3+ papers.
    * **If total papers are less than 6**, you may reduce the number of required areas to **2-3**, but MUST still adhere to the BROADness principle.
4. Each research direction needs to be described as clearly as possible..
5. Interest names should be 3-6 words (avoid single-word topics).
6. Areas must be **DISTINCT** with minimal overlap.
7. **OUTPUT IN ENGLISH**: All content must be in English.

**CRITICAL - Think BROAD categories, NOT specific techniques**:
Return JSON format (ALL IN ENGLISH):
{{
  "research_interests": [
    {{
      "name": "<BROAD research area name>",
      "description": "<Describe the characteristics of the area in as much detail as possible.>"
    }},
    {{
      "name": "<BROAD research area name>",
      "description": "<Describe the characteristics of the area in as much detail as possible.>"
    }}
  ]
}}
Return ONLY valid JSON with 2-4 broad research areas, no markdown, no explanation.
"""

# 判断描述是否需要细化
PROMPT_EVALUATE_DESCRIPTION_QUALITY = """
You are evaluating the quality of a research interest description.

Research Interest Name: {interest_name}
Current Description: {current_description}

**EVALUATION CRITERIA**:
A description needs refinement if it is:
1. A simple keyword list (e.g., "Topic A, Topic B, Topic C")
2. Too short or vague (less than 30 characters)
3. Missing key details about methods, goals, or applications

A description is already good if it:
1. Contains complete sentences
2. Describes methods, goals, or specific techniques
3. Provides clear context about the research area

**YOUR TASK**: 
Decide if this description needs refinement based on papers.

Return ONLY a JSON object:
{{
  "needs_refinement": true/false,
  "reason": "Brief explanation why it needs or doesn't need refinement"
}}
Your response (JSON only):
"""

# 细化研究方向描述（基于该方向下的论文）
PROMPT_REFINE_INTEREST_DESCRIPTION = """
Based on the following papers in the research area "{interest_name}", generate a detailed 2-3 sentence description of this research interest.

Papers in this area:
{papers_in_category}

Current description: {current_description}

**YOUR TASK**:
- If current description is a keyword list (e.g., "Topic A, Topic B, Topic C"), expand it into complete sentences
- If current description is empty or vague, create a comprehensive new description
- If current description is already detailed, enhance it with insights from the papers

Generate a more detailed and specific description based on the actual paper titles and abstracts.

The description should:
1. Be 2-3 complete sentences (not just keywords)
2. Capture the key themes, methods, and applications from the papers
3. Be specific and technical (not generic)
4. Include concrete examples or techniques when possible
5. **OUTPUT IN ENGLISH**: All content must be in English

**CRITICAL**: Return ONLY a JSON object with "value" field containing the description:

{{
  "value": "Your detailed description here"
}}

Your response (JSON only):
"""

# 第三阶段：论文分类
PROMPT_CLASSIFY_PUBLICATION = """
Classify this research paper into ONE category from the list below.

**Categories**:
{interests_list}

**Paper**:
{description}

**Instructions**:
1. Read the paper title and abstract carefully
2. Identify the MAIN research topic (not just mentioned keywords)
3. Match to the MOST RELEVANT category above
4. Be PRECISE: Don't put all papers in one category - distribute them based on actual topic match
5. You MUST choose from the categories above - choose the closest match even if not perfect

**Critical**: Return ONLY a JSON object with "value" field containing the exact category name from the list above:

{{
  "value": "Category Name Here"
}}

Example:
{{
  "value": "Cognitive Knowledge Structure"
}}

Your response (JSON only):
"""

# ============================ SCORING AND EVALUATION FUNCTIONS ============================

def check_url_redirect(url: str, max_redirects: int = 5) -> Tuple[str, bool]:
    """
    检查URL是否有重定向，返回最终URL和是否发生了重定向
    
    Args:
        url: 原始URL
        max_redirects: 最大重定向次数
        
    Returns:
        (final_url, redirected)
    """
    try:
        # 使用HEAD请求检查重定向，避免下载完整内容
        response = requests.head(url, allow_redirects=True, timeout=10, headers=config.UA)
        final_url = response.url
        
        # 规范化URL比较
        original_normalized = normalize_url(url)
        final_normalized = normalize_url(final_url)
        
        redirected = original_normalized != final_normalized
        
        if redirected:
            print(f"[URL Redirect] {url} → {final_url}")
        
        return final_url, redirected
        
    except Exception as e:
        print(f"[URL Redirect] Failed to check redirect for {url}: {e}")
        return url, False

def verify_profile_identity(author_name: str, platform: str, url: str, content: str, 
                          llm_client) -> Tuple[bool, float, str]:
    """
    使用LLM验证profile是否属于目标作者
    
    Args:
        author_name: 目标作者姓名
        platform: 平台类型 (linkedin, twitter, scholar, etc.)
        url: profile URL
        content: 页面内容预览
        llm_client: LLM客户端
        
    Returns:
        (is_target_author, confidence, reason)
    """
    # 对于权威学术平台，降低验证要求
    if platform in ['orcid', 'openreview', 'scholar', 'semanticscholar']:
        # 简单的名字匹配检查
        author_words = set(author_name.lower().split())
        content_lower = content.lower()
        
        # 检查是否有足够的名字匹配
        name_matches = sum(1 for word in author_words if len(word) > 2 and word in content_lower)
        if name_matches >= len(author_words) * 0.6:  # 60%的名字词汇匹配
            return True, 0.8, f"Academic platform with name match"
    
    # 对于社交平台，使用LLM严格验证
    if platform in ['linkedin', 'twitter', 'researchgate']:
        try:
            prompt = PROMPT_VERIFY_PROFILE_IDENTITY(author_name, platform, url, content)
            result = llm.safe_structured(llm_client, prompt, schemas.LLMSelectSpecVerifyIdentity)
            
            if result:
                is_target = bool(getattr(result, 'is_target_author', False))
                confidence = float(getattr(result, 'confidence', 0.0))
                reason = str(getattr(result, 'reason', 'LLM verification'))
                return is_target, confidence, reason
        except Exception as e:
            print(f"[Profile Verification] LLM failed for {platform}: {e}")
    
    # 默认：基于内容的简单验证
    author_words = set(author_name.lower().split())
    content_lower = content.lower()
    name_matches = sum(1 for word in author_words if len(word) > 2 and word in content_lower)
    
    if name_matches >= len(author_words) * 0.7:
        return True, 0.6, "Basic name matching"
    
    return False, 0.2, "Insufficient name match"

# ============================ SMART PLATFORM URL MANAGEMENT ============================

def should_update_platform_url(profile: AuthorProfile, platform_type: str, new_url: str, author_name: str) -> bool:
    """判断是否应该更新平台URL - 使用LLM验证关键更新"""
    current_url = profile.platforms.get(platform_type)
    if not current_url:
        return True  # 没有现有URL，直接添加
    
    # 对于LinkedIn和Twitter，如果现有URL质量很低，需要LLM验证
    if platform_type in ['linkedin', 'twitter']:
        current_quality = assess_url_quality(current_url, platform_type, author_name)
        new_quality = assess_url_quality(new_url, platform_type, author_name)
        
        # 如果新URL质量显著更高，且现有URL质量很低，直接更新
        if current_quality < 0.3 and new_quality > 0.6:
            return True
        
        # 如果质量相近，保持现有URL
        if abs(new_quality - current_quality) < 0.2:
            return False
    
    return assess_url_quality(new_url, platform_type, author_name) > assess_url_quality(current_url, platform_type, author_name)

def update_platform_url(profile: AuthorProfile, platform_type: str, new_url: str, author_name: str):
    """智能更新平台URL，优先保留更准确的链接"""
    
    if should_update_platform_url(profile, platform_type, new_url, author_name):
        profile.platforms[platform_type] = new_url

def assess_url_quality(url: str, platform: str, author: str) -> float:
    """评估URL质量"""
    score = 0.0
    author_lower = author.lower().replace(' ', '')
    author_parts = [part.lower() for part in author.split()]
    
    # 包含作者名字的URL质量更高
    if any(part in url.lower() for part in author_parts if len(part) > 2):
        score += 0.5
    
    # 特定平台的质量指标
    if platform == 'scholar' and 'citations?user=' in url:
        score += 0.4
    elif platform == 'github' and not any(bad in url for bad in ['/orgs/', '/topics/', '/search']):
        score += 0.4
    elif platform == 'linkedin':
        if '/in/' in url and not '/directory/' in url:
            score += 0.6  # LinkedIn个人档案
            # 检查用户名是否与作者相关
            linkedin_username = url.split('/in/')[-1].split('/')[0].split('?')[0]
            if any(part.lower() in linkedin_username.lower() for part in author_parts if len(part) > 2):
                score += 0.4
        elif '/directory/' in url:
            score -= 0.5  # 目录页面质量很低
    elif platform == 'twitter':
        if not any(bad in url for bad in ['/status/', '/search', '?lang=', '/hashtag/']):
            # 检查用户名是否与作者相关
            twitter_username = url.split('/')[-1].split('?')[0]
            name_match = any(part.lower() in twitter_username.lower() for part in author_parts if len(part) > 2)
            if name_match:
                score += 0.7  # 用户名匹配的Twitter账号
            else:
                score += 0.2  # 用户名不匹配的Twitter账号质量低
    elif platform == 'orcid' and re.search(r'\d{4}-\d{4}-\d{4}-\d{4}', url):
        score += 0.5
    elif platform == 'openreview' and 'profile?id=' in url:
        score += 0.4
    elif platform == 'homepage':
        # 个人域名优于托管服务
        if any(domain in url for domain in ['.com/', '.org/', '.net/', '.edu/']):
            score += 0.5
        if 'github.io' in url:
            score += 0.3
    
    # 惩罚明显错误的URL
    if any(bad in url.lower() for bad in ['directory', 'search', 'random', 'example']):
        score -= 0.3
        
    return max(0.0, score)

def validate_social_link_for_author(platform: str, url: str, author_name: str) -> bool:
    """
    验证社交媒体链接是否真的属于目标作者
    
    Args:
        platform: 平台类型
        url: 链接URL
        author_name: 目标作者姓名
        
    Returns:
        是否有效
    """
    if not url or not url.startswith('http'):
        return False
    
    author_words = [word.lower() for word in author_name.split() if len(word) > 2]
    url_lower = url.lower()
    
    # Twitter/X validation
    if platform == 'twitter':
        if 'x.com/' in url_lower or 'twitter.com/' in url_lower:
            # Exclude invalid URLs
            if any(bad in url_lower for bad in ['/status/', '/search', '/hashtag/', '?lang=', '/i/']):
                return False
            
            # Extract username
            username_part = url_lower.split('/')[-1].split('?')[0]
            
            # Check username validity
            if len(username_part) < 3 or len(username_part) > 20:
                return False
            
            # Check if username matches author name
            name_match = any(word in username_part for word in author_words)
            return name_match
        return False
    
    # LinkedIn validation
    elif platform == 'linkedin':
        if 'linkedin.com/in/' in url_lower:
            # Exclude directory pages
            if '/directory/' in url_lower:
                return False
            
            username_part = url_lower.split('/in/')[-1].split('/')[0].split('?')[0]
            
            # Check username length
            if len(username_part) < 3:
                return False
            
            # Check if username matches author name
            name_match = any(word in username_part for word in author_words)
            return name_match
        return False
    
    # GitHub validation
    elif platform == 'github':
        if 'github.com/' in url_lower:
            # Exclude organization and search pages
            if any(bad in url_lower for bad in ['/orgs/', '/search', '/topics/', '/trending']):
                return False
            
            username_part = url_lower.split('github.com/')[-1].split('/')[0].split('?')[0]
            
            if len(username_part) < 2:
                return False
            
            # GitHub username usually relates to author name
            name_match = any(word in username_part for word in author_words)
            return name_match
        return False
    
    # Scholar validation
    elif platform == 'scholar':
        return 'citations?user=' in url_lower
    
    # Default validation for other platforms
    return True

def check_url_accessibility(url: str, timeout: int = 3) -> Tuple[bool, str]:
    """
    快速检查URL是否可访问（轻量级，只做HEAD请求）
    Args:
        url: 要检查的URL
        timeout: 超时时间（秒），建议2-3秒   
    Returns:
        (is_accessible, reason)
    """
    try:
        # 使用HEAD请求（不下载内容，速度快）
        response = requests.head(
            url, 
            timeout=timeout, 
            allow_redirects=True,
            headers={'User-Agent': 'Mozilla/5.0 (Academic Research Bot)'}
        )
        
        # 2xx和3xx状态码认为可访问
        if 200 <= response.status_code < 400:
            return True, f"Accessible (HTTP {response.status_code})"
        else:
            return False, f"HTTP {response.status_code}"
    except requests.exceptions.Timeout:
        return False, "Timeout (>3s)"
    except requests.exceptions.ConnectionError:
        return False, "Connection failed"
    except requests.exceptions.TooManyRedirects:
        return False, "Redirect loop"
    except requests.exceptions.SSLError:
        # SSL错误时尝试用GET验证（有些网站HEAD不支持但GET可以）
        try:
            response = requests.get(url, timeout=timeout, stream=True, headers=config.UA)
            response.raw.read(100)  # 只读100字节
            return True, f"Accessible via GET (HTTP {response.status_code})"
        except:
            return False, "SSL certificate error"
    except Exception as e:
        return False, f"Error: {str(e)[:30]}"

def _normalize_identity_term(term: str) -> Optional[str]:
    """Normalize degree/organization terms for identity verification."""
    if not term:
        return None
    cleaned = re.sub(r"[^a-z0-9 ]+", " ", str(term).lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) < 3:
        return None
    return cleaned


def _collect_identity_clues(profile, openreview_result: Dict[str, Any]) -> Dict[str, set[str]]:
    """Collect degree/organization hints from education & industrial experience for homonym filtering."""
    degree_terms: set[str] = set()
    org_terms: set[str] = set()
    def _add_term(value: str, target: set[str]):
        normalized = _normalize_identity_term(value)
        if normalized:
            target.add(normalized)
    # OpenReview baseline data
    for item in openreview_result.get('or_career_education') or []:
        _add_term(item.get('degree_or_position', ''), degree_terms)
        institution = item.get('institution', '')
        if isinstance(institution, dict):
            institution = institution.get('name', '')
        _add_term(institution, org_terms)

    for item in openreview_result.get('or_industrial_experience') or []:
        _add_term(item.get('position', '') or item.get('title', ''), degree_terms)
        _add_term(item.get('organization', '') or item.get('company', ''), org_terms)

    # Aggregated profile data (if available)
    for item in getattr(profile, 'career_education_history', []) or []:
        _add_term(getattr(item, 'degree_or_position', ''), degree_terms)
        _add_term(getattr(item, 'institution', ''), org_terms)

    for item in getattr(profile, 'industrial_experience', []) or []:
        _add_term(getattr(item, 'position', ''), degree_terms)
        _add_term(getattr(item, 'organization', ''), org_terms)

    return {
        'degree_terms': {t for t in degree_terms if len(t) >= 3},
        'org_terms': {t for t in org_terms if len(t) >= 3},
    }


def _passes_basic_identity(name_match_ratio: float, author_name_parts: List[str], content_lower: str, has_identity_clues: bool) -> Tuple[bool, str]:
    """Stricter name-based validation to avoid homonyms when clues are missing."""

    if not author_name_parts:
        return False, "No valid author name parts"

    last_name = author_name_parts[-1]
    if last_name not in content_lower:
        return False, "Last name not found in content"

    threshold = 0.5 if has_identity_clues else 0.75
    if name_match_ratio < threshold:
        return False, f"Name match below {int(threshold * 100)}% threshold when {'with' if has_identity_clues else 'without'} clues"

    return True, ""


def _passes_degree_verification(preview_content: str, identity_clues: Dict[str, set[str]]) -> Tuple[bool, List[str], List[str], bool]:
    """Check whether homepage preview overlaps with known education/experience clues."""

    normalized_content = _normalize_identity_term(preview_content) or ""
    compressed_content = normalized_content.replace(" ", "")

    def _match_terms(terms: set[str]) -> List[str]:
        matched = []
        for term in terms:
            compressed_term = term.replace(" ", "")
            if term in normalized_content or compressed_term in compressed_content:
                matched.append(term)
        return sorted(set(matched))

    matched_degrees = _match_terms(identity_clues.get('degree_terms', set()))
    matched_orgs = _match_terms(identity_clues.get('org_terms', set()))

    has_clues = bool(identity_clues.get('degree_terms') or identity_clues.get('org_terms'))
    if not has_clues:
        return True, matched_degrees, matched_orgs, False

    has_match = bool(matched_degrees or matched_orgs)
    return has_match, matched_degrees, matched_orgs, True



def _parse_web_search_result(result: Any) -> List[Dict[str, str]]:
    """
    解析 web_search 工具返回的结果为统一格式
    Args:
        result: web_search 返回的结果（可能是字符串、列表或字典）
    Returns:
        统一格式的搜索结果列表 [{'url': '...', 'title': '...', 'snippet': '...'}, ...]
    """
    import json
    parsed = []
    
    if isinstance(result, str):
        try:
            data = json.loads(result)
            if isinstance(data, list):
                parsed = data
            elif isinstance(data, dict):
                if 'results' in data:
                    parsed = data['results']
                elif 'url' in data:
                    parsed = [data]
        except json.JSONDecodeError:
            pass
    elif isinstance(result, list):
        parsed = result
    elif isinstance(result, dict):
        if 'results' in result:
            parsed = result['results']
        else:
            parsed = [result]
    
    # 标准化字段名
    standardized = []
    for item in parsed:
        if isinstance(item, dict):
            standardized.append({
                'url': item.get('url', item.get('link', '')),
                'title': item.get('title', ''),
                'snippet': item.get('snippet', item.get('content', item.get('description', ''))),
            })
    
    return standardized

def _execute_web_search_query(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """
    使用 web_search 模块执行搜索查询
    Args:
        query: 搜索查询
        max_results: 最大结果数
    Returns:
        搜索结果列表
    """
    try:
        results = search.searxng_search(query, pages=1, k_per_query=max_results)
        return _parse_web_search_result(results)
    except Exception as e:
        print(f"[Web Search] Error executing query '{query[:50]}...': {e}")
        return []

def _execute_homepage_search_v2(
    author_name: str, 
    known_affiliation: str = "",
    paper_title: str = "",
    timeout_seconds: int = 30
) -> List[Dict]:
    """
    执行homepage搜索策略（回退机制专用）- 新版本
    使用 name + 机构 + homepage 的形式搜索，有效降低重名风险
    搜索策略：
    - 有机构：所有查询都带机构信息，零重名风险
    - 无机构：必须有论文标题，否则直接放弃
    
    Args:
        author_name: 作者姓名
        known_affiliation: 已知的机构/组织信息
        paper_title: 论文标题（用于无机构时的备选搜索）
        timeout_seconds: 总超时时间（默认30秒）
        
    Returns:
        搜索结果列表 [{'url': '...', 'title': '...', 'snippet': '...'}, ...]
    """
    print(f"[Homepage Fallback V2] Starting search for '{author_name}'")
    print(f"  - Affiliation: {known_affiliation or '(none)'}")
    print(f"  - Paper title: {paper_title if paper_title else '(none)'}")
    # 构建搜索查询（参考 web_search_fallback.py 的策略）
    from .web_search_fallback import extract_paper_title_keywords
    homepage_search_strategies = []
    paper_keywords = extract_paper_title_keywords(paper_title)
    if known_affiliation:
        # ==== 有机构信息：所有查询都带机构 ====
        affiliation_clean = known_affiliation.split(',')[0].strip()
        # Query 1: 姓名 + 机构（最精确）
        homepage_search_strategies.append((f'"{author_name}" {affiliation_clean}', 5))
        # Query 2: 姓名 + 机构 + homepage
        homepage_search_strategies.append((f'"{author_name}" {affiliation_clean} homepage', 3))
        # Query 3: 如果有论文关键词，增加第3个查询
        if paper_keywords:
            homepage_search_strategies.append((f'"{author_name}" {affiliation_clean} {paper_keywords}', 3))
        print(f"[Homepage Fallback V2] Using affiliation-based search (low ambiguity)")
    else:
        # ==== 无机构信息：必须有论文标题 ====
        if not paper_keywords:
            # 无机构且无论文关键词：直接放弃，风险太高
            print(f"[Homepage Fallback V2] Skipped: No affiliation and no paper title keywords (HIGH AMBIGUITY RISK)")
            return []
        # Query 1: 姓名 + 论文关键词
        homepage_search_strategies.append((f'"{author_name}" {paper_keywords}', 5))
        # Query 2: 姓名 + 论文关键词 + author
        homepage_search_strategies.append((f'"{author_name}" {paper_keywords} author', 3))
        print(f"[Homepage Fallback V2]    Using paper-title-based search (medium ambiguity)")
        print(f"[Homepage Fallback V2]    Paper keywords: {paper_keywords}")
    # 限制最多3个查询
    homepage_search_strategies = homepage_search_strategies[:3]
    
    start_time = time.time()
    all_results = []
    
    try:
        num_strategies = len(homepage_search_strategies)
        max_workers = get_optimal_workers(num_strategies, 'io_bound')
        max_workers = min(max_workers, num_strategies)
        
        print(f"[Homepage Fallback V2] Launching {num_strategies} strategies with {max_workers} workers")
        for idx, (query, _) in enumerate(homepage_search_strategies, 1):
            print(f"  Query {idx}: {query}")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_strategy = {}
            for idx, (query, max_results) in enumerate(homepage_search_strategies):
                future = executor.submit(
                    _execute_web_search_query,
                    query,
                    max_results
                )
                future_to_strategy[future] = (idx + 1, query)
            
            # 收集结果
            completed_count = 0
            for future in as_completed(future_to_strategy.keys(), timeout=timeout_seconds):
                strategy_num, query = future_to_strategy[future]
                elapsed = time.time() - start_time
                
                try:
                    res = future.result(timeout=1)
                    completed_count += 1
                    
                    if res:
                        all_results.extend(res)
                        print(f"[Homepage Fallback V2] Strategy {strategy_num} → {len(res)} results ({elapsed:.1f}s)")
                    else:
                        print(f"[Homepage Fallback V2] Strategy {strategy_num} → no results ({elapsed:.1f}s)")
                        
                except Exception as e:
                    print(f"[Homepage Fallback V2] Strategy {strategy_num} error: {str(e)}")
                    completed_count += 1
            
            print(f"[Homepage Fallback V2] Completed {completed_count}/{num_strategies} strategies")
            
    except TimeoutError:
        elapsed = time.time() - start_time
        print(f"[Homepage Fallback V2] Timeout after {elapsed:.1f}s, collected {len(all_results)} results")
        for future in future_to_strategy.keys():
            future.cancel()
    except Exception as e:
        print(f"[Homepage Fallback V2] Error: {e}")
    
    elapsed = time.time() - start_time
    print(f"[Homepage Fallback V2] Total time: {elapsed:.1f}s, found {len(all_results)} candidates")
    return all_results


def extract_social_links_from_content(content: str, base_url: str = "") -> Dict[str, str]:
    """从页面内容中提取社交媒体链接 - 增强版"""
    social_links = {}
    
    # 更全面的社交媒体链接模式，包括更多变体
    patterns = {
        'scholar': [
            r'https?://scholar\.google\.com/citations\?user=([A-Za-z0-9_\-]+)',
            r'https?://scholar\.google\.com/citations\?hl=[^&]*&user=([A-Za-z0-9_\-]+)',
            r'scholar\.google\.com/citations\?user=([A-Za-z0-9_\-]+)',  # 无协议版本
        ],
        'github': [
            r'https?://github\.com/([A-Za-z0-9_\-]+)(?:/[^"\s]*)?',
            r'github\.com/([A-Za-z0-9_\-]+)',  # 无协议版本
        ],
        'linkedin': [
            r'https?://(?:www\.)?linkedin\.com/in/([A-Za-z0-9_\-]+)',
            r'linkedin\.com/in/([A-Za-z0-9_\-]+)',  # 无协议版本
        ],
        'twitter': [
            r'https?://(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)',
            r'(?:x\.com|twitter\.com)/([A-Za-z0-9_]+)',  # 无协议版本
            r'@([A-Za-z0-9_]+)',  # @username 格式
        ],
        'orcid': [
            r'https?://orcid\.org/(\d{4}-\d{4}-\d{4}-\d{4})',
            r'orcid\.org/(\d{4}-\d{4}-\d{4}-\d{4})',
        ],
        'openreview': [
            r'https?://openreview\.net/profile\?id=([A-Za-z0-9_\-\.%~]+)',
            r'openreview\.net/profile\?id=([A-Za-z0-9_\-\.%~]+)',
        ],
        'huggingface': [
            r'https?://huggingface\.co/([A-Za-z0-9_\-]+)',
            r'huggingface\.co/([A-Za-z0-9_\-]+)',
        ],
    }
        
    for platform, pattern_list in patterns.items():
        for pattern in pattern_list:
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                print(f"[Regex Match] {platform}: {pattern} found {len(matches)} matches: {matches[:3]}")  # 显示前3个匹配
                
                # 取第一个匹配的链接，但排除明显错误的
                valid_match = None
                for match in matches:
                    # 对Twitter特殊处理@username格式
                    if platform == 'twitter' and pattern.startswith(r'@'):
                        # 排除过短或明显不是用户名的匹配
                        if len(match) >= 3 and not any(bad in match.lower() for bad in ['http', 'www', 'com']):
                            valid_match = match
                            break
                    else:
                        # 其他平台的常规处理
                        if match and len(match) > 2:
                            valid_match = match
                            break
                
                if valid_match:
                    # 构建完整URL
                    if platform == 'scholar':
                        social_links[platform] = f"https://scholar.google.com/citations?user={valid_match}"
                    elif platform == 'github':
                        social_links[platform] = f"https://github.com/{valid_match}"
                    elif platform == 'linkedin':
                        social_links[platform] = f"https://www.linkedin.com/in/{valid_match}"
                    elif platform == 'twitter':
                        social_links[platform] = f"https://x.com/{valid_match}"
                    elif platform == 'orcid':
                        social_links[platform] = f"https://orcid.org/{valid_match}"
                    elif platform == 'openreview':
                        social_links[platform] = f"https://openreview.net/profile?id={valid_match}"
                    elif platform == 'huggingface':
                        social_links[platform] = f"https://huggingface.co/{valid_match}"
                    
                    break  # 找到有效匹配后跳出内层循环
    
    print(f"[Regex Summary] Extracted {len(social_links)} social links: {list(social_links.keys())}")
    return social_links

# ============================ PROFILE MERGING FUNCTIONS ============================

def process_homepage_candidate(candidate: ProfileCandidate, author_name: str, paper_title: str, 
                             profile: AuthorProfile, protected_platforms: set, llm_ext) -> bool:
    """
    处理homepage类型的候选者
    
    Args:
        candidate: 候选者信息
        author_name: 目标作者姓名
        paper_title: 论文标题
        profile: 当前作者档案
        protected_platforms: 受保护的平台集合
        llm_ext: LLM客户端
        
    Returns:
        是否成功处理
    """
    print(f"[Homepage Candidate] Processing: {candidate.url}")
    
    # Check if this is a trusted source (from OpenReview)
    is_trusted = getattr(candidate, 'trusted_source', False)
    
    # 1. 检查URL重定向
    final_url, redirected = check_url_redirect(candidate.url)
    working_url = final_url if redirected else candidate.url
    
    if redirected:
        print(f"[Homepage Redirect] Using final URL: {working_url}")
    
    # 2. 所有 homepage 都来自 OpenReview（trusted source），跳过验证
    # Note: All homepage URLs come from OpenReview profile, no need for validation
    if not is_trusted:
        print(f"[Homepage] Warning: Non-trusted homepage URL encountered (should not happen in current flow)")
        return False
    
    print(f"[Homepage Trusted] From OpenReview, skipping validation, directly processing")
    
    # 3. 身份验证通过，进行全面抓取（使用最终URL）
    print(f"[Homepage] Identity verified, starting comprehensive fetch")
    homepage_result = fetch_homepage_comprehensive(working_url, author_name, max_chars=50000, include_subpages=True, max_subpages=6)
    
    if not homepage_result['success']:
        print(f"[Homepage] Comprehensive fetch failed, using fallback")
        txt = fetch_text(working_url, max_chars=30000, snippet=candidate.snippet)
        if not txt or len(txt) < config.MIN_TEXT_LENGTH:
            return False
        print(f"[Homepage] Fallback content fetched: {len(txt)} characters")
    else:
        txt = homepage_result['text_content']
        print(f"[Homepage] Successfully fetched comprehensive content: {len(txt)} characters")
        
        # 3. 直接从HTML提取的高质量链接
        html_social_links = homepage_result['social_platforms']
        html_emails = homepage_result['emails']
        
        print(f"[Homepage Integration] Adding {len(html_social_links)} social links and {len(html_emails)} emails")
        
        # 添加社交平台链接（最高优先级，但需要验证）
        for platform, url in html_social_links.items():
            if platform not in profile.platforms and validate_social_link_for_author(platform, url, author_name):
                profile.platforms[platform] = url
                protected_platforms.add(platform)
                print(f"[Homepage Direct] Added {platform}: {url}")
            elif not validate_social_link_for_author(platform, url, author_name):
                print(f"[Homepage Rejected] Invalid {platform} link: {url}")
        
        # 添加邮箱（经过过滤）
        for email in html_emails:
            if email not in profile.emails and is_email_relevant_to_author(email, author_name):
                profile.emails.append(email)
                print(f"[Homepage Direct] Added email: {email}")
    return True

def process_regular_candidate(candidate: ProfileCandidate, author_name: str, 
                            profile: AuthorProfile, protected_platforms: set, llm_ext) -> bool:
    """
    处理非homepage类型的候选者
    
    Args:
        candidate: 候选者信息
        author_name: 目标作者姓名
        profile: 当前作者档案
        protected_platforms: 受保护的平台集合
        llm_ext: LLM客户端
        
    Returns:
        是否成功处理
    """
    # 1. 确定平台类型
    host = domain_of(candidate.url)
    platform_type = determine_platform_type(candidate.url, host)
    
    if not platform_type:
        return False
    
    # 2. 抓取内容
    max_chars = config.FETCH_MAX_CHARS
    txt = fetch_text(candidate.url, max_chars=max_chars, snippet=candidate.snippet)
    
    if not txt or len(txt) < config.MIN_TEXT_LENGTH:
        print(f"[Regular Candidate] Failed to fetch sufficient content from {candidate.url}")
        return False
    
    print(f"[Regular Candidate] Fetched {len(txt)} characters from {candidate.url}")
    
    # 3. 对社交平台进行身份验证
    if platform_type in ['linkedin', 'twitter', 'researchgate']:
        is_target, confidence, reason = verify_profile_identity(
            author_name, platform_type, candidate.url, txt[:1000], llm_ext
        )
        print(f"[Profile Verification] {platform_type}: {is_target} (conf: {confidence:.2f})")
        
        if not is_target or confidence < 0.6:
            print(f"[Profile Rejected] {platform_type} profile rejected")
            return False
    
    # 4. 更新平台URL
    if platform_type not in protected_platforms:
        update_platform_url(profile, platform_type, candidate.url, author_name)
    else:
        print(f"[Skipped Platform] {platform_type} already protected by homepage")
    
    # 5. LLM内容提取
    dump = txt[:8000]
    platform_hint = get_platform_hint(host)
    prompt = PROFILE_EXTRACT_PROMPT(author_name, dump, platform_hint)
    
    try:
        ext = llm.safe_structured(llm_ext, prompt, schemas.LLMAuthorProfileSpec)
        if ext:
            process_extracted_profile_info(ext, candidate.url, author_name, profile, protected_platforms, is_homepage=False)
            return True
    except Exception as e:
        print(f"[Regular LLM] Extraction failed for {candidate.url}: {e}")
    
    return False

def determine_platform_type(url: str, host: str) -> str:
    """确定平台类型 - 增强homepage检测"""
    url_lower = url.lower()
    
    # 权威学术平台
    if 'orcid.org' in host:
        return 'orcid'
    elif 'openreview.net' in host:
        return 'openreview'
    elif 'scholar.google.' in host:
        return 'scholar'
    elif 'semanticscholar.org' in host:
        return 'semanticscholar'
    elif 'dblp.org' in host:
        return 'dblp'
    
    # 机构网站
    elif host.endswith('.edu') or host.endswith('.ac.nz') or host.endswith('.ac.uk'):
        return 'university'
    
    # 个人网站检测 - 增强版
    elif 'github.io' in host:
        return 'homepage'
    elif any(personal_indicator in host for personal_indicator in [
        'personal', 'homepage', 'home', 'about', 'profile'
    ]):
        return 'homepage'
    elif any(domain_pattern in host for domain_pattern in [
        '.com', '.org', '.net', '.me', '.io'
    ]) and not any(platform in host for platform in [
        'github.com', 'linkedin.com', 'twitter.com', 'x.com', 'facebook.com',
        'instagram.com', 'youtube.com', 'medium.com', 'reddit.com'
    ]):
        # 可能是个人域名，进一步检查URL路径
        if any(indicator in url_lower for indicator in [
            'personal', 'homepage', 'home', 'about', 'profile', 'cv', 'resume'
        ]) or len(host.split('.')) <= 2:  # 简单域名如 yuzheyang.com
            return 'homepage'
    
    # 代码和专业平台
    elif 'github.com' in host:
        return 'github'
    elif 'huggingface.co' in host:
        return 'huggingface'
    elif 'researchgate.net' in host:
        return 'researchgate'
    
    # 社交媒体
    elif 'x.com' in host or 'twitter.com' in host:
        return 'twitter'
    elif 'linkedin.com' in host:
        return 'linkedin'
    
    # 排除明显不相关的网站
    elif any(blocked in host for blocked in [
        'wikipedia', 'news', 'blog', 'forum', 'reddit', 'youtube', 'facebook'
    ]):
        return None
    
    return None

def get_platform_hint(host: str) -> str:
    """获取平台提示"""
    if 'openreview.net' in host:
        return "openreview"
    elif 'scholar.google.' in host:
        return "google_scholar"
    elif 'orcid.org' in host:
        return "orcid"
    elif 'semanticscholar.org' in host:
        return "semantic_scholar"
    elif host.endswith('.edu') or host.endswith('.ac.uk') or host.endswith('.ac.nz'):
        return "university"
    elif 'github.com' in host:
        return "github"
    return "generic"

def process_extracted_profile_info(ext, url: str, author_name: str, profile: AuthorProfile, 
                                 protected_platforms: set, is_homepage: bool = False):
    """处理LLM提取的profile信息"""
    # 处理个人主页URL
    personal_homepage = getattr(ext, 'personal_homepage', '') or getattr(ext, 'homepage_url', '')
    if personal_homepage == url:
        personal_homepage = None  # 当前页面不是个人主页
    
    if is_homepage and not personal_homepage:
        personal_homepage = url
    
    if 'github.io' in url and not profile.homepage_url:
        profile.homepage_url = url
    
    # 处理社交链接
    social_links = getattr(ext, 'social_links', {}) or {}
    
    if is_homepage:
        # 个人网站：强制更新所有社交链接（但需要验证）
        for social_platform, social_url in social_links.items():
            if social_url and social_url != url and validate_social_link_for_author(social_platform, social_url, author_name):
                profile.platforms[social_platform] = social_url
                protected_platforms.add(social_platform)
                print(f"[Protected LLM] {social_platform}: {social_url}")
            elif social_url and not validate_social_link_for_author(social_platform, social_url, author_name):
                print(f"[LLM Rejected] Invalid {social_platform} link: {social_url}")
        
        # 从内容提取额外链接（如果还没有足够的链接）
        if len(social_links) < 3:  # 如果LLM提取的链接不够
            extracted_links = extract_social_links_from_content(fetch_text(url, max_chars=10000))
            for social_platform, social_url in extracted_links.items():
                if social_platform not in profile.platforms:
                    profile.platforms[social_platform] = social_url
                    protected_platforms.add(social_platform)
                    print(f"[Protected HTML] {social_platform}: {social_url}")
    else:
        # 非个人网站：只有在平台未被保护时才更新
        for social_platform, social_url in social_links.items():
            if social_url and social_url != url and social_platform not in protected_platforms:
                update_platform_url(profile, social_platform, social_url, author_name)
    
    # 创建incoming profile并合并
    cleaned_aliases = clean_aliases(getattr(ext, 'aliases', []) or [], author_name)
    
    incoming = AuthorProfile(
        name=getattr(ext, 'name', '') or author_name,
        aliases=cleaned_aliases,
        platforms={}, ids={}, 
        homepage_url=personal_homepage,
        affiliation_current=getattr(ext, 'affiliation_current','') or None,
        emails=list(getattr(ext, 'emails', []) or []),
        interests=list(getattr(ext, 'interests', []) or []),
        selected_publications=list(getattr(ext, 'selected_publications', []) or []),
        confidence=0.4,
        notable_achievements=list(getattr(ext, 'notable_achievements', []) or []),
        social_impact=getattr(ext, 'social_impact', '') or None,
        career_stage=getattr(ext, 'career_stage', '') or None,
        overall_score=0.0
    )
    
    # 合并profiles，但不返回值因为profile是引用传递
    merged = merge_profiles(profile, incoming)
    # 更新profile的属性
    for attr in ['name', 'aliases', 'platforms', 'ids', 'homepage_url', 'affiliation_current', 
                 'emails', 'interests', 'selected_publications', 'notable_achievements', 
                 'social_impact', 'career_stage', 'confidence']:
        setattr(profile, attr, getattr(merged, attr))

def clean_aliases(raw_aliases: List[str], author_name: str) -> List[str]:
    """清理aliases，只保留真正的作者别名"""
    cleaned_aliases = []
    author_words = set(author_name.lower().split())
    
    for alias in raw_aliases:
        if alias and alias != author_name:
            alias_words = set(alias.lower().split())
            # 如果别名与作者名有重叠词汇，可能是真正的别名
            if len(alias_words & author_words) > 0 or len(alias.split()) <= 3:
                cleaned_aliases.append(alias)
    
    return cleaned_aliases[:5]  # 限制别名数量

def merge_profiles(base: AuthorProfile, incoming: AuthorProfile, keep_base_platforms: bool = False) -> AuthorProfile:
    """Merge two author profiles with trust ranking"""
    # Merge platforms and IDs
    for k, v in incoming.platforms.items():
        if not keep_base_platforms:
            base.platforms.setdefault(k, v)
    for k, v in incoming.ids.items():
        if not keep_base_platforms:
            base.ids.setdefault(k, v)
    
    # 合并homepage_url - 优先保留非平台URL的个人网站
    if incoming.homepage_url:
        if not base.homepage_url:
            base.homepage_url = incoming.homepage_url
        elif 'github.io' in incoming.homepage_url and 'github.io' not in base.homepage_url:
            # 个人域名优于github.io
            base.homepage_url = incoming.homepage_url
        elif 'github.io' in base.homepage_url and 'github.io' not in incoming.homepage_url:
            pass
    
    # Merge names and aliases
    if not base.name and incoming.name:
        if not keep_base_platforms:
            base.name = incoming.name
    # Add aliased to base
    for a in incoming.aliases:
        if a and a not in base.aliases:
            base.aliases.append(a)

    # Merge basic fields (prefer non-empty)
    if not base.affiliation_current and incoming.affiliation_current:
        base.affiliation_current = incoming.affiliation_current
    if not base.social_impact and incoming.social_impact:
        base.social_impact = incoming.social_impact
    if not base.career_stage and incoming.career_stage:
        base.career_stage = incoming.career_stage
        
    # Merge list fields with deduplication
    for e in incoming.emails:
        if e and e not in base.emails:
            base.emails.append(e)
    for i in incoming.interests:
        if i and i not in base.interests:
            base.interests.append(i)
    
    # Merge notable achievements
    if hasattr(incoming, 'notable_achievements') and incoming.notable_achievements:
        for achievement in incoming.notable_achievements:
            if achievement and achievement not in base.notable_achievements:
                base.notable_achievements.append(achievement)

    # Merge publications with deduplication
    def key(pub):
        return re.sub(r'\s+', ' ', (pub.get('title','') or '').strip().lower())

    seen = {key(p) for p in base.selected_publications}
    for p in incoming.selected_publications:
        if key(p) not in seen:
            base.selected_publications.append(p)
            seen.add(key(p))

    # Update confidence
    base.confidence = min(1.0, base.confidence + 0.1)
    return base

# ============================ MAIN DISCOVERY FUNCTIONS ============================

def scrape_openreview_profile_html(profile_url: str) -> Dict[str, str]:
    """
    从 OpenReview profile HTML 页面中提取社交链接（补充 API 数据）
    Args:
        profile_url: OpenReview profile URL (e.g., https://openreview.net/profile?id=~xxx)
    
    Returns:
        Dict of discovered links: {'homepage': url, 'google_scholar': url, ...}
    """
    print(f"[OpenReview HTML] Scraping profile page: {profile_url}")
    discovered_links = {}
    try:
        response = requests.get(profile_url, timeout=30, headers=config.UA)
        
        if not response.ok:
            print(f"[OpenReview HTML] ❌ HTTP {response.status_code}")
            return discovered_links
        
        html = response.text
        soup = BeautifulSoup(html, 'html.parser')
        
        # 方法1: 查找所有 <a> 标签
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href'].strip()
            text = a_tag.get_text().strip().lower()
            
            # Homepage (通常标记为 "Homepage", "Website", "Personal Page" 等)
            if any(keyword in text for keyword in ['homepage', 'website', 'personal page', 'home page']):
                if href.startswith('http') and 'homepage' not in discovered_links:
                    discovered_links['homepage'] = href
                    print(f"[OpenReview HTML] 📄 Found homepage: {href}")
            
            # Google Scholar
            if 'scholar.google.com/citations?user=' in href:
                discovered_links['google_scholar'] = href
                print(f"[OpenReview HTML] 🎓 Found Google Scholar: {href}")
            
            # ORCID
            if 'orcid.org/' in href and re.search(r'\d{4}-\d{4}-\d{4}-\d{4}', href):
                discovered_links['orcid'] = href
                print(f"[OpenReview HTML] 🆔 Found ORCID: {href}")
            
            # DBLP
            if ('dblp.org/pid/' in href or 'dblp.org/pers/' in href) and 'dblp' not in discovered_links:
                discovered_links['dblp'] = href
                print(f"[OpenReview HTML] 📚 Found DBLP: {href}")
            
            # LinkedIn
            if 'linkedin.com/in/' in href and 'linkedin' not in discovered_links:
                discovered_links['linkedin'] = href
                print(f"[OpenReview HTML] 💼 Found LinkedIn: {href}")
            
            # GitHub
            if 'github.com/' in href and 'github.io' not in href and 'github' not in discovered_links:
                # 排除 organization 页面
                if not any(org in href for org in ['/orgs/', '/organizations/']):
                    discovered_links['github'] = href
                    print(f"[OpenReview HTML] 💻 Found GitHub: {href}")
        
        # 方法2: 查找特定的 data 属性（如果 OpenReview 使用）
        for elem in soup.find_all(attrs={'data-homepage': True}):
            discovered_links['homepage'] = elem['data-homepage']
        
        for elem in soup.find_all(attrs={'data-gscholar': True}):
            discovered_links['google_scholar'] = elem['data-gscholar']
        
        print(f"[OpenReview HTML] ✅ Discovered {len(discovered_links)} links from HTML")
        
    except Exception as e:
        print(f"[OpenReview HTML] ❌ Error: {str(e)[:100]}")
    
    return discovered_links

def _normalize_name_tokens(name: str) -> List[str]:
    """Normalize a human name into lowercase tokens.

    - Removes parentheses content, e.g. "Jiarui (Gary) Ji" -> "Jiarui  Ji"
    - Unifies common separators: "_ . -" -> space
    - Keeps only Latin letters and CJK characters
    """
    if not name:
        return []
    name = name.lower().strip()
    # remove nickname / alias in parentheses
    name = re.sub(r"\([^)]*\)", " ", name)
    # unify separators
    name = name.replace("_", " ").replace(".", " ").replace("-", " ")
    # keep only letters and CJK characters
    name = re.sub(r"[^a-z\u4e00-\u9fff\s]", " ", name)
    tokens = [t for t in name.split() if t]
    return tokens

def _score_openreview_candidate(author_name: str, candidate: Dict[str, Any]) -> float:
    """Score how well a candidate OpenReview profile matches the author name.

    Uses bidirectional token coverage (query->cand, cand->query) plus small
    bonuses for same surname and same token count.
    """
    query_tokens = _normalize_name_tokens(author_name)
    if not query_tokens:
        return 0.0

    raw = ""
    if isinstance(candidate, dict):
        raw = candidate.get("name") or candidate.get("id") or ""
    raw = str(raw)
    # remove OpenReview username decorations: ~Name_Surname1 -> Name Surname
    raw = re.sub(r"^~", "", raw)
    raw = re.sub(r"\d+$", "", raw)

    cand_tokens = _normalize_name_tokens(raw)
    if not cand_tokens:
        return 0.0

    common = len(set(query_tokens) & set(cand_tokens))
    if common == 0:
        return 0.0

    # bidirectional coverage
    recall_q = common / len(query_tokens)
    recall_c = common / len(cand_tokens)
    score = 0.5 * (recall_q + recall_c)

    # bonus: same surname (last token)
    if query_tokens[-1] == cand_tokens[-1]:
        score += 0.2

    # bonus: same token count (roughly same name structure)
    if len(query_tokens) == len(cand_tokens):
        score += 0.05

    return score


def _get_paper_info_from_s2_or_arxiv(paper_title: str, api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    通过 title 从 Semantic Scholar 或 arXiv 获取论文信息（包括作者列表）
    
    Args:
        paper_title: 论文标题
        api_key: API密钥（未使用，保持兼容性）
    
    Returns:
        Dict包含:
        - title: 论文标题
        - authors: 作者列表 ['John Doe', 'Jane Smith', ...]
        - author_ids: 作者ID列表（S2有，arXiv没有）
        - abstract: 摘要
        - year: 年份
        - venue: 会议/期刊
        - data_source: 'semantic_scholar' 或 'arxiv'
        如果未找到，返回None
    """
    print(f"[Paper Info] Fetching paper info for: {paper_title[:80]}...")
    
    # 先尝试 Semantic Scholar
    try:
        from .semantic_paper_search import SemanticScholarClient
        ss_client = SemanticScholarClient(api_key=config.SEMANTIC_SCHOLAR_API_KEY if config.SEMANTIC_SCHOLAR_API_KEY else None, requests_per_second=1.0, timeout=5.0)
        
        ss_data = ss_client.get_paper_full_details(
            title=paper_title,
            min_match_score=0.75
        )
        
        if ss_data and ss_data.get('authors'):
            print(f"[Paper Info] ✅ Found via Semantic Scholar")
            print(f"  - Authors: {len(ss_data.get('authors', []))}")
            return {
                'title': ss_data.get('title', paper_title),
                'authors': ss_data.get('authors', []),
                'author_ids': ss_data.get('author_ids', []),
                'abstract': ss_data.get('abstract', ''),
                'year': ss_data.get('year'),
                'venue': ss_data.get('venue', ''),
                'data_source': 'semantic_scholar'
            }
    except Exception as e:
        print(f"[Paper Info] ⚠️ Semantic Scholar failed: {e}")
    
    # 回退到 arXiv
    try:
        from .arxiv_fallback import ArxivFallbackClient
        arxiv_client = ArxivFallbackClient(requests_per_second=0.33)
        
        arxiv_data = arxiv_client.search_by_title(title=paper_title)
        
        if arxiv_data and arxiv_data.get('authors'):
            print(f"[Paper Info] ✅ Found via arXiv")
            print(f"  - Authors: {len(arxiv_data.get('authors', []))}")
            return {
                'title': arxiv_data.get('title', paper_title),
                'authors': arxiv_data.get('authors', []),
                'author_ids': [],  # arXiv 没有作者ID
                'abstract': arxiv_data.get('abstract', ''),
                'year': arxiv_data.get('year'),
                'venue': arxiv_data.get('venue', 'arXiv preprint'),
                'data_source': 'arxiv'
            }
    except Exception as e:
        print(f"[Paper Info] ⚠️ arXiv failed: {e}")
    
    print(f"[Paper Info] ❌ Failed to fetch paper info from S2 or arXiv")
    return None


def _normalize_author_name(name: str) -> List[str]:
    """
    标准化作者名字，提取token用于匹配
    
    Returns:
        名字token列表（小写）
    """
    if not name:
        return []
    # 去除多余空格，转换为小写
    name = re.sub(r'\s+', ' ', name.strip().lower())
    # 分割成token
    tokens = name.split()
    # 过滤空token
    return [t for t in tokens if t]


def _normalize_author_for_match(name: str) -> List[str]:
    """Normalize author name for fuzzy matching (lowercase, strip punctuation)."""

    name = name or ""
    cleaned = re.sub(r"[^A-Za-z0-9\s]", " ", name.lower())
    tokens = [t for t in re.split(r"\s+", cleaned) if t]
    expanded_tokens = []
    for token in tokens:
        expanded_tokens.append(token)
        if len(token) > 1:
            expanded_tokens.append(token[0])
    return expanded_tokens


def _author_similarity(a: str, b: str) -> float:
    """Combine Jaccard and Levenshtein-like similarity for author names."""

    tokens_a = set(_normalize_author_for_match(a))
    tokens_b = set(_normalize_author_for_match(b))

    if not tokens_a or not tokens_b:
        return 0.0

    jaccard = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    levenshtein_like = difflib.SequenceMatcher(None, a.lower(), b.lower()).ratio()
    return (jaccard + levenshtein_like) / 2


def _align_author_ids(expected_authors: List[str], candidate: SemanticScholarPaper) -> List[Optional[str]]:
    """Align Semantic Scholar authorIds to the expected author order."""

    aligned: List[Optional[str]] = []
    used_idx = set()

    for expected in expected_authors:
        best_idx = None
        best_score = 0.0
        for idx, author in enumerate(candidate.authors):
            if idx in used_idx:
                continue
            score = _author_similarity(expected, author.name)
            if score > best_score:
                best_idx = idx
                best_score = score
        if best_idx is not None:
            used_idx.add(best_idx)
            aligned.append(candidate.authors[best_idx].authorId)
        else:
            aligned.append(None)
    return aligned


def _score_candidate_authors(input_authors: List[str], candidate_authors: List[str]) -> float:
    if not input_authors or not candidate_authors:
        return 0.0
    scores = []
    for expected in input_authors:
        best = max((_author_similarity(expected, cand) for cand in candidate_authors), default=0.0)
        scores.append(best)
    return sum(scores) / len(scores)


def _select_best_semantic_candidate(
    candidates: List[SemanticScholarPaper],
    input_authors: List[str],
    target_year: Optional[int] = None,
) -> Optional[SemanticScholarPaper]:
    best: Optional[SemanticScholarPaper] = None
    best_score = 0.0
    for cand in candidates:
        if target_year and cand.year:
            if abs(int(cand.year) - int(target_year)) > 1:
                continue
        cand_score = _score_candidate_authors(input_authors, [a.name for a in cand.authors])
        if cand_score > best_score:
            best_score = cand_score
            best = cand
    return best


def search_paper_candidates_with_fallback(
    paper_title: str,
    authors: List[str],
    year: Optional[int] = None,
    google_search_fn=None,
    semantic_client: Optional[SemanticScholarSearchClient] = None,
) -> List[Dict[str, Any]]:
    """Search paper candidates via Google Scholar, then Semantic Scholar fallback."""

    start_ts = time.time()
    authors = authors or []
    google_candidates: List[Dict[str, Any]] = []
    try:
        if google_search_fn:
            google_candidates = google_search_fn(paper_title, authors, year)
    except Exception as e:
        print(f"[Semantic Fallback] ⚠️ Google Scholar search error: {e}")

    if google_candidates:
        return google_candidates

    if not config.USE_SEMANTIC_SCHOLAR_FALLBACK:
        print("[Semantic Fallback] 🚫 Disabled via config, skipping Semantic Scholar")
        return []

    semantic_client = semantic_client or SemanticScholarSearchClient(
        api_key=config.SEMANTIC_SCHOLAR_API_KEY if config.SEMANTIC_SCHOLAR_API_KEY else None,
        requests_per_second=1.0,
        timeout=8.0,
    )

    query = f"{paper_title} {' '.join(authors)}".strip()
    try:
        results = semantic_client.search_papers(query=query)
    except Exception as e:
        print(f"[Semantic Fallback] ❌ Semantic Scholar search failed: {e}")
        return []

    best = _select_best_semantic_candidate(results, authors, target_year=year)
    if not best:
        print("[Semantic Fallback] ⚠️ No Semantic Scholar candidates matched author list")
        return []

    aligned_ids = _align_author_ids(authors, best)
    elapsed = (time.time() - start_ts) * 1000
    print(
        f"[Semantic Fallback] ✅ Selected Semantic Scholar candidate (source=semantic_scholar, {elapsed:.1f} ms)"
    )

    return [
        {
            "paperId": best.paperId,
            "title": best.title,
            "authors": [a.name for a in best.authors],
            "semantic_scholar_author_ids": aligned_ids,
            "year": best.year,
            "venue": best.venue,
            "source": "semantic_scholar",
        }
    ]

def _match_author_lists(s2_authors: List[str], or_authors: List[str], first_author: str = None) -> Optional[int]:
    """
    匹配两个作者列表，找到目标作者在 OpenReview 列表中的位置
    
    Args:
        s2_authors: Semantic Scholar/arXiv 的作者列表
        or_authors: OpenReview 的作者列表
        first_author: 目标作者名字（可选，用于精确匹配）
    
    Returns:
        目标作者在 or_authors 中的索引，如果未找到返回None
    """
    if not s2_authors or not or_authors:
        return None
    
    # 策略1: 如果提供了 first_author，在 or_authors 中查找精确 token 匹配
    if first_author:
        first_author_tokens = set(_normalize_author_name(first_author))
        for idx, or_author in enumerate(or_authors):
            or_author_tokens = set(_normalize_author_name(or_author))
            # 检查是否有足够的token匹配（至少匹配姓氏）
            if or_author_tokens and first_author_tokens:
                common = first_author_tokens & or_author_tokens
                if len(common) >= min(2, len(first_author_tokens), len(or_author_tokens)):
                    return idx
        
        # 策略1.5: 使用模糊匹配（相似度 >= 0.6）来找到目标作者
        best_idx = None
        best_score = 0.0
        for idx, or_author in enumerate(or_authors):
            similarity = _author_similarity(first_author, or_author)
            if similarity > best_score and similarity >= 0.6:
                best_score = similarity
                best_idx = idx
        
        if best_idx is not None:
            return best_idx
    
    # 策略2: 如果没有提供 first_author，匹配 s2_authors[0] (第一作者) 与 or_authors
    # 注意：只有在没有提供 first_author 时才使用此策略
    if not first_author and s2_authors:
        s2_first = s2_authors[0]
        s2_first_tokens = set(_normalize_author_name(s2_first))
        
        for idx, or_author in enumerate(or_authors):
            or_author_tokens = set(_normalize_author_name(or_author))
            if or_author_tokens and s2_first_tokens:
                common = s2_first_tokens & or_author_tokens
                # 至少匹配2个token，或者匹配姓氏（最后一个token）
                if len(common) >= min(2, len(s2_first_tokens), len(or_author_tokens)):
                    return idx
    
    # 策略3: 如果都没匹配上，返回 None（不再默认返回0，避免错误关联到一作）
    # 调用者应该使用 fallback 机制（如 name-based 搜索）
    return None


def _get_paperhash(paper_title: str, authors: List[str]) -> str:
    """
    使用 openreview.tools.get_paperhash 生成 paperhash。
    注意：必须安装 openreview 库才能使用此功能。
    安装命令：pip install openreview-py
    
    Args:
        paper_title: 论文标题
        authors: 作者列表
    
    Returns:
        paperhash 字符串
    Raises:
        ImportError: 如果未安装 openreview 库
        ValueError: 如果作者列表为空
    """
    if not authors or len(authors) == 0:
        raise ValueError("authors list is required to build paperhash")
    
    try:
        from openreview import tools
    except ImportError:
        raise ImportError(
            "openreview 库未安装，无法生成 paperhash。\n"
            "请运行以下命令安装：pip install openreview-py"
        )
    first_author = authors[0] if authors else ""
    paperhash = tools.get_paperhash(first_author=first_author, title=paper_title)
    return paperhash


def _search_openreview_notes_by_title(paper_title: str, authors: List[str] = None, api_key: str = None, max_results: int = 10) -> Optional[Dict[str, Any]]:
    """
    通过论文标题和作者列表在 OpenReview 中搜索 notes，提取 authors 和 authorids
    
    使用 paperhash 方法进行精确匹配（需要提供 authors）
    
    Args:
        paper_title: 论文标题
        authors: 作者列表（必需，用于生成 paperhash）
        api_key: API密钥（未使用，保持兼容性）
        max_results: 最多返回结果数
    
    Returns:
        Dict包含:
        - matched_note: 匹配的note信息
        - authors: 作者列表
        - authorids: 作者ID列表（有序对应）
        如果未找到或 authors 为空，返回None
    """
    if not authors or len(authors) == 0:
        return None
    
    def get_value(data, default=''):
        """从 OpenReview note content 中提取值"""
        if isinstance(data, dict):
            return data.get('value', default)
        return data if data is not None else default
    
    try:
        # 生成 paperhash（优先使用 openreview 库，否则使用本地实现）
        paperhash = _get_paperhash(paper_title, authors)
        
        if not paperhash:
            return None
        
        time.sleep(0.5)  # Rate limiting
        
        # 使用 paperhash 通过 REST API 查询
        notes_url = "https://api2.openreview.net/notes"
        notes_params = {
            'paperhash': paperhash,
            'limit': max_results,
            'offset': 0
        }
        
        response = requests.get(
            notes_url,
            params=notes_params,
            timeout=30,
            headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
        )
        
        if response.status_code == 200:
            notes_data = response.json()
            notes = notes_data.get('notes', [])
            
            if notes and len(notes) > 0:
                best_note = notes[0]  # paperhash 应该只返回一个精确匹配
                note_content = best_note.get('content', {})
                note_authors = get_value(note_content.get('authors'), [])
                note_authorids = get_value(note_content.get('authorids'), [])
                
                # 确保 authors 和 authorids 是列表
                if not isinstance(note_authors, list):
                    note_authors = []
                if not isinstance(note_authorids, list):
                    note_authorids = []
                
                return {
                    'matched_note': best_note,
                    'authors': note_authors,
                    'authorids': note_authorids,
                    'match_score': 1.0,
                    'method': 'paperhash'
                }
            else:
                pass  # No notes found
        else:
            pass  # HTTP error
            
    except Exception as e:
        pass  # Silently fail
    
    return None


def search_openreview_profile_by_title(
    paper_title: str,
    first_author: str = None,
    api_key: str = None,
    paper_info: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    通过论文标题获取 OpenReview profile
    
    流程：
    1. 检查缓存（优先使用 OpenReviewClient 的缓存）
    2. 从 Semantic Scholar/arXiv 获取论文信息（包括作者）
    3. 使用 title 和 authors 生成 paperhash，在 OpenReview 中查询获取 authorids
    4. 验证作者列表匹配，使用 authorid 获取 profile
    5. 更新 paper_info 中的 author_ids 列表（将 OpenReview 获取到的 author_ids 填入）
    6. 如果失败，返回 None（由调用者处理 fallback）
    
    Args:
        paper_title: 论文标题
        first_author: 第一作者名字（可选，用于精确匹配）
        api_key: API密钥
        paper_info: 可选的论文信息（如果已获取，避免重复请求）
    
    Returns:
        OpenReview profile 字典，如果失败返回 None
        profile 字典中会包含 'updated_paper_info' 字段，包含更新后的 author_ids 列表
    """
    # ====== CHECK CACHE FIRST ======
    or_authors = None
    or_authorids = None
    s2_authors = None
    cache_hit = False
    
    # ====== DEDUP: 记录已搜索过的作者，避免重复调用 search_openreview_profile ======
    _searched_authors_cache = {}  # {author_name_lower: profile_or_None}
    def _search_profile_with_dedup(author_name: str) -> Optional[Dict[str, Any]]:
        """带去重的 search_openreview_profile 调用"""
        if not author_name:
            return None
        key = author_name.lower().strip()
        if key in _searched_authors_cache:
            # 已搜索过，直接返回缓存结果
            return _searched_authors_cache[key]
        # 未搜索过，执行搜索并缓存
        result = search_openreview_profile(author_name, api_key=api_key)
        _searched_authors_cache[key] = result
        return result
    try:
        from .openreview_client import get_openreview_client
        or_client = get_openreview_client()
        cached_paper = or_client.cache.get_paper(paper_title)
        
        if cached_paper and cached_paper.authorids:
            # Cache hit - use cached data directly, skip API calls
            or_authors = cached_paper.authors or []  # 确保不为 None
            or_authorids = cached_paper.authorids or []  # 确保不为 None
            cache_hit = True
            
            if not paper_info:
                paper_info = {
                    'title': paper_title,
                    'authors': or_authors,
                    'author_ids': list(or_authorids),
                    'data_source': 'openreview_cache'
                }
            s2_authors = paper_info.get('authors', []) or or_authors
    except ImportError:
        pass
    except Exception:
        pass  # Silently continue with API if cache fails
    # ====== END CACHE CHECK ======
    
    # 如果调用方已经提供了包含作者列表的 paper_info，则复用，避免重复调用 S2/arXiv
    if paper_info and paper_info.get('authors'):
        s2_authors = paper_info.get('authors', [])
    # Only do API calls if cache miss
    if not cache_hit and not (paper_info and paper_info.get('authors')):
        fallback_candidates = search_paper_candidates_with_fallback(
            paper_title=paper_title,
            authors=[first_author] if first_author else [],
            year=None,
        )

        if fallback_candidates:
            candidate = fallback_candidates[0]
            paper_info = {
                'title': candidate.get('title', paper_title),
                'authors': candidate.get('authors', []),
                'author_ids': candidate.get('semantic_scholar_author_ids', []),
                'year': candidate.get('year'),
                'venue': candidate.get('venue', ''),
                'data_source': candidate.get('source', 'semantic_scholar')
            }
        # Step 1.1: 如果仍然为空，再尝试 Semantic Scholar / arXiv 兜底
        if not paper_info:
            paper_info = _get_paper_info_from_s2_or_arxiv(paper_title, api_key=api_key)
            if not paper_info:
                print(f"[OpenReview By Title] ❌ Failed to get paper info")
                return None  
        s2_authors = paper_info.get('authors', [])
        if not s2_authors:
            print(f"[OpenReview By Title] ❌ No authors found")
            return None
        
        # Step 2: 使用 paperhash 在 OpenReview 中查询获取 authorids
        or_note_data = _search_openreview_notes_by_title(paper_title, authors=s2_authors, api_key=api_key)
        if not or_note_data:
            print(f"[OpenReview By Title] ❌ Failed to find note in OpenReview")
            return None
        
        or_authors = or_note_data.get('authors', [])
        or_authorids = or_note_data.get('authorids', [])
        
        if not or_authors or not or_authorids:
            print(f"[OpenReview By Title] ❌ No authors/authorids in OpenReview note")
            return None
    
    # 确保 authors 和 authorids 长度一致，且不为 None
    if not or_authors or not or_authorids:
        print(f"[OpenReview By Title]  or_authors or or_authorids is None/empty")
        return None
    min_len = min(len(or_authors), len(or_authorids))
    or_authors = or_authors[:min_len]
    or_authorids = or_authorids[:min_len]
    
    # ✅ Step 2.5: 更新 paper_info 中的 author_ids 列表
    # 将 OpenReview 获取到的 author_ids 映射回 s2_authors 列表
    paper_author_ids = paper_info.get('author_ids', [])
    # 确保 author_ids 列表长度与 authors 列表一致
    while len(paper_author_ids) < len(s2_authors):
        paper_author_ids.append(None)
    
    # 匹配 OpenReview 的 authors 和 s2_authors，更新对应的 author_ids
    for or_idx, or_author in enumerate(or_authors):
        if or_idx >= len(or_authorids):
            continue
        or_authorid = or_authorids[or_idx]
        
        # 在 s2_authors 中查找匹配的作者
        for s2_idx, s2_author in enumerate(s2_authors):
            # 简单的名称匹配（可以改进为更智能的匹配）
            if s2_author and or_author and s2_author.lower().strip() == or_author.lower().strip():
                # 检查是否是有效的 OpenReview ID（以 ~ 开头）
                # 处理数字类型：如果是数字类型，转换为字符串后检查
                if isinstance(or_authorid, (int, float)):
                    or_authorid_str = str(or_authorid)
                elif isinstance(or_authorid, str):
                    or_authorid_str = or_authorid
                else:
                    or_authorid_str = None
                
                is_valid_id = (or_authorid_str is not None and 
                              isinstance(or_authorid_str, str) and 
                              or_authorid_str.strip().startswith('~'))
                current_id = paper_author_ids[s2_idx] if s2_idx < len(paper_author_ids) else None
                
                if is_valid_id:
                    # 如果 OpenReview 返回的是有效 ID，更新它
                    paper_author_ids[s2_idx] = or_authorid
                else:
                    # 如果 OpenReview 返回的是无效 ID（如 DBLP 链接、纯数字等），尝试通过 name 获取 ID
                    if not current_id or not isinstance(current_id, str) or not current_id.strip().startswith('~'):
                        # 通过 name 搜索获取 ID
                        name_profile = _search_profile_with_dedup(s2_author)
                        if name_profile:
                            profile_id = name_profile.get('openreview_id', '')
                            if profile_id and isinstance(profile_id, str) and profile_id.strip().startswith('~'):
                                if s2_idx < len(paper_author_ids):
                                    paper_author_ids[s2_idx] = profile_id
                            else:
                                if s2_idx < len(paper_author_ids):
                                    paper_author_ids[s2_idx] = None
                        else:
                            if s2_idx < len(paper_author_ids):
                                paper_author_ids[s2_idx] = None
                break
    
    # 更新 paper_info 中的 author_ids
    paper_info['author_ids'] = paper_author_ids
    
    # Step 3: 验证作者列表是否匹配
    target_idx = _match_author_lists(s2_authors, or_authors, first_author=first_author)
    if target_idx is None or target_idx >= len(or_authorids):
        print(f"[OpenReview By Title] ❌ Failed to match author")
        return None
    
    target_authorid = or_authorids[target_idx]
    target_author_name = or_authors[target_idx] if target_idx < len(or_authors) else None
    
    # Step 3.5: 找到目标作者在 s2_authors 中的位置，保存 author_position 和 target_author_index
    s2_target_idx = None
    if target_author_name:
        # 在 s2_authors 中查找匹配的作者
        for s2_idx, s2_author in enumerate(s2_authors):
            if s2_author and target_author_name and s2_author.lower().strip() == target_author_name.lower().strip():
                s2_target_idx = s2_idx
                break
        
        # 如果通过精确匹配没找到，尝试使用 first_author 参数
        if s2_target_idx is None and first_author:
            for s2_idx, s2_author in enumerate(s2_authors):
                if s2_author and first_author:
                    # 使用简单的包含匹配
                    if first_author.lower().strip() in s2_author.lower().strip() or s2_author.lower().strip() in first_author.lower().strip():
                        s2_target_idx = s2_idx
                        break
        
        # 如果还是没找到，使用 target_idx（假设 OpenReview 和 S2 的作者顺序一致）
        if s2_target_idx is None and target_idx < len(s2_authors):
            s2_target_idx = target_idx
    
    # 保存 author_position (1-based) 和 target_author_index (0-based) 到 paper_info
    if s2_target_idx is not None:
        paper_info['target_author_index'] = s2_target_idx
        paper_info['author_position'] = s2_target_idx + 1  # 1-based position
    else:
        # 如果找不到，至少保存 target_idx（来自 OpenReview）
        paper_info['target_author_index'] = target_idx
        paper_info['author_position'] = target_idx + 1
    
    # Step 4: 检查 authorid 是否是有效的 OpenReview ID（以 ~ 开头）
    def is_valid_openreview_id(authorid: str) -> bool:
        """检查是否是有效的 OpenReview profile ID"""
        if not authorid or not isinstance(authorid, str):
            return False
        return authorid.strip().startswith('~')
    
    # Step 5: 通过 authorid 获取 profile
    from .direct_homepage_evaluation import _fetch_openreview_by_id
    
    # 如果匹配到的 authorid 无效，先检查 paper_author_ids 中是否已经有通过 name 搜索获取的有效 ID
    if not is_valid_openreview_id(target_authorid):
        # 使用 Step 3.5 中已经计算好的 s2_target_idx（从 paper_info 中获取）
        s2_target_idx = paper_info.get('target_author_index')
        
        # 检查 paper_author_ids 中是否已经有有效的 ID
        if s2_target_idx is not None and s2_target_idx < len(paper_author_ids):
            filled_id = paper_author_ids[s2_target_idx]
            if is_valid_openreview_id(filled_id):
                target_authorid = filled_id  # 使用已填充的有效 ID
            else:
                # 使用 name 搜索作为最后的 fallback
                if target_author_name:
                    profile = _search_profile_with_dedup(target_author_name)
                    if profile:
                        profile_id = profile.get('openreview_id', '')
                        if is_valid_openreview_id(profile_id):
                            paper_author_ids[s2_target_idx] = profile_id
                        paper_info['author_ids'] = paper_author_ids
                        profile['updated_paper_info'] = paper_info
                        return profile
                    else:
                        return None
                else:
                    return None
        else:
            # 使用 name 搜索作为 fallback
            if target_author_name:
                profile = _search_profile_with_dedup(target_author_name)
                if profile:
                    paper_info['author_ids'] = paper_author_ids
                    profile['updated_paper_info'] = paper_info
                    return profile
                else:
                    return None
            else:
                return None
    
    # 使用有效的 OpenReview ID 获取 profile
    profile = _fetch_openreview_by_id(target_authorid, api_key=api_key)
    
    if profile:
        # 验证返回的 profile 是否包含有效的 OpenReview ID
        profile_id = profile.get('openreview_id', '') or target_authorid
        if is_valid_openreview_id(profile_id):
            # 确保 profile 中包含 openreview_id 字段
            if 'openreview_id' not in profile:
                profile['openreview_id'] = target_authorid
            # 将更新后的 paper_info 附加到 profile 中
            profile['updated_paper_info'] = paper_info
            return profile
        else:
            # 如果通过 ID 获取的 profile 没有有效 ID，尝试通过名称搜索作为备选
            if target_author_name:
                name_profile = _search_profile_with_dedup(target_author_name)
                if name_profile:
                    name_profile_id = name_profile.get('openreview_id', '')
                    if is_valid_openreview_id(name_profile_id):
                        # 更新 paper_info 中对应作者的 author_id
                        for s2_idx, s2_author in enumerate(s2_authors):
                            if s2_author and target_author_name and s2_author.lower().strip() == target_author_name.lower().strip():
                                if s2_idx < len(paper_author_ids):
                                    paper_author_ids[s2_idx] = name_profile_id
                                break
                        name_profile['updated_paper_info'] = paper_info
                        return name_profile
            return profile  # 即使没有有效 ID，也返回已获取的 profile
    
    # 如果通过 ID 获取失败，尝试通过作者名称搜索作为备选
    if target_author_name:
        profile = _search_profile_with_dedup(target_author_name)
        if profile:
            profile_id = profile.get('openreview_id', '')
            if is_valid_openreview_id(profile_id):
                # 更新 paper_info 中对应作者的 author_id
                for s2_idx, s2_author in enumerate(s2_authors):
                    if s2_author and target_author_name and s2_author.lower().strip() == target_author_name.lower().strip():
                        if s2_idx < len(paper_author_ids):
                            paper_author_ids[s2_idx] = profile_id
                        break
            profile['updated_paper_info'] = paper_info
            return profile
    return None


# =============== 线程安全锁机制 (用于 OpenReview profile 查询去重) ===============
_openreview_search_locks = {}  # {author_name_lower: threading.Lock}
_openreview_locks_lock = threading.Lock()  # 保护锁字典的锁

def _get_openreview_author_lock(author_name: str) -> threading.Lock:
    """获取指定作者的专用锁（线程安全）"""
    key = author_name.lower().strip()
    with _openreview_locks_lock:
        if key not in _openreview_search_locks:
            _openreview_search_locks[key] = threading.Lock()
        return _openreview_search_locks[key]


def search_openreview_profile(author_name: str, api_key: str = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Search for OpenReview profile - 线程安全包装器
    确保同一作者同时只有一个线程执行 API 查询，避免重复日志和网络请求
    """
    from .openreview_client import get_openreview_client
    
    # =============== CACHE CHECK (第一次，无锁快速检查) ===============
    or_client = get_openreview_client()
    cached_profile = or_client.cache.get_profile(author_name)
    if cached_profile:
        return cached_profile
    
    # =============== 获取作者专用锁 ===============
    author_lock = _get_openreview_author_lock(author_name)
    with author_lock:
        # =============== CACHE CHECK (第二次，锁内双重检查) ===============
        cached_profile = or_client.cache.get_profile(author_name)
        if cached_profile:
            return cached_profile
        
        # 执行实际查询（在锁内，确保同一作者只查询一次）
        return _search_openreview_profile_impl(author_name, api_key, max_retries, or_client)


def _search_openreview_profile_impl(author_name: str, api_key: str, max_retries: int, or_client) -> Optional[Dict[str, Any]]:
    """
    Search for OpenReview profile and extract comprehensive author information
    使用新的OpenReview爬虫逻辑，一次性获取完整信息
    
    🆕 改进：API + HTML 双重获取，最大化链接发现
    🆕 添加重试机制，提高网络请求的健壮性
    
    Args:
        author_name: 作者姓名
        api_key: API密钥（未使用，保持兼容性）
        max_retries: 最大重试次数（默认3次）
        or_client: OpenReview 客户端实例
    
    Returns:
        Dict with structured author info including:
        - openreview_url: Profile URL
        - names: List of author names
        - personal_links: Dict of personal links (homepage, gscholar, orcid, dblp, linkedin)
        - education: List of education history
        - relations: List of advisors/relations
        - publications: List of publications
        - coauthors: Dict of coauthors
        None if no OpenReview profile exists or all retries failed
    """
    import random
    
    # 重试逻辑：对网络错误进行重试
    for attempt in range(max_retries):
        try:
            # Step 1: Search OpenReview API
            api_search_url = "https://api2.openreview.net/profiles/search"
            params = {
                'term': author_name,
                'es': 'true'
            }
            
            time.sleep(1.0 + random.uniform(0, 0.5))  # Rate limiting: 1-1.5s
            
            try:
                response = requests.get(
                    api_search_url, 
                    params=params,
                    timeout=30,
                    headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
                )
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout, 
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)  # 指数退避：2s, 4s, 6s
                    print(f"[OpenReview] ⚠️ Network error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[OpenReview] ❌ Network error after {max_retries} attempts: {e}")
                    return None
            
            if response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            if response.status_code != 200:
                if attempt < max_retries - 1 and response.status_code >= 500:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                return None
            
            data = response.json()
            profiles = data.get('profiles', [])
            
            if not profiles:
                return None

            best_profile = None
            best_score = 0.0
            for cand in profiles:
                score = _score_openreview_candidate(author_name, cand)
                if score > best_score:
                    best_score = score
                    best_profile = cand

            # Minimum acceptable similarity score (configurable via backend.config)
            min_score = getattr(config, "OPENREVIEW_NAME_MIN_SCORE", 0.7)
            if not best_profile or best_score < min_score:
                print(f"[OpenReview] No profile passes name similarity threshold for {author_name} (best_score={best_score:.2f})")
                return None

            profile_id = best_profile.get('id', '')
            if not profile_id:
                return None
            
            print(f"[OpenReview ByName] Found: {author_name} → {profile_id}")
            openreview_url = f"https://openreview.net/profile?id={profile_id}"
            
            # Step 2: Fetch detailed profile
            profile_url = f"https://api2.openreview.net/profiles"
            profile_params = {'id': profile_id}
            
            time.sleep(1.0 + random.uniform(0, 0.5))  # Rate limiting
            
            try:
                profile_response = requests.get(
                    profile_url,
                    params=profile_params,
                    timeout=30,
                    headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
                )
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.Timeout, 
                    requests.exceptions.RequestException) as e:
                if attempt < max_retries - 1:
                    wait_time = (attempt + 1) * 2 + random.uniform(0, 1)
                    print(f"[OpenReview] ⚠️ Network error fetching profile (attempt {attempt + 1}/{max_retries}): {type(e).__name__}, retrying in {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                else:
                    print(f"[OpenReview] ❌ Network error fetching profile after {max_retries} attempts: {e}")
                    return None
            
            if profile_response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            
            if profile_response.status_code != 200:
                if attempt < max_retries - 1 and profile_response.status_code >= 500:
                    wait_time = (attempt + 1) * 2
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            
            try:
                profile_data = profile_response.json()
            except Exception as e:
                print(f"[OpenReview] ❌ Failed to parse JSON response for {author_name} (profile_id={profile_id}): {type(e).__name__}: {e}")
                return None
            
            if 'profiles' not in profile_data or not profile_data['profiles']:
                print(f"[OpenReview] ❌ Empty profile data for {author_name} (profile_id={profile_id}): 'profiles' field missing or empty in response")
                if 'profiles' in profile_data:
                    print(f"[OpenReview] Response structure: {list(profile_data.keys())}")
                return None
            
            full_profile = profile_data['profiles'][0]
            content = full_profile.get('content', {})
            
            # Step 3: Extract structured info
            names_list = []
            for name in content.get('names', []):
                names_list.append({
                    'first': name.get('first', ''),
                    'middle': name.get('middle', ''),
                    'last': name.get('last', ''),
                    'username': name.get('username', ''),
                    'preferred': name.get('preferred', False)
                })
            
            personal_links = {}
            if content.get('homepage'):
                personal_links['homepage'] = content['homepage']
            if content.get('gscholar'):
                personal_links['google_scholar'] = content['gscholar']
            if content.get('orcid'):
                personal_links['orcid'] = content['orcid']
            if content.get('dblp'):
                personal_links['dblp'] = content['dblp']
            if content.get('linkedin'):
                personal_links['linkedin'] = content['linkedin']
            
            education_list = []
            # OpenReview API 可能返回 'education' 或 'history' 字段
            # 检查 content 和 profile 顶层
            education_data = content.get('education') or content.get('history') or full_profile.get('education') or full_profile.get('history') or []
            if not isinstance(education_data, list):
                education_data = []
            for item in education_data:
                # 处理 institution 可能是对象或字符串的情况
                institution_name = ''
                if isinstance(item.get('institution'), dict):
                    institution_name = item.get('institution', {}).get('name', '')
                elif isinstance(item.get('institution'), str):
                    institution_name = item.get('institution', '')
                
                # 处理 start 和 end 可能是数字或字符串的情况
                start = item.get('start', '')
                end = item.get('end', '')
                if start is not None:
                    start = str(start) if start else ''
                if end is not None:
                    end = str(end) if end else ''
                
                education_list.append({
                    'position': item.get('position', ''),
                    'institution': institution_name,
                    'start': start,
                    'end': end
                })
            
            relations_list = []
            for relation in content.get('relations', []):
                relations_list.append({
                    'name': relation.get('name', ''),
                    'email': relation.get('email', ''),
                    'relation': relation.get('relation', ''),
                    'start': relation.get('start', ''),
                    'end': relation.get('end', '')
                })
            
            all_author_ids = [profile_id]
            for name in names_list:
                if name['username'] and name['username'] not in all_author_ids:
                    all_author_ids.append(name['username'])
            
            # Step 4: Fetch publications (允许部分失败，不中断整个流程)
            publications_list = []
            seen_ids = set()
            
            for author_id in all_author_ids:
                try:
                    time.sleep(0.5)  # 避免429
                    notes_url = f"https://api2.openreview.net/notes"
                    notes_params = {
                        'content.authorids': author_id,
                        'details': 'replyCount,invitation,original',
                        'limit': 1000,
                        'offset': 0
                    }
                    
                    try:
                        notes_response = requests.get(
                            notes_url,
                            params=notes_params,
                            timeout=30,
                            headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
                        )
                    except (requests.exceptions.ConnectionError, 
                            requests.exceptions.Timeout, 
                            requests.exceptions.RequestException) as e:
                        # 获取论文失败不影响整体流程，继续处理
                        print(f"[OpenReview] ⚠️ Failed to fetch publications for {author_id}: {type(e).__name__}")
                        continue
                    
                    if notes_response.status_code == 200:
                        notes_data = notes_response.json()
                        notes = notes_data.get('notes', [])
                        
                        for note in notes:
                            note_id = note.get('id')
                            if note_id and note_id not in seen_ids:
                                seen_ids.add(note_id)
                                
                                # 提取论文信息
                                note_content = note.get('content', {})
                                
                                def get_value(data, default=''):
                                    if isinstance(data, dict):
                                        return data.get('value', default)
                                    return data if data is not None else default
                                
                                pub_info = {
                                    'id': note_id,
                                    'title': get_value(note_content.get('title'), ''),
                                    'authors': get_value(note_content.get('authors'), []),
                                    'authorids': get_value(note_content.get('authorids'), []),
                                    'abstract': get_value(note_content.get('abstract'), ''),
                                    'keywords': get_value(note_content.get('keywords'), []),
                                    'venue': get_value(note_content.get('venue'), ''),
                                    'year': note.get('pdate', 0) // 1000 // 86400 // 365 + 1970 if note.get('pdate') else None,  # 粗略转换
                                    'pdf': get_value(note_content.get('pdf'), ''),
                                    'url': f"https://openreview.net/forum?id={note_id}"
                                }
                                
                                publications_list.append(pub_info)
                except Exception as e:
                    # 单个author_id失败不影响整体
                    continue
            
            # Step 5: Extract coauthors
            coauthors_dict = {}
            
            for pub in publications_list:
                authors = pub.get('authors', [])
                authorids = pub.get('authorids', [])
                
                for i, author in enumerate(authors):
                    if not author:
                        continue
                    
                    # 检查是否是当前作者
                    is_current_author = False
                    if i < len(authorids) and authorids[i] in all_author_ids:
                        is_current_author = True
                    
                    if not is_current_author:
                        coauthors_dict[author] = coauthors_dict.get(author, 0) + 1
            
            # Sort coauthors by collaboration count
            coauthors_sorted = dict(sorted(coauthors_dict.items(), key=lambda x: x[1], reverse=True))
            
            # Step 6: Scrape HTML for additional links (允许失败，不影响主要数据)
            try:
                html_links = scrape_openreview_profile_html(openreview_url)
                for platform, url in html_links.items():
                    if not personal_links.get(platform):
                        personal_links[platform] = url
            except Exception as e:
                print(f"[OpenReview] ⚠️ Failed to scrape HTML links: {type(e).__name__}")

            # 成功获取数据，返回结果
            result = {
                'openreview_url': openreview_url,
                'openreview_id': profile_id,  # 添加有效的 OpenReview ID
                'names': names_list,
                'personal_links': personal_links,  # homepage is inside personal_links['homepage']
                'education': education_list,
                'relations': relations_list,
                'publications': publications_list,
                'coauthors': coauthors_sorted,
                'raw_profile': full_profile  # 保留原始数据以备用
            }
            # =============== CACHE SET ===============
            or_client.cache.set_profile(author_name, result)
            return result
            
        except Exception as e:
            # 非网络错误（如JSON解析错误等），不需要重试
            if attempt < max_retries - 1:
                # 某些异常可能是暂时的，重试一次
                wait_time = (attempt + 1) * 1
                print(f"[OpenReview] ⚠️ Unexpected error (attempt {attempt + 1}/{max_retries}): {type(e).__name__}, retrying in {wait_time:.1f}s...")
                time.sleep(wait_time)
                continue
            else:
                print(f"\n❌ OpenReview extraction failed after {max_retries} attempts: {e}")
                print(f"{'='*80}\n")
                return None
    
    # 所有重试都失败
    print(f"[OpenReview] ❌ All {max_retries} attempts failed")
    return None

def merge_homepage_data_to_profile(homepage_data: Dict, profile: AuthorProfile, priority: str = 'primary'):
    """
    合并 Homepage 数据到 AuthorProfile
    
    Args:
        homepage_data: 从 homepage_parser 解析的数据
        profile: 目标 AuthorProfile
        priority: 'primary' | 'secondary' | 'filling'
    """
    if priority == 'primary':
        # 主要来源：大胆填充/覆盖
        # Research Interests
        if homepage_data.get('research_interests'):
            interests = homepage_data['research_interests']
            if isinstance(interests, list):
                # 合并到现有兴趣（去重）
                existing = set(i.lower() for i in profile.interests)
                for interest in interests:
                    if interest.lower() not in existing:
                        profile.interests.append(interest)
                print(f"    ✓ Added research interests: {len(interests)} items")
        
        # Awards
        if homepage_data.get('awards'):
            awards = homepage_data['awards']
            if isinstance(awards, list):
                for award in awards:
                    if isinstance(award, dict):
                        award_name = award.get('name', '')
                        if award_name:
                            profile.notable_achievements.append(award_name)
                print(f"    ✓ Added awards: {len(awards)} items")
        
        # Experience
        if homepage_data.get('experience'):
            # 暂存到 profile（后续可以添加到 AuthorProfile 的 experience 字段）
            print(f"    ✓ Extracted experience data")
        
        # Teaching
        if homepage_data.get('teaching'):
            # 暂存到 profile
            print(f"    ✓ Extracted teaching data")
        
        # Contact
        if homepage_data.get('contact'):
            contacts = homepage_data['contact']
            if isinstance(contacts, dict):
                # 更新邮箱
                if contacts.get('email') and contacts['email'] not in profile.emails:
                    profile.emails.append(contacts['email'])
                
                # 更新其他平台链接
                for platform, url in contacts.items():
                    if platform != 'email' and url:
                        if platform not in profile.platforms:
                            profile.platforms[platform] = url
                
                print(f"    ✓ Updated contact information")


def merge_scholar_data_to_profile(scholar_data: Dict, profile: AuthorProfile, priority: str = 'secondary'):
    """
    合并 Google Scholar 数据到 AuthorProfile
    Args:
        scholar_data: 从 parse_google_scholar 解析的数据
        profile: 目标 AuthorProfile
        priority: 'primary' | 'secondary'
    """
    if priority == 'primary':
        # Scholar 作为主要来源（中完整度）
        # Research Interests
        if scholar_data.get('research_interests'):
            interests = scholar_data['research_interests']
            if isinstance(interests, list):
                existing = set(i.lower() for i in profile.interests)
                for interest in interests:
                    if interest.lower() not in existing:
                        profile.interests.append(interest)
                print(f"    ✓ Filled research interests: {len(interests)} items")
        
        # Affiliation
        if not profile.affiliation_current and scholar_data.get('affiliation'):
            profile.affiliation_current = scholar_data['affiliation']
            print(f"    ✓ Filled affiliation: {profile.affiliation_current}")
        
        # Citation Stats
        if scholar_data.get('citation_stats'):
            stats = scholar_data['citation_stats']
            profile.social_impact = f"h-index: {stats.get('h_index', 0)}, citations: {stats.get('citations_all', 0)}"
            print(f"    ✓ Filled social impact: {profile.social_impact}")
        
        # Publications（添加时间筛选）
        if scholar_data.get('publications'):
            import datetime
            current_year = datetime.datetime.utcnow().year
            recent_year_threshold = current_year - 3  # 最近3年
            
            pubs = scholar_data['publications']
            existing_titles = {pub.get('title', '').lower() for pub in profile.selected_publications}
            new_count = 0
            
            for pub in pubs:  # ✅ 不限制数量，只按时间筛选
                # ✅ 只添加最近3年的论文
                pub_year = pub.get('year')
                if not pub_year or not isinstance(pub_year, int) or pub_year < recent_year_threshold:
                    continue
                
                if pub.get('title') and pub['title'].lower() not in existing_titles:
                    profile.selected_publications.append({
                        'title': pub['title'],
                        'year': pub_year,
                        'venue': pub.get('authors', ''),
                        'citations': pub.get('citations', 0),
                        'abstract': pub.get('abstract', '')
                    })
                    existing_titles.add(pub['title'].lower())
                    new_count += 1
            
            if new_count > 0:
                print(f"    ✓ Added {new_count} new publications from Scholar (recent 3 years)")
    
    elif priority == 'secondary':
        # Scholar 作为辅助来源（高完整度）
        # 只更新引用数和补充研究兴趣
        if scholar_data.get('publications'):
            scholar_pubs = {pub['title'].lower(): pub for pub in scholar_data['publications']}
            update_count = 0
            for pub in profile.selected_publications:
                pub_title_lower = pub.get('title', '').lower()
                if pub_title_lower in scholar_pubs and not pub.get('citations'):
                    pub['citations'] = scholar_pubs[pub_title_lower].get('citations', 0)
                    update_count += 1
            if update_count > 0:
                print(f"    ✓ Updated citation counts for {update_count} papers")
        
        # 补充研究兴趣
        if scholar_data.get('research_interests'):
            interests = scholar_data['research_interests']
            existing = set(i.lower() for i in profile.interests)
            new_interests = [i for i in interests if i.lower() not in existing]
            profile.interests.extend(new_interests)
            if new_interests:
                print(f"    ✓ Added {len(new_interests)} supplementary research interests")
        
        # 更新 social_impact（如果缺失）
        if not profile.social_impact and scholar_data.get('citation_stats'):
            stats = scholar_data['citation_stats']
            profile.social_impact = f"h-index: {stats.get('h_index', 0)}, citations: {stats.get('citations_all', 0)}"
            print(f"    ✓ Added social impact: {profile.social_impact}")


def merge_orcid_data_to_profile(orcid_data: Dict, profile: AuthorProfile, priority: str = 'filling'):
    """
    合并 ORCID 数据到 AuthorProfile
    
    Args:
        orcid_data: 从 parse_orcid 解析的数据
        profile: 目标 AuthorProfile
        priority: 'filling'（只填充缺失字段）
    """
    # ORCID 主要用于填充
    # Contact
    if orcid_data.get('contact'):
        contacts = orcid_data['contact']
        if isinstance(contacts, dict):
            if contacts.get('email') and not profile.emails:
                profile.emails.append(contacts['email'])
                print(f"    ✓ Filled email from ORCID")
    
    # Employment (填充 affiliation)
    if not profile.affiliation_current and orcid_data.get('employment'):
        employment = orcid_data['employment']
        if isinstance(employment, list) and employment:
            latest = employment[0]
            if isinstance(latest, dict):
                pos = latest.get('position', '')
                inst = latest.get('institution', '')
                if pos and inst:
                    profile.affiliation_current = f"{pos} at {inst}"
                    print(f"    ✓ Filled affiliation from ORCID")
    
    # Education (补充)
    if orcid_data.get('education'):
        print(f"    ✓ Extracted education data from ORCID")


def classify_openreview_career_education(
    openreview_result: Dict[str, Any],
    author_name: str,
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    """
    对 OpenReview 的 Career & Education History 进行分类，生成 OR baseline。

    输入：
        openreview_result['education']: OpenReview 原始 career & education 列表
    
    输出：
        在 openreview_result 中新增两个 baseline 字段（保留原始 education）：
        - or_career_education: 所有非工业界 career（学生阶段 + 学术职位 + 课程教学）
        - or_industrial_experience: 工业/公司经历
    """
    try:
        # 原始 career history
        career_history = openreview_result.get('education') or []
        if not career_history:
            # 如果没有 career history，初始化空 baseline
            openreview_result['or_career_education'] = []
            openreview_result['or_industrial_experience'] = []
            return openreview_result

        # 构建 LLM 分类用的文本
        lines: List[str] = []
        lines.append("=== OpenReview Career & Education History ===")
        lines.append(f"Name: {author_name}")
        lines.append("")
        lines.append("Career & Education History:")
        lines.append("")

        for idx, item in enumerate(career_history):
            position = item.get('position', '') or ''
            # 处理 institution：可能是字符串或对象（包含 name 字段）
            institution_raw = item.get('institution', '') or ''
            if isinstance(institution_raw, dict):
                institution = institution_raw.get('name', '') or ''
            else:
                institution = str(institution_raw) if institution_raw else ''
            start = item.get('start', '') or ''
            end = item.get('end', '') or 'Present'
            line = f"[{idx}] {position} at {institution} ({start}–{end})"
            lines.append(line)

        career_text = "\n".join(lines)

        # 使用九维度抽取器，将 OpenReview career 文本划分为：
        # - career_education: 所有非工业 career 阶段（学生 + 学术 + 教学）
        # - experience: 工业/公司经历
        extractor = NineDimensionExtractor(api_key=api_key)
        dim_results = extractor.extract_from_text(
            text=career_text,
            author_name=author_name,
            source_name="openreview_career_baseline",
            target_dimensions=["career_education", "experience"],
        )

        # 初始化 OR baseline 列表
        openreview_result['or_career_education'] = []
        openreview_result['or_industrial_experience'] = []

        # 1) Career & Education History（所有非工业 career 阶段）
        car_dim = dim_results.get("career_education")
        if car_dim and car_dim.success and car_dim.data:
            llm_career = car_dim.data.get("career_education") or []
            openreview_result['or_career_education'] = llm_career

        # 2) 工业/公司经历（Industrial Experience baseline）
        exp_dim = dim_results.get("experience")
        if exp_dim and exp_dim.success and exp_dim.data:
            llm_exp = exp_dim.data.get("experiences") or []
            openreview_result['or_industrial_experience'] = llm_exp

        print(f"[OR Baseline] Career & Education: {len(openreview_result['or_career_education'])} items")
        print(f"[OR Baseline] Industrial Experience: {len(openreview_result['or_industrial_experience'])} items")
        
        # 如果 LLM 分类失败或返回空结果，回退到直接使用原始 education 数据
        if not openreview_result['or_career_education'] and career_history:
            print(f"[OR Baseline] ⚠️ LLM classification returned no results, falling back to raw education data")
            # 直接将原始 education 数据转换为 or_career_education 格式
            fallback_career = []
            for item in career_history:
                position = item.get('position', '') or ''
                institution_raw = item.get('institution', '') or ''
                if isinstance(institution_raw, dict):
                    institution = institution_raw.get('name', '') or ''
                else:
                    institution = str(institution_raw) if institution_raw else ''
                start = item.get('start', '') or ''
                end = item.get('end', '') or ''
                
                # 构建 duration 字符串
                if start and end:
                    duration = f"{start}-{end}"
                elif start:
                    duration = f"{start}-Present"
                else:
                    duration = ""
                
                if position or institution:
                    fallback_career.append({
                        'degree_or_position': position,
                        'institution': institution,
                        'duration': duration,
                        'department': '',
                        'field': '',
                        'advisor': '',
                        'description': ''
                    })
            
            if fallback_career:
                openreview_result['or_career_education'] = fallback_career
                print(f"[OR Baseline] ✅ Fallback: Using {len(fallback_career)} raw education items")

    except Exception as e:
        # 分类失败时初始化空 baseline，不影响主流程
        print(f"[OR Baseline] ⚠️ Failed to classify career & education: {type(e).__name__}: {e}")
        import traceback
        print(f"[OR Baseline] Traceback: {traceback.format_exc()}")
        # 如果分类失败，尝试使用原始 education 数据作为回退
        career_history = openreview_result.get('education') or []
        if career_history:
            print(f"[OR Baseline] 🔄 Fallback: Using {len(career_history)} raw education items")
            fallback_career = []
            for item in career_history:
                position = item.get('position', '') or ''
                institution_raw = item.get('institution', '') or ''
                if isinstance(institution_raw, dict):
                    institution = institution_raw.get('name', '') or ''
                else:
                    institution = str(institution_raw) if institution_raw else ''
                start = item.get('start', '') or ''
                end = item.get('end', '') or ''
                
                if start and end:
                    duration = f"{start}-{end}"
                elif start:
                    duration = f"{start}-Present"
                else:
                    duration = ""
                
                if position or institution:
                    fallback_career.append({
                        'degree_or_position': position,
                        'institution': institution,
                        'duration': duration,
                        'department': '',
                        'field': '',
                        'advisor': '',
                        'description': ''
                    })
            openreview_result['or_career_education'] = fallback_career
        else:
            openreview_result['or_career_education'] = []
        openreview_result['or_industrial_experience'] = []

    return openreview_result


def fetch_all_sources_parallel_v2(profile: AuthorProfile, openreview_result: Dict, author_name: str, api_key: str = None) -> AuthorProfile:
    """
    并发爬取所有可用数据源并智能合并
    三大主源（必须尝试）：
    1. Homepage - 最全面的个人信息
    2. OpenReview - 已获取，用于交叉验证
    3. Google Scholar - 论文和引用数据
    
    辅助源（可选）：
    4. ORCID - 标准化数据
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import re
    personal_links = openreview_result.get('personal_links', {})
    # 收集所有可用源
    sources_to_fetch = {}
    homepage_fallback_candidates = []  # 存储回退搜索结果
    identity_clues = _collect_identity_clues(profile, openreview_result)
    # 三大主源 - Homepage 特殊处理（可访问性验证 + 回退）
    if personal_links.get('homepage'):
        homepage_url = personal_links['homepage']
        print(f"[Multi-Source] Homepage found in OpenReview: {homepage_url}")
        # 验证OpenReview homepage的可访问性
        print(f"[Multi-Source] Verifying homepage accessibility...")
        is_accessible, reason = check_url_accessibility(
            homepage_url, 
            timeout=getattr(config, 'OPENREVIEW_HOMEPAGE_VERIFICATION_TIMEOUT', 3)
        )
        if is_accessible:
            # ✅ 可访问 - 使用OpenReview链接
            print(f"[Multi-Source] ✅ Homepage accessible: {reason}")
            sources_to_fetch['homepage'] = homepage_url
        else:
            print(f"[Multi-Source] ⚠️ Homepage NOT accessible: {reason}")
    else:
        print(f"[Multi-Source] ⚠️ No homepage found in OpenReview")
        homepage_url = None
    
    # 统一处理：不可访问或没有 homepage 的情况，都执行回退搜索
    if not sources_to_fetch.get('homepage'):
        # 检查是否启用回退机制
        if getattr(config, 'OPENREVIEW_HOMEPAGE_FALLBACK_ENABLED', True):
            print(f"[Multi-Source] 🔄 FALLBACK: Starting homepage search...")
            # 获取机构信息和论文标题（用于新版本搜索策略）
            known_affiliation = profile.affiliation_current or ""
            # 尝试从论文列表中获取代表性论文标题
            paper_title = ""
            if profile.selected_publications:
                paper_title = profile.selected_publications[0].get('title', '')
            # 执行新版本回退搜索（使用 name + 机构 + homepage 策略）
            timeout_seconds = getattr(config, 'HOMEPAGE_SEARCH_TIMEOUT', 30)
            fallback_results = _execute_homepage_search_v2(
                author_name=author_name,
                known_affiliation=known_affiliation,
                paper_title=paper_title,
                timeout_seconds=timeout_seconds
            )
            if fallback_results:
                homepage_fallback_candidates = fallback_results
                print(f"[Multi-Source] ✅ Fallback found {len(fallback_results)} alternative homepages")
                validated_candidates = []  # 存储所有通过验证的候选
                author_name_parts = author_name.lower().replace('-', ' ').split()
                author_name_parts = [p for p in author_name_parts if len(p) > 2]  # 过滤短词
                print(f"[Homepage Validation] Trying up to {min(len(fallback_results), 5)} candidates...")
                # 最多验证前5个候选
                for idx, result in enumerate(fallback_results[:5], 1):
                    candidate_url = result['url']
                    print(f"\n[Homepage Validation] Candidate {idx}: {candidate_url}")
                    try:
                        # 步骤0：过滤明显不是homepage的URL
                        print(f"[Homepage Validation] Step 0: Check URL type...")
                        if any(candidate_url.lower().endswith(ext) for ext in ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz']):
                            print(f"[Homepage Validation] ❌ Rejected: File download (PDF/DOC/etc), not a homepage")
                            continue
                        excluded_domains = [
                            'arxiv.org', 'doi.org', 'acm.org', 'ieee.org', 
                            'springer.com', 'sciencedirect.com', 'researchgate.net',
                            'semanticscholar.org', 'openreview.net/profile'  # OpenReview profile不算homepage
                        ]
                        if any(domain in candidate_url.lower() for domain in excluded_domains):
                            print(f"[Homepage Validation] ❌ Rejected: Academic platform, not personal homepage")
                            continue
                        
                        if 'scholar.google.com/citations' in candidate_url.lower():
                            print(f"[Homepage Validation] 📚 Detected Google Scholar profile - Bonus discovery!")
                            print(f"[Homepage Validation] Scholar URL: {candidate_url}")
                            
                            try:
                                existing_scholar = None
                                if hasattr(profile, 'platforms') and isinstance(profile.platforms, dict):
                                    existing_scholar = profile.platforms.get('scholar')
                                
                                if not existing_scholar:
                                    # 保存到platforms字典
                                    if not hasattr(profile, 'platforms'):
                                        profile.platforms = {}
                                    profile.platforms['scholar'] = candidate_url
                                    print(f"[Homepage Validation] Saved Scholar URL to profile (was missing)")
                                else:
                                    print(f"[Homepage Validation] ℹ️ Profile already has Scholar URL: {existing_scholar}")
                                print(f"[Homepage Validation] 📊 Will include Scholar content in profile extraction")
                                import requests
                                from bs4 import BeautifulSoup
                                print(f"[Homepage Validation] Attempting to extract homepage URL from Scholar...")
                                scholar_response = requests.get(candidate_url, timeout=10, headers=config.UA)
                                if scholar_response.ok:
                                    scholar_soup = BeautifulSoup(scholar_response.text, 'html.parser')
                                    # Google Scholar的homepage链接通常在 <div id="gsc_prf_ivh"> 下
                                    homepage_link = None
                                    # 方法1：查找"Homepage"或"Website"链接
                                    for a_tag in scholar_soup.find_all('a', href=True):
                                        text = a_tag.get_text().strip().lower()
                                        if text in ['homepage', 'website', 'personal page', 'home']:
                                            homepage_link = a_tag['href']
                                            print(f"[Homepage Validation] ✅ Found homepage link from Scholar: {homepage_link}")
                                            break
                                    # 方法2：查找 gsc_prf_ivh div中的链接
                                    if not homepage_link:
                                        ivh_div = scholar_soup.find('div', {'id': 'gsc_prf_ivh'})
                                        if ivh_div:
                                            for a_tag in ivh_div.find_all('a', href=True):
                                                href = a_tag['href']
                                                # 排除Google Scholar自身的链接和邮箱
                                                if 'scholar.google' not in href and 'mailto:' not in href:
                                                    homepage_link = href
                                                    print(f"[Homepage Validation] ✅ Found homepage link from Scholar profile: {homepage_link}")
                                                    break
                                    if homepage_link:
                                        # 将提取的homepage加入候选列表（继续验证）
                                        print(f"[Homepage Validation] 🔄 Will validate extracted homepage: {homepage_link}")
                                        candidate_url = homepage_link
                                        result['url'] = homepage_link  # 更新结果中的URL
                                        print(f"[Homepage Validation] New candidate URL: {candidate_url}")
                                    else:
                                        print(f"[Homepage Validation] ℹ️ No homepage link found on Scholar page")
                                        print(f"[Homepage Validation] Will still use Scholar content as information source")
                                        # 不跳过，Scholar本身也有价值
                                        continue
                                else:
                                    print(f"[Homepage Validation] ⚠️ Failed to fetch Scholar page (HTTP {scholar_response.status_code})")
                                    continue
                                    
                            except Exception as e:
                                print(f"[Homepage Validation] ⚠️ Error processing Scholar: {str(e)[:100]}")
                                import traceback
                                traceback.print_exc()
                                continue
                        # 步骤1：尝试爬取少量内容（快速验证）
                        print(f"[Homepage Validation] Step 1: Quick fetch (max 3000 chars)...")
                        preview_content = fetch_text(candidate_url, max_chars=3000, snippet=result.get('snippet', ''))
                        if not preview_content or len(preview_content) < 100:
                            print(f"[Homepage Validation] ❌ Failed: Content too short ({len(preview_content) if preview_content else 0} chars)")
                            continue
                        
                        print(f"[Homepage Validation] ✅ Fetched {len(preview_content)} chars")
                        
                        # 步骤2：验证内容是否包含作者名（名字匹配）
                        print(f"[Homepage Validation] Step 2: Verify author identity...")
                        content_lower = preview_content.lower()
                        
                        # 计算名字匹配度
                        name_match_count = sum(1 for part in author_name_parts if part in content_lower)
                        name_match_ratio = name_match_count / len(author_name_parts) if author_name_parts else 0
                        
                        print(f"[Homepage Validation] Name match: {name_match_count}/{len(author_name_parts)} parts ({name_match_ratio:.0%})")
                        
                        has_identity_clues = bool(identity_clues.get('degree_terms') or identity_clues.get('org_terms'))
                        basic_ok, basic_reason = _passes_basic_identity(name_match_ratio, author_name_parts, content_lower, has_identity_clues)
                        if not basic_ok:
                            print(f"[Homepage Validation] ❌ Failed: {basic_reason}")
                        # 步骤2.5：使用教育/工业经历进行同名消歧
                        print(f"[Homepage Validation] Step 2.5: Degree/affiliation verification...")
                        degree_ok, matched_degrees, matched_orgs, has_clues = _passes_degree_verification(preview_content, identity_clues)
                        if not degree_ok:
                            print(f"[Homepage Validation] ❌ Rejected: No overlap with known degrees/organizations")
                            continue
                        if matched_degrees or matched_orgs:
                            print(f"[Homepage Validation] ✅ Degree/affiliation hints matched → degrees: {matched_degrees}, orgs: {matched_orgs}")
                        elif has_clues:
                            print(f"[Homepage Validation] ℹ️ Degree clues exist but none matched; accepted via strong name match")
                        else:
                            print(f"[Homepage Validation] ℹ️ No degree/organization clues available, strengthened name threshold applied")
                        # 步骤3：检查是否是团队/实验室页面（降低优先级）
                        print(f"[Homepage Validation] Step 3: Check page type...")
                        url_lower = candidate_url.lower()
                        is_team_site = any(indicator in url_lower for indicator in [
                            '/lab/', '/group/', '/team/', '/center/', '/institute/', '/department/',
                            '-lab.', '-group.', '-team.'
                        ])
                        
                        # 同时检查标题和内容
                        title_lower = result.get('title', '').lower()
                        is_team_content = any(keyword in content_lower or keyword in title_lower for keyword in [
                            'laboratory', 'research group', 'research team', 'lab members', 'group members'
                        ])
                        
                        is_team_page = is_team_site or is_team_content
                        
                        if is_team_page:
                            print(f"[Homepage Validation] ⚠️ Detected team/lab page")
                            # 检查是否有强个人信息
                            personal_indicators = ['email', 'cv', 'publications', 'biography', 'education']
                            has_personal_info = sum(1 for ind in personal_indicators if ind in content_lower) >= 2
                            
                            if not (has_personal_info and name_match_ratio >= 0.7):
                                print(f"[Homepage Validation] ❌ Rejected: Team page without strong personal info")
                                continue
                            else:
                                print(f"[Homepage Validation] ✅ Accepted: Team page with strong personal info")
                        
                        # 步骤4：通过验证！添加到候选列表
                        print(f"[Homepage Validation] ✅✅ PASSED validation")
                        validated_candidates.append({
                            'url': candidate_url,
                            'preview_content': preview_content,
                            'name_match_ratio': name_match_ratio,
                            'is_team_page': is_team_page,
                            'result': result
                        })
                    
                    except Exception as e:
                        print(f"[Homepage Validation] ❌ Error: {str(e)[:100]}")
                        import traceback
                        traceback.print_exc()
                        continue
                
                # 如果有通过验证的候选，进行完整爬取并比较信息量
                if validated_candidates:
                    print(f"\n[Homepage Selection] ✅ {len(validated_candidates)} candidates passed validation")
                    print(f"[Homepage Selection] Now fetching full content to compare information richness...")
                    
                    # ✅ 根据候选数量动态调整并行worker数量
                    num_candidates = len(validated_candidates)
                    num_workers = min(num_candidates, 3)  # 最多3个并行worker，避免过多并发
                    
                    print(f"[Homepage Selection] Using {num_workers} parallel workers for {num_candidates} candidates")
                    
                    # 定义单个homepage的爬取函数
                    def fetch_single_homepage(candidate_info):
                        """爬取单个候选homepage并计算得分"""
                        idx, candidate = candidate_info
                        candidate_url = candidate['url']
                        
                        try:
                            print(f"\n[Homepage Selection] [{idx}/{num_candidates}] Fetching: {candidate_url}")
                            
                            # 爬取完整内容
                            homepage_data = fetch_homepage_comprehensive(
                                url=candidate_url,
                                author_name=author_name,
                                max_chars=50000,
                                include_subpages=False,  # 快速模式，不抓取子页面
                                max_subpages=0
                            )
                            
                            if homepage_data and homepage_data.get('success'):
                                # 计算信息丰富度得分
                                text_content = homepage_data.get('text_content', '')
                                social_platforms = homepage_data.get('social_platforms', {})
                                emails = homepage_data.get('emails', [])
                                
                                info_score = (
                                    len(text_content) * 1.0 +           # 文本内容（主要指标）
                                    len(social_platforms) * 200 +       # 社交平台（每个+200分）
                                    len(emails) * 100 +                 # 邮箱（每个+100分）
                                    candidate['name_match_ratio'] * 500 # 名字匹配度（加权）
                                )
                                
                                # 如果是团队页面，降低得分
                                if candidate['is_team_page']:
                                    info_score *= 0.8
                                
                                result = {
                                    'url': candidate_url,
                                    'data': homepage_data,
                                    'info_score': info_score,
                                    'text_length': len(text_content),
                                    'social_count': len(social_platforms),
                                    'email_count': len(emails),
                                    'name_match_ratio': candidate['name_match_ratio'],
                                    'is_team_page': candidate['is_team_page']
                                }
                                
                                print(f"[Homepage Selection] [{idx}/{num_candidates}] ✅ Success - Score: {info_score:.0f}")
                                print(f"  - Text: {len(text_content)} chars, Social: {len(social_platforms)}, Email: {len(emails)}")
                                
                                return result
                            else:
                                print(f"[Homepage Selection] [{idx}/{num_candidates}] ❌ Fetch failed")
                                return None
                                
                        except Exception as e:
                            print(f"[Homepage Selection] [{idx}/{num_candidates}] ❌ Error: {str(e)[:100]}")
                            import traceback
                            traceback.print_exc()
                            return None
                    
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    import time
                    
                    fetched_homepages = []
                    start_time = time.time()
                    
                    with ThreadPoolExecutor(max_workers=num_workers) as executor:
                        # 提交所有任务
                        future_to_candidate = {
                            executor.submit(fetch_single_homepage, (idx+1, candidate)): candidate
                            for idx, candidate in enumerate(validated_candidates)
                        }
                        
                        # 收集结果
                        for future in as_completed(future_to_candidate):
                            try:
                                result = future.result(timeout=30)
                                if result:
                                    fetched_homepages.append(result)
                            except Exception as e:
                                print(f"[Homepage Selection] ⚠️ Future error: {str(e)[:100]}")
                    
                    elapsed_time = time.time() - start_time
                    print(f"\n[Homepage Selection] ✅ Parallel fetch completed in {elapsed_time:.1f}s")
                    print(f"[Homepage Selection] Successfully fetched {len(fetched_homepages)}/{num_candidates} homepages")
                    
                    # 选择信息量最丰富的homepage
                    if fetched_homepages:
                        # 按信息得分排序
                        fetched_homepages.sort(key=lambda x: x['info_score'], reverse=True)
                        best_homepage = fetched_homepages[0]
                        
                        print(f"\n[Homepage Selection] 📊 Comparison results:")
                        for idx, hp in enumerate(fetched_homepages, 1):
                            marker = "👑 BEST" if idx == 1 else f"   #{idx}"
                            print(f"{marker} - Score: {hp['info_score']:.0f} | {hp['url']}")
                            print(f"       Text: {hp['text_length']}c, Social: {hp['social_count']}, Email: {hp['email_count']}, Match: {hp['name_match_ratio']:.0%}")
                        selected_url = best_homepage['url']
                        selected_data = best_homepage['data']
                        
                        print(f"\n[Homepage Selection] ✅✅ SELECTED BEST (Primary): {selected_url}")
                        print(f"[Homepage Selection] Reason: Highest information richness (score: {best_homepage['info_score']:.0f})")
                        print(f"[Homepage Selection] 📚 Total validated homepages to use: {len(fetched_homepages)}")
                        
                        # 1. 清除原有的homepage配置（包括OpenReview的）
                        if 'homepage' in sources_to_fetch:
                            old_url = sources_to_fetch['homepage']
                            print(f"[Homepage Selection] 🔄 Replacing old homepage: {old_url}")
                            del sources_to_fetch['homepage']
                        
                        # 2. 使用新的validated URL（不需要再爬取，直接用已获取的数据）
                        sources_to_fetch['homepage'] = selected_url
                        
                        # 3. 更新profile的homepage URL（主URL）
                        profile.homepage_url = selected_url
                        print(f"[Multi-Source] ✅ Updated profile.homepage_url to: {selected_url}")
                        
                        # 4. 更新platforms字典
                        if hasattr(profile, 'platforms') and isinstance(profile.platforms, dict):
                            profile.platforms['homepage'] = selected_url
                            print(f"[Multi-Source] ✅ Updated profile.platforms['homepage']")
                        
                        # 5. ✅ 关键改进：保存所有验证通过的homepage数据（用于信息提取）
                        print(f"[Multi-Source] ✅ Preserving all {len(fetched_homepages)} validated homepage(s) for profile extraction")
                        profile._prefetched_homepage_data = selected_data  # 主homepage数据
                        profile._all_validated_homepages = fetched_homepages  # 所有验证通过的homepage数据
                        
                        # 输出保留的所有homepage
                        print(f"[Multi-Source] 📝 All validated homepages that will be used:")
                        for idx, hp in enumerate(fetched_homepages, 1):
                            marker = "🏆 Primary" if idx == 1 else f"📖 Additional #{idx-1}"
                            print(f"  {marker}: {hp['url']} (Score: {hp['info_score']:.0f}, Text: {hp['text_length']}c)")
                        
                    else:
                        print(f"\n[Multi-Source] ❌ All validated candidates failed full fetch")
                else:
                    print(f"\n[Multi-Source] ❌ No candidates passed validation from {len(fallback_results)} alternatives")
                    print(f"[Multi-Source] Reason: All candidates failed validation (content fetch or author identity)")
            else:
                print(f"[Multi-Source] ❌ Fallback search found no alternatives")
            
            # 如果原来有 homepage_url 但不可访问，保留作为参考
            if homepage_url:
                print(f"[Multi-Source] ℹ️ Keeping OpenReview URL as reference: {homepage_url} (inaccessible)")
        else:
            print(f"[Multi-Source] ⚠️ Fallback disabled, skipping homepage search")
    
    # ✅ Google Scholar处理（包含fallback中发现的）
    if personal_links.get('google_scholar'):
        sources_to_fetch['scholar'] = personal_links['google_scholar']
    elif hasattr(profile, 'platforms') and isinstance(profile.platforms, dict):
        fallback_scholar = profile.platforms.get('scholar')
        if fallback_scholar:
            sources_to_fetch['scholar'] = fallback_scholar
            print(f"[Multi-Source] Adding Scholar URL discovered from fallback: {fallback_scholar}")
    # OpenReview数据已有，不需要重复爬取
    
    # 辅助源
    if personal_links.get('orcid'):
        sources_to_fetch['orcid'] = personal_links['orcid']
    
    print(f"[Multi-Source] Available sources: {list(sources_to_fetch.keys())}")
    if homepage_fallback_candidates:
        print(f"[Multi-Source] + {len(homepage_fallback_candidates)} fallback homepage candidates in reserve")
    
    # ==== Step 1: OpenReview Career 分类生成 baseline ====
    openreview_result = classify_openreview_career_education(
        openreview_result=openreview_result,
        author_name=author_name,
        api_key=api_key,
    )
    print(f"[Multi-Source] ✅ classify_openreview_career_education completed")
    # 更新 profile 中的 openreview_data（确保包含分类后的 or_career_education 和 or_industrial_experience）
    profile._openreview_data = openreview_result
    print(f"[Multi-Source] ✅ Updated profile._openreview_data with classified career/education data")
    
    if not sources_to_fetch:
        print(f"[Multi-Source] No additional sources to fetch, using OpenReview data only")
        return profile
    
    # 定义爬取函数
    def fetch_homepage_safe(url):
        try:
            # ✅ 关键：先检查是否有预取的数据（来自fallback validation）
            if hasattr(profile, '_prefetched_homepage_data'):
                prefetched_data = profile._prefetched_homepage_data
                # 验证预取的URL是否匹配当前要爬取的URL
                if prefetched_data:
                    print(f"[Multi-Source] [Homepage] ✅ Using pre-fetched data (skipping duplicate fetch)")
                    
                    # 从预取数据中提取文本（✅ 正确的键名）
                    text_content = prefetched_data.get('text_content', '')
                    
                    if text_content and len(text_content) >= 100:
                        print(f"[Multi-Source] [Homepage] ✅ Pre-fetched data contains {len(text_content)} chars")
                        
                        # 构建完整文本（和原逻辑相同）
                        text_parts = []
                        text_parts.append(f"========== HOMEPAGE FULL TEXT ==========")
                        text_parts.append(f"URL: {url}")
                        text_parts.append("")
                        text_parts.append(text_content[:30000])  # 保留前30k字符
                        text_parts.append("")
                        text_parts.append("="*80)
                        
                        combined_text = '\n'.join(text_parts)
                        
                        # 清除预取数据（已使用）
                        delattr(profile, '_prefetched_homepage_data')
                        
                        return {'text': combined_text, 'metadata': prefetched_data}
                    else:
                        print(f"[Multi-Source] [Homepage] ⚠️ Pre-fetched data invalid, will re-fetch")
                        delattr(profile, '_prefetched_homepage_data')
            
            print(f"[Multi-Source] [Homepage] Fetching: {url}")
            basic_result = fetch_homepage_comprehensive(
                url,
                author_name=author_name,
                max_chars=50000,
                include_subpages=False,
                max_subpages=0
            )

            if not basic_result or not basic_result.get('success'):
                print(f"[Multi-Source] [Homepage] ❌ Traditional parser failed")
                return None

            result = _run_homepage_agent_upgrade(
                result=basic_result,
                url=url,
                author_name=author_name,
                max_chars=50000,
                log_prefix="[Multi-Source] [Homepage]",
                api_key=api_key  # 🆕 传递 api_key，使 Agent 能够运行 LLM 提取
            )
            if 'extracted_dimensions' not in result:
                result.setdefault('extracted_dimensions', {})
            if result.get('extracted_dimensions'):
                print(f"[Multi-Source] [Homepage] 📊 Reused extracted dimensions from existing metadata: {list(result['extracted_dimensions'].keys())}")
            
            text_content = result.get('text_content', '')
            if not text_content:
                print(f"[Multi-Source] [Homepage] ❌ No content extracted")
                return None
                
            print(f"[Multi-Source] [Homepage] ✅ Extracted {len(text_content)} chars of text")
            
            # 🆕 确保 metadata 中包含 extracted_dimensions（如果存在）
            if not result.get('metadata'):
                result['metadata'] = {}
            if result.get('extracted_dimensions') and 'extracted_dimensions' not in result['metadata']:
                result['metadata']['extracted_dimensions'] = result.get('extracted_dimensions')
                print(f"[Multi-Source] [Homepage] ✅ Stored extracted_dimensions in result['metadata']")
            meta = result.get('extracted_dimensions_meta') or result['metadata'].get('extracted_dimensions_meta')
            if meta:
                result['metadata']['extracted_dimensions_meta'] = meta
            
            # 🆕 提取 extraction_text（如果存在）
            extraction_text = result.get('extraction_text', '')
            if not extraction_text and result.get('agent_metadata'):
                extraction_text = result['agent_metadata'].get('extraction_text', '')
            
            # 构建完整文本
            text_parts = []
            text_parts.append(f"========== HOMEPAGE FULL TEXT ==========")
            text_parts.append(f"URL: {url}")
            text_parts.append("")
            text_parts.append(text_content[:30000])  # 保留前30k字符
            text_parts.append("")
            text_parts.append("="*80)
            
            combined_text = '\n'.join(text_parts)
            
            print(f"[Multi-Source] [Homepage] ✅ Success: {len(combined_text)} chars total")
            
            # 返回包含 extraction_text 的 metadata
            return {'text': combined_text, 'metadata': result, 'extraction_text': extraction_text}
            
        except Exception as e:
            print(f"[Multi-Source] [Homepage] ❌ Error: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            return None
    
    def fetch_scholar_safe(url):
        try:
            print(f"[Multi-Source] [Scholar] Fetching: {url[:60]}...")
            from .homepage_parser import HomepageParser
            parser = HomepageParser(api_key=api_key)
            
            # 使用Scholar专用方法
            result = parser.parse_google_scholar(url)
            
            if result and not result.get('error'):
                # 构建文本（包含论文列表、引用数等）
                text_parts = []
                text_parts.append(f"Google Scholar Profile: {url}")
                text_parts.append("="*80)
                
                if result.get('h_index'):
                    text_parts.append(f"\nH-index: {result['h_index']}")
                if result.get('total_citations'):
                    text_parts.append(f"Total Citations: {result['total_citations']}")
                if result.get('i10_index'):
                    text_parts.append(f"i10-index: {result['i10_index']}")
                
                if result.get('publications'):
                    text_parts.append(f"\nPublications ({len(result['publications'])}):")
                    for pub in result['publications'][:20]:
                        title = pub.get('title', '')
                        year = pub.get('year', '')
                        citations = pub.get('citations', 0)
                        text_parts.append(f"  - {title} ({year}) - Citations: {citations}")
                
                combined_text = '\n'.join(text_parts)
                print(f"[Multi-Source] [Scholar] ✅ Success: {len(combined_text)} chars")
                print(f"[Multi-Source] [Scholar]    - Publications: {len(result.get('publications', []))}")
                
                return {'text': combined_text, 'metadata': result}
            else:
                error_msg = result.get('error', 'Unknown') if result else 'No result'
                print(f"[Multi-Source] [Scholar] ⚠️ Failed: {error_msg} (likely 403)")
                return None
                
        except Exception as e:
            print(f"[Multi-Source] [Scholar] ❌ Error: {str(e)[:100]}")
            return None
    
    def fetch_orcid_safe(url):
        try:
            print(f"[Multi-Source] [ORCID] Fetching: {url[:60]}...")
            from .homepage_parser import HomepageParser
            parser = HomepageParser(api_key=api_key)
            
            # 🔧 FIX: 使用关键字参数明确指定timeout，减少超时时间（10秒）和重试次数（1次），快速失败
            try:
                orcid_html = parser.fetch_page(url=url, use_scholar_session=False, max_retries=1, timeout=10)
            except TypeError as te:
                # 兼容旧版本：如果timeout参数不支持，使用默认参数
                print(f"[Multi-Source] [ORCID] ⚠️ timeout parameter not supported, using defaults: {te}")
                orcid_html = parser.fetch_page(url, use_scholar_session=False, max_retries=1)
            
            if orcid_html and len(orcid_html) > 500:
                orcid_text = parser.extract_structured_text(orcid_html)
                print(f"[Multi-Source] [ORCID] ✅ Success: {len(orcid_text)} chars")
                return {'text': orcid_text}
            else:
                print(f"[Multi-Source] [ORCID] ⚠️ No data (skipping, will use other sources)")
                return None
        except requests.exceptions.Timeout:
            print(f"[Multi-Source] [ORCID] ❌ Timeout (skipping)")
            return None
        except Exception as e:
            print(f"[Multi-Source] [ORCID] ❌ Error (skipping): {type(e).__name__}: {str(e)[:80]}")
            return None
    
    # 爬取函数映射
    fetch_functions = {
        'homepage': fetch_homepage_safe,
        'scholar': fetch_scholar_safe,
        'orcid': fetch_orcid_safe
    }
    
    fetch_results = {}
    failed_sources = []
    
    # ✅ 允许所有源并行获取（无需限制为4）
    max_parallel = len(sources_to_fetch)
    with ThreadPoolExecutor(max_workers=max_parallel) as executor:
        futures = {}
        for source_name, source_url in sources_to_fetch.items():
            if source_name in fetch_functions:
                future = executor.submit(fetch_functions[source_name], source_url)
                futures[future] = source_name
        
        # 策略：让每个任务独立超时，正在爬取的任务不会被打断
        for future in as_completed(futures):
            source_name = futures[future]
            try:
                # ✅ 单任务超时：180秒（足够长，适应全量爬取）
                # 如果真的卡住，这个会触发；如果正在爬取，会等待完成
                result = future.result(timeout=180)
                if result:
                    fetch_results[source_name] = result
                    print(f"[Multi-Source] ✓ {source_name} completed successfully")
                else:
                    failed_sources.append(source_name)
                    print(f"[Multi-Source] ○ {source_name} returned no data")
            except TimeoutError:
                # ⏱️ 单任务真的超时了（180秒无响应）
                failed_sources.append(source_name)
                print(f"[Multi-Source] ⏱ {source_name} timeout after 180s (真正卡住)")
            except Exception as e:
                failed_sources.append(source_name)
                print(f"[Multi-Source] ✗ {source_name} error: {str(e)[:60]}")
    
    # 📊 详细的结果统计
    print(f"\n[Multi-Source] Fetch summary:")
    print(f"  - Attempted: {len(sources_to_fetch)} sources")
    print(f"  - Successful: {len(fetch_results)} sources")
    if fetch_results:
        print(f"  - ✅ Success: {', '.join(fetch_results.keys())}")
    if failed_sources:
        print(f"  - ❌ Failed/Skipped: {', '.join(failed_sources)}")
    
    if not fetch_results:
        print(f"[Multi-Source] ⚠️ No additional sources fetched successfully")
        print(f"[Multi-Source] ✅ Will continue with OpenReview data only")
    
    # ==========================================================================
    # 智能合并数据 - 以Homepage和OpenReview为主
    # ==========================================================================
    
    print(f"\n[Multi-Source] Merging data from all sources...")
    
    # 1. 提取邮箱（优先级：Homepage > OpenReview）
    emails_set = set()
    if 'homepage' in fetch_results:
        homepage_text = fetch_results['homepage'].get('text', '')
        # 简单的邮箱提取
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        found_emails = re.findall(email_pattern, homepage_text)
        emails_set.update(found_emails[:3])  # 最多3个
    
    if emails_set:
        profile.emails = list(emails_set)
        print(f"[Multi-Source]   ✅ Emails: {len(profile.emails)}")
    
    # ==========================================================================
    # 数据保存 - 保存原始文本供九维度提取使用
    # ==========================================================================
    
    print(f"\n[Multi-Source] Saving raw data to profile...")
    
    # 保存Homepage文本（最重要的数据源）
    if 'homepage' in fetch_results:
        primary_text = fetch_results['homepage'].get('text', '') or ''  # 确保不为 None
        extraction_text = fetch_results['homepage'].get('extraction_text', '') or ''  # 确保不为 None
        profile._homepage_metadata = fetch_results['homepage'].get('metadata', {}) or {}
        
        # 确保 metadata 中包含 extracted_dimensions（如果存在）
        if not profile._homepage_metadata:
            profile._homepage_metadata = {}
        if fetch_results['homepage'].get('metadata', {}).get('extracted_dimensions'):
            if 'extracted_dimensions' not in profile._homepage_metadata:
                profile._homepage_metadata['extracted_dimensions'] = fetch_results['homepage']['metadata']['extracted_dimensions']
                print(f"[Multi-Source] ✅ Stored extracted_dimensions in profile._homepage_metadata: {len(profile._homepage_metadata['extracted_dimensions'])} dimensions")
        if fetch_results['homepage'].get('metadata', {}).get('extracted_dimensions_meta'):
            profile._homepage_metadata['extracted_dimensions_meta'] = fetch_results['homepage']['metadata']['extracted_dimensions_meta']
        
        # 检查是否有额外的验证通过的homepage
        if hasattr(profile, '_all_validated_homepages') and len(profile._all_validated_homepages) > 1:
            print(f"[Multi-Source] 📚 Merging {len(profile._all_validated_homepages)} validated homepage(s)...")
            
            all_homepage_texts = []
            
            # 主homepage（已经从fetch_results获取）
            all_homepage_texts.append(f"========== PRIMARY HOMEPAGE ==========")
            all_homepage_texts.append(f"URL: {profile.homepage_url}")
            all_homepage_texts.append(primary_text)
            all_homepage_texts.append("="*80)
            
            # 额外的验证通过的homepage（从预取数据获取）
            for idx, hp_data in enumerate(profile._all_validated_homepages[1:], 2):
                hp_url = hp_data['url']
                hp_text = hp_data['data'].get('text_content', '')
                
                if hp_text and len(hp_text) >= 100:
                    all_homepage_texts.append(f"\n========== ADDITIONAL HOMEPAGE #{idx-1} ==========")
                    all_homepage_texts.append(f"URL: {hp_url}")
                    all_homepage_texts.append(hp_text[:30000])  # 限制每个额外homepage的长度
                    all_homepage_texts.append("="*80)
                    print(f"[Multi-Source]   📖 Added homepage #{idx-1}: {hp_url} ({len(hp_text)} chars)")
            
            profile._homepage_raw_text = '\n'.join(all_homepage_texts)
            print(f"[Multi-Source]   ✅ Homepage (merged): {len(profile._homepage_raw_text)} chars from {len(profile._all_validated_homepages)} source(s)")
        else:
            profile._homepage_raw_text = primary_text
            print(f"[Multi-Source]   ✅ Homepage: {len(profile._homepage_raw_text)} chars saved")
        
        if extraction_text:
            profile._homepage_extraction_text = extraction_text
            print(f"[Multi-Source]   📝 Homepage (extraction): {len(extraction_text)} chars (智能压缩)")
        else:
            # 如果没有extraction_text，回退到完整文本
            profile._homepage_extraction_text = profile._homepage_raw_text
    else:
        profile._homepage_raw_text = ''
        profile._homepage_extraction_text = ''
        print(f"[Multi-Source]   ⚠️ Homepage: No data")
    
    # 保存Scholar文本（论文和引用数据）
    if 'scholar' in fetch_results:
        profile._scholar_raw_text = fetch_results['scholar'].get('text', '') or ''  # 确保不为 None
        profile._scholar_html = fetch_results['scholar'].get('html', '') or ''  # 确保不为 None
        profile._scholar_metadata = fetch_results['scholar'].get('metadata', {}) or {}  # 保存结构化数据
        print(f"[Multi-Source]   ✅ Scholar: {len(profile._scholar_raw_text)} chars saved")
    else:
        profile._scholar_raw_text = ''
        profile._scholar_html = ''
        profile._scholar_metadata = {}
        print(f"[Multi-Source]   ⚠️ Scholar: No data (often blocked)")
    
    # 保存ORCID文本（辅助源）
    if 'orcid' in fetch_results:
        profile._orcid_raw_text = fetch_results['orcid'].get('text', '') or ''  # 确保不为 None
        print(f"[Multi-Source]   ✅ ORCID: {len(profile._orcid_raw_text)} chars saved")
    else:
        profile._orcid_raw_text = ''
    # ==== Step 2: 构建多源文本（排除 OR career，避免重复）====
    # 🆕 OpenReview 文本只包含：个人链接 + 论文列表
    # 不再包含 education/experience/teaching（这三块已在 Step 1 处理）
    openreview_text_parts = []
    openreview_text_parts.append(f"=== OpenReview Profile ===")
    openreview_text_parts.append(f"Name: {author_name}")
    
    # 添加个人链接
    if personal_links:
        openreview_text_parts.append(f"\nPersonal Links:")
        for link_type, link_url in personal_links.items():
            openreview_text_parts.append(f"  - {link_type}: {link_url}")

    # 添加论文列表（这块仍然需要给九维 LLM）
    publications = openreview_result.get('publications', [])
    if publications:
        openreview_text_parts.append(f"\nPublications ({len(publications)}):")
        for pub in publications[:20]:  # 最多20篇
            openreview_text_parts.append(f"  - {pub.get('title', '')} ({pub.get('venue', '')}, {pub.get('year', '')})")
    
    profile._openreview_raw_text = '\n'.join(openreview_text_parts)
    print(f"[Multi-Source]   ✅ OpenReview: {len(profile._openreview_raw_text)} chars (publications only, career excluded)")
    # 统计总数据量
    total_chars = (
        len(profile._homepage_raw_text) +
        len(profile._scholar_raw_text) +
        len(profile._orcid_raw_text) +
        len(profile._openreview_raw_text)
    )
    print(f"\n[Multi-Source] ✅ Total raw data: {total_chars:,} chars from {len(fetch_results)} sources")
    print(f"[Multi-Source] Ready for nine-dimension extraction\n")
    return profile
def discover_author_profile(first_author: str, paper_title: str, aliases: List[str] = None,
                         k_queries: int = 40, author_id: str = None, api_key: str = None,
                         openreview_result: Optional[Dict[str, Any]] = None,
                         paper_info: Optional[Dict[str, Any]] = None) -> AuthorProfile:
    """Main function to discover comprehensive author profile with OpenReview priority
    
    Args:
        first_author: Author name
        paper_title: Paper title (optional)
        aliases: List of author name aliases
        k_queries: Number of queries for search
        author_id: OpenReview author ID (optional)
        api_key: API key for LLM services
        openreview_result: Optional pre-fetched OpenReview result (to avoid duplicate API calls)
    """
    aliases = aliases or []
    # Phase 0: MANDATORY OpenReview Check - If no OpenReview, skip the candidate entirely
    if openreview_result is None:
        import random
        delay = random.uniform(0.1, 0.5)
        time.sleep(delay)
        
        if paper_title:
            print(f"[Author Discovery] Phase 0: Trying title-based search for {paper_title[:60]}...")
            openreview_result = search_openreview_profile_by_title(
                paper_title=paper_title,
                first_author=first_author,
                api_key=api_key,
                paper_info=paper_info
            )
            
            if openreview_result:
                print(f"[Author Discovery] ✅ Title-based search succeeded")
            else:
                print(f"[Author Discovery] ⚠️ Title-based search failed, falling back to name search")
                # 回退到原来的 name-based 搜索
                openreview_result = search_openreview_profile(first_author, api_key=api_key)
        else:
            # 没有 paper_title，使用原来的 name-based 搜索
            print(f"[Author Discovery] Phase 0: Checking OpenReview for {first_author} (no paper_title provided)...")
            openreview_result = search_openreview_profile(first_author, api_key=api_key)
    else:
        print(f"[Author Discovery] Phase 0: Using provided OpenReview result (skipping API call)")
    
    if not openreview_result:
        print(f"[Author Discovery] ❌ No OpenReview profile found for {first_author}, SKIPPING CANDIDATE")
        return None  # STRICT: No OpenReview = Skip candidate
    
    # ============================ DEDUPLICATION CHECK ============================
    # 提取 OpenReview ID 并进行去重检查
    openreview_id = openreview_result.get('openreview_id', '')
    # 如果 openreview_id 为空，尝试从 URL 中提取
    if not openreview_id:
        openreview_url = openreview_result.get('openreview_url', '')
        if openreview_url:
            # 使用正则表达式从 URL 中提取 ID
            match = ID_PATTERNS['openreview'].search(openreview_url)
            if match:
                openreview_id = match.group(1)
    # 检查是否已处理过该 ID
    if openreview_id:
        if openreview_id in _processed_openreview_ids:
            print(f"[Author Discovery] ⚠️ Duplicate OpenReview ID '{openreview_id}' for {first_author}, SKIPPING CANDIDATE")
            return None  # 已处理过，跳过该候选人
        
        # 将 ID 加入集合
        _processed_openreview_ids.add(openreview_id)
        print(f"[Author Discovery] ✅ Added OpenReview ID '{openreview_id}' to processed set (total: {len(_processed_openreview_ids)})")
    else:
        print(f"[Author Discovery] ⚠️ Warning: Could not extract OpenReview ID for {first_author}, continuing without deduplication")
    # ============================ END DEDUPLICATION CHECK ============================
    print(f"[Author Discovery] ✅ Found OpenReview profile: {openreview_result['openreview_url']}")
    names_from_openreview = openreview_result.get('names', [])
    openreview_aliases = []
    for name_obj in names_from_openreview:
        full_name = f"{name_obj.get('first', '')} {name_obj.get('middle', '')} {name_obj.get('last', '')}".strip()
        if full_name and full_name != first_author:
            openreview_aliases.append(full_name)
    
    # Extract platforms from personal_links
    personal_links = openreview_result.get('personal_links', {})
    platforms_dict = {}
    if personal_links.get('homepage'):
        platforms_dict['homepage'] = personal_links['homepage']
    if personal_links.get('google_scholar'):
        platforms_dict['scholar'] = personal_links['google_scholar']
    if personal_links.get('orcid'):
        platforms_dict['orcid'] = personal_links['orcid']
    if personal_links.get('linkedin'):
        platforms_dict['linkedin'] = personal_links['linkedin']
    # Always add OpenReview profile
    platforms_dict['openreview'] = openreview_result['openreview_url']
    
    # Extract education info to infer affiliation
    education_list = openreview_result.get('education', [])
    current_affiliation = None
    if education_list:
        # Use most recent (first) education entry
        latest_edu = education_list[0]
        position = latest_edu.get('position', '')
        institution = latest_edu.get('institution', '')
        if position and institution:
            current_affiliation = f"{position} at {institution}"
        elif institution:
            current_affiliation = institution
    
    # Extract publications from OpenReview with time filtering
    publications_from_openreview = openreview_result.get('publications', [])
    
    # ✅ 时间筛选：只保留最近 3 年的论文
    import datetime
    current_year = datetime.datetime.utcnow().year
    recent_year_threshold = current_year - 3  # 最近3年
    
    selected_pubs = []
    for pub in publications_from_openreview:
        pub_year = pub.get('year')
        
        # 跳过没有年份信息的论文
        if not pub_year:
            continue
        
        # ✅ 只保留最近 3 年的论文
        if isinstance(pub_year, int) and pub_year >= recent_year_threshold:
            selected_pubs.append({
                'title': pub.get('title', ''),
                'authors': pub.get('authors', []),  # ✅ FIX: 保留作者列表用于计算 paper_score
                'year': pub_year,
                'venue': pub.get('venue', ''),
                'url': pub.get('url', ''),
                'citations': 0,  # OpenReview不提供引用数
                'abstract': pub.get('abstract', '')  # ✅ 保留摘要用于后续分类
            })
    
    # 按年份降序排序（最新的在前）
    selected_pubs.sort(key=lambda x: x.get('year', 0), reverse=True)
    
    # Build profile
    profile = AuthorProfile(
        name=first_author,
        aliases=list(set(aliases + openreview_aliases))[:10],  # 合并并去重
        platforms=platforms_dict,
        ids={},
        homepage_url=personal_links.get('homepage'),
        affiliation_current=current_affiliation,
        emails=[],
        interests=[],
        selected_publications=selected_pubs,
        confidence=0.7,  # OpenReview数据置信度较高
        notable_achievements=[],
        social_impact=None,
        career_stage=None,
        overall_score=0.0
    )
    
    # Store OpenReview data for EnhancedProfile generation
    profile._openreview_data = openreview_result
    
    print(f"[OpenReview] {first_author}: {len(profile.selected_publications)} pubs, {len(profile.platforms)} platforms")
    
    # Collect and fetch all available sources in parallel
    profile = fetch_all_sources_parallel_v2(profile, openreview_result, first_author, api_key)  
    # Phase 4: Supplement with Semantic Scholar data if author_id provided
    if author_id:
        print(f"\n[Author Discovery] Phase 4: Supplementing with Semantic Scholar...")
        try:
            from .semantic_paper_search import SemanticScholarClient
            from . import config
            s2_client = SemanticScholarClient(api_key=config.SEMANTIC_SCHOLAR_API_KEY if config.SEMANTIC_SCHOLAR_API_KEY else None)
            
            # Get h-index and citation metrics
            s2_profile = s2_client.get_author_profile_info(author_id)
            if s2_profile:
                h_index = s2_profile.get('hIndex', 0)
                citation_count = s2_profile.get('citationCount', 0)
                paper_count = s2_profile.get('paperCount', 0)
                
                if h_index > 0 or citation_count > 0:
                    profile.social_impact = f"h-index: {h_index}, citations: {citation_count}, papers: {paper_count}"
                    print(f"[S2] Added metrics: {profile.social_impact}")
                
                # Supplement affiliation if not set
                if not profile.affiliation_current and s2_profile.get('affiliations'):
                    profile.affiliation_current = s2_profile['affiliations'][0].get('name', '')
            
            # Update publications with citation counts
            papers = s2_client.get_author_papers(author_id, limit=50, sort="citationCount")
            if papers:
                # Merge with OpenReview publications
                s2_pubs_dict = {p['title'].lower().strip(): p for p in papers}
                for pub in profile.selected_publications:
                    pub_title_lower = pub.get('title', '').lower().strip()
                    if pub_title_lower in s2_pubs_dict:
                        # Update citation count from S2
                        pub['citations'] = s2_pubs_dict[pub_title_lower].get('citationCount', 0)
                
                print(f"[S2] Updated citation counts for {len(profile.selected_publications)} papers")
            
        except Exception as e:
            print(f"[S2] Failed to fetch from Semantic Scholar: {e}")

    # 最终档案精炼
    profile = refine_author_profile(profile, first_author)
    
    # 计算综合评分
    profile.overall_score = calculate_overall_score(profile)
    
    return profile

# ============================ S2 TOP PAPERS SELECTION ============================
# ============================ EVALUATION SYSTEM (4 DIMENSIONS) ============================
# Note: Actual prompts are now in evaluation_prompts.py
# This section only contains the evaluation execution logic


def evaluate_single_dimension(dimension_name: str, prompt_template: str, 
                             user_query: str, profile_data: Dict[str, Any],
                             api_key: str = None) -> Dict[str, Any]:
    """
    评估单个维度（用于并行处理）- 增强错误处理，避免print崩溃
    添加超时控制，确保单次评估不超过 25 秒
    
    Args:
        dimension_name: 维度名称
        prompt_template: Prompt 模板（来自evaluation_prompts.py）
        user_query: 用户查询
        profile_data: 候选人档案数据（已格式化）
        api_key: LLM API key
    
    Returns:
        {"dimension": str, "score": int, "explanation": str}
    """
    import time
    from concurrent.futures import TimeoutError as FutureTimeoutError
    
    def _call_llm():
        """内部函数：执行 LLM 调用"""
        # 填充 prompt - profile_data已经包含所有必需字段
        prompt = prompt_template.format(**profile_data)
        
        llm_instance = llm.get_llm("evaluate", temperature=0.1, api_key=api_key)
        
        # 使用schemas中定义的LLMSingleDimensionEval
        response = llm.safe_structured(llm_instance, prompt, schemas.LLMSingleDimensionEval)
        
        if response and hasattr(response, 'score') and hasattr(response, 'explanation'):
            return {
                "dimension": dimension_name,
                "score": response.score,
                "explanation": response.explanation
            }
        else:
            return {"dimension": dimension_name, "score": 3, "explanation": "LLM response format error"}
    
    try:
        # 使用 ThreadPoolExecutor 包装 LLM 调用，设置 25 秒超时（留出 5 秒缓冲）
        start_time = time.time()
        max_timeout = 25
        
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_call_llm)
            try:
                result = future.result(timeout=max_timeout)
                elapsed = time.time() - start_time
                if elapsed > 20:  # 如果超过 20 秒，记录警告
                    safe_print(f"[evaluate_dimension] {dimension_name} took {elapsed:.1f}s (slow but completed)")
                return result
            except FutureTimeoutError:
                elapsed = time.time() - start_time
                safe_print(f"[evaluate_dimension] {dimension_name} timeout after {elapsed:.1f}s (max {max_timeout}s), using default score 3")
                # 尝试取消 future
                try:
                    future.cancel()
                except:
                    pass
                return {"dimension": dimension_name, "score": 3, "explanation": f"Evaluation timeout ({elapsed:.1f}s > {max_timeout}s)"}
                
    except (TimeoutError, OSError, IOError, ConnectionError) as e:
        # Network/timeout errors - return default score without detailed logging
        safe_print(f"[evaluate_dimension] {dimension_name} network error (using default score 3): {type(e).__name__}")
        return {"dimension": dimension_name, "score": 3, "explanation": f"Network timeout"}
        
    except Exception as e:
        safe_print(f"[evaluate_dimension] LLM call failed for {dimension_name}: {str(e)}")
        return {"dimension": dimension_name, "score": 3, "explanation": f"LLM error: {type(e).__name__}"}


def evaluate_profile_4d_parallel(profile: AuthorProfile, 
                                 enhanced_profile: Optional[schemas.EnhancedAuthorProfile],
                                 user_query: str,
                                 api_key: str = None) -> schemas.EvaluationResult:
    """
    新的4维度并行评估系统 - 增强错误处理
    
    维度：
    1. Topic Match（主题匹配度）- 5分
    2. Venue Fit & Evidence Strength（发表质量与领域适配度）- 5分
    3. Recency & Momentum（研究新鲜度与持续性，IN QUERY DOMAIN）- 5分
    4. Role Leadership Fit（领导/主导贡献度）- 5分
    总分：20分
    """
    safe_print(f"\n📊 Starting 4-dimension parallel evaluation for {profile.name}...")
    
    # 准备完整的档案数据用于evaluation_prompts.py
    import datetime
    current_year = datetime.datetime.utcnow().year
    
    # 提取研究兴趣和关键词
    research_interests_list = []
    research_keywords_list = []
    research_focus_list = []
    
    if enhanced_profile:
        if enhanced_profile.research_interests:
            research_interests_list = [f"{ri.name}: {ri.description}" for ri in enhanced_profile.research_interests]
        # 从enhanced_profile提取其他研究关键词
        if hasattr(enhanced_profile, 'research_keywords'):
            research_keywords_list = enhanced_profile.research_keywords or []
    
    # Fallback to basic profile
    if not research_interests_list and profile.interests:
        research_interests_list = profile.interests
    
    # 构建representative papers列表（格式化为evaluation_prompts期望的格式）
    representative_papers_list = []
    publications_list = profile.selected_publications or []
    for pub in publications_list[:10]:  # 取前10篇代表作
        paper_dict = {
            'Title': pub.get('title', ''),
            'Venue': pub.get('venue', ''),
            'Year': pub.get('year'),
            'Type': 'Conference Paper' if pub.get('venue') else 'Preprint'
        }
        representative_papers_list.append(paper_dict)
    
    # Publication Overview（简化格式）
    publication_overview_list = [
        f"{pub.get('title', '')} ({pub.get('venue', '')}, {pub.get('year', '?')})"
        for pub in publications_list[:15]
    ]
    
    # Top-tier hits（近24个月的高质量论文）
    recent_pubs = [p for p in publications_list if isinstance(p.get('year'), int) and p.get('year') >= current_year - 2]
    top_tier_hits_list = [
        f"{pub.get('title', '')} - {pub.get('venue', '')} {pub.get('year', '')}"
        for pub in recent_pubs
    ]
    
    # Highlights（从notable achievements提取）
    highlights_list = profile.notable_achievements[:10] if profile.notable_achievements else []
    
    # Service/Talks（如果有的话）
    service_talks_list = []
    
    # Honors/Grants（从notable achievements提取）
    honors_grants_list = profile.notable_achievements[:10] if profile.notable_achievements else []
    
    # 创建一个mock candidate_overview对象用于format_candidate_info_for_eval
    class MockCandidateOverview:
        def __init__(self):
            self.name = profile.name
            self.research_keywords = research_keywords_list or research_interests_list
            self.research_focus = research_focus_list or research_interests_list
            self.representative_papers = representative_papers_list
            self.publication_overview = publication_overview_list
            self.top_tier_hits = top_tier_hits_list
            self.highlights = highlights_list
            self.service_talks = service_talks_list
            self.honors_grants = honors_grants_list
    
    mock_overview = MockCandidateOverview()
    
    # 使用evaluation_prompts.py中的helper函数格式化数据
    profile_data = format_candidate_info_for_eval(
        candidate_overview=mock_overview,
        user_query=user_query,
        research_field="",  # Will be filled from spec if available
        query_venues=None,
        author_priority=None
    )
    
    # 定义4个评估任务 - 使用新的独立prompt
    evaluation_tasks = [
        ("Topic Match", PROMPT_TOPIC_MATCH),
        ("Venue Fit & Evidence Strength", PROMPT_VENUE_FIT),
        ("Recency & Momentum", PROMPT_RECENCY_MOMENTUM),
        ("Role Leadership Fit", PROMPT_ROLE_LEADERSHIP)
    ]
    
    # ✅ 激进并发控制：4维评估可以开更多
    from .dynamic_concurrency import get_llm_workers
    max_workers = get_llm_workers(4)  # 移除上限，完全依赖动态并发控制
    
    # Parallel evaluation with improved timeout protection
    # 使用轮询方式替代 as_completed，避免某个 future 卡住导致整个循环阻塞
    results = []
    import time
    start_time = time.time()
    max_total_timeout = 120  # 总超时时间：120秒（4个维度，每个最多30秒）
    per_dimension_timeout = 30  # 每个维度的超时时间：30秒（正常情况下LLM应在2-5秒内响应，30秒足够包括重试）
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 记录每个 future 的提交时间
        future_to_dim = {}
        future_start_times = {}
        
        for dim_name, prompt in evaluation_tasks:
            future = executor.submit(
                evaluate_single_dimension,
                dim_name,
                prompt,
                user_query,
                profile_data,
                api_key
            )
            future_to_dim[future] = dim_name
            future_start_times[future] = time.time()
        
        safe_print(f"  📊 Starting evaluation for {len(evaluation_tasks)} dimensions...")
        
        # 使用轮询方式检查 future 状态
        completed_dims = set()
        remaining_futures = set(future_to_dim.keys())
        
        while remaining_futures and (time.time() - start_time) < max_total_timeout:
            futures_to_remove = []
            
            for future in list(remaining_futures):
                dim_name = future_to_dim[future]
                
                if dim_name in completed_dims:
                    futures_to_remove.append(future)
                    continue
                
                # 检查单个维度的超时（从该维度开始时间计算）
                dimension_elapsed = time.time() - future_start_times[future]
                
                if future.done():
                    # Future 已完成，获取结果
                    try:
                        result = future.result(timeout=1)
                        results.append(result)
                        completed_dims.add(dim_name)
                        futures_to_remove.append(future)
                        safe_print(f"  ✓ {dim_name}: {result['score']}/5")
                    except TimeoutError:
                        safe_print(f"  ⚠️  Timeout getting result for {dim_name}, using default score 3")
                        results.append({"dimension": dim_name, "score": 3, "explanation": "Evaluation timeout"})
                        completed_dims.add(dim_name)
                        futures_to_remove.append(future)
                    except Exception as e:
                        safe_print(f"  ⚠️  Error getting result for {dim_name}: {str(e)[:50]}")
                        results.append({"dimension": dim_name, "score": 3, "explanation": f"Error: {str(e)[:30]}"})
                        completed_dims.add(dim_name)
                        futures_to_remove.append(future)
                elif dimension_elapsed > per_dimension_timeout:
                    # 单个维度超时，使用默认值
                    safe_print(f"  ⚠️  Timeout for {dim_name} ({dimension_elapsed:.1f}s), using default score 3")
                    results.append({"dimension": dim_name, "score": 3, "explanation": "Evaluation timeout"})
                    completed_dims.add(dim_name)
                    futures_to_remove.append(future)
                    try:
                        future.cancel()
                    except:
                        pass
            
            # 移除已处理的 future
            for future in futures_to_remove:
                remaining_futures.discard(future)
            
            # 如果还有未完成的，等待一小段时间再检查
            if remaining_futures:
                time.sleep(0.5)
            else:
                break
        
        # 处理最终超时未完成的维度
        for future in remaining_futures:
            dim_name = future_to_dim[future]
            if dim_name not in completed_dims:
                safe_print(f"  ⚠️  Final timeout for {dim_name}, using default score 3")
                results.append({"dimension": dim_name, "score": 3, "explanation": "Overall evaluation timeout"})
                try:
                    future.cancel()
                except:
                    pass
    
    # 按原始顺序排序 - 使用新的维度名称
    dimension_order = ["Topic Match", "Venue Fit & Evidence Strength", "Recency & Momentum", "Role Leadership Fit"]
    results_sorted = sorted(results, key=lambda x: dimension_order.index(x['dimension']) if x['dimension'] in dimension_order else 999)
    
    # 构建 EvaluationItem 列表
    items = [
        schemas.EvaluationItem(
            dimension=r['dimension'],
            score=r['score'],
            justification=r['explanation']
        )
        for r in results_sorted
    ]
    
    # 计算总分和雷达图
    total_score = sum(item.score for item in items)
    
    # ✅ 修复：使用前端期望的键名映射
    dimension_key_mapping = {
        "Topic Match": "topic_match",
        "Venue Fit & Evidence Strength": "venue_fit",
        "Recency & Momentum": "recency_momentum",
        "Role Leadership Fit": "leadership"
    }
    
    radar = {
        dimension_key_mapping.get(item.dimension, item.dimension.lower().replace(" ", "_")): item.score 
        for item in items
    }
    
    # ✅ 新增：构建详细分数字典（包含解释）
    details = {
        item.dimension: f"{item.score}/5 - {item.justification}"
        for item in items
    }
    
    safe_print(f"✅ Evaluation complete. Total Score: {total_score}/20")    
    return schemas.EvaluationResult(
        items=items,
        total_score=total_score,
        radar=radar,
        details=details  # ✅ 添加详细解释
    )

# ============================ CANDIDATE OVERVIEW BUILDER (NEW 9-FIELD STRUCTURE) ============================

def build_candidate_overview_from_enhanced(
    enhanced_profile: schemas.EnhancedAuthorProfile,
    eval_result: schemas.EvaluationResult,
    trigger_paper_title: str = "",
    trigger_paper_url: str = "",
    trigger_paper_venue: str = "",
    trigger_paper_score: float = 0.0,
    trigger_paper_explanation: str = ""
) -> schemas.CandidateOverview:
    """
    Build CandidateOverview directly from EnhancedAuthorProfile (9-field structure)
    
    This is the NEW unified structure that matches the desired output format.
    """
    safe_print(f"[Build Overview] Creating 9-field CandidateOverview from EnhancedAuthorProfile...")
    
    # Extract name from introduction
    candidate_name = enhanced_profile.introduction.name or "Unknown"
    
    # 从候选人的 all_publications 中找到 trigger paper，使用它的评分更新 trigger_paper_score
    # 这样可以确保前端显示的分数和计算 paper_score 使用的分数一致
    final_trigger_paper_score = trigger_paper_score
    final_trigger_paper_explanation = trigger_paper_explanation
    
    if trigger_paper_url and enhanced_profile.selected_research:
        all_pubs = enhanced_profile.selected_research.all_publications or []
        for pub in all_pubs:
            pub_url = getattr(pub, "url", "")
            if pub_url and pub_url == trigger_paper_url:
                pub_score = getattr(pub, "relevance_score", 0)
                if pub_score > 0:
                    final_trigger_paper_score = pub_score
                    final_trigger_paper_explanation = getattr(pub, "relevance_explanation", trigger_paper_explanation)
                    safe_print(f"[Build Overview] ✅ Updated trigger_paper_score from {trigger_paper_score} to {final_trigger_paper_score} (from candidate's all_publications)")
                break
    
    # Compute trigger paper author position (index/label/total)
    # ✅ 优先使用 PublicationInfo 中预先计算的位置信息，如果没有则回退到运行时计算
    trigger_paper_position = None
    trigger_paper_position_index = None
    trigger_paper_total_authors = None
    try:
        if trigger_paper_url and enhanced_profile.selected_research:
            all_pubs = enhanced_profile.selected_research.all_publications or []
            # Find the trigger publication by URL
            for pub in all_pubs:
                pub_url = getattr(pub, "url", "")
                if pub_url and pub_url == trigger_paper_url:
                    # ✅ 优先使用预先计算的位置信息
                    pre_computed_pos = getattr(pub, "candidate_author_position", None)
                    pre_computed_label = getattr(pub, "candidate_author_position_label", None)
                    pre_computed_total = getattr(pub, "total_authors", None)
                    
                    if pre_computed_pos is not None:
                        trigger_paper_position_index = pre_computed_pos
                        trigger_paper_position = pre_computed_label
                        trigger_paper_total_authors = pre_computed_total
                        safe_print(f"[Build Overview] ✅ Using pre-computed position: {trigger_paper_position}")
                    else:
                        # 回退到运行时计算
                        authors = getattr(pub, "authors", []) or []
                        if authors:
                            pos_idx, pos_label, total = compute_author_position(candidate_name, authors)
                            if pos_idx is not None:
                                trigger_paper_position_index = pos_idx
                                trigger_paper_position = pos_label
                                trigger_paper_total_authors = total
                            else:
                                trigger_paper_total_authors = len(authors)
                        else:
                            trigger_paper_total_authors = 0
                    break
    except Exception as _pos_err:
        pass

    # Build CandidateOverview with 9-field structure
    overview = schemas.CandidateOverview(
        name=candidate_name,
        # 1. Introduction (General Background)
        introduction=enhanced_profile.introduction,
        # 2. Research Interests (with descriptions)
        research_interests=enhanced_profile.research_interests,
        # 3. Selected Research (publications grouped by category)
        selected_research=enhanced_profile.selected_research,
        # 4. Awards & Honors
        awards=enhanced_profile.awards,
        # 5. Professional Services
        professional_services=enhanced_profile.professional_services,
        # 6. Career & Education History
        career_education_history=enhanced_profile.career_education_history,
        # 7. Industrial Experience
        industrial_experience=enhanced_profile.industrial_experience,
        # 8. Contact
        contact=enhanced_profile.contact,
        # Metadata
        trigger_paper_title=trigger_paper_title,
        trigger_paper_url=trigger_paper_url,
        trigger_paper_venue=trigger_paper_venue,
        trigger_paper_score=final_trigger_paper_score,  # ✅ 使用更新后的分数
        trigger_paper_explanation=final_trigger_paper_explanation,  # ✅ 使用更新后的解释
        trigger_paper_position=trigger_paper_position,
        trigger_paper_position_index=trigger_paper_position_index,
        trigger_paper_total_authors=trigger_paper_total_authors,
        radar=eval_result.radar,
        total_score=eval_result.total_score,
        detailed_scores=eval_result.details
    )
    # Pre-populate candidate category from LLM-determined role if available
    if getattr(enhanced_profile, "current_role", None):
        overview.candidate_category = enhanced_profile.current_role.category or "Unknown"
    safe_print(f"[Overview] {candidate_name}: Score {eval_result.total_score}/20, {len(enhanced_profile.research_interests)} interests, {len(enhanced_profile.selected_research.all_publications)} pubs")
    return overview
# ============================ OLD BUILDERS (DEPRECATED - TO BE REMOVED) ============================
def build_candidate_overview_lightweight(profile: AuthorProfile, eval_result: schemas.EvaluationResult, top_pubs: List[Dict[str, Any]], 
                                       trigger_paper_title: str = None, trigger_paper_url: str = None, trigger_paper_venue: str = None) -> schemas.CandidateOverview:
    """构建轻量级候选人概览 - 借鉴Targeted Search的简化模式，避免复杂LLM提取"""
    print(f"[Lightweight Mode] Building candidate overview for {profile.name}")
    
    # 基础信息（不依赖LLM）
    profiles_display: Dict[str, str] = {}
    if profile.homepage_url:
        profiles_display["Homepage"] = profile.homepage_url
    if 'scholar' in profile.platforms:
        profiles_display["Google Scholar"] = profile.platforms['scholar']
    if 'twitter' in profile.platforms:
        profiles_display["X (Twitter)"] = profile.platforms['twitter']
    if 'openreview' in profile.platforms:
        profiles_display["OpenReview"] = profile.platforms['openreview']
    if 'linkedin' in profile.platforms:
        profiles_display["LinkedIn"] = profile.platforms['linkedin']
    if 'github' in profile.platforms:
        profiles_display["GitHub"] = profile.platforms['github']

    # 简化的论文信息提取
    publication_overview_list = []
    rep_papers: List[schemas.RepresentativePaper] = []
    if top_pubs:
        publication_overview_list = [
            p.get('title', '').strip() for p in top_pubs[:5] 
            if isinstance(p, dict) and p.get('title', '').strip()
        ]
        # 简化代表作
        for p in top_pubs[:3]:
            if isinstance(p, dict) and p.get('title'):
                rep_papers.append(schemas.RepresentativePaper(
                    title=p.get('title',''),
                    venue=p.get('venue',''),
                    year=p.get('year'),
                    type="Conference Paper",  # 简化分类
                    links=p.get('url','')
                ))
    
    # 使用已有的基础信息
    research_focus = profile.interests[:6] if profile.interests else []
    research_keywords = profile.interests[:8] if profile.interests else []
    honors_list = list(profile.notable_achievements[:3]) if profile.notable_achievements else []
    
    # 简化的高光信息
    highlights = []
    if profile.social_impact:
        highlights.append(f"Impact: {profile.social_impact}")
    if profile.notable_achievements:
        highlights.extend(profile.notable_achievements[:2])
    
    # 构建轻量级概览
    overview = schemas.CandidateOverview(
        name=profile.name,
        email=profile.emails[0] if profile.emails else "",
        current_role_affiliation=profile.affiliation_current or "",
        current_status="",  # 轻量模式不提取详细状态
        research_keywords=research_keywords,
        research_focus=research_focus,
        profiles=profiles_display,
        publication_overview=publication_overview_list,
        top_tier_hits=[f"{p.get('venue', 'arXiv')} {p.get('year','')}" for p in top_pubs[:5]],
        honors_grants=honors_list,
        service_talks=[],  # 轻量模式暂不提取
        open_source_projects=["GitHub projects available" if 'github' in profile.platforms else ""],
        representative_papers=rep_papers,
        trigger_paper_title=trigger_paper_title or "",
        trigger_paper_url=trigger_paper_url or "",
        trigger_paper_venue=trigger_paper_venue or "",
        highlights=highlights,
        radar=eval_result.radar,
        total_score=eval_result.total_score,
        detailed_scores=eval_result.details
    )
    
    print(f"[Lightweight Mode] ✅ Successfully built overview with basic fields")
    return overview

# ============================ ENHANCED PROFILE GENERATION ============================

def filter_survey_and_position_papers(
    publications: List[Dict[str, Any]], 
    api_key: str = None,
    use_llm_verification: bool = False
) -> Tuple[List[Dict[str, Any]], int]:
    """
    过滤掉 survey 和 position paper
    策略：
    1. 基于标题关键词的快速过滤（高效，可能误判）
    2. 可选：使用 LLM 对可疑论文进行二次验证（准确但成本高）
    Args:
        publications: 论文列表
        api_key: LLM API key（用于二次验证，可选）
        use_llm_verification: 是否使用 LLM 验证可疑论文（默认 False，仅关键词过滤）
        
    Returns:
        Tuple[过滤后的论文列表, 被过滤的论文数量]
    """
    import re
    # Survey/Review/Position Paper 关键词模式
    strong_patterns = [
        r'\b(survey|review|overview|position paper|position statement)\b',
        r'\ba survey of\b',
        r'\ba review of\b',
        r'\ban overview of\b',
        r'\bsurvey on\b',
        r'\breview on\b',
        r'\bposition paper\b',
        r'\bposition statement\b',
        r'\bstate of the art\b',
        r'\bstate-of-the-art\b',
        r'\bsota\b',
        r'\btaxonomy\b',
        r'\bcomprehensive survey\b',
        r'\bcomprehensive review\b',
        r'\bsystematic review\b',
        r'\bsystematic survey\b',
        r'\bliterature review\b',
        r'\bliterature survey\b',
    ]
    
    # 弱信号（可能是 survey/position paper，需要结合其他信息）
    weak_patterns = [
        r'\btowards\b',  # "Towards X" 可能是 position paper
        r'\bperspective\b',
        r'\bperspectives\b',
        r'\bchallenges?\b.*\b(future|ahead)\b',
        r'\bfuture directions?\b',
        r'\bdirections?\s+(for|in)\b',
    ]
    
    filtered_pubs = []
    filtered_count = 0
    suspicious_papers = []  # 可疑论文（弱信号匹配）
    
    for pub in publications:
        title = pub.get('title', '').strip()
        if not title:
            # 没有标题的论文保留（可能是数据不完整）
            filtered_pubs.append(pub)
            continue
        
        title_lower = title.lower()
        
        # 检查强信号
        is_survey_or_position = False
        for pattern in strong_patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                is_survey_or_position = True
                filtered_count += 1
                print(f"[Paper Filter] ❌ Filtered (strong signal): {title[:60]}...")
                break
        
        if is_survey_or_position:
            continue
        
        # 检查弱信号
        has_weak_signal = False
        for pattern in weak_patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                has_weak_signal = True
                break
        
        if has_weak_signal:
            # 弱信号论文：如果有 abstract，进一步检查
            abstract = pub.get('abstract', '')
            if abstract and len(abstract) > 100:
                # 检查 abstract 中是否包含 survey 相关词汇
                abstract_lower = abstract.lower()
                abstract_survey_keywords = [
                    'survey', 'review', 'overview', 'comprehensive', 
                    'systematic', 'taxonomy', 'literature review',
                    'position paper', 'state of the art'
                ]
                abstract_has_survey = any(kw in abstract_lower for kw in abstract_survey_keywords)
                
                if abstract_has_survey:
                    # Abstract 也包含 survey 关键词，很可能是 survey paper
                    filtered_count += 1
                    print(f"[Paper Filter] ❌ Filtered (weak signal + abstract): {title[:60]}...")
                    continue
                else:
                    # 弱信号但 abstract 不明确，标记为可疑
                    suspicious_papers.append(pub)
            else:
                # 没有 abstract 或 abstract 太短，标记为可疑
                suspicious_papers.append(pub)
        else:
            # 没有弱信号，保留
            filtered_pubs.append(pub)
    
    # 可选：使用 LLM 验证可疑论文
    if use_llm_verification and api_key and suspicious_papers:
        print(f"[Paper Filter] LLM verifying {len(suspicious_papers)} suspicious papers...")
        try:
            from . import llm
            llm_client = llm.get_llm("filter", temperature=0.1, api_key=api_key)
            
            for pub in suspicious_papers:
                title = pub.get('title', '')
                abstract = pub.get('abstract', '')[:500]  # 限制长度
                
                prompt = f"""Determine if this paper is a survey, review, or position paper.
                Title: {title}
                Abstract: {abstract[:500] if abstract else "No abstract available"}
                Please classify this paper:
                - If it is a survey/review/overview/position paper, return "survey"
                - If it is a research paper, return "research"
                Return only one word: "survey" or "research"."""
                # 使用统一的 invoke + safe_get 将返回值规范为纯文本，避免内部对字符串调用 .content 导致错误
                resp = llm_client.invoke(prompt)
                response = llm.safe_get(resp, "content", "") or llm.safe_get(resp, "text", "") or str(resp)
                if response and 'survey' in response.lower():
                    filtered_count += 1
                    print(f"[Paper Filter] ❌ Filtered (LLM verified): {title[:60]}...")
                else:
                    filtered_pubs.append(pub)
                    print(f"[Paper Filter] ✅ Kept (LLM verified): {title[:60]}...")
        except Exception as e:
            print(f"[Paper Filter] ⚠️ LLM verification failed: {e}, keeping suspicious papers")
            filtered_pubs.extend(suspicious_papers)
    else:
        # 不使用 LLM 验证，保留可疑论文（保守策略）
        filtered_pubs.extend(suspicious_papers)
        if suspicious_papers:
            print(f"[Paper Filter] ℹ️ Kept {len(suspicious_papers)} suspicious papers (no LLM verification)")
    return filtered_pubs, filtered_count

def filter_arxiv_papers(
    publications: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], int]:
    """
    过滤掉 arXiv（corr）收录的论文
    Args:
        publications: 论文列表   
    Returns:
        Tuple[过滤后的论文列表, 被过滤的论文数量]
    """
    filtered_pubs = []
    filtered_count = 0
    # arXiv 相关的关键词模式
    arxiv_venue_keywords = [
        'arxiv',
        'arxiv.org',
        'arxiv preprint',
        'corr'
    ]
    # arXiv URL 模式
    arxiv_url_patterns = [
        'arxiv.org/abs/',
        'arxiv.org/pdf/',
        'arxiv.org/e-print/'
    ]
    for pub in publications:
        # 检查是否为 trigger paper（trigger paper 不应该被过滤）
        if pub.get('_is_trigger', False):
            filtered_pubs.append(pub)
            continue  
        # 检查 venue 字段
        venue = pub.get('venue', '').lower().strip()
        is_arxiv = False
        if venue:
            for keyword in arxiv_venue_keywords:
                if keyword in venue:
                    is_arxiv = True
                    break
        # 检查 URL 字段
        if not is_arxiv:
            pub_url = pub.get('url', '') or pub.get('link', '') or pub.get('arxiv_url', '')
            if pub_url:
                pub_url_lower = pub_url.lower()
                for pattern in arxiv_url_patterns:
                    if pattern in pub_url_lower:
                        is_arxiv = True
                        break
        if is_arxiv:
            filtered_count += 1
            title = pub.get('title', '')[:60] if pub.get('title') else 'Unknown'
            print(f"[Paper Filter] Filtered arXiv paper: {title}...")
        else:
            filtered_pubs.append(pub)
    return filtered_pubs, filtered_count
def collect_all_publications_for_interests(
    profile: Optional[AuthorProfile], 
    openreview_data: Dict[str, Any], 
    nine_dim_data: Dict[str, Any] = None,
    user_query: str = None,
    api_key: str = None,
    exclude_paper_urls: set = None,
    trigger_paper: Dict[str, Any] = None,
    s2_author_id: str = None,
    author_name: str = None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Stage 3: 收集所有论文用于生成研究方向（多源合并 + 去重 + 评分）
    论文获取逻辑（按优先级）：
    1. Semantic Scholar Author API: 通过 s2_author_id 获取全量论文（主要数据源）
    2. OpenReview: 补充论文（有 abstract，最可信）
    3. Trigger Paper: 触发搜索的论文（强制包含）
    
    最后进行去重、API 补充元数据、LLM 评分
    
    Args:
        profile: AuthorProfile（包含 Homepage + Google Scholar 数据）
        openreview_data: OpenReview 原始数据
        nine_dim_data: 九维度提取的数据（未使用，保留参数兼容性）
        user_query: 用户查询（用于论文评分，可选）
        api_key: LLM API key（用于论文评分，可选）
        exclude_paper_urls: 要排除的论文URL集合（避免与Reference Papers重复）
        trigger_paper: 触发搜索的论文（Dict包含title, url, venue等），会被强制包含
        s2_author_id: Semantic Scholar authorId，用于获取作者的全量论文
        author_name: 候选人名字，用于名字匹配计算作者位置
        
    Returns:
        Tuple[所有论文列表（已去重）, 高分论文列表(6-8分)]
    """
    from datetime import datetime
    from .utils import normalize_url
    
    if exclude_paper_urls is None:
        exclude_paper_urls = set()
    
    all_pubs = []
    seen_titles = set()
    excluded_count = 0
    
    # 🎯 优先添加 trigger paper（如果提供）
    if trigger_paper and trigger_paper.get('title'):
        trigger_title = trigger_paper.get('title', '').lower().strip()
        if trigger_title:
            print(f"[collect_pubs] 🎯 Adding trigger paper: {trigger_paper.get('title', '')[:60]}...")
            trigger_pub = {
                'title': trigger_paper.get('title', ''),
                'authors': trigger_paper.get('authors', []),
                'venue': trigger_paper.get('venue', ''),
                'year': trigger_paper.get('year'),
                'url': trigger_paper.get('url', ''),
                'abstract': trigger_paper.get('abstract', ''),
                'citationCount': trigger_paper.get('citationCount', 0),
                # 预先写入作者位置信息，避免后续阶段重新用名字匹配
                'candidate_author_position': trigger_paper.get('candidate_author_position'),
                'total_authors': trigger_paper.get('total_authors'),
                # 传递特殊角色标记（从 PDF 检测得到）
                'is_corresponding_author': trigger_paper.get('is_corresponding_author'),
                'is_cofirst_author': trigger_paper.get('is_cofirst_author'),
                '_source': 'trigger',
                '_is_trigger': True  # 标记为trigger paper
            }
            # ✅ FIX: 如果 trigger paper 已经有评分，保留它（避免重新评分导致不一致）
            if trigger_paper.get('score'):
                trigger_pub['relevance_score'] = trigger_paper.get('score')
                trigger_pub['relevance_explanation'] = trigger_paper.get('explanation', '')
                print(f"[collect_pubs] Preserving trigger paper score: {trigger_paper.get('score')}/8")
            if trigger_paper.get('relevance_score') is not None:
                trigger_pub['relevance_score'] = trigger_paper.get('relevance_score')
            if trigger_paper.get('quality_score') is not None:
                trigger_pub['quality_score'] = trigger_paper.get('quality_score')
            if trigger_paper.get('relevance_dimensions'):
                trigger_pub['relevance_dimensions'] = trigger_paper.get('relevance_dimensions')
            if trigger_paper.get('quality_dimensions'):
                trigger_pub['quality_dimensions'] = trigger_paper.get('quality_dimensions')
            if trigger_paper.get('quality_score') is not None:
                print(f"[collect_pubs] Preserving trigger paper quality score: {trigger_paper.get('quality_score')}/8")
            all_pubs.append(trigger_pub)
            seen_titles.add(trigger_title)
    
    # 1. Semantic Scholar Author API：通过 s2_author_id 获取全量论文（主要数据源）
    s2_count = 0
    s2_failed = False
    if s2_author_id:
        print(f"\n[collect_pubs] Fetching papers via Semantic Scholar Author API (ID: {s2_author_id})")
        try:
            s2_client = SemanticScholarSearchClient()
            s2_papers = s2_client.fetch_author_papers(s2_author_id, limit=1000)
            
            if s2_papers:
                s2_pubs_dict = s2_client.convert_papers_to_dict_list(s2_papers, self_author_id=s2_author_id)
                
                for pub in s2_pubs_dict:
                    title = pub.get('title', '').lower().strip()
                    # 检查论文URL是否在排除列表中
                    pub_url = pub.get('url', '') or pub.get('link', '')
                    if pub_url:
                        normalized_pub_url = normalize_url(pub_url)
                        if normalized_pub_url in exclude_paper_urls:
                            excluded_count += 1
                            continue  # 跳过已在Reference Papers中的论文
                    
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_pubs.append(pub)
                        s2_count += 1
                
                print(f"  Semantic Scholar: {s2_count} 篇论文（全量获取）")
            else:
                print(f"  Semantic Scholar: No papers found for author ID {s2_author_id}")
                s2_failed = True
        except Exception as e:
            print(f"  Semantic Scholar fetch failed: {e}")
            import traceback
            traceback.print_exc()
            s2_failed = True
    else:
        print(f"\n[collect_pubs] No S2 author ID provided, skipping Semantic Scholar fetch")
        s2_failed = True
    # 2. Fallback: 如果 S2 获取失败或论文数过少，使用 OpenReview + Google Scholar 作为备用数据源
    if s2_failed or s2_count == 0:
        print(f"\n[collect_pubs] S2 failed or empty, falling back to OpenReview + Google Scholar...")
        # 2.1 OpenReview 论文（全部，有 abstract）
        openreview_pubs = openreview_data.get('publications', [])
        openreview_count = 0
        for pub in openreview_pubs:
            title = pub.get('title', '').lower().strip()
            pub_url = pub.get('url', '') or pub.get('link', '')
            if pub_url:
                normalized_pub_url = normalize_url(pub_url)
                if normalized_pub_url in exclude_paper_urls:
                    excluded_count += 1
                    continue
            if title and title not in seen_titles:
                seen_titles.add(title)
                # 如果没有预计算位置，使用名字匹配 fallback
                if pub.get('candidate_author_position') is None and author_name:
                    from .utils import compute_author_position
                    authors = pub.get('authors', [])
                    if authors:
                        position_index, position_label, total_authors = compute_author_position(author_name, authors)
                        if position_index is not None:
                            pub['candidate_author_position'] = position_index
                            pub['total_authors'] = total_authors
                all_pubs.append(pub)
                openreview_count += 1
        if openreview_count > 0:
            print(f"  OpenReview: {openreview_count} 篇论文（带 abstract）")
        # 2.2 Google Scholar 论文（最近 3 年）
        scholar_count = 0
        if profile and hasattr(profile, '_scholar_metadata'):
            from datetime import datetime
            scholar_metadata = getattr(profile, '_scholar_metadata', {})
            scholar_pubs = scholar_metadata.get('publications', [])
            if scholar_pubs:
                recent_year_threshold = datetime.now().year - 3
                for pub in scholar_pubs:
                    raw_year = pub.get('year')
                    try:
                        if isinstance(raw_year, int):
                            pub_year = raw_year
                        elif isinstance(raw_year, str):
                            pub_year = int(raw_year)
                        else:
                            pub_year = 0
                    except (TypeError, ValueError):
                        pub_year = 0
                    if pub_year and pub_year >= recent_year_threshold:
                        title = pub.get('title', '').lower().strip()
                        pub_url = pub.get('url', '') or pub.get('link', '')
                        if pub_url:
                            normalized_pub_url = normalize_url(pub_url)
                            if normalized_pub_url in exclude_paper_urls:
                                excluded_count += 1
                                continue
                        
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            # 如果没有预计算位置，使用名字匹配 fallback
                            if pub.get('candidate_author_position') is None and author_name:
                                from .utils import compute_author_position
                                authors = pub.get('authors', [])
                                if authors:
                                    position_index, position_label, total_authors = compute_author_position(author_name, authors)
                                    if position_index is not None:
                                        pub['candidate_author_position'] = position_index
                                        pub['total_authors'] = total_authors
                            all_pubs.append(pub)
                            scholar_count += 1
                if scholar_count > 0:
                    print(f"  Google Scholar: {scholar_count} 篇论文（最近 3 年）")
    
    if excluded_count > 0:
        print(f"  Excluded {excluded_count} papers already in Reference Papers")
    
    # 过滤掉 survey 和 position paper
    print(f"\n[Paper Filter] Filtering survey and position papers from {len(all_pubs)} papers...")
    all_pubs, filtered_count = filter_survey_and_position_papers(
        all_pubs, 
        api_key=api_key,
        use_llm_verification=True #开启llm筛选
    )
    if filtered_count > 0:
        print(f"[Paper Filter] Filtered {filtered_count} survey/position papers, {len(all_pubs)} papers remaining")
    else:
        print(f"[Paper Filter] No survey/position papers detected, all {len(all_pubs)} papers kept")
    
    # 过滤掉 arXiv（corr）收录的论文
    print(f"\n[Paper Filter] Filtering arXiv papers from {len(all_pubs)} papers...")
    all_pubs, arxiv_filtered_count = filter_arxiv_papers(all_pubs)
    if arxiv_filtered_count > 0:
        print(f"[Paper Filter] Filtered {arxiv_filtered_count} arXiv papers, {len(all_pubs)} papers remaining")
    else:
        print(f"[Paper Filter] No arXiv papers detected, all {len(all_pubs)} papers kept")
    
    # Enrich publications
    try:
        all_pubs = enrich_publications_for_profile(all_pubs)
    except Exception as e:
        print(f"[Enrich] Error: {e}")
    
    # 获取完整信息后进行去重（使用智能去重策略）
    try:
        from .agents import deduplicate_papers
        print(f"\n[Paper Deduplication] Deduplicating {len(all_pubs)} papers after enrichment...")
        # 将 all_pubs 作为 all_serp 传入，传入空的 all_scored_papers
        deduplicated_serp, _ = deduplicate_papers(all_pubs, {})
        all_pubs = deduplicated_serp
        print(f"[Paper Deduplication] After deduplication: {len(all_pubs)} unique papers")
    except Exception as e:
        print(f"[Paper Deduplication] ⚠️ Deduplication failed, using original list: {e}")
        # 如果去重失败，继续使用原始列表
    
    # 论文评分（如果提供了 user_query）
    high_quality_pubs = []
    if user_query:
        try:
            high_quality_pubs = score_and_filter_publications(all_pubs, user_query, api_key)
            print(f"  Paper scoring is complete; {len(high_quality_pubs)} high-quality papers have been selected.")
        except Exception as e:
            print(f"  The paper was graded incorrectly (skip grading).: {e}")
    
    print(f"   Collected: {len(all_pubs)} papers ({len(high_quality_pubs)} high-quality)")
    return all_pubs, high_quality_pubs

def score_and_filter_publications(
    publications: List[Dict[str, Any]], 
    user_query: str, 
    api_key: str
) -> List[Dict[str, Any]]:
    """
    使用 LLM 对论文进行评分，筛选出 6-8 分的高质量论文
    
    Args:
        publications: 论文列表
        user_query: 用户查询
        api_key: LLM API key
        
    Returns:
        6-8 分的高质量论文列表（带评分信息）
        注意：返回 6 分及以上用于展示，但在 agents.py 中搜索新人才时会再过滤为 7 分以上
    """
    if not publications:
        return []
    
    # 评分前检查：统计有多少论文缺少摘要
    papers_without_abstract = []
    papers_with_abstract = []
    for pub in publications:
        abstract = pub.get('abstract', '')
        if not abstract or len(str(abstract)) < 50:
            papers_without_abstract.append(pub.get('title', 'Unknown')[:60])
        else:
            papers_with_abstract.append(pub)
    
    print(f"\n{'='*80}")
    print(f"[Score Publications] Start LLM scoring of {len(publications)} papers.")
    print(f"[Score Publications] User Query: {user_query}")
    print(f"{'='*80}\n")
    
    try:
        from . import search
        from . import llm as llm_module
        from .dynamic_concurrency import get_llm_workers
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        # 获取 LLM 实例
        llm_instance = llm_module.get_llm("score", temperature=0.1, api_key=api_key)
        
        # 并行评分
        max_workers = get_llm_workers(len(publications))
        
        scored_pubs = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有评分任务
            future_to_pub = {}
            for idx, pub in enumerate(publications, 1):
                title = pub.get('title', '')
                abstract = pub.get('abstract', '')
                introduction = pub.get('introduction', '')
                
                if not title:
                    continue
                
                existing_score = pub.get('relevance_score', 0)
                if existing_score > 0:
                    paper_type = "trigger paper" if pub.get('_is_trigger') else "paper"
                    print(f"[Score Publications] ⏭️  Skipping {paper_type} (already scored): {title} → {existing_score}/8")
                    if existing_score > 6:
                        scored_pubs.append(pub)
                    continue
                
                future = executor.submit(
                    search.score_paper_with_llm,
                    title,              # title
                    abstract,           # abstract
                    user_query,         # user_query
                    llm_instance,       # llm
                    introduction=introduction,  # introduction
                    pdf_url=pub.get("pdf_url", "") or pub.get("url", "")  # PDF URL for full PDF scoring
                )
                future_to_pub[future] = pub
            
            # 收集结果（30秒单个超时，60秒总超时）
            completed_count = 0
            timeout_count = 0
            error_count = 0
            try:
                for future in as_completed(future_to_pub, timeout=60):
                    pub = future_to_pub[future]
                    title = pub.get('title', '')[:50]
                    
                    try:
                        result = future.result(timeout=30)
                        score = result.get('score', 0)
                        explanation = result.get('explanation', '')
                        has_error = result.get('has_error', False)
                        
                        # 将评分信息添加到论文中（仅在评分不存在时，避免覆盖trigger paper的预评分）
                        if 'relevance_score' not in pub or pub.get('relevance_score', 0) == 0:
                            pub['relevance_score'] = score
                        if 'relevance_explanation' not in pub or not pub.get('relevance_explanation'):
                            pub['relevance_explanation'] = explanation
                        pub['has_scoring_error'] = has_error
                        
                        # 保存质量评分和维度信息 (修复: paper_quality_score默认为0, 10维度详情不显示)
                        if 'quality_score' not in pub or pub.get('quality_score', 0.0) == 0.0:
                            pub['quality_score'] = result.get('quality_score', 0.0)
                        if 'relevance_dimensions' not in pub or not pub.get('relevance_dimensions'):
                            pub['relevance_dimensions'] = result.get('relevance_dimensions', {})
                        if 'quality_dimensions' not in pub or not pub.get('quality_dimensions'):
                            pub['quality_dimensions'] = result.get('quality_dimensions', {})
                        
                        completed_count += 1
                        
                        # ✅ 区分正常评分和API错误
                        if has_error:
                            error_count += 1
                            print(f"[{completed_count}/{len(publications)}] ⚠️ API Error: {title} → {score}/8 (will retry later)")
                        else:
                            print(f"[{completed_count}/{len(publications)}] ✅ Grading completed: {title} → {score}/8")
                            
                            # 筛选 6-8 分的论文用于展示（排除错误论文）
                            # 注意：搜索新人才时会在 agents.py 中再过滤为 7 分以上
                            if score > 6:
                                scored_pubs.append(pub)
                        
                    except TimeoutError:
                        timeout_count += 1
                        print(f"[{completed_count + timeout_count}/{len(publications)}] ⏱️ Timeout: {title}")
                        pub['relevance_score'] = 0
                        pub['relevance_explanation'] = "Timeout"
                        
                    except Exception as e:
                        error_count += 1
                        print(f"[{completed_count + error_count}/{len(publications)}] ❌ Error: {title} - {e}")
                        pub['relevance_score'] = 0
                        pub['relevance_explanation'] = f"Error: {str(e)}"
                        
            except TimeoutError:
                print(f"[Score Publications] ⚠️ Overall timeout (60s), some papers were not graded.")
        
        # 按评分降序排序
        scored_pubs.sort(key=lambda x: x.get('relevance_score', 0), reverse=True)
        
        # 验证高质量论文的 authors 字段完整性
        papers_missing_authors = []
        for pub in scored_pubs:
            authors = pub.get('authors', [])
            # 检查 authors 是否为空或格式错误
            if not authors or (isinstance(authors, list) and len(authors) == 0):
                papers_missing_authors.append(pub.get('title', 'Unknown')[:60])
        
        print(f"\n{'='*80}")
        print(f"[Score Publications] Grading completed:")
        print(f"  - Total: {len(publications)} papers")
        print(f"  - Successfully graded: {completed_count - error_count} papers")
        print(f"  - High-quality (6-8 points): {len(scored_pubs)} papers")
        if len(scored_pubs) > 0:
            print(f"  - Scoring distribution:")
            for score in [8, 7, 6]:
                count = sum(1 for p in scored_pubs if p.get('relevance_score') == score)
                if count > 0:
                    print(f"    • {score} points: {count} papers")
        print(f"  - Timeout: {timeout_count} papers")
        print(f"  - API Errors (KeyError/AttributeError): {error_count} papers")
        if error_count > 0:
            print(f"    ⚠️  High error rate may indicate API rate limiting (reduce concurrency)")
        
        # ⚠️ 警告：高质量论文缺少 authors 字段
        if papers_missing_authors:
            print(f"\n⚠️  WARNING: {len(papers_missing_authors)}/{len(scored_pubs)} high-quality papers missing 'authors' field!")
            print(f"    This will cause paper_score = 0.0 for affected candidates.")
            print(f"    Papers with missing authors:")
            for title in papers_missing_authors[:3]:
                print(f"      - {title}...")
            if len(papers_missing_authors) > 3:
                print(f"      ... and {len(papers_missing_authors) - 3} more")
            print(f"    → Solution: Ensure enrichment step properly fills authors from Semantic Scholar/arXiv")
        print(f"{'='*80}\n")
        return scored_pubs
    except ImportError as e:
        print(f"[Score Publications] ❌ Import failed: {e}")
        return []
    except Exception as e:
        print(f"[Score Publications] ❌ Scoring error: {e}")
        import traceback
        traceback.print_exc()
        return []

def enrich_publications_for_profile(publications: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    增强论文信息，确保每篇论文都有 introduction、abstract、url
    策略：
    1. 检查缺失字段
    2. Semantic Scholar API 补充
    3. arXiv API 回退
    4. 如果都失败，设置默认值
    Args:
        publications: 论文列表
    Returns:
        增强后的论文列表（原地修改 + 返回）
    """
    if not publications:
        return publications
    print(f"[Enrich] Processing {len(publications)} papers (parallel mode)...")
    try:
        from .semantic_paper_search import SemanticScholarClient
        from .arxiv_fallback import ArxivFallbackClient
        from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
        from .dynamic_concurrency import get_extraction_workers
        import threading
        from . import config
        
        s2_client = SemanticScholarClient(api_key=config.SEMANTIC_SCHOLAR_API_KEY if config.SEMANTIC_SCHOLAR_API_KEY else None)
        arxiv_client = ArxivFallbackClient()
        
        # Thread-safe counters
        lock = threading.Lock()
        enriched_count = 0
        s2_success = 0
        arxiv_success = 0
        already_complete = 0
        failed_count = 0
        
        # Helper function to enrich a single paper
        def enrich_single_paper(pub_index_tuple):
            """Thread-safe enrichment of a single paper"""
            nonlocal enriched_count, s2_success, arxiv_success, failed_count
            
            i, pub = pub_index_tuple
            title = (pub.get('title') or '').strip()
            if not title:
                return False
            
            has_abstract = pub.get('abstract') and len(str(pub.get('abstract', ''))) > 50
            has_introduction = pub.get('introduction') and len(str(pub.get('introduction', ''))) > 20
            has_url = pub.get('url') and pub['url'].strip()
            has_authors = pub.get('authors') and (isinstance(pub.get('authors'), list) and len(pub.get('authors', [])) > 0)
            
            # Already complete, skip (only if all fields including authors are present)
            if has_abstract and has_introduction and has_url and has_authors:
                return True
            
            year_hint = str(pub.get('year', '')) if pub.get('year') else None
            venue_hint = pub.get('venue', '')
            
            # Try S2 first
            try:
                s2_data = s2_client.get_paper_full_details(
                    title=title,
                    year=year_hint,
                    venue=venue_hint,
                    min_match_score=0.6
                )
                
                if s2_data:
                    with lock:
                        if not has_abstract and s2_data.get('abstract'):
                            pub['abstract'] = s2_data['abstract']
                        if not has_introduction and s2_data.get('tldr'):
                            pub['introduction'] = s2_data['tldr']
                        if not has_url and s2_data.get('url'):
                            pub['url'] = s2_data['url']
                        if not pub.get('citation_count'):
                            pub['citation_count'] = s2_data.get('citation_count', 0)
                        if not pub.get('authors') and s2_data.get('authors'):
                            pub['authors'] = s2_data['authors']
                        
                        enriched_count += 1
                        s2_success += 1
                    return True
            except Exception:
                pass
            # S2 failed, try arXiv
            try:
                arxiv_data = arxiv_client.search_by_title(
                    title=title,
                    year=year_hint
                )
                
                if arxiv_data:
                    with lock:
                        if not has_abstract and arxiv_data.get('abstract'):
                            pub['abstract'] = arxiv_data['abstract']
                        if not has_introduction:
                            pub['introduction'] = ""  # arXiv has no TLDR
                        if not has_url and arxiv_data.get('arxiv_url'):
                            pub['url'] = arxiv_data['arxiv_url']
                        if not pub.get('authors') and arxiv_data.get('authors'):
                            pub['authors'] = arxiv_data['authors']
                        
                        enriched_count += 1
                        arxiv_success += 1
                    return True
            except Exception:
                pass
            # Both failed, set defaults
            with lock:
                failed_count += 1
                if not has_abstract:
                    pub['abstract'] = ""
                if not has_introduction:
                    pub['introduction'] = ""
                if not has_url:
                    pub['url'] = ""
            return False
        
        # Count already complete papers
        for pub in publications:
            title = (pub.get('title') or '').strip()
            if not title:
                continue
            has_abstract = pub.get('abstract') and len(str(pub.get('abstract', ''))) > 50
            has_introduction = pub.get('introduction') and len(str(pub.get('introduction', ''))) > 20
            has_url = pub.get('url') and pub['url'].strip()
            if has_abstract and has_introduction and has_url:
                already_complete += 1
        # Filter papers that need enrichment
        pubs_to_process = [
            (i, pub) for i, pub in enumerate(publications)
            if (pub.get('title') or '').strip() and not (
                pub.get('abstract') and len(str(pub.get('abstract', ''))) > 50 and
                pub.get('introduction') and len(str(pub.get('introduction', ''))) > 20 and
                pub.get('url') and pub['url'].strip() and
                pub.get('authors') and (isinstance(pub.get('authors'), list) and len(pub.get('authors', [])) > 0)
            )
        ]
        
        if not pubs_to_process:
            print(f"[Enrich] All {len(publications)} papers already complete")
            return publications
        
        # ✅ 激进并发：移除硬编码上限
        max_workers = get_extraction_workers(len(pubs_to_process))  # 完全依赖动态并发控制
        print(f"[Enrich] Using {max_workers} workers for {len(pubs_to_process)} papers")
        
        # Parallel processing with timeout protection
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(enrich_single_paper, pub_tuple): pub_tuple[0]
                for pub_tuple in pubs_to_process
            }
            
            for future in as_completed(futures, timeout=None):
                try:
                    future.result(timeout=30)  # 30s per paper max
                except FutureTimeoutError:
                    with lock:
                        failed_count += 1
                except Exception:
                    with lock:
                        failed_count += 1
        
        print(f"[Enrich] {enriched_count}/{len(publications)} enriched (S2: {s2_success}, arXiv: {arxiv_success}, Failed: {failed_count})")
        
        # ✅ 验证 authors 字段完整性
        papers_still_without_authors = 0
        for pub in publications:
            authors = pub.get('authors', [])
            if not authors or (isinstance(authors, list) and len(authors) == 0):
                papers_still_without_authors += 1
        
        if papers_still_without_authors > 0:
            print(f"⚠️  [Enrich] {papers_still_without_authors}/{len(publications)} papers still missing authors after enrichment")
            print(f"    → These papers may cause paper_score = 0.0 if scored > 6")
        
    except ImportError as e:
        print(f"[Enrich Profile Pubs] Using original data")
        
        # 确保所有论文都有必需字段
        for pub in publications:
            if not pub.get('abstract'):
                pub['abstract'] = ""
            if not pub.get('introduction'):
                pub['introduction'] = ""
            if not pub.get('url'):
                pub['url'] = ""
    
    except Exception as e:
        print(f"[Enrich Profile Pubs] ❌ Enrichment error: {e}")
        import traceback
        traceback.print_exc()
        
        # 确保所有论文都有必需字段
        for pub in publications:
            if not pub.get('abstract'):
                pub['abstract'] = ""
            if not pub.get('introduction'):
                pub['introduction'] = ""
            if not pub.get('url'):
                pub['url'] = ""
    return publications

def refine_interest_descriptions(interests: List[Dict[str, str]], classified_pubs: Dict[str, List[Dict]], api_key: str = None) -> List[Dict[str, str]]:
    """
    基于分类后的论文细化研究方向描述
    Args:
        interests: 初步的研究方向列表
        classified_pubs: 已分类的论文 {interest_name: [pub1, pub2, ...]}
        api_key: LLM API key
    Returns:
        细化后的研究方向列表
    """
    
    refined_interests = []
    
    for interest in interests:
        interest_name = interest['name']
        current_desc = interest['description']
        
        # 获取该方向下的论文
        pubs_in_category = classified_pubs.get(interest_name, [])
        
        if not pubs_in_category:
            # 没有论文的方向，保持原描述
            refined_interests.append(interest)
            continue
        
        # 构建论文文本（最多5篇）
        papers_text = []
        for pub in pubs_in_category[:5]:
            title = pub.get('title', '')
            abstract = pub.get('abstract', '')
            
            paper_block = f"• {title}"
            if abstract and abstract.strip():
                paper_block += f"\n  Abstract: {abstract[:300]}"
            papers_text.append(paper_block)
        
        papers_in_category_text = "\n\n".join(papers_text)
        
        prompt = PROMPT_REFINE_INTEREST_DESCRIPTION.format(
            interest_name=interest_name,
            papers_in_category=papers_in_category_text,
            current_description=current_desc
        )
        
        try:
            # ✅ 第一步：判断是否需要细化（如果已有描述）
            needs_refinement = True  # 默认需要细化
            
            if current_desc and len(current_desc) > 0:                
                eval_prompt = PROMPT_EVALUATE_DESCRIPTION_QUALITY.format(
                    interest_name=interest_name,
                    current_description=current_desc
                )
                
                try:
                    llm_eval = llm.get_llm("extract", temperature=0.1, api_key=api_key)
                    eval_response = llm.safe_structured(llm_eval, eval_prompt, schemas.LLMDescriptionQualitySpec)
                    
                    if eval_response and hasattr(eval_response, 'needs_refinement'):
                        needs_refinement = eval_response.needs_refinement
                        reason = eval_response.reason if hasattr(eval_response, 'reason') else ""
                        
                        if needs_refinement:
                            print(f"  → Needs refinement: {reason}")
                        else:
                            print(f"  → Already good: {reason}")
                            # 保留原描述，不调用细化
                            refined_interests.append(interest)
                            print(f"  ✅ '{interest_name}': Original description retained (LLM determined good quality)")
                            continue
                    else:
                        print(f"  ⚠️  LLM failed to determine, defaulting to refinement")
                        needs_refinement = True
                        
                except Exception as e:
                    print(f"  ⚠️  Determination error: {e}, defaulting to refinement")
                    needs_refinement = True
            
            # ✅ 第二步：如果需要细化，调用 LLM 生成新描述
            if needs_refinement:                
                llm_instance = llm.get_llm("extract", temperature=0.3, api_key=api_key)
                response = llm.safe_structured(llm_instance, prompt, schemas.LLMStringSpec)
                
                if response and hasattr(response, 'value') and response.value.strip():
                    refined_desc = response.value.strip()
                    refined_interests.append({
                        'name': interest_name,
                        'description': refined_desc
                    })
                else:
                    # 细化失败，保持原描述
                    refined_interests.append(interest)
                
        except Exception as e:
            # 细化失败，保持原描述
            refined_interests.append(interest)
    
    return refined_interests

def synthesize_research_interests_from_papers(publications: List[Dict[str, Any]], bio: str = "", scholar_tags: List[str] = None, api_key: str = None) -> List[Dict[str, str]]:
    """
    从论文标题直接生成研究方向
    
    Args:
        publications: 论文列表
        bio: 学者简介（可选）
        scholar_tags: Google Scholar 标签（可选）
        api_key: LLM API key
        
    Returns:
        研究方向列表 [{"name": str, "description": str}, ...]
    """
    if not publications:
        return []
    
    print(f"Generating research interests from {len(publications)} paper titles...")
    
    scholar_tags = scholar_tags or []
    
    # 构建论文列表文本（title + abstract if available）
    papers_list_parts = []
    abstract_count = 0
    
    for i, pub in enumerate(publications[:30], 1):  # 最多30篇
        title = pub.get('title', '')
        abstract = pub.get('abstract', '')
        
        paper_text = f"{i}. {title}"
        if abstract and abstract.strip():
            paper_text += f"\n   Abstract: {abstract[:400]}"  # 截断到400字符
            abstract_count += 1
        
        papers_list_parts.append(paper_text)
    
    papers_list = "\n\n".join(papers_list_parts)
    
    print(f"  📊 Paper information: {len(publications[:30])} titles, {abstract_count} with abstracts")
    
    prompt = PROMPT_SYNTHESIZE_INTERESTS_FROM_PAPERS.format(
        papers_list=papers_list,
        bio=bio[:300] if bio else "N/A",
        scholar_tags=', '.join(scholar_tags[:5]) if scholar_tags else "N/A"
    )
    
    try:
        llm_instance = llm.get_llm("extract", temperature=0.3, api_key=api_key)
        response = llm.safe_structured(llm_instance, prompt, schemas.LLMResearchInterestsSpec)
        
        if response and hasattr(response, 'research_interests'):
            # 验证每个兴趣都有 name 和 description
            valid_interests = []
            for item in response.research_interests[:4]:  # 限制3-4个
                # Handle both Pydantic model objects and dict format
                if hasattr(item, 'name') and hasattr(item, 'description'):
                    # Pydantic model object (LLMResearchInterestItem)
                    if item.name:  # Only add if name is non-empty
                        valid_interests.append({
                            'name': item.name,
                            'description': item.description or ''
                        })
                elif isinstance(item, dict) and 'name' in item and 'description' in item:
                    # Dict format (fallback for JSON parsing)
                    valid_interests.append({
                        'name': item['name'],
                        'description': item['description']
                    })
            
            print(f"  ✅ Generated {len(valid_interests)} research interests:")
            for interest in valid_interests:
                print(f"     • {interest.get('name', '')}")
            
            return valid_interests
        
        print(f"  ⚠️  No response or incorrect format")
        return []
        
    except Exception as e:
        print(f"  ❌ Research interests generation failed: {e}")
        return []


def classify_single_publication(pub: Dict[str, Any], interests: List[Dict[str, str]], api_key: str = None) -> str:
    """
    单个论文分类（用于并行处理，支持模糊匹配 + 关键词兜底）
    
    Args:
        pub: 论文字典
        interests: 研究方向列表
        api_key: LLM API key
        
    Returns:
        分类结果（类别名称）
    """
    if not interests:
        return "Other"
    
    # 论文信息
    title = pub.get('title', '')
    abstract = pub.get('abstract', '')
    
    # 构建兴趣列表文本
    interests_list = "\n".join([f"{i+1}. {interest['name']}" for i, interest in enumerate(interests)])
    
    # 构建论文描述（标题 + 摘要截断）
    description = f"**Title**: {title}"
    if abstract and len(abstract.strip()) > 0:
        # 截断摘要到约 600 字符（约 150 词）
        abstract_truncated = abstract[:600].strip()
        if len(abstract) > 600:
            abstract_truncated += "..."
        description += f"\n\n**Abstract**: {abstract_truncated}"
    
    prompt = PROMPT_CLASSIFY_PUBLICATION.format(
        interests_list=interests_list,
        description=description
    )
    
    try:
        llm_instance = llm.get_llm("extract", temperature=0.1, api_key=api_key)
        response = llm.safe_structured(llm_instance, prompt, schemas.LLMStringSpec)
        
        # 提取分类结果
        if response and hasattr(response, 'value'):
            category = response.value.strip().strip('"\'')
        else:
            # LLM 返回为空，使用关键词兜底
            print(f"  ⚠️  LLM returned empty: '{title[:40]}...'")
            return _classify_by_keywords(pub, interests)
        
        # 验证是否是有效的兴趣名称
        interest_names = [interest['name'] for interest in interests]
        
        # 1. Exact match
        if category in interest_names:
            return category
        
        # 2. 模糊匹配（包含匹配 + 大小写不敏感）
        category_lower = category.lower()
        for interest_name in interest_names:
            interest_lower = interest_name.lower()
            if interest_lower in category_lower or category_lower in interest_lower:
                return interest_name
        
        return _classify_by_keywords(pub, interests)
        
    except Exception as e:
        print(f"  ⚠️  LLM classification error for '{title[:50]}...': {e}, using keyword fallback...")
        return _classify_by_keywords(pub, interests)


def _classify_by_keywords(pub: Dict[str, Any], interests: List[Dict[str, str]]) -> str:
    """
    改进的关键词匹配分类策略（兜底用）
    
    Args:
        pub: 论文字典
        interests: 研究方向列表
        
    Returns:
        分类结果（类别名称或 "Other"）
    """
    title = pub.get('title', '').lower()
    abstract = pub.get('abstract', '').lower()
    text = title + " " + abstract
    
    # 计算每个方向的匹配分数
    scores = []
    for interest in interests:
        interest_name = interest['name']
        interest_desc = interest.get('description', '')
        
        # 提取方向名称中的关键词（去除常见停用词）
        stopwords = {'and', 'or', 'in', 'for', 'with', 'the', 'of', 'to', 'a', 'an', 'on', 'using', 'based'}
        interest_keywords = [
            word.lower() 
            for word in interest_name.split() 
            if word.lower() not in stopwords and len(word) > 2
        ]
        
        # 也从描述中提取技术词汇（改进：提取所有3字以上的词）
        import re
        # 提取描述中的技术术语（包括小写开头的专业词汇）
        desc_words = re.findall(r'\b[a-zA-Z]{3,}(?:[-_][a-zA-Z]+)*\b', interest_desc)
        # 过滤停用词和常见词
        common_words = {'the', 'and', 'for', 'that', 'with', 'from', 'this', 'their', 'which', 'these', 'such', 'systems', 'methods', 'models', 'learning', 'neural', 'networks'}
        desc_keywords = [w.lower() for w in desc_words if w.lower() not in common_words][:8]
        interest_keywords.extend(desc_keywords)
        
        # 去重
        interest_keywords = list(dict.fromkeys(interest_keywords))
        
        # 计算匹配分数（改进：更精细的权重）
        score = 0
        matched_keywords = []
        for keyword in interest_keywords:
            if len(keyword) <= 2:
                continue
            if keyword in text:
                # 标题匹配：高权重
                if keyword in title:
                    score += 5
                    matched_keywords.append(keyword)
                # 摘要匹配：中等权重
                else:
                    score += 2
                    matched_keywords.append(keyword)
        
        # 额外加成：多个关键词匹配
        if len(matched_keywords) >= 3:
            score += 3
        elif len(matched_keywords) >= 2:
            score += 1
        
        scores.append((interest_name, score, matched_keywords))
    
    # 选择得分最高的方向
    scores.sort(key=lambda x: x[1], reverse=True)
    best_match, best_score, best_keywords = scores[0]
    
    # 总是返回得分最高的方向（即使得分为0也要选择最相关的）
    if best_score > 0:
        print(f"    → Keyword fallback: '{best_match}' (score={best_score}, matched: {', '.join(best_keywords[:3])})")
    else:
        print(f"    → Keyword fallback: '{best_match}' (score={best_score}, no strong match - choosing closest category)")
    return best_match


def classify_publications_parallel(publications: List[Dict[str, Any]], 
                                  interests: List[Dict[str, str]], 
                                  api_key: str = None,
                                  max_workers: int = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    第三阶段：并行分类论文到研究兴趣
    
    Args:
        publications: 论文列表
        interests: 研究兴趣列表
        api_key: LLM API key
        max_workers: 最大并行worker数
    
    Returns:
        Dict[interest_name, List[publications]]
    """
    if not publications or not interests:
        return {}
    
    # 统计有摘要的论文数量
    pubs_with_abstract = sum(1 for pub in publications if pub.get('abstract', '').strip())
    
    print(f"\n🔄 Classifying {len(publications)} papers...")
    print(f"    Research categories: {', '.join([i['name'] for i in interests])}")
    
    # 初始化分类结果
    classified = {interest['name']: [] for interest in interests}
    classified['Other'] = []
    
    # ✅ 动态并发控制
    if max_workers is None:
        from .dynamic_concurrency import get_llm_workers
        max_workers = get_llm_workers(len(publications))  # ✅ 移除上限，论文分类可以开很多并发
    
    # Parallel classification
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有分类任务
        future_to_pub = {
            executor.submit(classify_single_publication, pub, interests, api_key): pub
            for pub in publications
        }
        
        # 收集结果（总体超时 40 秒，单个论文 30 秒）
        for future in as_completed(future_to_pub, timeout=40):
            pub = future_to_pub[future]
            try:
                interest_name = future.result(timeout=30)
                classified[interest_name].append(pub)
                
                # 显示是否使用了摘要
                has_abstract = " (+abstract)" if pub.get('abstract', '').strip() else ""
                print(f"  ✓ {pub.get('title', '')}{has_abstract} → {interest_name}")
                
            except TimeoutError:
                print(f"  ⚠️  Timeout: '{pub.get('title', '')}'")
                classified['Other'].append(pub)
            except Exception as e:
                print(f"  ⚠️  Error: '{pub.get('title', '')}': {e}")
                classified['Other'].append(pub)
    
    # 移除空分类
    classified = {k: v for k, v in classified.items() if v}
    
    print(f"\n✅ Paper classification complete. Results:")
    for interest_name, pubs in classified.items():
        print(f"   - {interest_name}: {len(pubs)} 篇")
    
    other_count = len(classified.get("Other", []))
    valid_categories = {k: v for k, v in classified.items() if k != "Other"}
    
    if other_count > 0 and not valid_categories:
        
        # 重新使用关键词分类
        classified_retry = {interest['name']: [] for interest in interests}
        for pub in classified.get("Other", []):
            category = _classify_by_keywords(pub, interests)
            if category != "Other":
                classified_retry[category].append(pub)
        
        # 移除空分类
        classified_retry = {k: v for k, v in classified_retry.items() if v}
        
        if classified_retry:
            print(f"✅ Forced keyword allocation results:")
            for interest_name, pubs in classified_retry.items():
                print(f"   - {interest_name}: {len(pubs)} papers")
            return classified_retry
        else:
            print(f"⚠️  Keyword allocation also failed, returning original 'Other' classification")
    
    return classified

BACKGROUND_MIN_CHARS = 60
BACKGROUND_VERB_PATTERN = re.compile(
    r"\b(am|is|are|was|were|works?|working|research(?:es|ing)?|focus(?:es|ing)?|stud(?:y|ies|ying)|leads?|serves?)\b",
    re.IGNORECASE
)
BACKGROUND_NOISE_KEYWORDS = [
    "block or report",
    "pinned",
    "repositories",
    "public ",
    "forks",
    "stars",
    "watching",
    "loading",
    "achievements highlights",
    "github",
    "followers",
    "contributions in the last year",
    "projects",
    "issues",
    "pull requests",
    "clone",
    "commit",
    "license",
    "readme"
]

def _is_trustworthy_background_text(text: str, api_key: Optional[str] = None) -> tuple[bool, str]:
    """
    使用LLM判断从homepage提取的bio_text是否真实可信
    
    如果提供了api_key，使用LLM进行智能判断；否则使用简单的规则检查作为fallback。
    
    Args:
        text: 待检查的bio文本
        api_key: LLM API key（可选）
    
    Returns:
        (is_trustworthy, reason) 元组
    """
    if not text:
        return False, "empty"
    
    normalized = " ".join(text.split())
    
    # 基础检查：如果文本太短，直接拒绝（避免浪费LLM调用）
    if len(normalized) < 30:  # 降低最小长度要求，让LLM判断更灵活
        return False, "too short"
    
    # 如果有api_key，使用LLM判断
    if api_key:
        try:
            llm_client = llm.get_llm("extract", temperature=0.1, api_key=api_key)
            
            prompt = (
                "You are classifying whether an extracted webpage text is a real researcher biography/background "
                "or just webpage noise (UI/navigation/metadata). This is NOT a fact-checking task.\n\n"
                f"TEXT TO EVALUATE:\n{normalized}\n\n"
                "Label the text as one of the following:\n"
                "1. BIO: real biographical/background content about a researcher.\n"
                "2. NOISE: webpage UI elements, navigation menus, footers, metadata, or irrelevant fragments.\n\n"
                "Strong signals for BIO (any of these):\n"
                "- Full sentences describing a person’s role, affiliation, education, research interests, awards, or career.\n"
                "- First/third-person academic self-description (e.g., 'I am a PhD student...', "
                "'She is an assistant professor...', 'His research focuses on...').\n"
                "- Mentions of degrees, universities, advisors, labs, research areas, publications in narrative form.\n\n"
                "Strong signals for NOISE (any of these):\n"
                "- Mostly navigation/menu words: 'Home', 'About', 'People', 'Publications', 'Projects', "
                "'Contact', 'News', 'Teaching', 'Blog', etc.\n"
                "- Page boilerplate/metadata: copyright, 'all rights reserved', cookies, "
                "share buttons, 'last updated', view counts, tag clouds.\n"
                "- Is only a short fragment, list of keywords, or UI labels with no real sentences.\n\n"
                "Mixed text rule:\n"
                "- If BIO-like content is present but less than half of the text and UI/menu noise dominates, label NOISE.\n\n"
                "Return JSON only with:\n"
                "{\n"
                "  \"is_trustworthy\": <true if BIO else false>,\n"
                "  \"reason\": \"<brief reason>\",\n"
                "}"
            )
            response = llm.safe_structured(llm_client, prompt, schemas.LLMTrustworthyBackgroundSpec)
            
            if response:
                is_trustworthy = getattr(response, 'is_trustworthy', False)
                reason = getattr(response, 'reason', 'LLM evaluation')
                return is_trustworthy, reason
            else:
                # LLM调用失败，fallback到规则检查
                print(f"[Background Check] ⚠️ LLM evaluation failed, falling back to rule-based check")
        
        except Exception as e:
            # LLM调用出错，fallback到规则检查
            print(f"[Background Check] ⚠️ LLM evaluation error: {e}, falling back to rule-based check")
    
    # Fallback: 使用规则检查（当没有api_key或LLM调用失败时）
    if len(normalized) < BACKGROUND_MIN_CHARS:
        return False, "too short"
    
    # 英文比例检查：至少30%是ASCII字符（更宽松，允许多语言bio）
    ascii_ratio = sum(1 for c in normalized if ord(c) < 128) / max(len(normalized), 1)
    if ascii_ratio < 0.3:
        return False, "low english ratio"
    
    lowered = normalized.lower()
    
    # 噪音关键词检查：过滤GitHub UI元素等明显噪音
    for keyword in BACKGROUND_NOISE_KEYWORDS:
        if keyword in lowered:
            return False, f"contains noise keyword '{keyword}'"
    
    # 描述性内容检查
    has_verb = BACKGROUND_VERB_PATTERN.search(normalized)
    academic_keywords = [
        r"\b(professor|researcher|phd|doctor|university|institute|lab|laboratory|research|publication|paper|journal)\b",
        r"\b(education|degree|bachelor|master|doctoral|postdoc|postdoctoral)\b"
    ]
    has_academic_keyword = any(re.search(pattern, normalized, re.IGNORECASE) for pattern in academic_keywords)
    
    if not has_verb and not has_academic_keyword:
        return False, "missing descriptive verbs or academic keywords"
    
    return True, "ok"
# 学位映射表：标准学位标签 -> 各种变体关键词列表
# ============================ 学位映射表 ============================
DEGREE_MAPPING = {
    "phd": [
        "phd", "ph.d", "ph.d.", "dphil", "d phil",
        "doctor of philosophy", "philosophiae doctor",
        "doctoral student", "doctoral candidate", "phd student", "phd candidate",
        "candidate for phd", "pursuing phd",
        "博士", "博士生", "博士研究生", "攻读博士"
    ],
    "master": [
        "master", "master's", "masters",
        "msc", "m.sc", "ms", "m.s", "ma", "m.a",
        "meng", "m.eng", "mba", "mphil", "m.phil", "mres",
        "master of science", "master of engineering", "master of arts",
        "硕士", "硕士生"
    ],
    "bachelor": [
        "bachelor", "bachelor's", "bachelors",
        "bsc", "b.sc", "bs", "b.s", "ba", "b.a",
        "beng", "b.eng", "b.e",
        "bachelor of science", "bachelor of arts", "bachelor of engineering",
        "undergraduate", "undergrad", "undergraduate student",
        "学士", "学士学位", "本科", "本科生"
    ],
    "postdoc": [
        "postdoc", "post-doctoral", "postdoctoral", "post-doctorate",
        "postdoctoral researcher", "postdoctoral fellow", "postdoctoral associate",
        "postdoctoral scholar", "postdoctoral fellowship",
        "postdoctoral research fellow", "postdoc research fellow",
        "博士后"
    ]
}

# ============================ 学术职位映射表 ============================
ACADEMIC_POSITION_MAPPING = {
    "professor": [
        "professor", "prof", "full professor", "prof.",
        "教授", "正教授"
    ],
    "associate_professor": [
        "associate professor", "assoc prof", "associate prof", "assoc. prof",
        "副教授"
    ],
    "assistant_professor": [
        "assistant professor", "asst prof", "assistant prof", "asst. prof",
        "助理教授"
    ],
    "lecturer": [
        "lecturer", "senior lecturer", "principal lecturer", "adjunct lecturer",
        "讲师"
    ],
    "research_fellow": [
        "research fellow", "academic fellow", "university fellow",
        "fellow", "research fellowship"
    ],
    "visiting_scholar": [
        "visiting scholar", "visiting student",
        "访问学者", "访问学生"
    ],
    "visiting_researcher": [
        "visiting researcher", "visiting research associate", "visiting research assistant",
        "visiting scientist", "visiting research scientist",
        "访问研究员", "访问研究助理"
    ],
    "visiting_professor": [
        "visiting professor", "visiting associate professor", "visiting assistant professor",
        "visiting faculty", "adjunct professor",
        "访问教授", "客座教授"
    ]
}

# ============================ 教学角色映射表 ============================
TEACHING_ROLE_MAPPING = {
    "instructor": [
        "instructor", "course instructor", "adjunct instructor",
        "教员"
    ],
    "lecturer": [
        "lecturer", "teaching lecturer", "course lecturer",
        "讲师"
    ],
    "teaching_assistant": [
        "teaching assistant", "ta", "course assistant", "ca",
        "teaching fellow", "tf", "graduate teaching assistant", "gta",
        "助教"
    ]
}

# ============================ 所有允许的标准标签 ============================
# 用于验证：只有映射表中的内容才会被保留
ALLOWED_CAREER_EDUCATION_LABELS = set(
    list(DEGREE_MAPPING.keys()) +
    list(ACADEMIC_POSITION_MAPPING.keys()) +
    list(TEACHING_ROLE_MAPPING.keys())
)


def _extract_field_from_degree_title(degree_or_position: str) -> Optional[str]:
    """
    从 "Ph.D. in XXX" 或 "M.S. in XXX" 格式中提取专业信息。
    
    Args:
        degree_or_position: 学位/职位字符串（如 "Ph.D. in Electrical Engineering", "M.S. in Computer Science"）
    
    Returns:
        提取的专业信息（如 "Electrical Engineering", "Computer Science"），
        如果无法提取则返回 None
    """
    if not degree_or_position:
        return None
    
    import re
    
    # 匹配模式：学位 in 专业
    # 支持：Ph.D. in XXX, Ph.D in XXX, PhD in XXX, M.S. in XXX, M.S in XXX, MS in XXX, B.S. in XXX 等
    patterns = [
        r"(?:ph\.?\s*d\.?|phd|doctor\s+of\s+philosophy)\s+in\s+(.+?)(?:\s+@|\s*$)",
        r"(?:m\.?\s*s\.?|ms|master|m\.?\s*sc|msc|m\.?\s*eng|meng)\s+in\s+(.+?)(?:\s+@|\s*$)",
        r"(?:b\.?\s*s\.?|bs|bachelor|b\.?\s*sc|bsc|b\.?\s*eng|beng|b\.?\s*e)\s+in\s+(.+?)(?:\s+@|\s*$)",
        r"(?:m\.?\s*a\.?|ma|master\s+of\s+arts)\s+in\s+(.+?)(?:\s+@|\s*$)",
        r"(?:b\.?\s*a\.?|ba|bachelor\s+of\s+arts)\s+in\s+(.+?)(?:\s+@|\s*$)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, degree_or_position, re.IGNORECASE)
        if match:
            field = match.group(1).strip()
            # 移除可能的尾随标点符号
            field = field.rstrip('.,;:')
            if field:
                return field
    
    return None

def _normalize_career_education_label(title: str) -> Optional[str]:
    """
    统一的归一化函数：将学位/学术职位/教学角色映射到标准标签。
    只保留映射表中的内容，如果无法映射则返回 None。
    
    支持 "Ph.D. in XXX" 格式：会正确识别为对应的学位类型。
    
    Args:
        title: 原始标题字符串（如 "Ph.D.", "Ph.D. in CS", "MS", "Professor", "TA"）
    
    Returns:
        标准标签（如 "phd", "master", "professor", "teaching_assistant"），
        如果无法映射到任何标准标签则返回 None
    """
    if not title:
        return None
    
    t = (title or "").lower().replace(".", "").strip()
    if not t:
        return None
    
    # 匹配策略：优先匹配更长的 variant，避免短 variant 误匹配
    def find_best_match(mapping_dict):
        best_match = None
        best_length = 0
        for standard_label, variants in mapping_dict.items():
            for variant in variants:
                # 检查 variant 是否在 t 中，并且是完整的单词匹配
                if variant == t:
                    # 完全匹配，优先级最高
                    return standard_label
                elif variant in t:
                    # 部分匹配：确保 variant 是完整的单词序列（避免部分匹配）
                    # 例如："associate professor" 不应该匹配到 "visiting associate professor"
                    variant_start = t.find(variant)
                    if variant_start != -1:
                        # 检查 variant 前面是否是空格或开头
                        before_ok = (variant_start == 0 or t[variant_start - 1] == " ")
                        # 检查 variant 后面是否是空格或结尾
                        after_pos = variant_start + len(variant)
                        after_ok = (after_pos == len(t) or t[after_pos] == " ")
                        # 只有前后都是边界时才匹配
                        if before_ok and after_ok:
                            if len(variant) > best_length:
                                best_match = standard_label
                                best_length = len(variant)
        return best_match
    
    # 1. 先检查学位映射
    match = find_best_match(DEGREE_MAPPING)
    if match:
        return match
    
    # 2. 检查学术职位映射
    match = find_best_match(ACADEMIC_POSITION_MAPPING)
    if match:
        return match
    
    # 3. 检查教学角色映射
    match = find_best_match(TEACHING_ROLE_MAPPING)
    if match:
        return match
    
    # 无法映射到任何标准标签
    return None

def build_enhanced_profile(openreview_data: Dict[str, Any], 
                          author_name: str,
                          api_key: str = None,
                          profile: Optional[AuthorProfile] = None,
                          user_query: str = None,
                          exclude_paper_urls: set = None,
                          trigger_paper: Dict[str, Any] = None,
                          s2_author_id: str = None) -> Optional[schemas.EnhancedAuthorProfile]:
    """
    构建增强的9字段档案
    Args:
        openreview_data: OpenReview 原始数据
        author_name: 作者姓名
        api_key: LLM API key
        profile: AuthorProfile（包含从多源采集的数据）
        user_query: 用户查询（用于论文评分，可选）
        exclude_paper_urls: 要排除的论文URL集合（避免与Reference Papers重复）
        trigger_paper: 触发搜索的论文信息（Dict包含title, url, venue, score等）
        s2_author_id: Semantic Scholar authorId，用于获取作者的论文列表
    
    Returns:
        EnhancedAuthorProfile 对象
    """
    if exclude_paper_urls is None:
        exclude_paper_urls = set()
    if not openreview_data:
        return None
    
    print(f"\n🔨 Building enhanced profile for {author_name}...")
    print(f"[build_enhanced_profile] Using multi-source data: {profile is not None}")
    print(f"[build_enhanced_profile] User query for paper scoring: {user_query[:100] if user_query else 'None'}")
    if trigger_paper and trigger_paper.get('title'):
        print(f"[build_enhanced_profile] 🎯 Trigger paper: {trigger_paper.get('title', '')}")
    
    # Initialize high_quality_pubs list (will be populated if user_query is provided)
    high_quality_pubs = []
    # ==========================================================================
    # 三层 Profile 构建逻辑：
    # Stage 1: OpenReview baseline → 初始结构化 profile（最可信）
    # Stage 2: Homepage 九维增量 → 完善 profile 字段（主要信息源）
    # Stage 3: Google Scholar → 仅补充论文列表（不影响其他字段）
    # ==========================================================================
    print(f"Stage 1: OpenReview Baseline - Building initial structured profile")
    print(f"[Stage 1] OpenReview provides: career/education, industrial experience, services, publications")
    print(f"[Stage 1] These will serve as the trusted baseline for profile construction")
    # ========== Stage 2: Homepage 九维增量完善 ==========
    print(f"Stage 2: Homepage Enhancement - Refining profile with 9D incremental extraction")
    
    nine_dim_data = None
    if profile:
        try:
            from .nine_dimension_extractor import extract_nine_dimensions_multi_source
            
            nine_dim_data = extract_nine_dimensions_multi_source(
                author_name=author_name,
                profile=profile,
                openreview_data=openreview_data,
                api_key=api_key
            )
            
            print(f"[Stage 2] ✅ Homepage 9D extraction completed")
            
        except Exception as e:
            print(f"[Stage 2] ❌ Homepage 9D extraction failed: {e}")
            import traceback
            traceback.print_exc()
    
    # 1. Introduction - 暂时只设置 name，position/affiliation 将在 profile 构建完成后由新的 role 决策函数填充
    intro_name = author_name  # Default to input parameter
    
    # 尝试从 9D background 获取更准确的 name
    if nine_dim_data and nine_dim_data.get('background'):
        bg = nine_dim_data['background']
        extracted_name = bg.get('name', '').strip()
        if not intro_name or intro_name.lower() in ['unknown', 'n/a', '']:
            if extracted_name and extracted_name.lower() not in ['unknown', 'n/a', '']:
                intro_name = extracted_name
                print(f"  ✅ Using name from 9D background extraction: {intro_name}")
    
    # 创建临时 introduction（position/affiliation 稍后填充）
    introduction = schemas.IntroductionInfo(
        name=intro_name,
        position="",  # Will be filled by determine_current_role_from_profile
        affiliation="",  # Will be filled by determine_current_role_from_profile
        text=f"I am {intro_name}."  # Will be updated after role determination
    )
    
    # 2. Research Interests - 优先使用 Homepage 提供的，LLM 生成作为 fallback
    research_interests = []
    research_interests_raw = []
    publications_for_analysis = []  # 初始化论文列表
    
    homepage_interests = []
    nine_dim_interests = []
    # 来源1: AuthorProfile.interests（来自 Phase 3 多源采集）
    if profile:
        homepage_interests = getattr(profile, 'interests', []) or []
    
    # 来源2: 九维度提取（如果可用）- 新格式支持 name + description
    nine_dim_interests_with_desc = []
    if nine_dim_data and nine_dim_data.get('research_interests'):
        ri_data = nine_dim_data['research_interests'].get('interests', [])
        
        if isinstance(ri_data, list) and len(ri_data) > 0:
            # 检查格式：新格式是 [{"name": "...", "description": "..."}]
            if isinstance(ri_data[0], dict) and 'name' in ri_data[0]:
                # 新格式：带描述（检查是否真的有描述）
                nine_dim_interests_with_desc = []
                has_any_desc = False
                
                for item in ri_data:
                    desc = item.get('description', '').strip()
                    if desc:
                        has_any_desc = True
                    nine_dim_interests_with_desc.append({
                        'name': item['name'],
                        'description': desc
                    })
                
                nine_dim_interests = [item['name'] for item in ri_data]
            else:
                # 旧格式：只有名称
                nine_dim_interests = ri_data    
    
    # 只有当 homepage 没有提取到时，才用九维度的
    if homepage_interests:
        final_interests = homepage_interests
    elif nine_dim_interests:
        final_interests = nine_dim_interests
    else:
        final_interests = []
    
    # 检查是否有有效描述
    has_descriptions = False
    
    if nine_dim_interests_with_desc:
        valid_desc_count = sum(1 for item in nine_dim_interests_with_desc if item.get('description', '').strip())
        has_descriptions = valid_desc_count > 0
    
    # 确定实际使用的数据源（用于日志显示）
    actual_source = "Unknown"
    if homepage_interests and final_interests == homepage_interests:
        actual_source = "Homepage parser (Structured HTML parsing)"
    elif nine_dim_interests and final_interests == nine_dim_interests:
        actual_source = "Nine-dimension LLM extraction"
    print(f"\n📚 Research interests generation strategy...")
    print(f"  📋 Data source: {actual_source}")
    print(f"  📋 Number of tags: {len(final_interests)}")
    # 策略 1：如果 Homepage/Scholar 有研究方向标签，直接使用（不管有几个）
    if len(final_interests) > 0:
        # 检查是否已有描述
        if has_descriptions:
            # Homepage 已有完整描述，直接使用
            research_interests_raw = [
                {
                    'name': item['name'],
                    'description': item.get('description', '')
                }
                for item in nine_dim_interests_with_desc[:5]
            ]
            research_interests = [
                schemas.ResearchInterest(name=ri['name'], description=ri['description'])
                for ri in research_interests_raw
            ]
        else:
            # 只有方向名称，没有描述，需要后续用 LLM 基于论文生成描述
            print(f"  ✅ Using Homepage provided {len(final_interests)} research interest names")
            print(f"  🔄 However, a description needs to be generated based on the paper....")
            research_interests_raw = [
                {'name': interest, 'description': ''}
                for interest in final_interests[:5]  # 最多取5个
            ]
            research_interests = [
                schemas.ResearchInterest(name=ri['name'], description=ri['description'])
                for ri in research_interests_raw
            ]
        
        if user_query and api_key:
            print(f"Stage 3: Publication Collection - Merging papers from all sources")
            print(f"[Stage 3] Collecting from: OpenReview → Homepage → Google Scholar")
            print(f"[Stage 3] Scholar only supplements publication list, does not affect other fields")
            print(f"\n📊 Collecting and scoring publications for query relevance...")
            publications_for_analysis, high_quality_pubs = collect_all_publications_for_interests(
                profile, openreview_data, nine_dim_data, user_query=user_query, api_key=api_key, exclude_paper_urls=exclude_paper_urls, trigger_paper=trigger_paper, s2_author_id=s2_author_id, author_name=author_name
            )
            print(f"  ✅ Scored {len(high_quality_pubs)} high-quality publications (score ≥ 6)")
    
    else:
        print(f"Stage 3: Publication Collection - Merging papers from all sources")
        print(f"[Stage 3] Collecting from: OpenReview → Homepage → Google Scholar")
        print(f"[Stage 3] Scholar only supplements publication list, does not affect other fields")
        
        publications_for_analysis, high_quality_pubs = collect_all_publications_for_interests(
            profile, openreview_data, nine_dim_data, user_query=user_query, api_key=api_key, exclude_paper_urls=exclude_paper_urls, trigger_paper=trigger_paper, s2_author_id=s2_author_id, author_name=author_name
            )
        
        # ✅ 确保有论文才生成研究方向
        if not publications_for_analysis or len(publications_for_analysis) == 0:
            print(f"  ⚠️  No publications from collect function")
            print(f"  🔄 Trying to get publications from OpenReview directly...")
            # 🆕 保底：直接从 openreview_data 获取论文
            publications_for_analysis = openreview_data.get('publications', [])
            
            if not publications_for_analysis:
                print(f"  ❌ No publications available from any source, cannot generate research interests")
                research_interests = []
                research_interests_raw = []
            else:
                print(f"  ✅ Found {len(publications_for_analysis)} publications from OpenReview")
        
        if publications_for_analysis:
            try:
                # 获取 bio 和 scholar_tags（如果可用）
                bio = ""
                scholar_tags = final_interests  # 使用已有的标签辅助生成
                if profile:
                    bio = getattr(profile, 'bio', '') or ""
                
                # ✅ 直接从论文标题生成研究方向（新版：跳过关键词提取）
                print(f"\n📊 Generating research interests...")
                research_interests_raw = synthesize_research_interests_from_papers(
                    publications=publications_for_analysis,
                    bio=bio,
                    scholar_tags=scholar_tags,
                    api_key=api_key
                )
                if research_interests_raw:
                    research_interests = [
                        schemas.ResearchInterest(name=ri['name'], description=ri['description'])
                        for ri in research_interests_raw
                    ]
                    print(f"  ✅ Research interests: {len(research_interests)} broad directions (based on {len(publications_for_analysis)} papers)")
                else:
                    print(f"  ❌ Research interests generation failed")
                    research_interests = []
                    research_interests_raw = []
            except Exception as e:
                print(f"  ⚠️  Research interests extraction failed: {e}")
                import traceback
                traceback.print_exc()
                research_interests = []
                research_interests_raw = []
    
    # 3. Selected Research - 使用相同的论文列表进行分类
    selected_research_by_category = {}
    all_publications = []
    
    try:
        # ✅ 收集论文（确保论文已收集）
        if not publications_for_analysis:
            # 策略1使用了 Homepage 标签，现在需要收集论文用于分类
            print(f"\n📚 Collecting papers for classification...")
            publications_for_analysis, high_quality_pubs = collect_all_publications_for_interests(
                profile, openreview_data, nine_dim_data, user_query=user_query, api_key=api_key, exclude_paper_urls=exclude_paper_urls, trigger_paper=trigger_paper, s2_author_id=s2_author_id, author_name=author_name
            )
        
        # 使用收集到的论文列表
        publications = publications_for_analysis if publications_for_analysis else []
        print(f"\n📂 Paper classification: classifying {len(publications)} papers into research interests...")
        
        # Convert all publications to PublicationInfo
        conversion_errors = 0
        for pub in publications:
            try:
                authors_raw = pub.get('authors', [])
                if isinstance(authors_raw, str):
                    authors = [a.strip() for a in authors_raw.split(',') if a.strip()]
                elif isinstance(authors_raw, list):
                    authors = authors_raw
                else:
                    authors = []
                
                # 确保 keywords 是列表
                keywords_raw = pub.get('keywords', [])
                if isinstance(keywords_raw, str):
                    # 如果是字符串，分割为列表
                    keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
                elif isinstance(keywords_raw, list):
                    keywords = keywords_raw
                else:
                    keywords = []
                
                # ✅ 安全地转换 year 为 int
                raw_year = pub.get('year')
                year_int = None
                if raw_year is not None:
                    try:
                        if isinstance(raw_year, int):
                            year_int = raw_year
                        elif isinstance(raw_year, str):
                            year_int = int(raw_year) if raw_year.strip() else None
                        else:
                            year_int = None
                    except (TypeError, ValueError):
                        year_int = None
                        print(f"  ⚠️  Year conversion failed: {pub.get('title', '')} year='{raw_year}'")
                
                # 优先使用预先写入的作者位置信息（针对 trigger paper），避免后续依赖名字匹配
                precomputed_pos = pub.get('candidate_author_position')
                precomputed_total = pub.get('total_authors')
                position_index = None
                position_label = None
                total_authors_val = None
                
                # 获取特殊角色标记（从 PDF 检测得到）
                is_cofirst = pub.get('is_cofirst_author', False)
                is_corresponding = pub.get('is_corresponding_author', False)
                
                if authors:
                    if precomputed_pos is not None:
                        position_index = precomputed_pos
                        total_authors_val = precomputed_total or len(authors)
                        # 生成人类可读的位置标签 - 优先显示特殊角色
                        if is_cofirst:
                            # 共同一作：显示 co-first author
                            position_label = "co-first author"
                        elif is_corresponding:
                            # 通讯作者：显示 corresponding author
                            position_label = "corresponding author"
                        elif position_index == 1:
                            position_label = "1st author"
                        elif total_authors_val and position_index == total_authors_val and total_authors_val > 1:
                            position_label = f"last author ({position_index}/{total_authors_val})"
                        elif position_index == 2:
                            position_label = "2nd author"
                        elif position_index == 3:
                            position_label = "3rd author"
                        else:
                            if total_authors_val:
                                position_label = f"author #{position_index}/{total_authors_val}"
                            else:
                                position_label = f"author #{position_index}"
                    else:
                        position_index, position_label, total_authors_val = compute_author_position(author_name, authors)
                        # 即使通过名字匹配获取位置，也要检查特殊角色标记
                        if is_cofirst:
                            position_label = "co-first author"
                        elif is_corresponding:
                            position_label = "corresponding author"
                
                pub_info = schemas.PublicationInfo(
                    title=pub.get('title', ''),
                    authors=authors,
                    venue=pub.get('venue', ''),
                    year=year_int,
                    url=pub.get('url', ''),
                    abstract=pub.get('abstract', ''),
                    keywords=keywords,
                    relevance_score=pub.get('relevance_score', 0),
                    relevance_explanation=pub.get('relevance_explanation', ''),
                    # Multi-dimensional scoring fields
                    quality_score=pub.get('quality_score', 0.0),
                    relevance_dimensions=pub.get('relevance_dimensions', {}),
                    quality_dimensions=pub.get('quality_dimensions', {}),
                    # 计算并保存候选人在该论文中的作者位置
                    candidate_author_position=position_index,
                    candidate_author_position_label=position_label,
                    total_authors=total_authors_val,
                    # Corresponding author flag (from PDF detection)
                    is_corresponding_author=pub.get('is_corresponding_author', None),
                    # Co-first author flag (from PDF detection)
                    is_cofirst_author=pub.get('is_cofirst_author', None)
                )
                all_publications.append(pub_info)
            except Exception as e:
                conversion_errors += 1
                print(f"  ⚠️  Conversion of paper failed, skipping: {pub.get('title', 'Unknown')} - {e}")
        
        if conversion_errors > 0:
            print(f"  ⚠️  {conversion_errors}/{len(publications)} papers conversion failed")
        
        print(f"  ✅ Successfully converted {len(all_publications)}/{len(publications)} papers to PublicationInfo")
        # Classify publications into research interests
        if research_interests_raw and publications:
            print(f"\n  🔄 Starting paper classification...")
            print(f"     Input: {len(publications)} papers")
            print(f"     Target directions: {len(research_interests_raw)}")
            for ri in research_interests_raw:
                print(f"       - {ri['name']}")
            
            try:
                classified = classify_publications_parallel(
                    publications, 
                    research_interests_raw, 
                    api_key
                )
            except Exception as e:
                print(f"  ⚠️  Publication classification failed: {e}")
                import traceback
                # 使用关键词兜底
                classified = {}
                for interest in research_interests_raw:
                    classified[interest['name']] = []
                for pub in publications:
                    category = _classify_by_keywords(pub, research_interests_raw)
                    if category in classified:
                        classified[category].append(pub)
            
            # 1. 保留所有原始描述（包括关键词列表）
            # 2. 对有论文的方向，尝试生成更详细的描述
            # 3. 生成成功 → 替换；生成失败 → 保留原描述
            min_papers_for_description = 1
            
            # 统计描述状态
            interests_with_desc = [ri for ri in research_interests_raw if ri.get('description', '').strip()]
            interests_without_desc = [ri for ri in research_interests_raw if not ri.get('description', '').strip()]
            
            print(f"\n  📊 Description status statistics:")
            print(f"     - With description: {len(interests_with_desc)}")
            print(f"     - Without description: {len(interests_without_desc)}")
            
            # 对所有有论文的方向尝试细化描述
            interests_with_papers = [
                ri for ri in research_interests_raw 
                if ri['name'] in classified and len(classified[ri['name']]) >= min_papers_for_description
            ]
            
            if interests_with_papers and len(publications) >= min_papers_for_description:
                print(f"  🔄 Trying to refine the description of {len(interests_with_papers)} directions based on {len(publications)} papers...")
                
                # 调用细化函数（会尝试生成更详细的描述）
                refined_interests = refine_interest_descriptions(
                    interests=interests_with_papers,
                    classified_pubs=classified,
                    api_key=api_key
                )
                
                # 合并结果：用细化后的描述替换原描述
                final_interests_list = []
                
                for ri in research_interests_raw:
                    # 找到细化后的版本
                    refined_version = next((r for r in refined_interests if r['name'] == ri['name']), None)
                    if refined_version:
                        # 检查是否真的生成了新描述
                        new_desc = refined_version.get('description', '').strip()
                        old_desc = ri.get('description', '').strip()
                        
                        if new_desc and len(new_desc) > len(old_desc):
                            # LLM 生成了更详细的描述，使用新的
                            final_interests_list.append(refined_version)
                            print(f"  ✅ '{ri['name']}': Description enhanced ({len(old_desc)} → {len(new_desc)} chars)")
                        else:
                            # LLM 生成失败或不够详细，保留原描述
                            final_interests_list.append(ri)
                            if old_desc:
                                print(f"  ℹ️  '{ri['name']}': Original description retained")
                            else:
                                print(f"  ⚠️  '{ri['name']}': LLM refinement failed, keeping without description")
                    else:
                        # 没有被细化（没有论文或论文太少）
                        final_interests_list.append(ri)
                
                research_interests_raw = final_interests_list
                
                # 更新 research_interests
                research_interests = [
                    schemas.ResearchInterest(name=ri['name'], description=ri['description'])
                    for ri in research_interests_raw
                ]
            else:
                print(f"  ℹ️  Papers insufficient or no direction has papers, keeping original description")
            
            # ✅ 只保留有论文的研究方向（过滤掉没有匹配论文的空方向）
            total_classified_papers = 0
            papers_without_authors_in_classification = 0
            seen_titles_across_categories = set()
            
            for interest_name, pubs in classified.items():
                if interest_name != "Other" and pubs:  # 只添加有论文的方向
                    pub_list = []
                    seen_titles_in_category = set()
                    
                    for pub in pubs:
                        try:
                            # 🆕 去重：检查该类别中是否已有相同标题的论文
                            pub_title = pub.get('title', '').lower().strip()
                            if pub_title in seen_titles_in_category:
                                print(f"  ⏭️  Skipping duplicate paper in '{interest_name}': {pub.get('title', '')}")
                                continue
                            
                            if pub_title in seen_titles_across_categories:
                                print(f"  ⏭️  Skipping paper already in another category: {pub.get('title', '')} (current: '{interest_name}')")
                                continue
                            
                            seen_titles_in_category.add(pub_title)
                            seen_titles_across_categories.add(pub_title)
                            total_classified_papers += 1
                            
                            # ✅ 确保 authors 是列表（处理字符串格式）
                            authors_raw = pub.get('authors', [])
                            if isinstance(authors_raw, str):
                                # 字符串格式：'A Aljović, Z Lin, W Wang, ...'
                                authors = [a.strip() for a in authors_raw.split(',') if a.strip()]
                            elif isinstance(authors_raw, list):
                                authors = authors_raw
                            else:
                                authors = []
                            
                            if not authors:
                                papers_without_authors_in_classification += 1
                            
                            # 确保 keywords 是列表
                            keywords_raw = pub.get('keywords', [])
                            if isinstance(keywords_raw, str):
                                keywords = [k.strip() for k in keywords_raw.split(',') if k.strip()]
                            elif isinstance(keywords_raw, list):
                                keywords = keywords_raw
                            else:
                                keywords = []
                            
                            raw_year = pub.get('year')
                            year_int = None
                            if raw_year is not None:
                                try:
                                    if isinstance(raw_year, int):
                                        year_int = raw_year
                                    elif isinstance(raw_year, str):
                                        year_int = int(raw_year) if raw_year.strip() else None
                                except (TypeError, ValueError):
                                    year_int = None
                            
                            # 优先使用预先写入的作者位置信息（针对 trigger paper），避免后续依赖名字匹配
                            precomputed_pos = pub.get('candidate_author_position')
                            precomputed_total = pub.get('total_authors')
                            position_index = None
                            position_label = None
                            total_authors_val = None
                            
                            # 获取特殊角色标记（从 PDF 检测得到）
                            is_cofirst = pub.get('is_cofirst_author', False)
                            is_corresponding = pub.get('is_corresponding_author', False)
                            
                            if authors:
                                if precomputed_pos is not None:
                                    position_index = precomputed_pos
                                    total_authors_val = precomputed_total or len(authors)
                                    # 生成人类可读的位置标签 - 优先显示特殊角色
                                    if is_cofirst:
                                        position_label = "co-first author"
                                    elif is_corresponding:
                                        position_label = "corresponding author"
                                    elif position_index == 1:
                                        position_label = "1st author"
                                    elif total_authors_val and position_index == total_authors_val and total_authors_val > 1:
                                        position_label = f"last author ({position_index}/{total_authors_val})"
                                    elif position_index == 2:
                                        position_label = "2nd author"
                                    elif position_index == 3:
                                        position_label = "3rd author"
                                    else:
                                        if total_authors_val:
                                            position_label = f"author #{position_index}/{total_authors_val}"
                                        else:
                                            position_label = f"author #{position_index}"
                                else:
                                    position_index, position_label, total_authors_val = compute_author_position(author_name, authors)
                                    # 即使通过名字匹配获取位置，也要检查特殊角色标记
                                    if is_cofirst:
                                        position_label = "co-first author"
                                    elif is_corresponding:
                                        position_label = "corresponding author"
                            
                            pub_list.append(schemas.PublicationInfo(
                                title=pub.get('title', ''),
                                authors=authors,
                                venue=pub.get('venue', ''),
                                year=year_int,
                                url=pub.get('url', ''),
                                abstract=pub.get('abstract', ''),
                                keywords=keywords,
                                relevance_score=pub.get('relevance_score', 0),
                                relevance_explanation=pub.get('relevance_explanation', ''),
                                # Multi-dimensional scoring fields
                                quality_score=pub.get('quality_score', 0.0),
                                relevance_dimensions=pub.get('relevance_dimensions', {}),
                                quality_dimensions=pub.get('quality_dimensions', {}),
                                # 计算并保存候选人在该论文中的作者位置
                                candidate_author_position=position_index,
                                candidate_author_position_label=position_label,
                                total_authors=total_authors_val,
                                # Corresponding author flag (from PDF detection)
                                is_corresponding_author=pub.get('is_corresponding_author', None),
                                # Co-first author flag (from PDF detection)
                                is_cofirst_author=pub.get('is_cofirst_author', None)
                            ))
                        except Exception as e:
                            print(f"  Conversion of classified paper failed: {pub.get('title', '')} - {e}")
                            import traceback
                            traceback.print_exc()
                    selected_research_by_category[interest_name] = pub_list
            
            # 研究方向是从论文聚合生成的，如果没有论文被分到这个方向，说明分类失败或方向不准确
            interests_kept = []
            interests_removed = []
            
            for interest in research_interests_raw:
                interest_name = interest['name']
                has_papers = interest_name in selected_research_by_category and len(selected_research_by_category.get(interest_name, [])) > 0
                
                if has_papers:
                    # ✅ 只保留有论文的方向
                    interests_kept.append(interest)
                else:
                    # ❌ 删除没有论文的方向
                    interests_removed.append(interest_name)
            
            # 更新为保留的方向
            research_interests_raw = interests_kept
            research_interests = [
                schemas.ResearchInterest(name=ri['name'], description=ri['description'])
                for ri in research_interests_raw
            ]
            
            selected_research_by_category = {
                name: pubs 
                for name, pubs in selected_research_by_category.items() 
                if pubs  # 只保留非空列表
            }
            # 统计分类结果
            categories_with_papers = len(selected_research_by_category)
            other_papers = len(classified.get("Other", []))
            
            print(f"  ✅ Publications classified:")
            print(f"     - {categories_with_papers} categories with papers kept")
            if interests_removed:
                print(f"     - Removed {len(interests_removed)} empty categories (no matching papers): {', '.join(interests_removed[:3])}{'...' if len(interests_removed) > 3 else ''}")
            if other_papers > 0:
                print(f"     - {other_papers} papers classified as 'Other' (not shown in profile)")
            if papers_without_authors_in_classification > 0:
                print(f"\n⚠️  WARNING: {papers_without_authors_in_classification}/{total_classified_papers} classified papers missing authors!")
                print(f"    → This will cause paper_score = 0.0 for this candidate.")
        else:
            print(f"  ⚠️  Cannot classify: research_interests={len(research_interests_raw)}, publications={len(publications)}")
    except Exception as e:
        print(f"  ⚠️  Publication classification failed: {e}")
        import traceback
        traceback.print_exc()
        selected_research_by_category = {}
        # 即使分类失败，也要保留已收集的论文
        print(f"  ℹ️  Even though classification failed, {len(all_publications)} papers have been successfully converted")
    
    # Create SelectedResearch object
    selected_research = schemas.SelectedResearch(
        by_category=selected_research_by_category,
        all_publications=all_publications
    )
    
    print(f"   Research Interests: {len(research_interests)} 个")
    for ri in research_interests:
        print(f"     - {ri.name}")
    print(f"   Selected Research by_category: {len(selected_research_by_category)} 个分类")
    for category_name, pubs in selected_research_by_category.items():
        print(f"     - {category_name}: {len(pubs)} 篇论文")
    print(f"   All Publications: {len(all_publications)} 篇")
    # 4. Awards - 使用统一的 awards 列表
    awards = []
    
    # 优先级1：九维度提取
    if nine_dim_data and nine_dim_data.get('awards'):
        awards_dim = nine_dim_data['awards'] or {}
        awards_list = awards_dim.get('awards', []) or []
        
        # 直接转换为 AwardInfo，保留所有字段
        for award_dict in awards_list:
            awards.append(schemas.AwardInfo(
                name=award_dict.get('name', ''),
                year=str(award_dict.get('year', '') or ''),
                organization=award_dict.get('organization', ''),
                paper_title=award_dict.get('paper_title', ''),
                venue=award_dict.get('venue', ''),
                project_name=award_dict.get('project_name', ''),
                description=award_dict.get('description', ''),
                award_type=award_dict.get('award_type', 'personal')
            ))
    
    # Priority 2: AuthorProfile
    elif profile and profile.notable_achievements:
        import re
        for achievement in profile.notable_achievements:
            year_match = re.search(r'\b(20\d{2})\b', achievement)
            year_str = year_match.group(1) if year_match else ""
            awards.append(schemas.AwardInfo(
                name=achievement,
                year=year_str,
                organization=""
            ))
        print(f"  ✅ Awards from multi-source fallback: {len(awards)} items")
    else:
        print(f"  ○ No awards data available")
    
    # 去重：基于 (name, year) 去掉重复奖项
    if awards:
        seen_keys = set()
        deduped_awards = []
        for a in awards:
            key = ((a.name or '').strip().lower(), (a.year or '').strip())
            if key in seen_keys:
                continue
            seen_keys.add(key)
            deduped_awards.append(a)
        if len(deduped_awards) != len(awards):
            print(f"  🧹 Deduplicated awards: {len(awards)} -> {len(deduped_awards)}")
        awards = deduped_awards
    
    # 5. Professional Services - 优先使用九维度提取
    services = []
    
    # 优先级1：九维度提取
    service_source = "none"
    if nine_dim_data and nine_dim_data.get('services'):
        services_data = nine_dim_data['services'].get('services', [])
        talks_data = nine_dim_data['services'].get('invited_talks', [])
        
        for svc_dict in services_data:
            services.append(schemas.ServiceInfo(
                role=svc_dict.get('role', ''),
                venue=svc_dict.get('venue', ''),
                year=svc_dict.get('year', ''),
                description=svc_dict.get('description', '')
            ))
        
        for talk_dict in talks_data:
            services.append(schemas.ServiceInfo(
                role="Invited Talk",
                venue=talk_dict.get('venue', ''),
                year=talk_dict.get('year', ''),
                description=talk_dict.get('title', '')
            ))
        
        if services:
            service_source = "nine_dim_agent"
    
    else:
        print(f"  ○ No services data available")
    
    print(f"  📊 Professional services collected: {len(services)} entries (source={service_source})")
    
    # 6. Career & Education History - 合并所有非工业界职业经历（学生阶段 + 学术职位 + 课程教学）
    career_education_items = []
    # 去重逻辑：仅基于 degree_key 进行去重
    # 如果 degree_key 相同，优先保留 OpenReview 的数据
    # 只有当 degree_key 唯一时，才保留其他来源的数据
    seen_career_edu_degree_keys = {}  # degree_key -> (index, source) 其中 source 是 'openreview' 或 'nine_dim'

    # 优先级1：OpenReview baseline（统一 Career & Education History）
    or_career_edu = openreview_data.get('or_career_education', [])
    if or_career_edu:
        for item_dict in or_career_edu:
            degree_or_pos = (item_dict.get('degree_or_position') or item_dict.get('degree') or item_dict.get('position') or '').strip()
            institution = (item_dict.get('institution', '') or '').strip()
            department = item_dict.get('department', '')
            duration = item_dict.get('duration', '')
            field = item_dict.get('field', '')
            advisor = item_dict.get('advisor', '')
            description = item_dict.get('description', '')
            
            # 如果 field 为空，尝试从 degree_or_position 中提取（处理 "Ph.D. in XXX" 格式）
            if not field and degree_or_pos:
                extracted_field = _extract_field_from_degree_title(degree_or_pos)
                if extracted_field:
                    field = extracted_field

            if not degree_or_pos and not institution:
                continue
            
            # 提前过滤掉 institution 为占位符值的情况（避免通过验证函数）
            invalid_institution_values = [
                "not specified", "n/a", "na", "unknown", 
                "none", "tbd", "to be determined", "待定", "未指定"
            ]
            if institution and institution.lower() in invalid_institution_values:
                print(f"  ⚠️ Career/Edu OR: Skipped placeholder institution - {degree_or_pos} at '{institution}'")
                continue

            # 数据验证：检查记录是否合理
            item_dict_for_validation = {
                'degree_or_position': degree_or_pos,
                'institution': institution,
                'duration': duration
            }
            if not _validate_career_education_item(item_dict_for_validation):
                print(f"  ⚠️ Career/Edu OR: Skipped invalid item - {degree_or_pos} at {institution}")
                continue

            # 使用新的归一化函数：只保留映射表中的内容
            norm_key = _normalize_career_education_label(degree_or_pos)
            if not norm_key:
                # OpenReview 数据优先保留，但如果无法映射到标准标签，仍然跳过
                print(f"  ⚠️ Career/Edu OR: Skipped unmapped label - {degree_or_pos} at {institution}")
                continue

            # 仅使用 degree_key 作为去重 key
            degree_key = norm_key
            
            new_item = schemas.CareerEducationInfo(
                degree_or_position=degree_or_pos,
                institution=institution,
                department=department,
                duration=_normalize_duration_format(duration),  # 统一时间段格式
                field=field,
                advisor=advisor,
                description=description,
            )

            # 检查是否已存在相同的 degree_key
            if degree_key in seen_career_edu_degree_keys:
                # 已存在相同的 degree_key，比较信息量，保留更完整的那条
                idx, existing_source = seen_career_edu_degree_keys[degree_key]
                existing = career_education_items[idx]

                existing_score = int(bool(existing.duration)) + int(bool(existing.field)) + int(bool(existing.advisor)) + int(bool(existing.department))
                new_score = int(bool(new_item.duration)) + int(bool(new_item.field)) + int(bool(new_item.advisor)) + int(bool(new_item.department))

                if (new_score > existing_score) or (
                    new_score == existing_score and len(new_item.degree_or_position or "") > len(existing.degree_or_position or "")
                ):
                    career_education_items[idx] = new_item
                    # 保持来源为 'openreview'（因为这是 OpenReview 数据）
                    seen_career_edu_degree_keys[degree_key] = (idx, 'openreview')
                    print(f"  🔄 Career/Edu OR: Replaced - {degree_or_pos} at {institution}")
                continue

            # 首次出现这个 degree_key：正常新增
            career_education_items.append(new_item)
            seen_career_edu_degree_keys[degree_key] = (len(career_education_items) - 1, 'openreview')

        print(f"  ✅ Career/Edu from OR baseline: {len(career_education_items)} items")

    # 优先级2：九维度提取（仅补充 OR 没有覆盖到的职业经历，OR 为第一来源）
    if nine_dim_data and nine_dim_data.get('career_education'):
        career_edu_data = nine_dim_data['career_education'].get('career_education', [])
        added_from_9d = 0
        skipped_unmapped = 0
        
        for item_dict in career_edu_data:
            degree_or_pos = (item_dict.get('degree_or_position', '') or item_dict.get('degree', '') or item_dict.get('position', '')).strip()
            institution = (item_dict.get('institution', '') or '').strip()
            department = item_dict.get('department', '')
            duration = item_dict.get('duration', '')
            field = item_dict.get('field', '')
            advisor = item_dict.get('advisor', '')
            description = item_dict.get('description', '')
            
            # 如果 field 为空，尝试从 degree_or_position 中提取（处理 "Ph.D. in XXX" 格式）
            if not field and degree_or_pos:
                extracted_field = _extract_field_from_degree_title(degree_or_pos)
                if extracted_field:
                    field = extracted_field

            if not degree_or_pos and not institution:
                continue
            
            # 提前过滤掉 institution 为占位符值的情况（避免通过验证函数）
            invalid_institution_values = [
                "not specified", "n/a", "na", "unknown", 
                "none", "tbd", "to be determined", "待定", "未指定"
            ]
            if institution and institution.lower() in invalid_institution_values:
                print(f"  ⚠️ Career/Edu 9D: Skipped placeholder institution - {degree_or_pos} at '{institution}'")
                continue

            # 数据验证：检查记录是否合理
            item_dict_for_validation = {
                'degree_or_position': degree_or_pos,
                'institution': institution,
                'duration': duration
            }
            if not _validate_career_education_item(item_dict_for_validation):
                print(f"  ⚠️ Career/Edu 9D: Skipped invalid item - {degree_or_pos} at {institution}")
                continue

            # 使用新的归一化函数：只保留映射表中的内容，如果无法映射则直接舍弃
            norm_key = _normalize_career_education_label(degree_or_pos)
            if not norm_key:
                # 从 homepage 获取时，如果无法映射到标准标签，直接舍弃
                skipped_unmapped += 1
                print(f"  ⚠️ Career/Edu 9D: Skipped unmapped label (not in mapping table) - {degree_or_pos} at {institution}")
                continue

            # 仅使用 degree_key 作为去重 key
            degree_key = norm_key
            
            # 如果已存在相同的 degree_key，优先保留 OpenReview 的数据，跳过 9D
            if degree_key in seen_career_edu_degree_keys:
                idx, source = seen_career_edu_degree_keys[degree_key]
                print(f"  ⚠️ Career/Edu 9D: Skipped duplicate degree_key (priority: {source}) - {degree_or_pos} at {institution}")
                continue

            new_item = schemas.CareerEducationInfo(
                degree_or_position=degree_or_pos,
                institution=institution,
                department=department,
                duration=_normalize_duration_format(duration),  # 统一时间段格式
                field=field,
                advisor=advisor,
                description=description
            )

            # 首次出现这个 degree_key：正常新增
            career_education_items.append(new_item)
            seen_career_edu_degree_keys[degree_key] = (len(career_education_items) - 1, 'nine_dim')
            added_from_9d += 1
        
        if skipped_unmapped > 0:
            print(f"  ⚠️ Career/Edu 9D: Skipped {skipped_unmapped} items with unmapped labels")

        if added_from_9d > 0:
            print(f"  ✅ Career/Edu from 9D: added {added_from_9d}, total {len(career_education_items)} items")
    
    # 排序 Career & Education History：按时间从新到旧
    if career_education_items:
        career_education_items = _sort_career_education_items(career_education_items)
        print(f"  🔄 Sorted {len(career_education_items)} career/education items by time (newest first)")
    else:
        print(f"  ○ No career & education history available")
    # 7. Industrial Experience - 使用 OR baseline + 其它源补充
    experience_items = []
    seen_exp_keys = set()

    # 优先级1：OpenReview baseline（已分类好的工业/公司经历）
    or_experience = openreview_data.get('or_industrial_experience', [])
    if or_experience:
        for exp_dict in or_experience:
            position = (exp_dict.get('position', '') or '').strip()
            organization = (exp_dict.get('company') or exp_dict.get('organization') or exp_dict.get('institution') or '').strip()
            duration = exp_dict.get('duration', '')
            description = exp_dict.get('description', '')

            if not position and not organization:
                continue

            # 使用 (position, organization, duration_key) 作为去重 key，保留同一机构不同时间段的同名职位
            duration_key = _normalize_duration_for_key(duration)
            key = (position.lower().strip(), organization.lower().strip(), duration_key)
            if key[0] and key[1] and key in seen_exp_keys:
                continue

            experience_items.append(schemas.ExperienceInfo(
                position=position,
                organization=organization,
                duration=duration,
                description=description
            ))
            if key[0] and key[1]:
                seen_exp_keys.add(key)

        print(f"  ✅ Experience from OR baseline: {len(experience_items)} items")

    # 优先级2：九维度提取（用于补全 OR 没有的工业经历）
    if nine_dim_data and nine_dim_data.get('experience'):
        exp_data = nine_dim_data['experience'].get('experiences', [])
        added_from_9d = 0
        
        for exp_dict in exp_data:
            position = (exp_dict.get('position', '') or '').strip()
            organization = (exp_dict.get('company') or exp_dict.get('organization') or '').strip()
            duration = exp_dict.get('duration', '')
            description = exp_dict.get('description', '')

            if not position and not organization:
                continue

            # 使用 (position, organization, duration_key) 作为去重 key，保留同一机构不同时间段的同名职位
            duration_key = _normalize_duration_for_key(duration)
            key = (position.lower().strip(), organization.lower().strip(), duration_key)
            if key[0] and key[1] and key in seen_exp_keys:
                continue

            experience_items.append(schemas.ExperienceInfo(
                position=position,
                organization=organization,
                duration=duration,
                description=description
            ))
            if key[0] and key[1]:
                seen_exp_keys.add(key)
            added_from_9d += 1

        if added_from_9d > 0:
            print(f"  ✅ Experience from other sources (9D): added {added_from_9d} items, total {len(experience_items)} items")
    
    # 排序 Industrial Experience：按时间从新到旧
    if experience_items:
        experience_items = _sort_experience_items(experience_items)
        print(f"  🔄 Sorted {len(experience_items)} industrial experience items by time (newest first)")
    else:
        print(f"  ○ No industrial experience data available")

    # 统一裁决：根据职位标题重新划分 Career & Education vs Industrial，并去重
    career_education_items, experience_items = _reconcile_career_and_industrial_lists(
        career_education_items,
        experience_items,
    )
    print(f"  🔄 Reconciled career & industrial lists: {len(career_education_items)} career items, {len(experience_items)} industrial items")
    
    if experience_items:
        experience_items = _sort_experience_items(experience_items)
        print(f"  🔄 Re-sorted {len(experience_items)} industrial experience items by time (newest first)")
    if career_education_items:
        career_education_items = _sort_career_education_items(career_education_items)
        print(f"  🔄 Re-sorted {len(career_education_items)} career & education items by time (newest first)")
    if not research_interests:
        print(f"  ⚠️  All research interests generation strategies failed")
        if profile and profile.interests:
            print(f"  🔄 Final fallback: using {len(profile.interests)} original tags")
            for interest_text in profile.interests[:3]:
                if interest_text and len(interest_text) > 2:
                    research_interests.append(schemas.ResearchInterest(
                        name=interest_text,
                        description=""
                    ))
                    research_interests_raw.append({'name': interest_text, 'description': ''})
            print(f"  ✅ Fallback research interests: {len(research_interests)} items")
        else:
            print(f"  ❌ Unable to generate any research interests")
    
    # 9. Contact - 优先使用九维度提取
    email_final = ""
    phone_final = ""
    social_links_final = {}
    
    # 优先级1：九维度提取
    if nine_dim_data and nine_dim_data.get('contact'):
        contact_data = nine_dim_data['contact']
        email_final = contact_data.get('email', '')
        phone_final = contact_data.get('phone', '')
        social_links_final = contact_data.get('social_links', {})
        print(f"  ✅ Contact from 9D extraction")
    
    # 优先级2：AuthorProfile（Phase 3 已补充）
    if not email_final and profile and profile.emails:
        email_final = profile.emails[0]
        print(f"  ✅ Email from multi-source")
    
    # 验证email格式（最终检查）
    if email_final:
        if not _is_valid_email_format(email_final):
            print(f"  ⚠️  Invalid email format, discarding: {email_final}")
            email_final = ""
        elif '@openreview.net' in email_final.lower():
            print(f"  ⚠️  OpenReview placeholder email, discarding: {email_final}")
            email_final = ""
    
    # 合并社交链接（九维度 + profile.platforms + personal_links + openreview）
    platforms_final = profile.platforms if profile else {}
    personal_links = openreview_data.get('personal_links', {})
    openreview_url = openreview_data.get('openreview_url', '') or openreview_data.get('profile_url', '')
    
    # 构建完整的 contact 信息，包含所有平台链接
    contact = schemas.ContactInfo(
        email=email_final,
        phone=phone_final,
        # Homepage: 优先九维度提取 > profile > personal_links
        homepage=social_links_final.get('homepage', '') or platforms_final.get('homepage', '') or personal_links.get('homepage', ''),
        # Google Scholar: 优先九维度提取 > profile > personal_links
        google_scholar=social_links_final.get('google_scholar', '') or social_links_final.get('scholar', '') or platforms_final.get('scholar', '') or personal_links.get('google_scholar', ''),
        # GitHub
        github=social_links_final.get('github', '') or platforms_final.get('github', '') or personal_links.get('github', ''),
        # Twitter
        twitter=social_links_final.get('twitter', '') or platforms_final.get('twitter', '') or personal_links.get('twitter', ''),
        # LinkedIn
        linkedin=social_links_final.get('linkedin', '') or platforms_final.get('linkedin', '') or personal_links.get('linkedin', ''),
        # ORCID
        orcid=social_links_final.get('orcid', '') or platforms_final.get('orcid', '') or personal_links.get('orcid', ''),
        # DBLP: 优先九维度提取 > profile > personal_links
        dblp=social_links_final.get('dblp', '') or platforms_final.get('dblp', '') or personal_links.get('dblp', ''),
        # OpenReview: 添加 OpenReview 档案链接
        openreview=openreview_url
    )
    
    print(f"  ✅ Contact info assembled")
    print(f"    - Email: {email_final[:30] if email_final else 'None'}...")
    all_links = [contact.homepage, contact.google_scholar, contact.github, contact.linkedin, contact.orcid, contact.dblp, contact.openreview]
    print(f"    - Social links: {len([v for v in all_links if v])} platforms")
    if openreview_url:
        print(f"    - OpenReview: ✓ ({openreview_url})")
    else:
        print(f"    - OpenReview: ✗ (missing)")
    
    # 构建完整档案（即使部分字段为空也返回）
    try:
        # 记录数据来源
        data_sources = ['OpenReview']
        if profile:
            if profile.homepage_url:
                data_sources.append('Homepage')
            if profile.platforms.get('scholar'):
                data_sources.append('Google Scholar')
            if profile.platforms.get('orcid'):
                data_sources.append('ORCID')
        
        enhanced_profile = schemas.EnhancedAuthorProfile(
            introduction=introduction,
            research_interests=research_interests,
            selected_research=selected_research,
            awards=awards,
            professional_services=services,
            career_education_history=career_education_items,
            industrial_experience=experience_items,
            contact=contact,
            data_sources=data_sources,
            high_quality_publications=high_quality_pubs  # Add scored publications
        )
        
        print(f"✅ Profile: {author_name} ({len(research_interests)} interests, {len(all_publications)} pubs)")
        print(f"   High-quality publications (score ≥ 6): {len(high_quality_pubs)} papers")
        
        # 🆕 在返回前，根据 career_education_history 和 industrial_experience 决定当前 role
        print(f"\n[Role Determination] Determining current role from career history...")
        from .role_determination import determine_current_role_from_profile
        
        role_category, role_text, affiliation, explanation = determine_current_role_from_profile(
            enhanced_profile=enhanced_profile,
            api_key=api_key
        )

        role_category = (role_category or "Unknown").strip() or "Unknown"
        role_text = (role_text or "").strip()
        affiliation = (affiliation or "").strip()
        explanation = explanation or ""

        missing_position = _is_placeholder_text(role_text)
        missing_affiliation = _is_placeholder_text(affiliation)
        if role_category != "Unknown" and missing_position and missing_affiliation:
            print(
                f"[Role Determination] ⚠️ {author_name}: Downgrading role category to Unknown "
                "because both position and affiliation are missing."
            )
            role_category = "Unknown"
            role_text = ""
            affiliation = ""
            if explanation:
                explanation = explanation + " | Forced Unknown due to missing position/affiliation."
            else:
                explanation = "Forced Unknown due to missing position/affiliation."
        
        enhanced_profile.current_role = schemas.CurrentRoleInfo(
            category=role_category,
            role_text=role_text,
            affiliation=affiliation,
            explanation=explanation
        )
        # 更新 introduction 的 position 和 affiliation
        enhanced_profile.introduction.position = role_text
        enhanced_profile.introduction.affiliation = affiliation
        
        # 生成 intro text 的新策略：
        # 1. 如果有 homepage，检查 9D bio_text 是否真实，真实则使用，否则fallback到结构化生成
        # 2. 如果没有 homepage，基于结构化字段（career + experience + interests）生成真实 background
        has_homepage = bool(profile and profile.homepage_url)
        
        if has_homepage:
            # 有 homepage：检查 9D background 的 bio_text 是否真实
            bio_text_from_9d = ""
            if nine_dim_data and nine_dim_data.get('background'):
                bio_text_from_9d = nine_dim_data['background'].get('bio_text', '').strip()
            
            # 检查 bio_text 是否真实可信
            if bio_text_from_9d:
                is_trustworthy, reason = _is_trustworthy_background_text(bio_text_from_9d, api_key=api_key)
                if is_trustworthy:
                    enhanced_profile.introduction.text = bio_text_from_9d
                    print(f"[Background] ✅ Using 9D bio_text from homepage ({len(bio_text_from_9d)} chars)")
                else:
                    # bio_text 不真实，fallback 到结构化生成方法
                    print(f"[Background] ⚠️ 9D bio_text rejected (reason: {reason}), generating from structured data...")
                    generated_bio = _generate_background_from_structured(
                        author_name=author_name,
                        current_position=role_text,
                        current_affiliation=affiliation,
                        career_education_history=enhanced_profile.career_education_history,
                        industrial_experience=enhanced_profile.industrial_experience,
                        research_interests=enhanced_profile.research_interests,
                        api_key=api_key
                    )
                    if generated_bio:
                        enhanced_profile.introduction.text = generated_bio
            else:
                # 没有 bio_text，fallback 到结构化生成方法
                print(f"[Background] No bio_text from 9D, generating from structured data...")
                generated_bio = _generate_background_from_structured(
                    author_name=author_name,
                    current_position=role_text,
                    current_affiliation=affiliation,
                    career_education_history=enhanced_profile.career_education_history,
                    industrial_experience=enhanced_profile.industrial_experience,
                    research_interests=enhanced_profile.research_interests,
                    api_key=api_key
                )
                if generated_bio:
                    enhanced_profile.introduction.text = generated_bio
        else:
            # 没有 homepage：基于结构化字段生成真实 background
            print(f"[Background] No homepage, generating from structured data...")
            generated_bio = _generate_background_from_structured(
                author_name=author_name,
                current_position=role_text,
                current_affiliation=affiliation,
                career_education_history=enhanced_profile.career_education_history,
                industrial_experience=enhanced_profile.industrial_experience,
                research_interests=enhanced_profile.research_interests,
                api_key=api_key
            )
            if generated_bio:
                enhanced_profile.introduction.text = generated_bio
        
        print(f"[Role Determination] ✅ Updated introduction: {role_text} at {affiliation}")
        
        # Translate all Chinese content to English before returning
        print(f"\n[Translation] Checking and translating Chinese content to English...")
        enhanced_profile = _translate_profile_to_english(enhanced_profile, api_key=api_key)
        
        return enhanced_profile
        
    except Exception as e:
        print(f"❌ Critical error building enhanced profile schema: {e}")
        import traceback
        traceback.print_exc()
        return None


def _contains_chinese(text: str) -> bool:
    """
    Check if text contains Chinese characters.
    Args:
        text: Text to check
    Returns:
        True if text contains Chinese characters, False otherwise
    """
    if not text or not isinstance(text, str):
        return False
    # Check for Chinese characters (CJK Unified Ideographs)
    return bool(re.search(r'[\u4e00-\u9fff]', text))

def _translate_text_to_english(text: str, api_key: str = None) -> str:
    """
    Translate Chinese text to English using LLM.
    Args:
        text: Text to translate (may contain Chinese)
        api_key: LLM API key
    Returns:
        Translated text in English, or original text if translation fails
    """
    if not text or not isinstance(text, str):
        return text
    # Skip if no Chinese characters
    if not _contains_chinese(text):
        return text
    
    try:
        llm_instance = llm.get_llm("translate", temperature=0.1, api_key=api_key)
        
        prompt = f"""Translate the following text to English. 
            Preserve all technical terms, names, and proper nouns exactly as they are.
            Only translate the Chinese content to English, keep English parts unchanged.

            Text to translate:
            {text}

            Return ONLY the translated text in English, nothing else.
        """
        response = llm_instance.invoke(prompt)
        translated = (
            llm.safe_get(response, "content", "")
            or llm.safe_get(response, "text", "")
            or str(response)
        )

        if isinstance(translated, dict) and "text" in translated:
            translated = translated["text"]
        elif isinstance(translated, list):
            translated = "\n".join(str(item) for item in translated)
        elif not isinstance(translated, str):
            translated = str(translated)

        translated = translated.strip()
        
        if translated and len(translated) > 0:
            print(f"[Translation] ✅ Translated {len(text)} chars → {len(translated)} chars")
            return translated
        else:
            print(f"[Translation] ⚠️ Translation returned empty, keeping original")
            return text
            
    except Exception as e:
        print(f"[Translation] ⚠️ Translation failed: {e}, keeping original text")
        return text


def _decode_response_text(response) -> str:
    """
    Decode HTTP responses with a reasonable charset fallback.
    Fixes cases where UTF-8 pages are mislabeled as ISO-8859-1, producing mojibake.
    """
    if response is None:
        return ""
    try:
        content = response.content
    except Exception:
        content = None
    if not content:
        return response.text if hasattr(response, "text") else ""
    encoding = (getattr(response, "encoding", None) or "").strip().lower()
    if not encoding or encoding == "iso-8859-1":
        try:
            detected = getattr(response, "apparent_encoding", None)
        except Exception:
            detected = None
        if detected:
            encoding = detected.strip().lower()
    if not encoding:
        encoding = "utf-8"
    try:
        return content.decode(encoding, errors="replace")
    except Exception:
        return content.decode("utf-8", errors="replace")

def _translate_profile_to_english(profile: schemas.EnhancedAuthorProfile, api_key: str = None) -> schemas.EnhancedAuthorProfile:
    """
    Recursively translate all Chinese content in EnhancedAuthorProfile to English.
    
    Args:
        profile: EnhancedAuthorProfile to translate
        api_key: LLM API key
        
    Returns:
        Profile with all Chinese content translated to English
    """
    if not profile:
        return profile
    
    translation_count = 0
    
    # 1. Introduction
    if profile.introduction:
        if profile.introduction.name and _contains_chinese(profile.introduction.name):
            profile.introduction.name = _translate_text_to_english(profile.introduction.name, api_key)
            translation_count += 1
        if profile.introduction.position and _contains_chinese(profile.introduction.position):
            profile.introduction.position = _translate_text_to_english(profile.introduction.position, api_key)
            translation_count += 1
        if profile.introduction.affiliation and _contains_chinese(profile.introduction.affiliation):
            profile.introduction.affiliation = _translate_text_to_english(profile.introduction.affiliation, api_key)
            translation_count += 1
        if profile.introduction.text and _contains_chinese(profile.introduction.text):
            profile.introduction.text = _translate_text_to_english(profile.introduction.text, api_key)
            translation_count += 1
    if hasattr(profile, "current_role") and profile.current_role:
        current_role = profile.current_role
        if current_role.role_text and _contains_chinese(current_role.role_text):
            current_role.role_text = _translate_text_to_english(current_role.role_text, api_key)
            translation_count += 1
        if current_role.affiliation and _contains_chinese(current_role.affiliation):
            current_role.affiliation = _translate_text_to_english(current_role.affiliation, api_key)
            translation_count += 1
        if current_role.explanation and _contains_chinese(current_role.explanation):
            current_role.explanation = _translate_text_to_english(current_role.explanation, api_key)
            translation_count += 1
    
    # 2. Research Interests
    if profile.research_interests:
        for interest in profile.research_interests:
            if interest.name and _contains_chinese(interest.name):
                interest.name = _translate_text_to_english(interest.name, api_key)
                translation_count += 1
            if interest.description and _contains_chinese(interest.description):
                interest.description = _translate_text_to_english(interest.description, api_key)
                translation_count += 1
    
    # 3. Selected Research (Publications)
    if profile.selected_research:
        # Publications by category
        if profile.selected_research.by_category:
            for category, pubs in profile.selected_research.by_category.items():
                for pub in pubs:
                    if hasattr(pub, 'title') and pub.title and _contains_chinese(pub.title):
                        pub.title = _translate_text_to_english(pub.title, api_key)
                        translation_count += 1
                    if hasattr(pub, 'venue') and pub.venue and _contains_chinese(pub.venue):
                        pub.venue = _translate_text_to_english(pub.venue, api_key)
                        translation_count += 1
                    if hasattr(pub, 'abstract') and pub.abstract and _contains_chinese(pub.abstract):
                        pub.abstract = _translate_text_to_english(pub.abstract, api_key)
                        translation_count += 1
        
        # All publications
        if profile.selected_research.all_publications:
            for pub in profile.selected_research.all_publications:
                if hasattr(pub, 'title') and pub.title and _contains_chinese(pub.title):
                    pub.title = _translate_text_to_english(pub.title, api_key)
                    translation_count += 1
                if hasattr(pub, 'venue') and pub.venue and _contains_chinese(pub.venue):
                    pub.venue = _translate_text_to_english(pub.venue, api_key)
                    translation_count += 1
                if hasattr(pub, 'abstract') and pub.abstract and _contains_chinese(pub.abstract):
                    pub.abstract = _translate_text_to_english(pub.abstract, api_key)
                    translation_count += 1
    
    # 4. Awards
    if profile.awards:
        for award in profile.awards:
            if hasattr(award, 'name') and award.name and _contains_chinese(award.name):
                award.name = _translate_text_to_english(award.name, api_key)
                translation_count += 1
            if hasattr(award, 'description') and award.description and _contains_chinese(award.description):
                award.description = _translate_text_to_english(award.description, api_key)
                translation_count += 1
            if hasattr(award, 'venue') and award.venue and _contains_chinese(award.venue):
                award.venue = _translate_text_to_english(award.venue, api_key)
                translation_count += 1
            if hasattr(award, 'paper_title') and award.paper_title and _contains_chinese(award.paper_title):
                award.paper_title = _translate_text_to_english(award.paper_title, api_key)
                translation_count += 1
            if hasattr(award, 'project_name') and award.project_name and _contains_chinese(award.project_name):
                award.project_name = _translate_text_to_english(award.project_name, api_key)
                translation_count += 1
    
    # 5. Professional Services
    if profile.professional_services:
        for service in profile.professional_services:
            if hasattr(service, 'role') and service.role and _contains_chinese(service.role):
                service.role = _translate_text_to_english(service.role, api_key)
                translation_count += 1
            if hasattr(service, 'venue') and service.venue and _contains_chinese(service.venue):
                service.venue = _translate_text_to_english(service.venue, api_key)
                translation_count += 1
            if hasattr(service, 'description') and service.description and _contains_chinese(service.description):
                service.description = _translate_text_to_english(service.description, api_key)
                translation_count += 1
    
    # 6. Career & Education History
    if profile.career_education_history:
        for item in profile.career_education_history:
            if hasattr(item, 'degree_or_position') and item.degree_or_position and _contains_chinese(item.degree_or_position):
                item.degree_or_position = _translate_text_to_english(item.degree_or_position, api_key)
                translation_count += 1
            if hasattr(item, 'institution') and item.institution and _contains_chinese(item.institution):
                item.institution = _translate_text_to_english(item.institution, api_key)
                translation_count += 1
            if hasattr(item, 'department') and item.department and _contains_chinese(item.department):
                item.department = _translate_text_to_english(item.department, api_key)
                translation_count += 1
            if hasattr(item, 'field') and item.field and _contains_chinese(item.field):
                item.field = _translate_text_to_english(item.field, api_key)
                translation_count += 1
            if hasattr(item, 'advisor') and item.advisor and _contains_chinese(item.advisor):
                item.advisor = _translate_text_to_english(item.advisor, api_key)
                translation_count += 1
            if hasattr(item, 'description') and item.description and _contains_chinese(item.description):
                item.description = _translate_text_to_english(item.description, api_key)
                translation_count += 1
    
    # 7. Industrial Experience
    if profile.industrial_experience:
        for exp in profile.industrial_experience:
            if hasattr(exp, 'position') and exp.position and _contains_chinese(exp.position):
                exp.position = _translate_text_to_english(exp.position, api_key)
                translation_count += 1
            if hasattr(exp, 'company') and exp.company and _contains_chinese(exp.company):
                exp.company = _translate_text_to_english(exp.company, api_key)
                translation_count += 1
            if hasattr(exp, 'description') and exp.description and _contains_chinese(exp.description):
                exp.description = _translate_text_to_english(exp.description, api_key)
                translation_count += 1
    
    # 8. High-quality publications
    if profile.high_quality_publications:
        for pub in profile.high_quality_publications:
            if isinstance(pub, dict):
                if pub.get('title') and _contains_chinese(pub['title']):
                    pub['title'] = _translate_text_to_english(pub['title'], api_key)
                    translation_count += 1
                if pub.get('venue') and _contains_chinese(pub['venue']):
                    pub['venue'] = _translate_text_to_english(pub['venue'], api_key)
                    translation_count += 1
                if pub.get('abstract') and _contains_chinese(pub['abstract']):
                    pub['abstract'] = _translate_text_to_english(pub['abstract'], api_key)
                    translation_count += 1
    if translation_count > 0:
        print(f"[Translation] ✅ Translated {translation_count} field(s) from Chinese to English")
    else:
        print(f"[Translation] ✅ No Chinese content found, profile is already in English")
    return profile

def _generate_background_from_structured(
    author_name: str,
    current_position: str,
    current_affiliation: str,
    career_education_history: List[schemas.CareerEducationInfo],
    industrial_experience: List[schemas.ExperienceInfo],
    research_interests: List[schemas.ResearchInterest],
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    基于结构化字段生成 background bio text（用于没有 homepage 的情况）
    
    只使用已经提取好的真实字段，不允许 LLM 虚构信息
    
    如果没有 Career & Education History 和 Industrial Experience，返回 None（无法判断角色）
    """
    # 检查是否有足够的信息来判断角色
    has_career_education = bool(career_education_history)
    has_industrial_experience = bool(industrial_experience)
    
    # 如果两者都没有，无法判断角色，不生成 background
    if not has_career_education and not has_industrial_experience:
        print(f"[Background] ⚠️ No career/education history or industrial experience, skipping background generation")
        return None
    
    # 构建 career_education_text
    career_edu_lines = []
    if career_education_history:
        for item in career_education_history[:5]:  # 只取前5条
            degree_or_pos = item.degree_or_position or "N/A"
            institution = item.institution or "N/A"
            duration = item.duration or ""
            
            line = f"- {degree_or_pos} at {institution}"
            if duration:
                line += f" ({duration})"
            career_edu_lines.append(line)
    
    has_career_education = bool(career_edu_lines)
    career_education_text = "\n".join(career_edu_lines) if has_career_education else None
    
    # 构建 industrial_experience_text
    industrial_lines = []
    if industrial_experience:
        for item in industrial_experience[:3]:  # 只取前3条
            position = item.position or "N/A"
            organization = item.organization or "N/A"
            duration = item.duration or ""
            
            line = f"- {position} at {organization}"
            if duration:
                line += f" ({duration})"
            industrial_lines.append(line)
    
    has_industrial_experience = bool(industrial_lines)
    industrial_experience_text = "\n".join(industrial_lines) if has_industrial_experience else None
    
    # 构建 research_interests_text
    interests_lines = []
    if research_interests:
        for interest in research_interests[:3]:  # 只取前3个
            interests_lines.append(f"- {interest.name}")
    
    has_research_interests = bool(interests_lines)
    research_interests_text = "\n".join(interests_lines) if has_research_interests else None
    
    # 构建 prompt，只包含有数据的部分
    input_sections = []
    input_sections.append(f"Author Name: {author_name}")
    input_sections.append("")
    input_sections.append("Current Role:")
    input_sections.append(f"- Position: {current_position or 'Unknown'}")
    input_sections.append(f"- Affiliation: {current_affiliation or 'Unknown'}")
    
    if has_career_education:
        input_sections.append("")
        input_sections.append("Career & Education History:")
        input_sections.append(career_education_text)
    
    if has_industrial_experience:
        input_sections.append("")
        input_sections.append("Industrial Experience:")
        input_sections.append(industrial_experience_text)
    
    if has_research_interests:
        input_sections.append("")
        input_sections.append("Research Interests:")
        input_sections.append(research_interests_text)
    
    input_data_text = "\n".join(input_sections)
    
    # LLM Prompt
    prompt = f"""You are a professional academic profile writer. Generate a concise, factual background paragraph based ONLY on the provided structured data.

CRITICAL RULES:
1. **ZERO FABRICATION**: Only use information explicitly provided below
2. **NO ASSUMPTIONS**: Do not infer schools, companies, awards, or positions not listed
3. **EXACT NAMES**: Use institution/company names EXACTLY as provided
4. **CONCISE**: 2-4 sentences, first person ("I am...")
5. **NEVER MENTION MISSING DATA**: Do NOT mention "no publicly listed history", "no education history", or similar phrases about missing information. Only describe what IS provided.

INPUT DATA:
{input_data_text}

TASK:
Write a 2-4 sentence background paragraph that:
1. States current position and affiliation
2. Mentions 1-2 key past positions/degrees (ONLY if provided in Career & Education History section)
3. Briefly mentions research focus (ONLY if Research Interests are provided)

IMPORTANT: If a section is not provided above, do NOT mention its absence. Only describe what is actually available.

OUTPUT: Return ONLY the paragraph (no preamble).
"""
    
    try:
        llm_instance = llm.get_llm("background_gen", temperature=0.3, api_key=api_key)
        response = llm_instance.invoke(prompt)
        
        if hasattr(response, 'content'):
            bio_text = response.content.strip()
        else:
            bio_text = str(response).strip()
        
        # 清理引号
        bio_text = bio_text.strip('"').strip("'")
        
        if bio_text and len(bio_text) > 20:
            print(f"[Background] ✅ Generated from structured data ({len(bio_text)} chars)")
            return bio_text
        else:
            print(f"[Background] ⚠️ Generated text too short, using fallback")
            return _fallback_bio(author_name, current_position, current_affiliation)
            
    except Exception as e:
        print(f"[Background] ❌ Generation error: {e}")
        return _fallback_bio(author_name, current_position, current_affiliation)


def _fallback_bio(author_name: str, current_position: str, current_affiliation: str) -> str:
    """简单的 fallback 模板"""
    if current_position and current_affiliation:
        return f"I am a {current_position} at {current_affiliation}."
    elif current_affiliation:
        return f"I am at {current_affiliation}."
    else:
        return f"I am {author_name}."


# ============================ ORCHESTRATOR ============================

def orchestrate_candidate_report(first_author: str, paper_title: str, paper_url: str = None, paper_venue: str = None,
                                 paper_score: int = 0, paper_explanation: str = "", paper_authors: List[str] = None,
                                 aliases: List[str] = None, k_queries: int = 40, author_id: str = None,
                                 s2_author_id: str = None,
                                 api_key: str = None, use_lightweight_mode: bool = False, user_query: str = None, 
                                 exclude_paper_urls: set = None,
                                 paper_info: Optional[Dict[str, Any]] = None,
                                 is_corresponding_author: bool = False,
                                 is_cofirst_author: bool = False) -> Tuple[Optional[AuthorProfile], Optional[schemas.CandidateOverview], Optional[schemas.EvaluationResult], Optional[schemas.EnhancedAuthorProfile]]:
    """
    Run discovery with homepage enforcement, build 9-field profile, evaluate 4D, and return overview.
    NEW: CandidateOverview now uses EnhancedAuthorProfile's 9-field structure directly.
    Args:
        exclude_paper_urls: Set of paper URLs to exclude from candidate's publication list (to avoid duplication with Reference Papers)
        is_corresponding_author: Whether this candidate is a corresponding author on the trigger paper (for higher weight)
        is_cofirst_author: Whether this candidate is a co-first author on the trigger paper (gets first-author weight)
    """
    if exclude_paper_urls is None:
        exclude_paper_urls = set()
    # 尝试获取作者 profile，异常时设置为 None（后续可能触发 Web Search Fallback）
    profile = None
    try:
        profile = discover_author_profile(first_author, paper_title, aliases, k_queries=k_queries, author_id=author_id, api_key=api_key, paper_info=paper_info)
    except Exception as e:
        print(f"[orchestrate] ⚠️ discover_author_profile failed: {e}")
        import traceback
        traceback.print_exc()
        profile = None

    # 🎯 构造 trigger paper 字典（用于强制包含在候选人的论文列表中）
    trigger_paper_dict = None
    if paper_title and paper_url:
        # 在第一次触发阶段就记录候选人在该论文中的作者位置，后续不再依赖名字匹配
        candidate_position_index = None
        total_authors = len(paper_authors) if paper_authors else None
        if paper_authors and first_author:
            first_author_stripped = first_author.strip().lower()
            for idx, name in enumerate(paper_authors, 1):
                if name and name.strip().lower() == first_author_stripped:
                    candidate_position_index = idx
                    break
        # 从 paper_info 中提取多维度评分字段
        relevance_score = paper_score  # Default to paper_score
        quality_score = 0.0
        relevance_dimensions = {}
        quality_dimensions = {}
        if paper_info:
            relevance_score = paper_info.get('relevance_score', float(paper_score))
            quality_score = paper_info.get('quality_score', 0.0)
            relevance_dimensions = paper_info.get('relevance_dimensions', {})
            quality_dimensions = paper_info.get('quality_dimensions', {})
        trigger_paper_dict = {
            'title': paper_title,
            'url': paper_url,
            'venue': paper_venue or '',
            'score': paper_score,
            'explanation': paper_explanation,
            'authors': paper_authors or [],  # 添加作者列表，用于计算 paper score
            # 预先写入候选人在触发论文中的作者位置，避免后续再次计算导致不一致
            'candidate_author_position': candidate_position_index,
            'total_authors': total_authors,
            # 多维度评分字段 (10 dimensions: 5 relevance + 5 quality)
            'relevance_score': relevance_score,
            'quality_score': quality_score,
            'relevance_dimensions': relevance_dimensions,
            'quality_dimensions': quality_dimensions,
            # Corresponding author flag (detected from PDF)
            'is_corresponding_author': is_corresponding_author,
            # Co-first author flag (detected from PDF)
            'is_cofirst_author': is_cofirst_author,
            # Note: year, abstract 等字段可能缺失，collect函数会处理
        }
    
    # Build Enhanced Profile FIRST (this IS the candidate overview now)
    enhanced_profile = None
    if profile and hasattr(profile, '_openreview_data') and profile._openreview_data:
        try:
            # ✅ 传递 profile 参数，使用多源采集的数据
            enhanced_profile = build_enhanced_profile(
                profile._openreview_data,
                first_author,
                api_key=api_key,
                profile=profile,  # 新增：传递包含多源数据的 profile
                user_query=user_query,  # 新增：传递 user_query 用于论文评分
                exclude_paper_urls=exclude_paper_urls,  # 新增：排除已在Reference Papers中的论文
                trigger_paper=trigger_paper_dict,
                s2_author_id=s2_author_id  # 传递 S2 authorId 用于获取论文
            )
        except Exception as e:
            print(f"[orchestrate] ⚠️  Enhanced profile build failed: {e}")
            enhanced_profile = None
    elif profile:
        # profile 存在但没有 OpenReview 数据
        print(f"[orchestrate] ⚠️  No OpenReview data available for enhanced profile")
        # Try to fetch fresh OpenReview data
        try:
            print(f"[orchestrate] 🔄 Attempting to fetch fresh OpenReview data for {first_author}...")
            from . import extraction
            openreview_result = extraction.search_openreview_profile(first_author, api_key=api_key)
            if openreview_result:
                print(f"[orchestrate] ✅ Successfully fetched fresh OpenReview data")
                profile._openreview_data = openreview_result
                # ✅ 传递 profile 参数
                enhanced_profile = build_enhanced_profile(
                    openreview_result,
                    first_author,
                    api_key=api_key,
                    profile=profile,  # 新增：传递包含多源数据的 profile
                    user_query=user_query,  # 新增：传递 user_query 用于论文评分
                    exclude_paper_urls=exclude_paper_urls,  # 新增：排除已在Reference Papers中的论文
                    trigger_paper=trigger_paper_dict,
                    s2_author_id=s2_author_id  # 传递 S2 authorId 用于获取论文
                )
            else:
                print(f"[orchestrate] ❌ Failed to fetch OpenReview data")
        except Exception as e:
            print(f"[orchestrate] ❌ Error fetching fresh OpenReview data: {e}")
    else:
        # profile 为 None，直接进入 Web Search Fallback
        print(f"[orchestrate]  No profile available, will try Web Search Fallback")
    
    # If no enhanced profile, try Web Search Fallback
    if not enhanced_profile:
        print(f"[orchestrate] No enhanced profile, trying Web Search Fallback...")
        print(f"[orchestrate] Debug - Search parameters:")
        print(f"  - Author name: {first_author}")
        print(f"  - Paper title: {paper_title}")
        
        try:
            from .web_search_fallback import ( 
                WebSearchFallbackDiscovery,
                convert_web_candidate_to_enhanced_profile
            )
            
            # 从 profile 或 paper_info 获取已知的 affiliation
            known_affiliation = ""
            if profile and hasattr(profile, 'affiliation_current') and profile.affiliation_current:
                known_affiliation = profile.affiliation_current
            elif paper_info and paper_info.get('affiliations'):
                # 如果论文信息中有作者机构
                affiliations = paper_info.get('affiliations', [])
                if affiliations:
                    known_affiliation = affiliations[0] if isinstance(affiliations, list) else str(affiliations)       
            print(f"  - Known affiliation: {known_affiliation or '(none)'}")
            # 执行 Web Search Fallback
            fallback = WebSearchFallbackDiscovery(api_key=api_key)
            web_candidate = fallback.discover_candidate(
                author_name=first_author,
                known_affiliation=known_affiliation,
                paper_title=paper_title
            )
            
            if web_candidate:
                print(f"[orchestrate] Web Search Fallback succeeded for: {web_candidate.name}")
                # 转换为 EnhancedAuthorProfile
                enhanced_profile = convert_web_candidate_to_enhanced_profile(
                    web_candidate,
                    trigger_paper=trigger_paper_dict,
                    user_query=user_query
                )
                print(f"[orchestrate] Converted to EnhancedAuthorProfile")
            else:
                print(f"[orchestrate] Web Search Fallback returned no result")
                
        except Exception as e:
            print(f"[orchestrate] Web Search Fallback error: {e}")
            import traceback
            traceback.print_exc()
    
    # Final check: if still no enhanced profile, cannot proceed
    if not enhanced_profile:
        print(f"[orchestrate] Cannot build overview without enhanced profile (all sources failed)")
        return None, None, None, None
    
    # NEW: Use 4-dimension parallel evaluation (REQUIRED: must have user_query)
    if not user_query:
        print("[orchestrate] ❌ ERROR: No user_query provided, cannot evaluate candidate")
        return None, None, None, None
    
    eval_res = evaluate_profile_4d_parallel(profile, enhanced_profile, user_query, api_key=api_key)
    # Build CandidateOverview from EnhancedAuthorProfile (9-field structure)
    overview = build_candidate_overview_from_enhanced(
        enhanced_profile=enhanced_profile,
        eval_result=eval_res,
        trigger_paper_title=paper_title,
        trigger_paper_url=paper_url,
        trigger_paper_venue=paper_venue,
        trigger_paper_score=paper_score,
        trigger_paper_explanation=paper_explanation
    )
    return profile, overview, eval_res, enhanced_profile
# ============================ PROFILE REFINEMENT ============================

def enhance_career_stage_detection(profile: AuthorProfile) -> str:
    """
    增强的career stage检测，从多个来源综合判断
    Args:
        profile: 作者档案
    Returns:
        推断的career stage
    """
    stage_indicators = []
    
    # 1. 从affiliation中提取线索
    if profile.affiliation_current:
        affiliation_lower = profile.affiliation_current.lower()
        
        if any(keyword in affiliation_lower for keyword in ['professor', 'prof']):
            if 'assistant' in affiliation_lower:
                stage_indicators.append(('assistant_prof', 0.8))
            elif 'associate' in affiliation_lower:
                stage_indicators.append(('associate_prof', 0.8))
            elif 'full' in affiliation_lower or 'chair' in affiliation_lower:
                stage_indicators.append(('full_prof', 0.8))
            else:
                stage_indicators.append(('professor', 0.6))
        elif any(keyword in affiliation_lower for keyword in ['postdoc', 'postdoctoral', 'research fellow']):
            stage_indicators.append(('postdoc', 0.8))
        elif any(keyword in affiliation_lower for keyword in ['phd student', 'doctoral student', 'graduate student']):
            stage_indicators.append(('phd_student', 0.8))
        elif any(keyword in affiliation_lower for keyword in ['researcher', 'scientist']):
            if any(company in affiliation_lower for company in ['google', 'microsoft', 'amazon', 'meta', 'openai', 'anthropic']):
                stage_indicators.append(('industry_researcher', 0.7))
            else:
                stage_indicators.append(('researcher', 0.6))
        elif any(keyword in affiliation_lower for keyword in ['engineer', 'developer', 'manager']):
            stage_indicators.append(('industry', 0.7))
    
    # 2. 从notable achievements中提取线索
    for achievement in profile.notable_achievements:
        achievement_lower = achievement.lower()
        
        if any(keyword in achievement_lower for keyword in ['dissertation award', 'phd thesis']):
            stage_indicators.append(('recent_phd', 0.6))
        elif any(keyword in achievement_lower for keyword in ['young researcher', 'rising star', 'early career']):
            stage_indicators.append(('early_career', 0.7))
        elif any(keyword in achievement_lower for keyword in ['fellow', 'distinguished']):
            stage_indicators.append(('senior_researcher', 0.8))
    
    # 3. 从social impact中提取线索
    if profile.social_impact:
        impact_lower = profile.social_impact.lower()
        
        # 解析h-index和citations来推断career stage
        import re
        h_index_match = re.search(r'h-?index[:\s]*(\d+)', impact_lower)
        citation_match = re.search(r'citation[s]?[:\s]*(\d+)', impact_lower)
        paper_match = re.search(r'paper[s]?[:\s]*(\d+)', impact_lower)
        
        h_index = int(h_index_match.group(1)) if h_index_match else 0
        citations = int(citation_match.group(1)) if citation_match else 0
        papers = int(paper_match.group(1)) if paper_match else 0
        
        # 根据学术指标推断career stage
        if h_index >= 30 or citations >= 5000:
            stage_indicators.append(('senior_researcher', 0.7))
        elif h_index >= 15 or citations >= 1000:
            stage_indicators.append(('mid_career', 0.6))
        elif h_index >= 5 or citations >= 200:
            stage_indicators.append(('early_career', 0.6))
        elif papers <= 5 and citations <= 100:
            stage_indicators.append(('student_or_early', 0.5))
    
    # 4. 综合判断
    if not stage_indicators:
        return "unknown"
    
    # 按置信度排序，选择最可能的stage
    stage_indicators.sort(key=lambda x: x[1], reverse=True)
    best_stage, best_confidence = stage_indicators[0]
    
    # 如果有多个高置信度的指标，进行进一步判断
    high_confidence_stages = [stage for stage, conf in stage_indicators if conf >= 0.7]
    
    if len(high_confidence_stages) > 1:
        # 优先级：教授 > 研究员 > 博士后 > 学生
        priority_order = ['full_prof', 'associate_prof', 'assistant_prof', 'professor', 
                         'senior_researcher', 'industry_researcher', 'researcher', 
                         'postdoc', 'phd_student', 'student_or_early']
        
        for priority_stage in priority_order:
            if priority_stage in high_confidence_stages:
                return priority_stage
    
    return best_stage

def refine_author_profile(profile: AuthorProfile, target_author: str) -> AuthorProfile:
    """最终精炼作者档案，确保数据质量"""
    
    # 1. 清理aliases - 移除明显不相关的名字
    target_words = set(target_author.lower().split())
    refined_aliases = []
    
    for alias in profile.aliases:
        if not alias or alias == profile.name:
            continue
            
        alias_words = set(alias.lower().split())
        
        # 更严格的别名检查
        is_valid_alias = False
        
        # 1. 检查是否有共同的实质性词汇（长度>2）
        common_words = [word for word in (alias_words & target_words) if len(word) > 2]
        if len(common_words) > 0:
            is_valid_alias = True
        
        # 2. 检查是否是名字的部分或变体
        target_first = target_author.split()[0].lower() if target_author.split() else ""
        target_last = target_author.split()[-1].lower() if len(target_author.split()) > 1 else ""
        
        if (target_first and target_first in alias.lower()) or (target_last and target_last in alias.lower()):
            is_valid_alias = True
        
        # 3. 排除明显不相关的名字
        if any(bad_indicator in alias.lower() for bad_indicator in ['rex', 'cook', 'evans', 'dante', 'ortega', 'camerino']):
            is_valid_alias = False
        
        # 4. 排除过长的名字（可能是其他人）
        if len(alias.split()) > 4:
            is_valid_alias = False
        
        if is_valid_alias:
            refined_aliases.append(alias)
    
    profile.aliases = refined_aliases[:3]  # 限制为最多3个别名
    
    # 2. 验证和清理平台链接
    verified_platforms = {}
    for platform, url in profile.platforms.items():
        if url and url.startswith('http') and len(url) > 10:
            # 基本URL验证
            verified_platforms[platform] = url
    
    profile.platforms = verified_platforms
    
    # 3. 清理兴趣领域 - 去重和规范化
    refined_interests = []
    seen_interests = set()
    
    for interest in profile.interests:
        if interest:
            # 规范化兴趣描述
            normalized = interest.strip().lower()
            if normalized not in seen_interests and len(normalized) > 2:
                seen_interests.add(normalized)
                refined_interests.append(interest.strip())
    
    profile.interests = refined_interests[:8]  # 限制兴趣数量
    
    # 4. 清理论文列表
    refined_publications = []
    seen_titles = set()
    
    for pub in profile.selected_publications:
        if isinstance(pub, dict) and pub.get('title'):
            title_normalized = pub['title'].lower().strip()
            if title_normalized not in seen_titles:
                seen_titles.add(title_normalized)
                refined_publications.append(pub)
    
    # 限制论文数量 to 10
    profile.selected_publications = refined_publications[:10]  
    
    # 5. 清理Notable成就
    refined_achievements = []
    for achievement in profile.notable_achievements:
        if achievement and len(achievement.strip()) > 5:
            refined_achievements.append(achievement.strip())
    
    # 限制成就数量 to 10
    profile.notable_achievements = refined_achievements[:10] 
    
    # 6. 增强career stage检测
    if not profile.career_stage or profile.career_stage == "assistant_prof":  # 如果没有或者是默认值
        enhanced_stage = enhance_career_stage_detection(profile)
        if enhanced_stage and enhanced_stage != "unknown":
            profile.career_stage = enhanced_stage
            print(f"[Enhanced Career Stage] Updated to: {enhanced_stage}")
    
    return profile

# ============================ SCORING SYSTEM ============================

def calculate_overall_score(profile: AuthorProfile) -> float:
    """计算作者的综合评分 (0-100)"""
    score = 0.0
    
    # 1. 平台权威性评分 (0-25分)
    platform_score = 0
    platform_weights = {
        'orcid': 8, 'openreview': 7, 'scholar': 6, 'semanticscholar': 5, 
        'university': 6, 'github': 3, 'homepage': 4
    }
    for platform in profile.platforms:
        if platform in platform_weights:
            platform_score += platform_weights[platform]
    score += min(25, platform_score)
    
    # 2. 信息完整性评分 (0-20分)
    completeness = 0
    if profile.affiliation_current: completeness += 4
    if profile.emails: completeness += 3
    if profile.interests: completeness += 4
    if profile.homepage_url: completeness += 3
    if len(profile.aliases) > 0: completeness += 2
    if len(profile.selected_publications) > 0: completeness += 4
    score += completeness
    
    # 3. Notable成就评分 (0-25分)
    notable_score = 0
    if profile.notable_achievements:
        for achievement in profile.notable_achievements:
            achievement_lower = achievement.lower()
            if any(keyword in achievement_lower for keyword in 
                   ['best paper', 'outstanding paper', 'award']):
                notable_score += 8
            elif any(keyword in achievement_lower for keyword in 
                     ['fellow', 'ieee fellow', 'acm fellow']):
                notable_score += 10
            elif any(keyword in achievement_lower for keyword in 
                     ['rising star', 'young researcher']):
                notable_score += 6
            elif any(keyword in achievement_lower for keyword in 
                     ['keynote', 'invited speaker']):
                notable_score += 5
            elif any(keyword in achievement_lower for keyword in 
                     ['startup', 'founder', 'entrepreneur']):
                notable_score += 4
            else:
                notable_score += 2
    score += min(25, notable_score)
    
    # 4. 学术影响力评分 (0-20分)
    impact_score = 0
    if profile.social_impact:
        impact_text = profile.social_impact.lower()
        # 解析h-index
        import re
        h_index_match = re.search(r'h-?index[:\s]*(\d+)', impact_text)
        if h_index_match:
            h_index = int(h_index_match.group(1))
            if h_index >= 50: impact_score += 20
            elif h_index >= 30: impact_score += 15
            elif h_index >= 20: impact_score += 12
            elif h_index >= 10: impact_score += 8
            elif h_index >= 5: impact_score += 5
        
        # 解析引用数
        citation_match = re.search(r'citation[s]?[:\s]*(\d+)', impact_text)
        if citation_match:
            citations = int(citation_match.group(1))
            if citations >= 10000: impact_score += 10
            elif citations >= 5000: impact_score += 8
            elif citations >= 1000: impact_score += 6
            elif citations >= 500: impact_score += 4
            elif citations >= 100: impact_score += 2
    
    # 论文数量作为影响力指标
    pub_count = len(profile.selected_publications)
    if pub_count >= 20: impact_score += 8
    elif pub_count >= 10: impact_score += 6
    elif pub_count >= 5: impact_score += 4
    elif pub_count >= 3: impact_score += 2
    
    score += min(20, impact_score)
    
    # 5. 职业阶段调整 (0-10分)
    stage_score = 0
    if profile.career_stage:
        stage_lower = profile.career_stage.lower()
        if 'full_prof' in stage_lower or 'professor' in stage_lower:
            stage_score += 10
        elif 'associate_prof' in stage_lower or 'associate professor' in stage_lower:
            stage_score += 8
        elif 'assistant_prof' in stage_lower or 'assistant professor' in stage_lower:
            stage_score += 6
        elif 'postdoc' in stage_lower:
            stage_score += 4
        elif 'phd' in stage_lower or 'student' in stage_lower:
            stage_score += 2
        elif 'industry' in stage_lower:
            stage_score += 7
    score += stage_score
    
    return min(100.0, score)

# ============================ ADDITIONAL PUBLICATIONS FUNCTIONS ============================

def fetch_author_publications_via_s2(author_id: str, k: int = 10) -> List[Dict[str, Any]]:
    """
    Fetch additional publications via Semantic Scholar API
    """
    publications = []

    try:
        from .semantic_paper_search import SemanticScholarClient
        from . import config
        s2_client = SemanticScholarClient(api_key=config.SEMANTIC_SCHOLAR_API_KEY if config.SEMANTIC_SCHOLAR_API_KEY else None)
        
        papers = s2_client.get_author_papers(author_id, limit=k, sort="citationCount")
        
        for paper in papers:
            pub_info = {
                'title': paper.get('title', ''),
                'year': paper.get('year'),
                'venue': paper.get('venue', ''),
                'url': paper.get('url', ''),
                'citations': paper.get('citationCount', 0),
                'authors': paper.get('authors', [])
            }
            publications.append(pub_info)
            
        print(f"[S2 Publications] Fetched {len(publications)} papers for author {author_id}")
        
    except Exception as e:
        print(f"[S2 publications] Error: {e}")

    return publications

# ============================ COMPREHENSIVE HOMEPAGE FETCHER ============================

def fetch_homepage_comprehensive(url: str, author_name: str = "", max_chars: int = 50000,
                                include_subpages: bool = True, max_subpages: int = 6) -> Dict[str, Any]:
    """
    专门处理homepage链接的全面抓取函数
    从整个HTML内容中提取各种社交媒体链接和其他作者信息
    支持自动发现和抓取subpage内容

    Args:
        url: homepage URL
        author_name: 作者姓名（用于验证和匹配）
        max_chars: 最大字符限制
        include_subpages: 是否包含subpage抓取（默认为True）
        max_subpages: 最大抓取的subpage数量

    Returns:
        Dict containing:
        - 'full_html': 完整的HTML内容（包含subpages）
        - 'extracted_links': 提取的各种链接
        - 'emails': 邮箱列表
        - 'social_platforms': 社交媒体平台链接
        - 'text_content': 文本内容
        - 'title': 页面标题
        - 'subpages': subpage信息（如果启用）
        - 'total_subpages': subpage总数
        - 'successful_subpages': 成功抓取的subpage数
    """
    print(f"[Homepage Fetcher] Starting comprehensive fetch for: {url}")
    print(f"[Homepage Fetcher] Subpages enabled: {include_subpages}")

    # 如果启用subpages，使用增强版函数
    if include_subpages:
        return fetch_homepage_comprehensive_with_subpages(
            url=url,
            author_name=author_name,
            max_chars=max_chars,
            max_subpages=max_subpages,
            subpage_timeout=8
        )

    # 否则使用原有逻辑（保持向后兼容）
    result = {
        'full_html': '',
        'extracted_links': {},
        'emails': [],
        'social_platforms': {},
        'text_content': '',
        'title': '',
        'success': False,
        'subpages': [],
        'total_subpages': 0,
        'successful_subpages': 0
    }

    try:
        # 使用requests获取完整HTML内容
        r = requests.get(url, timeout=30, headers=config.UA)

        if not r.ok:
            print(f"[Homepage Fetcher] HTTP error {r.status_code} for {url}")
            return result

        html_content = _decode_response_text(r)
        result['full_html'] = html_content[:max_chars]  # 限制大小但保留完整性
        result['success'] = True

        # 使用BeautifulSoup解析HTML
        soup = BeautifulSoup(html_content, 'html.parser')

        # 1. 提取页面标题
        title = extract_title_unified(html_content)
        result['title'] = title
        print(f"[Homepage Fetcher] Extracted title: {title}")

        # 2. 提取所有链接
        all_links = extract_all_links_from_html(html_content, url)
        result['extracted_links'] = all_links

        # 3. 专门提取社交媒体平台链接
        social_platforms = extract_social_platforms_from_html(html_content, url)
        result['social_platforms'] = social_platforms
        print(f"[Homepage Fetcher] Found {len(social_platforms)} social platforms")

        # 4. 提取邮箱地址（带作者名过滤）
        emails = extract_emails_from_html(html_content, author_name)
        result['emails'] = emails
        print(f"[Homepage Fetcher] Found {len(emails)} email addresses")

        # 5. 提取主要文本内容（用于LLM处理）
        text_content = extract_main_text(html_content, url)
        result['text_content'] = text_content[:30000]  # 限制文本内容大小

        # 6. 打印提取结果摘要
        print(f"[Homepage Fetcher] Summary:")
        print(f"  - Title: {title}")
        print(f"  - Social platforms: {list(social_platforms.keys())}")
        print(f"  - Emails: {emails}")
        print(f"  - Total links found: {len(all_links)}")

        return result

    except Exception as e:
        print(f"[Homepage Fetcher] Error fetching {url}: {e}")
        return result


def extract_all_links_from_html(html_content: str, base_url: str = "") -> Dict[str, List[str]]:
    """
    从HTML内容中提取所有类型的链接

    Args:
        html_content: HTML内容
        base_url: 基础URL（用于相对链接转换）

    Returns:
        分类后的链接字典
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    links = {
        'all': [],
        'mailto': [],
        'http': [],
        'https': [],
        'relative': []
    }

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()

        # 跳过空链接和JavaScript
        if not href or href.startswith('javascript:') or href == '#':
            continue

        links['all'].append(href)

        if href.startswith('mailto:'):
            links['mailto'].append(href)
        elif href.startswith('http://'):
            links['http'].append(href)
        elif href.startswith('https://'):
            links['https'].append(href)
        elif not href.startswith(('http://', 'https://', 'mailto:')):
            # 相对链接
            links['relative'].append(href)

    return links


def extract_social_platforms_from_html(html_content: str, base_url: str = "") -> Dict[str, str]:
    """
    从HTML内容中专门提取社交媒体和学术平台链接

    Args:
        html_content: HTML内容
        base_url: 基础URL

    Returns:
        平台名称到URL的映射字典
    """
    soup = BeautifulSoup(html_content, 'html.parser')
    platforms = {}

    # 定义平台识别规则
    platform_patterns = {
        'scholar': [
            r'scholar\.google\.com/citations\?user=',
            r'scholar\.google\.com/citations/',
        ],
        'github': [
            r'github\.com/[A-Za-z0-9\-_]+',
        ],
        'twitter': [
            r'(?:x\.com|twitter\.com)/[A-Za-z0-9_]+',
        ],
        'linkedin': [
            r'linkedin\.com/in/[A-Za-z0-9\-_]+',
        ],
        'orcid': [
            r'orcid\.org/\d{4}-\d{4}-\d{4}-\d{4}',
        ],
        'openreview': [
            r'openreview\.net/profile\?id=',
        ],
        'semanticscholar': [
            r'semanticscholar\.org/author/',
        ],
        'dblp': [
            r'dblp\.org/pid/',
            r'dblp\.org/pers/',
        ],
        'researchgate': [
            r'researchgate\.net/profile/',
        ],
        'huggingface': [
            r'huggingface\.co/[A-Za-z0-9\-_]+',
        ]
    }

    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()

        for platform, patterns in platform_patterns.items():
            for pattern in patterns:
                if re.search(pattern, href, re.IGNORECASE):
                    if platform not in platforms:  # 保留第一个匹配的链接
                        # 确保URL是完整的
                        if not href.startswith(('http://', 'https://')):
                            if base_url:
                                if href.startswith('/'):
                                    from urllib.parse import urljoin
                                    href = urljoin(base_url, href)
                                else:
                                    href = f"{base_url.rstrip('/')}/{href}"
                        platforms[platform] = href
                        print(f"[Platform Found] {platform}: {href}")
                        break

    return platforms


def extract_emails_from_html(html_content: str, author_name: str = "") -> List[str]:
    """
    从HTML内容中提取邮箱地址，并过滤掉明显不属于目标作者的邮箱
    支持各种反爬虫邮箱格式，如 "name - at - domain.com"

    Args:
        html_content: HTML内容
        author_name: 目标作者姓名，用于过滤

    Returns:
        过滤后的邮箱地址列表
    """
    import urllib.parse
    soup = BeautifulSoup(html_content, 'html.parser')
    emails = set()  # 使用set去重

    # 1. 从mailto链接中提取
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        if href.startswith('mailto:'):
            email = href[7:].split('?')[0]  # 移除可能的查询参数
            # 处理URL编码
            email = urllib.parse.unquote(email)
            if '@' in email and is_email_relevant_to_author(email, author_name):
                emails.add(email.lower())

    # 2. 从文本内容中提取（使用正则表达式）
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    text_content = soup.get_text()
    found_emails = re.findall(email_pattern, text_content, re.IGNORECASE)

    for email in found_emails:
        if is_email_relevant_to_author(email, author_name):
            emails.add(email.lower())

    # 3. 从特定属性中提取（有些网站把邮箱放在data属性中）
    for tag in soup.find_all(attrs={'data-email': True}):
        email = tag.get('data-email', '').strip()
        if '@' in email and is_email_relevant_to_author(email, author_name):
            emails.add(email.lower())

    # 4. 处理反爬虫邮箱格式（如 "name - at - domain.com"）
    # 查找可能的反爬虫邮箱模式
    obfuscated_patterns = [
        r'\b([A-Za-z0-9._%+-]+)\s*-\s*at\s*-\s*([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',  # name - at - domain.com
        r'\b([A-Za-z0-9._%+-]+)\s*\[at\]\s*([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',    # name [at] domain.com
        r'\b([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',         # name @ domain.com (文本中的@)
        r'\b([A-Za-z0-9._%+-]+)\s*\(at\)\s*([A-Za-z0-9.-]+\.[A-Z|a-z]{2,})\b',    # name (at) domain.com
    ]

    # 从所有文本内容中查找
    for pattern in obfuscated_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for username, domain in matches:
            email = f"{username}@{domain}".lower()
            if is_email_relevant_to_author(email, author_name):
                emails.add(email)

    # 5. 处理HTML中的反爬虫格式（从href和文本内容）
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        text = a_tag.get_text().strip()

        # 处理href中的URL编码反爬虫格式
        if 'mailto:' in href:
            decoded_href = urllib.parse.unquote(href)
            for pattern in obfuscated_patterns:
                matches = re.findall(pattern, decoded_href, re.IGNORECASE)
                for username, domain in matches:
                    email = f"{username}@{domain}".lower()
                    if is_email_relevant_to_author(email, author_name):
                        emails.add(email)

        # 处理文本中的反爬虫格式
        for pattern in obfuscated_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for username, domain in matches:
                email = f"{username}@{domain}".lower()
                if is_email_relevant_to_author(email, author_name):
                    emails.add(email)

    # 6. 处理其他可能的邮箱链接格式
    # 处理onclick事件中的邮箱
    for tag in soup.find_all(attrs={'onclick': True}):
        onclick = tag.get('onclick', '').strip()
        if 'mailto:' in onclick or 'email' in onclick.lower():
            # 提取onclick中的mailto链接
            mailto_match = re.search(r'mailto:([^\s\'"]+)', onclick, re.IGNORECASE)
            if mailto_match:
                email = urllib.parse.unquote(mailto_match.group(1))
                if '@' in email and is_email_relevant_to_author(email, author_name):
                    emails.add(email.lower())

    # 7. 处理JavaScript混淆的邮箱
    # 查找可能的JavaScript邮箱构造
    script_tags = soup.find_all('script')
    for script in script_tags:
        if script.string:
            script_content = script.string
            # 查找可能的邮箱构造模式
            js_email_patterns = [
                r'mailto:\s*[\'"]([^\'"]+)[\'"]',  # mailto:"email"
                r'([a-zA-Z0-9._%+-]+)\s*\+\s*[\'"](@[^\'"]+)[\'"]',  # name + "@domain"
                r'[\'"]([a-zA-Z0-9._%+-]+@[^\'"]+)[\'"]',  # "email@domain"
                r'var\s+email\s*=\s*[\'"]([^\'"]+)[\'"]',  # var email = "email"
                r'([a-zA-Z0-9._%+-]+)\s*\+\s*["\'](@[^"\']+)["\']',  # name + "@domain"
            ]
            for pattern in js_email_patterns:
                matches = re.findall(pattern, script_content, re.IGNORECASE)
                for match in matches:
                    if isinstance(match, tuple) and len(match) == 2:
                        # 处理 name + "@domain" 格式
                        username, domain_part = match
                        email = username + domain_part
                    else:
                        # 处理其他格式
                        email = match if isinstance(match, str) else ''.join(match)
                    email = urllib.parse.unquote(email)
                    if '@' in email and is_email_relevant_to_author(email, author_name):
                        emails.add(email.lower())

    # 8. 处理其他属性中的邮箱（如data-href, data-mail等）
    other_attrs = ['data-href', 'data-mail', 'data-email', 'data-contact']
    for attr in other_attrs:
        for tag in soup.find_all(attrs={attr: True}):
            attr_value = tag.get(attr, '').strip()
            if attr_value.startswith('mailto:'):
                email = urllib.parse.unquote(attr_value[7:].split('?')[0])
                if '@' in email and is_email_relevant_to_author(email, author_name):
                    emails.add(email.lower())
            elif '@' in attr_value:
                email = urllib.parse.unquote(attr_value)
                if is_email_relevant_to_author(email, author_name):
                    emails.add(email.lower())

    # 9. 处理实体编码的邮箱
    # BeautifulSoup会自动解码实体，所以直接从解码后的文本中提取
    decoded_text = soup.get_text()

    # 查找解码后的邮箱（已经被BeautifulSoup转换为<email>格式）
    decoded_email_pattern = r'<([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})>'
    matches = re.findall(decoded_email_pattern, decoded_text, re.IGNORECASE)
    for email in matches:
        if is_email_relevant_to_author(email, author_name):
            emails.add(email.lower())

    # 同时检查原始HTML中的实体编码（以防万一）
    raw_html = str(soup)
    entity_patterns = [
        r'&lt;([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})&gt;',  # &lt;email&gt;
        r'&#60;([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})&#62;',  # &#60;email&#62;
    ]
    for pattern in entity_patterns:
        matches = re.findall(pattern, raw_html, re.IGNORECASE)
        for email in matches:
            if is_email_relevant_to_author(email, author_name):
                emails.add(email.lower())

    # 10. 处理分段的反爬虫格式（如name[at]domain[dot]com）
    extended_obfuscated_patterns = [
        r'\b([A-Za-z0-9._%+-]+)\s*\[at\]\s*([A-Za-z0-9.-]+)\s*\[dot\]\s*([a-zA-Z]{2,})\b',  # name [at] domain [dot] com
        r'\b([A-Za-z0-9._%+-]+)\s*\(at\)\s*([A-Za-z0-9.-]+)\s*\(dot\)\s*([a-zA-Z]{2,})\b',  # name (at) domain (dot) com
        r'\b([A-Za-z0-9._%+-]+)\s*@\s*([A-Za-z0-9.-]+)\s*\.\s*([a-zA-Z]{2,})\b',         # name @ domain . com
    ]

    for pattern in extended_obfuscated_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for match in matches:
            if len(match) == 3:
                username, domain, tld = match
                email = f"{username}@{domain}.{tld}".lower()
                if is_email_relevant_to_author(email, author_name):
                    emails.add(email)

    # 11. 处理图片alt文本中的邮箱（有些网站用图片显示邮箱）
    for img in soup.find_all('img', alt=True):
        alt_text = img.get('alt', '').strip()
        # 检查alt文本是否包含邮箱信息
        for pattern in obfuscated_patterns + extended_obfuscated_patterns:
            matches = re.findall(pattern, alt_text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple) and len(match) >= 2:
                    if len(match) == 2:
                        username, domain = match
                        email = f"{username}@{domain}".lower()
                    elif len(match) == 3:
                        username, domain, tld = match
                        email = f"{username}@{domain}.{tld}".lower()
                    if is_email_relevant_to_author(email, author_name):
                        emails.add(email)

    return list(emails)

def _is_valid_email_format(email: str) -> bool:
    """
    验证email是否符合基本格式要求
    
    Args:
        email: 待验证的email地址
    
    Returns:
        bool: True if valid format, False otherwise
    """
    if not email or not isinstance(email, str):
        return False
    
    email = email.strip()
    
    # 基本格式验证：必须包含@和.
    if '@' not in email or '.' not in email:
        return False
    
    # 检查@的位置（不能在开头或结尾）
    at_pos = email.find('@')
    if at_pos <= 0 or at_pos >= len(email) - 2:
        return False
    
    # 分割用户名和域名
    parts = email.split('@')
    if len(parts) != 2:
        return False
    
    username, domain = parts
    
    # 用户名验证（不能为空，不能包含空格）
    if not username or ' ' in username or len(username) < 1:
        return False
    
    # 域名验证（必须有.，不能包含空格）
    if '.' not in domain or ' ' in domain or len(domain) < 3:
        return False
    
    # 域名后缀验证（至少2个字符）
    domain_parts = domain.split('.')
    if len(domain_parts[-1]) < 2:
        return False
    
    # 排除明显的非email文本
    invalid_patterns = [
        'http://', 'https://', 'www.', 
        '..', '@@', '< ', ' >',
        '[at]', '(at)', '[dot]', '(dot)',
        'email protected', 'your email', 'example@'
    ]
    email_lower = email.lower()
    if any(pattern in email_lower for pattern in invalid_patterns):
        return False
    
    # 正则表达式最终验证
    import re
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, email):
        return False
    
    return True


def is_email_relevant_to_author(email: str, author_name: str) -> bool:
    """
    判断邮箱是否可能属于目标作者 - 增强版
    
    Args:
        email: 邮箱地址
        author_name: 目标作者姓名
        
    Returns:
        是否相关
    """
    if not author_name or not email:
        return False
    
    email_lower = email.lower().strip()
    
    # 排除明显的系统/通用邮箱，但要考虑上下文
    system_emails = [
        'admin@', 'support@', 'webmaster@',
        'noreply@', 'no-reply@', 'help@', 'service@', 'office@',
        'secretary@', 'dept@', 'department@', 'marketing@', 'sales@'
    ]

    # 对于'info@'和'contact@'，如果作者名匹配，则不视为系统邮箱
    author_words = [word.lower() for word in author_name.split() if len(word) > 2]
    email_username = email_lower.split('@')[0] if '@' in email_lower else ''

    # 检查是否是明显的系统邮箱
    is_system_email = False
    for prefix in system_emails:
        if email_lower.startswith(prefix):
            is_system_email = True
            break

    # 特殊处理'info@'和'contact@' - 如果用户名与作者名相关，则不视为系统邮箱
    if email_lower.startswith(('info@', 'contact@')):
        if any(word in email_username for word in author_words) or email_username in author_name.lower():
            is_system_email = False

    if is_system_email:
        return False
    
    # 排除明显的垃圾邮箱或占位符
    spam_patterns = ['****', 'xxx@', 'example@', 'dummy@', 'fake@']
    # 只过滤完全匹配的测试邮箱，不过滤包含'test'的一般邮箱
    if any(spam in email_lower for spam in spam_patterns) or email_lower == 'test@test.com':
        return False
    
    # 排除明显不是个人邮箱的地址
    if any(company in email_lower for company in [
        '@google.com', '@microsoft.com', '@amazon.com', '@meta.com', 
        '@apple.com', '@nvidia.com', '@openai.com', '@anthropic.com'
    ]):
        # 这些大公司邮箱通常不是学者的主要联系邮箱
        return False
    
    # 提取邮箱用户名和域名
    if '@' not in email_lower:
        return False
    
    email_username, email_domain = email_lower.split('@', 1)
    author_words = [word.lower() for word in author_name.split() if len(word) > 2]
    
    # 强匹配：邮箱用户名包含作者姓名的关键词
    name_match_score = 0
    for word in author_words:
        if word in email_username:
            name_match_score += 1
    
    # 如果有强名字匹配，直接接受
    if name_match_score >= len(author_words) * 0.5:
        return True
    
    # 检查是否是学术机构邮箱
    academic_domains = ['.edu', '.ac.uk', '.ac.nz', '.ac.jp', '.edu.cn', '.ac.cn', '.ac.']
    is_academic = any(domain in email_domain for domain in academic_domains)
    
    # 学术邮箱 + 有一定名字匹配度
    if is_academic and name_match_score > 0:
        return True
    
    # 如果没有名字匹配且不是学术邮箱，拒绝
    if name_match_score == 0 and not is_academic:
        return False
    
    # 其他情况保守接受
    return True

def discover_subpages(base_url: str, html_content: str, author_name: str = "", max_subpages: int = 10) -> List[Dict[str, str]]:
    """
    从主页面HTML内容中发现有价值的subpage链接

    Args:
        base_url: 基础URL
        html_content: 主页面的HTML内容
        author_name: 作者姓名（用于相关性判断）
        max_subpages: 最大抓取的subpage数量

    Returns:
        subpage信息列表，每个包含 'url', 'title', 'type' 等信息
    """
    from urllib.parse import urlparse, urljoin

    print(f"[Subpage Discovery] Discovering subpages for: {base_url}")

    subpages = []
    anchors = []  # ✅ 新增：存储锚点链接
    soup = BeautifulSoup(html_content, 'html.parser')

    # 解析基础URL
    parsed_base = urlparse(base_url)
    base_domain = parsed_base.netloc
    base_path = parsed_base.path.rstrip('/')

    # ✅ 扩展有价值的subpage模式（新增：software, datasets, tools, awards, gallery, portfolio）
    valuable_subpages = [
        # 研究相关
        'research', 'publications', 'papers', 'projects', 'work', 'datasets', 'data',
        # 个人信息
        'about', 'bio', 'biography', 'cv', 'resume', 'vitae', 'profile', 'portfolio',
        # 教学相关
        'teaching', 'courses', 'students', 'supervision', 'mentoring',
        # 联系方式
        'contact', 'contact-me',
        # 其他有用页面
        'news', 'updates', 'blog', 'resources', 'software', 'code', 'tools',
        'group', 'team', 'lab', 'collaborators', 'alumni', 'awards', 'honors',
        'services', 'activities', 'talks', 'presentations', 'gallery'
    ]
    
    # ✅ 正则模式：匹配 publications-2024, papers-neurips 等变体
    import re
    valuable_patterns = [
        r'publication[s]?[-_]?\d*',
        r'paper[s]?[-_]?\d*',
        r'project[s]?[-_]?\w*',
        r'研究|论文|项目|成果'  # 中文关键词
    ]

    # 从链接中发现subpage
    for a_tag in soup.find_all('a', href=True):
        href = a_tag['href'].strip()
        link_text = a_tag.get_text().strip()

        # ✅ 处理锚点链接（如 #publications, /#publications）
        if href.startswith('#') or href.startswith('/#'):
            anchor_name = href.lstrip('#/').lower()
            # 检查是否是有价值的锚点
            if any(keyword in anchor_name for keyword in valuable_subpages):
                anchors.append({
                    'anchor': anchor_name,
                    'text': link_text,
                    'type': next((kw for kw in valuable_subpages if kw in anchor_name), 'other')
                })
                print(f"[Subpage Discovery] Found anchor: #{anchor_name}")
            continue

        # 跳过无效链接
        if not href or href.startswith(('javascript:', 'mailto:')):
            continue

        # 转换为绝对URL
        if not href.startswith(('http://', 'https://')):
            if href.startswith('/'):
                href = f"{parsed_base.scheme}://{base_domain}{href}"
            else:
                href = urljoin(base_url, href)

        # 检查是否是同一域名的内部链接
        parsed_href = urlparse(href)
        if parsed_href.netloc != base_domain:
            continue

        # 检查是否是subpage（不是根路径）
        href_path = parsed_href.path.rstrip('/')
        if not href_path or href_path == base_path:
            continue

        # 判断是否是有价值的subpage
        is_valuable = False
        subpage_type = 'other'

        # ✅ 1. 检查路径中的关键词（精确匹配）
        path_lower = href_path.lower()
        for keyword in valuable_subpages:
            if f'/{keyword}' in path_lower or f'/{keyword}/' in path_lower or path_lower.endswith(f'/{keyword}'):
                is_valuable = True
                subpage_type = keyword
                break

        # ✅ 2. 检查路径中的正则模式（模糊匹配）
        if not is_valuable:
            for pattern in valuable_patterns:
                if re.search(pattern, path_lower):
                    is_valuable = True
                    # 提取匹配的关键词作为类型
                    match = re.search(pattern, path_lower)
                    if match:
                        subpage_type = match.group(0)
                    break

        # ✅ 3. 检查链接文本中的关键词
        text_lower = link_text.lower()
        for keyword in valuable_subpages:
            if keyword in text_lower:
                is_valuable = True
                if subpage_type == 'other':
                    subpage_type = keyword
                break
        
        # ✅ 4. 检查链接文本中的特殊短语（"paper list", "我的研究"等）
        if not is_valuable:
            special_phrases = [
                'paper list', 'publication list', 'project list',
                'my research', 'my work', 'my papers',
                '我的研究', '我的论文', '研究成果'
            ]
            for phrase in special_phrases:
                if phrase in text_lower:
                    is_valuable = True
                    subpage_type = 'research'
                    break

        # 检查是否可能是个人页面（不包含常见排除词）
        if not is_valuable:
            exclude_patterns = ['login', 'admin', 'wp-', 'category', 'tag', 'author', 'search', 'feed', 'rss']
            if not any(pattern in path_lower for pattern in exclude_patterns):
                # 检查路径深度（2-3级路径可能是个人页面）
                path_parts = [p for p in href_path.split('/') if p]
                if 1 <= len(path_parts) <= 3:
                    # 检查是否包含作者名字的缩写或相关词
                    if author_name:
                        author_parts = [part.lower() for part in author_name.split() if len(part) > 2]
                        for part in author_parts:
                            if part in path_lower:
                                is_valuable = True
                                subpage_type = 'personal'
                                break

        if is_valuable:
            # 避免重复
            if not any(sp['url'] == href for sp in subpages):
                subpages.append({
                    'url': href,
                    'title': link_text[:100] if link_text else f"Subpage: {subpage_type}",
                    'type': subpage_type,
                    'path': href_path
                })

    # ✅ 处理锚点：提取锚点对应的HTML内容
    anchor_contents = []
    for anchor in anchors:
        anchor_name = anchor['anchor']
        # 查找对应的HTML元素（id或name属性）
        anchor_elem = soup.find(id=anchor_name) or soup.find(attrs={'name': anchor_name})
        if anchor_elem:
            # 提取该section的文本内容
            anchor_text = anchor_elem.get_text(strip=True, separator='\n')
            if len(anchor_text) > 100:  # 只保留有实质内容的锚点
                anchor_contents.append({
                    'anchor': anchor_name,
                    'type': anchor['type'],
                    'text': link_text or anchor_name,
                    'content': anchor_text[:10000]  # 限制10k字符
                })
                print(f"[Subpage Discovery] Extracted anchor content: #{anchor_name} ({len(anchor_text)} chars)")
    
    # 限制subpage数量
    subpages = subpages[:max_subpages]
    
    print(f"[Subpage Discovery] Found {len(subpages)} valuable subpages + {len(anchor_contents)} anchors")
    if subpages:
        subpage_info = [f"{sp['type']}" for sp in subpages]
        print(f"[Subpage Discovery]   Subpages: {subpage_info}")
    if anchor_contents:
        anchor_info = [f"#{a['anchor']}" for a in anchor_contents]
        print(f"[Subpage Discovery]   Anchors: {anchor_info}")

    return subpages, anchor_contents
def fetch_subpage_content(subpage_info: Dict[str, str], timeout: int = 10, use_dynamic_render: bool = False) -> Dict[str, Any]:
    """
    Args:
        subpage_info: subpage信息字典
        timeout: 请求超时时间
        use_dynamic_render: 是否使用动态渲染（Playwright）

    Returns:
        subpage抓取结果
    """
    url = subpage_info['url']
    print(f"[Subpage Fetch] Fetching: {url}")

    result = {
        'url': url,
        'title': subpage_info['title'],
        'type': subpage_info['type'],
        'success': False,
        'html_content': '',
        'text_content': '',
        'links': {},
        'emails': [],
        'error': '',
        'quality_score': 0,
        'is_dynamic': False,
        'child_pages': []
    }

    try:
        # ✅ 尝试动态渲染（如果启用）
        html_content = None
        if use_dynamic_render:
            try:
                from playwright.sync_api import sync_playwright
                import asyncio
                import sys

                if sys.platform.startswith("win"):
                    try:
                        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
                    except AttributeError:
                        pass

                try:
                    asyncio.get_running_loop()
                except RuntimeError:
                    asyncio.set_event_loop(asyncio.new_event_loop())

                print(f"[Subpage Fetch] Using Playwright for dynamic rendering...")
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    page.goto(url, timeout=timeout * 1000, wait_until='networkidle')
                    html_content = page.content()
                    browser.close()
                result['is_dynamic'] = True
                print(f"[Subpage Fetch] ✅ Dynamic rendering successful")
            except Exception as e:
                print(f"[Subpage Fetch] ⚠️ Dynamic rendering failed: {e}, falling back to static fetch")
                use_dynamic_render = False
        
        # ✅ 静态抓取（默认或动态渲染失败时）
        if not html_content:
            r = requests.get(url, timeout=timeout, headers=config.UA)
            if not r.ok:
                result['error'] = f"HTTP {r.status_code}"
                print(f"[Subpage Fetch] Failed: {result['error']}")
                return result
            html_content = _decode_response_text(r)

        result['html_content'] = html_content[:50000]  # Limit to 50k chars
        result['success'] = True

        # Extract links, emails, and text
        result['links'] = extract_all_links_from_html(html_content, url)
        result['emails'] = extract_emails_from_html(html_content, "")
        result['text_content'] = extract_main_text(html_content, url)[:10000]

        # ✅ 内容质量检测
        quality_score = _assess_content_quality(html_content, result['text_content'])
        result['quality_score'] = quality_score
        
        # ✅ 发现潜在的子页面（用于递归抓取）
        from urllib.parse import urlparse, urljoin
        soup = BeautifulSoup(html_content, 'html.parser')
        parsed_url = urlparse(url)
        base_domain = parsed_url.netloc
        current_depth = len([p for p in parsed_url.path.split('/') if p])
        
        # 只在深度 ≤ 2 时查找子页面
        if current_depth <= 2:
            for a_tag in soup.find_all('a', href=True):
                href = a_tag['href'].strip()
                if not href or href.startswith(('#', 'javascript:', 'mailto:')):
                    continue
                
                # 转换为绝对URL
                if not href.startswith(('http://', 'https://')):
                    href = urljoin(url, href)
                
                parsed_href = urlparse(href)
                if parsed_href.netloc != base_domain:
                    continue
                
                # 检查路径深度（不超过3层）
                href_depth = len([p for p in parsed_href.path.split('/') if p])
                if href_depth > 3:
                    continue
                
                # 检查是否包含年份或分页关键词
                href_lower = href.lower()
                if any(pattern in href_lower for pattern in ['/202', '/201', '/page', '/p=']):
                    result['child_pages'].append(href)

        print(f"[Subpage Fetch] Success: {len(result['text_content'])} chars, quality={quality_score:.1f}, child_pages={len(result['child_pages'])}")

    except Exception as e:
        result['error'] = str(e)
        print(f"[Subpage Fetch] Error: {e}")

    return result


def _detect_spa_framework(html_content: str) -> bool:
    """
    ✅ 检测页面是否使用SPA框架（需要动态渲染）
    
    Returns:
        True if SPA detected, False otherwise
    """
    spa_markers = [
        '<script src=',  # Generic JS framework
        'window.__NUXT__', 'window.__NEXT_DATA__',  # Next.js, Nuxt.js
        'id="app"', 'id="root"', 'id="__next"',  # Common SPA root elements
        'ng-app=', 'ng-controller=',  # Angular
        'v-app', 'v-cloak',  # Vue.js
        'react-root', 'data-reactroot',  # React
        'ember-application',  # Ember.js
    ]
    
    html_lower = html_content.lower()
    for marker in spa_markers:
        if marker.lower() in html_lower:
            return True
    
    return False


def _assess_content_quality(html_content: str, text_content: str) -> float:
    """
    ✅ 评估页面内容质量
    
    Returns:
        质量分数 0-10
    """
    score = 0.0
    
    # 1. 文本长度（最多3分）
    text_len = len(text_content)
    if text_len > 5000:
        score += 3.0
    elif text_len > 2000:
        score += 2.0
    elif text_len > 500:
        score += 1.0
    else:
        score += 0.5
    
    # 2. 学术链接密度（最多3分）
    soup = BeautifulSoup(html_content, 'html.parser')
    academic_links = soup.find_all('a', href=True)
    academic_count = 0
    for link in academic_links:
        href = link.get('href', '').lower()
        if any(pattern in href for pattern in ['doi.org', 'arxiv.org', 'pdf', 'paper', 'scholar.google']):
            academic_count += 1
    
    if academic_count > 50:
        score += 3.0
    elif academic_count > 20:
        score += 2.0
    elif academic_count > 5:
        score += 1.0
    
    # 3. 列表结构（最多2分）
    list_items = len(soup.find_all(['li', 'tr']))
    if list_items > 50:
        score += 2.0
    elif list_items > 20:
        score += 1.0
    elif list_items > 5:
        score += 0.5
    
    # 4. 标题结构（最多2分）
    headings = len(soup.find_all(['h1', 'h2', 'h3']))
    if headings > 10:
        score += 2.0
    elif headings > 5:
        score += 1.0
    elif headings > 2:
        score += 0.5
    
    return min(score, 10.0)


def merge_subpage_content(main_result: Dict[str, Any], subpage_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    将所有subpage内容合并到主结果中

    Args:
        main_result: 主页面抓取结果
        subpage_results: 所有subpage抓取结果

    Returns:
        合并后的完整结果
    """
    print(f"[Content Merge] Merging {len(subpage_results)} subpages into main content")

    # 合并HTML内容
    all_html_parts = [main_result['full_html']]
    all_text_parts = [main_result['text_content']]

    # 合并subpage信息
    subpage_summaries = []

    for subpage in subpage_results:
        if subpage['success']:
            # 添加到HTML集合中（用于后续LLM处理）
            if subpage['html_content']:
                all_html_parts.append(f"\n\n--- SUBPAGE: {subpage['title']} ({subpage['type']}) ---\n{subpage['html_content']}")

            # 添加到文本集合中
            if subpage['text_content']:
                all_text_parts.append(f"\n\n=== {subpage['title']} ({subpage['type']}) ===\n{subpage['text_content']}")

            # 收集subpage摘要信息
            subpage_summaries.append({
                'title': subpage['title'],
                'type': subpage['type'],
                'url': subpage['url'],
                'content_length': len(subpage['text_content']),
                'emails_found': len(subpage['emails'])
            })

            # 合并链接
            for link_type, links in subpage['links'].items():
                if link_type not in main_result['extracted_links']:
                    main_result['extracted_links'][link_type] = []
                main_result['extracted_links'][link_type].extend(links)

            # 合并邮箱（去重）
            for email in subpage['emails']:
                if email not in main_result['emails']:
                    main_result['emails'].append(email)

            # 合并社交平台链接
            subpage_social = extract_social_platforms_from_html(subpage['html_content'], subpage['url'])
            for platform, url in subpage_social.items():
                if platform not in main_result['social_platforms']:
                    main_result['social_platforms'][platform] = url

    # 更新合并后的内容
    main_result['full_html'] = '\n'.join(all_html_parts)[:100000]  # 限制总大小
    main_result['text_content'] = '\n'.join(all_text_parts)[:50000]  # 限制文本内容大小

    # 重要修复：从合并后的完整HTML中重新提取所有邮箱
    # 这确保了主页面和所有subpage的邮箱都被正确提取
    author_name_for_extraction = main_result.get('author_name', '')
    all_emails_in_merged_content = extract_emails_from_html(main_result['full_html'], author_name_for_extraction)

    # 合并所有找到的邮箱（包括主页面原始邮箱、subpage邮箱和重新提取的邮箱）
    original_emails = main_result.get('original_emails', [])
    final_emails = list(set(original_emails + main_result['emails'] + all_emails_in_merged_content))
    main_result['emails'] = final_emails

    # 从合并后的完整HTML中重新提取社交平台链接
    all_social_platforms = extract_social_platforms_from_html(main_result['full_html'], main_result.get('url', ''))
    # 合并社交平台（保留主页面原有的优先级）
    for platform, url in all_social_platforms.items():
        if platform not in main_result['social_platforms']:
            main_result['social_platforms'][platform] = url

    # 添加subpage信息
    main_result['subpages'] = subpage_summaries
    main_result['total_subpages'] = len(subpage_summaries)
    main_result['successful_subpages'] = len([s for s in subpage_summaries if s['content_length'] > 0])

    print(f"[Content Merge] Merged content: {len(main_result['full_html'])} chars HTML, {len(main_result['text_content'])} chars text")
    print(f"[Content Merge] Found additional {len([e for e in main_result['emails'] if e not in main_result.get('original_emails', [])])} emails")
    print(f"[Content Merge] Found additional {len([p for p in main_result['social_platforms'] if p not in main_result.get('original_platforms', [])])} social platforms")

    return main_result


# ============================ IMPROVED HOMEPAGE FETCHER WITH SUBPAGES ============================

def fetch_homepage_comprehensive_with_subpages(url: str, author_name: str = "", max_chars: int = 100000,
                                               max_subpages: int = 8, subpage_timeout: int = 8) -> Dict[str, Any]:
    """
    改进版的homepage抓取函数，能够自动发现和抓取subpage内容

    Args:
        url: homepage URL
        author_name: 作者姓名
        max_chars: 最大字符限制
        max_subpages: 最大抓取的subpage数量
        subpage_timeout: 单个subpage的超时时间

    Returns:
        包含主页面和所有subpage内容的完整结果字典
    """
    print(f"[Homepage Fetcher Enhanced] Starting comprehensive fetch with subpages for: {url}")

    result = {
        'full_html': '',
        'extracted_links': {},
        'emails': [],
        'social_platforms': {},
        'text_content': '',
        'title': '',
        'success': False,
        'subpages': [],
        'total_subpages': 0,
        'successful_subpages': 0,
        'error': ''
    }

    try:
        # 1. 首先抓取主页面
        print(f"[Homepage Fetcher Enhanced] Fetching main page...")
        r = requests.get(url, timeout=5, headers=config.UA)

        if not r.ok:
            result['error'] = f"Main page HTTP error {r.status_code}"
            print(f"[Homepage Fetcher Enhanced] {result['error']}")
            return result

        html_content = _decode_response_text(r)
        result['full_html'] = html_content[:max_chars]
        result['success'] = True

        # 2. 解析主页面内容
        soup = BeautifulSoup(html_content, 'html.parser')
        title = extract_title_unified(html_content)
        result['title'] = title
        print(f"[Homepage Fetcher Enhanced] Main page title: {title}")

        # 3. 提取主页面信息
        result['extracted_links'] = extract_all_links_from_html(html_content, url)
        result['social_platforms'] = extract_social_platforms_from_html(html_content, url)
        result['emails'] = extract_emails_from_html(html_content, author_name)
        result['text_content'] = extract_main_text(html_content, url)[:30000]
        result['author_name'] = author_name  # 保存作者名以供后续使用
        result['url'] = url  # 保存URL以供后续使用

        # 记录原始信息（用于后续比较）
        result['original_emails'] = result['emails'].copy()
        result['original_platforms'] = list(result['social_platforms'].keys())

        # 4. 发现subpages 和 anchors
        subpages, anchor_contents = discover_subpages(url, html_content, author_name, max_subpages)

        # ✅ 处理锚点内容：直接并入主文本
        if anchor_contents:
            print(f"[Homepage Fetcher Enhanced] Merging {len(anchor_contents)} anchor contents")
            anchor_texts = []
            for anchor in anchor_contents:
                anchor_texts.append(f"\n\n=== {anchor['text']} ({anchor['type']}) ===\n{anchor['content']}")
            result['text_content'] += "\n".join(anchor_texts)
            print(f"[Homepage Fetcher Enhanced] Added {sum(len(a['content']) for a in anchor_contents)} chars from anchors")

        if not subpages and not anchor_contents:
            print(f"[Homepage Fetcher Enhanced] No valuable subpages or anchors found")
            return result
        
        if not subpages:
            print(f"[Homepage Fetcher Enhanced] No subpages to fetch (only anchors)")
            return result

        # 5. 抓取subpage内容（并发处理 + 递归深度1）
        subpage_results = []
        all_child_pages = []  # ✅ 收集所有子页面用于递归
        
        try:
            from concurrent.futures import ThreadPoolExecutor, as_completed
            # ✅ 激进并发：subpage抓取是IO-bound，可以开更多
            max_workers = get_extraction_workers(len(subpages))
            print(f"[Subpage Fetch] Using {max_workers} workers for {len(subpages)} subpages")
            
            # ✅ 检测是否需要动态渲染（检查HTML中是否有SPA标记）
            use_dynamic = _detect_spa_framework(html_content)
            if use_dynamic:
                print(f"[Subpage Fetch] Detected SPA framework, will use dynamic rendering for subpages")
            
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                fut2info = {ex.submit(fetch_subpage_content, sp, subpage_timeout, use_dynamic): sp for sp in subpages}
                for fut in as_completed(fut2info):
                    try:
                        subpage_res = fut.result()
                        subpage_results.append(subpage_res)
                        
                        # ✅ 收集子页面（用于递归）
                        if subpage_res.get('child_pages'):
                            all_child_pages.extend(subpage_res['child_pages'])
                    except Exception as e:
                        print(f"[Subpage Fetch] error: {e}")
        except Exception as e:
            print(f"[Subpage Parallel] Falling back to sequential due to error: {e}")
            for subpage_info in subpages:
                subpage_result = fetch_subpage_content(subpage_info, subpage_timeout, use_dynamic=False)
                subpage_results.append(subpage_result)
                if subpage_result.get('child_pages'):
                    all_child_pages.extend(subpage_result['child_pages'])
        
        # ✅ 递归抓取子页面（深度=2，限制总数≤20）
        if all_child_pages:
            # 去重并限制数量
            unique_child_pages = list(set(all_child_pages))[:20]
            print(f"\n[Recursive Fetch] Found {len(unique_child_pages)} child pages, fetching...")
            
            try:
                with ThreadPoolExecutor(max_workers=min(max_workers, 10)) as ex:
                    child_info = [{'url': url, 'title': f"Child page", 'type': 'recursive'} for url in unique_child_pages]
                    child_fut = {ex.submit(fetch_subpage_content, ci, subpage_timeout, False): ci for ci in child_info}
                    
                    child_count = 0
                    for fut in as_completed(child_fut):
                        try:
                            child_res = fut.result()
                            # 只添加高质量的子页面
                            if child_res.get('quality_score', 0) >= 3.0:
                                subpage_results.append(child_res)
                                child_count += 1
                                print(f"[Recursive Fetch] ✅ Added child page (quality={child_res['quality_score']:.1f}): {child_res['url']}")
                        except Exception as e:
                            print(f"[Recursive Fetch] error: {e}")
                    
                    print(f"[Recursive Fetch] Completed: {child_count}/{len(unique_child_pages)} child pages added")
            except Exception as e:
                print(f"[Recursive Fetch] Failed: {e}")

        # 6. 合并所有内容
        result = merge_subpage_content(result, subpage_results)

        # 7. 最终统计
        print(f"[Homepage Fetcher Enhanced] Final summary:")
        print(f"  - Main page: {len(result['full_html'].split('--- SUBPAGE:')[0])} chars")
        print(f"  - Total content: {len(result['full_html'])} chars")
        print(f"  - Subpages found: {result['total_subpages']}")
        print(f"  - Subpages successful: {result['successful_subpages']}")
        print(f"  - Social platforms: {len(result['social_platforms'])}")
        print(f"  - Emails: {len(result['emails'])}")

        return result

    except Exception as e:
        result['error'] = str(e)
        print(f"[Homepage Fetcher Enhanced] Error: {e}")
        return result
