from typing import Dict, Any, List, Tuple, Union, Optional, Set
import json
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import copy
import warnings

# Suppress Streamlit ScriptRunContext warnings in ThreadPoolExecutor
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing script run context.*')

# ============================ GLOBAL SAFE PRINT UTILITY ============================

def safe_print(msg):
    """
    Safe print that won't crash on I/O errors.
    Critical for long-running searches where log files might disconnect.
    Prevents cascading failures when stdout/log_file becomes unavailable.
    """
    try:
        print(msg)
    except (OSError, IOError, BrokenPipeError):
        pass  # Silently ignore I/O errors

# Import critical modules
from . import schemas
from .schemas import (
    QuerySpec,
    ResearchState,
    QuerySpecDiff,
    CandidateOverview,
)
from . import utils
from . import config
from . import llm
# Import extraction module
try:
    from . import extraction
except Exception as e:
    print(f"[Agents] ⚠️ Extraction module not available: {e}")
    extraction = None
# Import search module (as module, not variable to avoid overwrite)
search = None
try:
    from . import search as search_module
    search = search_module
except Exception as e:
    print(f"[Agents] ⚠️ Search module not available: {e}")
    import traceback
    traceback.print_exc()

    class _DummySearch:
        @staticmethod
        def score_paper_with_llm(title, abstract, user_query, llm_instance):
            return {"score": 4, "explanation": "Fallback scoring (search unavailable)"}

        @staticmethod
        def fetch_text(url, max_chars, snippet=""):
            return snippet or ""

    search = _DummySearch()

# Import author position utilities
from .utils import compute_author_position, get_author_position_coef

# Import arXiv API module (not used anymore - Agent provides all info)
arxiv_api = None
# Import semantic_paper_search module
SemanticScholarClient = None
try:
    from .semantic_paper_search import SemanticScholarClient
except Exception as e:
    print(f"[Agents] ⚠️ Semantic Scholar client not available: {e}")
# Import author_discovery module
orchestrate_candidate_report = None
try:
    from .author_discovery import orchestrate_candidate_report
except Exception as e:
    print(f"[Agents] orchestrate_candidate_report not available: {e}")
# ============================ SCORING UTILITY FUNCTIONS ============================
def calculate_paper_score(cand):
    """
    计算候选人的论文综合得分（新版本：分桶 + Top-K + 闸门机制）
    公式：
    - paper_base = (topic_score / 8) × venue_coef
    - contrib = paper_base × pos_coef
    - Lead 角色: 1st/last author
    - Support 角色: 2nd+ author
    - 有Lead: score = lead_sum + β × support_sum
    - 无Lead: score = α × support_sum (强折扣)
    作者位次系数：
    - 一作（1st）：1.0
    - 尾作（last）：0.8
    - 二作（2nd）：0.4
    - 三作及之后：0.4/n
    
    Venue系数：
    - Workshop/arXiv：0.5
    - 正式会议（Accepted）：1.0
    """
    candidate_name = cand.name.lower().strip()
    
    selected_research = getattr(cand, "selected_research", None)
    if not selected_research:
        return 0.0
    
    all_publications = getattr(selected_research, "all_publications", [])
    if not all_publications:
        return 0.0
    
    # 分桶收集 Lead 和 Support 论文的贡献值
    lead_contribs = []    # 1st/last author
    support_contribs = [] # 2nd+ author
    
    for pub in all_publications:
        relevance_score = getattr(pub, "relevance_score", 0)
        if relevance_score < 6:
            continue  # 跳过低分论文
        
        # 获取作者位置
        position_index = getattr(pub, "candidate_author_position", None)
        total_authors = getattr(pub, "total_authors", None)
        
        if position_index is None:
            authors = getattr(pub, "authors", [])
            if not authors:
                continue
            position_index, _, total_authors = compute_author_position(candidate_name, authors)
        
        if position_index is None:
            continue
        
        # Check if this author is a corresponding author or co-first author for this paper
        is_corresponding = getattr(pub, "is_corresponding_author", False)
        is_cofirst = getattr(pub, "is_cofirst_author", False)
        # 计算位置系数 (now includes corresponding author and co-first author weight)
        pos_coef = get_author_position_coef(position_index, total_authors, is_corresponding=is_corresponding, is_cofirst=is_cofirst)
        
        # 计算 venue 系数
        venue = getattr(pub, "venue", "").lower()
        venue_coef = 0.5 if any(kw in venue for kw in ["workshop", "arxiv", "preprint"]) else 1.0
        
        # 计算论文基础分（归一化到 0-1）
        paper_base = (relevance_score / 8.0) * venue_coef
        # 计算贡献值
        contrib = paper_base * pos_coef
        # 分桶: Lead = {1st, last}, Support = {其他}
        is_first = (position_index == 1)
        is_last = (total_authors and position_index == total_authors and total_authors > 1)
        if is_first or is_last:
            lead_contribs.append(contrib)
        else:
            support_contribs.append(contrib)
    # Top-K 截断
    lead_contribs.sort(reverse=True)
    support_contribs.sort(reverse=True)
    lead_sum = sum(lead_contribs[:config.LEAD_TOP_K])
    support_sum = sum(support_contribs[:config.SUPPORT_TOP_K])
    # 闸门机制
    if lead_sum == 0:
        # 无 Lead 论文，强折扣
        score_paper = config.ALPHA_NO_LEAD * support_sum
    else:
        # 有 Lead 论文
        score_paper = lead_sum + config.BETA_SUPPORT * support_sum
    return score_paper

def calculate_final_score(cand):
    """
    计算候选人的最终评分
    公式：final_score = 10 * score_paper + (venue_fit + recency + role_leadership)
    """
    score_paper = getattr(cand, "score_paper", 0.0)
    radar = getattr(cand, "radar", {})
    
    if not isinstance(radar, dict):
        return 10 * score_paper
    venue_fit = radar.get("venue_fit", 0)
    recency = radar.get("recency_momentum", 0)
    role_leadership = radar.get("leadership", 0)
    final_score = 10 * score_paper + (venue_fit + recency + role_leadership)
    return final_score

# ============================ DIVERSITY RERANK FUNCTIONS ============================
# 多样性重排：避免同一篇论文的多个作者霸榜
# 新版本：分位数归一化 + 自适应gap窗口 + MMR混合

def _robust_normalize(scores: List[float], eps: float = 1e-6) -> Tuple[List[float], float, float]:
    """
    分位数归一化：把原始分数映射到 0~100
    Args:
        scores: 原始分数列表
        eps: 避免除零的小量
    Returns:
        (norm_scores, p10, p90)
    """
    import numpy as np
    
    if not scores:
        return [], 0.0, 100.0
    scores_arr = np.array(scores)
    p10 = float(np.percentile(scores_arr, 10))
    p90 = float(np.percentile(scores_arr, 90))
    spread = max(p90 - p10, eps)
    
    # 归一化到 0~100，并 clamp
    norm_scores = []
    for s in scores:
        norm = 100.0 * (s - p10) / spread
        norm = max(0.0, min(100.0, norm))  # clamp to [0, 100]
        norm_scores.append(norm)
    return norm_scores, p10, p90

def _get_eligible_window(
    remaining_candidates: List[Any],
    orig_norm_map: Dict[str, float],
    top_norm: float,
    remaining_slots: int,
    W: int,
    gap_start: float = 15.0,
    gap_step: float = 10.0,
    gap_max: float = 80.0
) -> Tuple[List[Any], float]:
    """
    获取符合条件的候选窗口（自适应 gap 放宽）
    Args:
        remaining_candidates: 剩余候选人列表
        orig_norm_map: 候选人名称 -> 归一化分数
        top_norm: 当前最高归一化分数
        remaining_slots: 剩余需要选择的人数
        W: 窗口大小
        gap_start: 初始 gap
        gap_step: 每次放宽的步长
        gap_max: 最大 gap
    Returns:
        (eligible_candidates, final_gap)
    """
    # 先按 orig_norm 降序排序，取 Top-W
    sorted_remaining = sorted(
        remaining_candidates, 
        key=lambda c: orig_norm_map.get(c.name, 0), 
        reverse=True
    )
    window = sorted_remaining[:W]
    # 最小候选池大小
    min_pool = min(len(window), max(10, 3 * remaining_slots))
    gap = gap_start
    while gap <= gap_max:
        eligible = [c for c in window if orig_norm_map.get(c.name, 0) >= top_norm - gap]
        if len(eligible) >= min_pool:
            return eligible, gap
        gap += gap_step
    # gap 已放到最大，返回整个 window
    return window, gap_max

def _mmr_score(
    orig_norm: float,
    div_norm: float,
    lam: float = 0.85,
    floor_ratio: float = 0.75
) -> float:
    """
    MMR 混合计算：质量主导 + 多样性辅助
    Args:
        orig_norm: 原始分数的归一化值 (0~100)
        div_norm: 多样性分数的归一化值 (0~100)
        lam: 原始分数权重（默认 0.85）
        floor_ratio: 护栏比例（防止砸穿）
    Returns:
        mmr 分数
    """
    mmr = lam * orig_norm + (1 - lam) * div_norm
    # 护栏：mmr 不能低于原始分的 floor_ratio
    floor = orig_norm * floor_ratio
    mmr = max(mmr, floor)
    return mmr

def _decay_function(used_count: int) -> float:
    """
    共享论文的衰减函数
    used=0 → 1.0（第一次出现吃满）
    used=1 → 0.3
    used=2 → 0.1
    used>=3 → 0.0
    """
    if used_count == 0:
        return 1.0
    elif used_count == 1:
        return 0.3
    elif used_count == 2:
        return 0.1
    else:
        return 0.0

def _get_candidate_top_papers(cand, all_scored_papers: Dict[str, Any], K: int = None) -> List[Dict[str, Any]]:
    """
    获取候选人的高质量论文（按贡献分降序）
    Args:
        K: 最大论文数，None 表示不限制
    Returns:
        List of dicts: [{"paper_url": str, "title": str, "venue": str, "score": int, 
                         "contribution": float, "position_index": int, "total_authors": int}, ...]
    """
    candidate_name = cand.name.lower().strip()
    selected_research = getattr(cand, "selected_research", None)
    if not selected_research:
        return []
    all_publications = getattr(selected_research, "all_publications", [])
    if not all_publications:
        return []
    papers = []
    for pub in all_publications:
        relevance_score = getattr(pub, "relevance_score", 0)
        if relevance_score < 6:  # 只考虑高质量论文
            continue
        paper_url = getattr(pub, "url", "") or ""
        title = getattr(pub, "title", "") or ""
        venue = getattr(pub, "venue", "") or ""
        # 获取作者位置信息
        position_index = getattr(pub, "candidate_author_position", None)
        total_authors = getattr(pub, "total_authors", None)
        if position_index is None:
            authors = getattr(pub, "authors", [])
            if authors:
                position_index, _, total_authors = compute_author_position(candidate_name, authors)
        
        if position_index is None:
            continue
        
        # 获取特殊角色标记
        is_corresponding = getattr(pub, "is_corresponding_author", False)
        is_cofirst = getattr(pub, "is_cofirst_author", False)
        # 计算贡献分
        author_coef = get_author_position_coef(position_index, total_authors, is_corresponding=is_corresponding, is_cofirst=is_cofirst)
        venue_lower = venue.lower() if venue else ""
        venue_coef = 0.5 if any(kw in venue_lower for kw in ["workshop", "arxiv", "preprint"]) else 1.0
        contribution = relevance_score * author_coef * venue_coef
        
        papers.append({
            "paper_url": paper_url,
            "title": title,
            "venue": venue,
            "score": relevance_score,
            "contribution": contribution,
            "position_index": position_index,
            "total_authors": total_authors,
        })
    
    # 按贡献分降序
    papers.sort(key=lambda x: x["contribution"], reverse=True)
    # 如果 K 不为 None，则取 Top-K；否则返回所有
    if K is not None:
        return papers[:K]
    return papers
def _pick_trigger_papers(
    cand,
    cand_papers: List[Dict[str, Any]],
    used_count: Dict[str, int],
    all_scored_papers: Dict[str, Any],
    M: int = 1,
    show_L: int = 1
) -> List['schemas.TriggerPaperInfo']:
    """
    为候选人选择触发论文（优先独立证据）
    Args:
        cand: 候选人对象
        cand_papers: 候选人的论文列表
        used_count: 论文使用计数器
        all_scored_papers: 所有评分论文字典
        M: 配额上限
        show_L: 展示论文数量（默认 1 篇）
    Returns:
        List of TriggerPaperInfo
    """
    # 获取候选人的所有论文（用于查找特殊角色标记）
    selected_research = getattr(cand, "selected_research", None)
    all_publications = getattr(selected_research, "all_publications", []) if selected_research else []
    
    indep = [(p, p["contribution"]) for p in cand_papers if used_count.get(p["paper_url"], 0) < M]
    shared = [(p, p["contribution"]) for p in cand_papers if used_count.get(p["paper_url"], 0) >= M]
    
    # 独立集合按贡献分降序
    indep.sort(key=lambda x: x[1], reverse=True)
    result = []
    # 优先从独立集合选
    for p, contrib in indep:
        if len(result) >= show_L:
            break
        
        # 获取作者位置描述（考虑特殊角色）
        pos_idx = p.get("position_index")
        total = p.get("total_authors")
        
        # 从原始论文对象获取特殊角色标记
        is_cofirst_for_display = False
        is_corresponding_for_display = False
        for pub in all_publications:
            if getattr(pub, "url", "") == p["paper_url"]:
                is_cofirst_for_display = getattr(pub, "is_cofirst_author", False)
                is_corresponding_for_display = getattr(pub, "is_corresponding_author", False)
                break
        
        pos_str = None
        if is_cofirst_for_display:
            pos_str = "co-first"
        elif is_corresponding_for_display:
            pos_str = "corresponding"
        elif pos_idx is not None:
            if pos_idx == 1:
                pos_str = "first"
            elif total and pos_idx == total:
                pos_str = "last"
            elif pos_idx == 2:
                pos_str = "second"
            else:
                pos_str = f"{pos_idx}th"
        
        trigger_info = schemas.TriggerPaperInfo(
            paper_url=p["paper_url"],
            title=p["title"],
            venue=p["venue"],
            score=p["score"],
            contribution=contrib,
            is_shared=False,
            author_position=pos_str,
            author_position_index=pos_idx,
            total_authors=total,
        )
        result.append(trigger_info)
    
    # 如果独立集合不够，用共享集合补齐
    if len(result) < show_L:
        # 共享集合按衰减后的贡献排序
        shared.sort(key=lambda x: x[1] * _decay_function(used_count.get(x[0]["paper_url"], 0)), reverse=True)
        
        for p, contrib in shared:
            if len(result) >= show_L:
                break
            
            # 获取作者位置描述（考虑特殊角色）
            pos_idx = p.get("position_index")
            total = p.get("total_authors")
            # 从原始论文对象获取特殊角色标记
            is_cofirst_for_display = False
            is_corresponding_for_display = False
            for pub in all_publications:
                if getattr(pub, "url", "") == p["paper_url"]:
                    is_cofirst_for_display = getattr(pub, "is_cofirst_author", False)
                    is_corresponding_for_display = getattr(pub, "is_corresponding_author", False)
                    break
            pos_str = None
            if is_cofirst_for_display:
                pos_str = "co-first"
            elif is_corresponding_for_display:
                pos_str = "corresponding"
            elif pos_idx is not None:
                if pos_idx == 1:
                    pos_str = "first"
                elif total and pos_idx == total:
                    pos_str = "last"
                elif pos_idx == 2:
                    pos_str = "second"
                else:
                    pos_str = f"{pos_idx}th"
            
            trigger_info = schemas.TriggerPaperInfo(
                paper_url=p["paper_url"],
                title=p["title"],
                venue=p["venue"],
                score=p["score"],
                contribution=contrib * _decay_function(used_count.get(p["paper_url"], 0)),
                is_shared=True,  # 标记为共享证据
                author_position=pos_str,
                author_position_index=pos_idx,
                total_authors=total,
            )
            result.append(trigger_info)
    return result
def _calculate_diversity_score(
    cand,
    cand_papers: List[Dict[str, Any]],
    used_count: Dict[str, int],
    M: int = 1,
    L: int = 2,
    ratio_threshold: float = 0.5,
    penalty_factor: float = 0.6
) -> Tuple[float, float]:
    """
    计算候选人的多样性调整后评分
    Args:
        cand: 候选人对象
        cand_papers: 候选人的 Top-K 论文
        used_count: 论文使用计数
        M: 配额上限
        L: indep_sum 取独立 Top-L
        ratio_threshold: 独立性阈值
        penalty_factor: 惩罚因子
    Returns:
        (diversity_adjusted_score, independent_ratio)
    """
    indep = [(p, p["contribution"]) for p in cand_papers if used_count.get(p["paper_url"], 0) < M]
    shared = [(p, p["contribution"]) for p in cand_papers if used_count.get(p["paper_url"], 0) >= M]
    
    # 独立集合按贡献分降序
    indep.sort(key=lambda x: x[1], reverse=True)
    
    # indep_sum: 取独立 Top-L 的总和
    indep_sum = sum(x[1] for x in indep[:L])
    
    # shared_sum: 应用 decay 后的总和
    shared_sum = sum(
        contrib * _decay_function(used_count.get(p["paper_url"], 0))
        for p, contrib in shared
    )
    
    paper_adj = indep_sum + shared_sum
    
    # 获取 extras (venue_fit + recency + role_leadership)
    radar = getattr(cand, "radar", {})
    if not isinstance(radar, dict):
        extras = 0
    else:
        extras = radar.get("venue_fit", 0) + radar.get("recency_momentum", 0) + radar.get("leadership", 0)
    
    # 计算调整后的评分
    import math
    score_adj = 10 * math.log1p(paper_adj) + extras
    
    # 独立性比例
    total_contrib = indep_sum + shared_sum + 1e-9
    ratio = indep_sum / total_contrib
    
    # 如果独立性太低，应用惩罚
    if ratio < ratio_threshold:
        score_adj *= penalty_factor
    
    return score_adj, ratio
def diversity_rerank_topN(
    candidates: List[Any],
    all_scored_papers: Dict[str, Any],
    N: int,
    M: int = 1
) -> List[Any]:
    """
    多样性重排：选择 Top-N 候选人，避免同一篇论文的作者霸榜
    新版本算法：
    1. 分位数归一化（p10/p90 → 0~100）
    2. 自适应 gap 窗口（不跨档插队）
    3. MMR 混合（质量主导 + 多样性辅助）
    Args:
        candidates: 候选人列表
        all_scored_papers: 所有评分论文
        N: 目标数量 (Top-N)
        M: 论文配额上限（一篇论文最多支撑 M 个候选人，默认 1 为最强多样性）
    Returns:
        重排后的 Top-N 候选人列表（已设置 trigger_papers 字段）
    """
    import math
    # ==================== 硬编码参数 ====================
    L = 2                    # indep_sum 取独立 Top-2
    ratio_threshold = 0.5    # 独立性阈值
    penalty_factor = 0.6     # 低独立性惩罚因子
    # MMR 参数
    lam = 0.85               # 原始分数权重
    floor_ratio = 0.75       # 护栏比例
    # 窗口参数
    gap_start = 15.0         # 初始 gap
    gap_step = 10.0          # 每次放宽步长
    gap_max = 80.0           # 最大 gap
    print(f"\n{'='*80}")
    print(f"[Diversity Rerank] Starting diversity-aware Top-{N} selection (M={M})")
    print(f"{'='*80}")
    print(f"[Diversity Rerank] Total candidates to rerank: {len(candidates)}")
    print(f"[Diversity Rerank] Algorithm: Percentile Normalization + Adaptive Gap Window + MMR")
    if not candidates:
        return []
    # ==================== Step 0: 预计算 ====================
    # 预计算每个候选人的所有高质量论文
    cand_papers_cache: Dict[str, List[Dict[str, Any]]] = {}
    for cand in candidates:
        cand_papers_cache[cand.name] = _get_candidate_top_papers(cand, all_scored_papers, K=None)
    
    # ==================== Step 1: 分位数归一化 original_score ====================
    original_scores = [getattr(c, "final_score", 0.0) for c in candidates]
    orig_norm_list, p10, p90 = _robust_normalize(original_scores)
    # 构建候选人名称 -> 归一化分数的映射
    orig_norm_map: Dict[str, float] = {}
    for i, cand in enumerate(candidates):
        orig_norm_map[cand.name] = orig_norm_list[i]
    print(f"[Diversity Rerank] Score normalization: p10={p10:.2f}, p90={p90:.2f}")
    # ==================== Step 3: 归一化 diversity_score（used_count=0 时的初始值） ====================
    empty_used_count: Dict[str, int] = {}
    div_raw_scores = []
    for cand in candidates:
        cand_papers = cand_papers_cache.get(cand.name, [])
        if not cand_papers:
            div_raw = getattr(cand, "final_score", 0.0)
        else:
            div_raw, _ = _calculate_diversity_score(
                cand, cand_papers, empty_used_count, M, L, ratio_threshold, penalty_factor
            )
        div_raw_scores.append(div_raw)
    
    div_norm_list, div_p10, div_p90 = _robust_normalize(div_raw_scores)
    div_norm_init_map: Dict[str, float] = {}
    for i, cand in enumerate(candidates):
        div_norm_init_map[cand.name] = div_norm_list[i]
    print(f"[Diversity Rerank] Diversity normalization (init): p10={div_p10:.2f}, p90={div_p90:.2f}")
    # ==================== Step 4-6: Greedy 选择 ====================
    selected = []
    used_count: Dict[str, int] = {}  # paper_url -> used count
    remaining = list(candidates)
    W = max(50, 5 * N)  # 窗口大小
    print(f"[Diversity Rerank] Window size W={W}, λ={lam}, floor_ratio={floor_ratio}")
    print(f"[Diversity Rerank] Gap: start={gap_start}, step={gap_step}, max={gap_max}")
    print(f"\n[Diversity Rerank] Selection process:")
    
    while len(selected) < N and remaining:
        remaining_slots = N - len(selected)
        # 确定当前最高归一化分数（以剩余候选人中的最高分为准）
        top_norm = max(orig_norm_map.get(c.name, 0) for c in remaining)
        # Step 4: 获取 eligible 窗口
        eligible, used_gap = _get_eligible_window(
            remaining, orig_norm_map, top_norm, remaining_slots, W, gap_start, gap_step, gap_max
        )
        # Step 5: 在 eligible 中用 MMR 选择最佳
        best_cand = None
        best_mmr = -1e18
        best_div_raw = 0.0
        best_ratio = 1.0
        
        for cand in eligible:
            cand_papers = cand_papers_cache.get(cand.name, [])
            
            # 计算当前 used_count 下的 diversity_score
            if not cand_papers:
                div_raw = getattr(cand, "final_score", 0.0)
                ratio = 1.0
            else:
                div_raw, ratio = _calculate_diversity_score(
                    cand, cand_papers, used_count, M, L, ratio_threshold, penalty_factor
                )
            
            # 将 div_raw 映射到归一化空间（使用初始化时的 p10/p90）
            div_spread = max(div_p90 - div_p10, 1e-6)
            div_norm_dynamic = 100.0 * (div_raw - div_p10) / div_spread
            div_norm_dynamic = max(0.0, min(100.0, div_norm_dynamic))
            # 获取原始分数的归一化值
            orig_norm = orig_norm_map.get(cand.name, 0)
            # MMR 计算
            mmr = _mmr_score(orig_norm, div_norm_dynamic, lam, floor_ratio)
            if mmr > best_mmr:
                best_cand = cand
                best_mmr = mmr
                best_div_raw = div_raw
                best_ratio = ratio
        
        if best_cand is None:
            break
        
        # Step 6: 为最佳候选人选择触发论文（只展示 1 篇）
        best_cand_papers = cand_papers_cache.get(best_cand.name, [])
        best_triggers = _pick_trigger_papers(
            best_cand, best_cand_papers, used_count, all_scored_papers, M, show_L=1
        )
        
        # 更新候选人的 trigger_papers 和 diversity 字段
        best_cand.trigger_papers = best_triggers
        best_cand.diversity_adjusted_score = best_div_raw  # 保存原始 diversity 分数用于 debug
        best_cand.independent_ratio = best_ratio
        
        # 同时更新旧的单个 trigger_paper_* 字段（向后兼容）
        if best_triggers:
            first_trigger = best_triggers[0]
            best_cand.trigger_paper_title = first_trigger.title
            best_cand.trigger_paper_url = first_trigger.paper_url
            best_cand.trigger_paper_venue = first_trigger.venue
            best_cand.trigger_paper_score = first_trigger.score
            best_cand.trigger_paper_position = first_trigger.author_position
            best_cand.trigger_paper_position_index = first_trigger.author_position_index
            best_cand.trigger_paper_total_authors = first_trigger.total_authors
        
        selected.append(best_cand)
        remaining.remove(best_cand)
        # 关键：只对“展示触发论文”更新 used_count
        for trigger in best_triggers:
            paper_url = trigger.paper_url
            if paper_url:
                used_count[paper_url] = used_count.get(paper_url, 0) + 1
        # 打印选择信息
        orig_score = getattr(best_cand, "final_score", 0.0)
        orig_norm_val = orig_norm_map.get(best_cand.name, 0)
        trigger_title = best_triggers[0].title[:35] + "..." if best_triggers and len(best_triggers[0].title) > 35 else (best_triggers[0].title if best_triggers else "N/A")
        shared_mark = "※" if best_triggers and best_triggers[0].is_shared else ""
        print(f"  #{len(selected):2d} {best_cand.name[:20]:20s} | orig={orig_score:.1f}(norm={orig_norm_val:.1f}) mmr={best_mmr:.1f} gap={used_gap:.0f} | {trigger_title}{shared_mark}")
    print(f"\n[Diversity Rerank] Selected {len(selected)} candidates")
    return selected
def compute_diversity_metrics(selected_candidates: List[Any]) -> Dict[str, Any]:
    """
    计算多样性效果指标（用于日志验收）
    Returns:
        {
            "unique_paper_coverage": float,  # 唯一论文覆盖率
            "max_paper_frequency": int,       # 最大论文出现次数
            "duplicate_rate": float,          # 重复率
            "total_triggers": int,            # 触发论文总数
            "unique_triggers": int,           # 唯一触发论文数
        }
    """
    all_trigger_urls = []
    for cand in selected_candidates:
        triggers = getattr(cand, "trigger_papers", [])
        for t in triggers:
            if t.paper_url:
                all_trigger_urls.append(t.paper_url)
    
    total_triggers = len(all_trigger_urls)
    unique_triggers = len(set(all_trigger_urls))
    
    if total_triggers == 0:
        return {
            "unique_paper_coverage": 1.0,
            "max_paper_frequency": 0,
            "duplicate_rate": 0.0,
            "total_triggers": 0,
            "unique_triggers": 0,
        }
    
    # 计算每篇论文出现次数
    from collections import Counter
    freq = Counter(all_trigger_urls)
    max_freq = max(freq.values()) if freq else 0
    
    # 唯一论文覆盖率
    coverage = unique_triggers / len(selected_candidates) if selected_candidates else 1.0
    
    # 重复率
    duplicate_count = total_triggers - unique_triggers
    duplicate_rate = duplicate_count / total_triggers if total_triggers > 0 else 0.0
    
    return {
        "unique_paper_coverage": coverage,
        "max_paper_frequency": max_freq,
        "duplicate_rate": duplicate_rate,
        "total_triggers": total_triggers,
        "unique_triggers": unique_triggers,
    }

def deduplicate_papers(all_serp: List[Dict[str, Any]], all_scored_papers: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    对论文列表进行去重（智能保留高质量版本）
    去重策略（任一条件满足即判定为重复）：
    1. arXiv ID 相同（从URL提取）
    2. OpenReview ID 相同
    3. DOI 相同
    4. 标题高度相似(≥85%) + 作者显著重叠(≥50%)
    **质量优先策略**：
    当发现重复论文时，根据venue质量决定保留哪个版本：
    - 等级3 (顶级会议正式录用) > 等级2 (其他会议) > 等级1 (Workshop) > 等级0 (arXiv/Preprint)
    - 例如：同一篇论文既在"ICLR Workshop 2024"又在"ICLR 2025"正式录用
      → 保留"ICLR 2025"版本，舍弃Workshop版本
    Args:
        all_serp: 所有SERP结果列表
        all_scored_papers: 所有已评分的论文字典
    Returns:
        去重后的 (all_serp, all_scored_papers)
    """
    from difflib import SequenceMatcher
    import re
    
    print(f"\n{'='*80}")
    print(f"[Paper Deduplication] Starting deduplication")
    print(f"{'='*80}")
    print(f"[Paper Deduplication] Before deduplication:")
    print(f"  - Total SERP items: {len(all_serp)}")
    print(f"  - Total scored papers: {len(all_scored_papers)}")
    
    def extract_arxiv_id(url: str) -> Optional[str]:
        """从URL中提取arXiv ID"""
        if not url:
            return None
        match = re.search(r'arxiv\.org/(?:abs|pdf)/([0-9]+\.[0-9]+)', url.lower())
        if match:
            return match.group(1)
        return None
    
    def extract_openreview_id(url: str) -> Optional[str]:
        """从URL中提取OpenReview ID"""
        if not url:
            return None
        # 匹配 openreview.net/forum?id=pzS8MJkf06
        match = re.search(r'openreview\.net/forum\?id=([A-Za-z0-9_-]+)', url)
        if match:
            return match.group(1)
        return None
    
    def extract_doi(url: str) -> Optional[str]:
        """从URL中提取DOI"""
        if not url:
            return None
        # 匹配 doi.org/10.xxxx/xxxx 或其他DOI格式
        match = re.search(r'doi\.org/(10\.[0-9]+/[^\s]+)', url.lower())
        if match:
            return match.group(1)
        return None
    
    def normalize_title(title: str) -> str:
        """标准化论文标题：小写、去除多余空格和标点，移除venue后缀"""
        title = title.lower().strip()
        
        # 移除常见的venue后缀模式（在标题末尾）
        # 例如："— Accepted by ES-FoMo III 2025", "— Accepted by CoRR 2025", "— Accepted at ICLR 2024"
        venue_patterns = [
            r'\s*—\s*accepted\s+by\s+[^—]+$',  # "— Accepted by ..."
            r'\s*—\s*accepted\s+at\s+[^—]+$',  # "— Accepted at ..."
            r'\s*—\s*published\s+in\s+[^—]+$',  # "— Published in ..."
            r'\s*—\s*in\s+[^—]+$',              # "— In ..."
            r'\s*\(accepted\s+by\s+[^)]+\)\s*$', # "(Accepted by ...)"
            r'\s*\(accepted\s+at\s+[^)]+\)\s*$', # "(Accepted at ...)"
            r'\s*\[accepted\s+by\s+[^\]]+\]\s*$', # "[Accepted by ...]"
            r'\s*\[accepted\s+at\s+[^\]]+\]\s*$', # "[Accepted at ...]"
        ]
        for pattern in venue_patterns:
            title = re.sub(pattern, '', title, flags=re.IGNORECASE)
        
        # 移除常见标点符号
        title = re.sub(r'[:\-_\.,;!?]', ' ', title)
        # 移除多余空格
        title = re.sub(r'\s+', ' ', title)
        return title.strip()
    
    def normalize_authors(authors: List[str]) -> List[str]:
        """标准化作者列表：小写、去除空格"""
        if not authors:
            return []
        return [author.lower().strip() for author in authors]
    
    def titles_similar(title1: str, title2: str, threshold: float = 0.85) -> bool:
        """
        判断两个标题是否相似（使用序列匹配）
        降低阈值到85%以捕获标题略有修改的情况（如添加会议名、年份等）
        """
        norm_title1 = normalize_title(title1)
        norm_title2 = normalize_title(title2)
        
        # 完全相同
        if norm_title1 == norm_title2:
            return True
        
        # 序列相似度（降低到85%以捕获更多变体）
        ratio = SequenceMatcher(None, norm_title1, norm_title2).ratio()
        return ratio >= threshold
    def authors_overlap_significant(authors1: List[str], authors2: List[str], threshold: float = 0.50) -> bool:
        """
        判断两个作者列表是否有显著重叠（≥50%）
        降低阈值以捕获作者顺序变化或添加/删除合作者的情况
        """
        norm_authors1 = set(normalize_authors(authors1))
        norm_authors2 = set(normalize_authors(authors2))
        
        if not norm_authors1 or not norm_authors2:
            return False
        
        # 计算Jaccard相似度：交集 / 并集
        intersection = len(norm_authors1 & norm_authors2)
        union = len(norm_authors1 | norm_authors2)
        
        if union == 0:
            return False
        
        overlap_ratio = intersection / union
        return overlap_ratio >= threshold
    
    def get_venue_quality(venue: str) -> int:
        """
        判断venue的质量等级（用于去重时保留更高质量的版本）
        返回值：数字越大，质量越高
        
        等级：
        - 3: 顶级会议正式录用 (ICLR, ICML, NeurIPS, ACL, EMNLP, etc.)
        - 2: 其他会议正式录用
        - 1: Workshop
        - 0: arXiv preprint / 未知
        """
        if not venue:
            return 0
        
        venue_lower = venue.lower()
        
        # Workshop 标识
        workshop_keywords = ['workshop', 'ws', 'w@', 'findings']
        if any(kw in venue_lower for kw in workshop_keywords):
            return 1
        
        # arXiv / preprint
        if 'arxiv' in venue_lower or 'preprint' in venue_lower:
            return 0
        
        # 顶级会议列表（从config获取）
        top_conferences = [
            'iclr', 'icml', 'neurips', 'nips',  # ML
            'acl', 'emnlp', 'naacl', 'coling', 'eacl',  # NLP
            'cvpr', 'iccv', 'eccv',  # CV
            'kdd', 'www', 'sigir', 'wsdm',  # DM/IR
            'aaai', 'ijcai',  # AI
            'sigmod', 'vldb', 'icde',  # DB
        ]
        
        for conf in top_conferences:
            if conf in venue_lower:
                return 3
        
        # 其他会议（有会议名但不是workshop/arxiv）
        return 2
    
    def should_keep_new_over_old(new_venue: str, old_venue: str) -> bool:
        """
        判断在去重时是否应该保留新版本而不是旧版本
        规则：保留venue质量更高的版本
        """
        new_quality = get_venue_quality(new_venue)
        old_quality = get_venue_quality(old_venue)
        
        # 如果新版本质量更高，保留新版本
        return new_quality > old_quality
    
    def papers_are_duplicate(paper1_data: Dict, paper2_data: Dict) -> Tuple[bool, str]:
        """
        判断两篇论文是否重复
        返回: (是否重复, 原因)
        """
        url1 = paper1_data.get("url", "")
        url2 = paper2_data.get("url", "")
        title1 = paper1_data.get("title", "")
        title2 = paper2_data.get("title", "")
        authors1 = paper1_data.get("authors", [])
        authors2 = paper2_data.get("authors", [])
        
        # 策略1: arXiv ID相同
        arxiv1 = extract_arxiv_id(url1)
        arxiv2 = extract_arxiv_id(url2)
        if arxiv1 and arxiv2 and arxiv1 == arxiv2:
            return True, f"Same arXiv ID: {arxiv1}"
        
        # 策略2: OpenReview ID相同
        openreview1 = extract_openreview_id(url1)
        openreview2 = extract_openreview_id(url2)
        if openreview1 and openreview2 and openreview1 == openreview2:
            return True, f"Same OpenReview ID: {openreview1}"
        
        # 策略3: DOI相同
        doi1 = extract_doi(url1)
        doi2 = extract_doi(url2)
        if doi1 and doi2 and doi1 == doi2:
            return True, f"Same DOI: {doi1}"
        
        # 策略4: 标题高度相似 + 作者显著重叠
        if titles_similar(title1, title2):
            # 如果有作者信息，检查作者重叠
            if authors1 and authors2:
                if authors_overlap_significant(authors1, authors2):
                    return True, "Similar title + significant author overlap"
            else:
                # 如果作者信息缺失，但标题高度相似（≥90%），也判定为重复
                # 这可以处理venue后缀不同但核心标题相同的情况
                norm_title1 = normalize_title(title1)
                norm_title2 = normalize_title(title2)
                if norm_title1 == norm_title2:
                    return True, "Identical normalized title (venue suffix removed)"
                # 如果标准化后完全相同，说明只是venue后缀不同
                from difflib import SequenceMatcher
                ratio = SequenceMatcher(None, norm_title1, norm_title2).ratio()
                if ratio >= 0.90:  # 提高阈值到90%以处理venue后缀差异
                    return True, f"Very similar title (ratio={ratio:.2f}, venue suffix may differ)"
        
        return False, ""
    
    # 去重 all_serp（优先保留正式会议版本）
    unique_serp = []
    seen_papers = []
    duplicate_count_serp = 0
    replaced_count_serp = 0  # 被更高质量版本替换的数量
    
    for serp_item in all_serp:
        title = serp_item.get("title", "")
        
        if not title:
            unique_serp.append(serp_item)
            continue
        
        # 检查是否与已有论文重复
        is_duplicate = False
        
        for idx, seen_paper in enumerate(seen_papers):
            is_dup, reason = papers_are_duplicate(serp_item, seen_paper)
            if is_dup:
                is_duplicate = True
                duplicate_count_serp += 1
                
                # 比较venue质量
                new_venue = serp_item.get("venue", "")
                old_venue = seen_paper.get("venue", "")
                
                if should_keep_new_over_old(new_venue, old_venue):
                    # 新版本质量更高，替换旧版本
                    replaced_count_serp += 1
                    print(f"[Paper Deduplication] 🔄 Replacing with higher-quality venue:")
                    print(f"  - Title: {title[:80]}...")
                    print(f"  - Old venue: '{old_venue}' → New venue: '{new_venue}'")
                    print(f"  - Reason: {reason}")
                    
                    # 替换seen_papers和unique_serp中的旧版本
                    seen_papers[idx] = serp_item
                    # 在unique_serp中找到并替换
                    for i, item in enumerate(unique_serp):
                        if item.get("url") == seen_paper.get("url") or item.get("title") == seen_paper.get("title"):
                            unique_serp[i] = serp_item
                            break
                else:
                    # 保留旧版本
                    print(f"[Paper Deduplication] 🔍 Found duplicate (keeping existing):")
                    print(f"  - Title: {title[:80]}...")
                    print(f"  - Existing venue: '{old_venue}' (kept)")
                    print(f"  - New venue: '{new_venue}' (skipped)")
                    print(f"  - Reason: {reason}")
                
                break
        
        if not is_duplicate:
            unique_serp.append(serp_item)
            seen_papers.append(serp_item)
    
    # 去重 all_scored_papers（优先保留正式会议版本）
    unique_scored_papers = {}
    seen_scored_papers = []
    duplicate_count_scored = 0
    replaced_count_scored = 0  # 被更高质量版本替换的数量
    
    for url, paper in all_scored_papers.items():
        title = paper.title
        
        # 从 all_serp 中找到对应的完整信息
        paper_data = None
        for serp_item in all_serp:
            if serp_item.get("url") == url or serp_item.get("title") == title:
                paper_data = serp_item
                break
        
        # 如果没找到，构造最小paper_data
        if not paper_data:
            paper_data = {
                "title": title,
                "url": url,
                "authors": [],
                "venue": paper.venue  # 从paper对象获取venue
            }
        
        if not title:
            unique_scored_papers[url] = paper
            continue
        
        # 检查是否与已有论文重复
        is_duplicate = False
        duplicate_url = None
        
        for idx, seen_data in enumerate(seen_scored_papers):
            is_dup, reason = papers_are_duplicate(paper_data, seen_data)
            if is_dup:
                is_duplicate = True
                duplicate_url = seen_data.get("url")
                duplicate_count_scored += 1
                
                # 比较venue质量
                new_venue = paper_data.get("venue", "")
                old_venue = seen_data.get("venue", "")
                
                if should_keep_new_over_old(new_venue, old_venue):
                    # 新版本质量更高，替换旧版本
                    replaced_count_scored += 1
                    print(f"[Paper Deduplication] 🔄 Replacing scored paper with higher-quality venue:")
                    print(f"  - Title: {title[:80]}...")
                    print(f"  - Old venue: '{old_venue}' → New venue: '{new_venue}'")
                    print(f"  - Reason: {reason}")
                    
                    # 替换seen_scored_papers
                    seen_scored_papers[idx] = paper_data
                    # 从unique_scored_papers中删除旧版本，添加新版本
                    unique_scored_papers.pop(duplicate_url, None)
                    unique_scored_papers[url] = paper
                else:
                    # 保留旧版本
                    print(f"[Paper Deduplication] 🔍 Found duplicate scored paper (keeping existing):")
                    print(f"  - Title: {title[:80]}...")
                    print(f"  - Existing venue: '{old_venue}' (kept)")
                    print(f"  - New venue: '{new_venue}' (skipped)")
                    print(f"  - Reason: {reason}")
                
                break
        
        if not is_duplicate:
            unique_scored_papers[url] = paper
            seen_scored_papers.append(paper_data)
    
    print(f"\n[Paper Deduplication] ✅ Deduplication complete:")
    print(f"  - SERP items: {len(all_serp)} → {len(unique_serp)} (removed {duplicate_count_serp - replaced_count_serp}, replaced {replaced_count_serp})")
    print(f"  - Scored papers: {len(all_scored_papers)} → {len(unique_scored_papers)} (removed {duplicate_count_scored - replaced_count_scored}, replaced {replaced_count_scored})")
    print(f"  - Quality-based replacements: {replaced_count_serp + replaced_count_scored} (workshop → conference)")
    print(f"{'='*80}\n")
    
    return unique_serp, unique_scored_papers


def _expand_candidates_from_related_papers_once(
    candidates_accum: Dict[str, Any],
    enhanced_profiles_accum: Dict[str, Any],
    all_scored_papers: Dict[str, Any],
    spec: Any,
    api_key: str = None,
) -> None:
    """
    Based on high_quality_publications from existing candidates, expand to find new authors (one round only).
    
    Args:
        candidates_accum: Accumulated candidates dict (will be updated in-place)
        enhanced_profiles_accum: Accumulated enhanced profiles dict (will be updated in-place)
        all_scored_papers: All scored papers dict (will be updated in-place)
        spec: QuerySpec for search parameters
        api_key: API key for LLM calls
    """
    print(f"\n{'='*80}")
    print(f"[Related Papers Expansion] Starting second-round author discovery")
    print(f"{'='*80}\n")
    
    # Step 1: Freeze seed candidates (only use first-round candidates' related papers)
    seed_candidate_names = set(candidates_accum.keys())
    print(f"[Related Papers] Seed candidates: {len(seed_candidate_names)}")
    
    if not seed_candidate_names:
        print(f"[Related Papers] No seed candidates, skipping expansion")
        return
    
    # Step 2: Collect all high_quality_publications from seed candidates
    all_related_papers = []
    for cand_name in seed_candidate_names:
        profile = enhanced_profiles_accum.get(cand_name)
        if not profile:
            continue
        
        high_quality_pubs = getattr(profile, 'high_quality_publications', [])
        if not high_quality_pubs:
            continue
        
        # 只使用 6 分及以上的论文来搜索新人才（虽然 profile 中存储了 6 分以上的）
        for pub in high_quality_pubs:
            score = pub.get('relevance_score', 0)
            if score > 6:  # 搜索新人才时只用 6 分以上
                pub['_source_candidate'] = cand_name
                all_related_papers.append(pub)
    
    print(f"[Related Papers] Collected {len(all_related_papers)} high-quality publications from {len(seed_candidate_names)} candidates")
    
    if not all_related_papers:
        print(f"[Related Papers] No related papers found, skipping expansion")
        return
    
    # Step 3: Deduplicate related papers (URL + title+year)
    existing_urls = set(all_scored_papers.keys())
    seen_urls = set()
    seen_title_keys = set()
    new_related_papers = []
    
    def normalize_title(title: str) -> str:
        import re
        title = title.lower().strip()
        title = re.sub(r'[^\w\s]', '', title)
        title = re.sub(r'\s+', ' ', title)
        return title
    
    for pub in all_related_papers:
        url = pub.get('url', '').strip()
        title = pub.get('title', '').strip()
        year = pub.get('year', '')
        
        if not title:
            continue
        
        # Skip if URL already in all_scored_papers (first-round paper)
        if url and url in existing_urls:
            continue
        
        # Skip if URL already seen in this round
        if url and url in seen_urls:
            continue
        
        # Deduplicate by normalized title + year
        title_key = f"{normalize_title(title)}::{year}"
        if title_key in seen_title_keys:
            continue
        
        # This is a truly new related paper
        if url:
            seen_urls.add(url)
        seen_title_keys.add(title_key)
        new_related_papers.append(pub)
    
    print(f"[Related Papers] After deduplication: {len(new_related_papers)} new papers")
    
    if not new_related_papers:
        print(f"[Related Papers] No new papers after deduplication, skipping expansion")
        return
    
    # Step 4: Add new related papers to all_scored_papers (no LLM scoring)
    for pub in new_related_papers:
        url = pub.get('url', '').strip()
        if not url:
            continue
        
        # Use existing relevance_score from high_quality_publications
        score = int(pub.get('relevance_score', 7))
        score = max(1, min(8, score))  # Clamp to 1-8
        
        paper = schemas.PaperWithScore(
            url=url,
            title=pub.get('title', ''),
            abstract=pub.get('abstract', '') or '',
            introduction=pub.get('introduction', '') or '',
            venue=pub.get('venue', '') or 'Unknown venue',
            score=score,
            explanation=pub.get('relevance_explanation', ''),
            authors=pub.get('authors', []),
            associated_candidates=[],
            # Multi-dimensional scoring fields
            relevance_score=float(pub.get('relevance_score', score)),
            quality_score=float(pub.get('quality_score', 0.0)),
            relevance_dimensions=pub.get('relevance_dimensions', {}),
            quality_dimensions=pub.get('quality_dimensions', {}),
            relevance_explanation=pub.get('relevance_explanation', '')
        )
        
        if url not in all_scored_papers:
            all_scored_papers[url] = paper
    
    print(f"[Related Papers] Added {len(new_related_papers)} new papers to all_scored_papers")
    
    # Step 5: Extract authors from new related papers (with deduplication)
    existing_authors = set(candidates_accum.keys())
    new_authors_this_round = set()
    related_candidates_to_process = []
    
    for pub in new_related_papers:
        paper_title = pub.get('title', '')
        paper_url = pub.get('url', '')
        authors = pub.get('authors', [])
        
        if not authors or not isinstance(authors, list):
            continue
        
        for author_name in authors:
            author_name = author_name.strip()
            if not author_name:
                continue
            
            # Validate author name
            is_valid, reason = is_valid_author_name(author_name)
            if not is_valid:
                continue
            
            # Skip if already a candidate or already added in this round
            if author_name in existing_authors or author_name in new_authors_this_round:
                continue
            
            new_authors_this_round.add(author_name)
            related_candidates_to_process.append((author_name, None, paper_title, paper_url))
    
    print(f"[Related Papers] Extracted {len(related_candidates_to_process)} new authors from {len(new_related_papers)} papers")
    
    if not related_candidates_to_process:
        print(f"[Related Papers] No new authors to process, expansion complete")
        return
    
    # Step 6: Process new authors with orchestrate_candidate_report (parallel)
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from .dynamic_concurrency import get_candidate_workers
    import psutil
    
    required_candidates = spec.top_n or 5
    max_workers = get_candidate_workers(
        candidates=len(related_candidates_to_process),
        required=required_candidates
    )
    
    # Memory safety check
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    memory_safe_workers = int(available_memory_gb / 0.5)
    max_workers = min(max_workers, memory_safe_workers)
    
    print(f"[Related Papers] Processing {len(related_candidates_to_process)} authors with {max_workers} workers")
    
    # Build user query
    user_query_parts = []
    if spec.keywords:
        user_query_parts.append(" ".join(spec.keywords))
    user_query = " ".join(user_query_parts) or "research papers"
    
    def _submit_related_candidate(ex, author_name, author_id, paper_title, paper_url):
        paper_venue = ""
        paper_score = 0
        paper_explanation = ""
        paper_authors = []
        paper_info = None
        
        if paper_url and paper_url in all_scored_papers:
            sp = all_scored_papers[paper_url]
            paper_venue = sp.venue
            paper_score = sp.score
            paper_explanation = sp.explanation
            paper_authors = sp.authors
            paper_info = {
                "title": sp.title,
                "abstract": sp.abstract,
                "authors": sp.authors,
                "venue": sp.venue,
                "url": sp.url,
                "relevance_score": getattr(sp, 'relevance_score', float(paper_score)),
                "quality_score": getattr(sp, 'quality_score', 0.0),
                "relevance_dimensions": getattr(sp, 'relevance_dimensions', {}),
                "quality_dimensions": getattr(sp, 'quality_dimensions', {}),
            }
        
        exclude_urls = set(all_scored_papers.keys())
        if paper_url and paper_url in exclude_urls:
            exclude_urls.remove(paper_url)
        
        return ex.submit(
            orchestrate_candidate_report,
            first_author=author_name,
            paper_title=paper_title,
            paper_url=paper_url,
            paper_venue=paper_venue,
            paper_score=paper_score,
            paper_explanation=paper_explanation,
            paper_authors=paper_authors,
            aliases=[author_name],
            author_id=author_id,
            api_key=api_key,
            user_query=user_query,
            exclude_paper_urls=exclude_urls,
            paper_info=paper_info,
        )
    
    new_candidates_count = 0
    
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {}
        idx = 0
        
        # Submit initial batch
        while idx < len(related_candidates_to_process) and len(futures) < max_workers:
            author_name, author_id, paper_title, paper_url = related_candidates_to_process[idx]
            fut = _submit_related_candidate(ex, author_name, author_id, paper_title, paper_url)
            futures[fut] = (author_name, author_id, paper_title, paper_url)
            idx += 1
        
        # Process results
        while futures:
            for fut in as_completed(list(futures.keys()), timeout=None):
                author_name, author_id, paper_title, paper_url = futures.pop(fut)
                
                try:
                    result = fut.result()
                    if result is None:
                        print(f"[Related Papers] {author_name}: orchestrate returned None")
                        profile, overview, eval_res, enhanced_profile = None, None, None, None
                    elif len(result) == 4:
                        profile, overview, eval_res, enhanced_profile = result
                    else:
                        profile, overview, eval_res = result
                        enhanced_profile = None
                except Exception as e:
                    print(f"[Related Papers] {author_name} error: {str(e)[:100]}")
                    profile, overview, eval_res, enhanced_profile = None, None, None, None
                
                if overview:
                    category = classify_candidate_overview(overview, api_key=api_key)
                    overview.score_paper = calculate_paper_score(overview)
                    overview.final_score = calculate_final_score(overview)
                    
                    candidates_accum[overview.name] = overview
                    overview.candidate_category = category
                    overview.discovered_in_round = -1  # Mark as related-papers round
                    
                    new_candidates_count += 1
                    print(f"[Related Papers] Added: {overview.name} ({category}) - score_paper={overview.score_paper:.1f}, final_score={overview.final_score:.2f}")
                    
                    if enhanced_profile:
                        enhanced_profiles_accum[overview.name] = enhanced_profile
                        if overview.name != author_name:
                            enhanced_profiles_accum[author_name] = enhanced_profile
                    
                    # Link paper to candidate
                    if paper_url and paper_url in all_scored_papers:
                        if overview.name not in all_scored_papers[paper_url].associated_candidates:
                            all_scored_papers[paper_url].associated_candidates.append(overview.name)
                
                # Submit next task
                if idx < len(related_candidates_to_process):
                    next_author_name, next_author_id, next_paper_title, next_paper_url = related_candidates_to_process[idx]
                    nfut = _submit_related_candidate(ex, next_author_name, next_author_id, next_paper_title, next_paper_url)
                    futures[nfut] = (next_author_name, next_author_id, next_paper_title, next_paper_url)
                    idx += 1
        
        # Cancel remaining
        for fut in list(futures.keys()):
            author_name, author_id, paper_title, paper_url = futures.pop(fut)
            fut.cancel()
    
    print(f"\n[Related Papers] Expansion complete:")
    print(f"  - New papers added to reference: {len(new_related_papers)}")
    print(f"  - New authors processed: {len(related_candidates_to_process)}")
    print(f"  - New candidates generated: {new_candidates_count}")
    print(f"  - Total candidates now: {len(candidates_accum)}")
    print(f"{'='*80}\n")


# ============================ AGENT FUNCTIONS ============================
def agent_parse_search_query(search_query: str, api_key: str = None) -> QuerySpec:
    """
    Parse a natural language search query into structured QuerySpec
    Args:
        search_query: Natural language query from user
        api_key: Optional API key for LLM calls
    Returns:
        QuerySpec: Structured search parameters
    """
    try:
        llm_instance = llm.get_llm("parse", temperature=0.3, api_key=api_key)

        conf_list = ", ".join(config.DEFAULT_CONFERENCES.keys())
        
        # 构建研究领域列表
        fields_list = ", ".join(list(config.CS_TOP_CONFERENCES.keys()))
        
        prompt = f"""
        You are a professional **Talent Recruitment Query Parser**. 
        Analyze the user's query and extract structured information in **strict JSON**. 
        Do NOT output any text outside JSON.

        === FIELDS TO EXTRACT ===
        1. **top_n** (int): Number of candidates. Extract numbers like "10 candidates", "20 people". Default: {config.DEFAULT_TOP_N} when not specified.
        2. **years** (int[]): Years for papers. Default: {config.DEFAULT_YEARS}.
        3. **venues** (string[]): Target conferences/journals. 
        Recognize names or variants in query. Known examples: {conf_list}.
        4. **keywords** (string[]): Technical terms or research topics.
        Rules:
        - Only extract exact phrases appearing in query (case-insensitive).
        - Keep multi-word phrases intact (e.g., "graph foundation model").
        - Ignore inferred/related terms, abbreviations, or general fields ("AI", "ML") unless explicitly mentioned.
        - Ignore negated terms (e.g., "not X", "no X", "without X").
        5. **research_field** (string): Primary field inferred from keywords/context.
        Options: {fields_list}.
        Mapping examples:
        - "robot", "navigation" → Robotics
        - "NLP", "language model" → Natural Language Processing
        - "vision", "segmentation" → Computer Vision
        - "deep learning", "neural" → Machine Learning
        - "database", "SQL" → Databases
        - "security", "encryption" → Computer Security
        Default: Machine Learning
        6. **must_be_current_student** (bool): True if mentions "current student", "PhD student", etc. Default true unless stated otherwise.
        7. **degree_levels** (string[]): Candidate category requirements. ONLY these 7 categories are valid:
        - "Master" / "Master student" → Currently enrolled Master students ONLY
        - "PhD" / "PhD student" → Currently enrolled PhD students ONLY
        - "Undergraduate" / "Undergraduate student" / "Bachelor" → Currently enrolled Undergraduate students ONLY
        - "Professor" → Professors at any level (Assistant/Associate/Full)
        - "Postdoc" → Postdoctoral researchers
        - "Industrial Researcher" → Researchers in industry/companies
        - "Institution Researcher" → Researchers in academic/research institutions
        - Extract EXACTLY as one of these 7 categories.
        - If explicit categories exist → use only them; else default: ["PhD", "Master"].
        8. **author_priority** (string[]): [DEPRECATED - System now extracts ALL authors] Keep as ["first"] for compatibility.
        9. **extra_constraints** (string[]): Other constraints such as region, institution, language, or experience.

        === EXTRACTION PRINCIPLES ===
        - Prefer explicit info; infer only for field classification.
        - Be concise and consistent.
        - Output clean, valid JSON only.

        User Query:
        {search_query}
        """

        query_spec = llm.safe_structured(llm_instance, prompt, schemas.QuerySpec)

        # 如果venues为空，根据研究领域智能选择会议
        if query_spec.venues == []:
            print("[Parse Query] No venues specified, selecting based on research field...")
            print(f"[Parse Query] Identified research field: {query_spec.research_field}")
            
            # 方案A：始终包含核心会议 + 研究领域会议
            selected_conferences = config.CORE_CONFERENCES.copy()  # 始终包含核心ML会议
            print(f"[Parse Query] Core ML conferences: {selected_conferences}")
            
            # 根据识别的研究领域添加专业会议
            if query_spec.research_field and query_spec.research_field in config.CS_TOP_CONFERENCES:
                field_conferences = config.CS_TOP_CONFERENCES[query_spec.research_field]
                print(f"[Parse Query] Field-specific conferences from '{query_spec.research_field}': {field_conferences}")
                selected_conferences.extend(field_conferences)
            else:
                print(f"[Parse Query] Warning: Research field '{query_spec.research_field}' not found in conference mapping")
            
            # 去重（保持顺序：核心会议优先，然后是领域会议）
            seen = set()
            deduped_conferences = []
            for conf in selected_conferences:
                if conf not in seen:
                    seen.add(conf)
                    deduped_conferences.append(conf)
            
            query_spec.venues = deduped_conferences
            
            print(f"[Parse Query] Final selected venues ({len(query_spec.venues)}): {query_spec.venues}")

        return query_spec

    except Exception as e:
        print(f"LLM解析失败，使用模拟数据: {e}")

        # 回退到模拟数据
        fallback_years = config.DEFAULT_YEARS.copy()
        fallback_keywords = []
        fallback_research_field = "Machine Learning"  # 默认研究领域
        
        fallback_venues = config.CORE_CONFERENCES.copy()  # 始终包含核心ML会议
        
        # 根据研究领域添加专业会议
        if fallback_research_field in config.CS_TOP_CONFERENCES:
            field_conferences = config.CS_TOP_CONFERENCES[fallback_research_field]
            fallback_venues.extend(field_conferences)
        
        # 去重（保持顺序）
        fallback_venues = list(dict.fromkeys(fallback_venues))
        
        print(f"[Fallback] Using default configuration:")
        print(f"  Years: {fallback_years}")
        print(f"  Research Field: {fallback_research_field}")
        print(f"  Venues: {fallback_venues}")
        print(f"  Keywords: {fallback_keywords}")
        
        return schemas.QuerySpec(
            top_n=10,
            years=fallback_years,
            venues=fallback_venues,
            keywords=fallback_keywords if fallback_keywords else ["machine learning"],
            research_field=fallback_research_field,
            must_be_current_student=True,
            degree_levels=["PhD", "Master"],  # Default: PhD and Master students
            author_priority=["first"],
        )
def _plan_terms(spec: QuerySpec) -> List[Dict[str, Any]]:
    """
    Generate structured search parameters with query variants (从严到松)
    第一个variant：原始keywords空格拼接（主query）
    后续variants：扩展备用（防止主query失效）
    Returns:
        List of dicts with structure:
        [
            {"keywords": ["kw1", "kw2"], "offset": 0, "exhausted": False, "name": "primary"},
            {"keywords": ["kw1"], "offset": 0, "exhausted": False, "name": "anchor_only"},
            ...
        ]
    """
    try:
        import re
        kw = spec.keywords if spec.keywords else []
        
        if not kw:
            return [{"keywords": ["machine learning"], "offset": 0, "exhausted": False, "name": "fallback"}]
        # ==================== 辅助函数 ====================
        
        def is_generic(k: str) -> bool:
            """判断keyword是否为泛词"""
            return k.lower().strip() in config.GENERIC_KEYWORDS
        def normalize_symbols(k: str) -> str:
            """标准化符号：连字符/下划线转空格"""
            k = k.replace("-", " ").replace("_", " ")
            # 压缩多余空格
            k = re.sub(r'\s+', ' ', k).strip()
            return k
        def filter_generic(keywords: List[str]) -> List[str]:
            """过滤掉泛词"""
            return [k for k in keywords if not is_generic(k)]
        def choose_anchor(keywords: List[str]) -> str:
            """
            选择最具体的keyword作为锚点
            优先级：非泛词 > 多词短语 > 长词
            """
            if not keywords:
                return ""
            scored = []
            for k in keywords:
                score = 0
                # 非泛词优先
                if not is_generic(k):
                    score += 1000
                # 多词短语加分
                score += len(k.split()) * 100
                # 长度加分
                score += len(k)
                scored.append((score, k))
            scored.sort(reverse=True)
            return scored[0][1]
        def is_valid_variant(keywords: List[str]) -> bool:
            """验证variant是否合法（不能全是泛词）"""
            if not keywords:
                return False
            # 至少有一个非泛词
            return any(not is_generic(k) for k in keywords)
        # ==================== 生成 Variants ====================
        variants = []
        # Variant 0: 原始keywords（主query，必须有）
        variants.append({
            "keywords": kw,
            "offset": 0,
            "exhausted": False,
            "name": "primary"
        })
        # ==================== 以下是备用variants ====================
        # Variant 1: 符号标准化（处理连字符问题，如 human-AI → human AI）
        norm_kw = [normalize_symbols(k) for k in kw]
        if norm_kw != kw:
            variants.append({
                "keywords": norm_kw,
                "offset": 0,
                "exhausted": False,
                "name": "normalized"
            })
        # Variant 2: 去掉泛词（如果有泛词如LLM/AI等）
        non_generic = filter_generic(kw)
        if len(non_generic) >= 1 and len(non_generic) < len(kw):
            variants.append({
                "keywords": non_generic,
                "offset": 0,
                "exhausted": False,
                "name": "drop_generic"
            })
        # Variant 3: 去掉最后一个keyword（缩小范围，只在>2个时）
        if len(kw) > 2 and is_valid_variant(kw[:-1]):
            variants.append({
                "keywords": kw[:-1],
                "offset": 0,
                "exhausted": False,
                "name": "drop_last"
            })
        # Variant 4: 锚点词单独查询（最宽松，但保证不是泛词）
        if len(non_generic) > 1:
            anchor = choose_anchor(non_generic)
            if anchor and not is_generic(anchor):
                variants.append({
                    "keywords": [anchor],
                    "offset": 0,
                    "exhausted": False,
                    "name": "anchor_only"
                })
        # ==================== 去重（跳过第一个primary） ====================
        unique_variants = [variants[0]]  # 保留primary
        seen_keys = {tuple(sorted([k.lower() for k in variants[0]["keywords"]]))}
        
        for v in variants[1:]:
            key = tuple(sorted([k.lower() for k in v["keywords"]]))
            if key not in seen_keys and len(v["keywords"]) > 0:
                seen_keys.add(key)
                unique_variants.append(v)
        # 限制最大数量
        max_variants = config.QUERY_POOL_CONFIG.get("max_variants", 5)
        unique_variants = unique_variants[:max_variants]
        # 打印生成的variants
        print(f"\n{'='*80}")
        print(f"[Query Pool] 生成 {len(unique_variants)} 个query variants:")
        for i, v in enumerate(unique_variants):
            query_str = " ".join(v["keywords"])
            print(f"  Variant {i} ({v['name']}): \"{query_str}\"")
        print(f"{'='*80}\n")
        return unique_variants
    except Exception as e:
        print(f"[_plan_terms] Error: {e}")
        import traceback
        traceback.print_exc()
        return [{"keywords": spec.keywords if spec.keywords else ["machine learning"], "offset": 0, "exhausted": False, "name": "error_fallback"}]

def _normalize_name_key(name: str) -> str:
    import re

    return re.sub(r"[^a-z\\s]", "", name.lower()).strip()


def _match_author_name(lead_name: str, author_name: str) -> bool:
    if not lead_name or not author_name:
        return False
    lead_key = _normalize_name_key(lead_name)
    author_key = _normalize_name_key(author_name)
    if not lead_key or not author_key:
        return False
    if lead_key == author_key:
        return True
    lead_parts = [p for p in lead_key.split() if p]
    author_parts = [p for p in author_key.split() if p]
    if not lead_parts or not author_parts:
        return False
    if lead_parts[-1] == author_parts[-1] and lead_parts[0][:1] == author_parts[0][:1]:
        return True
    return False


def _build_seed_inputs_from_leads(
    leads: List[Dict[str, Any]],
    api_key: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], List[Tuple[str, Optional[str], str, str]], Dict[str, Optional[str]]]:
    from .semantic_paper_search import SemanticScholarClient
    from .arxiv_fallback import ArxivFallbackClient
    from .utils import sanitize_author_ids

    seed_serp: List[Dict[str, Any]] = []
    seed_candidates: List[Tuple[str, Optional[str], str, str]] = []
    seed_name_to_id: Dict[str, Optional[str]] = {}
    seen_titles: set[str] = set()

    s2_client = SemanticScholarClient(
        api_key=config.SEMANTIC_SCHOLAR_API_KEY or None,
        timeout=12.0,
        requests_per_second=1.0,
    )
    arxiv_client = ArxivFallbackClient()

    for lead in leads:
        if not isinstance(lead, dict):
            continue
        name = (lead.get("name") or "").strip()
        paper_title = (lead.get("paper_title") or lead.get("paper") or lead.get("title") or "").strip()
        paper_url = (lead.get("paper_url") or lead.get("url") or "").strip()
        paper_year = lead.get("paper_year") or lead.get("year")
        paper_venue = lead.get("paper_venue") or lead.get("venue") or ""
        author_id = lead.get("author_id") or lead.get("authorId") or lead.get("s2_author_id")

        details = None
        if paper_title:
            details = s2_client.get_paper_full_details(
                title=paper_title,
                year=str(paper_year) if paper_year else None,
                venue=paper_venue or None,
                min_match_score=config.BING_PAPER_MIN_MATCH_SCORE,
            )
        if not details and paper_title:
            details = arxiv_client.search_by_title(paper_title, max_results=3)

        serp_item: Dict[str, Any] = {}
        if details:
            title = details.get("title", "") or paper_title
            if title:
                title_key = title.lower().strip()
                if title_key in seen_titles:
                    continue
                seen_titles.add(title_key)

            authors = details.get("authors", []) or []
            author_ids = sanitize_author_ids(details.get("author_ids", []))
            matched_author_id = author_id
            if name and author_ids:
                for a_name, a_id in zip(authors, author_ids):
                    if _match_author_name(name, a_name):
                        matched_author_id = a_id
                        break
            if name:
                seed_name_to_id[name] = matched_author_id

            serp_item = {
                "title": title,
                "authors": authors,
                "author_ids": author_ids,
                "venue": details.get("venue", "") or paper_venue or "",
                "year": details.get("year") or paper_year,
                "paper_id": details.get("paper_id", ""),
                "topic": "seed_leads",
                "source": details.get("data_source", "seed_leads"),
                "url": details.get("url") or paper_url,
                "abstract": details.get("abstract", "") or "",
                "introduction": details.get("tldr", "") or "",
                "pdf_url": details.get("pdf_url", "") or "",
            }
        elif paper_title or paper_url:
            title_key = paper_title.lower().strip() if paper_title else ""
            if title_key and title_key in seen_titles:
                continue
            if title_key:
                seen_titles.add(title_key)
            serp_item = {
                "title": paper_title,
                "authors": lead.get("authors", []) or [],
                "author_ids": sanitize_author_ids(lead.get("author_ids", [])) if lead.get("author_ids") else [],
                "venue": paper_venue or "",
                "year": paper_year,
                "paper_id": "",
                "topic": "seed_leads",
                "source": "seed_leads",
                "url": paper_url,
                "abstract": lead.get("abstract", "") or "",
                "introduction": lead.get("introduction", "") or "",
                "pdf_url": lead.get("pdf_url", "") or "",
            }

        if serp_item:
            seed_serp.append(serp_item)
            if not paper_title:
                paper_title = serp_item.get("title", "") or ""
            if not paper_url:
                paper_url = serp_item.get("url", "") or ""

        if name:
            seed_candidates.append((name, author_id, paper_title, paper_url))

    return seed_serp, seed_candidates, seed_name_to_id

def _filter_survey_position_papers(papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    过滤掉 survey 和 position paper
    这类论文通常不是原创研究，不适合用来发现新研究人才
    
    Args:
        papers: 论文列表，每个元素是一个 dict，包含 'title' 字段
        
    Returns:
        过滤后的论文列表
    """
    import re
    
    # Survey/Review/Position Paper 关键词模式（强信号）
    strong_patterns = [
        r'\b(survey|review|overview)\b',
        r'\ba survey of\b',
        r'\ba review of\b',
        r'\ban overview of\b',
        r'\bsurvey on\b',
        r'\breview on\b',
        r'\bposition paper\b',
        r'\bposition statement\b',
        r'\bstate of the art\b',
        r'\bstate-of-the-art\b',
        r'\btaxonomy\b',
        r'\bcomprehensive survey\b',
        r'\bcomprehensive review\b',
        r'\bsystematic review\b',
        r'\bsystematic survey\b',
        r'\bliterature review\b',
        r'\bliterature survey\b',
        r'\btutorial\b',  # Tutorial papers
    ]
    
    filtered_papers = []
    for paper in papers:
        title = paper.get('title', '').strip()
        if not title:
            # 没有标题的论文保留
            filtered_papers.append(paper)
            continue
        
        title_lower = title.lower()
        
        # 检查强信号
        is_survey_or_position = False
        for pattern in strong_patterns:
            if re.search(pattern, title_lower, re.IGNORECASE):
                is_survey_or_position = True
                if config.VERBOSE:
                    print(f"[Paper Filter] ❌ Filtered survey/position paper: {title}")
                break
        
        if not is_survey_or_position:
            filtered_papers.append(paper)
    
    return filtered_papers

def _run_search_terms(
    terms: List[Dict[str, Any]],
    k_per_query: int = 10,
    years: Optional[List[int]] = None,
    venues: Optional[List[str]] = None,
    seen_titles: Optional[Set[str]] = None,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Search for papers using Bing, then enrich with Semantic Scholar/arXiv metadata.
    Each call returns up to k_per_query papers, rotating through query variants.
    """
    if not terms:
        return [], False

    from .paper_search import build_bing_query, search_papers_bing

    all_results: List[Dict[str, Any]] = []
    seen_titles = {t.lower().strip() for t in (seen_titles or set()) if t}

    while len(all_results) < k_per_query:
        current_term = None
        for t in terms:
            if not t.get("exhausted", False):
                current_term = t
                break
        if current_term is None:
            print(f"[Bing Search] All terms exhausted, collected {len(all_results)} papers")
            break

        keywords = current_term.get("keywords", [])
        query = build_bing_query(keywords, years=years, venues=venues)
        attempts = current_term.get("offset", 0) + 1
        papers_needed = k_per_query - len(all_results)
        print(f"[Bing Search] Searching: {query} (attempt={attempts}, need={papers_needed})")

        new_results = search_papers_bing(
            query=query,
            k_per_query=papers_needed,
            keywords=keywords,
            years=years,
            venues=venues,
            seen_titles=seen_titles,
        )

        if not new_results:
            current_term["exhausted"] = True
            print(f"[Bing Search] No new papers for '{query}', marking variant exhausted")
            continue

        current_term["offset"] = attempts
        for paper in new_results:
            title_key = (paper.get("title") or "").lower().strip()
            if title_key:
                seen_titles.add(title_key)
        all_results.extend(new_results)

    has_more = any(not t.get("exhausted", False) for t in terms)
    if all_results:
        original_count = len(all_results)
        filtered_results = _filter_survey_position_papers(all_results)
        filtered_count = original_count - len(filtered_results)
        if filtered_count > 0:
            print(
                f"[Bing Search] ✗ Filtered {filtered_count} survey/position papers "
                f"(kept {len(filtered_results)}/{original_count})"
            )
        all_results = filtered_results
    print(f"[Bing Search] This round: {len(all_results)} papers collected, has_more={has_more}")
    return all_results, has_more

def _select_urls(
    serp: List[Dict[str, Any]], spec: QuerySpec, api_key: str = None
) -> Tuple[List[str], List[Dict[str, Any]], List[schemas.PaperWithScore]]:
    """
    Select URLs to fetch from Semantic Scholar search results.
    
    Strategy:
    1. Trust all papers from S2 Search (no filtering)
    2. Fetch full content (introduction + abstract)
    3. Score with LLM based on full content
    4. Sort by priority (score)
    5. Extract authors and process candidates
    
    Returns:
        urls: List of paper URLs to fetch
        serps_ordered: Corresponding SERP items
        scored_papers: Empty list (scoring happens after fetching)
    """
    print(f"[select] Processing {len(serp)} S2 search results")
    urls = []
    serps_ordered = []
    for item in serp:
        url = item.get("url", "")
        title = item.get("title", "")
        # Skip invalid items
        if not url or not title:
            continue
        urls.append(url)
        serps_ordered.append(item)
    
    print(f"[select] Selected {len(urls)} papers for content fetching and scoring")
    
    # Return empty scored_papers - scoring will happen after fetching full content
    return urls, serps_ordered, []
def clean_venue(venue: str) -> str:
    """
    清理venue字符串，移除"poster"、"oral"等展示类型信息
    特别处理 arXiv：arXiv 是预印本平台，不应该显示 "Accepted by arXiv" 等表述
    Args:
        venue: 原始venue字符串
    Returns:
        清理后的venue字符串
    """
    if not venue:
        return venue
    import re
    venue_lower = venue.lower().strip()
    # 特殊处理 arXiv：arXiv 是预印本平台，不是正式会议/期刊
    # 不应该显示 "Accepted by arXiv", "arxiv.org", "arXiv preprint" 等
    arxiv_patterns = [
        r'^accepted\s+(by|at|to)\s+arxiv',  # "Accepted by arXiv"
        r'^arxiv\.org$',                     # "arxiv.org"
        r'^arxiv$',                           # "arXiv"
        r'^corr$',                            # "CoRR" (Computing Research Repository)
        r'^arxiv\s*preprint$',               # "arXiv preprint"
        r'^arxiv:\s*\d+\.\d+',               # "arXiv:2312.12345"
    ]
    for pattern in arxiv_patterns:
        if re.match(pattern, venue_lower, re.IGNORECASE):
            return ""  # 返回空字符串，让上层逻辑决定显示什么
    # 移除常见的展示类型标识（不区分大小写）
    # 移除 "| Poster", "| Oral", "| Spotlight" 等
    venue = re.sub(r'\s*\|\s*(poster|oral|spotlight|workshop|demo|tutorial)\s*', '', venue, flags=re.IGNORECASE)
    # 移除 " (Poster)", " (Oral)" 等
    venue = re.sub(r'\s*\(poster|oral|spotlight|workshop|demo|tutorial\)\s*', '', venue, flags=re.IGNORECASE)
    # 移除开头的 "Poster: ", "Oral: " 等
    venue = re.sub(r'^(poster|oral|spotlight|workshop|demo|tutorial):\s*', '', venue, flags=re.IGNORECASE)
    # 移除 "Accepted by" 前缀（如果存在）
    venue = re.sub(r'^accepted\s+(by|at|to)\s+', '', venue, flags=re.IGNORECASE)
    return venue.strip()

def classify_candidate_category(role_text: str) -> str:
    """
    将候选人分类到7个类别之一
    
    类别（优先级从高到低）：
    1. Undergraduate Student - 本科生在读
    2. Master Student - 硕士在读
    3. PhD Student - 博士在读
    4. Professor - 教授（任何级别）
    5. Postdoc - 博士后
    6. Industrial Researcher - 工业研究员
    7. Institution Researcher - 机构研究员
    
    Args:
        role_text: 职位描述文本
    
    Returns:
        str: 类别名称，如果无法分类则返回"Unknown"
    """
    if not role_text or len(role_text) < 3:
        return "Unknown"
    
    # 标准化：转小写 + 去掉点号（处理"Ph.D."等缩写）
    t = role_text.lower().replace(".", "")
    
    # 按优先级匹配（避免误分类）
    
    # 1. Undergraduate Student（优先于Master/PhD，因为本科生是最低级别）
    if any(kw in t for kw in ["undergraduate student", "undergrad student", "bachelor student",
                              "b.s. student", "b.a. student", "bs student", "ba student",
                              "undergraduate", "undergrad"]):
        return "Undergraduate Student"
    
    # 2. Master Student
    if any(kw in t for kw in ["master student", "master's student", "ms student", 
                              "msc student", "m.s. student", "m.sc. student"]):
        return "Master Student"
    
    # 3. PhD Student（优先于Professor，因为有些教授title可能包含phd）
    if any(kw in t for kw in ["phd student", "ph.d. student", "doctoral student", 
                              "phd candidate", "doctoral candidate", "doctorate student"]):
        return "PhD Student"
    
    # 3. Professor（各级教授）
    if any(kw in t for kw in ["professor", "assistant professor", "associate professor", 
                              "full professor", "prof.", "chair professor"]):
        return "Professor"
    
    # 4. Postdoc（博士后）
    if any(kw in t for kw in ["postdoc", "post-doc", "postdoctoral", 
                              "postdoctoral researcher", "postdoctoral fellow"]):
        return "Postdoc"
    
    # 5. Industrial Researcher（工业研究员）
    # 检查是否在公司/企业工作
    if any(kw in t for kw in ["scientist at", "engineer at", "researcher at", 
                              "research scientist at", "applied scientist",
                              "staff scientist", "senior researcher at"]):
        # 进一步检查是否是公司（排除学术机构）
        if not any(uni in t for uni in ["university", "institute", "college", "lab"]):
            return "Industrial Researcher"
    
    # 6. Institution Researcher（机构研究员）
    if "researcher" in t and not any(kw in t for kw in ["student", "professor", "postdoc"]):
        return "Institution Researcher"
    
    # 无法分类
    return "Unknown"

def _llm_infer_candidate_category_from_profile(ov: CandidateOverview, role_text: str = "", api_key: str = None) -> Optional[str]:
    """
    LLM-based fallback for inferring candidate category from full profile.
    Uses background/education/industrial_experience/teaching fields with time information.
    Only called when rule-based classification returns "Unknown".
    Args:
        ov: CandidateOverview with structured profile fields
        role_text: Previously extracted role text (for context)
        api_key: Optional API key for LLM calls
    
    Returns:
        One of the 7 categories or "Unknown", or None if LLM fails
    """
    allowed_categories = [
        "Undergraduate Student",
        "Master Student",
        "PhD Student",
        "Professor",
        "Postdoc",
        "Industrial Researcher",
        "Institution Researcher",
        "Unknown"
    ]
    try:
        from .llm import get_llm
        
        llm_instance = get_llm("role_infer", temperature=0.1, api_key=api_key)
        
        # Extract structured fields
        name = getattr(ov, "name", "")
        introduction = getattr(ov, "introduction", None)
        education = getattr(ov, "education", []) or []
        industrial_exp = getattr(ov, "industrial_experience", []) or []
        teaching = getattr(ov, "teaching", []) or []
        
        # Build context from introduction
        intro_lines = []
        if introduction:
            intro_pos = getattr(introduction, "position", "") or ""
            intro_aff = getattr(introduction, "affiliation", "") or ""
            intro_text = getattr(introduction, "text", "") or ""
            if intro_pos or intro_aff:
                combined = f"Position: {intro_pos} at {intro_aff}".strip()
                intro_lines.append(combined)
            if intro_text:
                intro_lines.append(f"Intro: {intro_text}")
        
        # Build context from education (with time)
        edu_lines = []
        for edu in education[:5]:  # Limit to 5 most recent
            degree = getattr(edu, "degree", "")
            institution = getattr(edu, "institution", "")
            duration = getattr(edu, "duration", "")
            parts = [p for p in [degree, institution, duration] if p]
            if parts:
                edu_lines.append(" - " + ", ".join(parts))
        
        # Build context from industrial experience (with time)
        ind_lines = []
        for exp in industrial_exp[:5]:
            position = getattr(exp, "position", "")
            org = getattr(exp, "organization", "")
            duration = getattr(exp, "duration", "")
            parts = [p for p in [position, org, duration] if p]
            if parts:
                ind_lines.append(" - " + ", ".join(parts))
        
        # Build context from teaching (with time)
        teach_lines = []
        for t in teaching[:5]:
            # TeachingInfo might have 'title' or 'course' field
            title = getattr(t, "title", "") or getattr(t, "course", "")
            org = getattr(t, "organization", "")
            duration = getattr(t, "duration", "")
            parts = [p for p in [title, org, duration] if p]
            if parts:
                teach_lines.append(" - " + ", ".join(parts))
        
        # Assemble profile text
        profile_text_parts = []
        if name:
            profile_text_parts.append(f"Name: {name}")
        if role_text:
            profile_text_parts.append(f"Existing role text: {role_text}")
        if intro_lines:
            profile_text_parts.append("BACKGROUND:")
            profile_text_parts.extend(intro_lines)
        if edu_lines:
            profile_text_parts.append("EDUCATION:")
            profile_text_parts.extend(edu_lines)
        if ind_lines:
            profile_text_parts.append("INDUSTRIAL EXPERIENCE:")
            profile_text_parts.extend(ind_lines)
        if teach_lines:
            profile_text_parts.append("TEACHING:")
            profile_text_parts.extend(teach_lines)
        
        profile_text = "\n".join(profile_text_parts)
        
        prompt = f"""
        You are given structured profile information for a researcher. Your task is to infer the person's current primary role.
        You MUST answer with exactly one of the following categories:
        - Undergraduate Student
        - Master Student
        - PhD Student
        - Professor
        - Postdoc
        - Industrial Researcher
        - Institution Researcher
        - Unknown

        Guidelines:
        1. Prefer roles that are CURRENT (ongoing). Use durations like "now", "present", or missing end dates to infer current entries.
        2. If the person appears to be a student (Undergraduate/Master/PhD), only classify them as such if they are CURRENTLY enrolled.
        3. If they have both academic and industrial roles, choose the one that best represents their main current position.
        4. If information is too weak or ambiguous, choose "Unknown".
        5. CRITICAL: If the profile shows ONLY undergraduate enrollment with NO explicit evidence of pursuing a Master's or PhD, the role MUST be "Undergraduate Student".

        PROFILE:
        {profile_text}

        Answer with ONLY the category name from the list above.
        """.strip()
        
        response = llm_instance.invoke(prompt, enable_thinking=False)
        raw = getattr(response, "content", "") if response is not None else ""
        category = (raw or "").strip()
        
        # Normalize category (case-insensitive match)
        if category not in allowed_categories:
            category_upper = category.upper()
            for cat in allowed_categories:
                if cat.upper() == category_upper:
                    return cat
            return None
        
        return category
        
    except Exception as e:
        if config.VERBOSE:
            safe_print(f"[_llm_infer_candidate_category_from_profile] LLM fallback failed: {str(e)[:80]}")
        return None

def classify_candidate_overview(ov: CandidateOverview, api_key: str = None) -> str:
    """
    对候选人进行分类（不过滤）
    从CandidateOverview的introduction字段提取职位，分类到6个类别之一
    Args:
        ov: Candidate overview (9-field structure)
        api_key: Optional API key for LLM calls
    Returns:
        str: 类别名称 ("Undergraduate Student", "Master Student", "PhD Student", "Professor", "Postdoc", 
             "Industrial Researcher", "Institution Researcher", "Unknown")
    """
    candidate_name = getattr(ov, "name", "Unknown")
    
    try:
        pre_label = getattr(ov, "candidate_category", None)
        # 因为 "Unknown" 可能来自 determine_current_role_from_profile 的 LLM 判断，
        # 应该尊重这个判断，避免被后续的简单规则分类覆盖
        if pre_label is not None:
            safe_print(f"[Classify] {candidate_name} → {pre_label} (pre-labeled, skipping re-classification)")
            return pre_label
        # Extract role from introduction
        introduction = getattr(ov, "introduction", None)
        
        role_text = ""
        if introduction:
            position = getattr(introduction, "position", "") or ""
            affiliation = getattr(introduction, "affiliation", "") or ""
            
            # Combine position and affiliation
            if position and affiliation:
                role_text = f"{position} at {affiliation}"
            elif position:
                role_text = position
            elif affiliation:
                role_text = affiliation
            else:
                intro_text = getattr(introduction, "text", "") or ""
                role_text = intro_text
        
        # 使用新的分类函数
        category = classify_candidate_category(role_text)
        
        # 如果规则分类失败（Unknown），使用LLM基于完整profile进行兜底推断
        if category == "Unknown":
            llm_category = _llm_infer_candidate_category_from_profile(ov, role_text, api_key=api_key)
            if llm_category and llm_category != "Unknown":
                category = llm_category
                safe_print(f"[Classify] {candidate_name} → {category} (LLM fallback)")
            else:
                safe_print(f"[Classify] {candidate_name} → {category}")
        else:
            safe_print(f"[Classify] {candidate_name} → {category}")
        
        safe_print(f"   Role: {role_text[:80]}")
        
        return category
        
    except Exception as e:
        safe_print(f"[Classify] ⚠️ Exception for {candidate_name}: {str(e)[:50]}")
        return "Unknown"
# ============================ NAME VALIDATION ============================
def is_valid_author_name(name: str) -> tuple[bool, str]:
    """
    Validate if a string is a valid author name.
    Returns:
        (is_valid, reason)
    """
    if not config.ENABLE_NAME_FILTER:
        return True, "filter disabled"
    if not name or not isinstance(name, str):
        return False, "empty or invalid type"
    name = name.strip()
    # 1. Length check
    if len(name) < config.MIN_NAME_LENGTH:
        return False, f"too short (< {config.MIN_NAME_LENGTH} chars)"
    if len(name) > config.MAX_NAME_LENGTH:
        return False, f"too long (> {config.MAX_NAME_LENGTH} chars)"
    
    # 2. Must contain at least one letter
    if not any(c.isalpha() for c in name):
        return False, "no alphabetic characters"
    
    # 3. Cannot be all digits
    name_clean = name.replace(" ", "").replace(".", "").replace("-", "")
    if name_clean.isdigit():
        return False, "only digits"
    
    # 4. Email or URL check
    if "@" in name or "http" in name.lower() or "www." in name.lower():
        return False, "looks like email or URL"
    
    # 5. Institution keywords (common in error cases)
    institution_keywords = [
        "university", "institute", "lab", "laboratory", "research",
        "center", "centre", "college", "department", "school",
        "corporation", "company", "inc", "ltd", "llc", "foundation",
        "academy", "association", "society", "consortium"
    ]
    name_lower = name.lower()
    for keyword in institution_keywords:
        if keyword in name_lower:
            return False, f"institution keyword: {keyword}"
    
    # 6. Special symbols check (allow only: letters, spaces, hyphens, periods, apostrophes)
    import re
    allowed_pattern = r"^[a-zA-Z\u4e00-\u9fff\s.\-']+$"
    if not re.match(allowed_pattern, name):
        return False, "invalid characters"
    
    # 7. Check for single character names (unless Chinese)
    words = name.split()
    if len(words) == 1:
        # Allow Chinese names (2-4 characters)
        if all('\u4e00' <= ch <= '\u9fff' for ch in name if ch.strip()):
            if 2 <= len(name) <= 4:
                return True, "valid Chinese name"
        # Single letter or too short
        if len(name) <= 2:
            return False, "single letter or too short"
    
    # 8. Format validation for Western names
    if len(words) >= 2:
        # Check if words start with capital letters (common for real names)
        has_proper_caps = sum(1 for w in words if w and w[0].isupper()) >= len(words) * 0.5
        if not has_proper_caps:
            return False, "improper capitalization"
    
    # 9. All uppercase check (likely acronyms, not names)
    if name.isupper() and len(name) > 5 and " " not in name:
        return False, "all uppercase (likely acronym)"
    
    # 10. Suspicious patterns
    suspicious_patterns = [
        r'^\d+$',           # Pure numbers
        r'^[A-Z\s]+$',      # All caps multi-word (likely organization)
        r'.*\d{3,}.*',      # Contains 3+ consecutive digits
    ]
    for pattern in suspicious_patterns:
        if re.match(pattern, name):
            return False, f"matches suspicious pattern: {pattern}"
    
    return True, "passed all checks"

def agent_execute_search(
    spec: QuerySpec, 
    api_key: str = None, 
    progress_callback=None,
    log_callback=None,
    max_rounds_per_run: int = 1,
    resume_state: Optional[schemas.SearchTaskState] = None,
    seed_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Union[schemas.SearchResults, schemas.PartialSearchResults]:
    """Run real pipeline: plan -> search -> select -> fetch -> extract -> filter -> score -> sort.
    Supports incremental search with user decision points every N rounds.
    Returns:
        - SearchResults: If search is complete (all candidates found or user chose to finish)
        - PartialSearchResults: If need user decision (after max_rounds_per_run)
    Args:
        spec: Query specification
        api_key: API key for LLM calls
        progress_callback: Optional callback function(event: str, progress: float) for progress updates
        log_callback: Optional callback function(log_message: str) for real-time log streaming
        max_rounds_per_run: Number of candidate pool processing rounds before pausing (default: 1, changed from 2)
        resume_state: Optional task state to resume from
        seed_candidates: Optional list of seed candidate dicts from a prior web search (name, paper_title, paper_url, etc.)
    """
    # Capture logs for debugging
    import io
    import sys
    from pathlib import Path
    log_capture = io.StringIO()
    original_stdout = sys.stdout
    
    # Determine task_id early for log file naming
    # If resuming, use existing task_id; otherwise generate new one
    from .task_manager import create_task_state_from_spec, save_task_state, generate_task_id
    
    if resume_state:
        task_id_for_log = resume_state.task_id
        log_mode = 'a'  # Append mode for resumed tasks
        print(f"[Logging] Resuming task {task_id_for_log}, appending to existing log file")
    else:
        task_id_for_log = generate_task_id()
        log_mode = 'w'  # Write mode for new tasks
        print(f"[Logging] New task {task_id_for_log}, creating new log file")
    
    # Create log directory and file
    log_dir = Path(config.DATA_DIR) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{task_id_for_log}.log"
    
    # Open log file (append if resuming, write if new)
    log_file = open(log_file_path, log_mode, encoding='utf-8', buffering=1)  # Line buffering
    print(f"[Logging] Log file opened: {log_file_path} (mode: {log_mode})")
    
    # Create a tee that writes to terminal, log capture, log file, and frontend
    class TeeOutput:
        def __init__(self, *outputs):
            self.outputs = list(outputs)  # Convert to list for safe iteration
            self.buffer = ""  # Buffer for incomplete lines
            self.failed_outputs = set()  # Track failed outputs to skip them
        
        def write(self, data):
            # Write to all outputs (terminal, log capture, log file) with error protection
            for i, output in enumerate(self.outputs):
                # Skip outputs that have previously failed
                if i in self.failed_outputs:
                    continue
                    
                try:
                    output.write(data)
                except (OSError, IOError, ValueError) as e:
                    # Mark this output as failed and skip it in future writes
                    self.failed_outputs.add(i)
                    # Try to print to stderr if available
                    try:
                        import sys
                        sys.stderr.write(f"[TeeOutput] Output {i} failed: {e}\n")
                    except:
                        pass  # Even stderr might fail
                except Exception as e:
                    # Catch any other exception to prevent cascading failures
                    self.failed_outputs.add(i)
            
            # Send to frontend if log_callback is provided
            if log_callback and data:
                try:
                # Buffer the data and send complete lines
                    self.buffer += data
                    if '\n' in self.buffer:
                        lines = self.buffer.split('\n')
                        # Send all complete lines
                        for line in lines[:-1]:
                            if line.strip():  # Only send non-empty lines
                                try:
                                    log_callback(line)
                                except Exception:
                                    pass  # Don't break on callback errors
                        # Keep the incomplete line in buffer
                        self.buffer = lines[-1]
                except Exception:
                    pass  # Protect against buffer corruption
        
        def flush(self):
            for i, output in enumerate(self.outputs):
                if i in self.failed_outputs:
                    continue
                try:
                    output.flush()
                except (OSError, IOError, ValueError):
                    self.failed_outputs.add(i)
                except Exception:
                    self.failed_outputs.add(i)
            
            # Send any remaining buffer content
            if log_callback and self.buffer.strip():
                try:
                    log_callback(self.buffer)
                    self.buffer = ""
                except Exception:
                    pass
    
    sys.stdout = TeeOutput(original_stdout, log_capture, log_file)
    
    print("\n" + "="*80)
    print("🚀 AGENT_EXECUTE_SEARCH CALLED")
    print("="*80)
    print(f"Query Spec: top_n={spec.top_n}, keywords={spec.keywords}, venues={spec.venues}")
    print(f"Resume state: {resume_state is not None}")
    print("="*80 + "\n")
    
    def report_progress(event: str, pct: float, details: dict = None):
        """
        Helper to safely report progress with detailed information
        Args:
            event: Event name (e.g., "searching", "analyzing")
            pct: Progress percentage (0.0-1.0)
            details: Optional dict with detailed progress info:
                {
                    "round": "1/9",
                    "current_term": "text generation, diffusion model ACL 2025",
                    "papers_count": 37,
                    "papers_scores": {8: 5, 7: 8, 6: 4, ...},
                    "candidates": ["Xiaochuang Han", "Zhujin Gao"],
                    "processing_author": "Shansan Gong",
                    "search_query": "text generation, diffusion model",
                    "venues": ["ACL", "EMNLP"],
                    "years": [2025, 2024, 2023]
                }
        """
        if progress_callback:
            try:
                # Pack details into event string if provided
                if details:
                    import json
                    details_json = json.dumps(details, ensure_ascii=False)
                    event_with_details = f"{event}:::{details_json}"
                    progress_callback(event_with_details, pct)
                else:
                    progress_callback(event, pct)
            except Exception:
                pass
    
    # Initialize or resume state
    if resume_state:
        task_id = resume_state.task_id
        terms = resume_state.terms
        pos = resume_state.pos
        rounds_completed = resume_state.rounds_completed
        candidates_accum = resume_state.candidates_accum
        enhanced_profiles_accum = getattr(resume_state, 'enhanced_profiles_accum', {})  # With backward compatibility
        all_serp = resume_state.all_serp
        sources = resume_state.sources
        all_scored_papers = resume_state.all_scored_papers
        search_candidate_set = set(resume_state.search_candidate_set)
        selected_urls_set = resume_state.selected_urls_set
        selected_serp_url_set = resume_state.selected_serp_url_set
        # 恢复时检查是否还有未用完的terms
        has_more_papers = any(not t.get("exhausted", False) for t in terms)
        print(f"[Resume] Resumed search state: {sum(1 for t in terms if not t.get('exhausted', False))}/{len(terms)} terms still active")
        seed_candidates_input = []
        seed_serp = []
        seed_name_to_id = {}
    else:
        # Start new search
        report_progress("parsing", 0.05)
        terms = _plan_terms(spec)
        # Create new task state
        state = create_task_state_from_spec(spec, terms)
        # Ensure the task ID matches the one used for the log file so logs remain in a single file across resume / finish cycles
        state.task_id = task_id_for_log
        task_id = state.task_id
        pos = 0
        rounds_completed = 0
        candidates_accum = {}
        enhanced_profiles_accum = {}  # Initialize for new search
        all_serp = []
        sources = {}
        all_scored_papers = {}
        search_candidate_set = set()
        selected_urls_set = set()
        selected_serp_url_set = set()
        seed_candidates_input = seed_candidates or []
        seed_serp = []
        seed_name_to_id = {}
        if seed_candidates_input:
            seed_serp, seed_items, seed_name_to_id = _build_seed_inputs_from_leads(
                seed_candidates_input,
                api_key=api_key,
            )
            if seed_items:
                search_candidate_set.update(seed_items)
                print(f"[Seed Leads] Loaded {len(seed_items)} seed candidates")
            if seed_serp:
                print(f"[Seed Leads] Loaded {len(seed_serp)} seed papers")
            report_progress(
                "seed_leads",
                0.08,
                {
                    "lead_count": len(seed_candidates_input),
                    "seed_papers": len(seed_serp),
                    "seed_candidates": len(seed_items) if seed_items else 0,
                },
            )
    has_seed_input = bool(seed_candidates_input or seed_serp)
    chunk = config.SEARCH_BATCH_CHUNK
    rounds_this_run = 0  # Track rounds in current run
    has_more_papers = True  # 是否还有更多论文可获取
    MAX_TOTAL_ROUNDS = 50  # 安全上限，避免无限循环
    # ==================== Query Pool 控制变量 ====================
    # 计算当前应该使用的variant索引（考虑恢复任务的情况）
    current_term_idx = 0
    for idx, t in enumerate(terms):
        if not t.get("exhausted", False):
            current_term_idx = idx
            break
    consecutive_empty_batches = 0  # 当前query的连续空批次计数
    empty_batches_threshold = config.QUERY_POOL_CONFIG.get("empty_batches_threshold", 2)
    print(f"[Query Pool] 初始化: {len(terms)} 个variants, 空批次阈值={empty_batches_threshold}, 当前variant索引={current_term_idx}")
    
    # 规范化用户需求的类别（全局变量，用于统计有效候选人）
    requested_categories_for_count = []
    if spec.degree_levels:
        for d in spec.degree_levels:
            d_lower = d.lower().strip().replace("_", " ")
            if d_lower in ["phd", "phd student", "doctoral", "doctorate"]:
                requested_categories_for_count.append("PhD Student")
            elif d_lower in ["master", "master student", "msc", "ms"]:
                requested_categories_for_count.append("Master Student")
            elif d_lower in ["undergraduate", "undergraduate student", "undergrad", "bachelor", "bs", "ba"]:
                requested_categories_for_count.append("Undergraduate Student")
            elif d_lower in ["professor", "prof"]:
                requested_categories_for_count.append("Professor")
            elif d_lower in ["postdoc", "post-doc", "postdoctoral"]:
                requested_categories_for_count.append("Postdoc")
            elif d_lower in ["industrial researcher", "industry"]:
                requested_categories_for_count.append("Industrial Researcher")
            elif d_lower in ["institution researcher", "researcher"]:
                requested_categories_for_count.append("Institution Researcher")
    
    # 定义计算有效候选人数量的函数（避免代码重复）
    def count_matching_candidates():
        """统计符合学位要求且不是Unknown的候选人数量"""
        count = 0
        for c in candidates_accum.values():
            category = getattr(c, "candidate_category", "Unknown")
            if category != "Unknown":
                degree_match = (not requested_categories_for_count or category in requested_categories_for_count)
                if degree_match:
                    count += 1
        return count
    
    batch_matching_candidates_start = 0  # 每批次开始时的有效候选人数
    # ==============================================================
    # Main search loop - continue until all terms exhausted or max rounds reached
    # Build search query string for display
    if has_seed_input:
        search_query_display = "seed leads"
    elif spec.keywords:
        normalized_kw = [kw.replace("-", " ").replace("_", " ") for kw in spec.keywords]
        normalized_kw = [" ".join(kw.split()) for kw in normalized_kw]
        search_query_display = ", ".join(normalized_kw)
    else:
        search_query_display = "research papers"
    
    while has_more_papers and rounds_completed < MAX_TOTAL_ROUNDS:
        # ==================== Query Pool: Variant 切换检查 ====================
        # 检查是否所有variant都用完
        if current_term_idx >= len(terms):
            print(f"[Query Pool] 所有 {len(terms)} 个variants已exhausted，结束搜索")
            break
        # ==============================================================
        # Increment round counters at the START of each round
        rounds_completed += 1
        rounds_this_run += 1
        
        # ==================== Query Pool: 使用current_term_idx获取当前variant ====================
        # 注意：不再重新遍历terms，而是使用Query Pool管理的current_term_idx
        if current_term_idx >= len(terms):
            print(f"[Query Pool] current_term_idx={current_term_idx} 超出范围，结束搜索")
            break
        current_term = terms[current_term_idx]
        # ==============================================================
        remaining_terms = sum(1 for t in terms if not t.get("exhausted", False))
        current_offset = current_term.get("offset", 0) if current_term else 0
        current_keywords = current_term.get("keywords", []) if current_term else []
                
        print(f"[Round {rounds_completed}] Starting round {rounds_completed} (run: {rounds_this_run}/{max_rounds_per_run})")
        print(f"[Round {rounds_completed}] Current candidates: {len(candidates_accum)}, Target: {spec.top_n}")
        if has_seed_input:
            print(f"[Round {rounds_completed}] Seed leads provided; skipping paper search")
        else:
            print(f"[Round {rounds_completed}] Keywords: {current_keywords} (offset={current_offset})")
                
        # Report search progress with detailed info
        search_progress = 0.10 + min(0.20, rounds_completed / MAX_TOTAL_ROUNDS * 0.20)
        if has_seed_input:
            current_search_desc = "seed leads"
        else:
            current_search_desc = f"{', '.join(current_keywords)} (offset={current_offset})" if current_term else ""
                
        report_progress("searching", search_progress, {
            "round": f"{rounds_completed}",
            "current_term": current_search_desc,
            "search_query": search_query_display,
            "years": spec.years,
            "papers_count": len(all_scored_papers),
            "candidates": [c.name for c in candidates_accum.values()],
            "target_count": spec.top_n
        })
                
        # Call search backend - 每轮获取10篇
        print(f"[Round {rounds_completed}] Fetching 10 papers...")
        # ==================== Query Pool: 记录批次开始时的候选人数 ====================
        batch_matching_candidates_start = count_matching_candidates()
        # ==============================================================
        seed_serp_used = False
        if seed_serp:
            serp = seed_serp
            seed_serp = []
            seed_serp_used = True
            has_more_papers = False
            print(f"[Seed Leads] Using {len(serp)} seed papers for scoring")
        elif seed_candidates_input:
            serp = []
            seed_serp_used = True
            has_more_papers = False
            print("[Seed Leads] Using seed candidates without paper fetch")
        else:
            seen_titles = {
                p.title.lower().strip()
                for p in all_scored_papers.values()
                if getattr(p, "title", "").strip()
            }
            serp, has_more_papers = _run_search_terms(
                terms,
                k_per_query=10,
                years=spec.years,
                venues=spec.venues,
                seen_titles=seen_titles,
            )
        print(f"[Round {rounds_completed}] Got {len(serp)} papers, has_more={has_more_papers}")
        if seed_candidates_input and not seed_serp_used:
            has_more_papers = False
        
        # ==================== Query Pool: 检测当前variant是否exhausted ====================
        min_papers_threshold = config.QUERY_POOL_CONFIG.get("min_papers_per_batch", 5)
        if (not seed_serp_used) and (not seed_candidates_input) and (
            current_term.get("exhausted", False) or len(serp) < min_papers_threshold
        ):
            print(f"[Query Pool] Variant {current_term_idx} ({current_term['name']}) exhausted (获取={len(serp)}篇论文, 阈值={min_papers_threshold})")
            current_term["exhausted"] = True
            current_term_idx += 1
            consecutive_empty_batches = 0  # 重置空批次计数
            
            if current_term_idx < len(terms):
                next_variant = terms[current_term_idx]
                print(f"[Query Pool] 切换到Variant {current_term_idx}: {next_variant['keywords']} ({next_variant['name']})")
                print(f"[Query Pool] 立即开始下一轮搜索...")
                continue  # 立即跳到下一轮使用新variant获取论文
            else:
                print(f"[Query Pool] 所有variants已用完")
                has_more_papers = False
                break  # 结束搜索
        # ==============================================================
        if not serp:
            print(f"[Round {rounds_completed}] WARNING: No papers found in this round!")
        all_serp.extend(serp)
        selected_urls, selected_serp, batch_scored_papers = _select_urls(serp, spec, api_key)        
        # Save scored papers to our collection
        for paper in batch_scored_papers:
            if paper.url not in all_scored_papers:
                all_scored_papers[paper.url] = paper

        # 每轮搜索完成后进行论文去重
        print(f"[Round {rounds_completed}] Deduplicating papers after this round...")
        all_serp, all_scored_papers = deduplicate_papers(all_serp, all_scored_papers)
        print(f"[agent.execute_search] start remove duplicate serp items")
        # Deduplicate SERP items by URL
        new_serp_items = []
        for it in selected_serp:
            u = (it.get("url") or "").strip()
            if u and u not in selected_serp_url_set:
                selected_serp_url_set.add(u)
                new_serp_items.append(it)

        print(f"[agent.execute_search] start remove duplicate serp url")
        # Track URLs as well
        for u in selected_urls:
            if u not in selected_urls_set:
                selected_urls_set.add(u)

        # Report fetching progress (30% - 40%)
        report_progress("searching", 0.35)
        
        print(f"[agent.execute_search] new_serp_items: {len(new_serp_items)}")
        # Build user query for scoring (only keywords, no venue info)
        user_query = " ".join(spec.keywords) if spec.keywords else "research papers"
        
        # Score each paper using Agent-provided abstract
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        papers_to_score = []
        for serp_item in new_serp_items:
            url = serp_item.get("url", "")
            title = serp_item.get("title", "")
            
            # 获取 introduction 和 abstract
            introduction = serp_item.get("introduction") or ""  # Semantic Scholar TLDR，确保不是None
            abstract = serp_item.get("abstract", "") or serp_item.get("snippet", "")
            
            # Clean venue (arXiv-only papers will have empty venue after cleaning)
            venue_value = serp_item.get("venue", "")
            if venue_value and venue_value.strip():
                venue_value = clean_venue(venue_value)
            # Note: Empty venue means arXiv/preprint - frontend will handle display
            
            if config.VERBOSE:
                print(f"[papers_to_score] Title: {title}")
                print(f"  - Venue: '{venue_value}'")
                print(f"  - Introduction: {len(introduction) if introduction else 0} chars")
                print(f"  - Abstract: {len(abstract)} chars")
                print(f"  - pdf_url: {serp_item.get('pdf_url', '')}")
                print(f"  - authors: {serp_item.get('authors', [])}")

            
            papers_to_score.append({
                "url": url,  # 用于前端展示
                "title": title,
                "introduction": introduction,  # TLDR (会作为单独参数传给LLM)
                "abstract": abstract,  # 完整 abstract (会作为单独参数传给LLM)
                "venue": venue_value,
                "pdf_url": serp_item.get("pdf_url", ""),  # PDF URL for introduction extraction
                "authors": serp_item.get("authors", []),  # Author list from Semantic Scholar
                "serp_item": serp_item
            })
        # Score papers in parallel using existing score_paper_with_llm from search.py
        # Import dynamic_concurrency module
        get_optimal_workers = None
        get_candidate_workers = None
        get_extraction_workers = None
        get_llm_workers = None
        try:
            from .dynamic_concurrency import (
                get_optimal_workers, 
                get_candidate_workers, 
                get_extraction_workers, 
                get_llm_workers
            )
            print("[Agents] ✅ Dynamic concurrency module imported")
        except Exception as e:
            print(f"[Agents] ⚠️ Dynamic concurrency not available: {e}")
            # Provide fallback functions
            def get_optimal_workers(n): return min(n, 10)
            def get_candidate_workers(n): return min(n, 30)
            def get_extraction_workers(n): return min(n, 10)
            def get_llm_workers(n): return min(n, 5)
        max_workers = get_llm_workers(len(papers_to_score))
        print(f"[agent.execute_search] Scoring papers with {max_workers} parallel workers...")
        
        # Get LLM instance for scoring
        llm_instance = llm.get_llm("score", temperature=0.1, api_key=api_key)
        
        # Run scoring in parallel with timeout protection
        print(f"\n{'='*80}")
        print(f"[Paper Scoring] Starting LLM evaluation of {len(papers_to_score)} papers")
        print(f"{'='*80}\n")
        scored_results = []
        # Cache for PDF bytes - reuse for corresponding author detection
        paper_pdf_bytes_cache = {}  # paper_url -> pdf_bytes
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit scoring tasks using the existing score_paper_with_llm function
            future_to_paper = {}
            for idx, paper in enumerate(papers_to_score, 1):
                # 分别传递 introduction 和 abstract
                future = executor.submit(
                    search.score_paper_with_llm,
                    paper["title"],           # title
                    paper["abstract"],        # abstract
                    user_query,               # user_query
                    llm_instance,             # llm
                    introduction=paper.get("introduction", ""),  # introduction
                    pdf_url=paper.get("pdf_url", "")  # PDF URL for full PDF scoring
                )
                future_to_paper[future] = paper
            # Process results with timeout protection
            completed_count = 0
            timeout_count = 0
            error_count = 0
            try:
                # Use as_completed with overall timeout to prevent infinite waiting
                # Increased to 120s since we now only evaluate 5 Relevance dimensions (was 10 dimensions)
                for future in as_completed(future_to_paper, timeout=120):  # 120 seconds overall timeout
                    paper_info = future_to_paper[future]
                    try:
                        # Add per-task timeout (30 seconds per paper scoring for 5 Relevance dimensions)
                        result = future.result(timeout=30)  # Returns {"score": int, "explanation": str}
                        score = result.get("score", 4)  # Default to 4 if missing
                        # Ensure score is within valid range (1-8) for PaperWithScore schema
                        if score < 1:
                            score = 1
                        elif score > 8:
                            score = 8
                        paper_info["score"] = score
                        paper_info["explanation"] = result.get("explanation", "")
                        # Save multi-dimensional scoring fields to paper_info
                        paper_info["relevance_score"] = result.get("relevance_score", float(score))
                        paper_info["quality_score"] = result.get("quality_score", 0.0)
                        paper_info["relevance_dimensions"] = result.get("relevance_dimensions", {})
                        paper_info["quality_dimensions"] = result.get("quality_dimensions", {})
                        # Cache PDF bytes for corresponding author detection (avoid re-download)
                        pdf_bytes = result.get("pdf_bytes")
                        if pdf_bytes and paper_info.get("url"):
                            paper_pdf_bytes_cache[paper_info["url"]] = pdf_bytes
                            print(f"[PDF Cache] Cached {len(pdf_bytes)} bytes for: {paper_info['title']}")
                        completed_count += 1
                        
                        # Check if paper is low-score (< MIN_PAPER_SCORE)
                        # Low-score papers are kept in reference list but NOT used for candidate extraction
                        if paper_info["score"] < config.MIN_PAPER_SCORE:
                            print(f"[ScoreFilter]  Low-score paper (score: {paper_info['score']}) - kept in reference list, NOT for candidate extraction")
                        else:
                            print(f"[ScoreFilter]  High-score paper (score: {paper_info['score']}): {paper_info['title']}")
                        # Add ALL papers (both high and low score) to scored_results
                        scored_results.append(paper_info)
                        # Ensure venue is properly cleaned (arXiv-only papers should have empty venue)
                        venue_for_paper = paper_info.get("venue", "")
                        if venue_for_paper and venue_for_paper.strip():
                            venue_for_paper = clean_venue(venue_for_paper)
                        # After clean_venue, empty string means arXiv/preprint (let frontend handle display)
                        # Don't force "arXiv preprint" here - frontend will show appropriate label
                        paper = schemas.PaperWithScore(
                            url=paper_info["url"],
                            title=paper_info["title"],
                            abstract=paper_info["abstract"],              # Original abstract (~1500 chars)
                            introduction=paper_info.get("introduction") or "",  # Ensure not None, TLDR or empty string
                            venue=venue_for_paper,  # Conference/journal (always has value, includes year like "ACL 2025")
                            score=paper_info["score"],
                            explanation=paper_info.get("explanation", ""),  # LLM explanation for the score
                            authors=paper_info.get("authors", []),  # Author list for calculating paper score
                            associated_candidates=[],
                            # Multi-dimensional scoring fields
                            relevance_score=paper_info.get("relevance_score", 0.0),
                            quality_score=paper_info.get("quality_score", 0.0),
                            relevance_dimensions=paper_info.get("relevance_dimensions", {}),
                            quality_dimensions=paper_info.get("quality_dimensions", {}),
                            relevance_explanation=paper_info.get("explanation", "")
                        )
                        
                        # Add to all_scored_papers
                        if paper.url not in all_scored_papers:
                            all_scored_papers[paper.url] = paper
                            if config.VERBOSE:
                                print(f"[PaperWithScore] Added: {paper.title} | Venue: '{paper.venue}' | Score: {paper.score}")
                        
                    except TimeoutError:
                        timeout_count += 1
                        print(f"[agent.execute_search] ⏱Timeout scoring paper: {paper_info['title'][:80]}...")
                        print(f"  - Assigning default score of 4 (moderate relevance)")
                        # Assign default score for timed-out papers
                        paper_info["score"] = 4
                        paper_info["explanation"] = "LLM scoring timeout - assigned default score"
                        scored_results.append(paper_info)
                        
                        # Still create PaperWithScore for timed-out papers
                        timeout_venue = paper_info.get("venue", "")
                        if timeout_venue and timeout_venue.strip():
                            timeout_venue = clean_venue(timeout_venue)
                        paper = schemas.PaperWithScore(
                            url=paper_info["url"],
                            title=paper_info["title"],
                            abstract=paper_info["abstract"],
                            introduction=paper_info.get("introduction") or "",  # Ensure not None
                            venue=timeout_venue,
                            score=4,
                            explanation="LLM scoring timeout",
                            authors=paper_info.get("authors", []),  # Author list for calculating paper score
                            associated_candidates=[],
                            # Multi-dimensional scoring fields (timeout defaults)
                            relevance_score=4.0,
                            quality_score=0.0,
                            relevance_dimensions={},
                            quality_dimensions={},
                            relevance_explanation="LLM scoring timeout"
                        )
                        if paper.url not in all_scored_papers:
                            all_scored_papers[paper.url] = paper
                    
                    except Exception as e:
                        error_count += 1
                        print(f"[agent.execute_search] Error processing score result: {e}")
                        print(f"  - Paper: {paper_info['title']}")
                        import traceback
                        traceback.print_exc()
                        
            except TimeoutError:
                # Overall timeout - some tasks didn't complete
                print(f"\n[Scoring] Overall timeout reached (120s), processing remaining tasks...")
                # Track which futures were already processed in the as_completed loop
                processed_futures = {f for f in future_to_paper if f in {future for future in future_to_paper if future.done()}}
                # Only count futures that weren't processed in the as_completed loop
                already_processed_count = completed_count  # Save current count
                # Process any remaining futures that might have completed
                for future, paper_info in future_to_paper.items():
                    # Skip if already processed (result already in scored_results)
                    if any(p.get("url") == paper_info.get("url") for p in scored_results):
                        continue
                    if future.done():
                        try:
                            result = future.result(timeout=0)  # No wait, already done
                            score = result.get("score", 4)
                            # Ensure score is within valid range (1-8)
                            if score < 1:
                                score = 1
                            elif score > 8:
                                score = 8
                            paper_info["score"] = score
                            paper_info["explanation"] = result.get("explanation", "")
                            # Save multi-dimensional scoring fields to paper_info
                            paper_info["relevance_score"] = result.get("relevance_score", float(score))
                            paper_info["quality_score"] = result.get("quality_score", 0.0)
                            paper_info["relevance_dimensions"] = result.get("relevance_dimensions", {})
                            paper_info["quality_dimensions"] = result.get("quality_dimensions", {})
                            completed_count += 1
                            
                            if paper_info["score"] >= config.MIN_PAPER_SCORE:
                                scored_results.append(paper_info)
                                
                                remaining_venue = paper_info.get("venue", "")
                                if remaining_venue and remaining_venue.strip():
                                    remaining_venue = clean_venue(remaining_venue)
                                paper = schemas.PaperWithScore(
                                    url=paper_info["url"],
                                    title=paper_info["title"],
                                    abstract=paper_info["abstract"],
                                    introduction=paper_info.get("introduction") or "",  # Ensure not None
                                    venue=remaining_venue,
                                    score=paper_info["score"],
                                    explanation=paper_info.get("explanation", ""),
                                    authors=paper_info.get("authors", []),  # Author list for calculating paper score
                                    associated_candidates=[],
                                    # Multi-dimensional scoring fields
                                    relevance_score=paper_info.get("relevance_score", 0.0),
                                    quality_score=paper_info.get("quality_score", 0.0),
                                    relevance_dimensions=paper_info.get("relevance_dimensions", {}),
                                    quality_dimensions=paper_info.get("quality_dimensions", {}),
                                    relevance_explanation=paper_info.get("explanation", "")
                                )
                                if paper.url not in all_scored_papers:
                                    all_scored_papers[paper.url] = paper
                        except:
                            pass  # Ignore errors for already-completed tasks
                    else:
                        # Task didn't complete - cancel and assign default score
                        future.cancel()
                        timeout_count += 1
                        paper_info["score"] = 4
                        paper_info["explanation"] = "Overall timeout - assigned default score"
                        # NOTE: Default score 4 is used when scoring completely fails
                        # If Relevance dimensions were partially scored, we would use that score instead
                        scored_results.append(paper_info)
                        
                        overall_timeout_venue = paper_info.get("venue", "")
                        if overall_timeout_venue and overall_timeout_venue.strip():
                            overall_timeout_venue = clean_venue(overall_timeout_venue)
                        paper = schemas.PaperWithScore(
                            url=paper_info["url"],
                            title=paper_info["title"],
                            abstract=paper_info["abstract"],
                            introduction=paper_info.get("introduction") or "",  # Ensure not None
                            venue=overall_timeout_venue,
                            score=4,
                            explanation="Overall timeout (120s)",
                            authors=paper_info.get("authors", []),  # Author list for calculating paper score
                            associated_candidates=[],
                            # Multi-dimensional scoring fields (overall timeout defaults)
                            relevance_score=4.0,
                            quality_score=0.0,
                            relevance_dimensions={},
                            quality_dimensions={},
                            relevance_explanation="Overall timeout (120s)"
                        )
                        if paper.url not in all_scored_papers:
                            all_scored_papers[paper.url] = paper
            
            # Print scoring summary
            high_score_count = sum(1 for p in scored_results if p.get("score", 0) >= config.MIN_PAPER_SCORE)
            low_score_count = len(scored_results) - high_score_count
            print(f"\n[Scoring Summary] Processed {len(papers_to_score)} papers:")
            print(f"  - Successfully scored: {completed_count}")
            print(f"  - Timeouts: {timeout_count}")
            print(f"  - Errors: {error_count}")
            print(f"  - High-score papers (>= {config.MIN_PAPER_SCORE}): {high_score_count}")
            print(f"  - Low-score papers (< {config.MIN_PAPER_SCORE}): {low_score_count}")
        
        # Sort by score
        scored_results.sort(key=lambda x: x.get("score", 0), reverse=True)
        
        # Calculate filtering statistics
        total_scored = len(scored_results)
        high_score_papers_count = sum(1 for p in scored_results if p.get("score", 0) >= config.MIN_PAPER_SCORE)
        low_score_papers_count = total_scored - high_score_papers_count
        
        print(f"[agent.execute_search] Paper scoring completed:")
        print(f"  - Total papers scored: {total_scored}")
        print(f"  - High-score (>= {config.MIN_PAPER_SCORE}, for candidate extraction): {high_score_papers_count}")
        print(f"  - Low-score (< {config.MIN_PAPER_SCORE}, reference only): {low_score_papers_count}")
        
        print(f"[agent.execute_search] Score distribution (kept papers only):")
        
        # Calculate score distribution
        score_distribution = {}
        for score_val in range(8, 0, -1):
            count = sum(1 for p in all_scored_papers.values() if p.score == score_val)
            if count > 0:
                score_distribution[score_val] = count
                print(f"  Score {score_val}: {count} papers")

        # Report extraction progress with paper statistics
        report_progress("searching", 0.42, {
            "round": f"{rounds_completed}/{MAX_TOTAL_ROUNDS}",
            "search_query": search_query_display,
            "venues": spec.venues,
            "years": spec.years,
            "papers_count": len(all_scored_papers),
            "papers_scores": score_distribution,
            "candidates_count": len(candidates_accum),
            "target_count": spec.top_n,
            "current_action": "Extracting paper names"
        })
        high_score_papers = {url: paper for url, paper in all_scored_papers.items() if paper.score >= config.MIN_PAPER_SCORE}
        low_score_papers = {url: paper for url, paper in all_scored_papers.items() if paper.score < config.MIN_PAPER_SCORE}
        
        print(f"[agent.execute_search] Extracting ALL authors from HIGH-SCORE papers only (score >= {config.MIN_PAPER_SCORE})...")
        print(f"[agent.execute_search] Total papers: {len(all_scored_papers)}")
        print(f"  - High-score papers (>= {config.MIN_PAPER_SCORE}, for candidate extraction): {len(high_score_papers)}")
        print(f"  - Low-score papers (< {config.MIN_PAPER_SCORE}, kept in reference only): {len(low_score_papers)}")
        
        authors_found = 0
        filter_author_name_set = set()
        if search_candidate_set:
            filter_author_name_set = set(author_name for author_name, _, _, _ in search_candidate_set)
        seed_only_names = set()
        seed_author_ids = seed_name_to_id if isinstance(seed_name_to_id, dict) else {}
        if seed_candidates_input:
            seed_only_names = {
                author_name for author_name, _, _, _ in search_candidate_set if author_name
            }
            if seed_only_names:
                print(f"[Seed Leads] Restricting candidate extraction to {len(seed_only_names)} seed leads")
        
        # Import corresponding author detector for PDF-based detection
        try:
            from .corresponding_author_detector import detect_corresponding_authors, detect_cofirst_authors
            CORRESPOND_DETECTOR_AVAILABLE = True
        except ImportError:
            try:
                from corresponding_author_detector import detect_corresponding_authors, detect_cofirst_authors
                CORRESPOND_DETECTOR_AVAILABLE = True
            except ImportError:
                CORRESPOND_DETECTOR_AVAILABLE = False
                print("[Correspond Detector] Module not available, skipping corresponding author detection")
        
        # Cache for corresponding author indices per paper
        paper_corresponding_authors = {}  # paper_url -> list of 1-based author indices
        # Cache for co-first author indices per paper
        paper_cofirst_authors = {}  # paper_url -> list of 1-based author indices
        
        # 只从 high_score_papers 中提取作者
        for paper_url, scored_paper in high_score_papers.items():
            # 从原始 serp 中找到对应的完整数据（包含 authors）
            paper_authors = []
            paper_title = scored_paper.title
            
            # 在 selected_serp 中查找作者列表和author_ids
            paper_author_ids = []
            for serp_item in selected_serp:
                if serp_item.get("url") == paper_url and serp_item.get("authors"):
                    paper_authors = serp_item.get("authors", [])
                    # 提取 author_ids（如果可用）
                    paper_author_ids = serp_item.get("author_ids", [])
                    break
            
            # 如果没有作者列表，跳过这篇论文
            if not paper_authors:
                print(f"[Author Extraction] No authors found for paper: {paper_title} (skipping)")
                continue
            
            # ====== CORRESPONDING AUTHOR DETECTION (using cached PDF bytes) ======
            # Detect corresponding authors from PDF to give them higher weight
            corresponding_indices = []
            if CORRESPOND_DETECTOR_AVAILABLE and paper_url in paper_pdf_bytes_cache:
                try:
                    pdf_bytes = paper_pdf_bytes_cache[paper_url]
                    print(f"[Correspond Detector] Detecting corresponding authors for: {paper_title[:50]}...")
                    corresponding_indices, evidences = detect_corresponding_authors(
                        pdf_bytes=pdf_bytes,
                        authors=paper_authors,
                        api_key=api_key,
                        use_llm_fallback=True
                    )
                    if corresponding_indices:
                        paper_corresponding_authors[paper_url] = corresponding_indices
                        corresponding_names = [paper_authors[i-1] for i in corresponding_indices if i <= len(paper_authors)]
                        print(f"[Correspond Detector] Found corresponding author(s): {corresponding_names}")
                    else:
                        print(f"[Correspond Detector] No corresponding author detected for this paper")
                except Exception as e:
                    print(f"[Correspond Detector] Error detecting corresponding authors: {e}")
            elif CORRESPOND_DETECTOR_AVAILABLE and paper_url not in paper_pdf_bytes_cache:
                print(f"[Correspond Detector] No cached PDF for paper, skipping detection: {paper_title[:50]}...")
            # ====== END CORRESPONDING AUTHOR DETECTION ======
            
            # ====== CO-FIRST AUTHOR DETECTION (using cached PDF bytes) ======
            # Detect co-first authors from PDF to give them first-author weight
            cofirst_indices = []
            if CORRESPOND_DETECTOR_AVAILABLE and paper_url in paper_pdf_bytes_cache:
                try:
                    pdf_bytes = paper_pdf_bytes_cache[paper_url]
                    print(f"[CoFirst Detector] Detecting co-first authors for: {paper_title[:50]}...")
                    cofirst_indices, cofirst_evidences = detect_cofirst_authors(
                        pdf_bytes=pdf_bytes,
                        authors=paper_authors
                    )
                    if cofirst_indices:
                        paper_cofirst_authors[paper_url] = cofirst_indices
                        cofirst_names = [paper_authors[i-1] for i in cofirst_indices if i <= len(paper_authors)]
                        print(f"[CoFirst Detector] Found co-first author(s): {cofirst_names}")
                    else:
                        print(f"[CoFirst Detector] No co-first author detected for this paper")
                except Exception as e:
                    print(f"[CoFirst Detector] Error detecting co-first authors: {e}")
            # ====== END CO-FIRST AUTHOR DETECTION ======
            
            # Process ALL authors from this paper (1st, 2nd, 3rd, ..., last author)
            print(f"[Author Extraction] Processing {len(paper_authors)} authors from scored paper (score={scored_paper.score}): {paper_title}")
            if paper_author_ids:
                print(f"[Author Extraction] Found {len(paper_author_ids)} author IDs")
            
            for author_idx, author_name in enumerate(paper_authors, 1):
                author_name = author_name.strip()
                if not author_name:
                    continue
                if seed_only_names and author_name not in seed_only_names:
                    continue
                # Validate author name before adding
                is_valid, reason = is_valid_author_name(author_name)
                if not is_valid:
                    print(f"[NameFilter] Rejected author #{author_idx} '{author_name}': {reason}")
                    print(f"  - Paper: {paper_title}")
                    continue  # Skip this author
                
                # 尝试获取对应的 author_id（如果可用）
                author_id = None
                if paper_author_ids and author_idx - 1 < len(paper_author_ids):
                    author_id = paper_author_ids[author_idx - 1]  # author_idx是1-based，列表是0-based
                    # 处理空字符串的情况：如果 author_id 是空字符串，则设置为 None
                    if author_id and isinstance(author_id, str) and not author_id.strip():
                        author_id = None
                    if author_id:
                        print(f"[Author Extraction] Found author_id for {author_name}: {author_id}")
                    else:
                        print(f"[Author Extraction] No author_id for author #{author_idx} '{author_name}' (Semantic Scholar may not have ID for this author)")
                if author_name in seed_author_ids and seed_author_ids[author_name]:
                    author_id = seed_author_ids[author_name]
                
                # Add to candidate set if not already present
                if author_name not in filter_author_name_set:
                    filter_author_name_set.add(author_name)
                    search_candidate_set.add((author_name, author_id, paper_title, paper_url))
                    authors_found += 1
                    
                    # Determine author position label
                    if author_idx == 1:
                        position_label = "1st author"
                    elif author_idx == len(paper_authors):
                        position_label = f"last author ({author_idx}/{len(paper_authors)})"
                    else:
                        position_label = f"author #{author_idx}/{len(paper_authors)}"
                    
                    if config.VERBOSE:
                        print(f"[S2 Search] Valid {position_label}: {author_name} | Paper (score={scored_paper.score}): {paper_title[:50]}...")
        
        print(f"[S2 Search] Extracted {authors_found} valid authors from {len(high_score_papers)} high-score papers (score >= {config.MIN_PAPER_SCORE})")
        
        # Report analyzing progress
        report_progress("analyzing", 0.48)
        
        # Check if we have candidates to process
        if len(search_candidate_set) == 0:
            print(f"[agent.execute_search] No candidates in pool for this round (all filtered out or none found)")
            pass
        else:
            # ====== PREFETCH OPENREVIEW INFO FOR ALL HIGH-SCORE PAPERS ======
            # This is done ONCE at paper-level to avoid redundant API calls
            # Each author task will use cached paper info instead of hitting API
            try:
                from .openreview_client import get_openreview_client
                or_client = get_openreview_client()
                
                # Collect papers for prefetch
                papers_to_prefetch = []
                for paper_url, scored_paper in high_score_papers.items():
                    # Find authors from selected_serp
                    paper_authors = []
                    for serp_item in selected_serp:
                        if serp_item.get("url") == paper_url and serp_item.get("authors"):
                            paper_authors = serp_item.get("authors", [])
                            break
                    
                    if paper_authors:
                        papers_to_prefetch.append({
                            'title': scored_paper.title,
                            'authors': paper_authors
                        })
                
                if papers_to_prefetch:
                    print(f"[OpenReview Prefetch] Prefetching {len(papers_to_prefetch)} high-score papers...")
                    or_client.prefetch_papers(papers_to_prefetch)
                    print(f"[OpenReview Prefetch] Prefetch complete, cache will be used for candidate processing")
            except Exception as e:
                print(f"[OpenReview Prefetch] Warning: prefetch failed, will fallback to per-author queries: {e}")
            # ====== END PREFETCH ======
            
            # Prepare candidates item to search 
            items = list(search_candidate_set)
            print(f"[agent.execute_search] Candidate pool after name filtering: {len(items)} candidates")
            
            # Dynamic concurrency: intelligent resource management based on system state
            # Use dynamic concurrency manager to calculate optimal workers
            from .dynamic_concurrency import get_candidate_workers
            
            # Calculate optimal workers based on candidate count and required results
            required_candidates = spec.top_n or 5
            max_workers = get_candidate_workers(
                candidates=len(items), 
                required=required_candidates
            )
            
            # 检查可用内存来动态调整
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_safe_workers = int(available_memory_gb / 0.5)  # 每个候选人约0.5GB
            max_workers = min(max_workers, memory_safe_workers)  # 只受内存限制
            
            print(f"[Dynamic Concurrency] Optimal workers: {max_workers} for {len(items)} candidates")
            print(f"[Dynamic Concurrency] Required results: {required_candidates}, Processing: {min(len(items), max_workers * 2)}")

            # NOTE: Variable name 'first_name' is kept for compatibility with orchestrate_candidate_report,
            # but it now represents ANY author (1st, 2nd, 3rd, ..., last) from the paper
            def _submit_one(ex, first_name, first_id, paper_title, paper_url):
                # Get paper venue, score, explanation, and authors from all_scored_papers
                paper_venue = ""
                paper_score = 0
                paper_explanation = ""
                paper_authors = []
                paper_info = None
                if paper_url and paper_url in all_scored_papers:
                    sp = all_scored_papers[paper_url]
                    paper_venue = sp.venue
                    paper_score = sp.score
                    paper_explanation = sp.explanation
                    paper_authors = sp.authors  # 获取作者列表
                    paper_info = {
                        "title": sp.title,
                        "abstract": sp.abstract,
                        "authors": sp.authors,
                        "venue": sp.venue,
                        "url": sp.url,
                        "relevance_score": getattr(sp, 'relevance_score', float(paper_score)),
                        "quality_score": getattr(sp, 'quality_score', 0.0),
                        "relevance_dimensions": getattr(sp, 'relevance_dimensions', {}),
                        "quality_dimensions": getattr(sp, 'quality_dimensions', {}),
                        # Add corresponding author indices for weight calculation
                        "corresponding_author_indices": paper_corresponding_authors.get(paper_url, []),
                        # Add co-first author indices for weight calculation
                        "cofirst_author_indices": paper_cofirst_authors.get(paper_url, []),
                    }
                
                # Determine if this author is a corresponding author
                is_corresponding = False
                is_cofirst = False
                if paper_info:
                    # Find author index (1-based)
                    try:
                        author_index = paper_authors.index(first_name) + 1
                        
                        # Check corresponding author
                        if paper_info.get("corresponding_author_indices"):
                            is_corresponding = author_index in paper_info["corresponding_author_indices"]
                            if is_corresponding:
                                print(f"[Correspond Detector] Candidate '{first_name}' is a corresponding author")
                        
                        # Check co-first author
                        if paper_info.get("cofirst_author_indices"):
                            is_cofirst = author_index in paper_info["cofirst_author_indices"]
                            if is_cofirst:
                                print(f"[CoFirst Detector] Candidate '{first_name}' is a co-first author")
                    except ValueError:
                        pass  # Author not found in list
                
                # Build user query from spec
                user_query_parts = []
                if spec.keywords:
                    user_query_parts.append(" ".join(spec.keywords))
                user_query = " ".join(user_query_parts) or "research papers"
                
                # 传递已评分论文URL集合，避免候选人论文与Reference Papers重复
                # 但需要保留 trigger paper（发现候选人的那篇论文）
                exclude_urls = set(all_scored_papers.keys()) if all_scored_papers else set()
                # 从排除列表中移除 trigger paper，确保它出现在候选人的论文列表中
                if paper_url and paper_url in exclude_urls:
                    exclude_urls.remove(paper_url)
                    print(f"[Candidate Processing] Keeping trigger paper for {first_name}: {paper_title}")
                
                return ex.submit(
                    orchestrate_candidate_report,
                    first_author=first_name,
                    paper_title=paper_title,
                    paper_url=paper_url,
                    paper_venue=paper_venue,  # Pass venue information
                    paper_score=paper_score,  # Pass paper score
                    paper_explanation=paper_explanation,  # Pass paper explanation
                    paper_authors=paper_authors,  # Pass authors list for paper score calculation
                    aliases=[first_name],
                    author_id=first_id,
                    s2_author_id=first_id,  # S2 authorId 用于获取候选人论文
                    api_key=api_key,
                    user_query=user_query,  # NEW: Pass user query for evaluation
                    exclude_paper_urls=exclude_urls,  # NEW: Exclude papers already in Reference Papers (except trigger paper)
                    paper_info=paper_info,
                    is_corresponding_author=is_corresponding,  # Pass corresponding author flag
                    is_cofirst_author=is_cofirst,  # Pass co-first author flag
                )
            
            # Report candidate analysis start (50% - 80% will be dynamic)
            report_progress("analyzing", 0.50)
            
            print("="*50)
            print(f"[agent.execute_search] start submit search candidate task with: {len(items)} candidates")
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                # Submit initial window up to max_workers
                idx = 0
                futures = {}
                while idx < len(items) and len(futures) < max_workers:
                    first_name, first_id, paper_title, paper_url = items[idx]
                    fut = _submit_one(ex, first_name, first_id, paper_title, paper_url)
                    futures[fut] = (first_name, first_id, paper_title, paper_url)
                    idx += 1

                # Process as they complete; process all candidates in this round (no early stopping)
                processed_count = 0
                last_throttle_check = 0
                throttle_check_interval = 1  # Check every 3 candidates
                
                while futures:
                    # Resource monitoring: Check system resources periodically
                    if processed_count - last_throttle_check >= throttle_check_interval:
                        from .dynamic_concurrency import get_manager
                        manager = get_manager()
                        should_throttle, throttle_reason = manager.should_throttle()
                        
                        if should_throttle:
                            import time
                            safe_print(f"[Resource Monitor] {throttle_reason} - Pausing for 2 seconds...")
                            time.sleep(2)  # Brief pause to let system recover
                        
                        last_throttle_check = processed_count
                    
                    for fut in as_completed(list(futures.keys()), timeout=None):
                        first_name, first_id, paper_title, paper_url = futures.pop(fut)
                        processed_count += 1
                        
                        # remove this candidate from search_candidate_set
                        search_candidate_set.remove((first_name, first_id, paper_title, paper_url))
                        try:
                            result = fut.result()
                            # Handle None result (when orchestrate_candidate_report returns None)
                            if result is None:
                                safe_print(f"[orchestrate] {first_name}: orchestrate returned None")
                                profile, overview, eval_res, enhanced_profile = None, None, None, None
                            # Handle both old (3-tuple) and new (4-tuple) return values
                            elif len(result) == 4:
                                profile, overview, eval_res, enhanced_profile = result
                            else:
                                profile, overview, eval_res = result
                                enhanced_profile = None
                        except Exception as e:
                            safe_print(f"[orchestrate] {first_name} error: {str(e)[:100]}")
                            profile, overview, eval_res, enhanced_profile = None, None, None, None

                        if overview:
                            category = classify_candidate_overview(overview, api_key=api_key)
                            # 计算论文作者位置分数
                            overview.score_paper = calculate_paper_score(overview)
                            # 计算最终评分
                            overview.final_score = calculate_final_score(overview)
                            # 保存候选人到总列表（不论类别）
                            safe_print(f"[agent.execute_search] add candidate: {first_name} ({category}) - score_paper={overview.score_paper:.1f}, final_score={overview.final_score:.2f}")
                            candidates_accum[first_name] = overview
                            # 保存候选人类别（添加到overview对象）
                            overview.candidate_category = category
                            # 标记发现轮次（当前正在进行的轮次）
                            overview.discovered_in_round = rounds_completed + 1
                            # Also save enhanced_profile if available
                            if enhanced_profile:
                                # 使用 overview.name 作为key，确保前端能找到
                                # 如果 overview.name 和 first_name 不一致，两个都保存
                                enhanced_profiles_accum[overview.name] = enhanced_profile
                                safe_print(f"[agent.execute_search] Enhanced profile saved for: {overview.name}")
                                # 如果名字不一致，也用 first_name 作为key保存一份
                                if overview.name != first_name:
                                    enhanced_profiles_accum[first_name] = enhanced_profile
                                    safe_print(f"[agent.execute_search] Enhanced profile also saved with alias: {first_name}")
                            # Record paper-to-candidate mapping
                            paper_linked = False
                            
                            safe_print(f"[paper-candidate-link] Linking '{first_name}' to source paper...")
                            
                            # Direct URL match
                            if paper_url and paper_url in all_scored_papers:
                                if first_name not in all_scored_papers[paper_url].associated_candidates:
                                    all_scored_papers[paper_url].associated_candidates.append(first_name)
                                    safe_print(f"[paper-candidate-link] Linked '{first_name}' to paper (score={all_scored_papers[paper_url].score})")
                                    paper_linked = True
                            else:
                                safe_print(f"[paper-candidate-link] paper_url not in all_scored_papers")
                                safe_print(f"  - paper_url: {paper_url}")
                            
                            if not paper_linked:
                                safe_print(f"[paper-candidate-link] Failed to link '{first_name}'")
                            
                            # Report dynamic analyzing progress with detailed candidate info
                            # 计算符合要求的候选人数量（学位匹配）
                            def get_topic_match(cand):
                                radar = getattr(cand, "radar", {})
                                return radar.get("topic_match", 0) if isinstance(radar, dict) else 0
                            
                            requested_categories_for_progress = []
                            if spec.degree_levels:
                                for d in spec.degree_levels:
                                    d_lower = d.lower().strip().replace("_", " ")
                                    if d_lower in ["phd", "phd student", "doctoral", "doctorate"]:
                                        requested_categories_for_progress.append("PhD Student")
                                    elif d_lower in ["master", "master student", "msc", "ms"]:
                                        requested_categories_for_progress.append("Master Student")
                                    elif d_lower in ["undergraduate", "undergraduate student", "undergrad", "bachelor", "bs", "ba"]:
                                        requested_categories_for_progress.append("Undergraduate Student")
                                    elif d_lower in ["professor", "prof"]:
                                        requested_categories_for_progress.append("Professor")
                                    elif d_lower in ["postdoc", "post-doc", "postdoctoral"]:
                                        requested_categories_for_progress.append("Postdoc")
                                    elif d_lower in ["industrial researcher", "industry"]:
                                        requested_categories_for_progress.append("Industrial Researcher")
                                    elif d_lower in ["institution researcher", "researcher"]:
                                        requested_categories_for_progress.append("Institution Researcher")
                            
                            # 统计符合学位要求的候选人
                            matching_candidates_count = 0
                            
                            for c in candidates_accum.values():
                                category = getattr(c, "candidate_category", "Unknown")
                                # 只检查学位要求
                                degree_match = (not requested_categories_for_progress or category in requested_categories_for_progress)
                                if degree_match:
                                    matching_candidates_count += 1
                            target_for_progress = max(spec.top_n, matching_candidates_count + 1)
                            analyzing_progress = 0.50 + min(matching_candidates_count / target_for_progress, 1.0) * 0.25
                            
                            # Get top candidates for display (safely)
                            try:
                                top_candidates_list = sorted(
                                candidates_accum.values(),
                                key=lambda x: getattr(x, "total_score", 0),
                                reverse=True
                                )[:3]  # Top 3 for display
                            
                                # Extract affiliation from 9-field structure
                                top_candidates_info = []
                                for c in top_candidates_list:
                                    intro = getattr(c, "introduction", None)
                                    affiliation = ""
                                    if intro:
                                        position_str = getattr(intro, "position", "") or ""  # 重命名避免冲突
                                        aff = getattr(intro, "affiliation", "") or ""
                                        if position_str and aff:
                                            affiliation = f"{position_str} at {aff}"
                                        elif position_str:
                                            affiliation = position_str
                                        elif aff:
                                            affiliation = aff
                                    
                                    top_candidates_info.append({
                                    "name": c.name,
                                        "affiliation": affiliation,
                                    "score": getattr(c, "total_score", 0)
                                    })
                            
                                report_progress("analyzing", min(analyzing_progress, 0.75), {
                                    "round": f"{rounds_completed}/{MAX_TOTAL_ROUNDS}",
                                    "search_query": search_query_display,
                                    "venues": spec.venues,
                                    "years": spec.years,
                                    "papers_count": len(all_scored_papers),
                                    "papers_scores": score_distribution,
                                    "candidates_count": matching_candidates_count,  # 只显示符合要求的候选人数量
                                    "target_count": spec.top_n,
                                    "top_candidates": top_candidates_info,
                                    "processing_author": first_name,
                                    "current_action": f"Analyzing candidate profile"
                                })
                            except Exception as progress_err:
                                safe_print(f"[Progress] Error reporting progress: {str(progress_err)[:50]}")
                                # 计算符合要求的候选人
                                matching_count = len(candidates_accum)
                                safe_print(f"[agent.execute_search] matching candidates: {matching_count}/{len(candidates_accum)}")
                                safe_print(f"[agent.execute_search] target: {spec.top_n}")
                        else:
                            safe_print(f"[orchestrate] {first_name} -> overview is None")

                        # Rolling window: submit next task to keep max_workers saturated
                        if idx < len(items):
                            next_first_name, next_first_id, next_paper_title, next_paper_url = items[idx]
                            safe_print(f"[Rolling Window] Submitting next candidate: {next_first_name} ({idx+1}/{len(items)})")
                            nfut = _submit_one(ex, next_first_name, next_first_id, next_paper_title, next_paper_url)
                            futures[nfut] = (next_first_name, next_first_id, next_paper_title, next_paper_url)
                            idx += 1
                            safe_print(f"[Rolling Window] Active workers: {len(futures)}, Processed: {idx}/{len(items)})")

                # If enough gathered, best-effort cancel remaining
                for fut in list(futures.keys()):  # Create a copy of keys to avoid RuntimeError
                    first_name, first_id, paper_title, paper_url = futures.pop(fut)
                    search_candidate_set.remove((first_name, first_id, paper_title, paper_url))
                    fut.cancel()
            
            matching_final = len(candidates_accum)
            safe_print(f"[Search Complete] {matching_final} candidates found")
            safe_print("="*50)
        
        # ==================== Query Pool: 空批次检测与Variant切换 ====================
        # 统计批次结束时符合要求的候选人数量（排除Unknown）
        batch_matching_candidates_end = count_matching_candidates()
        new_matching_candidates_in_batch = batch_matching_candidates_end - batch_matching_candidates_start
        if new_matching_candidates_in_batch == 0:
            # 本批次空（没有符合要求的有效候选人）
            consecutive_empty_batches += 1
            print(f"[Query Pool] 空批次: 本批10篇论文未获取有效候选人（当前query连续{consecutive_empty_batches}批空）")
            print(f"[Query Pool] 本批次总候选人={len(candidates_accum)}, 符合要求={batch_matching_candidates_end}, 新增有效={new_matching_candidates_in_batch}")
            if consecutive_empty_batches >= empty_batches_threshold:
                # 连续多批空，切换到下一个variant
                print(f"[Query Pool] 当前query连续{consecutive_empty_batches}批空，切换到下一个variant")
                if current_term_idx < len(terms):
                    terms[current_term_idx]["exhausted"] = True
                current_term_idx += 1
                consecutive_empty_batches = 0  # 重置计数器
                
                if current_term_idx < len(terms):
                    next_variant = terms[current_term_idx]
                    print(f"[Query Pool] 切换到Variant {current_term_idx}: {next_variant['keywords']} ({next_variant['name']})")
                    # 立即跳到下一轮使用新variant
                    print(f"[Query Pool] 立即开始下一轮搜索...")
                    continue
                else:
                    print(f"[Query Pool] 所有variants已用完")
            else:
                # 第1批空，继续用同一query再来一批
                print(f"[Query Pool] 继续使用当前query再获取10篇论文...")
                continue
        else:
            # 有新的有效候选人，重置空批次计数
            consecutive_empty_batches = 0
            print(f"[Query Pool] 本批次新增{new_matching_candidates_in_batch}个有效候选人，累计{batch_matching_candidates_end}人")
        # ==============================================================
        # ========== 额外处理：对高分论文（7、8分）的一作进行二次检查 ==========
        print("\n" + "="*80)
        print("[First Author Recheck] Extracting first authors from high-score papers (≥6)...")
        print("="*80)
        
        # 1. 提取所有 score > 6 的论文的一作
        first_authors_to_recheck = []
        high_score_papers_for_first_author = {url: paper for url, paper in all_scored_papers.items() if paper.score > 6}
        
        for paper_url, paper in high_score_papers_for_first_author.items():
            # 从累积的所有 serp 中找到对应的作者列表
            paper_authors = []
            for serp_item in all_serp:
                if serp_item.get("url") == paper_url and serp_item.get("authors"):
                    paper_authors = serp_item.get("authors", [])
                    break
            
            # 只处理一作
            if paper_authors and len(paper_authors) > 0:
                first_author_name = paper_authors[0].strip()
                
                if not first_author_name:
                    continue
                
                # 验证一作名字是否合法
                is_valid, reason = is_valid_author_name(first_author_name)
                if not is_valid:
                    print(f"[FirstAuthorRecheck] Invalid first author name '{first_author_name}': {reason}")
                    continue
                
                # 检查这个一作是否已经在候选人列表中成功构建了档案
                if first_author_name in candidates_accum:
                    # 已经成功构建档案，跳过
                    print(f"[FirstAuthorRecheck] Skip (already processed): {first_author_name} | Paper (score={paper.score}): {paper.title}")
                    continue
                
                # 检查是否在 search_candidate_set 中（说明被提交过但可能失败了）
                already_in_set = any(name == first_author_name for name, _, _, _ in search_candidate_set)
                
                # 如果不在候选人列表中，或者在 search_candidate_set 中但没有成功构建档案，则重新处理
                if not already_in_set:
                    first_authors_to_recheck.append((first_author_name, None, paper.title, paper_url))
                    print(f"[FirstAuthorRecheck] ➕ Add to recheck: {first_author_name} | Paper (score={paper.score}): {paper.title}")
                else:
                    print(f"[FirstAuthorRecheck] In queue but not processed: {first_author_name} | Paper (score={paper.score}): {paper.title}")
        
        print(f"[FirstAuthorRecheck] Found {len(first_authors_to_recheck)} first authors to recheck")
        
        # 2. 如果有一作需要重新处理，则提交任务
        if first_authors_to_recheck:
            print(f"\n[FirstAuthorRecheck] Processing {len(first_authors_to_recheck)} first authors...")
            # 使用动态并发计算最优 worker 数量
            from .dynamic_concurrency import get_candidate_workers
            required_candidates = spec.top_n or 5
            max_workers_recheck = get_candidate_workers(
                candidates=len(first_authors_to_recheck), 
                required=required_candidates
            )
            
            # 安全上限
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            memory_safe_workers = int(available_memory_gb / 0.5)
            max_workers_recheck = min(max_workers_recheck, memory_safe_workers)
            
            print(f"[FirstAuthorRecheck] Using {max_workers_recheck} workers for {len(first_authors_to_recheck)} first authors")
            
            # 定义提交函数
            def _submit_first_author(ex, first_name, first_id, paper_title, paper_url):
                paper_venue = ""
                paper_score = 0
                paper_explanation = ""
                paper_authors = []
                paper_info = None
                if paper_url and paper_url in all_scored_papers:
                    sp = all_scored_papers[paper_url]
                    paper_venue = sp.venue
                    paper_score = sp.score
                    paper_explanation = sp.explanation
                    paper_authors = sp.authors  # 获取作者列表
                    paper_info = {
                        "title": sp.title,
                        "abstract": sp.abstract,
                        "authors": sp.authors,
                        "venue": sp.venue,
                        "url": sp.url,
                        "relevance_score": getattr(sp, 'relevance_score', float(paper_score)),
                        "quality_score": getattr(sp, 'quality_score', 0.0),
                        "relevance_dimensions": getattr(sp, 'relevance_dimensions', {}),
                        "quality_dimensions": getattr(sp, 'quality_dimensions', {}),
                    }
                
                user_query_parts = []
                if spec.keywords:
                    user_query_parts.append(" ".join(spec.keywords))
                user_query = " ".join(user_query_parts) or "research papers"
                
                # 传递已评分论文URL集合，避免候选人论文与Reference Papers重复
                exclude_urls = set(all_scored_papers.keys()) if all_scored_papers else set()
                
                return ex.submit(
                    orchestrate_candidate_report,
                    first_author=first_name,
                    paper_title=paper_title,
                    paper_url=paper_url,
                    paper_venue=paper_venue,
                    paper_score=paper_score,
                    paper_explanation=paper_explanation,
                    paper_authors=paper_authors,  # Pass authors list for paper score calculation
                    aliases=[first_name],
                    author_id=first_id,
                    api_key=api_key,
                    user_query=user_query,
                    exclude_paper_urls=exclude_urls,  # NEW: Exclude papers already in Reference Papers
                    paper_info=paper_info,
                )
            
            # 使用线程池处理一作
            with ThreadPoolExecutor(max_workers=max_workers_recheck) as ex:
                idx = 0
                futures = {}
                
                # 提交初始批次
                while idx < len(first_authors_to_recheck) and len(futures) < max_workers_recheck:
                    first_name, first_id, paper_title, paper_url = first_authors_to_recheck[idx]
                    fut = _submit_first_author(ex, first_name, first_id, paper_title, paper_url)
                    futures[fut] = (first_name, first_id, paper_title, paper_url)
                    idx += 1
                
                # 处理结果
                processed_count = 0
                while futures:
                    for fut in as_completed(list(futures.keys()), timeout=None):
                        first_name, first_id, paper_title, paper_url = futures.pop(fut)
                        processed_count += 1
                        
                        try:
                            result = fut.result()
                            # Handle None result (when orchestrate_candidate_report returns None)
                            if result is None:
                                safe_print(f"[FirstAuthorRecheck] {first_name}: orchestrate returned None")
                                profile, overview, eval_res, enhanced_profile = None, None, None, None
                            elif len(result) == 4:
                                profile, overview, eval_res, enhanced_profile = result
                            else:
                                profile, overview, eval_res = result
                                enhanced_profile = None
                        except Exception as e:
                            safe_print(f"[FirstAuthorRecheck] ❌ {first_name} error: {str(e)[:100]}")
                            profile, overview, eval_res, enhanced_profile = None, None, None, None
                        
                        if overview:
                            category = classify_candidate_overview(overview, api_key=api_key)
                            overview.score_paper = calculate_paper_score(overview)
                            overview.final_score = calculate_final_score(overview)
                            
                            safe_print(f"[FirstAuthorRecheck] Added first author: {first_name} ({category}) - score_paper={overview.score_paper:.1f}, final_score={overview.final_score:.2f}")
                            candidates_accum[first_name] = overview
                            overview.candidate_category = category
                            overview.discovered_in_round = rounds_completed + 1
                            
                            # 保存增强档案
                            if enhanced_profile:
                                enhanced_profiles_accum[overview.name] = enhanced_profile
                                safe_print(f"[FirstAuthorRecheck] Enhanced profile saved for: {overview.name}")
                                
                                if overview.name != first_name:
                                    enhanced_profiles_accum[first_name] = enhanced_profile
                            # 关联论文和候选人
                            if paper_url and paper_url in all_scored_papers:
                                if first_name not in all_scored_papers[paper_url].associated_candidates:
                                    all_scored_papers[paper_url].associated_candidates.append(first_name)
                                    safe_print(f"[FirstAuthorRecheck] Linked '{first_name}' to paper (score={all_scored_papers[paper_url].score})")
                        else:
                            safe_print(f"[FirstAuthorRecheck] {first_name} -> overview is None")
                        
                        # 提交下一个任务
                        if idx < len(first_authors_to_recheck):
                            next_first_name, next_first_id, next_paper_title, next_paper_url = first_authors_to_recheck[idx]
                            nfut = _submit_first_author(ex, next_first_name, next_first_id, next_paper_title, next_paper_url)
                            futures[nfut] = (next_first_name, next_first_id, next_paper_title, next_paper_url)
                            idx += 1
                
                # 取消剩余任务
                for fut in list(futures.keys()):
                    first_name, first_id, paper_title, paper_url = futures.pop(fut)
                    fut.cancel()
            
            matching_after_recheck = len(candidates_accum)
            print(f"\n[FirstAuthorRecheck] Complete: {matching_after_recheck} candidates found")
            print(f"[FirstAuthorRecheck] Added {processed_count} new first author candidates")
            print("="*80 + "\n")
        else:
            print("[FirstAuthorRecheck] No first authors need rechecking")
            print("="*80 + "\n")
        
        # ========== RELATED PAPERS EXPANSION (Second Round) ==========
        # NOTE: Disabled - no longer needed for talent discovery workflow
        # This step was used to expand candidates by querying their related papers,
        # but it adds unnecessary API calls and processing time.
        # _expand_candidates_from_related_papers_once(
        #     candidates_accum=candidates_accum,
        #     enhanced_profiles_accum=enhanced_profiles_accum,
        #     all_scored_papers=all_scored_papers,
        #     spec=spec,
        #     api_key=api_key,
        # )
        # ========== End of round - check if we need to pause ==========
        matching_for_round = len(candidates_accum)
        # 检查当前会议的状态
        current_venue_exhausted = current_term.get("exhausted", False) if current_term else True
        if current_venue_exhausted and has_more_papers:
            # 当前会议用完，还有其他会议
            next_term = None
            for t in terms:
                if not t.get("exhausted", False):
                    next_term = t
                    break
            next_venue = next_term.get("venue", "") if next_term else "N/A"
            print(f"\n[Round {rounds_completed}] Complete: '{current_venue}' exhausted, moving to '{next_venue}'")
        else:
            print(f"\n[Round {rounds_completed}] Complete: {matching_for_round} candidates found")
        print(f"  - Target: {spec.top_n} | Rounds: {rounds_this_run}/{max_rounds_per_run}")
        # 检查搜索词是否已经用完（基于 has_more_papers 而不是 pos）
        active_terms_remaining = sum(1 for t in terms if not t.get("exhausted", False))
        search_terms_exhausted = not has_more_papers or rounds_completed >= MAX_TOTAL_ROUNDS
        if search_terms_exhausted:
            if rounds_completed >= MAX_TOTAL_ROUNDS:
                print(f"[Search Complete] Reached max rounds limit ({MAX_TOTAL_ROUNDS})")
            else:
                print(f"[Search Complete] All venues exhausted ({active_terms_remaining} remaining)")
            print(f"[Search Complete] Proceeding to final ranking and results...")
        
        # Check if we need to pause and ask user for decision
        # 如果搜索词已用完，不暂停，直接完成
        if rounds_this_run >= max_rounds_per_run and not search_terms_exhausted:
            # Now 1 round = 1 cycle (changed from 2 rounds = 1 cycle)
            current_cycle = rounds_completed
            print(f"[Pause] Cycle information:")
            print(f"  - Completed cycle number: {current_cycle}")
            print(f"  - Completed rounds: {rounds_completed}")
            print(f"  - Accumulated candidates: {len(candidates_accum)}")
            if candidates_accum:
                print(f"  - Candidate list: {list(candidates_accum.keys())}...")
            print(f"  - Search progress: {pos}/{len(terms)} search terms")
            
            # Capture current logs before pausing
            current_logs = log_capture.getvalue()
            
            # Get existing logs from resume state if any (with backward compatibility)
            existing_logs = ""
            if resume_state and hasattr(resume_state, 'accumulated_logs'):
                existing_logs = resume_state.accumulated_logs
            
            # Combine logs
            combined_logs = existing_logs + current_logs
            
            print(f"\n[Pause] Saving task state with logs:")
            print(f"  - existing_logs: {len(existing_logs):,} chars")
            print(f"  - current_logs: {len(current_logs):,} chars")
            print(f"  - combined_logs: {len(combined_logs):,} chars")
            print(f"  - combined_logs preview: {combined_logs[:200]}...\n")
            
            # Save task state
            task_state = schemas.SearchTaskState(
                task_id=task_id,
                spec=spec,
                pos=pos,
                terms=terms,
                rounds_completed=rounds_completed,
                candidates_accum=candidates_accum,
                enhanced_profiles_accum=enhanced_profiles_accum,  # Save enhanced profiles
                all_serp=all_serp,
                sources=sources,
                all_scored_papers=all_scored_papers,
                search_candidate_set=list(search_candidate_set),
                selected_urls_set=selected_urls_set,
                selected_serp_url_set=selected_serp_url_set,
                accumulated_logs=combined_logs,  # Save accumulated logs
            )
            save_task_state(task_state)
            print(f"[Pause] Task state saved to disk\n")
            
            # Return partial results and ask user
            current_cycle = rounds_completed
            
            # 计算符合要求的候选人数量（学位匹配）
            matching_degree_count = 0
            requested_categories = []
            
            # 规范化用户需求的类别
            if spec.degree_levels:
                for d in spec.degree_levels:
                    d_lower = d.lower().strip().replace("_", " ")
                    if d_lower in ["phd", "phd student", "doctoral", "doctorate"]:
                        requested_categories.append("PhD Student")
                    elif d_lower in ["master", "master student", "msc", "ms"]:
                        requested_categories.append("Master Student")
                    elif d_lower in ["undergraduate", "undergraduate student", "undergrad", "bachelor", "bs", "ba"]:
                        requested_categories.append("Undergraduate Student")
                    elif d_lower in ["professor", "prof"]:
                        requested_categories.append("Professor")
                    elif d_lower in ["postdoc", "post-doc", "postdoctoral"]:
                        requested_categories.append("Postdoc")
                    elif d_lower in ["industrial researcher", "industry"]:
                        requested_categories.append("Industrial Researcher")
                    elif d_lower in ["institution researcher", "researcher"]:
                        requested_categories.append("Institution Researcher")
            
            # 统计符合要求的候选人（学位匹配）
            # 同时排除 Unknown 类别的候选人
            non_unknown_count = 0  # 非 Unknown 类别的候选人总数
            for cand in candidates_accum.values():
                category = getattr(cand, "candidate_category", "Unknown")
                # 跳过 Unknown 类别
                if category == "Unknown":
                    continue
                non_unknown_count += 1
                degree_match = (not requested_categories or category in requested_categories)
                if degree_match:
                    matching_degree_count += 1
            print(f"[Pause] Prepare to return PartialSearchResults:")
            print(f"  - task_id: {task_id}")
            print(f"  - rounds_completed: {rounds_completed}")
            print(f"  - total_candidates_found: {non_unknown_count} (排除 Unknown 后的候选人)")
            print(f"  - matching_degree_candidates: {matching_degree_count} (学位匹配)")
            print(f"  - requested_categories: {requested_categories or 'ALL'}")
            print(f"  - breakdown:")
            print(f"    • Total (excluding Unknown): {non_unknown_count}")
            print(f"    • Matching degree requirement: {matching_degree_count}")
            print(f"    • Will appear in Recommended: ~{min(matching_degree_count, spec.top_n)}")
            print(f"  - accumulated_logs size: {len(combined_logs)} chars")
            print(f"  - log_file_path: {log_file_path}")
            print(f"  - active_terms_remaining: {active_terms_remaining}")
            print(f"  ⚠️ Attention: candidates are not sorted yet (sorting happens when user clicks 'Finish')")
            
            partial_result = schemas.PartialSearchResults(
                task_id=task_id,
                need_user_decision=True,
                rounds_completed=rounds_completed,
                total_candidates_found=non_unknown_count,  # 使用排除 Unknown 后的数量
                matching_degree_candidates=matching_degree_count,  # 🆕 符合学位要求的数量
                current_candidates=list(candidates_accum.values()),  # Return raw list without sorting
                message=f"Completed search cycle {current_cycle}, found {non_unknown_count} candidates so far.",
                log_file_path=str(log_file_path)
            )
            
            # Restore stdout and close log file (will reopen in append mode if resumed)
            sys.stdout = original_stdout
            # 安全关闭日志文件（处理网络文件系统断开）
            try:
                log_file.close()
                print(f"[Logging] Log file closed (paused): {log_file_path}")
            except (OSError, IOError) as e:
                print(f"[Logging] Warning: Could not close log file (filesystem issue): {e}")
                # 文件系统问题，但不影响搜索结果
            return partial_result
    # ========== RANKING & SCORING (Step 3) - Outside search loop ==========
    # Report ranking progress (75% - 85%)
    report_progress("ranking", 0.78)
    # Capture accumulated logs so far
    accumulated_logs_so_far = log_capture.getvalue()
    # Get existing logs from resume state if any (with backward compatibility)
    existing_logs = ""
    if resume_state and hasattr(resume_state, 'accumulated_logs'):
        existing_logs = resume_state.accumulated_logs
    # Combine logs
    total_accumulated_logs = existing_logs + accumulated_logs_so_far
    # Create task state for final processing
    final_task_state = schemas.SearchTaskState(
        task_id=task_id,
        spec=spec,
        pos=pos,
        terms=terms,
        rounds_completed=rounds_completed,
        candidates_accum=candidates_accum,
        enhanced_profiles_accum=enhanced_profiles_accum,  # Pass enhanced profiles
        all_serp=all_serp,
        sources=sources,
        all_scored_papers=all_scored_papers,
        search_candidate_set=list(search_candidate_set),
        selected_urls_set=selected_urls_set,
        selected_serp_url_set=selected_serp_url_set,
        accumulated_logs=total_accumulated_logs,  # Pass accumulated logs to finish function
    )
    print(f"\n[agent.execute_search] Preparing to call agent_finish_search")
    print(f"  - Passing accumulated_logs: {len(total_accumulated_logs):,} chars")
    print(f"  - From existing logs: {len(existing_logs):,} chars")
    print(f"  - From current session: {len(accumulated_logs_so_far):,} chars\n")
    # Report ranking progress (85% - 90%)
    report_progress("ranking", 0.88)
    # Report finalizing progress (90% - 95%)
    report_progress("finalizing", 0.92)
    # Use unified finish function to rank and prepare results
    # Pass capture_logs=False so output is captured by our log_capture
    results = agent_finish_search(final_task_state, api_key, capture_logs=False)
    # Report completion (100%)
    report_progress("done", 1.0)
    # Restore stdout and capture logs
    sys.stdout = original_stdout
    captured_logs = log_capture.getvalue()
    
    # 安全关闭日志文件
    try:
        log_file.close()
        print(f"[Logging] Log file closed (completed): {log_file_path}")
    except (OSError, IOError) as e:
        print(f"[Logging] Warning: Could not close log file (filesystem issue): {e}")
        # 继续执行，不影响返回结果
    # Set the complete logs and log file path in results
    # This includes all stages: paper search, candidate discovery, and final ranking
    results.search_logs = captured_logs
    results.log_file_path = str(log_file_path)
    print(f"[agent.execute_search] Search completed")
    print(f"  - Complete search logs captured: {len(captured_logs):,} chars")
    return results

def agent_finish_search(task_state: schemas.SearchTaskState, api_key: str = None, capture_logs: bool = True) -> schemas.SearchResults:
    """Finish a paused search task by ranking and returning current candidates.
    This is called when the user chooses to stop searching and view current results.
    Can be called either from agent_execute_search or directly from frontend.
    Args:
        task_state: SearchTaskState containing accumulated candidates and papers
        api_key: API key for LLM calls (if needed)
        capture_logs: If True, capture logs in this function. If False, output directly (for external capture)
    Returns:
        SearchResults with final ranked candidates
    """
    # Capture logs for this finish operation (if needed)
    import io
    import sys
    from pathlib import Path
    
    # Open log file in append mode (task already has a log file from previous rounds)
    log_dir = Path(config.DATA_DIR) / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file_path = log_dir / f"{task_state.task_id}.log"
    
    if capture_logs:
        # Direct call from frontend - set up full logging infrastructure
        log_capture = io.StringIO()
        original_stdout = sys.stdout
        log_file = open(log_file_path, 'a', encoding='utf-8', buffering=1)  # Append mode
        
        # Create TeeOutput to write to all destinations
        class TeeOutput:
            def __init__(self, *outputs):
                self.outputs = outputs
            
            def write(self, data):
                for output in self.outputs:
                    output.write(data)
            
            def flush(self):
                for output in self.outputs:
                    output.flush()
        
        sys.stdout = TeeOutput(original_stdout, log_capture, log_file)
        print(f"[Logging] Appending to log file (finish): {log_file_path}")
    else:
        # Called from agent_execute_search - log file already open, just use it
        log_capture = None
        original_stdout = None
        log_file = None  # Don't open a new file, the caller already has it open
    
    # Check if we have accumulated logs from previous rounds
    print(f"\n[agent.finish_search] Checking for accumulated logs:")
    print(f"  - Has attribute 'accumulated_logs': {hasattr(task_state, 'accumulated_logs')}")
    if hasattr(task_state, 'accumulated_logs'):
        print(f"  - accumulated_logs length: {len(task_state.accumulated_logs)} chars")
        print(f"  - accumulated_logs preview: {task_state.accumulated_logs[:200]}..." if task_state.accumulated_logs else "  - accumulated_logs is empty")
    else:
        print(f"  - ⚠️ task_state does NOT have accumulated_logs attribute!")
    
    has_previous_logs = (hasattr(task_state, 'accumulated_logs') and 
                         task_state.accumulated_logs and 
                         len(task_state.accumulated_logs) > 100)  # Has meaningful content
    
    print(f"  - Result: has_previous_logs = {has_previous_logs}\n")
    print(f"[agent.finish_search] Input data:")
    print(f"  - Task ID: {task_state.task_id}")
    print(f"  - Total candidate number: {len(task_state.candidates_accum)}")
    if task_state.candidates_accum:
        print(f"  - Candidate list: {list(task_state.candidates_accum.keys())[:10]}...")
    print(f"  - Total paper number: {len(task_state.all_scored_papers)}")
    print(f"  - Target Top N: {task_state.spec.top_n}")    
    # Get spec from task state
    spec = task_state.spec
    candidates_accum = task_state.candidates_accum
    all_scored_papers = task_state.all_scored_papers
    
    # 辅助函数：获取 Topic Match 分数
    def get_topic_match_score(cand):
        """获取候选人的 Topic Match 分数"""
        radar = getattr(cand, "radar", {})
        if isinstance(radar, dict):
            return radar.get("topic_match", 0)
        return 0
    
    # Build reverse mapping: candidate_name -> highest paper score (for display only)
    candidate_to_best_paper_score = {}  # {name: paper_score (7-8)}
    
    for paper_url, paper in all_scored_papers.items():
        for cand_name in paper.associated_candidates:
            # Track the highest paper score (for display information only, not for sorting)
            current_paper_score = candidate_to_best_paper_score.get(cand_name, 0)
            candidate_to_best_paper_score[cand_name] = max(current_paper_score, paper.score)
    
    # ==========================================================================
    # 按类别分组候选人
    # ==========================================================================
    print(f"\n{'='*80}")
    print(f"[agent.finish_search] Classifying candidates by category")
    print(f"{'='*80}\n")
    candidates_before_filter = len(candidates_accum)
    # Save for later use in SearchResults
    total_candidates_before_filter = candidates_before_filter
    
    print(f"[Filtering Results]:")
    print(f"  - Total candidates: {candidates_before_filter}")
    
    # 获取 enhanced_profiles 用于检查角色信息
    enhanced_profiles = getattr(task_state, 'enhanced_profiles_accum', {})
    
    # 过滤掉没有成功获取role的候选人（position 或 affiliation 为空）
    candidates_without_role = []
    for cand_name, cand in list(candidates_accum.items()):
        enhanced_profile = enhanced_profiles.get(cand_name)
        # 如果没有 enhanced_profile 或 introduction，或者 position/affiliation 为空，则过滤掉
        if not enhanced_profile or not enhanced_profile.introduction:
            candidates_without_role.append(cand_name)
            candidates_accum.pop(cand_name, None)
        else:
            position = enhanced_profile.introduction.position or ""
            affiliation = enhanced_profile.introduction.affiliation or ""
            # 如果 position 或 affiliation 为空，说明没有成功获取role
            if not position.strip() or not affiliation.strip():
                candidates_without_role.append(cand_name)
                candidates_accum.pop(cand_name, None)
    
    if candidates_without_role:
        print(f"\n[Filter] Removing {len(candidates_without_role)} candidates without role (position or affiliation is empty):")
        for name in candidates_without_role:
            print(f"  - {name}")
        candidates_before_filter = len(candidates_accum)
        print(f"[Filter] Updated total candidates: {candidates_before_filter + len(candidates_without_role)} → {candidates_before_filter} (removed {len(candidates_without_role)} without role)")
    
    # 按类别分组所有候选人
    candidates_by_category = {
        "PhD Student": [],
        "Master Student": [],
        "Undergraduate Student": [],
        "Professor": [],
        "Postdoc": [],
        "Industrial Researcher": [],
        "Institution Researcher": [],
        "Unknown": []
    }
    
    for cand in candidates_accum.values():
        category = getattr(cand, "candidate_category", "Unknown")
        # 如果类别不在预定义列表中，归类为 "Unknown"
        if category not in candidates_by_category:
            print(f"[Warning] Unknown category '{category}' for candidate {cand.name}, classifying as 'Unknown'")
            category = "Unknown"
        candidates_by_category[category].append(cand)
    
    # 过滤掉所有"Unknown"类别的候选人
    unknown_count = len(candidates_by_category.get("Unknown", []))
    if unknown_count > 0:
        print(f"\n[Filter] Removing {unknown_count} candidates with 'Unknown' category")
        unknown_names = [cand.name for cand in candidates_by_category.get("Unknown", [])]
        for name in unknown_names:
            print(f"  - {name}")
        # 从candidates_by_category中移除Unknown类别
        candidates_by_category.pop("Unknown", None)
        
        # 从candidates_accum中移除这些候选人
        for name in unknown_names:
            candidates_accum.pop(name, None)
    
    # 更新total_candidates_before_filter，排除Unknown候选人
    total_candidates_before_filter = candidates_before_filter - unknown_count
    if unknown_count > 0:
        print(f"[Filter] Updated total_candidates_found: {candidates_before_filter} → {total_candidates_before_filter} (removed {unknown_count} Unknown)")
    
    # 统计各类别人数
    print(f"\n[Category Stats] Distribution (after Unknown removal):")
    total_classified = 0
    for category, cands in candidates_by_category.items():
        if cands:
            print(f"  - {category}: {len(cands)} candidates")
            total_classified += len(cands)
    print(f"  - Total: {total_classified} candidates\n")
    
    # 第二步：对每个类别内部按 Final Score 分数排序
    for category in candidates_by_category:
        # 确保所有候选人都有 final_score
        for cand in candidates_by_category[category]:
            if not hasattr(cand, 'final_score') or cand.final_score == 0.0:
                cand.score_paper = calculate_paper_score(cand)
                cand.final_score = calculate_final_score(cand)
        candidates_by_category[category] = sorted(
            candidates_by_category[category],
            key=lambda x: -getattr(x, "final_score", 0.0)  # Final Score 降序
        )
    # ==========================================================================
    # 根据用户需求的degree_levels，分recommended和others
    # ==========================================================================
    # 规范化用户需求的类别
    requested_categories = []
    if spec.degree_levels:
        for d in spec.degree_levels:
            d_lower = d.lower().strip().replace("_", " ")
            if d_lower in ["phd", "phd student", "doctoral", "doctorate"]:
                requested_categories.append("PhD Student")
            elif d_lower in ["master", "master student", "msc", "ms"]:
                requested_categories.append("Master Student")
            elif d_lower in ["undergraduate", "undergraduate student", "undergrad", "bachelor", "bs", "ba"]:
                requested_categories.append("Undergraduate Student")
            elif d_lower in ["professor", "prof"]:
                requested_categories.append("Professor")
            elif d_lower in ["postdoc", "post-doc", "postdoctoral"]:
                requested_categories.append("Postdoc")
            elif d_lower in ["industrial researcher", "industry"]:
                requested_categories.append("Industrial Researcher")
            elif d_lower in ["institution researcher", "researcher"]:
                requested_categories.append("Institution Researcher")
    print(f"[Recommended Selection] User requested categories: {requested_categories or 'ALL'}")
    print(f"[Recommended Selection] Target count (top_n): {spec.top_n}")
    # 收集requested类别的候选人
    recommended_pool = []
    for category in requested_categories:
        recommended_pool.extend(candidates_by_category.get(category, []))
    # 如果没有指定类别，使用所有候选人
    if not requested_categories:
        recommended_pool = list(candidates_accum.values())
    
    # 确保所有候选人都有 final_score（为旧数据提供兜底）
    for cand in recommended_pool:
        if not hasattr(cand, 'final_score') or cand.final_score == 0.0:
            cand.score_paper = calculate_paper_score(cand)
            cand.final_score = calculate_final_score(cand)
    # ==========================================================================
    # 多样性重排：避免同一篇论文的多个作者霸榜
    # 参数设置：
    #   N = spec.top_n (用户指定的目标数量)
    #   K = 5 (每人用于 rerank 的核心论文数)
    #   M = 1 (最强多样性：一篇论文最多支撑 1 个 Top 候选)
    #   L = 2 (indep_sum 取独立 Top-2)
    #   show_L = 2 (每个候选展示 2 篇触发论文)
    # ==========================================================================
    print(f"\n[Diversity Rerank] Applying diversity-aware ranking to avoid paper monopolization...") 
    recommended = diversity_rerank_topN(
        candidates=recommended_pool,
        all_scored_papers=all_scored_papers,
        N=spec.top_n,
        M=1  # 最强多样性
    )
    # 计算并打印多样性效果指标
    diversity_metrics = compute_diversity_metrics(recommended)
    print(f"\n[Diversity Metrics] Effectiveness indicators:")
    print(f"  - Unique paper coverage: {diversity_metrics['unique_paper_coverage']:.2%}")
    print(f"  - Max paper frequency: {diversity_metrics['max_paper_frequency']} (M=1 expects ~1)")
    print(f"  - Duplicate rate: {diversity_metrics['duplicate_rate']:.2%} (lower is better)")
    print(f"  - Total triggers: {diversity_metrics['total_triggers']}, Unique: {diversity_metrics['unique_triggers']}")
    recommended_names = set(c.name for c in recommended)
    print(f"\n[Recommended Candidates] Selected {len(recommended)}/{spec.top_n} from requested categories:")
    print(f"  (Sorted by Diversity-Adjusted Score, descending)")
    for i, cand in enumerate(recommended, 1):
        category = getattr(cand, "candidate_category", "Unknown")
        total_score = getattr(cand, "total_score", 0)
        topic_match = get_topic_match_score(cand)
        final_score = getattr(cand, "final_score", 0.0)
        diversity_score = getattr(cand, "diversity_adjusted_score", 0.0)
        indep_ratio = getattr(cand, "independent_ratio", 1.0)
        trigger_papers = getattr(cand, "trigger_papers", [])
        trigger_display = trigger_papers[0].title[:40] + "..." if trigger_papers and len(trigger_papers[0].title) > 40 else (trigger_papers[0].title if trigger_papers else "N/A")
        print(f"  #{i} {cand.name} ({category}) - DivScore: {diversity_score:.2f} (Ratio: {indep_ratio:.2f}) | Triggers: {trigger_display}")
    # ==========================================================================
    # Others: 剩余候选人，按类别分组
    # ==========================================================================
    others_by_category = {
        "PhD Student": [],
        "Master Student": [],
        "Undergraduate Student": [],
        "Professor": [],
        "Postdoc": [],
        "Industrial Researcher": [],
        "Institution Researcher": [],
        "Unknown": []
    }
    
    # 将所有未在recommended中的候选人放入others（按原类别）
    # 排除"Unknown"类别（已在前面被过滤）
    for category, cands in candidates_by_category.items():
        if category == "Unknown":  # 跳过Unknown类别
            continue
        for cand in cands:
            if cand.name not in recommended_names:
                others_by_category[category].append(cand)
    
    # 从others_by_category中移除Unknown键（确保不会出现在统计中）
    others_by_category.pop("Unknown", None)
    
    print(f"\n[Other Candidates] Distribution by category:")
    for category, cands in others_by_category.items():
        if cands:
            print(f"  - {category}: {len(cands)} candidates")
    
    # 旧逻辑兼容：additional仍然是除recommended之外的所有候选人（扁平列表）
    # 也按 Final Score 分数排序
    additional = []
    for cands in others_by_category.values():
        additional.extend(cands)
    # 确保 additional 列表中所有候选人都有 final_score（为旧数据提供兜底）
    for cand in additional:
        if not hasattr(cand, 'final_score') or cand.final_score == 0.0:
            # 重新计算缺失的分数
            cand.score_paper = calculate_paper_score(cand)
            cand.final_score = calculate_final_score(cand)
    # 确保 additional 列表也按 Final Score 排序
    additional = sorted(additional, key=lambda x: -getattr(x, "final_score", 0.0))
    # 为 additional 候选人也设置 trigger_papers（使用空的 used_count，不应用多样性惩罚）
    for cand in additional:
        cand_papers = _get_candidate_top_papers(cand, all_scored_papers, K=None)  # 不限制数量
        if cand_papers and not getattr(cand, 'trigger_papers', []):
            # 为 additional 候选人生成 trigger_papers（不应用多样性限制，只展示 1 篇）
            empty_used_count: Dict[str, int] = {}
            cand.trigger_papers = _pick_trigger_papers(
                cand, cand_papers, empty_used_count, all_scored_papers, M=999, show_L=1
            )
            # 同时更新旧的单个 trigger_paper_* 字段（向后兼容）
            if cand.trigger_papers:
                first_trigger = cand.trigger_papers[0]
                cand.trigger_paper_title = first_trigger.title
                cand.trigger_paper_url = first_trigger.paper_url
                cand.trigger_paper_venue = first_trigger.venue
                cand.trigger_paper_score = first_trigger.score
                cand.trigger_paper_position = first_trigger.author_position
                cand.trigger_paper_position_index = first_trigger.author_position_index
                cand.trigger_paper_total_authors = first_trigger.total_authors
    
    # Keep ALL papers in reference list (including low-score papers without candidates)
    # Low-score papers (< 6) won't have candidates but should still appear in reference list
    papers_to_keep = list(all_scored_papers.values())
    
    papers_with_candidates = [p for p in papers_to_keep if p.associated_candidates]
    papers_without_candidates = [p for p in papers_to_keep if not p.associated_candidates]
    
    print(f"[agent.finish_search] 📄 Reference papers (ALL papers included):")
    print(f"  - Total papers: {len(papers_to_keep)}")
    print(f"  - Papers with candidates (> 6 score): {len(papers_with_candidates)}")
    print(f"  - Papers without candidates (<= 6 score): {len(papers_without_candidates)}")
    print(f"  - Total candidates: {len(candidates_accum)}")
    
    # Debug: Show some examples
    if papers_with_candidates:
        print(f"  - Examples of HIGH-SCORE papers (with candidates):")
        for paper in list(papers_with_candidates)[:3]:
            print(f"    * Score {paper.score}: '{paper.title}' -> {paper.associated_candidates}")
    
    if papers_without_candidates:
        print(f"  - Examples of LOW-SCORE papers (without candidates, kept in reference):")
        for paper in list(papers_without_candidates)[:3]:
            print(f"    * Score {paper.score}: '{paper.title}'")
    
    # Sort reference papers by score (descending)
    # ALL papers are included (both high-score with candidates and low-score without candidates)
    reference_papers = sorted(
        papers_to_keep,
        key=lambda x: -x.score  # Sort by score descending (8>7>6>5>4>...)
    )
    print(f"[agent.finish_search] Reference Papers ranking:")
    print(f"  - Sorting: By score descending (8 > 7 > 6 > 5 > ...)")
    print(f"  - ALL papers included (both with and without candidates)")
    
    high_score_papers_count = sum(1 for p in reference_papers if p.score >= config.MIN_PAPER_SCORE)
    low_score_papers_count = sum(1 for p in reference_papers if p.score < config.MIN_PAPER_SCORE)
    print(f"  - High-score papers (> 6, with candidates): {high_score_papers_count}")
    print(f"  - Low-score papers (<= 6, reference only): {low_score_papers_count}")
    print(f"  - Total: {len(reference_papers)} papers")
    # Show score distribution (all scores)
    if reference_papers:
        print(f"\n Score distribution:")
        score_counts = {}
        for paper in reference_papers:
            score_counts[paper.score] = score_counts.get(paper.score, 0) + 1
        
        for score_val in sorted(score_counts.keys(), reverse=True):
            count = score_counts[score_val]
            if score_val >= config.MIN_PAPER_SCORE:
                label = "High-score (with candidates)"
            else:
                label = "Low-score (reference only)"
            print(f"    Score {score_val} ({label}): {count} papers")
    
    # Build user query for metadata
    # Build user query for display (only keywords, no venue info)
    user_query = " ".join(spec.keywords) if spec.keywords else "research papers"
    
    # Create and return SearchResults
    enhanced_profiles = getattr(task_state, 'enhanced_profiles_accum', {})
    if enhanced_profiles:
        displayed_names = {c.name for c in recommended} | {c.name for c in additional}
        enhanced_profiles = {
            name: profile for name, profile in enhanced_profiles.items()
            if name in displayed_names
            and profile
            and profile.introduction
            and profile.introduction.position
            and profile.introduction.position.strip()
            and profile.introduction.affiliation
            and profile.introduction.affiliation.strip()
        }
    
    # 准备返回结果（包含分类信息）
    # matching_candidates_count = 符合目标学位的候选人数 (recommended)
    # total_candidates_found = 所有候选人数 (recommended + additional)
    matching_candidates_count = len(recommended)  # 符合目标学位的候选人
    total_candidates_after_filtering = len(recommended) + len(additional)  # 实际显示的所有候选人数（排除Unknown）
    
    results = schemas.SearchResults(
        recommended_candidates=recommended,
        additional_candidates=additional,
        candidates_by_category=others_by_category,
        reference_papers=reference_papers,
        total_candidates_found=total_candidates_after_filtering,
        matching_candidates_count=matching_candidates_count,
        search_query=user_query,
        enhanced_profiles=enhanced_profiles
    )
    
    print(f"\n[agent.finish_search] 📊 Result Summary:")
    print(f"  - Recommended (matching target degree): {len(recommended)} candidates")
    print(f"  - Additional (other degrees): {len(additional)} candidates")
    print(f"  - Total displayed: {len(recommended) + len(additional)} candidates")
    print(f"  - Total candidates found: {total_candidates_after_filtering} (same as total displayed)")
    print(f"  - Matching candidates count: {matching_candidates_count} (same as recommended)")
    print(f"  - By category (for UI display):")
    for category, cands in others_by_category.items():
        if cands:
            print(f"    • {category}: {len(cands)}")
    print(f"  - Reference papers: {len(reference_papers)}")
    
    if enhanced_profiles:
        print(f"\n[agent.finish_search] ✅ Returning {len(enhanced_profiles)} enhanced profiles (8 fields each)")
    
    print(f"\n[agent.finish_search] ✅ Sorting completed, returning results (sorted by Topic Match):")
    print(f"  - Recommended candidates: {len(recommended)}")
    print(f"  - Additional candidates: {len(additional)}")
    print(f"  - Reference papers: {len(reference_papers)} (filtered from {len(all_scored_papers)} total)")
    print(f"  - Total candidates: {len(candidates_accum)}")
    print("🏁"*50 + "\n")
    
    # Restore stdout and capture logs (if we were capturing)
    if capture_logs:
        sys.stdout = original_stdout
        finish_logs = log_capture.getvalue()
        
        # ✅ 安全关闭日志文件
        if log_file:
            try:
                log_file.close()
                print(f"[Logging] Log file closed (finish): {log_file_path}")
            except (OSError, IOError) as e:
                print(f"[Logging] Warning: Could not close log file (filesystem issue): {e}")
        
        # 这样 task_state.accumulated_logs 就包含了完整的搜索历史（从开始到 finish）
        if hasattr(task_state, 'accumulated_logs'):
            # 如果已有 accumulated_logs，追加 finish_logs
            task_state.accumulated_logs += finish_logs
        else:
            # 如果没有（旧版本或首次运行），直接设置
            task_state.accumulated_logs = finish_logs
        
        print(f"[Logging] Updated task_state.accumulated_logs: {len(task_state.accumulated_logs):,} chars total")
        
        # ✅ 保存更新后的 task_state 到磁盘
        from .task_manager import save_task_state
        try:
            save_task_state(task_state)
            print(f"[Logging] ✅ Task state saved to disk with complete logs")
        except Exception as e:
            print(f"[Logging] ⚠️ Failed to save task state: {e}")
        
        # Set the complete logs and log file path in results
        # This includes accumulated_logs (if any) + final ranking stage
        results.search_logs = finish_logs
        results.log_file_path = str(log_file_path)
        
        print(f"[agent.finish_search] Generated logs: {len(finish_logs):,} chars")
        if has_previous_logs:
            print(f"  ✅ Includes logs from previous search rounds")
        else:
            print(f"  ℹ️ Only final ranking stage (no previous logs found)")
    else:
        pass
    
    if config.ENABLE_DATABASE_STORAGE:
        try:
            from .database import get_db
            print(f"\n[Database] 💾 Saving search results to database...")
            
            db = get_db(
                db_type=config.DB_TYPE,
                db_path=config.SQLITE_DB_PATH if config.DB_TYPE == "sqlite" else None
            )
            
            db.save_search_task(task_state, status="completed")
            
            saved_count = 0
            for name, candidate in candidates_accum.items():
                try:
                    db.save_candidate(candidate, task_id=task_state.task_id)
                    saved_count += 1
                except Exception as e:
                    print(f"[Database] Error saving {name}: {e}")
            
            stats = db.get_statistics()
            print(f"[Database] ✅ Saved {saved_count} candidates")
            print(f"[Database] 📊 Total in DB: {stats['total_candidates']} candidates")
        
        except Exception as e:
            print(f"[Database] ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    return results

def agent_adjust_search_parameters(
    current_spec: Dict[str, Any],
    user_input: str,
    chat_history: List[Dict[str, str]] = None,
) -> QuerySpec | None:
    """
    Use LLM to adjust search parameters based on a new user instruction and recent chat history.
    Args:
        current_spec: Existing query spec as dict
        user_input: New user adjustment instruction
        chat_history: Optional recent chat messages as a list of {"role": "user"|"assistant", "content": str}
    Returns:
        QuerySpec: Updated query spec
    """
    try:
        llm_instance = llm.get_llm("adjust", temperature=0.2)
        # Clamp history to last 10 messages
        chat_history = chat_history or []
        recent_msgs = chat_history[-10:]
        # Prepare a compact history string
        def fmt(m):
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            return f"{role.upper()}: {content}"
        history_text = "\n".join(fmt(m) for m in recent_msgs)

        conf_list = ", ".join(config.DEFAULT_CONFERENCES.keys())
        prompt = (
            "SYSTEM ROLE: You update a structured search spec for a recruitment/talent search engine.\n"
            "You MUST output STRICT JSON matching the QuerySpec schema (no extra keys, no comments, no prose).\n"
            "\n"
            "OBJECTIVE\n"
            "Given (1) the current JSON spec, (2) a new user instruction, and (3) recent conversation snippets,\n"
            "produce an UPDATED QuerySpec where ONLY the fields explicitly changed by the new instruction are modified.\n"
            "All other fields must remain identical to the current spec.\n"
            "\n"
            "SCHEMA (QuerySpec):\n"
            "{\n"
            '  "top_n": int,\n'
            '  "years": int[],\n'
            '  "venues": string[],\n'
            '  "keywords": string[],\n'
            '  "must_be_current_student": bool,\n'
            '  "degree_levels": string[],\n'
            '  "author_priority": string[],\n'
            '  "extra_constraints": string[]\n'
            "}\n"
            "\n"
            "PRECEDENCE & EDIT RULES (VERY IMPORTANT)\n"
            "1) New User Instruction > recent conversation context > Current Spec.\n"
            '2) If the instruction includes ADDITIVE language (e.g., "also include X", "add Y"), then UNION with the existing list.\n'
            '3) If it includes EXCLUSIVE language (e.g., "only X", "strictly X", "limit to X"), then REPLACE the list with exactly those items.\n'
            '4) If it includes NEGATION (e.g., "exclude X", "not X", "no X"), REMOVE those items from the list if present.\n'
            "5) If a field is NOT mentioned, DO NOT change it.\n"
            '6) For numbers in "top_n", parse the most salient integer in the instruction ("~", "around", "at least" → just use the integer).\n'
            '7) Years: extract explicit 4-digit years if present; if phrases like "last 2 years" appear, map to [CURRENT_YEAR, CURRENT_YEAR-1].\n'
            "   If years are not mentioned, DO NOT change them.\n"
            '8) must_be_current_student: set True if the instruction says "current/enrolled/active students only"; set False if it says\n'
            '   "alumni allowed", "graduates ok", "postdocs ok", or similar. If not mentioned, DO NOT change it.\n'
            "\n"
            "NORMALIZATION RULES\n"
            "- Venues canonicalization (case-insensitive → canonical UPPER names). Known venues include: {conf_list}.\n"
            '  Synonyms map to canonical: {"NIPS":"NeurIPS", "The Web Conference":"WWW", "WWW":"WWW"}.\n'
            "  Deduplicate while preserving the user-specified order.\n"
            '- Degree levels canonical set: ["PhD", "MSc", "Master", "Graduate", "Undergraduate", "Bachelor", "Postdoc"].\n'
            '  Map synonyms: {"MS":"MSc", "M.S.":"MSc", "MEng":"Master", "BSc":"Bachelor", "BS":"Bachelor"}.\n'
            '- Author priority canonical set: ["first", "last", "corresponding"]. Map synonyms: {"lead":"first", "senior":"last"}.\n'
            "- Keywords: trim whitespace, lower-case, deduplicate.\n"
            "\n"
            "CONSTRAINTS\n"
            "- ABSOLUTELY DO NOT invent defaults or remove existing values unless the instruction explicitly requests it or implies it\n"
            "  via exclusive/negation phrasing. If ambiguous, prefer ADD (union) rather than replace.\n"
            "- Output MUST be valid JSON for QuerySpec, including ALL fields. No nulls. No extra commentary.\n"
            "\n"
            "INPUTS\n"
            "=== Conversation (most recent last) ===\n"
            f"{history_text}\n"
            "\n"
            "=== Current Spec (JSON) ===\n"
            f"{json.dumps(current_spec, ensure_ascii=False)}\n"
            "\n"
            "=== New User Instruction ===\n"
            f"{user_input}\n"
            "\n"
            "OUTPUT FORMAT\n"
            "Return ONLY the final JSON for QuerySpec. No markdown, no code fences, no explanations.\n"
        )

        # Use schemas.QuerySpec for structured parsing
        adjusted = llm.safe_structured(llm_instance, prompt, schemas.QuerySpec)

        # If venues empty, apply sensible default: 3个核心会议 + 随机2个顶会
        if adjusted.venues == []:
            import random
            core_venues = config.CORE_CONFERENCES.copy()
            random_venues = random.sample(config.TOP_TIER_CONFERENCES, min(2, len(config.TOP_TIER_CONFERENCES)))
            adjusted.venues = core_venues + random_venues
            print(f"[Adjust Params] No venues after adjustment, using default:")
            print(f"  Core: {core_venues}")
            print(f"  Random: {random_venues}")
            print(f"  Final: {adjusted.venues}")

        return adjusted
    except Exception as e:
        return None

def agent_classify_user_adjustment(
    current_spec: Dict[str, Any],
    user_input: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> Dict[str, Any]:
    """
    Classify whether the user input is requesting a change to search parameters.

    Returns a dict: {"is_adjustment": bool, "help_instruction": str}
    If is_adjustment is False, help_instruction contains a concise instruction for the user.
    """
    try:
        llm_instance = llm.get_llm("classify", temperature=0.0)

        chat_history = chat_history or []
        recent_msgs = chat_history[-10:]
        def fmt(m):
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            return f"{role.upper()}: {content}"
        history_text = "\n".join(fmt(m) for m in recent_msgs)
        prompt = (
            "SYSTEM: You classify if the user's new message is asking to ADJUST the search parameters (like top_n, years, venues, keywords, must_be_current_student, degree_levels, author_priority, extra_constraints) or not.\n"
            'Return STRICT JSON with keys: {"is_adjustment": bool, "help_instruction": string}.\n'
            "If the message is NOT an adjustment (e.g., greeting, question, generic feedback without parameters), set is_adjustment=false and provide a short, concrete help_instruction that tells the user exactly how to specify changes (one sentence).\n"
            "If it IS an adjustment, set is_adjustment=true and help_instruction=''.\n\n"
            "=== Conversation (most recent last) ===\n"
            f"{history_text}\n\n"
            "=== Current Spec (JSON) ===\n"
            f"{json.dumps(current_spec, ensure_ascii=False)}\n\n"
            "=== New User Message ===\n"
            f"{user_input}\n\n"
            "OUTPUT: JSON only."
        )
        result = llm.safe_structured(
            llm_instance, prompt, schemas.UserAdjustmentClassification
        )
        # Expecting UserAdjustmentClassification object
        if isinstance(result, schemas.UserAdjustmentClassification):
            is_adjustment = result.is_adjustment
            help_instruction = result.help_instruction if not is_adjustment else ""
            return {
                "is_adjustment": is_adjustment,
                "help_instruction": help_instruction,
            }
        # Fallback
        is_adjustment = any(
            k in user_input.lower()
            for k in [
                "add",
                "remove",
                "only",
                "exclude",
                "top",
                "year",
                "venue",
                "keyword",
                "student",
                "degree",
                "author",
            ]
        )
        return {
            "is_adjustment": is_adjustment,
            "help_instruction": (
                "Please specify what to change, e.g., 'top_n 15' or 'add keywords: computer vision'."
                if not is_adjustment
                else ""
            ),
        }
    except Exception as e:
        print(f"LLM分类失败, 返回原始参数: {e.message}")
        return {
            "is_adjustment": False,
            "help_instruction": "Tell me what to change, e.g., 'set top_n to 15' or 'add venues: ACL, EMNLP'.",
        }

def agent_validate_search_request(
    user_input: str, chat_history: List[Dict[str, str]] | None = None
) -> Dict[str, Any]:
    """
    Validate whether the user input contains sufficient information for a meaningful search.

    Returns a dict: {"is_valid_search": bool, "search_terms_found": List[str],
                     "missing_elements": List[str], "suggestion": str}
    """
    try:
        llm_instance = llm.get_llm("validate", temperature=0.0)

        chat_history = chat_history or []
        recent_msgs = chat_history[-5:]  # Only look at recent context

        def fmt(m):
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            return f"{role.upper()}: {content}"

        history_text = "\n".join(fmt(m) for m in recent_msgs)
        prompt = (
            "SYSTEM: You validate if the user's message contains enough information for a meaningful talent search.\n"
            "IMPORTANT: Respond ONLY in English. All output must be in English.\n\n"
            "A valid search should contain at least one of:\n"
            "- Research areas/topics (e.g., 'machine learning', 'computer vision', 'NLP')\n"
            "- Academic terms (e.g., 'PhD students', 'researchers', 'graduate students')\n"
            "- Specific skills or expertise\n"
            "- Academic venues or conferences\n"
            "- Career stage or degree level\n\n"
            'Return STRICT JSON with keys: {"is_valid_search": bool, "search_terms_found": [str], "missing_elements": [str], "suggestion": str}.\n'
            "If valid, set is_valid_search=true and list the search terms found.\n"
            "If invalid, set is_valid_search=false, list what's missing, and provide a helpful suggestion.\n\n"
            "=== Recent Conversation ===\n"
            f"{history_text}\n\n"
            "=== User Message to Validate ===\n"
            f"{user_input}\n\n"
            "OUTPUT: JSON only."
        )
        result = llm.safe_structured(
            llm_instance, prompt, schemas.SearchValidationResult
        )
        if isinstance(result, schemas.SearchValidationResult):
            return {
                "is_valid_search": result.is_valid_search,
                "search_terms_found": result.search_terms_found,
                "missing_elements": result.missing_elements,
                "suggestion": result.suggestion,
            }
        # Fallback
        has_research_terms = any(
            term in user_input.lower()
            for term in [
                "machine learning",
                "ai",
                "artificial intelligence",
                "computer vision",
                "nlp",
                "natural language",
                "deep learning",
                "neural",
                "research",
                "phd",
                "student",
                "graduate",
                "academic",
                "paper",
                "publication",
                "conference",
                "journal",
            ]
        )
        return {
            "is_valid_search": has_research_terms,
            "search_terms_found": (
                [] if not has_research_terms else ["research terms detected"]
            ),
            "missing_elements": (
                [] if has_research_terms else ["research areas", "academic context"]
            ),
            "suggestion": (
                "Please specify what type of talent you're looking for, including research areas or academic background."
                if not has_research_terms
                else ""
            ),
        }
    except Exception as e:
        print(f"Search validation failed: {e}")
        return {
            "is_valid_search": False,
            "search_terms_found": [],
            "missing_elements": ["clear search criteria"],
            "suggestion": "Please describe what type of talent you're looking for, including research areas or academic background.",
        }
def agent_diff_search_parameters(
    current_spec: Dict[str, Any],
    user_input: str,
    chat_history: List[Dict[str, str]] | None = None,
) -> QuerySpecDiff | None:
    """
    Use LLM to produce a PARTIAL update (diff) for QuerySpec. Only include fields that change.
    """
    try:
        llm_instance = llm.get_llm("adjust_diff", temperature=0.2)

        chat_history = chat_history or []
        recent_msgs = chat_history[-10:]

        def fmt(m):
            role = m.get("role", "user")
            content = m.get("content", "").strip()
            return f"{role.upper()}: {content}"

        history_text = "\n".join(fmt(m) for m in recent_msgs)

        conf_list = ", ".join(config.DEFAULT_CONFERENCES.keys())
        prompt = (
            "SYSTEM ROLE: You update a structured search spec for a recruitment/talent search engine.\n"
            "You MUST output STRICT JSON matching the QuerySpecDiff schema (omit fields that do not change).\n\n"
            "OBJECTIVE\n"
            "Given (1) the current JSON spec, (2) a new user instruction, and (3) recent conversation snippets,\n"
            "produce a PARTIAL QuerySpecDiff with ONLY the fields changed by the new instruction.\n\n"
            "SCHEMA (QuerySpecDiff): {"
            " 'top_n': int?, 'years': int[]?, 'venues': string[]?, 'keywords': string[]?, 'must_be_current_student': bool?, 'degree_levels': string[]?, 'author_priority': string[]?, 'extra_constraints': string[]? }\n\n"
            "EDIT RULES\n"
            "- ADDITIVE language (e.g., 'also include X', 'add Y'): union with existing list and return the NEW list for that field.\n"
            "- EXCLUSIVE language (e.g., 'only X', 'strictly X', 'limit to X'): replace the list with exactly those items.\n"
            "- NEGATION (e.g., 'exclude X', 'no X'): remove those items if present and return the NEW list.\n"
            "- If a field is NOT mentioned, DO NOT include it in the output.\n"
            "- Numbers: set 'top_n' to the salient integer.\n"
            "- Years: handle explicit years or phrases like 'last 2 years' to [CURRENT_YEAR, CURRENT_YEAR-1]. If not mentioned, omit.\n\n"
            "NORMALIZATION\n"
            f"- Venues canonicalization. Known venues include: {conf_list}. Map 'NIPS'->'NeurIPS', 'The Web Conference'->'WWW'. Deduplicate, preserve user order.\n"
            "- Degree levels canonical set: ['PhD','MSc','Master','Graduate','Undergraduate','Bachelor','Postdoc'] with common synonyms.\n"
            "- Author priority canonical set: ['first','last','corresponding'] (lead->first, senior->last).\n"
            "- Keywords: lower-case, trim, deduplicate.\n\n"
            "CONSTRAINTS\n"
            "- Output MUST be valid JSON for QuerySpecDiff. No markdown, no prose.\n\n"
            "INPUTS\n"
            "=== Conversation (most recent last) ===\n"
            f"{history_text}\n\n"
            "=== Current Spec (JSON) ===\n"
            f"{json.dumps(current_spec, ensure_ascii=False)}\n\n"
            "=== New User Instruction ===\n"
            f"{user_input}\n\n"
            "OUTPUT FORMAT\n"
            "Return ONLY the final JSON for QuerySpecDiff."
        )
        diff = llm.safe_structured(llm_instance, prompt, QuerySpecDiff)
        return diff
    except Exception:
        return None

def merge_query_spec_with_diff(
    base_spec: Dict[str, Any], diff: QuerySpecDiff
) -> Dict[str, Any]:
    """Apply QuerySpecDiff fields onto base_spec and return a new dict."""
    try:
        updated = dict(base_spec)
        data = diff.dict(exclude_none=True)
        for k, v in data.items():
            updated[k] = v
        return updated
    except Exception:
        # Safe fallback: return original
        return base_spec

if __name__ == "__main__":
    agent_search_input = QuerySpec(
        top_n=3,
        years=[2025, 2024],
        venues=[
            "ICLR",
            "ICML",
            "NeurIPS",
            "ACL",
            "EMNLP",
            "NAACL",
            "KDD",
            "WWW",
            "AAAI",
            "IJCAI",
            "CVPR",
            "ECCV",
            "ICCV",
            "SIGIR",
        ],
        keywords=["LLM", "Large Language Model", "graph neural network"],
        must_be_current_student=True,
        degree_levels=["PhD", "MSc"],
        author_priority=["first", "last"],
        extra_constraints=[],
    )

    # For testing, you may need to provide an API key
    # agent_execute_search(agent_search_input, api_key="your_api_key_here")
    print("Test code disabled - requires API key")
def agent_generate_search_summary(results, query_spec):
    # Get count of candidates displayed to user (recommended + additional)
    displayed_count = len(results.recommended_candidates) + len(results.additional_candidates)
    # Get count of all candidates
    matching_count = getattr(results, 'matching_candidates_count', displayed_count)
    # Get total candidates found
    total_count = getattr(results, 'total_candidates_found', displayed_count)
    # ✅ 处理0个候选人的情况
    if matching_count == 0:
        if total_count > 0:
            # 找到了候选人但都被过滤了
            return f"Search completed, but no candidates met the matching criteria. Found {total_count} potential candidates but none matched your requirements closely enough. Consider broadening your search keywords or adjusting filters."
        else:
            # 完全没找到候选人
            return "Search completed, but no candidates were found. This might be because: 1) The keywords are too specific, 2) The conference/venue filters are too restrictive, or 3) There are limited publications in this area. Consider broadening your search criteria."
    # Build research areas text
    areas = []
    for c in results.recommended_candidates[:3]:
        if hasattr(c, 'research_focus'):
            areas.extend(c.research_focus[:2])
    
    areas_text = ", ".join(list(set(areas))[:5]) if areas else "relevant research areas"
    
    # Generate summary with context
    if matching_count > displayed_count:
        # We have more matching candidates than we're displaying
        return f"Found {matching_count} candidates matching your criteria (showing top {displayed_count} by relevance), specializing in {areas_text}."
    elif total_count > matching_count:
        # Some candidates were filtered out
        return f"Found {matching_count} candidates matching your criteria (filtered from {total_count} total), specializing in {areas_text}."
    else:
        # All candidates match and are shown
        return f"Found {displayed_count} candidates specializing in {areas_text}."
