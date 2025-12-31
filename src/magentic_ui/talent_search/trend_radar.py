"""
Trend Radar Module for Talent Search
Specialized search logic for trend radar use case

✅ 核心改进：完全复用 Targeted Search 的成熟搜索引擎
- search_by_direction: 直接调用 agent_execute_search
- search_by_name: 直接调用 orchestrate_candidate_report
"""

from typing import List, Dict, Any, Tuple, Optional

# Import from talent_search_module
from . import schemas
from .author_discovery import orchestrate_candidate_report


def search_by_name(
    author_name: str,
    direction_title: str = "",
    api_key: str = None
) -> Optional[schemas.CandidateOverview]:
    """
    从推文获取的人才名字进行搜索
    直接从 OpenReview 开始构建 profile，跳过前期搜索步骤
    
    Args:
        author_name: 人才名字
        direction_title: 研究方向标题（用于评估候选人）
        api_key: API key for LLM calls
        
    Returns:
        CandidateOverview or None if not found
    """
    try:
        print(f"\n[Trend Radar - Name Search] Searching: {author_name}")
        
        # 🔧 改进：直接使用方向名称作为 user_query（更精准）
        user_query = direction_title if direction_title else "Research expertise"
        
        # 直接调用 orchestrate_candidate_report - 跳过论文搜索阶段
        # 注意：orchestrate_candidate_report 返回 4 个值
        profile, overview, eval_res, enhanced_profile = orchestrate_candidate_report(
            first_author=author_name,
            paper_title="",           # 没有论文标题
            paper_url=None,
            aliases=[],               # 没有别名
            k_queries=10,             # 适度的查询数量
            author_id=None,
            api_key=api_key,
            use_lightweight_mode=True,  # 使用轻量级模式，提高速度
            user_query=user_query     # 🔧 直接使用方向名称
        )
        
        if overview is None:
            print(f"[Trend Radar - Name Search] {author_name} not found (filtered by internal logic)")
            return None
        
        print(f"[Trend Radar - Name Search] ✓ Found: {author_name}")
        return overview
        
    except Exception as e:
        print(f"[Trend Radar - Name Search] Error for {author_name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def search_by_names(
    names: List[str],
    direction_title: str = "",
    api_key: str = None,
    max_results: int = None
) -> List[schemas.CandidateOverview]:
    """
    批量搜索多个人名（推文来源）
    
    Args:
        names: 人才名字列表
        direction_title: 研究方向标题（用于评估候选人）
        api_key: API key for LLM calls
        max_results: 最多返回多少个结果（None表示全部）
        
    Returns:
        List of CandidateOverview objects
    """
    results = []
    target = max_results if max_results else len(names)
    
    print(f"\n[Trend Radar - Names Search] Starting search for {len(names)} names (target: {target})")
    
    for name in names:
        if max_results and len(results) >= max_results:
            print(f"[Trend Radar - Names Search] Reached target ({max_results}), stopping")
            break
            
        overview = search_by_name(name, direction_title=direction_title, api_key=api_key)
        if overview:
            results.append(overview)
    
    print(f"\n[Trend Radar - Names Search] Completed: Found {len(results)}/{len(names)} candidates")
    return results


def search_by_direction(
    direction_title: str,
    max_candidates: int = 3,
    api_key: str = None,
    progress_callback=None
) -> schemas.SearchResults:
    """
    为研究方向搜索人才 - 完全复用 Targeted Search 的成熟流程
    
    ✅ 新逻辑：直接调用 agent_execute_search（Targeted Search 的核心函数）
    - 不再自己实现论文搜索、作者提取、候选人构建
    - 复用所有成熟的搜索、评分、过滤逻辑
    - 保证与 Targeted Search 完全一致的质量和稳定性
    
    Args:
        direction_title: 研究方向标题（如 "Brain-Computer Interface"）
        max_candidates: 需要的候选人数量
        api_key: API key for LLM calls
        progress_callback: Optional callback function(stage, progress, message)
        
    Returns:
        SearchResults with recommended candidates
    """
    print(f"\n[Trend Radar - Direction Search] Starting search for: '{direction_title}'")
    print(f"[Trend Radar - Direction Search] Target candidates: {max_candidates}")
    print(f"[Trend Radar - Direction Search] 🔧 Using Targeted Search engine (agent_execute_search)")
    
    try:
        # 构建 QuerySpec - 使用方向名称作为关键词
        spec = schemas.QuerySpec(
            top_n=max_candidates,
            keywords=[direction_title],     # ✅ 方向名称作为搜索关键词
            venues=[],                       # 让系统自动推断相关会议
            years=[],                        # 让系统自动选择时间范围
            must_be_current_student=False,   # Trend Radar 不限学生身份
            degree_levels=["PhD", "Master", "Postdoc"],
            author_priority=["first", "last"]  # ✅ 第一作者和最后作者（导师）
        )
        
        # ✅ 核心改动：直接调用 Targeted Search 的完整搜索引擎
        from .agents import agent_execute_search
        
        print(f"[Trend Radar - Direction Search] Calling agent_execute_search with:")
        print(f"  - Keywords: {spec.keywords}")
        print(f"  - Target: {spec.top_n} candidates")
        print(f"  - Author priority: {spec.author_priority}")
        
        # 调用完整的搜索引擎（包含所有步骤）
        results = agent_execute_search(
            spec=spec,
            api_key=api_key,
            progress_callback=progress_callback,
            log_callback=None,
            max_rounds_per_run=1  # Trend Radar 只搜索一轮（快速模式）
        )
        
        print(f"[Trend Radar - Direction Search] ✅ Search completed!")
        print(f"  - Recommended: {len(results.recommended_candidates)}")
        print(f"  - Additional: {len(results.additional_candidates)}")
        print(f"  - Reference papers: {len(results.reference_papers)}")
        print(f"  - Total candidates found: {results.total_candidates_found}")
        
        return results
        
    except Exception as e:
        print(f"[Trend Radar - Direction Search] ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        
        # 返回空结果
        return schemas.SearchResults(
            recommended_candidates=[],
            additional_candidates=[],
            reference_papers=[],
            total_candidates_found=0,
            search_query=direction_title
        )


def search_with_fallback(
    generated_names: List[str],
    direction_title: str,
    max_candidates: int = 3,
    api_key: str = None
) -> Tuple[List[schemas.CandidateOverview], List[schemas.CandidateOverview]]:
    """
    智能人才搜索：优先搜索AI生成的人才姓名，不足时用方向搜索补齐
    
    Args:
        generated_names: AI生成的人才姓名列表
        direction_title: 研究方向标题
        max_candidates: 需要的候选人总数
        api_key: API key
        
    Returns:
        Tuple of (from_names, from_direction) - 两个候选人列表
    """
    print(f"\n[Trend Radar - Fallback Search] Starting")
    print(f"[Trend Radar - Fallback Search] Generated names: {len(generated_names)}")
    print(f"[Trend Radar - Fallback Search] Target candidates: {max_candidates}")
    
    # 先按姓名搜索（推文来源）
    from_names = search_by_names(
        names=generated_names,
        direction_title=direction_title,  # 🔧 添加 direction_title
        api_key=api_key,
        max_results=max_candidates  # 最多搜索到需要的数量
    )
    
    print(f"[Trend Radar - Fallback Search] From names: {len(from_names)} candidates")
    
    # 如果不足，用方向搜索补齐
    from_direction = []
    if len(from_names) < max_candidates:
        needed = max_candidates - len(from_names)
        print(f"[Trend Radar - Fallback Search] Need {needed} more candidates from direction search")
        
        direction_results = search_by_direction(
            direction_title=direction_title,
            max_candidates=needed,
            api_key=api_key
        )
        
        from_direction = direction_results.recommended_candidates
        print(f"[Trend Radar - Fallback Search] From direction: {len(from_direction)} candidates")
    
    print(f"[Trend Radar - Fallback Search] Total: {len(from_names) + len(from_direction)} candidates")
    return from_names, from_direction
