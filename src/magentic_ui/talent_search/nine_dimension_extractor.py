"""
Nine-Dimension Parallel Extractor
实现多源分级提取：Homepage → Scholar → ORCID → DBLP
每个数据源并行调用9个维度 prompts，按优先级填充字段
"""

from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass
import traceback
import re
import json
import time

from . import llm
from . import schemas
from .nine_dimension_prompts import (
    PROMPT_EXTRACT_BACKGROUND,
    PROMPT_EXTRACT_RESEARCH_INTERESTS,
    PROMPT_EXTRACT_PUBLICATIONS,
    PROMPT_EXTRACT_AWARDS,
    PROMPT_EXTRACT_SERVICES,
    PROMPT_EXTRACT_CAREER_EDUCATION,
    PROMPT_EXTRACT_EXPERIENCE,
    PROMPT_EXTRACT_CONTACT
)
from .dynamic_concurrency import get_llm_workers

# 统一缓存版本，确保 prompt / 规则更新后能触发刷新
EXTRACTED_DIMENSIONS_VERSION = "2025-11-17-en-v1"


def _print_dimension_content(dim_name: str, dim_data: Dict[str, Any], source: str = ""):
    """
    打印维度内容的摘要信息（用于调试）
    
    Args:
        dim_name: 维度名称
        dim_data: 维度数据字典
        source: 数据来源（可选）
    """
    if not dim_data:
        print(f"[Multi-Source] 📋 {dim_name} content: (empty)")
        return
    
    source_prefix = f"[{source}] " if source else ""
    
    # 根据维度类型打印不同的摘要信息
    if dim_name == 'services':
        services_list = dim_data.get('services', [])
        talks_list = dim_data.get('invited_talks', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(services_list)} services, {len(talks_list)} talks")
    elif dim_name == 'publications':
        pubs_list = dim_data.get('publications', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(pubs_list)} publications")
    elif dim_name == 'awards':
        awards_list = dim_data.get('awards', [])
        # 统计不同类型
        paper_count = sum(1 for a in awards_list if isinstance(a, dict) and a.get('award_type', '').lower() == 'paper')
        project_count = sum(1 for a in awards_list if isinstance(a, dict) and a.get('award_type', '').lower() == 'project')
        personal_count = sum(1 for a in awards_list if isinstance(a, dict) and a.get('award_type', '').lower() == 'personal')
        total_awards = len(awards_list)
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {total_awards} awards ({paper_count} paper, {project_count} project, {personal_count} personal)")
    elif dim_name == 'education':
        edu_list = dim_data.get('education', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(edu_list)} entries")
    elif dim_name == 'experience':
        exp_list = dim_data.get('experiences', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(exp_list)} entries")
    elif dim_name == 'career_education':
        career_edu_list = dim_data.get('career_education', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(career_edu_list)} entries")
    elif dim_name == 'research_interests':
        interests_list = dim_data.get('interests', [])
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(interests_list)} interests")
    elif dim_name == 'contact':
        email = dim_data.get('email', '') or 'N/A'
        social = dim_data.get('social_links', {}) or {}
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: email={email}, social_links={len(social)}")
    else:
        # 通用打印：显示所有字段的键
        keys = list(dim_data.keys())
        print(f"[Multi-Source] 📋 {source_prefix}{dim_name} content: {len(keys)} fields - {', '.join(keys)}")


def _validate_and_clean_awards(awards_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate unified awards data.
    
    Rules:
    1. Paper awards must have paper_title and venue
    2. Project awards must have project_name
    3. Personal awards must have name
    4. Remove invalid entries
    
    Args:
        awards_data: Raw awards data with unified awards list
        
    Returns:
        Cleaned awards data
    """
    if not awards_data or not isinstance(awards_data, dict):
        return {"awards": []}
    
    awards = awards_data.get('awards', [])
    if not isinstance(awards, list):
        return {"awards": []}
    
    cleaned = []
    
    # Common helper: normalize a name string (lowercase + collapse spaces)
    def _norm_name(s: str) -> str:
        return " ".join((s or '').strip().lower().split())

    # Obvious non-award phrases that should never be treated as awards
    banned_non_award_names = {
        "poster",
        "poster presentation",
        "oral presentation",
        "invited talk",
        "talk",
        "publication",
        "publication list",
        "one paper accepted",
        "paper accepted",
        "accepted paper",
        "camera-ready",
        "preprint",
        "acceptance",
        # Paper tracks/categories (not awards)
        "main",
        "demo",
        "findings",
        "workshop",
        "short paper",
        "short",
        "long paper",
        "long",
        "spotlight",
        "oral",
        "student abstract",
        "extended abstract",
    }
    
    for award in awards:
        if not isinstance(award, dict):
            continue
            
        award_type = award.get('award_type', 'personal').lower()
        name = (award.get('name') or '').strip()
        
        # 基础验证：必须有 name
        if not name or len(name) < 3:
            continue
        
        # 检查是否为非 award 短语
        norm_name = _norm_name(name)
        if norm_name in banned_non_award_names:
            print(f"[Awards Validation] ⚠️ Skipped award with non-award name: {name}")
            continue
        
        # 检查 track 前缀
        name_lower = name.lower()
        track_prefixes = ['main -', 'demo -', 'findings -', 'workshop -', 'short -', 'long -', 
                         'spotlight -', 'oral -', 'poster -']
        if any(name_lower.startswith(prefix) for prefix in track_prefixes):
            print(f"[Awards Validation] ⚠️ Skipped award with track prefix: {name}")
            continue
        
        # 根据类型验证必填字段
        if award_type == 'paper':
            paper_title = (award.get('paper_title') or '').strip()
            venue = (award.get('venue') or '').strip()
            if not paper_title or len(paper_title) < 5 or not venue:
                print(f"[Awards Validation] ⚠️ Skipped paper award with missing paper_title/venue: {name}")
                continue
        elif award_type == 'project':
            project_name = (award.get('project_name') or '').strip()
            if not project_name or len(project_name) < 3:
                print(f"[Awards Validation] ⚠️ Skipped project award with missing project_name: {name}")
                continue
        
        cleaned.append(award)
    
    total_removed = len(awards) - len(cleaned)
    if total_removed > 0:
        print(f"[Awards Validation] 🧹 Removed {total_removed} invalid award entries")
    
    return {"awards": cleaned}


@dataclass
class DimensionResult:
    """单个维度的提取结果"""
    dimension_name: str
    data: Optional[Dict[str, Any]]
    source: str  # 'homepage' / 'scholar' / 'orcid' / 'dblp'
    success: bool
    error: Optional[str] = None


class NineDimensionExtractor:
    """九维度并行提取器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.llm_client = llm.get_llm("extract", temperature=0.1, api_key=api_key)
    
    def extract_from_text(self, text: str, author_name: str, source_name: str, target_dimensions: List[str] = None) -> Dict[str, DimensionResult]:
        """
        从单个文本源并行提取维度
        
        Args:
            text: 文本内容
            author_name: 作者姓名
            source_name: 来源名称 ('homepage' / 'scholar' / 'orcid' / 'dblp')
            target_dimensions: 目标维度列表（None=提取所有9个维度，否则只提取指定维度）
            
        Returns:
            Dict[dimension_name, DimensionResult]
        """
        # 定义8个提取任务（合并了 education 和 teaching 为 career_education）
        all_extraction_tasks = [
            ("background", PROMPT_EXTRACT_BACKGROUND, schemas.LLMBackgroundSpec),
            ("research_interests", PROMPT_EXTRACT_RESEARCH_INTERESTS, schemas.LLMResearchInterestsRawSpec),
            ("publications", PROMPT_EXTRACT_PUBLICATIONS, schemas.LLMPublicationsSpec),
            ("awards", PROMPT_EXTRACT_AWARDS, schemas.LLMAwardsSpec),
            ("services", PROMPT_EXTRACT_SERVICES, schemas.LLMServicesSpec),
            ("career_education", PROMPT_EXTRACT_CAREER_EDUCATION, schemas.LLMCareerEducationSpec),
            ("experience", PROMPT_EXTRACT_EXPERIENCE, schemas.LLMExperienceSpec),
            ("contact", PROMPT_EXTRACT_CONTACT, schemas.LLMContactSpec),
        ]
        
        # 如果指定了目标维度，只提取这些维度
        normalized_targets = None
        if target_dimensions:
            alias_map = {
                "education": "career_education",
                "teaching": "career_education",
            }
            normalized_targets = []
            for dim in target_dimensions:
                mapped = alias_map.get(dim, dim)
                if mapped not in normalized_targets:
                    normalized_targets.append(mapped)
            if normalized_targets != target_dimensions:
                print(f"[9D] 🔄 Normalized target dimensions: {', '.join(normalized_targets)}")
            target_dimensions = normalized_targets
            extraction_tasks = [
                task for task in all_extraction_tasks 
                if task[0] in target_dimensions
            ]
            print(f"[9D] 🎯 Extracting {len(target_dimensions)} target dimensions: {', '.join(target_dimensions)}")
        else:
            extraction_tasks = all_extraction_tasks
        
        results = {}
        
        # Extract dimensions in parallel
        task_count = len(extraction_tasks)
        if task_count == 0:
            print(f"[9D] ⚠️ No valid extraction tasks for requested dimensions: {', '.join(target_dimensions or [])}")
            return {}
        
        max_workers = min(task_count, get_llm_workers(task_count))
        
        print(f"[9D] 🚀 Starting parallel extraction: {task_count} tasks, {max_workers} workers")
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_dim = {
                executor.submit(
                    self._extract_single_dimension,
                    dim_name,
                    prompt_func,
                    schema_cls,
                    author_name,
                    text,
                    source_name
                ): dim_name
                for dim_name, prompt_func, schema_cls in extraction_tasks
            }
            
            # 收集结果，添加进度跟踪
            completed_count = 0
            try:
                for future in as_completed(future_to_dim, timeout=20):  # 总体超时 20 秒
                    dim_name = future_to_dim[future]
                    try:
                        elapsed = time.time() - start_time
                        print(f"[9D] ⏳ Extracting {dim_name}... (elapsed: {elapsed:.1f}s, completed: {completed_count}/{task_count})")
                        result = future.result(timeout=15)
                        results[dim_name] = result
                        completed_count += 1
                        elapsed_after = time.time() - start_time
                        status = "✅" if result.success else "❌"
                        print(f"[9D] {status} {dim_name} completed in {elapsed_after - elapsed:.1f}s ({completed_count}/{task_count} done)")
                            
                    except TimeoutError as e:
                        elapsed = time.time() - start_time
                        print(f"[9D] ⏱️ {dim_name} timeout after {elapsed:.1f}s: {e}")
                        results[dim_name] = DimensionResult(
                            dimension_name=dim_name,
                            data=None,
                            source=source_name,
                            success=False,
                            error=f"Timeout after {elapsed:.1f}s"
                        )
                        completed_count += 1
                    except Exception as e:
                        elapsed = time.time() - start_time
                        print(f"[9D] ❌ {dim_name} failed after {elapsed:.1f}s: {e}")
                        results[dim_name] = DimensionResult(
                            dimension_name=dim_name,
                            data=None,
                            source=source_name,
                            success=False,
                            error=str(e)
                        )
                        completed_count += 1
            except TimeoutError:
                # 总体超时，处理剩余任务
                elapsed = time.time() - start_time
                print(f"[9D] ⚠️ Overall timeout ({elapsed:.1f}s), processing remaining tasks...")
                remaining = task_count - completed_count
                if remaining > 0:
                    print(f"[9D] ⚠️ {remaining} tasks still pending, marking as timeout")
                    for future, dim_name in future_to_dim.items():
                        if dim_name not in results:
                            results[dim_name] = DimensionResult(
                                dimension_name=dim_name,
                                data=None,
                                source=source_name,
                                success=False,
                                error="Overall timeout (20s)"
                            )
        
        success_count = sum(1 for r in results.values() if r.success)
        total_dims = len(target_dimensions) if target_dimensions else 9
        print(f"[9D] Extracted {success_count}/{total_dims} dimensions from {source_name}")
        
        return results
    
    def _extract_single_dimension(
        self,
        dimension_name: str,
        prompt_func,
        schema_cls,
        author_name: str,
        text: str,
        source_name: str
    ) -> DimensionResult:
        """
        提取单个维度
        
        Args:
            dimension_name: 维度名称
            prompt_func: Prompt 生成函数
            schema_cls: Pydantic schema 类
            author_name: 作者姓名
            text: 文本内容
            source_name: 来源名称
            
        Returns:
            DimensionResult
        """
        try:
            # 生成 prompt
            prompt = prompt_func(author_name, text)
            
            # 调用 LLM（添加超时保护）
            dim_start_time = time.time()
            result = llm.safe_structured(self.llm_client, prompt, schema_cls)
            dim_elapsed = time.time() - dim_start_time
            if dim_elapsed > 10:  # 如果超过 10 秒，记录警告
                print(f"[9D] ⚠️ {dimension_name} took {dim_elapsed:.1f}s (slow)")
            
            if result:
                # 转换为字典
                data = result.model_dump() if hasattr(result, 'model_dump') else dict(result)
                
                # 特殊处理：验证和清理 awards 数据
                if dimension_name == 'awards':
                    data = _validate_and_clean_awards(data)
                
                return DimensionResult(
                    dimension_name=dimension_name,
                    data=data,
                    source=source_name,
                    success=True
                )
            else:
                return DimensionResult(
                    dimension_name=dimension_name,
                    data=None,
                    source=source_name,
                    success=False,
                    error="LLM returned None"
                )
                
        except Exception as e:
            return DimensionResult(
                dimension_name=dimension_name,
                data=None,
                source=source_name,
                success=False,
                error=str(e)
            )


class MultiSourceNineDimensionCollector:
    """多源九维度采集器（按优先级填充）"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
        self.extractor = NineDimensionExtractor(api_key=api_key)
    def collect_from_sources(
        self,
        author_name: str,
        profile: Any,  # AuthorProfile
        openreview_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        智能融合三大主源数据，按优先级填充九维度
        
        1. 优先使用增量提取结果：如果 homepage 有已提取的 extracted_dimensions，直接使用
        2. 准备融合文本：Homepage + OpenReview + Scholar （三大主源）
        3. 补充提取：只对缺失的维度进行提取
        4. 辅助补充：ORCID/DBLP用于补充空白维度
        
        Args:
            author_name: 作者姓名
            profile: AuthorProfile 对象（包含多源采集的原始数据）
            openreview_data: OpenReview 原始数据
            
        Returns:
            Dict with merged 9-dimension data
        """
        
        # ========================================================================== 
        # Step 0: 优先使用增量提取的结果（如果存在）
        #         并在刷新时保留原始 homepage 结果作为备份，避免被更差/为空的 fused 结果覆盖
        # ========================================================================== 
        merged_dimensions = {}
        homepage_metadata = getattr(profile, '_homepage_metadata', {}) if profile else {}
        extracted_dimensions = homepage_metadata.get('extracted_dimensions', {}) if homepage_metadata else {}
        extracted_dimensions_meta = homepage_metadata.get('extracted_dimensions_meta', {}) if homepage_metadata else {}
        # 记录所有增量维度的备份（包括可能被判定为 stale 的维度）
        if extracted_dimensions:
            print(f"[Multi-Source] 🎯 Using incremental extraction results from homepage: {len(extracted_dimensions)}/9 dimensions")
            
            validated_dimensions = extracted_dimensions.copy()

            filtered_dimensions = {}
            for dim_name, dim_data in validated_dimensions.items():
                dim_meta = extracted_dimensions_meta.get(dim_name, {}) if isinstance(extracted_dimensions_meta, dict) else {}

                if dim_name in ("career_education", "experience"):
                    if not self._dimension_seems_about_author(dim_data, author_name, dim_meta):
                        print(f"[Multi-Source] ⚠️ Rejected {dim_name} from homepage (role may not belong to {author_name})")
                        continue

                filtered_dimensions[dim_name] = dim_data

            merged_dimensions = filtered_dimensions.copy()
            if filtered_dimensions:
                print(f"[Multi-Source] ✅ Loaded {len(filtered_dimensions)} validated dimensions: {', '.join(filtered_dimensions.keys())}")            
            if validated_dimensions:
                print(f"[Multi-Source] ✅ Loaded {len(validated_dimensions)} validated dimensions: {', '.join(validated_dimensions.keys())}")
            else:
                print(f"[Multi-Source] ⚠️ No valid dimensions from homepage 9D")
        else:
            print(f"[Multi-Source] ℹ️ No incremental dimensions available from homepage metadata")
        
        # ==========================================================================
        # Step 1: 构建补充文本（仅 Homepage）
        # ==========================================================================
        
        supplement_text_parts = []
        source_stats = {}
        
        # Homepage（唯一文本源，用于完善 profile）
        homepage_text = self._get_homepage_text(profile)
        if homepage_text and len(homepage_text) > 100:
            supplement_text_parts.append("="*80)
            supplement_text_parts.append("SOURCE: PERSONAL HOMEPAGE")
            supplement_text_parts.append("="*80)
            supplement_text_parts.append(homepage_text[:20000])  # 最多20K字符
            source_stats['homepage'] = len(homepage_text)
            print(f"[Stage 2] ✅ Homepage: {len(homepage_text):,} chars available for supplement")
        else:
            print(f"[Stage 2] ⚠️ Homepage: Not available")
        
        # Generate supplement text (only Homepage)
        if not supplement_text_parts:
            # 如果已经有增量提取结果，直接返回
            if merged_dimensions:
                print(f"[Stage 2] ✅ Using homepage incremental results, no supplement needed")
                return merged_dimensions
            print(f"[Stage 2] ⚠️ No data sources available, returning empty dimensions")
            return {}
        
        supplement_text = '\n'.join(supplement_text_parts)
        total_chars = len(supplement_text)
        
        # ==========================================================================
        # Step 2: 只对缺失的维度进行补充提取
        # ==========================================================================
        
        # 非论文字段：这些字段不受 Scholar 影响，只由 OpenReview + Homepage 决定
        non_publication_dimensions = ['background', 'research_interests', 'awards', 
                                      'services', 'career_education', 'experience', 'contact']
        
        # 检查哪些维度还缺失
        missing_dims = [d for d in non_publication_dimensions if d not in merged_dimensions]
        existing_dims = [d for d in non_publication_dimensions if d in merged_dimensions]

        # 不再识别 stale 维度，Homepage 九维结果受到完全保护
        # 原因：Homepage 是 Stage 2 的主要来源，其结果优先级高于任何文本提取
        print(f"[Stage 2] ℹ️ Homepage 9D dimensions are protected (no stale refresh)")
        if existing_dims:
            print(f"[Stage 2] ✅ Protected dimensions from homepage: {', '.join(existing_dims)}")

        # 只对缺失维度进行补充提取（不再有 refresh 逻辑）
        if missing_dims:
            print(f"[Stage 2] 📝 Supplement text: {total_chars:,} chars from Homepage")
            print(f"[Stage 2] 🔍 Extracting missing dimensions: {', '.join(missing_dims)}")

            # 仅对缺失维度进行提取
            supplement_results = self.extractor.extract_from_text(
                supplement_text,
                author_name,
                "homepage_supplement",
                target_dimensions=missing_dims
            )

            # 处理缺失维度：直接使用 supplement 结果
            for dim_name in missing_dims:
                if dim_name in supplement_results:
                    dim_result = supplement_results[dim_name]
                    if dim_result.success and dim_result.data:
                        merged_dimensions[dim_name] = dim_result.data
                        print(f"[Stage 2] ✅ Added {dim_name} from homepage supplement")
                        _print_dimension_content(dim_name, dim_result.data, "homepage supplement")
        elif existing_dims:
            # 如果所有维度都已存在，不需要补充提取
            print(f"[Stage 2] ✅ All {len(existing_dims)} dimensions already available from homepage 9D, no supplement needed")
        else:
            print(f"[Stage 2] ⚠️ No dimensions available (should not happen)")
        
        # ==========================================================================
        # Step 3: 辅助源补充（ORCID 等，仅在仍有空缺时使用）
        # ==========================================================================
        
        empty_dims = [d for d in non_publication_dimensions if d not in merged_dimensions]
        
        if empty_dims:
            print(f"\n[Stage 2] 🔍 Still missing {len(empty_dims)} dimensions, trying auxiliary sources...")
            print(f"[Stage 2] Empty dimensions: {', '.join(empty_dims)}")
            
            # ORCID辅助（保留）
            orcid_text = self._get_orcid_text(profile)
            if orcid_text and len(orcid_text) > 100:
                orcid_results = self.extractor.extract_from_text(orcid_text, author_name, "orcid", target_dimensions=empty_dims)
                
                filled_count = 0
                for dim_name in empty_dims:
                    if dim_name in orcid_results and orcid_results[dim_name].success and orcid_results[dim_name].data:
                        
                        merged_dimensions[dim_name] = orcid_results[dim_name].data
                        filled_count += 1
                        print(f"[Stage 2] ✅ Added {dim_name} from ORCID")                
                if filled_count == 0:
                    print(f"[Stage 2] ⚠️ ORCID did not fill any missing dimensions")
        
        # ==========================================================================
        # Step 4: 最终统计
        # ==========================================================================
        
        filled_dims = [d for d in non_publication_dimensions if d in merged_dimensions]
        print(f"\n[Stage 2] ✅ Profile fields complete: {len(filled_dims)}/{len(non_publication_dimensions)} dimensions")
        print(f"[Stage 2] Available dimensions: {', '.join(filled_dims)}")
        
        still_missing = [d for d in non_publication_dimensions if d not in merged_dimensions]
        if still_missing:
            print(f"[Stage 2] ⚠️ Still missing: {', '.join(still_missing)}")
        
        return merged_dimensions

    def _detect_stale_dimensions(self, dims: Dict[str, Any], meta: Dict[str, Any]) -> List[str]:
        """
        判断缓存是否需要刷新：
        1. 版本号不一致 → 全量刷新
        2. 字段包含中文/非 ASCII 内容 → 针对性刷新
        """
        if not dims:
            return []
        if not isinstance(meta, dict):
            meta = {}
        version = meta.get("version")
        if version != EXTRACTED_DIMENSIONS_VERSION:
            return list(dims.keys())
        stale = []
        quality_flags = meta.get("quality_flags", {}) if isinstance(meta.get("quality_flags", {}), dict) else {}
        for dim_name, dim_data in dims.items():
            if quality_flags.get(dim_name) == 'rejected_noise':
                stale.append(dim_name)
        return stale

    def _dimension_seems_about_author(
        self,
        dim_data: Any,
        author_name: str,
        dim_meta: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Heuristically verify that homepage roles/education refer to the target author."""

        author_lower = (author_name or "").lower()

        if isinstance(dim_meta, dict):
            owner_field = dim_meta.get("owner") or dim_meta.get("name") or dim_meta.get("author")
            if owner_field and author_lower and author_lower not in str(owner_field).lower():
                return False

        texts = self._collect_text_fragments(dim_data)
        if not texts:
            return True

        joined_text = " \n ".join(texts)
        joined_lower = joined_text.lower()

        if author_lower and author_lower in joined_lower:
            return True

        if re.search(r"\b(i|my|me|mine)\b", joined_lower):
            return True

        name_candidates = re.findall(r"\b([A-Z][a-z]+\s+[A-Z][a-z]+)\b", joined_text)
        org_stopwords = {"University", "Institute", "Laboratory", "College", "School", "Center", "Department"}
        suspicious_names = [
            name for name in name_candidates
            if not any(part in org_stopwords for part in name.split()) and (not author_lower or name.lower() != author_lower)
        ]

        if suspicious_names and author_lower and author_lower not in joined_lower:
            return False

        return True

    def _collect_text_fragments(self, value: Any) -> List[str]:
        """Flatten nested dimension content into a list of text fragments for checks."""
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            fragments: List[str] = []
            for item in value:
                fragments.extend(self._collect_text_fragments(item))
            return fragments
        if isinstance(value, dict):
            fragments: List[str] = []
            for v in value.values():
                fragments.extend(self._collect_text_fragments(v))
            return fragments
        return [str(value)]
    def _get_homepage_text(self, profile) -> Optional[str]:
        """
        从 profile 获取 homepage 文本
        优先使用 extraction_text（智能压缩版），fallback 到完整文本
        """
        # 优先：提取用文本（智能压缩，适合 LLM）
        extraction_text = getattr(profile, '_homepage_extraction_text', None)
        if extraction_text:
            return extraction_text
        
        # Fallback：完整文本
        return getattr(profile, '_homepage_raw_text', None) or ''
    
    def _get_openreview_text(self, profile) -> Optional[str]:
        """从 profile 获取 openreview 文本"""
        return getattr(profile, '_openreview_raw_text', None) or ''
    
    def _get_scholar_text(self, profile) -> Optional[str]:
        """从 profile 获取 scholar 文本"""
        return getattr(profile, '_scholar_raw_text', None) or ''
    
    def _get_orcid_text(self, profile) -> Optional[str]:
        """从 profile 获取 orcid 文本"""
        return getattr(profile, '_orcid_raw_text', None) or ''
    
    def _get_dblp_text(self, profile) -> Optional[str]:
        """从 profile 获取 dblp 文本"""
        return getattr(profile, '_dblp_raw_text', None) or ''
    
    def _is_better_dimension_data(self, existing_data: Dict, new_data: Dict, dim_name: str) -> bool:
        """
        判断新数据是否比现有数据更好
        
        判断标准：
        1. 列表型字段：新数据的列表长度更长
        2. 字符串字段：新数据的字符串更长且更详细
        3. 字典字段：新数据包含更多键值对
        
        Returns:
            True: 新数据更好，应该替换
            False: 现有数据更好，保留现有数据
        """
        if not existing_data or not new_data:
            return bool(new_data and not existing_data)
        
        # 列表型字段：比较列表长度
        list_fields = ['publications', 'awards', 'education', 'experiences', 'teaching', 'interests', 'services', 'invited_talks']
        
        if dim_name in list_fields:
            # 对于列表型维度，比较主要字段的列表长度
            existing_count = 0
            new_count = 0
            
            # 统计现有数据的项目数
            if isinstance(existing_data, dict):
                for field in list_fields:
                    if field in existing_data and isinstance(existing_data[field], list):
                        existing_count += len(existing_data[field])
                # 特殊处理 awards：统计统一的 awards 列表
                if 'awards' in existing_data and isinstance(existing_data['awards'], list):
                    existing_count += len(existing_data['awards'])
            
            # 统计新数据的项目数
            if isinstance(new_data, dict):
                for field in list_fields:
                    if field in new_data and isinstance(new_data[field], list):
                        new_count += len(new_data[field])
                # 特殊处理 awards：统计统一的 awards 列表
                if 'awards' in new_data and isinstance(new_data['awards'], list):
                    new_count += len(new_data['awards'])
            
            # 如果新数据有更多项目，认为更好
            if new_count > existing_count:
                return True
            elif new_count < existing_count:
                return False
            # 如果数量相同，比较内容的详细程度（字符串长度）
            else:
                existing_str_len = len(str(existing_data))
                new_str_len = len(str(new_data))
                return new_str_len > existing_str_len * 1.2  # 新数据至少多20%内容
        
        # 字符串型字段：比较字符串长度和详细程度
        else:
            existing_str = str(existing_data)
            new_str = str(new_data)
            
            existing_len = len(existing_str)
            new_len = len(new_str)
            
            # 如果新数据明显更长（至少多30%），认为更好
            if new_len > existing_len * 1.3:
                return True
            # 如果新数据更短，但现有数据很短（少于100字符），也可能替换
            elif existing_len < 100 and new_len > existing_len * 1.1:
                return True
            else:
                return False
    
    def _merge_dimension_data(self, merged: Dict, dim_name: str, new_data: Dict):
        """
        合并维度数据（补充模式，去重）
        
        对于列表型字段（publications, awards等），将新数据追加到已有数据（去重）
        对于字符串字段，如果新数据有内容且现有数据为空，则使用新数据；否则保留现有数据
        """
        if not merged[dim_name] or not new_data:
            # 如果现有数据为空，使用新数据
            if not merged[dim_name] and new_data:
                merged[dim_name] = new_data
            return
        
        existing = merged[dim_name]
        
        # 列表型字段：追加去重
        list_fields = ['publications', 'awards', 'education', 'experiences', 'teaching', 'interests', 'services', 'invited_talks']
        
        for field in list_fields:
            if field in existing and field in new_data:
                # 获取现有数据
                existing_items = existing[field]
                new_items = new_data[field]
                
                if isinstance(existing_items, list) and isinstance(new_items, list):
                    def _make_key(field_name: str, item: Dict[str, Any]) -> str:
                        if not isinstance(item, dict):
                            return ''
                        if field_name == 'services':
                            role = item.get('role', '') or ''
                            venue = item.get('venue', '') or ''
                            year = item.get('year', '') or ''
                            # 如果 role、venue、year 都为空，检查是否有 description
                            if not role and not venue:
                                description = item.get('description', '') or ''
                                if description:
                                    # 使用 description 的前50字符作为 key（避免过长）
                                    return f"desc|{description.strip().lower()[:50]}"
                                # 如果连 description 都没有，返回空字符串（会被跳过）
                                return ''
                            # 正常情况：使用 role|venue|year 作为 key
                            return f"{role.strip().lower()}|{venue.strip().lower()}|{str(year).strip().lower()}"
                        if field_name == 'invited_talks':
                            title = (item.get('title') or '').strip().lower()
                            # 🆕 如果 title 为空，使用 venue 作为 fallback
                            if not title:
                                venue = (item.get('venue') or '').strip().lower()
                                if venue:
                                    return f"venue|{venue}"
                            return title
                        return (item.get('title') or item.get('name') or item.get('course_name') or '').strip().lower()

                    existing_keys = set()
                    for item in existing_items:
                        key = _make_key(field, item)
                        if key:
                            existing_keys.add(key)
                    
                    added_count = 0
                    skipped_count = 0
                    for item in new_items:
                        key = _make_key(field, item)
                        if not key:
                            skipped_count += 1
                            continue
                        if key not in existing_keys:
                            existing_items.append(item)
                            existing_keys.add(key)
                            added_count += 1
                    
                    if skipped_count > 0:
                        print(f"[Multi-Source]   ⚠️ Skipped {skipped_count} {field} items (missing required fields)")
                    
                    if added_count > 0:
                        print(f"[Multi-Source]   ✅ Merged {added_count} new items into {dim_name}.{field} (deduplicated)")
                        # 打印合并后的维度内容摘要
                        if field == 'services' or field == 'invited_talks':
                            _print_dimension_content(dim_name, existing, f"after merge")
            elif field in new_data and field not in existing:
                # 如果新数据有该字段但现有数据没有，直接添加
                existing[field] = new_data[field]
        
        # 特殊处理 awards 维度的统一 awards 列表
        if dim_name == 'awards':
            if 'awards' in new_data:
                if 'awards' not in existing:
                    existing['awards'] = []
                
                existing_items = existing['awards']
                new_items = new_data['awards']
                if isinstance(existing_items, list) and isinstance(new_items, list):
                    def _make_award_key(item: Dict[str, Any]) -> str:
                        """根据 award 类型生成去重 key"""
                        if not isinstance(item, dict):
                            return ''
                        award_type = (item.get('award_type') or 'personal').lower()
                        name = (item.get('name') or '').strip().lower()
                        
                        if award_type == 'paper':
                            # Paper awards: use paper_title + venue as key
                            title = (item.get('paper_title') or '').strip().lower()
                            venue = (item.get('venue') or '').strip().lower()
                            if not title or not venue:
                                return ''
                            return f"paper|{title}|{venue}"
                        elif award_type == 'project':
                            # Project awards: use project_name + name as key
                            project_name = (item.get('project_name') or '').strip().lower()
                            if not project_name or not name:
                                return ''
                            return f"project|{project_name}|{name}"
                        else:
                            # Personal awards: use name + year as key
                            year = str(item.get('year') or '').strip().lower()
                            if not name:
                                return ''
                            return f"personal|{name}|{year}"
                    
                    existing_keys = set()
                    for item in existing_items:
                        key = _make_award_key(item)
                        if key:
                            existing_keys.add(key)
                    
                    added_count = 0
                    skipped_count = 0
                    for item in new_items:
                        key = _make_award_key(item)
                        if not key:
                            skipped_count += 1
                            continue
                        if key not in existing_keys:
                            existing_items.append(item)
                            existing_keys.add(key)
                            added_count += 1
                    
                    if skipped_count > 0:
                        print(f"[Multi-Source]   ⚠️ Skipped {skipped_count} awards items (missing required fields)")
                    
                    if added_count > 0:
                        print(f"[Multi-Source]   ✅ Merged {added_count} new items into awards (deduplicated)")
        
        # 对于非列表字段，如果现有数据为空或很短，使用新数据
        for key, value in new_data.items():
            if key not in list_fields:
                if key not in existing or not existing[key] or (isinstance(existing[key], str) and len(existing[key]) < 50):
                    if value and (not isinstance(value, str) or len(value) >= 50):
                        existing[key] = value


def extract_nine_dimensions_multi_source(
    author_name: str,
    profile: Any,
    openreview_data: Dict[str, Any],
    api_key: str = None
) -> Dict[str, Any]:
    """
    主入口：多源九维度提取
    
    Args:
        author_name: 作者姓名
        profile: AuthorProfile（包含各源的原始文本）
        openreview_data: OpenReview 数据
        api_key: LLM API key
        
    Returns:
        Dict with 9-dimension data ready for EnhancedAuthorProfile
    """
    collector = MultiSourceNineDimensionCollector(api_key=api_key)
    
    # 执行多源采集
    merged_dims = collector.collect_from_sources(
        author_name=author_name,
        profile=profile,
        openreview_data=openreview_data
    )
    
    return merged_dims

