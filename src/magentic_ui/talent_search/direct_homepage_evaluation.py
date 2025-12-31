"""
Direct homepage evaluation utilities (Updated to use 9-field EnhancedAuthorProfile).
Given a personal homepage URL, build an EnhancedAuthorProfile using the
9-field structure and return a frontend-ready payload.
"""
from __future__ import annotations

from typing import Dict, Any, List, Callable, Optional
import warnings
import requests
import time
import re

# Suppress Streamlit ScriptRunContext warnings in ThreadPoolExecutor
warnings.filterwarnings('ignore', message='.*ScriptRunContext.*')
warnings.filterwarnings('ignore', message='.*missing script run context.*')

import json

# Reuse the rich pipeline utilities already implemented
from . import author_discovery as ad
from . import schemas
from . import config
from . import llm
# ------------------- Input Analysis and Dispatching Helpers -------------------

def _is_url(text: str) -> bool:
    """Checks if a string is a valid-looking URL."""
    if not text:
        return False
    # Simple regex to catch http/https URLs. Will be normalized later.
    return re.match(r'^https?://', text.strip(), re.IGNORECASE) is not None

def _extract_name_from_text(text: str, api_key: str | None = None) -> Optional[str]:
    """Uses an LLM to extract a candidate's name from a block of text."""
    if not text or len(text) < 10:
        return None

    print(f"[Name Extractor] Attempting to extract name from text (length: {len(text)})...")
    try:
        llm_client = llm.get_llm("extract", temperature=0.0, api_key=api_key)
        prompt = f"""From the following resume or text, extract the full name of the candidate. 
        Return only the name and nothing else. If no name is found, return an empty string.

        Text: {text[:4000]}

        Candidate's Full Name:"""
        
        # Use invoke() instead of generate() - AzureChatOpenAI uses invoke()
        response = llm_client.invoke(prompt)
        
        # Extract content from response object
        if response:
            # Handle different response types
            if hasattr(response, 'content'):
                raw_name = response.content.strip()
            elif isinstance(response, str):
                raw_name = response.strip()
            else:
                raw_name = str(response).strip()
            
            if raw_name:
                # Validate the extracted name to ensure it's a plausible name
                cleaned_name = _clean_and_validate_author_name(raw_name)
                if cleaned_name:
                    print(f"[Name Extractor] Extracted and validated name: '{raw_name}' -> '{cleaned_name}'")
                    return cleaned_name
                else:
                    print(f"[Name Extractor] Extracted name '{raw_name}' failed validation.")
                    return None
        return None
    except Exception as e:
        print(f"[Name Extractor] Error during name extraction: {e}")
        import traceback
        traceback.print_exc()
        return None


def _clean_and_validate_author_name(name: str) -> Optional[str]:
    """
    清理和验证作者姓名，确保格式正确可用于 OpenReview 搜索
    
    Args:
        name: 原始姓名
        
    Returns:
        清理后的姓名，如果无效则返回 None
    """
    if not name or not isinstance(name, str):
        return None
    
    # 1. 基本清理
    name = name.strip()
    
    # 2. 移除常见的无效值
    invalid_values = ['unknown', 'n/a', 'na', 'none', '', 'null']
    if name.lower() in invalid_values:
        return None
    
    # 3. 移除特殊字符和标点（保留空格、连字符、点号）
    # 保留字母、空格、连字符、点号、撇号（用于 O'Brien 等）
    name = re.sub(r'[^a-zA-Z\s\-\.\']', '', name)
    name = re.sub(r'\s+', ' ', name)  # 合并多个空格
    name = name.strip()
    
    # 4. 验证长度（至少3个字符，最多100个字符）
    if len(name) < 3 or len(name) > 100:
        return None
    
    # 5. 验证至少包含两个词（first name + last name）
    name_parts = name.split()
    if len(name_parts) < 2:
        return None
    
    # 6. 验证每个部分至少2个字符（避免单个字母）
    if any(len(part) < 2 for part in name_parts):
        return None
    
    # 7. 移除常见的标题和前缀
    prefixes = ['dr.', 'dr', 'prof.', 'prof', 'professor', 'mr.', 'mr', 'mrs.', 'mrs', 'ms.', 'ms']
    first_part_lower = name_parts[0].lower()
    if first_part_lower in prefixes:
        name_parts = name_parts[1:]
        if len(name_parts) < 2:
            return None
    
    # 8. 移除常见的后缀
    suffixes = ['jr.', 'jr', 'sr.', 'sr', 'ii', 'iii', 'iv', 'phd', 'ph.d.']
    last_part_lower = name_parts[-1].lower()
    if last_part_lower in suffixes:
        name_parts = name_parts[:-1]
        if len(name_parts) < 2:
            return None
    
    # 9. 重新组合（保留最多4个部分，避免过长的名字）
    cleaned_name = ' '.join(name_parts[:4])
    
    # 10. 最终验证：至少包含 first name 和 last name
    if len(cleaned_name.split()) < 2:
        return None
    
    return cleaned_name


def _extract_name_from_title(title: str) -> Optional[str]:
    """
    从页面标题中提取作者姓名
    Args:
        title: 页面标题
        
    Returns:
        提取的作者姓名，如果无法提取则返回 None
    """
    if not title or not isinstance(title, str):
        return None
    
    title = title.strip()
    if not title:
        return None
    
    # 移除常见的后缀
    suffixes = [
        "'s website", "'s Website", "'s Homepage", "'s Page",
        " - Homepage", " - Website", " - Personal Page",
        " | Homepage", " | Website", " | Personal Page",
        " Homepage", " Website", " Personal Page",
        " - Research", " | Research", " Research"
    ]
    
    for suffix in suffixes:
        if title.lower().endswith(suffix.lower()):
            title = title[:-len(suffix)].strip()
            break
    
    # 移除常见的前缀
    prefixes = [
        "Welcome to ", "Homepage of ", "Personal Page of ",
        "Research Page of ", "Website of "
    ]
    
    for prefix in prefixes:
        if title.lower().startswith(prefix.lower()):
            title = title[len(prefix):].strip()
            break
    
    # 移除标题和前缀（Dr., Prof., etc.）
    title_parts = title.split()
    if len(title_parts) > 0:
        first_part = title_parts[0].lower()
        prefixes_to_remove = ['dr.', 'dr', 'prof.', 'prof', 'professor', 'mr.', 'mr', 'mrs.', 'mrs', 'ms.', 'ms']
        if first_part in prefixes_to_remove:
            title_parts = title_parts[1:]
            title = ' '.join(title_parts)
    
    # 清理和验证
    cleaned_name = _clean_and_validate_author_name(title)
    return cleaned_name


def _identify_candidates_from_name(name: str, api_key: str = None) -> List[Dict[str, Any]]:
    """Performs a lightweight search to find potential candidate matches."""
    print(f"[Identifier] Searching for candidates matching: {name}")
    # For now, we use the existing search_openreview_profile which returns one best match.
    # In a future version, this could be expanded to search multiple sources or return multiple results.
    profile = _try_search_openreview_with_variants(name, api_key=api_key)
    if not profile:
        return []

    # Format the profile into a candidate card structure
    candidates = []
    try:
        main_name_info = profile.get("names", [{}])[0]
        full_name = f"{main_name_info.get('first', '')} {main_name_info.get('last', '')}".strip()

        # Find current affiliation
        affiliation = "N/A"
        if profile.get("education"): 
            # OpenReview's history/education is not always sorted
            latest_entry = max(profile["education"], key=lambda x: x.get('end') or '0', default=None)
            if latest_entry:
                affiliation = latest_entry.get('institution', 'N/A')

        candidate_card = {
            "id": profile.get("openreview_id"),
            "name": full_name,
            "affiliation": affiliation,
            "homepage_url": profile.get("personal_links", {}).get("homepage"),
            "openreview_profile_url": profile.get("openreview_url"),
            "source": "OpenReview"
        }
        candidates.append(candidate_card)
    except Exception as e:
        print(f"[Identifier] Error formatting candidate card: {e}")

    return candidates

def _try_search_openreview_with_variants(author_name: str, api_key: str = None) -> Optional[Dict[str, Any]]:
    """
    尝试使用多种 name 变体搜索 OpenReview
    
    Args:
        author_name: 作者姓名
        api_key: API key
        
    Returns:
        OpenReview 数据，如果都失败则返回 None
    """
    # 变体1：原始姓名
    variants = [author_name]
    
    # 变体2：如果包含中间名，尝试只使用 first + last
    name_parts = author_name.split()
    if len(name_parts) >= 3:
        # First + Last
        variants.append(f"{name_parts[0]} {name_parts[-1]}")
    
    # 变体3：如果名字很长，尝试缩写中间名
    if len(name_parts) >= 3:
        # First + Middle Initial + Last
        middle_initials = '. '.join([part[0] for part in name_parts[1:-1]]) + '.'
        variants.append(f"{name_parts[0]} {middle_initials} {name_parts[-1]}")
    
    # 去重
    variants = list(dict.fromkeys(variants))
    
    print(f"[Direct Homepage] 🔍 Trying OpenReview search with {len(variants)} name variants:")
    for i, variant in enumerate(variants, 1):
        print(f"  [{i}] {variant}")
        result = ad.search_openreview_profile(variant, api_key=api_key)
        if result:
            print(f"[Direct Homepage] ✅ Found OpenReview profile with variant '{variant}'")
            return result
    
    print(f"[Direct Homepage] ❌ No OpenReview profile found with any variant")
    return None

def _fetch_openreview_by_id(profile_id: str, api_key: str = None, max_retries: int = 3) -> Optional[Dict[str, Any]]:
    """
    Directly fetch OpenReview profile by ID (more accurate than search by name)
    
    Args:
        profile_id: OpenReview profile ID (e.g., ~Ziming_Liu2)
        api_key: API key (unused for OpenReview, kept for consistency)
        max_retries: Maximum retry attempts for rate limiting
        
    Returns:
        Dict with structured profile data, same format as search_openreview_profile
    """
    import random
    
    for attempt in range(max_retries):
        try:
            # Use OpenReview API to get profile directly by ID
            profile_url = f"https://api2.openreview.net/profiles"
            profile_params = {'id': profile_id}
            
            time.sleep(1.0 + random.uniform(0, 0.5))  # Rate limiting: 1-1.5s
            response = requests.get(
                profile_url,
                params=profile_params,
                timeout=30,
                headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
            )
            
            if response.status_code == 429:
                # Rate limited - exponential backoff
                wait_time = (attempt + 1) * 3 + random.uniform(0, 2)
                if attempt < max_retries - 1:
                    time.sleep(wait_time)
                    continue
                else:
                    return None
            
            if response.status_code != 200:
                return None
            
            data = response.json()
            if 'profiles' not in data or not data['profiles']:
                return None
            
            profile = data['profiles'][0]
            content = profile.get('content', {})
            
            # Build structured data (same format as search_openreview_profile)
            openreview_url = f"https://openreview.net/profile?id={profile_id}"
            
            # Extract names
            names_list = []
            for name in content.get('names', []):
                names_list.append({
                    'first': name.get('first', ''),
                    'middle': name.get('middle', ''),
                    'last': name.get('last', ''),
                    'username': name.get('username', ''),
                    'preferred': name.get('preferred', False)
                })
            
            # Extract personal links
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
            
            # Extract education
            education_list = []
            education_data = content.get('education') or content.get('history') or profile.get('education') or profile.get('history') or []
            if not isinstance(education_data, list):
                education_data = []
            for item in education_data:
                institution_name = ''
                if isinstance(item.get('institution'), dict):
                    institution_name = item.get('institution', {}).get('name', '')
                elif isinstance(item.get('institution'), str):
                    institution_name = item.get('institution', '')
                
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
            
            # Extract relations
            relations_list = []
            for relation in content.get('relations', []):
                relations_list.append({
                    'name': relation.get('name', ''),
                    'email': relation.get('email', ''),
                    'relation': relation.get('relation', ''),
                    'start': relation.get('start', ''),
                    'end': relation.get('end', '')
                })
            
            # Fetch publications
            publications_list = []
            all_author_ids = [profile_id]
            
            for name in names_list:
                username = name.get('username', '')
                if username and username not in all_author_ids:
                    all_author_ids.append(username)
            
            seen_ids = set()
            seen_titles = set()
            
            for author_id in all_author_ids[:3]:
                try:
                    time.sleep(1.0)  # Rate limiting
                    notes_url = "https://api2.openreview.net/notes"
                    notes_params = {
                        'content.authorids': author_id,
                        'details': 'replyCount,invitation,original',
                        'limit': 100,
                        'offset': 0
                    }
                    
                    notes_response = requests.get(
                        notes_url,
                        params=notes_params,
                        timeout=30, 
                        headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
                    )
                    
                    if notes_response.status_code == 200:
                        notes_data = notes_response.json()
                        notes = notes_data.get('notes', [])
                        
                        for note in notes:
                            note_id = note.get('id')
                            if note_id and note_id not in seen_ids:
                                note_content = note.get('content', {})
                                
                                def get_value(data, default=''):
                                    if isinstance(data, dict):
                                        return data.get('value', default)
                                    return data if data is not None else default
                                
                                title = get_value(note_content.get('title'), '')
                                title_normalized = re.sub(r'[^\w\s]', '', title.lower()).strip()
                                title_normalized = re.sub(r'\s+', ' ', title_normalized)
                                
                                if title_normalized in seen_titles:
                                    continue
                                
                                seen_ids.add(note_id)
                                seen_titles.add(title_normalized)
                                
                                pub_info = {
                                    'id': note_id,
                                    'title': title,
                                    'authors': get_value(note_content.get('authors'), []),
                                    'abstract': get_value(note_content.get('abstract'), ''),
                                    'venue': get_value(note_content.get('venue'), ''),
                                    'year': note.get('pdate', 0) // 1000 // 86400 // 365 + 1970 if note.get('pdate') else None,
                                    'url': f"https://openreview.net/forum?id={note_id}"
                                }
                                publications_list.append(pub_info)
                except Exception:
                    pass
            
            return {
                'openreview_url': openreview_url,
                'openreview_id': profile_id,
                'names': names_list,
                'personal_links': personal_links,
                'education': education_list,
                'relations': relations_list,
                'publications': publications_list,
                'coauthors': {}
            }
        
        except Exception:
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    
    return None


def _to_frontend_payload_from_enhanced(enhanced_profile: schemas.EnhancedAuthorProfile, eval_result: Optional[schemas.EvaluationResult] = None) -> Dict[str, Any]:
    """Convert EnhancedAuthorProfile (9-field structure) to frontend payload.
    
    Args:
        enhanced_profile: EnhancedAuthorProfile object
        eval_result: Optional evaluation result (for scores)
        
    Returns:
        Dict suitable for frontend.candidate_profile.render
    """
    # Extract introduction
    intro = enhanced_profile.introduction
    
    # Extract research interests
    research_keywords = [ri.name for ri in enhanced_profile.research_interests[:6]]
    research_focus = [ri.name for ri in enhanced_profile.research_interests[:8]]
    
    # Extract contact info
    contact = enhanced_profile.contact
    profiles = {}
    if contact.homepage:
        profiles["Homepage"] = contact.homepage
    if contact.google_scholar:
        profiles["Google Scholar"] = contact.google_scholar
    if contact.github:
        profiles["GitHub"] = contact.github
    if contact.linkedin:
        profiles["LinkedIn"] = contact.linkedin
    if contact.orcid:
        profiles["ORCID"] = contact.orcid
    if contact.dblp:
        profiles["DBLP"] = contact.dblp
    if contact.openreview:
        profiles["OpenReview"] = contact.openreview
    if contact.twitter:
        profiles["Twitter/X"] = contact.twitter
    
    # Extract publications (get representative ones from selected_research)
    publication_titles = []
    top_tier_hits = []
    representative_papers = []
    
    for category_name, papers_list in enhanced_profile.selected_research.by_category.items():
        for pub in papers_list[:2]:  # Take top 2 from each category
            if pub.title not in [p["title"] for p in representative_papers]:
                representative_papers.append({
                    "title": pub.title,
                    "venue": pub.venue,
                    "year": pub.year,
                    "type": pub.note or "",  # Use note field or empty string
                    "links": pub.url
                })
                publication_titles.append(pub.title)
                if pub.venue:
                    top_tier_hits.append(f"{pub.venue} {pub.year or ''}")
    
    # Limit to 5 items
    publication_titles = publication_titles[:5]
    top_tier_hits = top_tier_hits[:5]
    representative_papers = representative_papers[:3]
    
    # Extract honors/grants
    honors_grants = [
        f"{award.name} ({award.year or 'N/A'}) - {award.organization or ''}"
        for award in enhanced_profile.awards[:5]
    ]
    
    # Extract service/talks
    service_talks = []
    for svc in enhanced_profile.professional_services[:10]:
        service_talks.append(f"{svc.role} @ {svc.venue} ({svc.year or 'N/A'})")
    
    # Extract open source projects (from teaching field as a fallback or we could add a separate field)
    # For now, we'll leave it empty or generate from contact.github
    open_source_projects = []
    if contact.github:
        open_source_projects = [f"See GitHub: {contact.github}"]
    
    # Extract highlights (from awards for now)
    highlights = [
        f"{award.name} ({award.year or 'N/A'})"
        for award in enhanced_profile.awards[:8]
    ]
    
    # Build current role & affiliation (已经由 role_determination 在 build_enhanced_profile 中填充好了)
    current_role_affiliation = f"{intro.position} at {intro.affiliation}" if intro.position and intro.affiliation else intro.affiliation or intro.position or ""
    
    # Evaluation scores
    total_score = 0
    detailed_scores = {}
    radar = {}
    if eval_result:
        total_score = eval_result.total_score
        detailed_scores = eval_result.details
        radar = eval_result.radar
    
    # Build enhanced_profile structure (serialize Pydantic models to dict)
    enhanced_profile_dict = {
        "introduction": {
            "name": intro.name,
            "current_position": intro.position,
            "affiliation": intro.affiliation,
            "text": intro.text
        },
        "current_role": {
            "category": enhanced_profile.current_role.category,
            "role_text": enhanced_profile.current_role.role_text,
            "affiliation": enhanced_profile.current_role.affiliation,
            "explanation": enhanced_profile.current_role.explanation
        },
        "research_interests": [
            {"name": ri.name, "description": ri.description or ""}
            for ri in enhanced_profile.research_interests
        ],
        "selected_research": {
            "by_category": {
                cat: [
                    {
                        "title": pub.title,
                        "venue": pub.venue,
                        "year": pub.year,
                        "url": pub.url,
                        "note": pub.note or ""
                    }
                    for pub in pubs
                ]
                for cat, pubs in enhanced_profile.selected_research.by_category.items()
            }
        },
        "awards": [
            {
                "name": award.name,
                "year": award.year,
                "organization": award.organization or ""
            }
            for award in enhanced_profile.awards
        ],
        "professional_services": [
            {
                "role": svc.role,
                "venue": svc.venue,
                "year": svc.year
            }
            for svc in enhanced_profile.professional_services
        ],
        "career_education_history": [
            {
                "degree_or_position": item.degree_or_position,
                "institution": item.institution,
                "department": item.department or "",
                "duration": item.duration or "",
                "field": item.field or "",
                "advisor": item.advisor or "",
                "description": item.description or ""
            }
            for item in enhanced_profile.career_education_history
        ],
        "industrial_experience": [
            {
                "position": exp.position,
                "organization": exp.organization,
                "duration": exp.duration or ""
            }
            for exp in enhanced_profile.industrial_experience
        ],
        "contact": {
            "email": contact.email or "",
            "homepage": contact.homepage or "",
            "google_scholar": contact.google_scholar or "",
            "github": contact.github or "",
            "linkedin": contact.linkedin or "",
            "twitter": contact.twitter or "",
            "orcid": contact.orcid or "",
            "dblp": contact.dblp or "",
            "openreview": contact.openreview or "",
            "phone": contact.phone or "",
            # Build social_links dict from individual fields for backward compatibility
            "social_links": {
                k: v for k, v in {
                    "homepage": contact.homepage,
                    "scholar": contact.google_scholar,
                    "github": contact.github,
                    "twitter": contact.twitter,
                    "linkedin": contact.linkedin,
                    "orcid": contact.orcid,
                    "dblp": contact.dblp,
                    "openreview": contact.openreview
                }.items() if v
            }
        }
    }
    
    payload = {
        "name": intro.name,
        "email": contact.email,
        "current_role_affiliation": current_role_affiliation,
        "current_status": intro.text,
        "research_keywords": research_keywords,
        "research_focus": research_focus,
        "profiles": profiles,
        "publication_overview": publication_titles,
        "top_tier_hits": top_tier_hits,
        "honors_grants": honors_grants,
        "service_talks": service_talks,
        "open_source_projects": open_source_projects,
        "representative_papers": representative_papers,
        "highlights": highlights,
        "radar": radar,
        "total_score": total_score,
        "detailed_scores": detailed_scores,
        "enhanced_profile": enhanced_profile_dict,  # ← Add the complete structure
    }
    return payload


def evaluate_candidate_input(
    user_input: str,
    api_key: str | None = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Universal entry point for evaluating a candidate from various inputs."""
    def _progress(event: str, pct: float) -> None:
        if on_progress is not None:
            on_progress(event, pct)

    _progress("analyzing_input", 0.01)

    # Path A: Input is a URL, proceed with direct deep evaluation
    if _is_url(user_input):
        print(f"[Dispatcher] Input detected as URL: {user_input}")
        return _evaluate_url_to_candidate_overview(
            homepage_url=user_input,
            author_hint="",
            api_key=api_key,
            on_progress=on_progress
        )

    # Path B: Input is free-form text, perform content-enhanced evaluation
    _progress("identifying_candidate", 0.1)
    print("[Dispatcher] Input detected as free-form text. Starting content-enhanced evaluation.")

    # 1. Extract a name from the text
    extracted_name = _extract_name_from_text(user_input, api_key=api_key)
    # If no name can be extracted, treat the whole input as a name query
    candidate_name = extracted_name or user_input
    validated_name = _clean_and_validate_author_name(candidate_name)

    if not validated_name:
        print(f"[Dispatcher] Could not identify a valid name from input: '{user_input}'")
        return {
            "error": "Could not identify a candidate from the provided text. Please try providing a full name or a URL.",
        }

    # 2. Proceed with name-based evaluation, passing the original text as extra content for profile enhancement
    print(f"[Dispatcher] Name identified as '{validated_name}'. Proceeding with evaluation, using original text as enhancement.")
    return _evaluate_name_to_candidate_overview(
        author_name=validated_name,
        extra_content=user_input, # Pass the full original text
        api_key=api_key,
        on_progress=on_progress
    )

def _evaluate_name_to_candidate_overview(
    author_name: str,
    extra_content: str = "",
    api_key: str | None = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Runs a targeted search pipeline starting only with a candidate's name."""
    def _progress(event: str, pct: float) -> None:
        if on_progress is not None:
            on_progress(event, pct)

    _progress("targeted_search", 0.10)
    print(f"[_evaluate_name] Starting targeted search for: {author_name}")

    try:
        discovered_profile = ad.discover_author_profile(
            first_author=author_name,
            paper_title="",
            aliases=[],
            k_queries=40,
            author_id=None,
            api_key=api_key,
            openreview_result=None
        )

        if not discovered_profile:
            raise RuntimeError(f"Could not find a public profile for '{author_name}'. Please try a more specific name or provide a URL.")

        # Attach the user-provided text to the profile for enhancement
        if extra_content:
            print(f"[_evaluate_name] Attaching {len(extra_content)} chars of extra content for profile enhancement.")
            # We store this in a new attribute that `build_enhanced_profile` will use.
            discovered_profile._user_provided_text = extra_content

        _progress("building_enhanced_profile", 0.40)
        print(f"[_evaluate_name] Building enhanced profile for: {discovered_profile.name}")

        openreview_data = getattr(discovered_profile, '_openreview_data', None)
        if not openreview_data:
            openreview_data = {"names": [{"first": author_name.split()[0], "last": author_name.split()[-1] if ' ' in author_name else ''}]}

        enhanced_profile = ad.build_enhanced_profile(
            openreview_data=openreview_data,
            author_name=discovered_profile.name,
            api_key=api_key,
            profile=discovered_profile,
            user_query=None
        )

        _progress("profile_built", 0.85)
        if not enhanced_profile:
            raise RuntimeError("Failed to build the enhanced profile from the discovered data.")

        _progress("converting_to_frontend", 0.95)
        payload = _to_frontend_payload_from_enhanced(enhanced_profile, eval_result=None)
        _progress("done", 1.0)
        return payload

    except Exception as e:
        print(f"[_evaluate_name] Error during name-based evaluation: {e}")
        return {"error": str(e)}

def _evaluate_url_to_candidate_overview(
    homepage_url: str,
    author_hint: str = "",
    api_key: str | None = None,
    on_progress: Optional[Callable[[str, float], None]] = None,
) -> Dict[str, Any]:
    """Run a homepage-only pipeline using the 9-field EnhancedAuthorProfile structure.
    
    - Fetch homepage (with subpages)
    - Create a mock openreview_data structure
    - Call build_enhanced_profile to generate 9-field structure
    - Return frontend-ready payload
    
    Args:
        homepage_url: Personal homepage URL
        author_hint: Optional author name hint
        api_key: LLM API key
        on_progress: Optional progress callback
        
    Returns:
        Dict suitable for frontend rendering
    """
    # Helper to emit progress safely
    def _progress(event: str, pct: float) -> None:
        """
        安全地发送进度更新，忽略 WebSocket 连接错误
        这些错误通常发生在：
        - 任务运行时间过长，WebSocket 连接超时
        - 前端页面被关闭或刷新
        - 网络连接中断
        这些错误不影响后台任务的执行，只是无法向前端发送进度更新
        """
        try:
            if on_progress is not None:
                on_progress(event, float(max(0.0, min(1.0, pct))))
        except (ConnectionError, OSError, RuntimeError) as e:
            pass
        except Exception as e:
            import os
            if os.getenv("DEBUG", "").lower() == "true":
                print(f"[Progress] ⚠️ Progress callback error (non-fatal): {type(e).__name__}")
            pass

    _progress("fetching_homepage", 0.02)
    # Detect if input is OpenReview URL
    is_openreview_url = "openreview.net" in homepage_url.lower()
    
    # Initialize variables
    openreview_data = None
    actual_homepage_url = None
    result = None
    prefetched_homepage_data: Optional[Dict[str, Any]] = None
    author_query = author_hint  # Will be extracted from URL if OpenReview and no hint
    profile_id: Optional[str] = None  # Keep OpenReview profile id for downstream targeted search
    
    # 1) If OpenReview URL: Use API to get structured data (same as Targeted Search)
    if is_openreview_url:
        print(f"[Direct Homepage] Detected OpenReview URL, using API for structured data")
        _progress("fetching_openreview_api", 0.05)
        try:
            # Extract profile ID directly from URL
            profile_id = None
            profile_id_match = re.search(r'id=([^&]+)', homepage_url)
            if profile_id_match:
                profile_id = profile_id_match.group(1)
                print(f"[Direct Homepage] Extracted profile ID from URL: {profile_id}")
            
            # If we have profile ID, fetch directly by ID (more accurate)
            # Otherwise use author_hint to search
            if profile_id:
                print(f"[Direct Homepage] Fetching OpenReview profile by ID: {profile_id}")
                openreview_data = _fetch_openreview_by_id(profile_id, api_key)
                
                # Extract author name from fetched data
                if openreview_data:
                    names = openreview_data.get('names', [])
                    if names:
                        first = names[0].get('first', '')
                        last = names[0].get('last', '')
                        author_query = f"{first} {last}".strip()
                        print(f"[Direct Homepage] Got author name from API: {author_query}")
                    else:
                        author_query = author_hint or ""
                else:
                    author_query = author_hint or ""
            else:
                # No profile ID in URL, use search by name
                author_query = author_hint or ""
                if author_query:
                    print(f"[Direct Homepage] Searching OpenReview by name: {author_query}")
                    openreview_data = ad.search_openreview_profile(author_query, api_key=api_key)
                else:
                    print(f"[Direct Homepage] No profile ID or author hint provided")
                    openreview_data = None
            
            if openreview_data:
                print(f"[Direct Homepage] ✅ Got structured OpenReview data via API")
                
                # Extract personal homepage from OpenReview data
                personal_links = openreview_data.get('personal_links', {})
                actual_homepage_url = personal_links.get('homepage', '')
                
                if actual_homepage_url:
                    print(f"[Direct Homepage] 📍 Found homepage in OpenReview: {actual_homepage_url}")
                    _progress("fetching_actual_homepage", 0.10)
                    
                    # Fetch the actual homepage content
                    result = ad.fetch_homepage_comprehensive(
                        actual_homepage_url, 
                        author_name=author_query or author_hint or "", 
                        include_subpages=True, 
                        max_subpages=6
                    )
                else:
                    print(f"[Direct Homepage] ⚠️ No homepage found in OpenReview, will use OpenReview page as source")
            else:
                print(f"[Direct Homepage] ⚠️ Failed to get OpenReview data via API, falling back to HTML scraping")
                is_openreview_url = False  # Fall back to HTML scraping
        except Exception as e:
            print(f"[Direct Homepage] ❌ Error fetching OpenReview API: {e}")
            is_openreview_url = False  # Fall back to HTML scraping
    
    # 1.b) For OpenReview URL: ALWAYS run targeted search using the known profile id/name
    if is_openreview_url and openreview_data:
        try:
            # Prefer name from OpenReview data
            if not author_query:
                names = openreview_data.get("names") or []
                if names:
                    first = names[0].get("first", "")
                    last = names[0].get("last", "")
                    author_query = f"{first} {last}".strip()
            # Clean and validate for targeted search
            extracted_author_name = _clean_and_validate_author_name(author_query or author_hint or "")
            use_targeted_search_flow = True
            
            _progress("targeted_search", 0.12)
            print(f"[Direct Homepage] 🎯 Running targeted search with OpenReview id (skip search): {profile_id} / {extracted_author_name}")
            
            discovered_profile = ad.discover_author_profile(
                first_author=extracted_author_name or author_query or author_hint or "",
                paper_title="",
                aliases=[],
                k_queries=40,
                author_id=profile_id,
                api_key=api_key,
                openreview_result=openreview_data  # reuse already-fetched data
            )
            
            if discovered_profile:
                targeted_search_success = True
                discovered_profile_from_targeted_search = discovered_profile
                
                # Ensure homepage is set; fall back to OpenReview homepage or input URL
                if not discovered_profile.homepage_url:
                    discovered_profile.homepage_url = actual_homepage_url or homepage_url
                
                # Keep openreview_data in sync
                if hasattr(discovered_profile, "_openreview_data") and discovered_profile._openreview_data:
                    openreview_data = discovered_profile._openreview_data
                else:
                    discovered_profile._openreview_data = openreview_data
                print(f"[Direct Homepage] Targeted search completed via OpenReview link")
            else:
                print(f"[Direct Homepage] Targeted search returned None for OpenReview link")
        except Exception as e:
            print(f"[Direct Homepage] Targeted search flow (OpenReview URL) failed: {e}")
            import traceback
            traceback.print_exc()
            targeted_search_success = False
    
    # 2) If not OpenReview OR API failed: Fetch URL directly (original behavior)
    # 对于普通网页，我们需要先提取作者姓名，然后立即走 targeted search 流程
    if not 'use_targeted_search_flow' in locals() or use_targeted_search_flow is None:
        use_targeted_search_flow = False
    if not 'extracted_author_name' in locals() or extracted_author_name is None:
        extracted_author_name = None
    final_author_name = None  # 将在后续确定
    if not 'targeted_search_success' in locals():
        targeted_search_success = False  # 标记 targeted search 是否成功
    if not 'discovered_profile_from_targeted_search' in locals():
        discovered_profile_from_targeted_search = None  # 存储从 targeted search 获取的 profile
    
    if not is_openreview_url or (is_openreview_url and not openreview_data):
        result = ad.fetch_homepage_comprehensive(
            homepage_url, 
            author_name=author_hint or "", 
            include_subpages=True, 
            max_subpages=6
        )
        
        # 优先从页面标题中提取作者姓名，如果成功则立即走 targeted search 流程
        if result and result.get("title") and not extracted_author_name:
            title = result.get("title", "")
            title_name = _extract_name_from_title(title)
            if title_name:
                extracted_author_name = title_name
                use_targeted_search_flow = True
                print(f"[Direct Homepage] 📝 Extracted author name from page title: '{title}' → '{title_name}'")
                print(f"[Direct Homepage] 🎯 Immediately starting targeted search flow (will search OpenReview for: {title_name})")
                
                # 立即执行 targeted search 流程（不等待 Agent）
                _progress("targeted_search", 0.10)
                try:
                    # 先尝试直接搜索 OpenReview（使用多种 name 变体）
                    openreview_result = _try_search_openreview_with_variants(extracted_author_name, api_key)
                    if openreview_result:
                        # 如果找到 OpenReview，使用 discover_author_profile
                        # 传递已搜索的 openreview_result，避免重复 API 调用
                        print(f"[Direct Homepage] ✅ Found OpenReview, proceeding with discover_author_profile (passing pre-fetched result)")
                        discovered_profile = ad.discover_author_profile(
                            first_author=extracted_author_name,
                            paper_title="",  # Resume evaluation 没有触发论文
                            aliases=[],
                            k_queries=40,
                            author_id=None,
                            api_key=api_key,
                            openreview_result=openreview_result  # 传递已搜索的结果
                        )
                        
                        if discovered_profile:
                            print(f"[Direct Homepage] ✅ Discovered profile via targeted search")
                            # 保存 discovered profile，后续会使用它替换初始的 profile
                            discovered_profile_from_targeted_search = discovered_profile
                            
                            # 更新 openreview_data（从 discovered profile 中获取）
                            if hasattr(discovered_profile, '_openreview_data') and discovered_profile._openreview_data:
                                openreview_data = discovered_profile._openreview_data
                                print(f"[Direct Homepage] ✅ Using OpenReview data from discovered profile")
                            else:
                                # 如果 discovered profile 没有 OpenReview，使用搜索结果
                                if openreview_result:
                                    openreview_data = openreview_result
                                    discovered_profile._openreview_data = openreview_result
                                    print(f"[Direct Homepage] ✅ Using OpenReview data from search result")
                            
                            # 确保 homepage URL 被正确设置
                            if not discovered_profile.homepage_url or discovered_profile.homepage_url == homepage_url:
                                discovered_profile.homepage_url = homepage_url
                            
                            # 标记：已经通过 targeted search 获取了完整 profile，后续不需要再走普通流程
                            print(f"[Direct Homepage] ✅ Targeted search completed successfully, will skip normal flow")
                            targeted_search_success = True
                        else:
                            print(f"[Direct Homepage] ⚠️ discover_author_profile returned None, will fall back to normal flow")
                            targeted_search_success = False
                    else:
                        print(f"[Direct Homepage] ⚠️ No OpenReview found for '{extracted_author_name}', will fall back to normal flow")
                        targeted_search_success = False
                except Exception as e:
                    print(f"[Direct Homepage] ⚠️ Targeted search flow failed: {e}")
                    import traceback
                    traceback.print_exc()
                    targeted_search_success = False
            else:
                targeted_search_success = False
        else:
            targeted_search_success = False
        # 检测页面是否需要九维度Agent，并强制运行一次以探测交互元素
        try:
            html_snapshot = result.get("full_html", "") if result else ""
            text_snapshot = result.get("text_content", "") if result else ""
            text_snapshot_len = len(text_snapshot or "")
            
            needs_agent = ad.page_requires_interaction(html_snapshot)
            print(
                "[Direct Homepage] 🤖 Running Nine-Dimension Agent for interactive detection "
                f"(regex_flag={needs_agent}, baseline_text_len={text_snapshot_len})"
            )
            from .nine_dimension_aware_agent import fetch_with_nine_dimension_awareness
            
            agent_text, agent_metadata = fetch_with_nine_dimension_awareness(
                url=homepage_url,
                author_name=author_hint or author_query or "",
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
                    "[Direct Homepage] 🎯 Agent detected interactive elements "
                    f"(candidates={interactive_candidates}, queue={interactive_queue_length}, interactions={total_interactions})"
                )
            else:
                print(
                    "[Direct Homepage] ℹ️ Agent detected no interactive elements "
                    f"(regex_flag={needs_agent})"
                )
            
            should_use_agent = bool(agent_text and agent_text.strip() and (needs_agent or has_interactive_elements))
            
            if should_use_agent:
                trimmed_text = agent_text[:30000]
                result = result or {}
                result["success"] = True
                result["text_content"] = trimmed_text
                result.setdefault("full_html", "")
                result["method"] = "nine_dimension_agent"
                if agent_metadata:
                    result["agent_metadata"] = agent_metadata
                    if agent_metadata.get("extracted_dimensions"):
                        print(f"[Direct Homepage] 📊 Agent extracted dimensions: {list(agent_metadata.get('extracted_dimensions').keys())}")
                        result["extracted_dimensions"] = agent_metadata.get("extracted_dimensions")
                extraction_text = agent_metadata.get("extraction_text", "") if agent_metadata else ""
                if extraction_text:
                    result["extraction_text"] = extraction_text
                
                #    确保 metadata 中包含 extracted_dimensions，以便 author_discovery.py 可以读取
                if not result.get("metadata"):
                    result["metadata"] = {}
                if agent_metadata and agent_metadata.get("extracted_dimensions"):
                    result["metadata"]["extracted_dimensions"] = agent_metadata.get("extracted_dimensions")
                    print(f"[Direct Homepage] ✅ Stored extracted_dimensions in result['metadata'] for MultiSourceNineDimensionCollector")
                    if agent_metadata.get("extracted_dimensions_meta"):
                        result["metadata"]["extracted_dimensions_meta"] = agent_metadata.get("extracted_dimensions_meta")
                    
                    #    尝试从 extracted_dimensions 中提取作者姓名
                    extracted_dims = agent_metadata.get("extracted_dimensions", {})
                    if extracted_dims.get("background") and extracted_dims["background"].get("name"):
                        raw_name = extracted_dims["background"]["name"].strip()
                        #    清理和验证 name
                        cleaned_name = _clean_and_validate_author_name(raw_name)
                        if cleaned_name:
                            extracted_author_name = cleaned_name
                            print(f"[Direct Homepage] 📝 Extracted author name from background: {raw_name} → {cleaned_name}")
                            use_targeted_search_flow = True
                        else:
                            print(f"[Direct Homepage] ⚠️ Extracted name '{raw_name}' failed validation")
                
                prefetched_homepage_data = {
                    "text_content": agent_text,
                    "extraction_text": extraction_text,
                    "agent_metadata": agent_metadata,
                    "extracted_dimensions": agent_metadata.get("extracted_dimensions", {}) if agent_metadata else {},
                    "extracted_dimensions_meta": agent_metadata.get("extracted_dimensions_meta", {}) if agent_metadata else {},
                    "fetched_via": "nine_dimension_agent",
                    "has_interactive_elements": has_interactive_elements,
                    "interactive_candidates": interactive_candidates,
                    "interactive_queue_length": interactive_queue_length,
                    "total_interactions": total_interactions,
                    "interactive_diagnostics": diagnostics,
                }
                print(f"[Direct Homepage] ✅ Nine-Dimension Agent fetched {len(agent_text)} chars (trimmed to {len(trimmed_text)})")
            else:
                print(
                    "[Direct Homepage] ℹ️ Retaining traditional parser output "
                    f"(should_use_agent={should_use_agent}, has_interactive={has_interactive_elements})"
                )
                
                #    即使不使用 Agent，也尝试从传统解析结果中提取作者姓名
                if not is_openreview_url and result and result.get("text_content"):
                    try:
                        # 使用简单的 LLM 调用提取作者姓名
                        from . import llm
                        llm_client = llm.get_llm("extract", temperature=0.1, api_key=api_key)
                        homepage_text = result.get("text_content", "")[:5000]  # 使用前5000字符
                        
                        if homepage_text and len(homepage_text) > 100:
                            extract_name_prompt = f"""Extract the author's full name from the following homepage content. Return only the name, nothing else.

Homepage content:
{homepage_text}

Author's full name:"""
                            
                            # Use invoke() instead of generate() - AzureChatOpenAI uses invoke()
                            response = llm_client.invoke(extract_name_prompt)
                            # Extract content from response object
                            if response:
                                if hasattr(response, 'content'):
                                    raw_name = response.content.strip()
                                elif isinstance(response, str):
                                    raw_name = response.strip()
                                else:
                                    raw_name = str(response).strip()
                                
                                if raw_name:
                                    # 清理和验证 name
                                    cleaned_name = _clean_and_validate_author_name(raw_name)
                                    if cleaned_name:
                                        extracted_author_name = cleaned_name
                                        use_targeted_search_flow = True
                                        print(f"[Direct Homepage] Extracted author name from traditional parser: {raw_name} → {cleaned_name}")
                                    else:
                                        print(f"[Direct Homepage] Extracted name '{raw_name}' failed validation")
                    except Exception as e:
                        print(f"[Direct Homepage] Failed to extract name from traditional parser: {e}")
        except Exception as e:
            print(f"[Direct Homepage] Nine-Dimension Agent upgrade failed: {e}")
    
    _progress("fetched_homepage", 0.15)
    
    # Check if fetch was successful
    if result and not result.get("success"):
        return {
            "error": "Failed to fetch homepage content",
            "name": author_hint or "Unknown",
            "email": "",
            "current_role_affiliation": "",
            "current_status": "",
            "research_keywords": [],
            "research_focus": [],
            "profiles": {"Homepage": homepage_url},
            "publication_overview": [],
            "top_tier_hits": [],
            "honors_grants": [],
            "service_talks": [],
            "open_source_projects": [],
            "representative_papers": [],
            "highlights": [],
            "radar": {},
            "total_score": 0,
            "detailed_scores": {},
        }
    
    # 2) Create AuthorProfile and store multi-source data
    # Use extracted author name if available
    profile_name = author_query or author_hint or ""
    
    profile = ad.AuthorProfile(
        name=profile_name,
        aliases=[], 
        platforms={}, 
        ids={}, 
        homepage_url=actual_homepage_url or homepage_url,  # Use actual homepage if found
        affiliation_current=None, 
        emails=[], 
        interests=[], 
        selected_publications=[],
        confidence=0.5, 
        notable_achievements=[], 
        social_impact=None, 
        career_stage=None, 
        overall_score=0.0
    )
    
    if prefetched_homepage_data:
        profile._prefetched_homepage_data = prefetched_homepage_data
    
    # Extract social links from fetched content
    html_social_links = result.get("social_platforms", {}) if result else {}
    html_emails = result.get("emails", []) if result else []
    
    for platform, url in html_social_links.items():
        if platform not in profile.platforms:
            profile.platforms[platform] = url
    
    # Add emails
    for email in html_emails:
        if email not in profile.emails:
            profile.emails.append(email)
    
    # ✅ Store multi-source raw text for 9-dimension extraction (same as Targeted Search)
    profile._homepage_raw_text = ""
    profile._openreview_raw_text = ""
    profile._scholar_raw_text = ""
    profile._orcid_raw_text = ""
    profile._dblp_raw_text = ""
    
    # Store homepage text if fetched
    if result:
        raw_text = result.get("text_content") or result.get("text") or ""
        profile._homepage_raw_text = raw_text
        print(f"[Direct Homepage] ✅ Stored homepage text: {len(profile._homepage_raw_text)} chars")
        
        #    存储 metadata（包含 extracted_dimensions）到 profile._homepage_metadata
        if result.get("metadata"):
            profile._homepage_metadata = result.get("metadata")
            if result.get("metadata", {}).get("extracted_dimensions"):
                print(f"[Direct Homepage] ✅ Stored _homepage_metadata with {len(result['metadata']['extracted_dimensions'])} extracted dimensions")
            if result.get("metadata", {}).get("extracted_dimensions_meta"):
                profile._homepage_metadata.setdefault("extracted_dimensions_meta", result["metadata"]["extracted_dimensions_meta"])
        elif result.get("extracted_dimensions"):
            # 兼容性：如果没有 metadata，但有 extracted_dimensions，创建一个 metadata 对象
            profile._homepage_metadata = {
                "extracted_dimensions": result.get("extracted_dimensions"),
                "extracted_dimensions_meta": result.get("extracted_dimensions_meta", {})
            }
            print(f"[Direct Homepage] ✅ Stored _homepage_metadata with {len(result['extracted_dimensions'])} extracted dimensions (from result['extracted_dimensions'])")
    
    # Store OpenReview data if obtained via API
    if openreview_data:
        profile._openreview_data = openreview_data  # Store for build_enhanced_profile
        
        # ✅ Convert OpenReview structured data to text (same as Targeted Search)
        openreview_text_parts = []
        openreview_text_parts.append(f"=== OpenReview Profile ===")
        openreview_text_parts.append(f"Name: {profile_name}")
        openreview_text_parts.append(f"OpenReview URL: {openreview_data.get('openreview_url', '')}")
        
        # Add personal links
        personal_links = openreview_data.get('personal_links', {})
        if personal_links:
            openreview_text_parts.append(f"\nPersonal Links:")
            for link_type, link_url in personal_links.items():
                openreview_text_parts.append(f"  - {link_type}: {link_url}")
        
        # Add education
        education_list = openreview_data.get('education', [])
        if education_list:
            openreview_text_parts.append(f"\nEducation:")
            for edu in education_list:
                openreview_text_parts.append(f"  - {edu.get('position', '')} at {edu.get('institution', '')} ({edu.get('start', '')}-{edu.get('end', 'Present')})")
        
        # Add publications
        publications = openreview_data.get('publications', [])
        if publications:
            openreview_text_parts.append(f"\nPublications ({len(publications)}):")
            for pub in publications[:20]:  # Max 20 papers
                openreview_text_parts.append(f"  - {pub.get('title', '')} ({pub.get('venue', '')}, {pub.get('year', '')})")
        
        # Add relations
        relations = openreview_data.get('relations', [])
        if relations:
            openreview_text_parts.append(f"\nRelations:")
            for rel in relations[:10]:  # Max 10 relations
                openreview_text_parts.append(f"  - {rel.get('name', '')} ({rel.get('relation', '')})")
        
        # Store as text for nine-dimension extraction
        profile._openreview_raw_text = '\n'.join(openreview_text_parts)
        print(f"[Direct Homepage] ✅ Stored OpenReview text: {len(profile._openreview_raw_text)} chars (formatted from API)")
        
        # Store platform URLs to profile.platforms for reference
        for platform_key, url in personal_links.items():
            if url:
                platform_name = {
                    'homepage': 'homepage',
                    'google_scholar': 'scholar',
                    'github': 'github',
                    'linkedin': 'linkedin',
                    'orcid': 'orcid',
                    'dblp': 'dblp',
                    'twitter': 'twitter'
                }.get(platform_key, platform_key)
                
                if platform_name not in profile.platforms:
                    profile.platforms[platform_name] = url
    
    # ✅ Fetch Google Scholar if available (after all data is ready)
    # Priority: Homepage discovered > OpenReview API > profile.platforms
    scholar_url = (
        html_social_links.get('scholar', '') or 
        (openreview_data.get('personal_links', {}).get('google_scholar', '') if openreview_data else '') or
        profile.platforms.get('scholar', '')
    )
    
    if scholar_url and not profile._scholar_raw_text:
        try:
            print(f"[Direct Homepage] 🔍 Fetching Google Scholar: {scholar_url[:80]}...")
            _progress("fetching_scholar", 0.22)
            
            scholar_result = ad.fetch_homepage_comprehensive(
                scholar_url,
                author_name=profile_name,
                include_subpages=False,
                max_subpages=0
            )
            if scholar_result and scholar_result.get('success'):
                scholar_text = scholar_result.get('text_content', '')
                if scholar_text and len(scholar_text) > 100:
                    profile._scholar_raw_text = scholar_text[:15000]  # Limit to 15K chars
                    print(f"[Direct Homepage] ✅ Stored Scholar text: {len(profile._scholar_raw_text)} chars")
                else:
                    print(f"[Direct Homepage] ⚠️ Scholar text too short or empty")
            else:
                print(f"[Direct Homepage] ⚠️ Failed to fetch Scholar (may be blocked)")
        except Exception as e:
            print(f"[Direct Homepage] ⚠️ Error fetching Scholar: {e}")
    
    _progress("creating_mock_data", 0.25)
    
    #    3) 如果是普通网页且成功提取到作者姓名，走 targeted search 流程
    #    如果之前已经从页面标题成功执行了 targeted search，这里直接使用结果
    if targeted_search_success and discovered_profile_from_targeted_search:
        print(f"[Direct Homepage] ✅ Using previously discovered profile from targeted search (skipping duplicate search)")
        # 使用之前发现的 profile 替换当前的 profile
        profile = discovered_profile_from_targeted_search
        
        # 合并从网页提取的数据到 discovered profile
        #    优先更新 _homepage_metadata（包含 Agent 提取的 extracted_dimensions）
        if result:
            if result.get("metadata"):
                profile._homepage_metadata = result.get("metadata")
                if result.get("metadata", {}).get("extracted_dimensions"):
                    print(f"[Direct Homepage] ✅ Updated _homepage_metadata with {len(result['metadata']['extracted_dimensions'])} extracted dimensions from Agent")
            elif result.get("extracted_dimensions"):
                profile._homepage_metadata = {
                    "extracted_dimensions": result.get("extracted_dimensions")
                }
                print(f"[Direct Homepage] ✅ Updated _homepage_metadata with {len(result['extracted_dimensions'])} extracted dimensions from Agent (from result['extracted_dimensions'])")
        
        # 合并 homepage 文本内容
        if profile._homepage_raw_text:
            # 如果 discovered profile 已经有 homepage 数据，合并它们
            existing_homepage = profile._homepage_raw_text
            new_homepage = result.get("text_content", "") if result else ""
            if new_homepage and new_homepage not in existing_homepage:
                profile._homepage_raw_text = f"{existing_homepage}\n\n========== ADDITIONAL HOMEPAGE CONTENT ==========\n{new_homepage}"
                print(f"[Direct Homepage] ✅ Merged additional homepage content into discovered profile")
        else:
            # 如果 discovered profile 没有 homepage 数据，使用当前提取的
            if result:
                profile._homepage_raw_text = result.get("text_content", "") or result.get("text", "") or ""
                print(f"[Direct Homepage] ✅ Added homepage content to discovered profile")
        
        # 更新 final_author_name
        final_author_name = extracted_author_name
    elif not is_openreview_url and use_targeted_search_flow and extracted_author_name and not targeted_search_success:
        # 如果从 Agent 的 background 维度提取到 name，但之前没有成功执行 targeted search
        # 尝试搜索 OpenReview，如果找到就走 discover_author_profile，否则只解析普通网页
        print(f"[Direct Homepage] 🎯 Extracted name from Agent background, trying OpenReview search: {extracted_author_name}")
        _progress("targeted_search", 0.30)
        
        try:
            #    先尝试直接搜索 OpenReview（使用多种 name 变体）
            openreview_result = _try_search_openreview_with_variants(extracted_author_name, api_key)
            
            if openreview_result:
                # 如果找到 OpenReview，使用 discover_author_profile
                # 传递已搜索的 openreview_result，避免重复 API 调用
                print(f"[Direct Homepage] ✅ Found OpenReview, proceeding with discover_author_profile (passing pre-fetched result)")
                discovered_profile = ad.discover_author_profile(
                    first_author=extracted_author_name,
                    paper_title="",  # Resume evaluation 没有触发论文
                    aliases=[],
                    k_queries=40,
                    author_id=None,
                    api_key=api_key,
                    openreview_result=openreview_result
                )
                
                if discovered_profile:
                    print(f"[Direct Homepage] ✅ Discovered profile via targeted search")
                    # 使用 discovered profile 替换当前的 profile
                    profile = discovered_profile
                    
                    # 更新 openreview_data（从 discovered profile 中获取）
                    if hasattr(profile, '_openreview_data') and profile._openreview_data:
                        openreview_data = profile._openreview_data
                        print(f"[Direct Homepage] ✅ Using OpenReview data from discovered profile")
                    else:
                        # 如果 discovered profile 没有 OpenReview，使用搜索结果
                        if openreview_result:
                            openreview_data = openreview_result
                            profile._openreview_data = openreview_result
                            print(f"[Direct Homepage] ✅ Using OpenReview data from search result")
                    
                    # 确保 homepage URL 被正确设置
                    if not profile.homepage_url or profile.homepage_url == homepage_url:
                        profile.homepage_url = homepage_url
                    
                    # 合并从网页提取的数据到 discovered profile
                    #    优先更新 _homepage_metadata（包含 Agent 提取的 extracted_dimensions）
                    if result:
                        if result.get("metadata"):
                            profile._homepage_metadata = result.get("metadata")
                            if result.get("metadata", {}).get("extracted_dimensions"):
                                print(f"[Direct Homepage] ✅ Updated _homepage_metadata with {len(result['metadata']['extracted_dimensions'])} extracted dimensions from Agent")
                        elif result.get("extracted_dimensions"):
                            profile._homepage_metadata = {
                                "extracted_dimensions": result.get("extracted_dimensions")
                            }
                            print(f"[Direct Homepage] ✅ Updated _homepage_metadata with {len(result['extracted_dimensions'])} extracted dimensions from Agent (from result['extracted_dimensions'])")
                    
                    # 合并 homepage 文本内容
                    if profile._homepage_raw_text:
                        # 如果 discovered profile 已经有 homepage 数据，合并它们
                        existing_homepage = profile._homepage_raw_text
                        new_homepage = result.get("text_content", "") if result else ""
                        if new_homepage and new_homepage not in existing_homepage:
                            profile._homepage_raw_text = f"{existing_homepage}\n\n========== ADDITIONAL HOMEPAGE CONTENT ==========\n{new_homepage}"
                            print(f"[Direct Homepage] ✅ Merged additional homepage content into discovered profile")
                    else:
                        # 如果 discovered profile 没有 homepage 数据，使用当前提取的
                        if result:
                            profile._homepage_raw_text = result.get("text_content", "") or result.get("text", "") or ""
                            print(f"[Direct Homepage] ✅ Added homepage content to discovered profile")
                    
                    # 更新 final_author_name 和标记成功
                    final_author_name = extracted_author_name
                    targeted_search_success = True
                else:
                    print(f"[Direct Homepage] ⚠️ discover_author_profile returned None, will fall back to normal flow")
                    targeted_search_success = False
            else:
                #    如果没找到 OpenReview，只解析普通网页，不走 discover_author_profile
                print(f"[Direct Homepage] ⚠️ No OpenReview found for '{extracted_author_name}', will only parse the provided homepage (no discover_author_profile)")
                targeted_search_success = False
        except Exception as e:
            print(f"[Direct Homepage] ⚠️ Targeted search flow failed: {e}")
            import traceback
            traceback.print_exc()
            targeted_search_success = False
    
    # 3) Prepare openreview_data for build_enhanced_profile
    if not openreview_data:
        # No API data: create mock structure from HTML scraping
        print(f"[Direct Homepage] Creating mock openreview_data from HTML scraping")
        
        openreview_data = {
            "openreview_url": homepage_url if is_openreview_url else html_social_links.get("openreview", ""),
            "names": [{"first": author_hint.split()[0] if author_hint else "", 
                      "middle": "", 
                      "last": author_hint.split()[-1] if author_hint and len(author_hint.split()) > 1 else ""}],
            "personal_links": {
                "homepage": "" if is_openreview_url else (actual_homepage_url or homepage_url),
                "google_scholar": html_social_links.get("scholar", ""),
                "github": html_social_links.get("github", ""),
                "linkedin": html_social_links.get("linkedin", ""),
                "dblp": html_social_links.get("dblp", ""),
                "orcid": html_social_links.get("orcid", ""),
                "twitter": html_social_links.get("twitter", ""),
            },
            "education": [],
            "publications": [],
            "coauthors": {},
            "relations": [],
        }
    else:
        # API data exists: use it directly (same as Targeted Search)
        print(f"[Direct Homepage] Using OpenReview API structured data (same as Targeted Search)")
        
        # ✅ Update OpenReview data with links discovered from Homepage
        # Homepage links are more up-to-date than OpenReview API
        if html_social_links:
            print(f"[Direct Homepage] Updating OpenReview data with {len(html_social_links)} links from Homepage")
            openreview_data.setdefault('personal_links', {})
            
            # Update with discovered links (keep existing if not found)
            if html_social_links.get('scholar'):
                openreview_data['personal_links']['google_scholar'] = html_social_links['scholar']
            if html_social_links.get('github'):
                openreview_data['personal_links']['github'] = html_social_links['github']
            if html_social_links.get('linkedin'):
                openreview_data['personal_links']['linkedin'] = html_social_links['linkedin']
            if html_social_links.get('twitter'):
                openreview_data['personal_links']['twitter'] = html_social_links['twitter']
            if html_social_links.get('orcid'):
                openreview_data['personal_links']['orcid'] = html_social_links['orcid']
            if html_social_links.get('dblp'):
                openreview_data['personal_links']['dblp'] = html_social_links['dblp']
        
        # Ensure homepage URL is set
        if actual_homepage_url:
            openreview_data['personal_links']['homepage'] = actual_homepage_url
    
    _progress("building_enhanced_profile", 0.35)
    
    # 4) Call build_enhanced_profile to generate 9-field structure
    # Use extracted_author_name (from targeted search) > author_query (from OpenReview URL) > author_hint
    if not final_author_name:
        final_author_name = author_query or author_hint or "Unknown"
    
    #    确保使用从页面标题提取的 name（如果存在）
    if extracted_author_name and not final_author_name:
        final_author_name = extracted_author_name
        print(f"[Direct Homepage] ✅ Using extracted_author_name as final_author_name: {final_author_name}")
    elif extracted_author_name and final_author_name != extracted_author_name:
        # 如果 extracted_author_name 存在且与 final_author_name 不同，优先使用 extracted_author_name
        print(f"[Direct Homepage] ⚠️ final_author_name ({final_author_name}) differs from extracted_author_name ({extracted_author_name}), using extracted_author_name")
        final_author_name = extracted_author_name
    
    print(f"[Direct Homepage] 📝 Final author_name for build_enhanced_profile: {final_author_name}")
    
    try:
        enhanced_profile = ad.build_enhanced_profile(
            openreview_data=openreview_data,
            author_name=final_author_name,
            api_key=api_key,
            profile=profile,
            user_query=None
        )
    except Exception as e:
        print(f"[Direct Homepage] Enhanced profile build failed: {e}")
        import traceback
        traceback.print_exc()
        enhanced_profile = None
    
    _progress("profile_built", 0.80)
    
    if not enhanced_profile:
        # Fallback: return minimal data
        return {
            "error": "Failed to build enhanced profile",
            "name": author_hint or "Unknown",
            "email": profile.emails[0] if profile.emails else "",
            "current_role_affiliation": "",
            "current_status": "",
            "research_keywords": [],
            "research_focus": [],
            "profiles": {"Homepage": homepage_url},
            "publication_overview": [],
            "top_tier_hits": [],
            "honors_grants": [],
            "service_talks": [],
            "open_source_projects": [],
            "representative_papers": [],
            "highlights": [],
            "radar": {},
            "total_score": 0,
            "detailed_scores": {},
        }
    
    _progress("converting_to_frontend", 0.90)
    
    # 5) Convert to frontend payload
    payload = _to_frontend_payload_from_enhanced(enhanced_profile, eval_result=None)
    
    _progress("done", 1.0)
    
    return payload
