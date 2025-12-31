"""
Web Search Fallback for Candidate Discovery
当 OpenReview 等主要来源无法获取候选人信息时，使用网络搜索作为兜底方案

使用场景：
- OpenReview API 找不到候选人
- Semantic Scholar 无完整 profile
- 需要补充候选人的公开信息

搜索策略：
- 使用 姓名 + 组织 构建搜索查询
- 筛选高价值 URL（个人主页、学校页面、学术档案）
- 爬取内容并用 LLM 提取结构化信息
"""

import json
import re
import logging
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from urllib.parse import urlparse

from . import search

logger = logging.getLogger(__name__)


@dataclass
class WebSearchCandidate:
    """Web搜索发现的候选人信息"""
    name: str
    affiliation: str = ""
    position: str = ""
    department: str = ""
    homepage_url: Optional[str] = None
    email: Optional[str] = None
    research_interests: List[str] = field(default_factory=list)
    publications: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    social_links: Dict[str, str] = field(default_factory=dict)
    bio_text: str = ""
    confidence: float = 0.5
    source_urls: List[str] = field(default_factory=list)
    is_verified: bool = False


# ============================ 高价值域名配置 ============================

# 高价值域名模式（个人主页、学术档案等）
HIGH_VALUE_DOMAINS = [
    '.edu',           # 美国大学
    '.ac.',           # 英国/日本等学术机构
    '.edu.cn',        # 中国大学
    '.edu.hk',        # 香港大学
    '.edu.sg',        # 新加坡大学
    'github.com',
    'github.io',      # GitHub Pages 个人主页
    'linkedin.com',
    'scholar.google',
    'dblp.org',
    'semanticscholar.org',
    'researchgate.net',
    'orcid.org',
    'csail.mit.edu',
    'stanford.edu',
    'berkeley.edu',
    'cmu.edu',
    'tsinghua.edu.cn',
    'pku.edu.cn',
]

# 排除的域名（社交媒体、购物等无关网站）
EXCLUDE_DOMAINS = [
    'twitter.com', 'x.com', 
    'facebook.com', 'fb.com',
    'youtube.com', 
    'amazon.com', 
    'pinterest.com', 
    'instagram.com',
    'tiktok.com',
    'weibo.com',
    'zhihu.com',       # 知乎可能有噪音
    'reddit.com',
    'quora.com',
    'medium.com',      # 博客平台噪音多
    'arxiv.org',       # 论文页面，不是个人主页
    'openreview.net',  # 已经尝试过了
    'papers.nips.cc',
    'proceedings.mlr.press',
]

# 个人主页关键词（用于评分）
HOMEPAGE_KEYWORDS = [
    'homepage', 'personal', 'home page', 'about me', 'bio',
    'professor', 'phd', 'researcher', 'scientist',
    'faculty', 'staff', 'people', 'members',
    '个人主页', '个人简介', '教师主页',
]

# 论文标题中的停用词（用于提取关键词时过滤）
TITLE_STOP_WORDS = {
    'a', 'an', 'the', 'of', 'for', 'and', 'or', 'in', 'on', 'at', 'to', 'with',
    'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'via', 'using', 'based', 'towards', 'toward', 'through', 'into',
    'that', 'this', 'these', 'those', 'it', 'its', 'their', 'our', 'your',
}


def extract_paper_title_keywords(paper_title: str, max_keywords: int = 4) -> str:
    """
    从论文标题中提取关键词（用于搜索查询）
    
    Args:
        paper_title: 论文标题
        max_keywords: 最大关键词数
    
    Returns:
        关键词字符串，例如 "Zhyper Factorized Hypernetworks LLM"
    """
    if not paper_title:
        return ""
    
    # 移除特殊字符，保留字母数字和空格
    clean_title = re.sub(r'[^\w\s-]', ' ', paper_title)
    
    # 分词
    words = clean_title.split()
    
    # 过滤停用词和短词，保留有意义的词
    keywords = []
    for word in words:
        word_lower = word.lower()
        # 跳过停用词、纯数字、太短的词
        if word_lower in TITLE_STOP_WORDS:
            continue
        if word.isdigit():
            continue
        if len(word) < 3:
            continue
        keywords.append(word)
        if len(keywords) >= max_keywords:
            break
    
    return ' '.join(keywords)


class WebSearchFallbackDiscovery:
    """Web搜索兜底发现器"""
    
    def __init__(self, api_key: str = None):
        self.api_key = api_key
    
    def discover_candidate(
        self,
        author_name: str,
        known_affiliation: str = "",
        paper_title: str = "",
    ) -> Optional[WebSearchCandidate]:
        """
        通过网络搜索发现候选人信息
        
        Args:
            author_name: 作者姓名 (从论文中提取)
            known_affiliation: 已知的组织/机构 (从论文中提取)
            paper_title: 触发论文标题 (用于验证身份)
        
        Returns:
            WebSearchCandidate 或 None
        """
        logger.info(f"[WebSearch Fallback] Starting search for: {author_name}")
        print(f"\n{'='*60}")
        print(f"[WebSearch Fallback] Searching for: {author_name}")
        if known_affiliation:
            print(f"[WebSearch Fallback] Affiliation: {known_affiliation}")
        if paper_title:
            print(f"[WebSearch Fallback] Paper title: {paper_title[:60]}...")
        print(f"{'='*60}")
        
        # Step 1: 构建搜索查询（名字 + 组织 或 名字 + 论文关键词）
        search_queries = self._build_search_queries(author_name, known_affiliation, paper_title)
        print(f"[WebSearch Fallback] Search queries: {search_queries}")
        
        # Step 2: 执行搜索
        search_results = self._execute_searches(search_queries)
        if not search_results:
            print(f"[WebSearch Fallback] No search results for {author_name}")
            return None
        print(f"[WebSearch Fallback] Found {len(search_results)} search results")
        
        # Step 3: 筛选有价值的URL
        valuable_urls = self._filter_valuable_urls(search_results, author_name)
        if not valuable_urls:
            print(f"[WebSearch Fallback] No valuable URLs for {author_name}")
            return None
        print(f"[WebSearch Fallback] Selected {len(valuable_urls)} valuable URLs")
        for item in valuable_urls:
            print(f"    - {item['url']} (score: {item['score']})")
        
        # Step 4: 爬取内容
        crawled_contents = self._crawl_urls(valuable_urls)
        if not crawled_contents:
            print(f"[WebSearch Fallback] Failed to crawl URLs for {author_name}")
            return None
        print(f"[WebSearch Fallback] Crawled {len(crawled_contents)} pages")
        
        # Step 5: LLM提取结构化Profile
        candidate = self._extract_profile_with_llm(
            author_name, known_affiliation, paper_title, crawled_contents
        )
        
        if candidate:
            print(f"[WebSearch Fallback] Successfully extracted profile for {candidate.name}")
            print(f"    - Position: {candidate.position}")
            print(f"    - Affiliation: {candidate.affiliation}")
            print(f"    - Confidence: {candidate.confidence:.2f}")
        else:
            print(f"[WebSearch Fallback] Failed to extract profile for {author_name}")
        
        return candidate
    
    def _build_search_queries(
        self,
        author_name: str,
        known_affiliation: str,
        paper_title: str = "",
    ) -> List[str]:
        """
        构建搜索查询
        
        策略：
        - 有机构：所有查询都带机构信息
        - 无机构：如果有论文标题就用 姓名 + 论文标题；若都没有则跳过
        
        这样可以有效降低重名风险
        """
        queries = []
        
        has_paper_title = bool(paper_title and paper_title.strip())
        
        if known_affiliation:
            # ==== 有机构信息：所有查询都带机构 ====
            affiliation_clean = known_affiliation.split(',')[0].strip()
            
            # Query 1: 姓名 + 机构（最精确）
            queries.append(f'"{author_name}" {affiliation_clean}')
            
            # Query 2: 姓名 + 机构 + homepage（保留机构限定）
            queries.append(f'"{author_name}" {affiliation_clean} homepage')
            
            print(f"[WebSearch Fallback] Using affiliation-based search (low ambiguity)")
        else:
            # ==== 无机构信息：优先使用论文标题 ==== 
            if not has_paper_title:
                # 无机构且无论文标题：直接放弃，风险太高
                print(f"[WebSearch Fallback] Skipped: No affiliation and no paper title (HIGH AMBIGUITY RISK)")
                return []  # 返回空列表，后续会跳过
            
            # Query 1: 姓名 + 论文标题
            queries.append(f'"{author_name}" {paper_title}')
            
            print(f"[WebSearch Fallback] Using paper-title-based search (medium ambiguity)")
            print(f"[WebSearch Fallback] Paper title: {paper_title}")
        
        return queries[:3]  # 最多3个查询
    
    def _execute_searches(self, queries: List[str]) -> List[Dict[str, Any]]:
        """执行搜索并合并结果"""
        all_results = []
        seen_urls = set()
        
        for query in queries:
            try:
                print(f"[WebSearch Fallback] Executing query: {query}")
                result = search.searxng_search(query, pages=1, k_per_query=10)
                parsed_results = self._parse_search_result(result)
                print(f"[WebSearch Fallback] Found {len(parsed_results)} results for query")
                
                for item in parsed_results:
                    url = item.get('url', '')
                    if url and url not in seen_urls:
                        seen_urls.add(url)
                        all_results.append(item)
                
                print(f"[WebSearch Fallback] Got {len(parsed_results)} results")
                
            except Exception as e:
                logger.error(f"[WebSearch Fallback] Search error for query '{query}': {e}")
                print(f"[WebSearch Fallback] Search error: {e}")
                # 打印更详细的错误信息以便调试
                import traceback
                traceback.print_exc()
        
        return all_results
    
    def _parse_search_result(self, result: Any) -> List[Dict[str, Any]]:
        """解析搜索工具返回的结果"""
        parsed = []
        
        if isinstance(result, str):
            # 清理 null bytes 和其他控制字符（双重保险）
            result = result.replace('\x00', '')
            result = ''.join(char for char in result if ord(char) >= 32 or char in '\n\r\t')
            # 尝试解析 JSON
            try:
                data = json.loads(result)
                if isinstance(data, list):
                    parsed = data
                elif isinstance(data, dict):
                    # Tavily 格式
                    if 'results' in data:
                        parsed = data['results']
                    elif 'url' in data:
                        parsed = [data]
            except json.JSONDecodeError as e:
                # 可能是纯文本格式，尝试其他解析
                print(f"[WebSearch Fallback] JSON decode error: {e}")
                # 尝试提取可能的 URL（兜底策略）
                import re
                urls = re.findall(r'https?://[^\s<>"\')]+', result)
                for url in urls[:5]:  # 最多取 5 个 URL
                    parsed.append({'url': url, 'title': '', 'snippet': ''})
                if parsed:
                    print(f"[WebSearch Fallback] Extracted {len(parsed)} URLs from raw text")
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
    
    def _filter_valuable_urls(
        self, 
        search_results: List[Dict], 
        author_name: str
    ) -> List[Dict[str, Any]]:
        """筛选有价值的URL"""
        
        valuable = []
        author_name_lower = author_name.lower()
        author_parts = author_name_lower.split()
        
        for item in search_results:
            url = item.get('url', '')
            title = item.get('title', '')
            snippet = item.get('snippet', '')
            
            if not url:
                continue
            
            url_lower = url.lower()
            
            # 排除不需要的域名
            if any(ex in url_lower for ex in EXCLUDE_DOMAINS):
                continue
            
            # 计算价值分数
            score = 0
            text_lower = (title + " " + snippet).lower()
            
            # 检查是否包含作者姓名（部分匹配）
            name_matches = sum(1 for part in author_parts if len(part) > 1 and part in text_lower)
            score += name_matches * 2
            
            # 检查 URL 是否包含作者姓名
            url_name_matches = sum(1 for part in author_parts if len(part) > 2 and part in url_lower)
            score += url_name_matches * 3  # URL 中出现名字更有价值
            
            # 检查是否是高价值域名
            for hv in HIGH_VALUE_DOMAINS:
                if hv in url_lower:
                    score += 3
                    break
            
            # 检查是否像个人主页
            if any(kw in text_lower for kw in HOMEPAGE_KEYWORDS):
                score += 2
            
            # 检查是否是 GitHub Pages 等个人站点
            if '.github.io' in url_lower or '/~' in url_lower or '/people/' in url_lower:
                score += 2
            
            # 只保留分数 >= 2 的结果
            if score >= 2:
                valuable.append({
                    'url': url,
                    'title': title,
                    'snippet': snippet,
                    'score': score
                })
        
        # 按分数排序，取前5个
        valuable.sort(key=lambda x: x['score'], reverse=True)
        return valuable[:5]
    
    def _crawl_urls(self, urls: List[Dict]) -> List[Dict[str, str]]:
        """爬取URL内容"""
        results = []
        
        for item in urls[:3]:  # 最多爬取3个页面
            url = item['url']
            try:
                print(f"[WebSearch Fallback] Crawling: {url}")
                content = search.fetch_text(url, max_chars=15000, snippet=item.get("snippet", ""))
                
                if content and len(content) > 100:
                    results.append({
                        'url': url,
                        'title': item.get('title', ''),
                        'content': content[:15000]  # 限制内容长度
                    })
                    print(f"[WebSearch Fallback] Got {len(content)} chars")
                else:
                    print(f"[WebSearch Fallback] Content too short or empty")
                    
            except Exception as e:
                logger.error(f"[WebSearch Fallback] Crawl error for {url}: {e}")
                print(f"[WebSearch Fallback] Crawl error: {str(e)}")
        
        return results
    
    def _extract_profile_with_llm(
        self,
        author_name: str,
        known_affiliation: str,
        paper_title: str,
        crawled_contents: List[Dict]
    ) -> Optional[WebSearchCandidate]:
        """使用LLM从爬取的内容中提取结构化Profile"""
        
        from . import llm as llm_module
        
        # 合并所有爬取的内容
        combined_content = "\n\n---PAGE BREAK---\n\n".join([
            f"Source: {c['url']}\nTitle: {c['title']}\nContent:\n{c['content']}"
            for c in crawled_contents
        ])
        
        prompt = f"""You are extracting structured profile information for a researcher.

Target Author: {author_name}
Known Affiliation: {known_affiliation or "Unknown"}
Related Paper: {paper_title or "Not specified"}

From the following web content, extract information ONLY about the target author.
Be careful not to confuse with other people mentioned on the page.

Web Content:
{combined_content[:12000]}

Extract the following in JSON format:
{{
    "name": "Full name as found on the page",
    "position": "Current position (e.g., PhD Student, Assistant Professor, Research Scientist)",
    "affiliation": "Current institution/company",
    "department": "Department or lab name if available",
    "email": "Email address if found",
    "homepage_url": "Personal homepage URL if different from source",
    "bio_text": "Brief bio or introduction (1-2 sentences)",
    "research_interests": ["list", "of", "research", "interests"],
    "publications": [
        {{"title": "paper title", "venue": "conference/journal", "year": 2024}}
    ],
    "education": [
        {{"degree": "PhD", "institution": "University Name", "field": "Computer Science", "duration": "2020-2024"}}
    ],
    "social_links": {{
        "google_scholar": "url if found",
        "github": "url if found",
        "linkedin": "url if found",
        "dblp": "url if found"
    }},
    "is_target_author": true,
    "confidence": 0.8
}}

IMPORTANT RULES:
1. Set "is_target_author" to false if the content is clearly about a DIFFERENT person
2. Set "confidence" between 0.0-1.0 based on how certain you are this is the target author
3. Only include information EXPLICITLY stated in the content, do NOT infer or guess
4. If a field is not found, use empty string "" or empty list []
5. For publications, only include 3-5 most recent/relevant ones

Output ONLY valid JSON, no other text."""

        try:
            llm_instance = llm_module.get_llm("extraction", temperature=0.1, api_key=self.api_key)
            response = llm_instance.invoke(prompt)
            
            # 解析响应
            response_text = response.content if hasattr(response, 'content') else str(response)
            
            # 提取JSON
            json_match = re.search(r'\{[\s\S]*\}', response_text)
            if json_match:
                data = json.loads(json_match.group())
                
                # 验证是否是目标作者
                if not data.get('is_target_author', True):
                    print(f"[WebSearch Fallback] Content is not about target author")
                    return None
                
                confidence = data.get('confidence', 0)
                if confidence < 0.3:
                    print(f"[WebSearch Fallback] Low confidence ({confidence:.2f}), skipping")
                    return None
                
                # 构建 WebSearchCandidate
                return WebSearchCandidate(
                    name=data.get('name', author_name),
                    affiliation=data.get('affiliation', known_affiliation),
                    position=data.get('position', ''),
                    department=data.get('department', ''),
                    homepage_url=data.get('homepage_url'),
                    email=data.get('email'),
                    bio_text=data.get('bio_text', ''),
                    research_interests=data.get('research_interests', []),
                    publications=data.get('publications', []),
                    education=data.get('education', []),
                    social_links=data.get('social_links', {}),
                    confidence=confidence,
                    source_urls=[c['url'] for c in crawled_contents],
                    is_verified=True
                )
                
        except json.JSONDecodeError as e:
            logger.error(f"[WebSearch Fallback] JSON parse error: {e}")
            print(f"[WebSearch Fallback] JSON parse error: {e}")
        except Exception as e:
            logger.error(f"[WebSearch Fallback] LLM extraction error: {e}")
            print(f"[WebSearch Fallback] LLM extraction error: {e}")
        
        return None


def convert_web_candidate_to_enhanced_profile(
    web_candidate: WebSearchCandidate,
    trigger_paper: Dict = None,
    user_query: str = ""
) -> 'schemas.EnhancedAuthorProfile':
    """
    将 WebSearchCandidate 转换为标准的 EnhancedAuthorProfile
    
    Args:
        web_candidate: Web搜索得到的候选人信息
        trigger_paper: 触发论文信息 dict
        user_query: 用户查询（用于论文评分）
    
    Returns:
        EnhancedAuthorProfile 对象
    """
    from . import schemas
    
    # 1. Introduction
    intro = schemas.IntroductionInfo(
        name=web_candidate.name,
        affiliation=web_candidate.affiliation,
        position=web_candidate.position,
        text=web_candidate.bio_text or f"{web_candidate.name} is a {web_candidate.position} at {web_candidate.affiliation}."
    )
    
    # 2. Current Role
    current_role = schemas.CurrentRoleInfo(
        category=_infer_role_category(web_candidate.position),
        role_text=web_candidate.position,
        affiliation=web_candidate.affiliation,
        explanation="Extracted from web search"
    )
    
    # 3. Research Interests
    interests = [
        schemas.ResearchInterest(name=ri, description="")
        for ri in (web_candidate.research_interests or [])
    ]
    
    # 4. Publications
    publications = []
    
    # 先添加 trigger paper（如果有）
    if trigger_paper:
        # Convert empty string year to None for Pydantic validation
        year_value = trigger_paper.get('year')
        if year_value == '' or year_value is None:
            year_value = None
        elif isinstance(year_value, str):
            try:
                year_value = int(year_value)
            except ValueError:
                year_value = None
        
        trigger_pub = schemas.PublicationInfo(
            title=trigger_paper.get('title', ''),
            venue=trigger_paper.get('venue', ''),
            year=year_value,
            url=trigger_paper.get('url', ''),
            authors=trigger_paper.get('authors', []),
            relevance_score=trigger_paper.get('score', 0),
            relevance_explanation=trigger_paper.get('explanation', ''),
            # Multi-dimensional scoring fields
            quality_score=trigger_paper.get('quality_score', 0.0),
            relevance_dimensions=trigger_paper.get('relevance_dimensions', {}),
            quality_dimensions=trigger_paper.get('quality_dimensions', {}),
            candidate_author_position=trigger_paper.get('candidate_author_position'),
            total_authors=trigger_paper.get('total_authors')
        )
        publications.append(trigger_pub)
    
    # 添加从网页提取的论文
    for pub in (web_candidate.publications or []):
        # Convert empty string year to None for Pydantic validation
        year_value = pub.get('year')
        if year_value == '' or year_value is None:
            year_value = None
        elif isinstance(year_value, str):
            try:
                year_value = int(year_value)
            except ValueError:
                year_value = None
        
        pub_info = schemas.PublicationInfo(
            title=pub.get('title', ''),
            venue=pub.get('venue', ''),
            year=year_value,
            url=pub.get('url', ''),
            authors=pub.get('authors', [])
        )
        publications.append(pub_info)
    
    selected_research = schemas.SelectedResearch(
        all_publications=publications,
        by_category={"Uncategorized": publications} if publications else {}
    )
    
    # 5. Career & Education History
    career_education = []
    for edu in (web_candidate.education or []):
        item = schemas.CareerEducationInfo(
            degree_or_position=edu.get('degree', ''),
            institution=edu.get('institution', ''),
            department=edu.get('department', ''),
            duration=edu.get('duration', ''),
            field=edu.get('field', ''),
            advisor=edu.get('advisor', '')
        )
        career_education.append(item)
    
    # 6. Contact
    social = web_candidate.social_links or {}
    contact = schemas.ContactInfo(
        email=web_candidate.email or "",
        homepage=web_candidate.homepage_url or "",
        google_scholar=social.get('google_scholar', ''),
        github=social.get('github', ''),
        linkedin=social.get('linkedin', ''),
        dblp=social.get('dblp', ''),
        orcid=social.get('orcid', '')
    )
    
    # 构建完整 EnhancedAuthorProfile
    return schemas.EnhancedAuthorProfile(
        introduction=intro,
        current_role=current_role,
        research_interests=interests,
        selected_research=selected_research,
        awards=[],
        professional_services=[],
        career_education_history=career_education,
        industrial_experience=[],
        contact=contact,
        confidence=web_candidate.confidence,
        data_sources=web_candidate.source_urls or ["web_search_fallback"]
    )


def _infer_role_category(position: str) -> str:
    """从 position 推断角色类别"""
    if not position:
        return "Unknown"
    
    pos_lower = position.lower()
    
    # 先检查 postdoc（在 phd 之前检查，避免误判）
    if any(kw in pos_lower for kw in ['postdoc', 'post-doc', 'postdoctoral']):
        return "Postdoc"
    elif any(kw in pos_lower for kw in ['phd', 'doctoral', 'ph.d']):
        return "PhD Student"
    elif any(kw in pos_lower for kw in ['master', 'msc', 'm.s.']):
        return "Master Student"
    elif any(kw in pos_lower for kw in ['undergrad', 'bachelor', 'b.s.', 'b.a.']):
        return "Undergraduate Student"
    elif any(kw in pos_lower for kw in ['professor', 'prof', 'faculty', 'lecturer']):
        return "Professor"
    elif any(kw in pos_lower for kw in ['research scientist', 'researcher', 'engineer']):
        # 检查是否是工业界
        if any(company in pos_lower for company in ['google', 'microsoft', 'meta', 'amazon', 'openai', 'deepmind']):
            return "Industrial Researcher"
        return "Institution Researcher"
    else:
        return "Unknown"
