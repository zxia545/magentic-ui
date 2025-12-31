"""
个人主页解析器
使用分层清洗 + 结构化提取 + LLM 解析的方式提取学者主页信息

新增功能：
- 智能交互式抓取（使用 nine_dimension_aware_agent）
- 九维度感知：自动识别并提取 9 个维度的信息
- LLM 智能过滤：对文字元素进行相关性判断，过滤无关元素
- 自动处理需要点击"Show More"等动态内容的页面
- 三层过滤机制：外部链接、内容链接、导航元素
"""
import requests
import re
import json
import time
import threading
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin
from bs4 import BeautifulSoup

# 尝试导入 Nine Dimension Aware Agent（九维度感知的交互代理）
try:
    from .nine_dimension_aware_agent import NineDimensionAwareAgent
    WEB_AGENT_AVAILABLE = True
except ImportError:
    WEB_AGENT_AVAILABLE = False
    print("[HomepageParser] ⚠️ Nine Dimension Aware Agent not available")
# ========== GOOGLE SCHOLAR RATE LIMITING ==========
# 使用信号量限制并发数，而不是完全串行化
_SCHOLAR_SEMAPHORE = threading.Semaphore(3)  # 最多3个线程同时访问Scholar
_SCHOLAR_LOCK = threading.Lock()  # 用于协调请求间隔
_LAST_SCHOLAR_REQUEST_TIME = 0
_MIN_SCHOLAR_INTERVAL = 3.0  # 每个请求之间至少间隔3秒
class HomepageParser:
    """学者个人主页解析器（支持 Homepage、Google Scholar、ORCID、DBLP）"""
    # Awards 和 Services 关键词（用于快速筛选）
    AWARDS_KW = r"award|honor|fellowship|scholar|best paper|rising star|outstanding|distinguished|prize"
    SERVICE_KW = r"\bpc\b|\bac\b|program chair|area chair|organizer|reviewer|editor|committee|chair|meta-reviewer"
    def __init__(self, api_key: str = None, enable_interactive: bool = False):
        """
        初始化解析器
        
        Args:
            api_key: LLM API key
            enable_interactive: 是否启用交互式抓取（需要 Playwright）
        """
        self.session = requests.Session()
        # ✅ 改进：使用完整的浏览器请求头，避免被识别为爬虫
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN,zh;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Cache-Control': 'max-age=0'
        })
        
        # 初始化 Nine Dimension Aware Agent（如果启用）
        # 使用九维度感知的交互代理，具有 LLM 智能过滤功能
        self.enable_interactive = enable_interactive and WEB_AGENT_AVAILABLE
        self.web_agent = None
        if self.enable_interactive:
            try:
                self.web_agent = NineDimensionAwareAgent(api_key=api_key, headless=True)
                print(f"[HomepageParser] ✅ Nine Dimension Aware Agent initialized (with LLM filtering)")
            except Exception as e:
                print(f"[HomepageParser] ⚠️ Failed to initialize Web Agent: {e}")
                self.enable_interactive = False
        # Google Scholar 特定配置
        self.scholar_session = requests.Session()
        self.scholar_session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9,zh-CN,zh;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        })
        # 初始化 LLM 客户端
        self.api_key = api_key
        self.llm_available = False
        self.llm = None
        
        try:
            from . import llm as llm_module
            self.llm = llm_module.get_llm("parse", temperature=0.1)
            self.llm_available = True
            print(f"[HomepageParser] LLM 初始化成功 (Azure OpenAI)")
        except Exception as e:
            print(f"[HomepageParser] LLM 初始化失败: {e}")
        # 章节关键词
        self.section_keywords = {
            'introduction': ['introduction', 'about', 'bio', 'biography', 'overview', 'about me'],
            'research_interests': ['research interests', 'research areas', 'interests', 'research'],
            'publications': ['publications', 'papers', 'selected publications', 'selected research', 'recent publications'],
            'education': ['education', 'academic background', 'degrees'],
            'experience': ['experience', 'employment', 'work experience', 'industrial experience', 'positions'],
            'awards': ['awards', 'honors', 'achievements', 'recognition'],
            'services': ['service', 'professional service', 'activities', 'committee', 'professional services'],
            'teaching': ['teaching', 'courses', 'instruction'],
            'contact': ['contact', 'email', 'reach me']
        }
    def fetch_page(self, url: str, use_scholar_session: bool = False, max_retries: int = 3, timeout: int = 15) -> Optional[str]:
        """
        获取网页 HTML（带重试机制 + 优化超时）
        
        Args:
            url: 网页 URL
            use_scholar_session: 是否使用 Scholar 专用 session
            max_retries: 最大重试次数
            timeout: 单次请求超时时间（秒）- 默认 15 秒，避免长时间等待
            
        Returns:
            HTML 内容
        """
        import time
        session = self.scholar_session if use_scholar_session else self.session
        
        for attempt in range(max_retries):
            try:
                # ✅ 添加延迟，避免过快请求
                if attempt > 0:
                    retry_delay = min(2 ** attempt, 5)  # 最多等待 5 秒
                    time.sleep(retry_delay)
                    print(f"[HomepageParser] 重试 {attempt + 1}/{max_retries}: {url[:60]}...")
                
                # 🔧 FIX: 使用可配置的 timeout，默认 15 秒（之前是 30 秒）
                response = session.get(url, timeout=timeout, allow_redirects=True)
                response.raise_for_status()
                response.encoding = response.apparent_encoding
                
                # ✅ 验证返回内容
                if len(response.text) < 100:
                    print(f"[HomepageParser] ⚠️ 返回内容过短 ({len(response.text)} chars)，可能被拦截")
                    if attempt < max_retries - 1:
                        continue
                
                return response.text
                
            except requests.exceptions.HTTPError as e:
                status_code = e.response.status_code if e.response else 0
                if status_code == 404:
                    print(f"[HomepageParser] 获取网页失败 {url}: HTTP 404 (Not Found)")
                    return None  # 404直接返回，不重试
                elif status_code == 403:
                    # 403错误通常表示被反爬虫拦截，重试意义不大
                    if attempt < max_retries - 1:
                        # 对于Scholar等站点，增加延迟后重试
                        if 'scholar.google' in url.lower():
                            wait_time = min(2 ** attempt * 2, 10)  # 最多等待10秒
                            print(f"[HomepageParser] HTTP 403 (Forbidden)，等待 {wait_time}s 后重试...")
                            time.sleep(wait_time)
                            continue
                        else:
                            print(f"[HomepageParser] HTTP 403 (Forbidden)，将重试...")
                            continue
                    else:
                        print(f"[HomepageParser] 获取网页失败 {url}: HTTP 403 (Forbidden - likely rate limited)")
                        return None
                elif status_code == 429:
                    # 429 Too Many Requests，需要更长的等待
                    wait_time = min(2 ** attempt * 3, 15)  # 最多等待15秒
                    if attempt < max_retries - 1:
                        print(f"[HomepageParser] HTTP 429 (Rate Limited)，等待 {wait_time}s 后重试...")
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"[HomepageParser] 获取网页失败 {url}: HTTP 429 (Rate Limited)")
                        return None
                elif attempt < max_retries - 1:
                    print(f"[HomepageParser] HTTP错误 {status_code}，将重试...")
                    continue
                else:
                    print(f"[HomepageParser] 获取网页失败 {url}: HTTP {status_code}")
                    return None
                    
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    print(f"[HomepageParser] 请求超时，将重试...")
                    continue
                else:
                    print(f"[HomepageParser] 获取网页失败 {url}: 请求超时")
                    return None
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    print(f"[HomepageParser] 未知错误，将重试: {str(e)[:50]}")
                    continue
                else:
                    print(f"[HomepageParser] 获取网页失败 {url}: {e}")
                    return None
        
        return None
    def extract_structured_text(self, html: str) -> str:
        """
        第 1 步：HTML → 分层结构化文本
        
        Args:
            html: HTML 内容
            
        Returns:
            结构化文本
        """
        try:
            # 尝试使用 trafilatura（最佳）
            import trafilatura
            text = trafilatura.extract(
                html,
                include_formatting=True,
                include_links=False,
                include_images=False,
                output_format='txt'
            )
            if text and len(text) > 100:
                return text
        except ImportError:
            pass
        # 备用方案：使用 BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        # 移除无关标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        # 保留标题结构
        for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            tag.insert_before('\n\n')
            tag.insert_after('\n')
        # 保留段落间隔
        for tag in soup.find_all('p'):
            tag.insert_after('\n')
        # 保留列表结构
        for tag in soup.find_all('li'):
            tag.string = '• ' + (tag.get_text(strip=True) or '')
        text = soup.get_text(separator=' ')
        return text
    
    def extract_text_by_heading_sections(self, html: str) -> List[Dict[str, str]]:
        """
        按照 h 标签分段提取内容
        从 h 标签到下一个 h 标签之间的内容作为一段
        
        Args:
            html: HTML 内容
            
        Returns:
            分段内容列表，每个元素包含 {'heading': '标题', 'content': '内容'}
        """
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除无关标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
            tag.decompose()
        
        sections = []
        
        # 找到所有 h 标签（按文档顺序）
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        
        if not headings:
            # 如果没有 h 标签，返回整个页面的文本
            text = soup.get_text(separator='\n', strip=True)
            if text:
                sections.append({
                    'heading': '',
                    'content': text
                })
            return sections
        
        # 获取 body 或整个文档
        body = soup.find('body') or soup
        
        # 处理每个 h 标签及其后续内容
        for i, heading in enumerate(headings):
            heading_text = heading.get_text(strip=True)
            
            # 找到下一个 h 标签（用于确定内容范围）
            next_heading = headings[i + 1] if i + 1 < len(headings) else None
            
            # 找到当前 h 标签和下一个 h 标签之间的所有元素
            content_parts = []
            
            # 获取所有元素（按文档顺序）
            all_elements = list(body.descendants)
            
            try:
                # 找到当前 heading 的位置
                start_idx = all_elements.index(heading)
                # 找到下一个 heading 的位置（如果有）
                end_idx = all_elements.index(next_heading) if next_heading else len(all_elements)
                
                # 提取之间的元素
                for elem in all_elements[start_idx + 1:end_idx]:
                    # 跳过 h 标签
                    if hasattr(elem, 'name') and elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                        break
                    
                    # 提取文本
                    if isinstance(elem, str):
                        text = elem.strip()
                    elif hasattr(elem, 'string') and elem.string:
                        text = str(elem.string).strip()
                    elif hasattr(elem, 'get_text'):
                        # 只获取直接文本，不递归（避免重复）
                        text = ''.join([str(child) for child in elem.children if isinstance(child, str)]).strip()
                    else:
                        continue
                    
                    if text and len(text) > 3:
                        content_parts.append(text)
            except (ValueError, AttributeError):
                # 遍历所有元素，找到在 heading 之后的内容
                found_heading = False
                for elem in all_elements:
                    if elem == heading:
                        found_heading = True
                        continue
                    if next_heading and elem == next_heading:
                        break
                    if found_heading:
                        if hasattr(elem, 'name') and elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                            break
                        if isinstance(elem, str):
                            text = elem.strip()
                            if text and len(text) > 3:
                                content_parts.append(text)
            
            # 合并内容（去重但保持顺序）
            seen = set()
            unique_parts = []
            for part in content_parts:
                part_clean = part.strip()
                part_lower = part_clean.lower()
                if part_lower and part_lower not in seen and len(part_clean) > 3:
                    seen.add(part_lower)
                    unique_parts.append(part_clean)
            
            content = ' '.join(unique_parts).strip()
            
            # 如果内容不为空，添加到 sections
            if content or heading_text:
                sections.append({
                    'heading': heading_text,
                    'content': content
                })
        
        # 如果第一个 h 标签之前有内容，也添加进去
        first_heading = headings[0]
        try:
            all_elements = list(body.descendants)
            first_idx = all_elements.index(first_heading)
            before_parts = []
            for elem in all_elements[:first_idx]:
                if hasattr(elem, 'name') and elem.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                    continue
                if isinstance(elem, str):
                    text = elem.strip()
                    if text and len(text) > 3:
                        before_parts.append(text)
            if before_parts:
                sections.insert(0, {
                    'heading': '',
                    'content': ' '.join(before_parts).strip()
                })
        except (ValueError, AttributeError):
            pass
        
        return sections
    
    def is_frame_structure(self, html: str) -> bool:
        """
        检测页面是否为 frame 架构
        Args:
            html: HTML 内容
        Returns:
            True 如果是 frame 架构，False 否则
        """
        if not html:
            return False
        
        # 检测 frame/iframe 标签
        frame_indicators = [
            '<frame',
            '<iframe',
            '<frameset',
            'frame src',
            'iframe src'
        ]
        html_lower = html.lower()
        for indicator in frame_indicators:
            if indicator in html_lower:
                return True
        # 检测是否有多个 h 标签（frame 架构通常会有多个 h 标签分段）
        try:
            soup = BeautifulSoup(html, 'html.parser')
            headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
            # 如果有 3 个或更多 h 标签，可能是 frame 架构
            if len(headings) >= 3:
                return True
        except:
            pass
        return False
    def pre_clean_text(self, text: str) -> str:
        """
        预清洗文本
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        # 规范化空白字符
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        # 去除常见无关内容
        noise_patterns = [
            r'(?i)(copyright|©).*?\d{4}.*',
            r'(?i)all rights reserved.*',
            r'(?i)(login|sign in|register).*',
            r'(?i)cookie policy.*',
            r'(?i)privacy policy.*',
        ]
        for pattern in noise_patterns:
            text = re.sub(pattern, '', text, flags=re.MULTILINE)
        # 去除连续的特殊字符
        text = re.sub(r'[=\-_]{5,}', '', text)
        return text.strip()
    def pre_extract_contacts(self, text: str) -> Dict[str, str]:
        """
        正则预提取联系方式
        Args:
            text: 文本内容
        Returns:
            提取的联系信息
        """
        contacts = {}
        # Email
        emails = re.findall(r'[\w\.-]+@[\w\.-]+\.\w+', text)
        if emails:
            contacts['email'] = emails[0]  # 取第一个
        # GitHub
        github = re.findall(r'https?://github\.com/[\w\-]+', text, re.IGNORECASE)
        if github:
            contacts['github'] = github[0]
        # Google Scholar
        scholar = re.findall(r'https?://scholar\.google\.[^\s<>\)]+', text, re.IGNORECASE)
        if scholar:
            contacts['google_scholar'] = scholar[0]
        # Twitter/X
        twitter = re.findall(r'https?://(?:twitter|x)\.com/[\w]+', text, re.IGNORECASE)
        if twitter:
            contacts['twitter'] = twitter[0]
        # ORCID
        orcid = re.findall(r'https?://orcid\.org/[\d\-]+', text, re.IGNORECASE)
        if orcid:
            contacts['orcid'] = orcid[0]
        # LinkedIn
        linkedin = re.findall(r'https?://(?:www\.)?linkedin\.com/in/[\w\-]+', text, re.IGNORECASE)
        if linkedin:
            contacts['linkedin'] = linkedin[0]
        return contacts
    def call_llm(self, prompt: str, max_tokens: int = 2000) -> Optional[Dict]:
        """
        调用 LLM 提取信息
        Args:
            prompt: 提示词
            max_tokens: 最大 token 数 (ignored, using config default)
        Returns:
            解析后的 JSON 结果
        """
        if not self.llm_available or not self.llm:
            return None
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise information extraction assistant. Always return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
            response = self.llm.invoke(messages)
            result_text = response.content.strip()
            # 移除可能的 markdown 代码块标记
            result_text = re.sub(r'^```json?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            return json.loads(result_text)
        except Exception as e:
            print(f"[HomepageParser] LLM 调用失败: {e}")
            return None
    def parse_homepage(self, url: str, use_interactive: bool = None) -> Dict:
        """
        解析个人主页
        
        Args:
            url: 个人主页 URL
            use_interactive: 是否使用交互式抓取（None=自动判断，True=强制使用，False=禁用）
        
        Returns:
            提取的结构化信息
        """
        print(f"[HomepageParser] 正在解析个人主页: {url}")
        
        # 决定是否使用交互式抓取
        should_use_interactive = False
        if use_interactive is True and self.enable_interactive:
            should_use_interactive = True
        elif use_interactive is None and self.enable_interactive:
            # 自动判断：检查 URL 是否可能包含动态内容
            dynamic_indicators = ['scholar', 'researchgate', 'linkedin', 'profile']
            should_use_interactive = any(indicator in url.lower() for indicator in dynamic_indicators)
        
        # 1. 获取页面内容
        text = None
        metadata = {}
        html = None  # 初始化 html 变量
        
        if should_use_interactive and self.web_agent:
            print(f"[HomepageParser] 🤖 Using Nine Dimension Aware Agent (with LLM filtering)")
            try:
                # 使用九维度感知的抓取方法
                text, metadata = self.web_agent.fetch_for_nine_dimensions(
                    url=url,
                    author_name="",  # 可以从 URL 或其他地方提取作者名
                    max_interactions=50,  # 全量爬取模式
                    required_dimensions=None  # None=全部九维度
                )
                print(f"[HomepageParser] ✅ Nine-dimension-aware fetch complete: {len(text)} chars")
                if metadata.get('dimension_coverage'):
                    coverage = metadata['dimension_coverage']
                    print(f"[HomepageParser] 📊 Dimension coverage: {coverage.get('covered', 0)}/{coverage.get('total', 9)} dimensions")
            except Exception as e:
                print(f"[HomepageParser] ⚠️ Nine-dimension-aware fetch failed, falling back to static: {e}")
                should_use_interactive = False
        
        if not should_use_interactive or not text:
            # 使用传统静态抓取
            print(f"[HomepageParser] 📄 Using static fetch")
            # 1. 获取 HTML
            html = self.fetch_page(url)
            if not html:
                return {'error': '无法获取网页内容'}
            # 2. 提取结构化文本
            text = self.extract_structured_text(html)
            if not text or len(text) < 50:
                return {'error': '无法提取有效文本'}
        
        print(f"[HomepageParser] 提取文本长度: {len(text)} 字符")
        # 3. 预清洗
        text = self.pre_clean_text(text)
        # 4. 预提取联系方式
        contacts = self.pre_extract_contacts(text)
        # 5. 使用 LLM 提取完整信息
        profile = {}
        if self.llm_available:
            # 检测是否为 frame 架构（且使用静态抓取）
            is_frame = False
            if not should_use_interactive and html is not None:
                is_frame = self.is_frame_structure(html)
            
            if is_frame:
                # Frame 架构模式：每 3 个段落调用一次 LLM
                print(f"[HomepageParser] 检测到 frame 架构，使用段落分组模式（每 3 个段落一组）")
                sections = self.extract_text_by_heading_sections(html)
                print(f"[HomepageParser] 提取到 {len(sections)} 个段落")
                # 过滤空段落
                valid_sections = [s for s in sections if s.get('content') and len(s.get('content', '').strip()) >= 20]
                if not valid_sections:
                    print(f"[HomepageParser] ⚠️ 没有有效的段落，回退到一次性提取")
                    is_frame = False
                else:
                    # 每 3 个段落组成一批，调用一次 LLM
                    all_results = []
                    batch_size = 3
                    for batch_start in range(0, len(valid_sections), batch_size):
                        batch_end = min(batch_start + batch_size, len(valid_sections))
                        batch_sections = valid_sections[batch_start:batch_end]
                        
                        # 构建批次内容
                        batch_content_parts = []
                        for i, section in enumerate(batch_sections):
                            heading = section.get('heading', '') or f"Section {batch_start + i + 1}"
                            content = section.get('content', '').strip()
                            if content:
                                batch_content_parts.append(f"## {heading}\n{content}")
                        
                        if not batch_content_parts:
                            continue
                        
                        batch_content = '\n\n'.join(batch_content_parts)
                        # 限制每批内容长度（避免过长）
                        batch_content = batch_content[:15000]
                        print(f"[HomepageParser] 处理批次 {batch_start // batch_size + 1} (段落 {batch_start + 1}-{batch_end}/{len(valid_sections)}): {len(batch_content)} 字符")
                        
                        prompt = f"""Extract structured information from these sections of a personal homepage.

                        Content from {len(batch_sections)} sections:
                        {batch_content}

                        CRITICAL RULES FOR RESEARCH INTERESTS:
                        - ONLY extract from sections explicitly titled "Research Interests", "Research Areas", "Research Focus", or similar
                        - Extract only SHORT, CONCISE field/topic names (e.g., "Machine Learning", "Natural Language Processing", "Computer Vision")
                        - DO NOT extract long descriptions or sentences
                        - If the section contains descriptions (sentences with verbs like "developing", "focusing on"), extract only the core topic keywords
                        - DO NOT infer from publications, paper titles, or project names
                        - If no such section exists, return empty list []

                        Extract and return JSON with these fields (extract from all sections in this batch):
                        1. research_interests: list of SHORT research field names (2-8 words max, e.g., "Machine Learning", "Bioinformatics")
                        2. awards: list of {{name, year, organization}}
                        3. experience: list of {{position, organization, duration}}
                        4. teaching: list of {{course, role, semester}}

                        Return only valid JSON. If a field is not present in these sections, return an empty list [] for that field."""
                        result = self.call_llm(prompt, max_tokens=3000)
                        if result:
                            all_results.append(result)
                    # 合并所有批次的结果
                    if all_results:
                        merged_profile = {
                            'research_interests': [],
                            'awards': [],
                            'experience': [],
                            'teaching': []
                        }
                        # 合并所有字段
                        for result in all_results:
                            if 'research_interests' in result:
                                merged_profile['research_interests'].extend(result['research_interests'])
                            if 'awards' in result:
                                merged_profile['awards'].extend(result['awards'])
                            if 'experience' in result:
                                merged_profile['experience'].extend(result['experience'])
                            if 'teaching' in result:
                                merged_profile['teaching'].extend(result['teaching'])
                        # 去重（对于 research_interests）
                        seen_interests = set()
                        unique_interests = []
                        for interest in merged_profile['research_interests']:
                            interest_lower = str(interest).lower().strip()
                            if interest_lower and interest_lower not in seen_interests:
                                seen_interests.add(interest_lower)
                                unique_interests.append(interest)
                        merged_profile['research_interests'] = unique_interests
                        profile.update(merged_profile)
                        print(f"[HomepageParser] Frame 架构提取完成: {len(unique_interests)} 研究兴趣, {len(merged_profile['awards'])} 奖项, {len(merged_profile['experience'])} 经历")
            if not is_frame:
                # 原来的方式：一次性提取
                text_for_llm = text[:12000]
                
                prompt = f"""Extract structured information from this personal homepage.
                Text (first {len(text_for_llm)} chars):
                {text_for_llm}

                CRITICAL RULES FOR RESEARCH INTERESTS:
                - ONLY extract from sections explicitly titled "Research Interests", "Research Areas", "Research Focus", or similar
                - Extract only SHORT, CONCISE field/topic names (e.g., "Machine Learning", "Natural Language Processing", "Computer Vision")
                - DO NOT extract long descriptions or sentences
                - If the section contains descriptions (sentences with verbs like "developing", "focusing on"), extract only the core topic keywords
                - DO NOT infer from publications, paper titles, or project names
                - If no such section exists, return empty list []

                Extract and return JSON with these fields:
                1. research_interests: list of SHORT research field names (2-8 words max, e.g., "Machine Learning", "Bioinformatics")
                2. awards: list of {{name, year, organization}}
                3. experience: list of {{position, organization, duration}}
                4. teaching: list of {{course, role, semester}}

                Return only valid JSON."""
                result = self.call_llm(prompt, max_tokens=3000)  # 增加输出 token 限制
                if result:
                    profile.update(result)
        # 6. 添加预提取的联系方式
        if contacts:
            profile['contact'] = contacts
        profile['_meta'] = {
            'homepage_url': url,
            'text_length': len(text)
        }
        return profile
    def parse_google_scholar(self, scholar_url: str) -> Dict:
        """
        解析 Google Scholar 个人页面
        使用信号量限制并发数（最多3个）+ 速率限制（3秒/请求）
        Args:
            scholar_url: Google Scholar URL
        Returns:
            提取的信息
        """
        global _SCHOLAR_SEMAPHORE, _SCHOLAR_LOCK, _LAST_SCHOLAR_REQUEST_TIME, _MIN_SCHOLAR_INTERVAL
        print(f"[HomepageParser] 正在解析 Google Scholar: {scholar_url}")
        # ========== 分级并发控制 ==========
        # 1. 信号量：最多3个线程同时访问Scholar
        with _SCHOLAR_SEMAPHORE:
            print(f"[Scholar Concurrency] Acquired semaphore slot (max 3 concurrent)")
            # 2. 速率限制：每个请求间隔至少3秒
            with _SCHOLAR_LOCK:
                current_time = time.time()
                elapsed = current_time - _LAST_SCHOLAR_REQUEST_TIME
                if elapsed < _MIN_SCHOLAR_INTERVAL:
                    wait_time = _MIN_SCHOLAR_INTERVAL - elapsed
                    print(f"[Scholar Rate Limit] Waiting {wait_time:.1f}s")
                    time.sleep(wait_time)
                _LAST_SCHOLAR_REQUEST_TIME = time.time()
            print(f"[Scholar] Proceeding with request...")
            # 3. 重试逻辑（Google Scholar 反爬严格）
            max_retries = 2
            html = None
            for attempt in range(max_retries):
                try:
                    html = self.fetch_page(scholar_url, use_scholar_session=True)
                    if html:
                        print(f"[Scholar] ✓ Successfully fetched page")
                        break  # 成功获取
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)  # 5秒, 10秒
                        print(f"[Scholar] Retry {attempt + 1}/{max_retries}, waiting {wait_time}s...")
                        time.sleep(wait_time)
                except Exception as e:
                    if attempt < max_retries - 1:
                        wait_time = 5 * (attempt + 1)
                        print(f"[Scholar] Attempt {attempt + 1} failed: {e}, retrying in {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        print(f"[Scholar] All retries failed: {e}")
                        return {'error': f'无法获取 Google Scholar 页面: {e}'}
            if not html:
                return {'error': '无法获取 Google Scholar 页面（重试后仍失败）'}
        # 解析在 semaphore 外部进行（释放slot给下一个线程）
        try:
            soup = BeautifulSoup(html, 'html.parser')
            scholar_data = {}
            # 提取作者名字
            name_elem = soup.find('div', {'id': 'gsc_prf_in'})
            if name_elem:
                scholar_data['name'] = name_elem.text.strip()
            # 提取机构
            affiliation_elem = soup.find('div', {'class': 'gsc_prf_il'})
            if affiliation_elem:
                scholar_data['affiliation'] = affiliation_elem.text.strip()
            # 提取研究兴趣
            interests = []
            interest_elems = soup.find_all('a', {'class': 'gsc_prf_inta'})
            for elem in interest_elems:
                interests.append(elem.text.strip())
            if interests:
                scholar_data['research_interests'] = interests
            # 提取引用统计
            citation_stats = {}
            stats_table = soup.find('table', {'id': 'gsc_rsb_st'})
            if stats_table:
                rows = stats_table.find_all('tr')
                if len(rows) >= 2:
                    values = [td.text.strip() for td in rows[1].find_all('td')]
                    
                    if len(values) >= 3:
                        citation_stats['citations_all'] = values[1]
                        citation_stats['h_index'] = values[2]
                        if len(values) >= 4:
                            citation_stats['i10_index'] = values[3]
            if citation_stats:
                scholar_data['citation_stats'] = citation_stats
            
            # 提取论文列表
            publications = []
            pub_rows = soup.find_all('tr', {'class': 'gsc_a_tr'})
            
            for row in pub_rows[:20]:  # 只提取前20篇
                pub = {}
                
                # 标题
                title_elem = row.find('a', {'class': 'gsc_a_at'})
                if title_elem:
                    pub['title'] = title_elem.text.strip()
                
                # 作者和期刊
                authors_elem = row.find('div', {'class': 'gs_gray'})
                if authors_elem:
                    pub['authors'] = authors_elem.text.strip()
                
                # 年份
                year_elem = row.find('span', {'class': 'gsc_a_h'})
                if year_elem:
                    pub['year'] = year_elem.text.strip()
                
                # 引用数
                cite_elem = row.find('a', {'class': 'gsc_a_ac'})
                if cite_elem and cite_elem.text.strip():
                    pub['citations'] = cite_elem.text.strip()
                
                if pub.get('title'):
                    publications.append(pub)
            
            if publications:
                scholar_data['publications'] = publications
            
            scholar_data['_meta'] = {
                'source': 'google_scholar',
                'url': scholar_url,
                'publications_count': len(publications)
            }
            
            print(f"[HomepageParser] ✓ Google Scholar 解析成功：{len(publications)} 篇论文")
            
            return scholar_data
            
        except Exception as e:
            print(f"[HomepageParser] ✗ Google Scholar 解析失败: {e}")
            return {'error': str(e)}
    
    def parse_orcid(self, orcid_url: str) -> Dict:
        """
        解析 ORCID 个人页面
        
        Args:
            orcid_url: ORCID URL
            
        Returns:
            提取的信息
        """
        print(f"[HomepageParser] 正在解析 ORCID: {orcid_url}")
        
        try:
            html = self.fetch_page(orcid_url)
            if not html:
                return {'error': '无法获取 ORCID 页面'}
            
            text = self.extract_structured_text(html)
            if not text or len(text) < 100:
                return {'error': '无法提取有效文本'}
            
            orcid_data = {}
            
            # 预提取联系方式
            contacts = self.pre_extract_contacts(text)
            if contacts:
                orcid_data['contact'] = contacts
            
            # 使用 LLM 提取教育和工作经历
            if self.llm_available:
                prompt = f"""Extract education and employment information from this ORCID profile.

Text:
{text[:3000]}

Return JSON:
{{
  "employment": [{{"position": "...", "institution": "...", "duration": "..."}}],
  "education": [{{"degree": "...", "institution": "...", "duration": "..."}}]
}}

Return only valid JSON."""

                result = self.call_llm(prompt)
                if result:
                    orcid_data.update(result)
            
            orcid_data['_meta'] = {
                'source': 'orcid',
                'url': orcid_url
            }
            
            print(f"[HomepageParser] ✓ ORCID 解析成功")
            
            return orcid_data
            
        except Exception as e:
            print(f"[HomepageParser] ✗ ORCID 解析失败: {e}")
            return {'error': str(e)}
    
    def parse_dblp(self, dblp_url: str) -> Dict:
        """
        解析 DBLP 个人页面
        
        Args:
            dblp_url: DBLP URL
            
        Returns:
            提取的信息
        """
        print(f"[HomepageParser] 正在解析 DBLP: {dblp_url}")
        
        try:
            html = self.fetch_page(dblp_url)
            if not html:
                return {'error': '无法获取 DBLP 页面'}
            
            soup = BeautifulSoup(html, 'html.parser')
            
            dblp_data = {}
            publications = []
            
            # DBLP 页面结构比较规范，直接解析
            pub_list = soup.find_all('li', class_='entry')
            
            for pub_elem in pub_list[:30]:  # 最多30篇
                pub = {}
                
                # 标题
                title_elem = pub_elem.find('span', class_='title')
                if title_elem:
                    pub['title'] = title_elem.text.strip()
                
                # 会议/期刊
                venue_elem = pub_elem.find('span', itemprop='isPartOf')
                if venue_elem:
                    pub['venue'] = venue_elem.text.strip()
                
                # 年份
                year_elem = pub_elem.find('span', itemprop='datePublished')
                if year_elem:
                    pub['year'] = year_elem.text.strip()
                
                if pub.get('title'):
                    publications.append(pub)
            
            if publications:
                dblp_data['publications'] = publications
            
            dblp_data['_meta'] = {
                'source': 'dblp',
                'url': dblp_url,
                'publications_count': len(publications)
            }
            
            print(f"[HomepageParser] ✓ DBLP 解析成功：{len(publications)} 篇论文")
            
            return dblp_data
            
        except Exception as e:
            print(f"[HomepageParser] ✗ DBLP 解析失败: {e}")
            return {'error': str(e)}
