"""
Nine-Dimension Aware Interactive Web Agent
九维度感知的智能网页交互代理
核心思路：
1. 明确告诉 LLM 我们需要的 9 个维度信息
2. LLM 根据当前页面内容判断哪些按钮/链接需要点击
3. 动态执行交互，直到收集足够的九维度数据
九维度：
1. Background (背景介绍)
2. Research Interests (研究兴趣)
3. Publications (论文发表)
4. Awards (获奖荣誉)
5. Services (学术服务)
6. Education (教育经历)
7. Experience (工作经历)
8. Teaching (教学经历)
9. Contact (联系方式)
"""
import json
import time
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
from uuid import uuid4
try:
    from playwright.sync_api import sync_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("[9D Agent] ⚠️ Playwright not available")
# 检测是否在 Streamlit 环境中（Playwright 不兼容）
try:
    import streamlit
    IN_STREAMLIT = True
    PLAYWRIGHT_AVAILABLE = False  # 强制禁用 Playwright
    print("[9D Agent] 🎯 Detected Streamlit environment - using Selenium instead")
except ImportError:
    IN_STREAMLIT = False

# 检查 Selenium 可用性（Streamlit 兼容）
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    import hashlib
    from typing import List, Dict
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException
    SELENIUM_AVAILABLE = True
    print("[9D Agent] ✅ Selenium available")
except ImportError:
    SELENIUM_AVAILABLE = False
    print("[9D Agent] ⚠️ Selenium not available")

from . import config
from .nine_dimension_extractor import (
    NineDimensionExtractor,
    EXTRACTED_DIMENSIONS_VERSION
)
from bs4 import BeautifulSoup
import re


# 九维度定义（用于 LLM 理解目标）
NINE_DIMENSIONS = {
    'background': {
        'name': 'Background',
        'description': 'Personal biography, research overview, current position',
        'keywords': ['about', 'bio', 'biography', 'introduction', 'overview', 'profile']
    },
    'research_interests': {
        'name': 'Research Interests',
        'description': 'Research areas, fields of interest, expertise',
        'keywords': ['research interests', 'research areas', 'interests', 'research', 'expertise']
    },
    'publications': {
        'name': 'Publications',
        'description': 'Papers, conference proceedings, journal articles',
        'keywords': ['publications', 'papers', 'research output', 'articles', 'selected publications']
    },
    'awards': {
        'name': 'Awards',
        'description': 'Honors, prizes, recognitions, fellowships',
        'keywords': ['awards', 'honors', 'achievements', 'prizes', 'recognition', 'fellowship']
    },
    'services': {
        'name': 'Academic Services',
        'description': 'Program committee, reviewing, editorial board',
        'keywords': ['service', 'committee', 'reviewer', 'editor', 'pc member', 'area chair']
    },
    'education': {
        'name': 'Education',
        'description': 'Degrees, universities, graduation dates',
        'keywords': ['education', 'academic background', 'degrees', 'phd', 'master', 'bachelor']
    },
    'experience': {
        'name': 'Work Experience',
        'description': 'Employment history, positions, institutions',
        'keywords': ['experience', 'employment', 'work', 'position', 'career', 'history']
    },
    'teaching': {
        'name': 'Teaching',
        'description': 'Courses taught, teaching activities',
        'keywords': ['teaching', 'courses', 'instruction', 'lectures', 'classes']
    },
    'contact': {
        'name': 'Contact Information',
        'description': 'Email, social media, office location',
        'keywords': ['contact', 'email', 'office', 'address', 'social media']
    }
}
@dataclass
class DimensionCoverage:
    """维度覆盖情况"""
    dimension_name: str
    covered: bool  # 是否已覆盖
    confidence: float  # 覆盖置信度 0-1
    content_preview: str  # 内容预览

def _print_dimension_content(dim_name: str, dim_data: Dict[str, Any], source: str = ""):
    """
    打印维度内容的摘要信息（用于调试）
    
    Args:
        dim_name: 维度名称
        dim_data: 维度数据字典
        source: 数据来源（可选）
    """
    if not dim_data:
        print(f"[9D Agent] 📋 {dim_name} content: (empty)")
        return
    
    source_prefix = f"[{source}] " if source else ""
    
    # 根据维度类型打印不同的摘要信息
    if dim_name == 'services':
        services_list = dim_data.get('services', [])
        talks_list = dim_data.get('invited_talks', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(services_list)} services, {len(talks_list)} talks")
        if services_list:
            for i, svc in enumerate(services_list[:3], 1):  # 只显示前3个
                role = svc.get('role', '') or 'N/A'
                venue = svc.get('venue', '') or 'N/A'
                year = svc.get('year', '') or 'N/A'
                print(f"  [{i}] {role} @ {venue} ({year})")
            if len(services_list) > 3:
                print(f"  ... and {len(services_list) - 3} more")
    elif dim_name == 'publications':
        pubs_list = dim_data.get('publications', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(pubs_list)} publications")
        if pubs_list:
            for i, pub in enumerate(pubs_list[:3], 1):  # 只显示前3个
                title = pub.get('title', '') or 'N/A'
                print(f"  [{i}] {title[:80]}...")
            if len(pubs_list) > 3:
                print(f"  ... and {len(pubs_list) - 3} more")
    elif dim_name == 'awards':
        awards_list = dim_data.get('awards', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(awards_list)} awards")
        if awards_list:
            for i, award in enumerate(awards_list[:3], 1):  # 只显示前3个
                name = award.get('name', '') or award.get('title', '') or 'N/A'
                year = award.get('year', '') or 'N/A'
                print(f"  [{i}] {name} ({year})")
            if len(awards_list) > 3:
                print(f"  ... and {len(awards_list) - 3} more")
    elif dim_name == 'education':
        edu_list = dim_data.get('education', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(edu_list)} entries")
        if edu_list:
            for i, edu in enumerate(edu_list[:3], 1):  # 只显示前3个
                degree = edu.get('degree', '') or 'N/A'
                institution = edu.get('institution', '') or 'N/A'
                print(f"  [{i}] {degree} @ {institution}")
            if len(edu_list) > 3:
                print(f"  ... and {len(edu_list) - 3} more")
    elif dim_name == 'experience':
        exp_list = dim_data.get('experiences', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(exp_list)} entries")
        if exp_list:
            for i, exp in enumerate(exp_list[:3], 1):  # 只显示前3个
                position = exp.get('position', '') or 'N/A'
                company = exp.get('company', '') or exp.get('institution', '') or 'N/A'
                print(f"  [{i}] {position} @ {company}")
            if len(exp_list) > 3:
                print(f"  ... and {len(exp_list) - 3} more")
    elif dim_name == 'career_education':
        career_edu_list = dim_data.get('career_education', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(career_edu_list)} entries")
        if career_edu_list:
            for i, item in enumerate(career_edu_list[:3], 1):  # 只显示前3个
                degree_or_pos = item.get('degree_or_position', '') or item.get('degree', '') or item.get('position', '') or 'N/A'
                institution = item.get('institution', '') or 'N/A'
                print(f"  [{i}] {degree_or_pos} @ {institution}")
            if len(career_edu_list) > 3:
                print(f"  ... and {len(career_edu_list) - 3} more")
    elif dim_name == 'research_interests':
        interests_list = dim_data.get('interests', [])
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(interests_list)} interests")
        if interests_list:
            interest_names = [i.get('name', '') or i.get('tag', '') for i in interests_list[:5]]
            print(f"  Tags: {', '.join(interest_names)}")
    elif dim_name == 'background':
        bio = dim_data.get('biography', '') or dim_data.get('text', '') or ''
        if bio:
            preview = bio[:100] + "..." if len(bio) > 100 else bio
            print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(bio)} chars - {preview}")
        else:
            print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: (empty)")
    elif dim_name == 'contact':
        email = dim_data.get('email', '') or 'N/A'
        social = dim_data.get('social_links', {}) or {}
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: email={email}, social_links={len(social)}")
    else:
        # 通用打印：显示所有字段的键
        keys = list(dim_data.keys())
        print(f"[9D Agent] 📋 {source_prefix}{dim_name} content: {len(keys)} fields - {', '.join(keys)}")

def _infer_target_dimensions_from_label(label: str, content_preview: str = "", panel_id: str = "") -> List[str]:
    """
    根据标签（tab名称/元素文本）、panel ID 和内容预览，智能推断应该提取哪些维度
    
    Args:
        label: 标签文本（如 "Publications", "Awards", "Services"）
        content_preview: 内容预览（可选，用于更精确判断）
        panel_id: Panel ID（可选，如 "publications", "awards" 等，通常比 label 更准确）
        
    Returns:
        目标维度列表（如果无法确定，返回 None 表示提取所有维度）
    """
    if not label and not panel_id:
        return None
    
    label_lower = (label or "").lower().strip()
    panel_id_lower = (panel_id or "").lower().strip()
    content_lower = (content_preview or "").lower()[:500]  # 只检查前500字符，避免过长
    combined_text = f"{label_lower} {panel_id_lower} {content_lower}"
    
    # 精确匹配：如果标签明确是某个维度，只提取该维度
    # 🆕 使用短语优先匹配：完整短语在前，单个词在后
    dimension_keywords_map = {
        'publications': ['publications', 'papers', 'research output', 'articles', 'selected publications', 'pubs', 'pub'],
        'awards': ['awards', 'honors', 'achievements', 'prizes', 'recognition', 'fellowship', 'grants', 'trophy'],
        'services': ['invited talks', 'invited talk', 'keynote talk', 'keynote talks', 'service', 'services', 'committee', 'reviewer', 'editor', 'pc member', 'area chair', 'program committee'],
        'education': ['education', 'academic background', 'degrees', 'phd', 'master', 'bachelor', 'university'],
        'experience': ['experience', 'employment', 'work', 'position', 'career', 'history', 'industry'],
        'teaching': ['teaching', 'courses', 'instruction', 'lectures', 'classes'],  # 🆕 移除 'talks'，避免与 invited talks 冲突
        'contact': ['contact', 'email', 'office', 'address', 'social media'],
        'research_interests': ['research interests', 'research areas', 'interests', 'research', 'expertise', 'focus'],
        'background': ['about', 'bio', 'biography', 'introduction', 'overview', 'profile']
    }
    
    matched_dims = []
    
    # 🆕 优先级1: 检查 panel_id（最准确，因为 panel_id 通常是语义化的）
    # 🆕 优先匹配完整短语，再匹配单个词
    if panel_id_lower:
        # 先检查完整短语（更精确）
        for dim_name, keywords in dimension_keywords_map.items():
            # 按长度排序，优先匹配长短语
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            for kw in sorted_keywords:
                if kw in panel_id_lower:
                    if dim_name not in matched_dims:
                        matched_dims.append(dim_name)
                    break  # 找到匹配后跳出，避免重复匹配
    
    # 🆕 优先级2: 检查 label（tab 名称/元素文本）
    # 🆕 优先匹配完整短语，再匹配单个词
    if label_lower:
        for dim_name, keywords in dimension_keywords_map.items():
            # 按长度排序，优先匹配长短语
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            for kw in sorted_keywords:
                if kw in label_lower:
                    if dim_name not in matched_dims:
                        matched_dims.append(dim_name)
                    break  # 找到匹配后跳出，避免重复匹配
    
    # 🆕 优先级3: 检查内容预览（如果前两者都没匹配到）
    # 🆕 优先匹配完整短语，再匹配单个词
    if not matched_dims and content_lower:
        # 从内容中提取关键词（更宽松的匹配）
        for dim_name, keywords in dimension_keywords_map.items():
            # 按长度排序，优先匹配长短语
            sorted_keywords = sorted(keywords, key=len, reverse=True)
            for kw in sorted_keywords:
                if kw in content_lower:
                    if dim_name not in matched_dims:
                        matched_dims.append(dim_name)
                    break  # 找到匹配后跳出，避免重复匹配
    
    # 如果只匹配到一个维度，只提取该维度
    if len(matched_dims) == 1:
        return matched_dims
    
    # 如果匹配到多个维度，返回这些维度（可能是组合页面，如 "Research & Publications"）
    if len(matched_dims) > 1:
        return matched_dims
    
    # 如果无法确定，返回 None（提取所有维度，保持向后兼容）
    return None

class NineDimensionAwareAgent:
    """九维度感知的交互代理"""
    
    def __init__(self, api_key: str = None, headless: bool = True):
        self.api_key = api_key
        self.headless = headless
        
        # 初始化 LLM
        self.llm_available = False
        self.llm = None
        try:
            from . import llm as llm_module
            self.llm = llm_module.get_llm("parse", temperature=0.1)
            self.llm_available = True
            print("[9D Agent] ✅ LLM initialized (Azure OpenAI)")
        except Exception as e:
            print(f"[9D Agent] ⚠️ LLM init failed: {e}") 
        
        # 🆕 初始化九维度提取器（用于每次点击后提取）
        self.nine_dim_extractor = None
        try:
            self.nine_dim_extractor = NineDimensionExtractor(api_key=api_key)
            print("[9D Agent] ✅ Nine Dimension Extractor initialized")
        except Exception as e:
            print(f"[9D Agent] ⚠️ Nine Dimension Extractor init failed: {e}") 

    def _wait_for_dom_ready(self, driver, timeout: int = 12) -> None:
        """
        等待页面出现 <body> 或 frame/frameset，兼容老式站点
        """
        if not SELENIUM_AVAILABLE:
            return
        try:
            WebDriverWait(driver, timeout).until(
                lambda d: (
                    len(d.find_elements(By.TAG_NAME, 'body')) > 0 or
                    len(d.find_elements(By.TAG_NAME, 'frame')) > 0 or
                    len(d.find_elements(By.TAG_NAME, 'iframe')) > 0 or
                    len(d.find_elements(By.TAG_NAME, 'frameset')) > 0
                )
            )
        except TimeoutException:
            print("[9D Agent] ⚠️ DOM未在限定时间内准备好（未检测到body/frame）")

    def _collect_text_from_frames(self, driver, max_depth: int = 4) -> Tuple[str, Dict[str, int]]:
        """
        递归遍历 frame/iframe，收集文本
        Returns: (文本, 统计信息)
        """
        stats = {
            'frame_count': 0,
            'iframe_count': 0,
            'max_depth_reached': 0,
        }
        collected: List[str] = []
        seen_hashes: Set[str] = set()

        def _collect(depth: int) -> None:
            if depth > max_depth:
                return
            stats['max_depth_reached'] = max(stats['max_depth_reached'], depth)
            try:
                bodies = driver.find_elements(By.TAG_NAME, 'body')
            except Exception:
                bodies = []

            for body in bodies:
                try:
                    text = body.text or ""
                except Exception:
                    text = ""
                normalized = re.sub(r'\s+', ' ', text).strip()
                if not normalized:
                    continue
                text_hash = hashlib.md5(normalized.encode('utf-8')).hexdigest()
                if text_hash in seen_hashes:
                    continue
                seen_hashes.add(text_hash)
                collected.append(text)

            if depth >= max_depth:
                return

            frame_elements = []
            try:
                frame_list = driver.find_elements(By.TAG_NAME, 'frame')
                stats['frame_count'] += len(frame_list)
                frame_elements.extend(frame_list)
            except Exception:
                pass

            try:
                iframe_list = driver.find_elements(By.TAG_NAME, 'iframe')
                stats['iframe_count'] += len(iframe_list)
                frame_elements.extend(iframe_list)
            except Exception:
                pass

            for frame in frame_elements:
                try:
                    driver.switch_to.frame(frame)
                    _collect(depth + 1)
                except Exception as frame_error:
                    try:
                        src = frame.get_attribute('src') or ''
                    except Exception:
                        src = ''
                    print(f"[9D Agent] ⚠️ 无法进入 frame (depth={depth+1}, src={src[:80]}): {str(frame_error)[:80]}")
                finally:
                    try:
                        driver.switch_to.parent_frame()
                    except Exception:
                        try:
                            driver.switch_to.default_content()
                        except Exception:
                            pass

        try:
            _collect(0)
        finally:
            try:
                driver.switch_to.default_content()
            except Exception:
                pass

        stats['segments'] = len(collected)
        stats['snippet_chars'] = sum(len(seg) for seg in collected)
        combined_text = "\n\n".join(collected)
        return combined_text, stats

    def _capture_page_text(
        self,
        driver,
        allow_frames: bool = True,
        label: str = "",
        fallback_to_source: bool = True
    ) -> str:
        """
        安全获取页面文本：
        1. 优先使用 <body>
        2. 若无 body，尝试遍历 frame/iframe
        3. 仍为空则使用 page_source 文本
        """
        self._wait_for_dom_ready(driver, timeout=15)
        text = ""
        try:
            body_elem = driver.find_element(By.TAG_NAME, 'body')
            text = body_elem.text or ""
        except Exception:
            text = ""

        if text and text.strip():
            return text

        if allow_frames:
            frame_text, stats = self._collect_text_from_frames(driver)
            if frame_text and frame_text.strip():
                label_suffix = f" ({label})" if label else ""
                total_frames = stats.get('frame_count', 0) + stats.get('iframe_count', 0)
                print(
                    f"[9D Agent] ℹ️ 使用 frame 文本{label_suffix}: "
                    f"frames={total_frames}, depth={stats.get('max_depth_reached', 0)}, "
                    f"chars={stats.get('snippet_chars', 0)}"
                )
                return frame_text

        if fallback_to_source:
            try:
                html = driver.page_source or ""
            except Exception:
                html = ""
            if html:
                try:
                    soup = BeautifulSoup(html, 'html.parser')
                    text = soup.get_text(separator='\n', strip=True)
                except Exception:
                    text = html
        return text or ""
    def fetch_for_nine_dimensions(
        self, 
        url: str, 
        author_name: str = "",
        max_interactions: int = 50,
        required_dimensions: List[str] = None
    ) -> Tuple[str, Dict[str, Any]]:
        """
        以九维度为目标进行智能交互式抓取
        - 每次点击/tab切换后，立即提取九维度信息
        - 累积提取结果，最终直接使用，无需再次处理所有文本
        
        Args:
            url: 目标 URL
            author_name: 作者姓名（用于上下文）
            max_interactions: 最大交互次数
            required_dimensions: 必需的维度列表（None=全部）
        Returns:
            (完整文本内容, 元数据)
            元数据中包含 extracted_dimensions，可直接用于构建 profile
        """
        if required_dimensions is None:
            required_dimensions = list(NINE_DIMENSIONS.keys())
        if author_name:
            print(f"[9D Agent] 👤 Author: {author_name}")
        if SELENIUM_AVAILABLE:
            print(f"[9D Agent] 🤖 Using Selenium for JavaScript rendering...")
            try:
                return self._fetch_with_selenium(url, author_name, max_interactions, required_dimensions)
            except Exception as e:
                print(f"[9D Agent] ⚠️ Selenium failed: {str(e)[:100]}")
                print(f"[9D Agent] 🔄 Falling back to enhanced static...")
                return self._fallback_enhanced_static(url, author_name, required_dimensions)
        else:
            print(f"[9D Agent] ℹ️ No dynamic rendering available, using enhanced static...")
            return self._fallback_enhanced_static(url, author_name, required_dimensions)
    def _fetch_with_selenium(
            self,
            url: str,
            author_name: str,
            max_interactions: int,
            required_dimensions: List[str]
        ) -> Tuple[str, Dict[str, Any]]:
        """
        Selenium智能爬取：获取所有可点击元素并按顺序点击，处理 tab/AJAX 内容。
        """
        driver = None
        try:
            # --- 启动浏览器 ---
            chrome_options = Options()
            if self.headless:
                chrome_options.add_argument('--headless=new')  # Updated headless mode
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            # chrome_options.add_argument('--remote-debugging-port=9222')  # Removed to prevent port conflicts and crashes
            chrome_options.add_argument('--disable-software-rasterizer')
            chrome_options.add_argument('--disable-extensions')
            chrome_options.add_argument('--window-size=1920,1080')
            import tempfile
            user_data_dir = tempfile.mkdtemp(prefix='chrome_user_data_')
            chrome_options.add_argument(f'--user-data-dir={user_data_dir}')
            chrome_options.add_argument(
                'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            #自动下载 chromedriver，并设置为系统变量
            import shutil
            import os
            
            # Check for common Linux Chromium paths (helpful for Docker/Linux)
            chromium_path = None
            
            # Try to find playwright browsers first
            try:
                # 尝试从 playwright 安装路径查找 (Linux default)
                home = os.path.expanduser("~")
                playwright_paths = [
                    os.path.join(home, ".cache/ms-playwright/chromium-*/chrome-linux/chrome"),
                    os.path.join(home, ".cache/ms-playwright/chromium-*/chrome-linux/chrome.exe")
                ]
                import glob
                for pattern in playwright_paths:
                    matches = glob.glob(pattern)
                    if matches:
                        chromium_path = matches[-1] # Use the latest one
                        print(f"[9D Agent] Found Playwright Chromium at: {chromium_path}")
                        break
            except Exception:
                pass

            if not chromium_path:
                # Avoid snap if possible or fallback
                std_paths = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"]
                for p in std_paths:
                    if os.path.exists(p):
                        chromium_path = p
                        print(f"[9D Agent] Found standard Chromium at: {chromium_path}")
                        break

            if not chromium_path:
                # Fallback to whatever 'which' finds (might be snap)
                chromium_path = shutil.which("chromium") or shutil.which("chromium-browser")
                if chromium_path:
                    print(f"[9D Agent] Found Chromium via path: {chromium_path}")

            if chromium_path and os.path.exists(chromium_path):
                chrome_options.binary_location = chromium_path
            else:
                 print("[9D Agent] No Chromium binary found explicitly, relying on WebDriver to find it...")

            try:
                # 1. Try using system installed chromedriver (common in Docker)
                service = None
                chromedriver_path = shutil.which("chromedriver") or "/usr/bin/chromedriver" or "/usr/lib/chromium-browser/chromedriver"
                
                if chromedriver_path and os.path.exists(chromedriver_path):
                     from selenium.webdriver.chrome.service import Service
                     service = Service(chromedriver_path)
                     print(f"[9D Agent] Using system chromedriver at: {chromedriver_path}")
                     driver = webdriver.Chrome(service=service, options=chrome_options)
                else:
                    # 2. Fallback to webdriver_manager
                    print("[9D Agent] System chromedriver not found, trying webdriver_manager...")
                    from webdriver_manager.chrome import ChromeDriverManager
                    from selenium.webdriver.chrome.service import Service
                    service = Service(ChromeDriverManager().install())
                    driver = webdriver.Chrome(service=service, options=chrome_options)

            except Exception as e:
                print(f"[9D Agent] ⚠️ Driver init with service failed: {e}, falling back to default...")
                try:
                    driver = webdriver.Chrome(options=chrome_options)
                except Exception as e2:
                    print(f"[9D Agent] ❌ All driver init attempts failed: {e2}")
                    raise e2

            driver.set_page_load_timeout(30)
            print(f"[9D Agent] 📄 Loading page: {url[:80]}...")
            driver.get(url)
            self._wait_for_dom_ready(driver, timeout=20)
            time.sleep(3)  # 等待 JS 渲染
            print(f"[9D Agent] ✅ Page loaded, extracting content...")
            # --- 初始化内容存储 ---
            initial_text = self._capture_page_text(driver, label="initial_load")
            if not initial_text.strip():
                print("[9D Agent] ⚠️ 初始页面文本为空，后续将依赖 frame/源码提取")
            structured_content = {
                'initial': {
                    'label': 'Initial Page',
                    'text': initial_text
                }
            }
            interaction_history = []
            
            # 🆕 初始化累积的九维度提取结果（每次点击后更新）
            accumulated_dimensions = {
                'background': None,
                'research_interests': None,
                'publications': None,
                'awards': None,
                'services': None,
                'education': None,
                'experience': None,
                'teaching': None,
                'contact': None
            }
            incremental_extracted_dimensions: Dict[str, Any] = {}
            def _missing_dims() -> List[str]:
                return [
                    dim for dim in required_dimensions
                    if accumulated_dimensions.get(dim) is None
                ]
            
            # 🆕 记录已提取的文本内容hash，避免重复提取相同内容
            extracted_text_hashes = set()
            
            # 🆕 对初始页面内容进行 LLM 提取（立即提取，避免内容堆积）
            # 即使没有 author_name，也可以进行增量提取（效果可能稍差，但仍有用）
            # 每次提取的结果会累积到 accumulated_dimensions，最终直接用于构建 profile
            if self.nine_dim_extractor and initial_text:
                # 🆕 检查内容是否已提取过（基于文本hash去重）
                import hashlib
                text_hash = hashlib.md5(initial_text[:1000].encode('utf-8')).hexdigest()[:12]  # 使用前1000字符的hash
                if text_hash not in extracted_text_hashes:
                    extracted_text_hashes.add(text_hash)
                    missing_dims = _missing_dims()
                    if not missing_dims:
                        print(f"[9D Agent] ⏭️ Skipping initial page extraction (all required dimensions already satisfied)")
                    else:
                        try:
                            print(f"[9D Agent] 🤖 Extracting {len(missing_dims)} remaining dimensions from initial page...")
                            dim_results = self.nine_dim_extractor.extract_from_text(
                                text=initial_text,
                                author_name=author_name or "",  # 允许空字符串
                                source_name='homepage_initial',
                                target_dimensions=missing_dims
                            )
                            for dim_name, dim_result in dim_results.items():
                                if dim_result.success and dim_result.data:
                                    accumulated_dimensions[dim_name] = dim_result.data
                                    print(f"[9D Agent] ✅ Extracted {dim_name} from initial page (will be used directly in profile)")
                                    _print_dimension_content(dim_name, dim_result.data, "initial page")
                            
                            extracted_dimensions_if_available = {
                                dim_name: dim_result.data
                                for dim_name, dim_result in dim_results.items()
                                if dim_result.success and dim_result.data
                            }
                            if extracted_dimensions_if_available:
                                incremental_extracted_dimensions.update(extracted_dimensions_if_available)
                                print(f"[9D Agent] 🧾 Saved incremental dimensions from initial page: {list(extracted_dimensions_if_available.keys())}")
                        except Exception as e:
                            print(f"[9D Agent] ⚠️ Failed to extract dimensions from initial page: {e}")
                else:
                    print(f"[9D Agent] ⏭️ Skipping initial page extraction (duplicate content detected)")
            # --- 预先提取隐藏但已存在于 DOM 中的内容 ---
            print(f"[9D Agent] 🔍 Extracting hidden sections...")
            hidden_sections = self._extract_hidden_sections(driver, required_dimensions)
            if hidden_sections:
                print(f"[9D Agent] ✅ Found {len(hidden_sections)} hidden sections")
            if hidden_sections:
                for idx, section in enumerate(hidden_sections, 1):
                    section_key = f"hidden_{idx}"
                    structured_content[section_key] = {
                        'label': section['label'],
                        'type': 'hidden_dom',
                        'text': section['text'],
                        'content_section': 'hidden_dom'
                    }
                    interaction_history.append({
                        'step': -(idx),
                        'element': section['label'],
                        'type': 'hidden_dom_capture',
                        'section_key': section_key,
                        'new_content_length': len(section['text']),
                        'is_icon': False,
                        'is_visual': False,
                        'has_emoji': False,
                        'matched_dimensions': section['matched_dimensions']
                    })
                    if section['matched_dimensions']:
                        dims = ', '.join(section['matched_dimensions'])
            
            # --- 🆕 优先处理所有 tab 内容（批量爬取） ---
            print(f"[9D Agent] 🔍 Processing tabs...")
            tab_contents, processed_tab_identifiers = self._fetch_all_tab_contents(driver)
            if tab_contents:
                print(f"[9D Agent] ✅ Extracted {len(tab_contents)} tab contents")
                for idx, tab_data in enumerate(tab_contents, 1):
                    tab_name = tab_data.get('tab_name', f'Tab_{idx}')
                    tab_content = tab_data.get('content', '')
                    tab_html = tab_data.get('html', '')
                    
                    if tab_content and len(tab_content.strip()) >= 20:
                        section_key = f"tab_{idx}_{tab_name.lower().replace(' ', '_')}"
                        structured_content[section_key] = {
                            'label': tab_name,
                            'type': 'tab',
                            'text': tab_content,
                            'html': tab_html,
                            'content_section': 'tab_content'
                        }
                        interaction_history.append({
                            'step': idx,
                            'element': tab_name,
                            'type': 'tab_batch',
                            'section_key': section_key,
                            'new_content_length': len(tab_content),
                            'is_icon': False,
                            'is_visual': False,
                            'has_emoji': False,
                            'matched_dimensions': []
                        })
                        
                        # 🆕 每次提取完 tab 内容后，立即调用 LLM 提取九维度信息（增量提取）
                        # 结果会累积到 accumulated_dimensions，最终直接用于构建 profile，无需再次处理文本
                        if self.nine_dim_extractor and tab_content:
                            # 🆕 检查内容是否已提取过（基于文本hash去重）
                            import hashlib
                            text_hash = hashlib.md5(tab_content[:1000].encode('utf-8')).hexdigest()[:12]  # 使用前1000字符的hash
                            if text_hash not in extracted_text_hashes:
                                extracted_text_hashes.add(text_hash)
                                try:
                                    missing_dims = _missing_dims()
                                    if not missing_dims:
                                        print(f"[9D Agent] ⏭️ Skipping tab '{tab_name}' extraction (all required dimensions already satisfied)")
                                        continue
                                    # 🆕 智能判断：根据 tab 名称、panel_id 和内容预览推断应该提取哪些维度
                                    panel_id = tab_data.get('panel_id', '') or ''
                                    target_dims = _infer_target_dimensions_from_label(
                                        label=tab_name, 
                                        content_preview=tab_content[:500],  # 🆕 增加内容预览长度
                                        panel_id=panel_id  # 🆕 传入 panel_id（更准确）
                                    )
                                    # 但如果推断失败，则 fallback 到缺失的维度
                                    if target_dims:
                                        # 优先提取推断出的维度（即使已存在，也可以用于补充/验证）
                                        # 但如果推断的维度中有缺失的维度，优先提取缺失的
                                        inferred_missing = [dim for dim in target_dims if dim in missing_dims]
                                        if inferred_missing:
                                            target_dims = inferred_missing  # 优先提取缺失的推断维度
                                        # 否则使用所有推断的维度（允许补充已有数据）
                                    else:
                                        # 推断失败，fallback 到缺失的维度
                                        target_dims = missing_dims
                                    if not target_dims:
                                        print(f"[9D Agent] ⏭️ Tab '{tab_name}' did not map to any remaining dimensions")
                                        continue
                                    print(f"[9D Agent] 🎯 Tab '{tab_name}' (panel_id='{panel_id}') → Extracting {len(target_dims)} dimensions: {', '.join(target_dims)}")
                                    
                                    dim_results = self.nine_dim_extractor.extract_from_text(
                                        text=tab_content,
                                        author_name=author_name or "",  # 允许空字符串
                                        source_name='homepage_tab',
                                        target_dimensions=target_dims  # 🆕 传入目标维度
                                    )
                                    # 更新累积的九维度结果（只更新非空的结果）
                                    for dim_name, dim_result in dim_results.items():
                                        if dim_result.success and dim_result.data:
                                            # 🆕 打印提取后的原始数据（用于调试）
                                            print(f"[9D Agent] 🔍 Raw extracted data for {dim_name} from tab '{tab_name}':")
                                            if dim_name == 'services':
                                                services_raw = dim_result.data.get('services', [])
                                                talks_raw = dim_result.data.get('invited_talks', [])
                                                print(f"  - services: {len(services_raw)} items")
                                                print(f"  - invited_talks: {len(talks_raw)} items")
                                                if services_raw:
                                                    for i, svc in enumerate(services_raw[:2], 1):
                                                        print(f"    [{i}] {svc}")
                                                if talks_raw:
                                                    for i, talk in enumerate(talks_raw[:2], 1):
                                                        print(f"    [{i}] {talk}")
                                            else:
                                                # 对于其他维度，打印前几个键
                                                keys = list(dim_result.data.keys())[:5]
                                                print(f"  - keys: {keys}")
                                            
                                            # 如果该维度还没有数据，或者新数据更完整，则更新
                                            if accumulated_dimensions[dim_name] is None:
                                                accumulated_dimensions[dim_name] = dim_result.data
                                                print(f"[9D Agent] ✅ Extracted {dim_name} from tab '{tab_name}' (will be used directly in profile)")
                                                _print_dimension_content(dim_name, dim_result.data, f"tab '{tab_name}'")
                                            else:
                                                # 🆕 打印合并前的数据
                                                print(f"[9D Agent] 🔍 Before merge - existing {dim_name}:")
                                                _print_dimension_content(dim_name, accumulated_dimensions[dim_name], "existing")
                                                
                                                # 合并数据（例如 publications 可以追加，会自动去重）
                                                old_data = accumulated_dimensions[dim_name]
                                                accumulated_dimensions[dim_name] = self._merge_dimension_data(
                                                    accumulated_dimensions[dim_name], 
                                                    dim_name, 
                                                    dim_result.data
                                                )
                                                print(f"[9D Agent] ✅ Merged {dim_name} from tab '{tab_name}' (will be used directly in profile)")
                                                _print_dimension_content(dim_name, accumulated_dimensions[dim_name], f"tab '{tab_name}' (merged)")
                                            
                                            incremental_extracted_dimensions[dim_name] = accumulated_dimensions[dim_name]
                                            print(f"[9D Agent] 🧾 Saved {dim_name} from tab '{tab_name}' into extracted_dimensions")
                                except Exception as e:
                                    print(f"[9D Agent] ⚠️ Failed to extract dimensions from tab '{tab_name}': {e}")
                            else:
                                print(f"[9D Agent] ⏭️ Skipping tab '{tab_name}' extraction (duplicate content detected)")
            # 将已处理的 tab 标识添加到 clicked_elements，避免后续重复点击
            clicked_elements = set(processed_tab_identifiers) if processed_tab_identifiers else set()
            
            # --- 获取所有可点击元素 ---
            print(f"[9D Agent] 🔍 Finding interactive elements...")
            all_elements = self._find_all_interactive_elements(driver)
            total_interactive_candidates = len(all_elements)
            interactive_candidate_counts = Counter(
                (elem.get('type') or 'unknown') for elem in all_elements
            )
            element_queue = []
            for elem in all_elements:
                elem_id = elem['text'] if elem['text'] else elem.get('id', str(uuid4()))
                elem_full_id = elem['text'] if elem['text'] else elem.get('id', str(uuid4()))
                elem_type = elem.get('type', 'unknown')
                
                # 🆕 排除已处理的 tab（使用多种匹配方式）
                is_processed_tab = False
                
                # 方法1: 检查文本是否匹配（只比较前50字符，避免长文本问题）
                if elem_id:
                    short_id = elem_id[:50].strip()
                    if short_id in clicked_elements:
                        is_processed_tab = True
                
                # 方法2: 如果是 tab 类型，检查位置是否匹配
                if not is_processed_tab and elem_type == 'tab':
                    try:
                        elem_obj = elem.get('element')
                        if elem_obj:
                            location = elem_obj.location
                            size = elem_obj.size
                            tag = elem_obj.tag_name
                            position_id = f"{tag}_{location['x']}_{location['y']}_{size['width']}_{size['height']}"
                            if position_id in clicked_elements:
                                is_processed_tab = True
                    except:
                        pass
                
                # 方法3: 检查 outerHTML hash
                if not is_processed_tab and elem_type == 'tab':
                    try:
                        elem_obj = elem.get('element')
                        if elem_obj:
                            outer_html = elem_obj.get_attribute('outerHTML') or ""
                            if outer_html:
                                import hashlib
                                html_hash = hashlib.md5(outer_html.encode()).hexdigest()[:12]
                                html_id = f"tab_html_{html_hash}"
                                if html_id in clicked_elements:
                                    is_processed_tab = True
                    except:
                        pass
                
                if is_processed_tab:
                    continue
                
                if elem_id not in [e.get('text') or e.get('id') for e in element_queue]:
                    element_queue.append(elem)
            
            pending_interactive_elements = len(element_queue)
            print(f"[9D Agent] 📋 Found {pending_interactive_elements} elements to interact with")
            # --- 循环点击 ---
            for step, target_elem in enumerate(element_queue):
                elem_label = target_elem['text'][:50] if target_elem['text'] else target_elem.get('id', 'unknown')
                elem_full_label = target_elem['text'] if target_elem['text'] else target_elem.get('id', str(uuid4()))
                elem_type = target_elem.get('type', 'unknown')

                if elem_full_label in clicked_elements:
                    continue

                print(f"[9D Agent] 🔘 Clicking element {step+1}/{len(element_queue)}: '{elem_label}' ({elem_type})")
                direct_href = target_elem.get('direct_href')
                if target_elem.get('direct_navigation') and direct_href:
                    success_direct, direct_text = self._direct_navigate(driver, direct_href, elem_label)
                    if success_direct and direct_text:
                        section_key = f"section_{step+1}_direct"
                        structured_content[section_key] = {
                            'label': elem_label,
                            'type': 'direct_navigation',
                            'text': direct_text,
                            'content_section': 'direct_navigation'
                        }
                        interaction_history.append({
                            'step': step + 1,
                            'element': elem_label,
                            'type': 'direct_navigation',
                            'section_key': section_key,
                            'new_content_length': len(direct_text),
                            'is_icon': target_elem.get('is_icon_only', False),
                            'is_visual': target_elem.get('is_visual_only', False),
                            'has_emoji': target_elem.get('has_emoji', False),
                            'matched_dimensions': target_elem.get('matched_dimensions', []),
                            'direct_url': direct_href
                        })
                        
                        # 🆕 每次提取完直接导航内容后，立即调用 LLM 提取九维度信息（增量提取）
                        # 结果会累积到 accumulated_dimensions，最终直接用于构建 profile
                        if self.nine_dim_extractor and direct_text:
                            # 🆕 检查内容是否已提取过（基于文本hash去重）
                            import hashlib
                            text_hash = hashlib.md5(direct_text[:1000].encode('utf-8')).hexdigest()[:12]  # 使用前1000字符的hash
                            if text_hash not in extracted_text_hashes:
                                extracted_text_hashes.add(text_hash)
                                try:
                                    missing_dims = _missing_dims()
                                    if not missing_dims:
                                        print(f"[9D Agent] ⏭️ Skipping direct nav '{elem_label}' extraction (all dimensions already satisfied)")
                                        continue
                                    # 🆕 智能判断：根据元素标签和内容预览推断应该提取哪些维度
                                    target_dims = _infer_target_dimensions_from_label(
                                        label=elem_label, 
                                        content_preview=direct_text[:500]  # 🆕 增加内容预览长度
                                    )
                                    if target_dims:
                                        target_dims = [dim for dim in target_dims if dim in missing_dims]
                                    if not target_dims:
                                        target_dims = missing_dims
                                    if not target_dims:
                                        print(f"[9D Agent] ⏭️ Direct nav '{elem_label}' did not map to any remaining dimensions")
                                        continue
                                    print(f"[9D Agent] 🎯 Direct nav '{elem_label}' → Extracting {len(target_dims)} dimensions: {', '.join(target_dims)}")
                                    
                                    dim_results = self.nine_dim_extractor.extract_from_text(
                                        text=direct_text,
                                        author_name=author_name or "",  # 允许空字符串
                                        source_name='homepage_direct_nav',
                                        target_dimensions=target_dims  # 🆕 传入目标维度
                                    )
                                    # 更新累积的九维度结果
                                    for dim_name, dim_result in dim_results.items():
                                        if dim_result.success and dim_result.data:
                                            if accumulated_dimensions[dim_name] is None:
                                                accumulated_dimensions[dim_name] = dim_result.data
                                                print(f"[9D Agent] ✅ Extracted {dim_name} from direct nav '{elem_label}' (will be used directly in profile)")
                                                _print_dimension_content(dim_name, dim_result.data, f"direct nav '{elem_label}'")
                                            else:
                                                accumulated_dimensions[dim_name] = self._merge_dimension_data(
                                                    accumulated_dimensions[dim_name], 
                                                    dim_name, 
                                                    dim_result.data
                                                )
                                                print(f"[9D Agent] ✅ Merged {dim_name} from direct nav '{elem_label}' (will be used directly in profile)")
                                                _print_dimension_content(dim_name, accumulated_dimensions[dim_name], f"direct nav '{elem_label}' (merged)")
                                except Exception as e:
                                    print(f"[9D Agent] ⚠️ Failed to extract dimensions from direct nav '{elem_label}': {e}")
                            else:
                                print(f"[9D Agent] ⏭️ Skipping direct nav '{elem_label}' extraction (duplicate content detected)")
                        
                        clicked_elements.add(elem_full_label)
                        time.sleep(1)
                        continue

                # 点击元素并抓取 panel 文本
                success, new_content, content_section = self._selenium_click_element_structured(driver, target_elem, elem_label)
                clicked_elements.add(elem_full_label)
                if success and new_content:
                    print(f"[9D Agent] ✅ Extracted {len(new_content)} chars from '{elem_label}'")
                    section_key = f"section_{step+1}_{elem_type}"
                    structured_content[section_key] = {
                        'label': elem_label,
                        'type': elem_type,
                        'text': new_content,
                        'content_section': content_section
                    }
                    interaction_history.append({
                        'step': step + 1,
                        'element': elem_label,
                        'type': elem_type,
                        'section_key': section_key,
                        'new_content_length': len(new_content),
                        'is_icon': target_elem.get('is_icon_only', False),
                        'is_visual': target_elem.get('is_visual_only', False),
                        'has_emoji': target_elem.get('has_emoji', False),
                        'matched_dimensions': target_elem.get('matched_dimensions', [])
                    })
                    
                    # 🆕 每次提取完内容后，立即调用 LLM 提取九维度信息（增量提取）
                    # 结果会累积到 accumulated_dimensions，最终直接用于构建 profile，无需再次处理文本
                    if self.nine_dim_extractor and new_content:
                        # 🆕 检查内容是否已提取过（基于文本hash去重）
                        import hashlib
                        text_hash = hashlib.md5(new_content[:1000].encode('utf-8')).hexdigest()[:12]  # 使用前1000字符的hash
                        if text_hash not in extracted_text_hashes:
                            extracted_text_hashes.add(text_hash)
                            try:
                                missing_dims = _missing_dims()
                                if not missing_dims:
                                    print(f"[9D Agent] ⏭️ Skipping '{elem_label}' extraction (all dimensions already satisfied)")
                                    continue
                                # 尝试从多个来源获取 panel_id
                                panel_id_for_inference = ''
                                if target_elem:
                                    panel_id_for_inference = target_elem.get('panel_id', '') or target_elem.get('panel_lookup', {}).get('inferred_panel_id', '') or ''
                                # 如果 target_elem 中没有，尝试从 elem_label 推断（转换为小写，去除特殊字符）
                                if not panel_id_for_inference and elem_label:
                                    import re
                                    panel_id_for_inference = re.sub(r'[^a-z0-9_-]', '', elem_label.lower().strip())
                                
                                target_dims = _infer_target_dimensions_from_label(
                                    label=elem_label, 
                                    content_preview=new_content[:500],
                                    panel_id=panel_id_for_inference  # 传入 panel_id 提高准确性
                                )
                                # 但如果推断失败，则 fallback 到缺失的维度
                                if target_dims:
                                    # 优先提取推断出的维度（即使已存在，也可以用于补充/验证）
                                    # 但如果推断的维度中有缺失的维度，优先提取缺失的
                                    inferred_missing = [dim for dim in target_dims if dim in missing_dims]
                                    if inferred_missing:
                                        target_dims = inferred_missing  # 优先提取缺失的推断维度
                                    # 否则使用所有推断的维度（允许补充已有数据）
                                else:
                                    # 推断失败，fallback 到缺失的维度
                                    target_dims = missing_dims
                                if not target_dims:
                                    print(f"[9D Agent] ⏭️ Element '{elem_label}' did not map to any remaining dimensions")
                                    continue
                                print(f"[9D Agent] 🎯 Element '{elem_label}' → Extracting {len(target_dims)} dimensions: {', '.join(target_dims)}")
                                
                                dim_results = self.nine_dim_extractor.extract_from_text(
                                    text=new_content,
                                    author_name=author_name or "",  # 允许空字符串
                                    source_name='homepage_interaction',
                                    target_dimensions=target_dims  # 🆕 传入目标维度
                                )
                                # 更新累积的九维度结果（只更新非空的结果）
                                for dim_name, dim_result in dim_results.items():
                                    if dim_result.success and dim_result.data:
                                        # 如果该维度还没有数据，或者新数据更完整，则更新
                                        if accumulated_dimensions[dim_name] is None:
                                            accumulated_dimensions[dim_name] = dim_result.data
                                            print(f"[9D Agent] ✅ Extracted {dim_name} from '{elem_label}' (will be used directly in profile)")
                                            _print_dimension_content(dim_name, dim_result.data, f"element '{elem_label}'")
                                        else:
                                            # 合并数据（例如 publications 可以追加，会自动去重）
                                            accumulated_dimensions[dim_name] = self._merge_dimension_data(
                                                accumulated_dimensions[dim_name], 
                                                dim_name, 
                                                dim_result.data
                                            )
                                            print(f"[9D Agent] ✅ Merged {dim_name} from '{elem_label}' (will be used directly in profile)")
                                            _print_dimension_content(dim_name, accumulated_dimensions[dim_name], f"element '{elem_label}' (merged)")
                                            
                                        incremental_extracted_dimensions[dim_name] = accumulated_dimensions[dim_name]
                                        print(f"[9D Agent] 🧾 Saved {dim_name} from '{elem_label}' into extracted_dimensions")
                            except Exception as e:
                                print(f"[9D Agent] ⚠️ Failed to extract dimensions from '{elem_label}': {e}")
                        else:
                            print(f"[9D Agent] ⏭️ Skipping '{elem_label}' extraction (duplicate content detected)")
                else:
                    pass

                time.sleep(1)
            
            print(f"[9D Agent] 📊 Processing extracted content...")
            # --- 合并与处理 ---
            full_text = self._merge_structured_content(structured_content)
            
            # 🆕 移除智能摘要阶段，不再生成 extraction_text
            # final_text = self._create_extraction_text(structured_content, section_groups)

            final_coverage = self._analyze_dimension_coverage(full_text, required_dimensions)
            covered_count = sum(1 for c in final_coverage.values() if c.covered)
            
            # 🆕 统计已提取的九维度数量
            extracted_dims_count = sum(1 for v in accumulated_dimensions.values() if v is not None)
            extracted_dimensions_final = {k: v for k, v in accumulated_dimensions.items() if v is not None}
            
            print(f"[9D Agent] ✅ Completed: {len(full_text)} chars, {covered_count}/{len(required_dimensions)} dimensions covered")
            print(f"[9D Agent] 📊 Extracted {extracted_dims_count}/9 dimensions via incremental LLM extraction")
            
            # 🆕 说明：增量提取的结果会直接用于构建 profile，无需再次处理所有文本
            if extracted_dims_count > 0:
                print(f"[9D Agent] 🎯 Incremental extraction results will be used directly in profile building:")
                for dim_name in extracted_dimensions_final.keys():
                    print(f"   ✅ {dim_name}")
            
            extracted_dimensions_meta = {
                'version': EXTRACTED_DIMENSIONS_VERSION,
                'generated_at': time.time(),
                'source': 'nine_dimension_aware_agent',
                'dimensions': list(extracted_dimensions_final.keys()),
                'quality_flags': {},
                'quality_reasons': {}
            }
            
            interactive_diagnostics = {
                'total_candidates': total_interactive_candidates,
                'queue_length': pending_interactive_elements,
                'candidates_by_type': dict(interactive_candidate_counts),
                'interactions_recorded': len(interaction_history)
            }
            metadata = {
                'url': url,
                'author_name': author_name,
                'method': 'selenium_interactive_segmented',
                'total_interactions': len(interaction_history),
                'total_text_length': len(full_text),
                'extraction_text_length': 0,  # 🆕 不再使用 extraction_text（增量提取已处理）
                'dimension_coverage': {
                    'covered': covered_count,
                    'total': len(required_dimensions),
                    'percentage': covered_count / len(required_dimensions) * 100
                },
                'dimensions_found': [d for d, c in final_coverage.items() if c.covered],
                'dimensions_missing': [d for d, c in final_coverage.items() if not c.covered],
                'has_interactive_candidates': total_interactive_candidates > 0,
                'interactive_candidates': total_interactive_candidates,
                'interactive_queue_length': pending_interactive_elements,
                'interactive_candidates_by_type': dict(interactive_candidate_counts),
                'interactive_diagnostics': interactive_diagnostics,
                'interaction_history': interaction_history,
                # 🆕 保存累积的九维度提取结果（每次爬取后立即提取，直接用于构建 profile）
                # 这些结果会被 MultiSourceNineDimensionCollector 优先使用，避免重复处理文本
                'extracted_dimensions': extracted_dimensions_final,
                'extracted_dimensions_meta': extracted_dimensions_meta,
                'extracted_dimensions_count': extracted_dims_count,
                'homepage_metadata': {
                    'extracted_dimensions': incremental_extracted_dimensions,
                    'extracted_dimensions_meta': extracted_dimensions_meta,
                    'interactive_diagnostics': interactive_diagnostics
                }
            }
            driver.quit()
            # 🆕 Cleanup user data dir
            if 'user_data_dir' in locals() and user_data_dir and os.path.exists(user_data_dir):
                try:
                    shutil.rmtree(user_data_dir, ignore_errors=True)
                except:
                    pass
            return full_text, metadata
        except Exception as e:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
            # 🆕 Cleanup user data dir
            if 'user_data_dir' in locals() and user_data_dir and os.path.exists(user_data_dir):
                try:
                    shutil.rmtree(user_data_dir, ignore_errors=True)
                except:
                    pass
            raise e
    def _fetch_all_tab_contents(self, driver) -> Tuple[List[Dict[str, Any]], Set[str]]:
        """
        批量爬取所有 tab 的内容（新思路）
        
        流程：
        1. 识别所有 tab 元素
        2. 依次点击每个 tab
        3. 等待内容加载
        4. 提取对应的 content
        5. 返回所有 tab 的内容和已处理的 tab 标识
        
        Returns:
            Tuple[List[Dict], Set[str]]: 
                - 每个 tab 的内容，格式为 [{'tab_name': str, 'content': str, 'html': str}, ...]
                - 已处理的 tab 标识集合，用于后续排除
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from bs4 import BeautifulSoup
        import re
        import hashlib
        
        results = []
        processed_tab_identifiers = set()  # 🆕 记录已处理的 tab 标识
        
        try:
            # Step 1: 定位所有 tab 元素
            tab_selectors = [
                '[role="tab"]',
                '.nav-link[role="tab"]',
                '.nav-tabs .nav-link',
                '.nav-tabs a',
                '.nav-tabs button',
                'ul.nav a[role="tab"]'
            ]
            
            tabs = []
            for selector in tab_selectors:
                try:
                    found_tabs = driver.find_elements(By.CSS_SELECTOR, selector)
                    for tab in found_tabs:
                        # 去重：基于元素位置和文本
                        if tab not in tabs and tab.is_displayed():
                            tabs.append(tab)
                except:
                    continue
            
            if not tabs:
                return results, processed_tab_identifiers
            
            print(f"[9D Agent] 📋 Found {len(tabs)} tabs to process")
            # 🆕 预先获取所有 tab-pane（用于顺序匹配）
            all_panes = []
            try:
                all_panes = driver.find_elements(By.CSS_SELECTOR, 
                    ".tab-pane, [role='tabpanel']")
            except:
                pass
            
            # Step 2: 循环点击每个 tab 并提取内容（使用顺序匹配策略）
            for i, tab in enumerate(tabs):
                print(f"[9D Agent] 📑 Processing tab {i+1}/{len(tabs)}...")
                try:
                    # 提取 tab 标识（用于日志）
                    tab_label = ""
                    try:
                        # 方法1: 文字
                        tab_label = tab.text.strip() if tab.text else ""
                        # 方法2: innerText/textContent
                        if not tab_label:
                            tab_label = (tab.get_attribute('innerText') or tab.get_attribute('textContent') or '').strip()
                        # 方法3: aria-label
                        if not tab_label:
                            tab_label = (tab.get_attribute('aria-label') or '').strip()
                        # 方法4: SVG 图标（data-icon）
                        if not tab_label:
                            try:
                                svg_info = driver.execute_script("""
                                    var el = arguments[0];
                                    var svg = el.querySelector('svg');
                                    if (svg) {
                                        var dataIcon = svg.getAttribute('data-icon') || '';
                                        var ariaLabel = svg.getAttribute('aria-label') || '';
                                        var title = svg.getAttribute('title') || '';
                                        return dataIcon || ariaLabel || title || '';
                                    }
                                    return '';
                                """, tab)
                                if svg_info:
                                    tab_label = svg_info.strip()
                            except:
                                pass
                        # 方法5: 从 tab 属性获取
                        if not tab_label:
                            try:
                                href = tab.get_attribute('href')
                                if href and href.startswith('#'):
                                    tab_label = href.lstrip('#')
                            except:
                                pass
                        if not tab_label:
                            tab_label = f"Tab_{i+1}"
                    except:
                        tab_label = f"Tab_{i+1}"
                    
                    # 滚动到 tab 并点击
                    try:
                        driver.execute_script("arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", tab)
                        time.sleep(0.3)
                    except:
                        pass
                    
                    # 🆕 记录点击前的内容hash（用于检测变化）
                    before_content_hash = None
                    try:
                        before_panel = driver.find_element(By.CSS_SELECTOR, 
                            ".tab-pane.show.active, [role='tabpanel'].show.active")
                        before_content_hash = hashlib.md5((before_panel.get_attribute('innerHTML') or "").encode()).hexdigest()[:8]
                    except:
                        pass
                    
                    # 点击 tab（使用 JS click 更可靠）
                    try:
                        driver.execute_script("arguments[0].click();", tab)
                    except:
                        try:
                            tab.click()
                        except:
                            continue
                    
                    # Step 3: 等待内容真正变化（通过hash比较）
                    content_changed = False
                    max_wait_time = 5
                    wait_interval = 0.2
                    waited_time = 0
                    
                    while waited_time < max_wait_time:
                        try:
                            current_panel = driver.find_element(By.CSS_SELECTOR, 
                                ".tab-pane.show.active, [role='tabpanel'].show.active")
                            current_content_hash = hashlib.md5((current_panel.get_attribute('innerHTML') or "").encode()).hexdigest()[:8]
                            
                            if before_content_hash is None or current_content_hash != before_content_hash:
                                content_changed = True
                                break
                        except:
                            pass
                        
                        time.sleep(wait_interval)
                        waited_time += wait_interval
                    
                    # 额外等待动画完成（Bootstrap fade）
                    time.sleep(0.5)
                    
                    # Step 4: 提取内容（改进的匹配策略 - 优先使用索引匹配，避免重复内容）
                    content_html = ""
                    content_text = ""
                    panel_id = "unknown"
                    
                    # 🆕 策略1: 尝试通过 tab 属性找到 panel id（最可靠）
                    panel_id_candidate = None
                    try:
                        href = tab.get_attribute('href')
                        if href and href.startswith('#'):
                            panel_id_candidate = href.lstrip('#').split()[0]
                    except:
                        pass
                    
                    if not panel_id_candidate:
                        try:
                            data_target = tab.get_attribute('data-bs-target') or tab.get_attribute('data-target')
                            if data_target and data_target.startswith('#'):
                                panel_id_candidate = data_target.lstrip('#').split()[0]
                        except:
                            pass
                    
                    if panel_id_candidate:
                        try:
                            panel = driver.find_element(By.ID, panel_id_candidate)
                            content_html = panel.get_attribute('innerHTML') or ""
                            content_text = panel.text.strip() if panel.text else ""
                            panel_id = panel_id_candidate
                            print(f"[9D Agent] 📄 Extracted from panel id='{panel_id}' (matched by tab attribute)")
                        except:
                            pass
                    
                    # 🆕 策略2: 尝试通过 tab 文本/ID 匹配 panel（适用于有文本的tab）
                    if (not content_text or len(content_text) < 20) and tab_label and tab_label != f"Tab_{i+1}":
                        # 将 tab_label 转换为可能的 panel id（小写，去除特殊字符）
                        import re
                        panel_id_candidate = re.sub(r'[^a-z0-9_-]', '', tab_label.lower().strip())
                        
                        if panel_id_candidate:
                            try:
                                panel = driver.find_element(By.ID, panel_id_candidate)
                                if panel.is_displayed() or 'show' in (panel.get_attribute('class') or '').lower():
                                    panel_content_html = panel.get_attribute('innerHTML') or ""
                                    panel_content_text = panel.text.strip() if panel.text else ""
                                    if panel_content_text and len(panel_content_text) >= 20:
                                        content_html = panel_content_html
                                        content_text = panel_content_text
                                        panel_id = panel_id_candidate
                                        print(f"[9D Agent] 📄 Extracted from panel id='{panel_id}' (matched by tab label)")
                            except:
                                pass
                    
                    # 🆕 策略3: 按索引匹配（第 i 个 tab 对应第 i 个 tab-pane）- 适用于图标tab
                    if not content_text or len(content_text) < 20:
                        if i < len(all_panes):
                            try:
                                panel = all_panes[i]
                                panel_content_html = panel.get_attribute('innerHTML') or ""
                                panel_content_text = panel.text.strip() if panel.text else ""
                                if panel_content_text and len(panel_content_text) >= 20:
                                    # 🆕 检查是否与之前提取的内容不同（避免重复）
                                    is_duplicate = False
                                    for prev_result in results:
                                        if prev_result.get('content') == panel_content_text:
                                            is_duplicate = True
                                            break
                                    
                                    if not is_duplicate:
                                        content_html = panel_content_html
                                        content_text = panel_content_text
                                        panel_id = panel.get_attribute('id') or f"panel_index_{i}"
                                        print(f"[9D Agent] 📄 Extracted from panel at index {i} (id='{panel_id}')")
                            except:
                                pass
                    
                    # 🆕 策略4: 提取当前 active 的 tab-pane（作为fallback，但要检查是否重复）
                    if not content_text or len(content_text) < 20:
                        try:
                            active_panel = driver.find_element(By.CSS_SELECTOR, 
                                ".tab-pane.show.active, [role='tabpanel'].show.active, .tab-pane.active")
                            active_panel_id = active_panel.get_attribute('id') or f"panel_index_{i}"
                            active_content_html = active_panel.get_attribute('innerHTML') or ""
                            active_content_text = active_panel.text.strip() if active_panel.text else ""
                            
                            if active_content_text and len(active_content_text) >= 20:
                                # 🆕 检查是否与之前提取的内容不同（避免重复）
                                is_duplicate = False
                                for prev_result in results:
                                    if prev_result.get('content') == active_content_text:
                                        is_duplicate = True
                                        break
                                
                                if not is_duplicate:
                                    content_html = active_content_html
                                    content_text = active_content_text
                                    panel_id = active_panel_id
                                    print(f"[9D Agent] 📄 Extracted from active panel (id='{panel_id}', changed={content_changed})")
                        except:
                            pass
                    
                    # 🆕 策略5: 如果还是为空，查找所有可见的 tab-pane（最后fallback）
                    if not content_text or len(content_text) < 20:
                        try:
                            visible_panels = driver.find_elements(By.CSS_SELECTOR, 
                                ".tab-pane, [role='tabpanel']")
                            for panel in visible_panels:
                                try:
                                    if panel.is_displayed():
                                        panel_content_html = panel.get_attribute('innerHTML') or ""
                                        panel_content_text = panel.text.strip() if panel.text else ""
                                        if panel_content_text and len(panel_content_text) >= 20:
                                            # 🆕 检查是否与之前提取的内容不同（避免重复）
                                            is_duplicate = False
                                            for prev_result in results:
                                                if prev_result.get('content') == panel_content_text:
                                                    is_duplicate = True
                                                    break
                                            
                                            if not is_duplicate:
                                                content_html = panel_content_html
                                                content_text = panel_content_text
                                                panel_id = panel.get_attribute('id') or f"panel_visible_{i}"
                                                print(f"[9D Agent] 📄 Extracted from visible panel (id='{panel_id}')")
                                                break
                                except:
                                    continue
                        except:
                            pass
                    
                    # 如果 HTML 为空但文本不为空，从文本生成简单 HTML
                    if not content_html and content_text:
                        content_html = f"<div>{content_text}</div>"
                    
                    # Step 5: 存储结果并记录已处理的 tab 标识
                    if content_text and len(content_text.strip()) >= 20:
                        results.append({
                            'tab_name': tab_label,
                            'tab_index': i,
                            'content': content_text,
                            'html': content_html,
                            'panel_id': panel_id
                        })
                        print(f"[9D Agent] ✅ Tab '{tab_label}': {len(content_text)} chars")
                    
                    # 🆕 记录已处理的 tab 标识（用于后续排除）
                    # 使用多种方式生成唯一标识，确保能正确匹配后续查找的元素
                    tab_identifiers = []
                    try:
                        # 方法1: 使用 tab 的简短文本（只取前50字符，避免过长文本）
                        if tab_label and tab_label != f"Tab_{i+1}":
                            short_label = tab_label[:50].strip()
                            if short_label:
                                tab_identifiers.append(short_label)
                        
                        # 方法2: 使用 tab 的位置和标签（更可靠）
                        try:
                            location = tab.location
                            size = tab.size
                            tag = tab.tag_name
                            position_id = f"{tag}_{location['x']}_{location['y']}_{size['width']}_{size['height']}"
                            tab_identifiers.append(position_id)
                        except:
                            pass
                        
                        # 方法3: 使用 outerHTML 的 hash（最可靠，但需要计算）
                        try:
                            outer_html = tab.get_attribute('outerHTML') or ""
                            if outer_html:
                                import hashlib
                                html_hash = hashlib.md5(outer_html.encode()).hexdigest()[:12]
                                tab_identifiers.append(f"tab_html_{html_hash}")
                        except:
                            pass
                        
                        # 方法4: 使用索引作为 fallback
                        tab_identifiers.append(f"tab_index_{i}")
                        
                    except:
                        tab_identifiers.append(f"tab_index_{i}")
                    
                    # 将所有标识添加到集合中
                    for identifier in tab_identifiers:
                        if identifier:
                            processed_tab_identifiers.add(identifier)
                    
                    # 短暂延迟，避免点击过快
                    time.sleep(0.5)
                    
                except Exception as e:
                    continue
            
        except Exception as e:
            pass
        
        return results, processed_tab_identifiers
    
    def _find_all_interactive_elements(self, driver) -> List[Dict]:
        """
        查找所有可交互元素（全量爬取模式 - 不限维度）
        优先查找 Tab，然后是其他按钮
    
        ⚠️ 排除外部链接（这些应该在后续并行爬取中处理）
        """
        return self._find_selenium_interactive_elements(driver, missing_dims=[])
    
    def _extract_hidden_sections(
        self,
        driver,
        required_dimensions: Optional[List[str]]
    ) -> List[Dict[str, Any]]:
        """
        提取 DOM 中已存在但被隐藏的内容（display:none/tab-pane/hidden）
        适用于无需点击即可直接读取的静态隐藏节点
        """
        try:
            page_source = driver.page_source
        except Exception as e:
            print(f"[Selenium] ⚠️ Unable to access page_source for hidden sections: {e}")
            return []

        if not page_source:
            return []

        soup = BeautifulSoup(page_source, 'html.parser')
        if required_dimensions is None:
            required_dimensions = list(NINE_DIMENSIONS.keys())

        selector_list = [
            '[hidden]',
            '[style*="display:none"]',
            '[style*="display: none"]',
            '[style*="visibility:hidden"]',
            '[style*="visibility: hidden"]',
            '.tab-pane',
            '.collapse',
            '.accordion-collapse',
            '.accordion-content',
            '.accordion-body',
            '.d-none',
            '.visually-hidden',
            '.sr-only'
        ]
        candidates = []
        collected_ids = set()
        def collect(nodes):
            for node in nodes:
                if not hasattr(node, 'attrs'):
                    continue
                node_id = id(node)
                if node_id in collected_ids:
                    continue
                collected_ids.add(node_id)
                candidates.append(node)
        for css in selector_list:
            try:
                collect(soup.select(css))
            except Exception:
                continue
        sections = []
        seen_text_hashes = set()

        for node in candidates:
            try:
                classes = ' '.join(node.get('class', [])) if node.has_attr('class') else ''
                class_lower = classes.lower()
                style_lower = (node.get('style') or '').lower()
                is_tab_pane = 'tab-pane' in class_lower
                is_collapse = any(keyword in class_lower for keyword in ['collapse', 'accordion', 'drawer'])
                is_marked_hidden = node.has_attr('hidden') or 'hidden' in class_lower
                style_hidden = any(token in style_lower for token in ['display:none', 'display: none', 'visibility:hidden', 'visibility: hidden', 'opacity:0'])

                is_hidden = is_marked_hidden or style_hidden

                if is_tab_pane:
                    is_hidden = True
                if is_collapse and 'show' not in class_lower and 'open' not in class_lower:
                    is_hidden = True

                if not is_hidden:
                    continue

                lines: List[str] = []
                try:
                    self._html_to_markdown(node, lines, level=0)
                    text = '\n'.join(line for line in lines if line is not None).strip()
                except Exception:
                    text = ''

                if not text:
                    text = node.get_text(separator='\n', strip=True)

                if not text or len(text) < 30:
                    continue

                label = None
                for attr in ['aria-label', 'data-title', 'data-label', 'data-tab', 'data-section', 'title']:
                    val = node.attrs.get(attr)
                    if val and isinstance(val, str) and val.strip():
                        label = val.strip()
                        break

                aria_labelledby = node.attrs.get('aria-labelledby')
                if not label and aria_labelledby:
                    labelled = soup.find(id=aria_labelledby)
                    if labelled:
                        label_candidate = labelled.get_text(strip=True)
                        if label_candidate:
                            label = label_candidate

                pane_id = node.attrs.get('id')
                if not label and pane_id:
                    linked = (
                        soup.find(attrs={'aria-controls': pane_id}) or
                        soup.find('a', attrs={'href': f'#{pane_id}'}) or
                        soup.find('button', attrs={'data-bs-target': f'#{pane_id}'}) or
                        soup.find(attrs={'data-target': f'#{pane_id}'})
                    )
                    if linked:
                        linked_text = linked.get_text(strip=True)
                        if linked_text:
                            label = linked_text

                if not label:
                    header = node.find_previous(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
                    if header:
                        header_text = header.get_text(strip=True)
                        if header_text:
                            label = header_text

                if not label:
                    label = f"Hidden Section ({node.name})"

                text_lower = text.lower()
                label_lower = label.lower()
                matched_dimensions: List[str] = []

                for dim in required_dimensions:
                    dim_info = NINE_DIMENSIONS.get(dim, {})
                    keywords = dim_info.get('keywords', [])
                    if any(kw in text_lower for kw in keywords) or any(kw in label_lower for kw in keywords):
                        if dim not in matched_dimensions:
                            matched_dimensions.append(dim)

                if not matched_dimensions and len(text) < 50:
                    continue

                text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
                if text_hash in seen_text_hashes:
                    continue
                seen_text_hashes.add(text_hash)

                sections.append({
                    'label': label,
                    'text': text,
                    'matched_dimensions': matched_dimensions
                })
            except Exception as hidden_err:
                print(f"[Selenium] ⚠️ Hidden section parse error: {hidden_err}")
                continue

        return sections
    
    def _is_external_link(self, elem, current_url: str) -> bool:
        """
        判断是否为外部平台链接（应该被排除，不在homepage内点击）
        
        外部平台包括：
        - Google Scholar / Semantic Scholar
        - GitHub / GitLab
        - LinkedIn
        - ORCID
        - ResearchGate
        - Twitter / X
        - DBLP
        - OpenReview
        - arXiv
        - 其他学术平台
        """
        try:
            href = elem.get_attribute('href')
            if not href:
                return False
            
            href_lower = href.lower()
            
            # 外部平台域名列表（这些平台我们会专门爬取）
            external_domains = [
                'scholar.google',
                'semanticscholar.org',
                'github.com',
                'gitlab.com',
                'linkedin.com',
                'orcid.org',
                'researchgate.net',
                'twitter.com',
                'x.com',
                'dblp.org',
                'dblp.uni-trier.de',
                'openreview.net',
                'arxiv.org',
                'acm.org',
                'dl.acm.org',
                'ieee.org',
                'ieeexplore.ieee.org',
                'springer.com',
                'link.springer.com',
                'sciencedirect.com',
                'nature.com',
                'science.org'
            ]
            
            # 检查是否包含外部域名
            for domain in external_domains:
                if domain in href_lower:
                    return True
            
            # 检查是否跳转到完全不同的域名（避免点击会离开当前网站）
            from urllib.parse import urlparse
            if href_lower.startswith('http://') or href_lower.startswith('https://'):
                current_domain = (urlparse(current_url).netloc or '').lower()
                link_domain = (urlparse(href).netloc or '').lower()
                
                # 如果域名完全不同，认为是外部链接
                if link_domain and current_domain and link_domain != current_domain:
                    # 但不包括子域名的情况（如 cs.stanford.edu -> web.stanford.edu）
                    current_base = '.'.join(current_domain.split('.')[-2:]) if '.' in current_domain else current_domain
                    link_base = '.'.join(link_domain.split('.')[-2:]) if '.' in link_domain else link_domain
                    if current_base.lower() != link_base.lower():
                        return True
            else:
                # 相对链接一律认为是站内链接
                return False
            
            return False
            
        except:
            return False
    
    def _is_content_link(self, elem) -> bool:
        """
        判断是否为"内容链接"（不应该点击，但要记录URL）
        
        内容链接包括：
        - PDF文件
        - 论文详情页（arXiv、ACM、IEEE等）
        - 项目页面
        - 数据集
        - 代码仓库
        
        特征：
        1. 链接到文件：PDF, DOC等
        2. 在列表中（通常是论文列表）
        3. 有特定class（paper-link, publication-link）
        """
        try:
            href = elem.get_attribute('href')
            if not href:
                return False
            
            href_lower = href.lower()
            
            # 1️⃣ 文件类型链接（不应该点击）
            file_extensions = ['.pdf', '.doc', '.docx', '.ppt', '.pptx', '.zip', '.tar', '.gz']
            if any(href_lower.endswith(ext) for ext in file_extensions):
                return True
            
            # 2️⃣ 论文/出版物详情页
            paper_patterns = [
                'arxiv.org/abs',
                'arxiv.org/pdf',
                'doi.org',
                'dl.acm.org/doi',
                'ieeexplore.ieee.org/document',
                'openreview.net/forum',
                'openreview.net/pdf',
                'proceedings.neurips.cc',
                'aclanthology.org',
                'papers.nips.cc',
                'openaccess.thecvf.com'
            ]
            if any(pattern in href_lower for pattern in paper_patterns):
                return True
            
            # 3️⃣ 检查父元素：是否在列表中（通常是论文列表）
            try:
                parent = elem.find_element(By.XPATH, '..')
                parent_tag = parent.tag_name.lower()
                
                # 🆕 例外：如果在导航区域（nav/masthead/菜单）内的列表项，不视为内容链接
                try:
                    cur = elem
                    for _ in range(6):
                        p = cur.find_element(By.XPATH, '..')
                        p_tag = (p.tag_name or '').lower()
                        p_cls = (p.get_attribute('class') or '').lower()
                        p_id = (p.get_attribute('id') or '').lower()
                        if p_tag == 'nav' or 'masthead' in p_cls or 'greedy-nav' in p_cls or 'visible-links' in p_cls or p_id == 'site-nav':
                            return False
                        cur = p
                except:
                    pass
                
                # 如果是列表项中的链接
                if parent_tag in ['li', 'ul', 'ol']:
                    return True
                
                # 如果父元素的父元素是列表
                try:
                    grandparent = parent.find_element(By.XPATH, '..')
                    if grandparent.tag_name.lower() in ['li', 'ul', 'ol']:
                        return True
                except:
                    pass
            except:
                pass
            
            # 4️⃣ 检查 class 名称
            try:
                elem_class = elem.get_attribute('class') or ''
                content_keywords = ['paper', 'publication', 'article', 'project', 'resource', 'download']
                if any(kw in elem_class.lower() for kw in content_keywords):
                    return True
            except:
                pass
            
            return False
            
        except:
            return False
    
    def _is_navigation_element(self, elem) -> bool:
        """
        判断是否为导航元素（应该点击）
        
        导航元素特征：
        1. role="tab" / role="button"
        2. 在导航区域（nav, header, aside, sidebar）
        3. class 包含 nav, tab, menu
        4. 是 button 或在 nav 区域的 a
        """
        try:
            # 1️⃣ 检查 role
            role = elem.get_attribute('role')
            if role in ['tab', 'button', 'menuitem']:
                return True
            
            # 2️⃣ 是 button 元素
            if elem.tag_name.lower() == 'button':
                return True
            
            # 3️⃣ 检查是否在导航区域
            try:
                # 向上查找父元素
                current = elem
                for _ in range(5):  # 最多向上查找5层
                    parent = current.find_element(By.XPATH, '..')
                    parent_tag = parent.tag_name.lower()
                    parent_class = (parent.get_attribute('class') or '').lower()
                    parent_id = (parent.get_attribute('id') or '').lower()
                    
                    # 在导航区域
                    if parent_tag in ['nav', 'header', 'aside']:
                        return True
                    
                    # class/id 包含导航关键词
                    nav_keywords = ['nav', 'menu', 'sidebar', 'tab', 'header']
                    if any(kw in parent_class or kw in parent_id for kw in nav_keywords):
                        return True
                    
                    current = parent
            except:
                pass
            
            # 4️⃣ 检查自身 class
            elem_class = (elem.get_attribute('class') or '').lower()
            if any(kw in elem_class for kw in ['nav', 'tab', 'menu', 'toggle']):
                return True
            
            return False
            
        except:
            return False
    
    def _find_selenium_interactive_elements(self, driver, missing_dims: List[str] = None) -> List[Dict]:
        """
        查找 Selenium 中的可交互元素
        改进：识别符号按钮、tab、section 等结构化元素
        
        ⚠️ 排除规则：
        1. 外部平台链接（Scholar、GitHub等）→ 不点击，另外爬取
        2. 内容链接（PDF、论文详情）→ 不点击，只记录URL
        3. 只点击内部导航元素（tab、button、menu）
        
        Args:
            driver: Selenium WebDriver
            missing_dims: 缺失的维度列表。如果为空/None，则不过滤，返回所有元素
        """
        elements = []
        current_url = driver.current_url  # 获取当前URL用于判断外部链接
        from urllib.parse import urlparse, urljoin

        def _normalize_url_no_fragment_local(url: str) -> str:
            if not url:
                return ""
            try:
                parsed = urlparse(url)
                normalized = parsed._replace(fragment='').geturl()
            except Exception:
                return url.rstrip('/')
            return normalized.rstrip('/')

        def _compute_direct_navigation(href_value: str) -> Tuple[bool, str]:
            if not href_value:
                return False, ""
            href_stripped = href_value.strip()
            if (
                not href_stripped
                or href_stripped == '#'
                or href_stripped.lower().startswith('javascript:')
            ):
                return False, ""
            try:
                abs_url = urljoin(current_url, href_stripped)
            except Exception:
                abs_url = href_stripped
            try:
                parsed_abs = urlparse(abs_url)
                parsed_current = urlparse(current_url)
            except Exception:
                return False, ""
            if not parsed_abs.scheme:
                return False, ""
            # 仅处理同域链接
            if parsed_abs.netloc and parsed_current.netloc:
                if parsed_abs.netloc.lower() != parsed_current.netloc.lower():
                    return False, ""
            target_base = _normalize_url_no_fragment_local(abs_url)
            current_base = _normalize_url_no_fragment_local(current_url)
            if target_base == current_base:
                return False, ""
            return True, abs_url

        filter_stats = {
            'external_links': 0,
            'content_links': 0,
            'non_navigation_links': 0,
            'no_text': 0,
            'not_visible': 0,
            'irrelevant_nav': 0
        }

        def _safe_href(el) -> str:
            try:
                return el.get_attribute('href') or ''
            except:
                return ''

        def _safe_tag(el) -> str:
            try:
                return (el.tag_name or '').lower()
            except:
                return ''
        
        # 如果 missing_dims 为空，则不过滤（全量爬取模式）
        keywords = []
        
        if missing_dims:
            for dim_name in missing_dims:
                keywords.extend(NINE_DIMENSIONS[dim_name]['keywords'])        
        # ✅ 改进1：优先查找 Tab 元素（结构化导航）
        try:
            tab_selectors = [
                '[role="tab"]',
                '[role="tablist"] a',
                '[role="tablist"] button',
                '.nav-tabs a',
                '.nav-tabs button',
                'ul.nav a',
                'ul.nav button'
            ]
            for selector in tab_selectors:
                try:
                    tab_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    for elem in tab_elements:
                        locator = (By.CSS_SELECTOR, selector)
                        if not elem.is_displayed():
                            filter_stats['not_visible'] += 1
                            continue
                        
                        # 🆕 排除外部链接（这些应该在后续并行爬取）
                        if elem.tag_name == 'a' and self._is_external_link(elem, current_url):
                            filter_stats['external_links'] += 1
                            continue
                        
                        # 🆕 排除内容链接（PDF、论文链接等，不应该点击）
                        if elem.tag_name == 'a' and self._is_content_link(elem):
                            filter_stats['content_links'] += 1
                            continue
                        
                        # ✅ 获取多种文本来源
                        text = self._extract_element_text(elem)
                        href_attr = _safe_href(elem)
                        direct_nav, direct_href = _compute_direct_navigation(href_attr)
                        tag_name_attr = _safe_tag(elem)
                        panel_lookup = self._extract_panel_lookup(elem)
                        
                        # 先爬后判策略：只要包含SVG/图标，就无条件保留
                        is_icon_only_tab = False
                        has_svg = False
                        has_fa_icon = False
                        
                        try:
                            # 检查是否有图标标记（SVG、FontAwesome等）
                            has_svg = len(elem.find_elements('css selector', 'svg')) > 0
                            class_attr = elem.get_attribute('class') or ''
                            has_fa_icon = 'fa-' in class_attr or 'icon' in class_attr.lower()
                            
                            if has_svg or has_fa_icon:
                                is_icon_only_tab = True
                                try:
                                    svg_elem = elem.find_element('css selector', 'svg[data-icon]')
                                    icon_name = svg_elem.get_attribute('data-icon')
                                    if icon_name:
                                        # 如果text为空或很短，使用图标名称
                                        if not text or len(text) < 2:
                                            text = f"icon-{icon_name}"  # 使用图标名称作为标识
                                except:
                                    pass
                                
                                # 如果text仍然为空，使用占位符
                                if not text or len(text) < 2:
                                    text = "icon-tab"  # 占位符
                        except:
                            pass
                        
                        if is_icon_only_tab:
                            pass  # 不执行导航检测，直接保留
                        else:
                            # 📝 文字元素：需要导航检测
                            if elem.tag_name == 'a' and not self._is_navigation_element(elem):
                                filter_stats['non_navigation_links'] += 1
                                continue
                        if not is_icon_only_tab and (not text or len(text) < 2):
                            filter_stats['no_text'] += 1
                            continue
                        if text and len(text) >= 1:  # 降低要求，允许icon-tab
                            has_emoji = self._contains_emoji(text)
                            is_visual_only = has_emoji or is_icon_only_tab
                            href_attr = ''
                            try:
                                href_attr = elem.get_attribute('href') or ''
                            except:
                                href_attr = ''
                            # 策略分支：
                            # 1. 视觉元素（emoji/icon）：无条件保留，先爬后判
                            # 2. 文字元素：先判断相关性，再决定是否爬取
                            if is_visual_only:
                                # 优先级：emoji=8, icon-only=7
                                priority = 8 if has_emoji else 7
                                elements.append({
                                    'element': elem,
                                    'text': text,
                                    'tag': elem.tag_name,
                                    'type': 'tab',
                                    'matched': False,  # 无法预判
                                    'matched_dimensions': [],
                                    'has_emoji': has_emoji,
                                    'is_icon_only': is_icon_only_tab,
                                    'is_visual_only': True,
                                    'priority': priority,
                                    'href': href_attr,
                                    'tag_name': tag_name_attr,
                                    'locator': locator,
                                    'direct_navigation': direct_nav,
                                    'direct_href': direct_href,
                                    'panel_lookup': panel_lookup
                                })
                            else:
                                # 📝 文字元素：智能过滤（先判后爬）
                                text_lower = text.lower()
                                
                                # 🚫 排除明确无关的导航
                                irrelevant_keywords = ['home', 'news', 'blog', 'media', 'press', 'events', 
                                                       'gallery', 'photos', 'videos', 'donate', 'join']
                                is_irrelevant = any(kw == text_lower or kw in text_lower.split() for kw in irrelevant_keywords)
                                if is_irrelevant:
                                    filter_stats['irrelevant_nav'] += 1
                                    continue  # 跳过无关导航
                                # ✅ 检查是否匹配九维度关键词
                                matched_dimensions = []
                                for dim_name, dim_info in NINE_DIMENSIONS.items():
                                    dim_keywords = dim_info['keywords']
                                    if any(kw in text_lower for kw in dim_keywords):
                                        matched_dimensions.append(dim_name)
                                
                                has_match = len(matched_dimensions) > 0
                                
                                # 检查是否是展开按钮（这些通常有用，不需要 LLM 判断）
                                expand_keywords = ['show', 'more', 'expand', 'see all', 'view all', 'load more']
                                is_expand = any(kw in text_lower for kw in expand_keywords)
                                
                                # 🆕 如果关键词匹配失败且不是展开按钮，使用 LLM 判断是否与九维度相关
                                should_keep = False
                                priority = 0
                                
                                if has_match:
                                    # 关键词匹配，直接保留
                                    should_keep = True
                                    priority = 10
                                elif is_expand:
                                    # 展开按钮，直接保留
                                    should_keep = True
                                    priority = 6
                                
                                # 保留元素
                                if should_keep:
                                    elements.append({
                                        'element': elem,
                                        'text': text,
                                        'tag': elem.tag_name,
                                        'type': 'tab',
                                        'matched': has_match,  # 只有关键词匹配才算 matched
                                        'matched_dimensions': matched_dimensions,
                                        'has_emoji': False,
                                        'is_icon_only': False,
                                        'is_visual_only': False,
                                        'priority': priority,
                                        'href': href_attr,
                                        'tag_name': tag_name_attr,
                                        'locator': locator,
                                        'direct_navigation': direct_nav,
                                        'direct_href': direct_href,
                                        'panel_lookup': panel_lookup
                                    })
                except:
                    continue
        except:
            pass
        
        # ✅ 改进2：查找其他可点击元素（包括侧边栏、下拉菜单等）
        try:
            clickable_selectors = [
                'button',
                'a',
                '[role="button"]',
                '[class*="expand"]',
                '[class*="show"]',
                '[class*="toggle"]',
                # 🆕 侧边栏常见选择器
                '.sidebar button',
                '.sidebar a',
                'aside button',
                'aside a',
                # 🆕 下拉菜单/手风琴
                '[class*="accordion"] button',
                '[class*="dropdown"] button',
                '[class*="menu"] button',
                '[class*="menu"] a',
                # 🆕 Section headers（可能可以展开）
                '[class*="section"] button',
                '[class*="section"] a'
            ]
            
            for selector in clickable_selectors:
                try:
                    clickable_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for elem in clickable_elements:
                        try:
                            locator = (By.CSS_SELECTOR, selector)
                            if not elem.is_displayed():
                                filter_stats['not_visible'] += 1
                                continue
                            
                            # ✅ 提取元素文本（包括隐藏的 aria-label 等）
                            text = self._extract_element_text(elem)
                            
                            if not text or len(text) < 2:
                                filter_stats['no_text'] += 1
                                continue
                            
                            # 检查是否包含视觉元素（emoji/图标）并在导航检测前计算标志
                            has_emoji = self._contains_emoji(text)
                            has_svg = len(elem.find_elements('css selector', 'svg')) > 0
                            class_attr = elem.get_attribute('class') or ''
                            has_fa_icon = 'fa-' in class_attr or 'icon' in class_attr.lower()
                            is_visual_only = has_emoji or has_svg or has_fa_icon
                            href_attr = _safe_href(elem)
                            direct_nav, direct_href = _compute_direct_navigation(href_attr)
                            tag_name_attr = _safe_tag(elem)
                            panel_lookup = self._extract_panel_lookup(elem)
                            
                            # 放宽：检测是否在导航区域（masthead/greedy-nav/site-nav/visible-links）
                            in_nav_area = False
                            try:
                                current = elem
                                for _ in range(6):
                                    parent = current.find_element(By.XPATH, '..')
                                    parent_class = (parent.get_attribute('class') or '').lower()
                                    parent_id = (parent.get_attribute('id') or '').lower()
                                    parent_tag = parent.tag_name.lower()
                                    if parent_tag == 'nav' or \
                                       'masthead' in parent_class or \
                                       'greedy-nav' in parent_class or \
                                       'visible-links' in parent_class or \
                                       parent_id == 'site-nav':
                                        in_nav_area = True
                                        break
                                    current = parent
                            except:
                                pass
                            
                            # 排除外部链接（这些应该在后续并行爬取）
                            if elem.tag_name == 'a' and self._is_external_link(elem, current_url):
                                filter_stats['external_links'] += 1
                                continue
                            
                            # 🆕 排除内容链接（PDF、论文链接等，不应该点击）
                            if elem.tag_name == 'a' and self._is_content_link(elem):
                                # 放宽：若在导航区域，允许通过，不按内容链接处理
                                if elem.tag_name == 'a' and self._is_content_link(elem) and (not in_nav_area):
                                    filter_stats['content_links'] += 1
                                    continue
                            
                            # 🆕 视觉元素（图标/emoji）跳过导航检测；
                            #    非视觉元素需要通过导航检测；但如果明确在导航区域则放宽保留
                            if (not is_visual_only) and elem.tag_name == 'a' and (not self._is_navigation_element(elem)) and (not in_nav_area):
                                filter_stats['non_navigation_links'] += 1
                                continue
                            
                            # 🆕 策略分支：
                            # 1. 视觉元素（emoji/图标）：无条件保留，先爬后判
                            # 2. 文字元素：先判断相关性，再决定是否爬取
                            
                            if is_visual_only:
                                # 🎨 视觉元素：保守爬取策略（先爬后判）
                                # 图标元素跳过导航检测
                                if elem.tag_name == 'a' and (has_svg or has_fa_icon) and self._is_navigation_element(elem):
                                    # 是导航图标，优先级更高
                                    priority = 8
                                else:
                                    priority = 7  # 普通emoji按钮
                                elements.append({
                                    'element': elem,
                                    'text': text,
                                    'tag': elem.tag_name,
                                    'type': 'button' if elem.tag_name == 'button' else 'link',
                                    'matched': False,  # 无法预判
                                    'matched_dimensions': [],
                                    'has_emoji': has_emoji,
                                    'is_visual_only': True,
                                    'priority': priority,
                                    'href': href_attr,
                                    'tag_name': tag_name_attr,
                                    'locator': locator,
                                    'direct_navigation': direct_nav,
                                    'direct_href': direct_href,
                                    'panel_lookup': panel_lookup
                                })
                            else:
                                # 📝 文字元素：智能过滤（先判后爬）
                                text_lower = text.lower()
                                
                                # 🆕 若处在导航区域，直接保留为可点击导航链接（提高优先级）
                                if in_nav_area:
                                    elements.append({
                                        'element': elem,
                                        'text': text,
                                        'tag': elem.tag_name,
                                        'type': 'link',
                                        'matched': False,
                                        'matched_dimensions': [],
                                        'has_emoji': False,
                                        'is_visual_only': False,
                                        'priority': 8,
                                        'href': href_attr,
                                        'tag_name': tag_name_attr,
                                        'locator': locator,
                                        'direct_navigation': direct_nav,
                                        'direct_href': direct_href,
                                        'panel_lookup': panel_lookup
                                    })
                                    continue
                                
                                irrelevant_keywords = ['home', 'news', 'blog', 'media', 'press', 'events', 
                                                       'gallery', 'photos', 'videos', 'donate', 'join']
                                is_irrelevant = any(kw == text_lower or kw in text_lower.split() for kw in irrelevant_keywords)
                                
                                if is_irrelevant:
                                    filter_stats['irrelevant_nav'] += 1
                                    continue  # 跳过无关导航
                                matched_dimensions = []
                                for dim_name, dim_info in NINE_DIMENSIONS.items():
                                    dim_keywords = dim_info['keywords']
                                    if any(kw in text_lower for kw in dim_keywords):
                                        matched_dimensions.append(dim_name)
                                
                                # 检查"展开"类按钮（这些通常有用）
                                expand_keywords = ['show', 'more', 'expand', 'see all', 'view all', 'load more']
                                is_expand = any(kw in text_lower for kw in expand_keywords)
                                
                                has_match = len(matched_dimensions) > 0
                                
                                should_keep = False
                                priority = 0
                                
                                if has_match:
                                    # 关键词匹配，直接保留
                                    should_keep = True
                                    priority = 9
                                elif is_expand:
                                    # 展开按钮，直接保留
                                    should_keep = True
                                    priority = 6
                                
                                # 保留元素
                                if should_keep:
                                    elements.append({
                                        'element': elem,
                                        'text': text,
                                        'tag': elem.tag_name,
                                        'type': 'button' if elem.tag_name == 'button' else 'link',
                                        'matched': has_match,  # 只有关键词匹配才算 matched
                                        'matched_dimensions': matched_dimensions,
                                        'has_emoji': False,
                                        'is_visual_only': False,
                                        'priority': priority,
                                        'href': href_attr,
                                        'tag_name': tag_name_attr,
                                        'locator': locator,
                                        'direct_navigation': direct_nav,
                                        'direct_href': direct_href,
                                        'panel_lookup': panel_lookup
                                    })
                        except:
                            continue
                except:
                    continue
        except:
            pass
        
        seen_elements = set()  # 使用 (text, location) 作为唯一标识
        unique_elements = []
        for elem_data in elements:
            text = elem_data['text']
            elem = elem_data['element']
            
            try:
                location = elem.location
                size = elem.size
                elem_id = f"{text}|{location['x']},{location['y']}|{size['width']},{size['height']}"
            except:
                # 如果获取位置失败，使用文本+tag作为标识
                elem_id = f"{text}|{elem.tag_name}"
            
            if elem_id not in seen_elements:
                seen_elements.add(elem_id)
                unique_elements.append(elem_data)
        
        # ✅ 按优先级排序
        unique_elements.sort(key=lambda x: x.get('priority', 0), reverse=True)
        
        return unique_elements
    
    def _extract_structured_text(self, elem, driver) -> str:
        """
        提取元素文本，保留层次结构
        将 HTML 结构转换为 Markdown 格式，保留标题、列表、段落
        """
        try:
            # 获取元素的 HTML
            html_content = elem.get_attribute('innerHTML')
            
            if not html_content:
                return elem.text
            
            # 使用 BeautifulSoup 解析
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # 转换为带格式的文本
            lines = []
            self._html_to_markdown(soup, lines, level=0)
            
            result = '\n'.join(lines)
            return result if result else elem.text
        
        except Exception as e:
            # 回退到简单文本提取
            return elem.text
    
    def _html_to_markdown(self, element, lines: list, level: int = 0):
        """
        递归将 HTML 元素转换为 Markdown 格式
        保留标题、列表、段落等结构
        """
        from bs4 import NavigableString, Tag
        
        # 跳过 script、style 等标签
        if isinstance(element, Tag) and element.name in ['script', 'style', 'noscript']:
            return
        
        # 处理文本节点
        if isinstance(element, NavigableString):
            text = str(element).strip()
            if text:
                indent = '  ' * level
                lines.append(f"{indent}{text}")
            return
        
        # 处理标签元素
        if isinstance(element, Tag):
            tag_name = element.name
            
            # 标题标签
            if tag_name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                text = element.get_text(strip=True)
                if text:
                    # Markdown 标题
                    level_mark = '#' * int(tag_name[1])
                    lines.append(f"\n{level_mark} {text}\n")
                return
            
            # 列表项
            elif tag_name == 'li':
                text = element.get_text(strip=True)
                if text:
                    indent = '  ' * level
                    lines.append(f"{indent}• {text}")
                return
            
            # 段落
            elif tag_name == 'p':
                text = element.get_text(strip=True)
                if text:
                    indent = '  ' * level
                    lines.append(f"{indent}{text}\n")
                return
            
            # 换行
            elif tag_name == 'br':
                lines.append('')
                return
            
            # 列表容器（递归处理子元素）
            elif tag_name in ['ul', 'ol']:
                for child in element.children:
                    self._html_to_markdown(child, lines, level)
                lines.append('')  # 列表后加空行
                return
            
            # 分区容器（递归处理）
            elif tag_name in ['div', 'section', 'article', 'main']:
                # 检查是否有特殊类名（如 section 标题）
                class_names = element.get('class', [])
                
                # 先处理直接文本节点
                for child in element.children:
                    self._html_to_markdown(child, lines, level)
                return
            
            # 其他内联元素
            else:
                # 递归处理子元素
                for child in element.children:
                    self._html_to_markdown(child, lines, level)
    
    def _extract_element_text(self, elem) -> str:
        """
        从元素中提取文本（包括符号按钮的隐藏文本）
        检查多个属性来获取元素的"意义"
        
        🆕 增强：识别emoji/符号对应的九维度
        """
        text_sources = []
        
        try:
            # 1. 显示文本
            visible_text = elem.text.strip()
            if visible_text:
                text_sources.append(visible_text)
            
            # 2. aria-label（无障碍标签，常用于符号按钮）
            aria_label = elem.get_attribute('aria-label')
            if aria_label and aria_label.strip():
                text_sources.append(aria_label.strip())
            
            # 3. title 属性
            title = elem.get_attribute('title')
            if title and title.strip():
                text_sources.append(title.strip())
            
            # 4. data-* 属性（可能包含描述）
            for attr in ['data-title', 'data-label', 'data-text', 'data-tab', 'data-tooltip', 'data-icon']:
                value = elem.get_attribute(attr)
                if value and value.strip():
                    text_sources.append(value.strip())
            
            # 5. placeholder（输入框）
            placeholder = elem.get_attribute('placeholder')
            if placeholder and placeholder.strip():
                text_sources.append(placeholder.strip())
            
            # 6. FontAwesome / Icon 类名（如果是纯图标元素）
            if not visible_text:
                try:
                    class_name = elem.get_attribute('class')
                    if class_name:
                        # 提取 FontAwesome 图标名称: fa-trophy, fa-user-friends, fa-leaf, etc.
                        import re
                        fa_matches = re.findall(r'fa-(\w+)', class_name)
                        if fa_matches:
                            # 使用最完整的图标名称（通常第一个匹配是最具体的）
                            icon_name = fa_matches[0]
                            # 将连字符转换为空格，使名称更易读
                            readable_name = icon_name.replace('-', ' ')
                            text_sources.append(f"icon {readable_name}")
                        
                        # 检查SVG子元素（包括data-icon属性）
                        try:
                            svg_elem = elem.find_element('css selector', 'svg[data-icon]')
                            icon_name = svg_elem.get_attribute('data-icon')
                            if icon_name:
                                readable_name = icon_name.replace('-', ' ')
                                text_sources.append(f"icon {readable_name}")
                        except:
                            pass
                        
                        # 检查SVG的class属性（FontAwesome SVG通常有类似svg-inline--fa fa-trophy的类）
                        try:
                            svg_elem = elem.find_element('css selector', 'svg')
                            svg_class = svg_elem.get_attribute('class') or ''
                            fa_matches = re.findall(r'fa-(\w+)', svg_class)
                            if fa_matches:
                                readable_name = fa_matches[0].replace('-', ' ')
                                text_sources.append(f"icon {readable_name}")
                        except:
                            pass
                except:
                    pass
            
            # 7. 子元素的文本（如果本身无文本）
            if not visible_text:
                try:
                    inner_html = elem.get_attribute('innerHTML')
                    if inner_html:
                        # 简单提取，避免太多噪音
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(inner_html, 'html.parser')
                        inner_text = soup.get_text(separator=' ', strip=True)
                        if inner_text and len(inner_text) < 100:  # 避免过长
                            text_sources.append(inner_text)
                except:
                    pass
        
        except:
            pass
        
        # 合并所有文本源（去重）
        if text_sources:
            # 选择最长且最有意义的文本
            return max(text_sources, key=len)
        
        return ""
    
    def _contains_emoji(self, text: str) -> bool:
        """
        🆕 检测文本中是否包含emoji符号
        
        使用Unicode范围判断是否为emoji：
        - Emoticons: 0x1F600 - 0x1F64F
        - Symbols & Pictographs: 0x1F300 - 0x1F5FF
        - Transport & Map: 0x1F680 - 0x1F6FF
        - Flags: 0x1F1E0 - 0x1F1FF
        - Miscellaneous Symbols: 0x2600 - 0x26FF
        - Dingbats: 0x2700 - 0x27BF
        """
        import re
        # Unicode emoji ranges
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002600-\U000026FF"  # misc symbols
            "\U00002700-\U000027BF"  # dingbats
            "\U0001F900-\U0001F9FF"  # supplemental symbols
            "\U0001FA00-\U0001FA6F"  # chess symbols
            "\U0001FA70-\U0001FAFF"  # symbols and pictographs extended-a
            "\U00002300-\U000023FF"  # misc technical
            "]+",
            flags=re.UNICODE
        )
        return bool(emoji_pattern.search(text))
    
    def _llm_decide_for_dimensions_selenium(
        self,
        current_text: str,
        missing_dimensions: List[str],
        interactive_elements: List[Dict],
        history: List[Dict]
    ) -> Dict[str, Any]:
        """LLM 决策（Selenium 版本）"""
        if not self.llm_available or not interactive_elements:
            return self._heuristic_decide_selenium(missing_dimensions, interactive_elements)
        
        # 使用简化的启发式（节省 LLM 调用）
        return self._heuristic_decide_selenium(missing_dimensions, interactive_elements)
    
    def _heuristic_decide_selenium(
        self, 
        missing_dimensions: List[str], 
        interactive_elements: List[Dict]
    ) -> Dict[str, Any]:
        """Selenium 启发式决策"""
        if not interactive_elements:
            return {'action': 'stop', 'reason': 'No elements'}
        
        # 优先选择匹配关键词的元素
        matched_elements = [e for e in interactive_elements if e.get('matched')]
        
        if matched_elements:
            return {
                'action': 'click',
                'target_element': matched_elements[0],
                'reason': 'Matched dimension keywords'
            }
        
        # 否则选择第一个"展开"类型的元素
        return {
            'action': 'click',
            'target_element': interactive_elements[0],
            'reason': 'Expand element'
        }
    
    def _group_sections_by_relevance(self, structured_content: Dict) -> Dict[str, list]:
        """
        将 sections 按主题分组，便于分段提取
        
        Returns:
            {
                'publications': [section_keys],
                'awards': [section_keys],
                'experience': [section_keys],
                ...
            }
        """
        groups = {
            'basic_info': [],      # Home, About, Bio
            'publications': [],    # Publications, Papers, Research
            'awards': [],          # Awards, Honors
            'experience': [],      # Experience, Education, CV
            'teaching': [],        # Teaching, Courses
            'service': [],         # Service, Activities
            'contact': [],         # Contact, Social
            'other': []            # 其他
        }
        
        # 关键词映射
        keywords_map = {
            'basic_info': ['home', 'about', 'bio', 'introduction', 'overview'],
            'publications': ['publication', 'paper', 'research', 'selected', 'preprint', 'conference', 'workshop', 'journal'],
            'awards': ['award', 'honor', 'prize', 'grant', 'fellowship'],
            'experience': ['experience', 'education', 'cv', 'resume', 'background', 'career'],
            'teaching': ['teach', 'course', 'lecture', 'class', 'academic position'],
            'service': ['service', 'activity', 'committee', 'review', 'chair'],
            'contact': ['contact', 'email', 'social', 'profile', 'link']
        }
        
        for key, content_obj in structured_content.items():
            if key == 'initial':
                groups['basic_info'].append(key)
                continue
            
            label = content_obj.get('label', '').lower()
            matched = False
            
            # 根据 label 匹配分组
            for group_name, keywords in keywords_map.items():
                if any(kw in label for kw in keywords):
                    groups[group_name].append(key)
                    matched = True
                    break
            
            if not matched:
                groups['other'].append(key)
        
        # 移除空组
        groups = {k: v for k, v in groups.items() if v}
        
        return groups
    
    def _create_extraction_text(self, structured_content: Dict, section_groups: Dict) -> str:
        """
        创建用于九维度提取的文本（智能摘要，避免过长）
        
        策略：
        1. 小段落：完整保留（< 2000字符）
        2. 长段落：取前1500 + 后500字符
        3. 超长段落（publications）：只保留标题和摘要信息
        """
        MAX_SECTION_CHARS = 2000
        MAX_LONG_SECTION_CHARS = 2000  # 前1500 + 后500
        
        parts = []
        
        for group_name, section_keys in section_groups.items():
            if not section_keys:
                continue
            parts.append(f"\n{'='*80}")
            parts.append(f"📂 {group_name.upper().replace('_', ' ')}")
            parts.append(f"{'='*80}\n")
            
            for section_key in section_keys:
                content_obj = structured_content.get(section_key, {})
                label = content_obj.get('label', 'Unknown')
                text = content_obj.get('text', '')
                
                if not text:
                    continue
                
                # 根据长度决定如何处理
                if len(text) <= MAX_SECTION_CHARS:
                    # 短内容：完整保留
                    parts.append(f"\n## {label}")
                    parts.append(text)
                else:
                    # 长内容：智能截取
                    if 'publication' in label.lower() or 'paper' in label.lower():
                        # Publications: 只保留前2000字符（通常包含最近的论文）
                        parts.append(f"\n## {label} (摘要)")
                        parts.append(text[:2000])
                        parts.append(f"\n... [剩余 {len(text) - 2000} 字符未显示，可从 section_key: {section_key} 获取]")
                    else:
                        # 其他长内容：首尾各取一部分
                        parts.append(f"\n## {label}")
                        parts.append(text[:1500])
                        parts.append(f"\n\n... [中间 {len(text) - 2000} 字符省略] ...\n\n")
                        parts.append(text[-500:])
                
                parts.append("")  # 空行分隔
        
        result = '\n'.join(parts)
        print(f"[9D Agent/Selenium] 📝 Extraction text: {len(result):,} chars (from {len(self._merge_structured_content(structured_content)):,} total)")
        return result
    
    def _merge_structured_content(self, structured_content: Dict) -> str:
        """合并所有结构化内容用于分析"""
        text_parts = []
        for key, content_obj in structured_content.items():
            text_parts.append(content_obj.get('text', ''))
        return '\n\n'.join(text_parts)
    
    def _merge_dimension_data(self, existing_data: Dict, dim_name: str, new_data: Dict) -> Dict:
        """
        合并维度数据（补充模式 + 智能去重）
        
        1. 列表型字段（publications, awards等）：追加去重（基于 title/name）
        2. 字符串字段（text, description等）：智能合并（保留更完整的内容）
        3. 避免重复内容生成
        
        对于列表型字段（publications, awards等），将新数据追加到已有数据
        对于字符串字段，如果新内容更完整，则更新（而不是简单的先到先得）
        """
        if not existing_data or not new_data:
            return existing_data or new_data
        
        merged = existing_data.copy()
        
        # 列表型字段：追加去重
        list_fields = ['publications', 'awards', 'education', 'experiences', 'teaching', 'interests', 'services', 'invited_talks']
        
        for field in list_fields:
            if field in merged and field in new_data:
                # 获取现有数据
                existing_items = merged[field]
                new_items = new_data[field]
                
                if isinstance(existing_items, list) and isinstance(new_items, list):
                    def _make_key(field_name: str, item: Dict[str, Any]) -> str:
                        """生成用于去重的 key"""
                        if not isinstance(item, dict):
                            return ''
                        if field_name == 'services':
                            role = item.get('role', '') or ''
                            venue = item.get('venue', '') or ''
                            year = item.get('year', '') or ''
                            # 🆕 如果 role 和 venue 都为空，使用 description 作为 fallback
                            if not role and not venue:
                                description = item.get('description', '') or ''
                                if description:
                                    return f"desc|{description.strip().lower()[:50]}"
                                return ''
                            # 正常情况：使用 role|venue|year 作为 key
                            return f"{role.strip().lower()}|{venue.strip().lower()}|{str(year).strip().lower()}"
                        elif field_name == 'invited_talks':
                            title = (item.get('title') or '').strip().lower()
                            # 🆕 如果 title 为空，使用 venue 作为 fallback
                            if not title:
                                venue = (item.get('venue') or '').strip().lower()
                                if venue:
                                    return f"venue|{venue}"
                            return title
                        else:
                            # 其他字段使用 title/name/course_name
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
                        print(f"[9D Agent]   ⚠️ Skipped {skipped_count} {field} items (missing required fields for deduplication)")
                    if added_count > 0:
                        print(f"[9D Agent]   ✅ Merged {added_count} new items into {dim_name}.{field} (deduplicated)")
            elif field in new_data and field not in merged:
                # 如果新数据有该字段但旧数据没有，直接添加
                merged[field] = new_data[field]
        
        string_fields = ['text', 'description', 'bio', 'overview', 'summary']
        for key, new_value in new_data.items():
            if key not in list_fields:
                if key in merged:
                    existing_value = merged[key]
                    # 如果新内容更长且更完整，则更新（避免保留不完整的内容）
                    if isinstance(new_value, str) and isinstance(existing_value, str):
                        if len(new_value.strip()) > len(existing_value.strip()) * 1.2:  # 新内容明显更长（20%以上）
                            merged[key] = new_value
                        # 否则保持原有内容（避免频繁替换）
                    elif new_value and not existing_value:
                        # 如果原有为空，使用新值
                        merged[key] = new_value
                else:
                    # 如果字段不存在，直接添加
                    merged[key] = new_value
        
        return merged
    
    def _format_structured_content_for_extraction(self, structured_content: Dict) -> str:
        """
        格式化结构化内容用于九维度提取
        保持 tab/section 的层次结构
        """
        formatted_parts = []
        
        for key, content_obj in structured_content.items():
            label = content_obj.get('label', 'Unknown Section')
            content_type = content_obj.get('type', 'content')
            text = content_obj.get('text', '')
            
            # 添加结构化标记
            formatted_parts.append("=" * 80)
            formatted_parts.append(f"SECTION: {label} ({content_type})")
            formatted_parts.append("=" * 80)
            formatted_parts.append(text)
            formatted_parts.append("")  # 空行分隔
        
        return '\n'.join(formatted_parts)
    
    def _find_content_panel(self, driver, elem, element_dict: Optional[Dict[str, Any]] = None, debug_prefix: str = ""):
        """
        简化的 panel 定位：通过 tab 文本找到对应的 panel id
        1. 提取 tab 的文本（如 "Publications"）
        2. 将文本转换为小写作为 panel id（如 "publications"）
        3. 通过 id 查找 panel 元素
        返回: panel 元素对象
        """
        from selenium.webdriver.common.by import By
        from selenium.common.exceptions import NoSuchElementException
        
        # 1. 提取 tab 的文本
        tab_text = ""
        try:
            # 方法1: 直接获取文本
            tab_text = elem.text.strip() if elem.text else ""
            
            # 方法2: 如果文本为空，尝试从 innerText/textContent 获取
            if not tab_text:
                tab_text = (elem.get_attribute('innerText') or elem.get_attribute('textContent') or '').strip()
            
            # 方法3: 如果还是为空，尝试从 aria-label 获取
            if not tab_text:
                tab_text = (elem.get_attribute('aria-label') or '').strip()
            
            # 方法4: 如果还是为空，尝试从 SVG 图标获取
            if not tab_text:
                try:
                    svg_info = driver.execute_script("""
                        var el = arguments[0];
                        var svg = el.querySelector('svg');
                        if (svg) {
                            var dataIcon = svg.getAttribute('data-icon') || '';
                            var ariaLabel = svg.getAttribute('aria-label') || '';
                            var title = svg.getAttribute('title') || '';
                            return dataIcon || ariaLabel || title || '';
                        }
                        return '';
                    """, elem)
                    if svg_info:
                        tab_text = svg_info.strip()
                except:
                    pass
            
        except Exception as e:
            pass
        
        # 2. 将文本转换为 panel id（小写，去除空格和特殊字符）
        if tab_text:
            # 转换为小写，去除前后空格
            panel_id = tab_text.lower().strip()
            # 去除特殊字符，只保留字母、数字、连字符和下划线
            import re
            panel_id = re.sub(r'[^a-z0-9_-]', '', panel_id)
            
            
            # 3. 通过 id 查找 panel
            try:
                panel = driver.find_element(By.ID, panel_id)
                return panel
            except NoSuchElementException:
                pass
            except Exception:
                pass
        
        # 4. Fallback: 尝试使用 aria-controls
        try:
            aria_controls = elem.get_attribute('aria-controls')
            if aria_controls:
                panel = driver.find_element(By.ID, aria_controls)
                return panel
        except Exception:
            pass
        
        # 5. Fallback: 尝试使用 data-bs-target 或 data-target
        for attr in ('data-bs-target', 'data-target', 'href'):
            try:
                val = elem.get_attribute(attr)
                if val and val.startswith('#'):
                    panel_id = val.lstrip('#').split()[0]
                    if panel_id:
                        panel = driver.find_element(By.ID, panel_id)
                        return panel
            except Exception:
                continue
        
        # 6. 最终 fallback: 返回 None
        return None

    def _classify_tab_behavior(self, elem, current_url: str) -> str:
        """
        粗略判断 tab 行为类型：
        - static: 前端切换，内容已在 DOM 中
        - ajax: 点击后通过 JS 异步加载
        - navigation: 实际是跳转链接
        """
        try:
            href_raw = (elem.get_attribute('href') or '').strip()
        except Exception:
            href_raw = ''
        data_remote_attrs = [
            'data-fetch', 'data-remote', 'data-url', 'data-load', 'data-source',
            'data-api', 'data-request', 'data-endpoint'
        ]
        ajax_indicators = any(
            (elem.get_attribute(attr) or '').strip() for attr in data_remote_attrs
        )
        try:
            data_action = (elem.get_attribute('data-action') or '').lower()
        except Exception:
            data_action = ''
        if data_action in ['load', 'fetch', 'ajax', 'remote']:
            ajax_indicators = True
        try:
            data_toggle = (elem.get_attribute('data-toggle') or '').lower()
        except Exception:
            data_toggle = ''
        try:
            data_bs_toggle = (elem.get_attribute('data-bs-toggle') or '').lower()
        except Exception:
            data_bs_toggle = ''
        toggle_value = data_toggle or data_bs_toggle
        if toggle_value in ['tab', 'pill', 'collapse']:
            return 'static'
        if href_raw:
            if href_raw.startswith('#'):
                return 'static'
            from urllib.parse import urljoin, urlparse
            try:
                current_parsed = urlparse(current_url or '')
            except Exception:
                current_parsed = None
            try:
                abs_href = urljoin(current_url or '', href_raw)
                href_parsed = urlparse(abs_href)
            except Exception:
                abs_href = href_raw
                href_parsed = None
            fragment_only = href_parsed and href_parsed.fragment and (
                (href_parsed.path or '') == (current_parsed.path if current_parsed else '')
            )
            if fragment_only:
                return 'static'
            if href_parsed and current_parsed:
                if href_parsed.netloc and href_parsed.netloc.lower() != (current_parsed.netloc or '').lower():
                    return 'navigation'
                href_path = (href_parsed.path or '').strip('/')
                current_path = (current_parsed.path or '').strip('/')
                if href_path and href_path != current_path:
                    return 'navigation'
            if ajax_indicators:
                return 'ajax'
            return 'static'
        return 'ajax' if ajax_indicators else 'static'
    
    def _build_tab_selectors(self, element_dict: Optional[Dict[str, Any]] = None) -> List[str]:
        selectors: List[str] = []
        if element_dict:
            lookup = element_dict.get('panel_lookup') or {}
            for candidate in lookup.get('candidates', []):
                method = candidate.get('method')
                value = (candidate.get('value') or '').strip()
                if not value:
                    continue
                if 'fallback:' in value.lower() or not value.startswith(('#', '.', '[', '*')):
                    continue
                if method == 'css':
                    selectors.append(value)
                elif method == 'id':
                    selectors.append(f"#{value.lstrip('#')}")
            resolved = lookup.get('resolved') or {}
            resolved_value = (resolved.get('value') or '').strip()
            resolved_method = (resolved.get('method') or '').strip()
            if resolved_value:
                if 'fallback:' not in resolved_value.lower() and (
                    resolved_value.startswith(('#', '.', '[', '*')) or resolved_method == 'id'
                ):
                    if resolved_method == 'css':
                        selectors.append(resolved_value)
                    elif resolved_method == 'id':
                        selectors.append(f"#{resolved_value.lstrip('#')}")
            inferred_id = (lookup.get('inferred_panel_id') or '').strip()
            if inferred_id:
                selectors.append(f"#{inferred_id.lstrip('#')}")
        default_selectors = [
            '.tab-pane[aria-hidden="false"]',
            '.tab-pane:not([hidden])',
            '[role="tabpanel"].show',
            '[role="tabpanel"][aria-hidden="false"]',
            '.tab-content .tab-pane.show',
            '.tab-content .tab-pane:not([hidden])',
            '.MuiTabPanel-root:not([hidden])',
            '.react-tabs__tab-panel--selected',
            '[aria-expanded="true"][role="tabpanel"]'
        ]
        for sel in default_selectors:
            if sel not in selectors:
                selectors.append(sel)
        return selectors
    
    def _extract_active_tab_content_from_source(
        self,
        driver,
        selectors: List[str],
        minimum_length: int = 20
    ) -> Tuple[str, Optional[str]]:
        """
        从 page_source 中提取当前激活 tab 的文本内容
        Returns: (text, matched_selector)
        """
        try:
            html = driver.page_source
        except Exception:
            return "", None
        if not html:
            return "", None
        soup = BeautifulSoup(html, 'html.parser')
        for sel in selectors:
            try:
                node = soup.select_one(sel)
            except Exception:
                node = None
            if node:
                text = node.get_text('\n', strip=True)
                if text and len(text) >= minimum_length:
                    return text, sel
        # fallback: aria-expanded true sections
        expanded_nodes = soup.select('[aria-expanded="true"]')
        for node in expanded_nodes:
            text = node.get_text('\n', strip=True)
            if text and len(text) >= minimum_length:
                return text, '[aria-expanded="true"]'
        return "", None
    
    def _extract_active_tab_content_with_wait(
        self,
        driver,
        element_dict: Optional[Dict[str, Any]],
        behavior: str,
        previous_text: str = "",
        before_inner_html: Optional[str] = None,
        debug_prefix: str = ""
    ) -> Tuple[bool, str]:
        """
        简化的 tab 内容提取：直接通过 panel id 获取内容
        """
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        import time as time_module
        import re

        timeout = 6 if behavior == 'static' else 10
        
        # 1. 从 element_dict 获取 tab 元素和文本
        tab_elem = None
        tab_text = ""
        if element_dict:
            tab_elem = element_dict.get('element')
            if tab_elem:
                try:
                    tab_text = tab_elem.text.strip() if tab_elem.text else ""
                    if not tab_text:
                        tab_text = (tab_elem.get_attribute('innerText') or tab_elem.get_attribute('textContent') or '').strip()
                    if not tab_text:
                        tab_text = (tab_elem.get_attribute('aria-label') or '').strip()
                except:
                    pass
                        
        # 2. 将 tab 文本转换为 panel id
        panel_id = None
        if tab_text:
            panel_id = tab_text.lower().strip()
            panel_id = re.sub(r'[^a-z0-9_-]', '', panel_id)
        
        # 3. 等待 panel 可见并提取内容
        if panel_id:
            try:
                # 等待 panel 元素出现并可见
                panel = WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.ID, panel_id))
                )
                
                # 等待 panel 可见（处理 Bootstrap fade 动画）
                WebDriverWait(driver, timeout).until(
                    EC.visibility_of_element_located((By.ID, panel_id))
                )
                
                # 额外等待动画完成
                time_module.sleep(0.5)
                
                # 提取文本内容
                text_content = panel.text.strip() if panel.text else ""
                
                if text_content and len(text_content) >= 20:
                    return True, text_content
            except Exception:
                pass
        
        # 4. Fallback: 尝试通过 _find_content_panel 找到 panel
        if tab_elem:
            try:
                panel = self._find_content_panel(driver, tab_elem, element_dict, debug_prefix)
                if panel:
                    text_content = panel.text.strip() if panel.text else ""
                    if text_content and len(text_content) >= 20:
                        return True, text_content
            except Exception:
                pass
        
        # 5. 最终 fallback: 查找当前可见的 tab-pane
        try:
            visible_panels = driver.find_elements(By.CSS_SELECTOR, 
                '.tab-pane.show, .tab-pane.active, [role="tabpanel"].show, [role="tabpanel"][aria-hidden="false"]')
            for panel in visible_panels:
                try:
                    if panel.is_displayed():
                        text_content = panel.text.strip() if panel.text else ""
                        if text_content and len(text_content) >= 20:
                            return True, text_content
                except:
                    continue
        except Exception:
            pass
        
        return False, ""
    
    def _is_panel_visible(self, panel) -> bool:
        """
        辅助方法：判断关联的内容面板是否真的处于可见/激活状态
        """
        try:
            if panel is None:
                return False
            if not panel.is_displayed():
                return False
            panel_class = (panel.get_attribute('class') or '').lower()
            aria_hidden = (panel.get_attribute('aria-hidden') or '').lower()
            if aria_hidden in ('true', '1'):
                return False
            # 如果没有明显的激活类，但仍处于显示状态，也认为可见
            return True
        except Exception:
            return False

    def _extract_panel_lookup(self, elem) -> Dict[str, Any]:
        """
        提取与 tab/按钮关联的面板信息（基于属性推断）
        返回包含候选定位信息的字典，便于后续快速定位内容面板
        """
        lookup: Dict[str, Any] = {'candidates': []}
        seen_candidates = set()

        def _add_candidate(method: str, value: str):
            nonlocal seen_candidates
            if not value:
                return
            key = f"{method}:{value}"
            if key in seen_candidates:
                return
            seen_candidates.add(key)
            lookup['candidates'].append({'method': method, 'value': value})

        try:
            aria_controls = (elem.get_attribute('aria-controls') or '').strip()
        except Exception:
            aria_controls = ''
        if aria_controls:
            _add_candidate('id', aria_controls)

        for attr in ['data-bs-target', 'data-target', 'data-panel', 'data-tab', 'data-section']:
            try:
                attr_value = (elem.get_attribute(attr) or '').strip()
            except Exception:
                attr_value = ''
            if not attr_value or attr_value in ('#', '##'):
                continue
            if attr_value.startswith('#'):
                _add_candidate('css', attr_value)
                _add_candidate('id', attr_value.lstrip('#'))
            else:
                _add_candidate('css', f'#{attr_value}')
                _add_candidate('id', attr_value)

        try:
            role_desc = (elem.get_attribute('data-controls') or '').strip()
        except Exception:
            role_desc = ''
        if role_desc:
            _add_candidate('css', role_desc)

        try:
            href_attr = (elem.get_attribute('href') or '').strip()
        except Exception:
            href_attr = ''
        fragment = ''
        if href_attr:
            if href_attr.startswith('#'):
                fragment = href_attr.lstrip('#')
            else:
                try:
                    from urllib.parse import urlparse
                    parsed_href = urlparse(href_attr)
                    fragment = parsed_href.fragment or ''
                except Exception:
                    fragment = ''
        if fragment:
            fragment = fragment.strip()
            _add_candidate('id', fragment)
            _add_candidate('css', f'#{fragment}')
            lookup['anchor_fragment'] = fragment

        if not lookup['candidates']:
            return {}
        return lookup

    def _is_tab_active(self, elem, panel_elem=None, element_dict: Optional[Dict[str, Any]] = None) -> bool:
        """
        检测 tab 是否已激活（避免重复点击）
        Args:
            elem: Selenium 元素对象
            
        Returns:
            True: tab 已激活，False: tab 未激活
        """
        try:
            if element_dict is not None and element_dict.get('panel_lookup'):
                lookup = element_dict['panel_lookup']
            else:
                lookup = self._extract_panel_lookup(elem) or {}
                if element_dict is not None and lookup:
                    element_dict['panel_lookup'] = lookup

            expected_ids = set()
            for candidate in lookup.get('candidates', []):
                method = candidate.get('method')
                value = (candidate.get('value') or '').strip()
                if not value:
                    continue
                if method == 'id':
                    expected_ids.add(value.lstrip('#'))
                elif method == 'css' and value.startswith('#'):
                    expected_ids.add(value.lstrip('#'))

            resolved_info = lookup.get('resolved') or {}
            resolved_value = (resolved_info.get('value') or '').strip()
            resolved_method = (resolved_info.get('method') or '').strip()
            if resolved_value:
                if resolved_method == 'id':
                    expected_ids.add(resolved_value.lstrip('#'))
                elif resolved_method == 'css' and resolved_value.startswith('#'):
                    expected_ids.add(resolved_value.lstrip('#'))

            inferred_panel_id = (lookup.get('inferred_panel_id') or '').strip()
            if inferred_panel_id:
                expected_ids.add(inferred_panel_id.lstrip('#'))

            inferred_labelledby = (lookup.get('inferred_panel_labelledby') or '').strip()
            if inferred_labelledby:
                expected_ids.add(inferred_labelledby.lstrip('#'))

            anchor_fragment = (lookup.get('anchor_fragment') or '').strip()
            if anchor_fragment:
                expected_ids.add(anchor_fragment.lstrip('#'))

            if not expected_ids:
                return False

            panel_is_visible = self._is_panel_visible(panel_elem)
            panel_matches_expected = False
            if panel_is_visible and panel_elem is not None:
                actual_id = ""
                try:
                    actual_id = (panel_elem.get_attribute('id') or '').strip()
                except Exception:
                    actual_id = ""
                if actual_id and actual_id.lstrip('#') in expected_ids:
                    panel_matches_expected = True
                else:
                    try:
                        aria_labelledby = (panel_elem.get_attribute('aria-labelledby') or '').strip()
                    except Exception:
                        aria_labelledby = ""
                    if aria_labelledby and aria_labelledby.lstrip('#') in expected_ids:
                        panel_matches_expected = True

            if not (panel_is_visible and panel_matches_expected):
                return False

            # 1. 检查 aria-selected 属性
            aria_selected = (elem.get_attribute('aria-selected') or '').lower()
            if aria_selected in ('true', '1'):
                return True
            
            # 2. 检查 aria-current
            aria_current = (elem.get_attribute('aria-current') or '').lower()
            if aria_current in ('page', 'true', '1'):
                return True

            return False
        except:
            return False
    
    def _resolve_click_element(self, driver, element_dict: Dict) -> Optional:
        """确保点击前元素引用有效，必要时根据 locator 重新获取"""
        elem = element_dict.get('element')    
        locator = element_dict.get('locator')
        if locator:
            by, value = locator
            try:
                refreshed = driver.find_element(by, value)
                element_dict['element'] = refreshed
                return refreshed
            except Exception:
                pass
        return None

    def _direct_navigate(
        self,
        driver,
        target_url: str,
        elem_label: str = ""
    ) -> Tuple[bool, str]:
        """直接导航到目标链接并抓取整页文本"""
        from selenium.webdriver.common.by import By
        import time

        label_display = elem_label or target_url
        try:
            driver.get(target_url)
            self._wait_for_dom_ready(driver, timeout=15)
            time.sleep(1.0)
            full_text = self._capture_page_text(driver, label=f"direct:{label_display}")
        except Exception:
            return False, ""

        if full_text and len(full_text.strip()) >= 50:
            return True, full_text

        return False, full_text or ""

    def _selenium_click_element_structured(
        self, 
        driver, 
        element_dict: Dict, 
        elem_label: str
    ) -> Tuple[bool, str, str]:
        """
        结构化点击元素，等待内容变化
        返回: (成功标志, 完整页面文本, 内容区域文本)
        """
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.common.by import By
        from selenium.webdriver.common.action_chains import ActionChains
        import hashlib
        import time
        import re
        from urllib.parse import urlparse, urljoin

        elem = self._resolve_click_element(driver, element_dict) if element_dict else None
        if elem is None:
            return False, "", ""        
        try:
            elem_tag = (getattr(elem, 'tag_name', '') or '').lower()
        except:
            elem_tag = ''
        
        try:
            href_after = elem.get_attribute('href') or ''
        except:
            href_after = ''

        stored_tag_name = element_dict.get('tag_name') if element_dict else ''
        if not elem_tag and stored_tag_name:
            elem_tag = stored_tag_name

        if not href_after and element_dict:
            href_after = element_dict.get('href', '')
        
        try:
            current_url_before = driver.current_url
        except:
            current_url_before = ''

        body_before = self._capture_page_text(driver, label="before_click")
        body_before_normalized = re.sub(r'\s+', ' ', body_before).strip() if body_before else ""
        
        def _format_reason(reason: str) -> str:
            return reason if reason else "Fallback"

        def _is_same_domain(target_url: str, base_url: str) -> bool:
            if not target_url:
                return False
            try:
                target_lower = target_url.lower()
                base_lower = (base_url or '').lower()
                if target_lower.startswith('http://') or target_lower.startswith('https://'):
                    target_domain = urlparse(target_lower).netloc or ''
                    base_domain = urlparse(base_lower).netloc or ''
                    if not target_domain or not base_domain:
                        return False
                    target_base = '.'.join(target_domain.split('.')[-2:]) if '.' in target_domain else target_domain
                    base_base = '.'.join(base_domain.split('.')[-2:]) if '.' in base_domain else base_domain
                    same = target_base == base_base
                    return same
                # 相对链接视为同域
                return True
            except:
                return False

        anchor_fragment = ""
        is_anchor_navigation = False
        if elem_tag == 'a' and href_after:
            try:
                parsed_href = urlparse(href_after)
                base_parsed = urlparse(current_url_before or '')
                if href_after.startswith('#'):
                    is_anchor_navigation = True
                    anchor_fragment = href_after.lstrip('#')
                elif parsed_href.fragment and _is_same_domain(href_after, current_url_before):
                    href_path = parsed_href.path or ''
                    base_path = base_parsed.path or ''
                    if not href_path or href_path == base_path:
                        is_anchor_navigation = True
                        anchor_fragment = parsed_href.fragment
                if is_anchor_navigation and anchor_fragment is None:
                    anchor_fragment = ""
            except Exception:
                is_anchor_navigation = False
                anchor_fragment = ""
        if not anchor_fragment and element_dict:
            lookup = element_dict.get('panel_lookup') or {}
            lookup_fragment = lookup.get('anchor_fragment')
            if lookup_fragment:
                anchor_fragment = lookup_fragment
                is_anchor_navigation = True

        def _navigate_and_capture(reason: str) -> Optional[Tuple[bool, str, str]]:
            if is_anchor_navigation:
                return None
            if elem_tag != 'a':
                return None
            href_to_use = href_after
            if not href_to_use:
                href_to_use = element_dict.get('href', '') if element_dict else ''
                if not href_to_use:
                    return None
            try:
                href_to_use = urljoin(current_url_before or '', href_to_use)
            except Exception:
                pass
            same_domain = _is_same_domain(href_to_use, current_url_before)
            if not same_domain:
                return None
            try:
                driver.get(href_to_use)
                self._wait_for_dom_ready(driver, timeout=12)
                time.sleep(1.0)
                full_text = self._capture_page_text(driver, label=f"navigate:{elem_label or href_to_use}")
            except Exception:
                return None
            if full_text and len(full_text.strip()) >= 50:
                return True, full_text, full_text
            return None
        
        def _capture_anchor_section_text(fragment: str) -> str:
            if not fragment:
                return ""
            search_targets = [
                (By.ID, fragment),
                (By.CSS_SELECTOR, f'[data-anchor="{fragment}"]'),
                (By.NAME, fragment),
                (By.CSS_SELECTOR, f'[data-section="{fragment}"]'),
                (By.CSS_SELECTOR, f'[data-tab="{fragment}"]'),
            ]
            target_elem = None
            for by, value in search_targets:
                try:
                    target_elem = driver.find_element(by, value)
                    if target_elem:
                        break
                except Exception:
                    continue
            if not target_elem:
                try:
                    target_elem = driver.find_element(By.CSS_SELECTOR, f'a[name="{fragment}"]')
                except Exception:
                    target_elem = None
            if not target_elem:
                return ""
            container = target_elem
            try:
                container = target_elem.find_element(
                    By.XPATH,
                    'ancestor-or-self::*[self::section or self::article or '
                    'contains(@class,"section") or contains(@class,"panel") or '
                    'contains(@class,"tab-pane") or contains(@class,"content")]'
                )
            except Exception:
                container = target_elem
            text_value = ""
            try:
                text_value = container.text or ""
            except Exception:
                text_value = ""
            if text_value and len(text_value.strip()) >= 10:
                return text_value
            try:
                text_value = self._extract_structured_text(container, driver)
            except Exception:
                text_value = ""
            return text_value or ""

        def _capture_section_by_label_text(label_text: str) -> str:
            if not label_text:
                return ""
            label_normalized = re.sub(r'\s+', ' ', label_text).strip().lower()
            if not label_normalized:
                return ""
            try:
                headings = driver.find_elements(By.CSS_SELECTOR, 'h1, h2, h3, h4, h5, h6')
            except Exception:
                headings = []
            for heading in headings:
                try:
                    heading_text = heading.text or ""
                except Exception:
                    heading_text = ""
                heading_text_norm = re.sub(r'\s+', ' ', heading_text).strip().lower()
                if not heading_text_norm:
                    continue
                if label_normalized in heading_text_norm:
                    section_elem = heading
                    # Try to expand to nearest section container
                    try:
                        section_elem = heading.find_element(By.XPATH, 'ancestor::section[1]')
                    except Exception:
                        try:
                            section_elem = heading.find_element(
                                By.XPATH,
                                'ancestor::div[contains(@class,"section") or contains(@class,"content") or contains(@class,"panel") or contains(@class,"entry-content")][1]'
                            )
                        except Exception:
                            section_elem = heading
                    try:
                        section_text = section_elem.text or ""
                    except Exception:
                        section_text = heading_text
                    if section_text and len(section_text.strip()) >= 30:
                        return section_text
            return ""

        def _normalize_url_no_fragment(url: str) -> str:
            if not url:
                return ""
            parsed = urlparse(url)
            normalized = parsed._replace(fragment='').geturl()
            return normalized.rstrip('/')

        try:
            
            elem_type = element_dict.get('type', '')
            panel_elem = None
            pre_panel_text = ""
            is_tab_element = elem_type == 'tab' or (elem.get_attribute('role') or '').lower() == 'tab'
            tab_behavior = None
            tab_identifier = None
            if is_tab_element:
                tab_behavior = self._classify_tab_behavior(elem, current_url_before)
                if tab_behavior == 'navigation':
                    is_tab_element = False

            if is_tab_element:
                # 先定位 panel（用于后续等待变化）
                debug_prefix_tab = f"[Tab: {elem_label}] "
                
                try:
                    tab_identifier = driver.execute_script("""
                        var el = arguments[0];
                        return {
                            outerHTML: el.outerHTML,
                            id: el.id || '',
                            className: el.className || '',
                            textContent: el.textContent || el.innerText || '',
                            x: el.getBoundingClientRect().x,
                            y: el.getBoundingClientRect().y,
                            // 尝试从 SVG 获取图标信息
                            svgIcon: (function() {
                                var svg = el.querySelector('svg');
                                if (svg) {
                                    return svg.getAttribute('data-icon') || 
                                           svg.getAttribute('aria-label') || 
                                           svg.getAttribute('title') || '';
                                }
                                return '';
                            })()
                        };
                    """, elem)
                except Exception:
                    pass
                
                try:
                    if tab_identifier:
                        tab_real_text = tab_identifier.get('svgIcon') or tab_identifier.get('textContent') or elem_label
                        if tab_real_text:
                            tab_real_text = tab_real_text.strip()[:50]
                    else:
                        tab_real_text = driver.execute_script("""
                            var el = arguments[0];
                            var svg = el.querySelector('svg');
                            if (svg) {
                                return svg.getAttribute('data-icon') || 
                                       svg.getAttribute('aria-label') || 
                                       svg.getAttribute('title') || '';
                            }
                            return el.textContent || el.innerText || el.getAttribute('aria-label') || '';
                        """, elem).strip()[:50]
                except:
                    tab_real_text = elem_label

                try:
                    panel_elem = self._find_content_panel(driver, elem, element_dict, debug_prefix=debug_prefix_tab)
                except Exception:
                    panel_elem = None
                if panel_elem is None:
                    try:
                        panel_elem = self._find_content_panel(driver, elem, element_dict, debug_prefix=debug_prefix_tab)
                    except Exception:
                        panel_elem = None
                # 记录点击前的 panel 内容和 innerHTML（用于检测变化）
                before_text = panel_elem.text or "" if panel_elem else ""
                try:
                    before_inner_html = panel_elem.get_attribute('innerHTML') or "" if panel_elem else ""
                except:
                    before_inner_html = ""
            else:
                panel_elem = self._find_content_panel(driver, elem, element_dict)
                before_text = panel_elem.text or "" if panel_elem else ""
                before_inner_html = ""

            # --- 面板 hash 函数---
            def hash_panel(panel):
                if panel is None:
                    return hashlib.md5(b"none_panel").hexdigest()
                try:
                    # 方法1: 基于文本内容（最重要）
                    text_content = panel.text or ""
                    text_hash = hashlib.md5(text_content.encode('utf-8')).hexdigest()[:8]
                    
                    # 方法2: 基于可见元素数量（辅助）
                    all_children = panel.find_elements(By.XPATH, './/*')
                    visible_count = len([c for c in all_children if c.is_displayed()])
                    
                    # 方法3: 基于关键元素（如列表项、段落等）
                    key_elements = panel.find_elements(By.XPATH, './/li | .//p | .//div[@class]')
                    key_count = len([e for e in key_elements if e.is_displayed()])
                    
                    # 组合 hash（文本内容权重最高）
                    combined = f"{text_hash}|{visible_count}|{key_count}"
                    return hashlib.md5(combined.encode('utf-8')).hexdigest()
                except Exception as e:
                    # 降级：使用简单的 HTML hash
                    try:
                        html = panel.get_attribute('innerHTML') or ""
                        return hashlib.md5(html.encode('utf-8')).hexdigest()
                    except:
                        return hashlib.md5(b"error_panel").hexdigest()

            if element_dict is not None:
                element_dict.pop('cached_full_text', None)
                element_dict.pop('cached_panel_text', None)
            
            if is_tab_element:
                before_text_length = len(before_text)
                text_normalized_before = re.sub(r'\s+', ' ', before_text).strip()
            else:
                if panel_elem is not None:
                    before_hash = hash_panel(panel_elem)
                    before_text = panel_elem.text or ""
                    before_text_length = len(before_text)
                    text_normalized_before = re.sub(r'\s+', ' ', before_text).strip()
                else:
                    before_hash = hash_panel(None)
                    before_text = ""
                    before_text_length = 0
                    text_normalized_before = ""

            content_updated = False
            if is_anchor_navigation and anchor_fragment:
                time.sleep(0.5)
                anchor_text = _capture_anchor_section_text(anchor_fragment)
                anchor_text_normalized = re.sub(r'\s+', ' ', anchor_text).strip() if anchor_text else ""
                if anchor_text_normalized and anchor_text_normalized != text_normalized_before and len(anchor_text_normalized) >= 50:
                    return True, anchor_text, anchor_text

            direct_navigation_attempted = False
            original_href_after = href_after
            if (
                elem_tag == 'a'
                and href_after
                and not is_anchor_navigation
                and not href_after.strip().lower().startswith('javascript:')
                and (not is_tab_element or tab_behavior == 'navigation')
            ):
                try:
                    direct_target = urljoin(current_url_before or '', href_after)
                except Exception:
                    direct_target = href_after
                normalized_target = _normalize_url_no_fragment(direct_target)
                normalized_current = _normalize_url_no_fragment(current_url_before)
                if (
                    direct_target
                    and normalized_target
                    and normalized_target != normalized_current
                    and _is_same_domain(direct_target, current_url_before)
                ):
                    href_after = direct_target
                    nav_result = _navigate_and_capture("Pre-click direct navigation")
                    if nav_result:
                        return nav_result
                    href_after = original_href_after
                    direct_navigation_attempted = True

            # --- 滚动到元素 ---
            try:
                driver.execute_script(
                    "arguments[0].scrollIntoView({behavior: 'auto', block: 'center'});", elem
                )
                time.sleep(0.5)
            except Exception:
                pass

            # --- 点击元素 ---
            click_elem = elem
            if is_tab_element:
                try:
                    # 检查当前元素是否是 SVG 或图标元素
                    tag_name = elem.tag_name.lower()
                    is_icon = tag_name in ['svg', 'i', 'path', 'use'] or 'icon' in (elem.get_attribute('class') or '').lower()
                    
                    if is_icon:
                        # 如果是图标，向上查找 button 或 a 标签
                        try:
                            container = elem.find_element(By.XPATH, './ancestor::button[1] | ./ancestor::a[1] | ./ancestor::*[@role="tab"][1]')
                            click_elem = container
                        except:
                            # 如果找不到容器，尝试使用 JS 查找
                            try:
                                container = driver.execute_script("""
                                    var el = arguments[0];
                                    var parent = el;
                                    while (parent && parent !== document.body) {
                                        if (parent.tagName === 'BUTTON' || parent.tagName === 'A' || 
                                            parent.getAttribute('role') === 'tab') {
                                            return parent;
                                        }
                                        parent = parent.parentElement;
                                    }
                                    return el;
                                """, elem)
                                if container and container != elem:
                                    click_elem = container
                            except:
                                pass
                    
                    if tab_identifier:
                        try:
                            # 使用保存的标识重新查找元素
                            tablist = driver.find_element(By.XPATH, '//*[@role="tablist"]')
                            all_tabs = tablist.find_elements(By.XPATH, './/*[@role="tab"]')
                            
                            for t in all_tabs:
                                try:
                                    t_identifier = driver.execute_script("""
                                        var el = arguments[0];
                                        return {
                                            outerHTML: el.outerHTML,
                                            id: el.id || '',
                                            x: el.getBoundingClientRect().x,
                                            y: el.getBoundingClientRect().y
                                        };
                                    """, t)
                                    
                                    # 比较标识
                                    if (t_identifier['outerHTML'] == tab_identifier['outerHTML'] or
                                        (tab_identifier['id'] and t_identifier['id'] == tab_identifier['id']) or
                                        (abs(t_identifier['x'] - tab_identifier['x']) < 1 and 
                                         abs(t_identifier['y'] - tab_identifier['y']) < 1)):
                                        click_elem = t
                                        break
                                except:
                                    continue
                        except Exception:
                            pass
                except Exception:
                    pass
            
            click_success = False
            for method, func in [
                ("JS click", lambda e=click_elem: driver.execute_script("arguments[0].click();", e)),
                ("Regular click", lambda e=click_elem: e.click()),
                ("ActionChains click", lambda e=click_elem: ActionChains(driver).move_to_element(e).click().perform())
            ]:
                try:
                    func()
                    click_success = True
                    break
                except Exception as e:
                    try:
                        err_msg = str(e).lower()
                    except:
                        err_msg = ""
                    if 'stale element' in err_msg:
                        nav_result = _navigate_and_capture("Direct navigation after stale element")
                        if nav_result:
                            return nav_result

            if not click_success:
                nav_result = _navigate_and_capture("Fallback navigation after click failure")
                if nav_result:
                    return nav_result
                if panel_elem is not None:
                    panel_text = panel_elem.text or ""
                    return False, panel_text, panel_text
                else:
                    return False, "", ""
            try:
                window_handles = driver.window_handles
            except Exception:
                window_handles = []
            if window_handles and len(window_handles) > 1:
                try:
                    driver.switch_to.window(window_handles[-1])
                except Exception:
                    pass
            try:
                current_url_after_click = driver.current_url or ""
            except Exception:
                current_url_after_click = ""

            if is_tab_element and tab_behavior in ('static', 'ajax'):
                debug_prefix_tab = f"[Tab: {elem_label}] "
                
                # 记录点击前的 panel innerHTML（用于比较变化）
                before_inner_html_str = before_inner_html if 'before_inner_html' in locals() else ""
                
                # 等待一小段时间让 tab 切换完成
                time.sleep(0.5)
                
                # 重新定位 panel（点击后 panel 可能已经切换）
                try:
                    updated_panel_elem = self._find_content_panel(driver, elem, element_dict, debug_prefix=debug_prefix_tab)
                    if updated_panel_elem:
                        panel_elem = updated_panel_elem
                except Exception:
                    pass
                
                # 使用等待方法提取 tab 内容（传入 before_inner_html 用于比较）
                success_tab, tab_text = self._extract_active_tab_content_with_wait(
                    driver,
                    element_dict,
                    tab_behavior,
                    previous_text=before_text,
                    before_inner_html=before_inner_html_str,
                    debug_prefix=debug_prefix_tab
                )
                if success_tab and tab_text and len(tab_text.strip()) >= 20:
                    # 验证内容是否真的变化了
                    tab_text_normalized = re.sub(r'\s+', ' ', tab_text).strip()
                    if tab_text_normalized != text_normalized_before:
                        if element_dict is not None:
                            element_dict['cached_full_text'] = tab_text
                            element_dict['cached_panel_text'] = tab_text
                        return True, tab_text, tab_text
                    # 继续执行后续的 panel 变化检测
                # 继续执行后续的 panel 变化检测
            #导航检测：若是同域 a 标签点击导致整页跳转，直接抓取整页文本
            try:
                if elem_tag == 'a' and href_after and not is_anchor_navigation and _is_same_domain(href_after, current_url_before):
                    # 等待 URL 改变或文档 ready
                    changed = False
                    try:
                        WebDriverWait(driver, 5).until(lambda d: (d.current_url or '').lower() != (current_url_before or '').lower())
                        changed = True
                    except:
                        pass
                    time.sleep(1.0)
                    try:
                        current_url_after_wait = driver.current_url or ""
                    except Exception:
                        current_url_after_wait = ""
                    try:
                        window_handles_after_wait = driver.window_handles
                    except Exception:
                        window_handles_after_wait = []
                    if window_handles_after_wait and len(window_handles_after_wait) > 1:
                        try:
                            driver.switch_to.window(window_handles_after_wait[-1])
                            current_url_after_wait = driver.current_url or ""
                        except Exception:
                            pass
                    full_text = self._capture_page_text(driver, label="post_navigation")
                    if full_text and len(full_text.strip()) >= 50:
                        return True, full_text, full_text
                    # 如果 URL 仍未变化且目标链接与当前不同，主动导航
                    try:
                        href_base = href_after.split('#')[0].rstrip('/')
                        current_base = (current_url_after_wait or '').split('#')[0].rstrip('/')
                    except Exception:
                        href_base = href_after
                        current_base = current_url_after_wait
                    if href_base and current_base and href_base.lower() != current_base.lower():
                        nav_result = _navigate_and_capture("Forced navigation after unchanged URL")
                        if nav_result:
                            return nav_result
            except Exception:
                pass

            # --- 等待 panel 内容变化（增强版：更敏感的多重检测）---
            def wait_panel_change(drv):
                """
                检测同一 panel 内的内容变化
                支持检测：文本替换、内容追加、列表项变化、DOM 结构变化等
                """
                try:
                    try:
                        cur_panel = panel_elem
                        _ = cur_panel.text
                    except:
                        # Stale element，重新定位
                        try:
                            # 先尝试查找tab-content容器（不依赖elem）
                            try:
                                # 方法1: 直接查找tab-content（不依赖elem）
                                tab_content = driver.find_element(By.CSS_SELECTOR, '.tab-content')
                            except:
                                # 方法2: 尝试通过elem查找（如果elem仍然有效）
                                try:
                                    parent = elem.find_element(By.XPATH, '..')
                                    tab_content = parent.find_element(By.XPATH, 
                                        './following-sibling::*[contains(@class, "tab-content")]')
                                except:
                                    # 方法3: 查找所有tab-content
                                    tab_contents = driver.find_elements(By.CSS_SELECTOR, '.tab-content, [class*="tab-content"]')
                                    if tab_contents:
                                        tab_content = tab_contents[0]
                                    else:
                                        raise Exception("Cannot find tab-content")
                            # 查找所有panel，返回第一个可见的
                                all_panels = tab_content.find_elements(By.XPATH,
                                    './/*[@role="tabpanel"] | .//*[contains(@class, "tab-pane")]')
                                for p in all_panels:
                                    if p.is_displayed():
                                        cur_panel = p
                                        break
                                else:
                                    # 最后fallback：返回tab-content本身
                                    cur_panel = tab_content
                        except Exception:
                            # Fallback：使用通用选择器
                            try:
                                cur_panel = driver.find_element(By.CSS_SELECTOR, 
                                    '[role="tabpanel"]:not([hidden]), .tab-pane:not([hidden])')
                            except:
                                # 最终fallback：使用 documentElement（兼容无 body 的 frameset）
                                try:
                                    cur_panel = driver.execute_script("return document.body || document.documentElement;")
                                except Exception:
                                    cur_panel = None
                    if cur_panel is None:
                        return False
                    
                    cur_hash = hash_panel(cur_panel)
                    cur_text_length = len(cur_panel.text or "")
                    cur_text = cur_panel.text or ""
                    # ✅ 多重判断：hash 变化 OR 文本长度显著变化 OR 文本内容变化
                    hash_changed = cur_hash != before_hash
                    length_changed = abs(cur_text_length - before_text_length) > 5
                    # 注意：text_normalized_before 已在外部作用域定义
                    text_normalized_cur = re.sub(r'\s+', ' ', cur_text).strip()
                    text_changed = text_normalized_cur != text_normalized_before
                    before_items = len(re.findall(r'•|^\d+\.|^[-*]', before_text, re.MULTILINE))
                    cur_items = len(re.findall(r'•|^\d+\.|^[-*]', cur_text, re.MULTILINE))
                    items_changed = abs(cur_items - before_items) > 0
                    before_headers = len(re.findall(r'^#+\s+', before_text, re.MULTILINE))
                    cur_headers = len(re.findall(r'^#+\s+', cur_text, re.MULTILINE))
                    headers_changed = abs(cur_headers - before_headers) > 0
                    
                    try:
                        if panel_elem is not None:
                            before_children = len([c for c in panel_elem.find_elements(By.XPATH, './/*') if c.is_displayed()])
                        else:
                            before_children = 0
                        cur_children = len([c for c in cur_panel.find_elements(By.XPATH, './/*') if c.is_displayed()])
                        dom_structure_changed = abs(cur_children - before_children) > 0
                    except:
                        # 如果before_children获取失败（stale element 或 None），跳过这个检测
                        dom_structure_changed = False
                    
                    try:
                        if panel_elem is not None:
                            before_links = len(panel_elem.find_elements(By.TAG_NAME, 'a'))
                        else:
                            before_links = 0
                        cur_links = len(cur_panel.find_elements(By.TAG_NAME, 'a'))
                        links_changed = abs(cur_links - before_links) > 0
                    except:
                        # 如果before_links获取失败（stale element 或 None），跳过这个检测
                        links_changed = False
                    
                    N = 100  # 检查前后100个字符
                    if len(before_text) >= N and len(cur_text) >= N:
                        before_start = text_normalized_before[:N]
                        before_end = text_normalized_before[-N:] if len(text_normalized_before) > N else ""
                        cur_start = text_normalized_cur[:N]
                        cur_end = text_normalized_cur[-N:] if len(text_normalized_cur) > N else ""
                        content_replaced = (before_start != cur_start) or (before_end != cur_end)
                    else:
                        content_replaced = False
                    
                    any_change = (hash_changed or length_changed or text_changed or 
                                 items_changed or headers_changed or dom_structure_changed or 
                                 links_changed or content_replaced)
                    
                    if any_change:
                        return True
                    else:
                        return False
                except Exception:
                    return False

            try:
                WebDriverWait(driver, 10).until(wait_panel_change)  # 从8秒增加到10秒
                content_updated = True
                time.sleep(2)
            except Exception:
                try:
                    try:
                        if panel_elem is not None:
                            _ = panel_elem.text
                            final_panel = panel_elem
                        else:
                            raise Exception("panel_elem is None")
                    except:
                        final_panel = self._find_content_panel(driver, elem, element_dict)
                    
                    if final_panel is None:
                        final_text_normalized = ""
                    else:
                        final_text = final_panel.text or ""
                        final_text_normalized = re.sub(r'\s+', ' ', final_text).strip()
                    
                    # 如果文本有明显差异（即使其他检测没触发），也认为有变化
                    if final_text_normalized != text_normalized_before:
                        text_diff = abs(len(final_text_normalized) - len(text_normalized_before))
                        if text_diff > 3:  # 至少3个字符的差异
                            # 继续处理，认为内容已变化
                            content_updated = True
                except Exception:
                    pass

            # --- 获取最终文本 ---
            try:
                try:
                    # 测试panel_elem是否仍然有效
                    if panel_elem is not None:
                        _ = panel_elem.text
                        final_panel_elem = panel_elem
                    else:
                        raise Exception("panel_elem is None")
                except:
                    # Stale element 或 None，重新定位
                    final_panel_elem = self._find_content_panel(driver, elem, element_dict)
                
                # 尝试多种方式获取文本
                final_text = ""
                if final_panel_elem:
                    try:
                        final_text = final_panel_elem.text
                        # 如果 panel 文本为空或太短，尝试获取 innerHTML 的文本
                        if not final_text or len(final_text.strip()) < 10:
                            final_text = final_panel_elem.get_attribute('innerText') or final_panel_elem.get_attribute('textContent') or ""
                    except Exception:
                        pass
                
                # 如果 panel 文本仍然为空，尝试获取整个 body
                if not final_text or len(final_text.strip()) < 10:
                    final_text = self._capture_page_text(driver, label="final_panel_fallback")

                body_after = self._capture_page_text(driver, label="post_click_diff")
                body_after_normalized = re.sub(r'\s+', ' ', body_after).strip() if body_after else ""

                if body_after_normalized and body_after_normalized != body_before_normalized and len(body_after_normalized) >= 80:
                    return True, body_after, body_after

                label_section_text = _capture_section_by_label_text(elem_label)
                label_section_normalized = re.sub(r'\s+', ' ', label_section_text).strip() if label_section_text else ""
                if label_section_normalized and label_section_normalized != text_normalized_before and len(label_section_normalized) >= 50:
                    return True, label_section_text, label_section_text
                
                if final_text and len(final_text.strip()) >= 50:
                    final_text_normalized = re.sub(r'\s+', ' ', final_text).strip()
                    if not content_updated:
                        if final_text_normalized != text_normalized_before:
                            diff_len = abs(len(final_text_normalized) - len(text_normalized_before))
                            if diff_len > 3:
                                content_updated = True
                    if not content_updated:
                        return False, "", ""
                    if element_dict is not None:
                        element_dict['cached_full_text'] = final_text
                        element_dict['cached_panel_text'] = final_text
                    return True, final_text, final_text
                else:
                    return False, final_text or "", final_text or ""
            except Exception:
                return False, "", ""

        except Exception as e:
            try:
                err_msg = str(e).lower()
            except:
                err_msg = ""
            if 'stale element' in err_msg:
                nav_result = _navigate_and_capture("Final fallback after exception")
                if nav_result:
                    return nav_result
            return False, "", ""

    def _selenium_click_element(self, driver, element_dict: Dict) -> Tuple[bool, str]:
        """Selenium 点击元素（旧方法，保留兼容性）"""
        success, full_text, _ = self._selenium_click_element_structured(driver, element_dict, "Unknown")
        return success, full_text
    
    def _extract_text(self, page: Page) -> str:
        """提取页面文本"""
        html = page.content()
        soup = BeautifulSoup(html, 'html.parser')
        
        # 移除无关标签
        for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            tag.decompose()
        
        return soup.get_text(separator='\n', strip=True)
    
    def _extract_interactive_elements(self, page: Page) -> List[Dict[str, str]]:
        """提取可交互元素"""
        elements = []
        
        selectors = [
            ('button:visible', 'button'),
            ('a:visible', 'link'),
            ('[class*="expand"]:visible', 'expandable'),
            ('[class*="show"]:visible', 'expandable'),
            ('[role="tab"]:visible', 'tab'),
        ]
        
        for selector, elem_type in selectors:
            try:
                handles = page.query_selector_all(selector)
                for handle in handles[:15]:  # 最多15个元素
                    try:
                        text = handle.inner_text()[:80].strip()
                        if not text or len(text) < 2:
                            continue
                        
                        # 检查是否可见
                        if not handle.is_visible():
                            continue
                        
                        elements.append({
                            'type': elem_type,
                            'text': text,
                            'handle': handle  # 保存handle用于点击
                        })
                    except:
                        continue
            except:
                continue
        
        return elements
    
    def _analyze_dimension_coverage(
        self, 
        text: str, 
        required_dimensions: List[str]
    ) -> Dict[str, DimensionCoverage]:
        """
        分析文本的维度覆盖情况（改进版：更严格的判断）
        
        Returns:
            {dimension_name: DimensionCoverage}
        """
        text_lower = text.lower()
        coverage = {}
        
        for dim_name in required_dimensions:
            dim_info = NINE_DIMENSIONS[dim_name]
            keywords = dim_info['keywords']
            
            # ✅ 改进1：计算实质性内容（排除单个词出现）
            substantial_hits = 0
            total_keyword_length = 0
            
            for kw in keywords:
                # 查找所有出现位置
                occurrences = []
                start = 0
                while True:
                    idx = text_lower.find(kw, start)
                    if idx == -1:
                        break
                    occurrences.append(idx)
                    start = idx + 1
                
                if occurrences:
                    # ✅ 改进2：检查关键词周围是否有实质内容
                    for idx in occurrences:
                        # 提取关键词前后各100字符
                        context = text[max(0, idx-100):min(len(text), idx+len(kw)+100)]
                        
                        # 实质性判断：周围至少有50个字符的内容
                        if len(context) > 50:
                            substantial_hits += 1
                            total_keyword_length += len(context)
                            break  # 每个关键词只计数一次
            
            # ✅ 改进3：更严格的覆盖判断
            # 需要：至少2个关键词匹配 或 单个关键词但周围内容充实
            if substantial_hits >= 2:
                covered = True
                confidence = min(substantial_hits / len(keywords), 1.0)
            elif substantial_hits == 1 and total_keyword_length > 200:
                covered = True
                confidence = 0.5
            else:
                covered = False
                confidence = 0.0
            
            # 提取预览（显示最相关的部分）
            preview = ""
            if substantial_hits > 0:
                for kw in keywords:
                    idx = text_lower.find(kw)
                    if idx != -1:
                        preview = text[max(0, idx-50):idx+150]
                        break
            
            coverage[dim_name] = DimensionCoverage(
                dimension_name=dim_name,
                covered=covered,
                confidence=confidence,
                content_preview=preview
            )
        
        return coverage
    
    def _llm_decide_for_dimensions(
        self,
        current_text: str,
        missing_dimensions: List[str],
        interactive_elements: List[Dict],
        history: List[Dict]
    ) -> Dict[str, Any]:
        """
        LLM 决策：点击哪个元素来获取缺失的维度
        """
        if not self.llm_available:
            return self._heuristic_decide(missing_dimensions, interactive_elements)
        
        # 构建元素描述
        elements_desc = []
        for i, elem in enumerate(interactive_elements[:10]):
            elements_desc.append(f"{i+1}. [{elem['type']}] \"{elem['text']}\"")
        
        # 构建缺失维度描述
        missing_dims_desc = []
        for dim_name in missing_dimensions:
            dim_info = NINE_DIMENSIONS[dim_name]
            missing_dims_desc.append(f"- {dim_info['name']}: {dim_info['description']}")
        
        prompt = f"""You are analyzing a researcher's webpage to extract profile information.

CURRENT SITUATION:
- We need to extract the following information dimensions:
{chr(10).join(missing_dims_desc)}

CURRENT PAGE CONTENT (first 800 chars):
{current_text[:800]}

AVAILABLE INTERACTIVE ELEMENTS:
{chr(10).join(elements_desc)}

INTERACTION HISTORY:
{json.dumps(history[-3:], indent=2) if history else 'None'}

TASK:
Decide which element to click to get information about the MISSING dimensions.
Consider:
1. Element text that matches dimension keywords (e.g., "Publications" for publications dimension)
2. Expandable sections that might reveal more content
3. Tabs or links that navigate to detailed pages

Respond in JSON format:
{{
  "action": "click" or "stop",
  "element_index": 1-{len(elements_desc)} (if action is click),
  "expected_dimensions": ["dimension_name1", "dimension_name2"],
  "reason": "Brief explanation"
}}

If no element seems helpful, respond with "stop".
"""
        
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a precise web interaction decision maker. Return valid JSON only.",
                },
                {"role": "user", "content": prompt},
            ]
            response = self.llm.invoke(messages)
            
            result_text = response.content.strip()
            result_text = re.sub(r'^```json?\s*', '', result_text)
            result_text = re.sub(r'\s*```$', '', result_text)
            
            decision_data = json.loads(result_text)
            
            if decision_data['action'] == 'stop':
                return {'action': 'stop', 'reason': decision_data.get('reason', 'LLM decided to stop')}
            
            elif decision_data['action'] == 'click':
                elem_idx = decision_data.get('element_index', 1) - 1
                if 0 <= elem_idx < len(interactive_elements):
                    return {
                        'action': 'click',
                        'target_element': interactive_elements[elem_idx],
                        'expected_dimensions': decision_data.get('expected_dimensions', []),
                        'reason': decision_data.get('reason', 'LLM suggested')
                    }
        
        except Exception as e:
            print(f"[9D Agent] ⚠️ LLM decision failed: {e}")
        
        return self._heuristic_decide(missing_dimensions, interactive_elements)
    
    def _heuristic_decide(
        self, 
        missing_dimensions: List[str], 
        interactive_elements: List[Dict]
    ) -> Dict[str, Any]:
        """启发式决策（无 LLM 时）"""
        # 为每个元素打分
        element_scores = []
        
        for elem in interactive_elements:
            text_lower = elem['text'].lower()
            score = 0
            matched_dims = []
            
            for dim_name in missing_dimensions:
                dim_keywords = NINE_DIMENSIONS[dim_name]['keywords']
                for kw in dim_keywords:
                    if kw in text_lower:
                        score += 10
                        matched_dims.append(dim_name)
                        break
            
            # 额外加分：展开/显示更多
            if any(kw in text_lower for kw in ['show', 'more', 'expand', 'see all', 'view all']):
                score += 5
            
            element_scores.append((score, elem, matched_dims))
        
        # 选择最高分
        element_scores.sort(reverse=True, key=lambda x: x[0])
        
        if element_scores and element_scores[0][0] > 0:
            score, elem, matched_dims = element_scores[0]
            return {
                'action': 'click',
                'target_element': elem,
                'expected_dimensions': matched_dims,
                'reason': f'Heuristic: matched keywords (score={score})'
            }
        
        return {'action': 'stop', 'reason': 'Heuristic: no relevant elements'}
    
    def _execute_click(self, page: Page, element: Dict) -> Tuple[bool, str]:
        """执行点击"""
        try:
            before_text = self._extract_text(page)
            before_length = len(before_text)
            
            # 点击
            handle = element.get('handle')
            if handle:
                handle.click(timeout=5000)
            else:
                return False, ""
            
            # 等待内容加载
            time.sleep(2)
            page.wait_for_load_state('networkidle', timeout=10000)
            
            # 获取新内容
            after_text = self._extract_text(page)
            after_length = len(after_text)
            
            new_content_length = after_length - before_length
            
            if new_content_length > 100:
                return True, after_text
            else:
                return False, ""
        
        except Exception as e:
            print(f"[9D Agent] ❌ Click failed: {e}")
            return False, ""
    
    def _fallback_enhanced_static(
        self, 
        url: str, 
        author_name: str,
        required_dimensions: List[str]
    ) -> Tuple[str, Dict]:
        """
        增强的静态抓取 - 使用 requests-html 或 selenium-wire（如果可用）
        支持基本的 JavaScript 渲染
        """
        print(f"[9D Agent] 📄 Using enhanced static fallback")
        # 方案 1: 尝试使用 requests-html（支持 JavaScript 渲染）
        try:
            from requests_html import HTMLSession
            print(f"[9D Agent] ✨ Using requests-html for JS rendering")
            session = HTMLSession()
            response = session.get(url, timeout=15)
            
            # 渲染 JavaScript
            response.html.render(timeout=20, sleep=2)
            
            # 提取文本
            text = response.html.text
            
            # 分析维度覆盖
            coverage = self._analyze_dimension_coverage(text, required_dimensions)
            covered_count = sum(1 for c in coverage.values() if c.covered)
            
            session.close()
            
            print(f"[9D Agent] ✅ Enhanced static: {len(text)} chars")
            print(f"[9D Agent]   - Coverage: {covered_count}/{len(required_dimensions)} dimensions")
            
            return text, {
                'method': 'enhanced_static_js',
                'text_length': len(text),
                'dimension_coverage': {
                    'covered': covered_count,
                    'total': len(required_dimensions),
                    'percentage': covered_count / len(required_dimensions) * 100
                }
            }
        
        except ImportError:
            print(f"[9D Agent] ⚠️ requests-html not available, using basic static")
        except Exception as e:
            print(f"[9D Agent] ⚠️ requests-html failed: {str(e)[:80]}")
        
        # 方案 2: 基础静态抓取
        return self._fallback_static(url)
    
    def _fallback_static(self, url: str) -> Tuple[str, Dict]:
        """基础静态抓取 fallback"""
        print(f"[9D Agent] 📄 Using basic static fallback")
        try:
            import requests
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.9',
            }
            
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            response.encoding = response.apparent_encoding
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 移除无关标签
            for tag in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
                tag.decompose()
            
            text = soup.get_text(separator='\n', strip=True)
            
            print(f"[9D Agent] ✅ Basic static: {len(text)} chars")
            
            return text, {
                'method': 'basic_static',
                'text_length': len(text)
            }
        
        except Exception as e:
            print(f"[9D Agent] ❌ Static fallback failed: {e}")
            return "", {'error': str(e), 'method': 'failed'}


# ==================== 便捷函数 ====================

def fetch_with_nine_dimension_awareness(
    url: str,
    author_name: str = "",
    api_key: str = None,
    required_dimensions: List[str] = None,
    headless: bool = True
) -> Tuple[str, Dict[str, Any]]:
    """
    便捷函数：以九维度为目标的智能抓取
    
    Args:
        url: 目标 URL
        author_name: 作者姓名
        api_key: LLM API key
        required_dimensions: 需要的维度列表（None=全部）
        headless: 无头模式
    
    Returns:
        (完整文本, 元数据)
    """
    agent = NineDimensionAwareAgent(api_key=api_key, headless=headless)
    return agent.fetch_for_nine_dimensions(
        url=url,
        author_name=author_name,
        max_interactions=50,
        required_dimensions=required_dimensions
    )
