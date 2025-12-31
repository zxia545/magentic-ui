"""
arXiv API Fallback Module
====================================
当 Semantic Scholar 未找到论文时，尝试从 arXiv API 获取数据
"""

import requests
import time
import re
import threading
from typing import Optional, Dict, Any
import xml.etree.ElementTree as ET


class ArxivFallbackClient:
    """arXiv API 客户端（作为 S2 的回退方案）"""
    
    def __init__(self, requests_per_second: float = 0.33):
        """
        Args:
            requests_per_second: 速率限制（arXiv建议每3秒1次，即0.33 rps）
        """
        self.base_url = "http://export.arxiv.org/api/query"
        self._last_call_ts = 0.0
        self._min_interval = 1.0 / requests_per_second
        self._lock = threading.Lock()
    
    def _throttle(self):
        """速率限制（线程安全）"""
        with self._lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)
            self._last_call_ts = time.monotonic()

    def _compute_backoff(self, attempt: int, base_wait: float, retry_after: Optional[str] = None) -> float:
        """根据 Retry-After 或指数退避计算等待时间"""
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass
        return base_wait * attempt
    
    def _normalize_title(self, title: str) -> str:
        """
        最小化标题归一化：只去除首尾空白和多余空格，保留所有标点符号
        避免过度处理导致匹配失败（如 Theory-of-Mind?, GPT-3 等）
        """
        # 只去除多余空格，保留所有标点符号（包括连字符、问号等）
        title = re.sub(r'\s+', ' ', title).strip()
        return title
    
    def search_by_title(
        self,
        title: str,
        year: Optional[str] = None,
        max_results: int = 3
    ) -> Optional[Dict[str, Any]]:
        """
        通过标题搜索arXiv论文
        Args:
            title: 论文标题
            year: 可选年份过滤
            max_results: 最多返回结果数（用于匹配）
        Returns:
            论文详情字典，未找到返回None
        """
        print(f"[arXiv Fallback] Searching for: {title[:80]}...")
        
        # 构建查询（使用原始标题，只做最小清理）
        normalized_title = self._normalize_title(title)
        print(f"[arXiv Fallback] Query string: {normalized_title[:80]}...")
        
        # arXiv API 查询参数
        params = {
            "search_query": f'ti:"{normalized_title}"',
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending"
        }
        
        try:
            response = None
            max_attempts = 4
            base_backoff = 3.0

            for attempt in range(1, max_attempts + 1):
                try:
                    self._throttle()
                    response = requests.get(self.base_url, params=params, timeout=10)
                    if response.status_code == 429:
                        wait = self._compute_backoff(attempt, base_backoff, response.headers.get("Retry-After"))
                        print(f"[arXiv Fallback] Rate limited (429). Waiting {wait:.1f}s before retry {attempt}/{max_attempts}")
                        time.sleep(wait)
                        continue
                    response.raise_for_status()
                    break
                except requests.RequestException as e:
                    if attempt >= max_attempts:
                        print(f"[arXiv Fallback] Request error after {attempt} attempts: {e}")
                        return None
                    wait = self._compute_backoff(attempt, base_backoff)
                    print(f"[arXiv Fallback] Request error ({e}), retrying in {wait:.1f}s ({attempt}/{max_attempts})")
                    time.sleep(wait)
            else:
                return None
            
            if response is None:
                print("[arXiv Fallback] No valid response received after retries")
                return None

            # 解析 Atom XML 响应
            root = ET.fromstring(response.content)
            
            # 命名空间
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            # 查找所有条目
            entries = root.findall('atom:entry', ns)
            
            if not entries:
                print(f"[arXiv Fallback] No results found")
                return None
            
            # 获取第一个结果（最相关）
            entry = entries[0]
            
            # 提取标题并验证匹配度
            arxiv_title = entry.find('atom:title', ns).text.strip()
            arxiv_title_normalized = self._normalize_title(arxiv_title)
            
            # 简单的标题匹配验证（降低阈值到40%，因为保留了标点符号）
            query_words = set(normalized_title.lower().split())
            result_words = set(arxiv_title_normalized.lower().split())
            if query_words and result_words:
                overlap = len(query_words & result_words) / len(query_words)
                print(f"[arXiv Fallback] Title match score: {overlap:.1%}")
                print(f"[arXiv Fallback] Found title: '{arxiv_title}'")
                if overlap < 0.4:
                    print(f"[arXiv Fallback] Low title match ({overlap:.1%}), skipping")
                    return None
            
            # 提取数据
            arxiv_id = entry.find('atom:id', ns).text.split('/abs/')[-1]
            
            # 提取摘要
            summary = entry.find('atom:summary', ns)
            abstract = summary.text.strip() if summary is not None else ""
            
            # 提取作者
            authors = []
            for author in entry.findall('atom:author', ns):
                name_elem = author.find('atom:name', ns)
                if name_elem is not None:
                    authors.append(name_elem.text.strip())
            
            # 提取发布日期
            published = entry.find('atom:published', ns)
            pub_year = None
            if published is not None:
                pub_date = published.text
                pub_year = int(pub_date[:4]) if pub_date else None
            
            result = {
                "paper_id": arxiv_id,
                "title": arxiv_title,
                "abstract": abstract,
                "tldr": "",  # arXiv 没有 TLDR
                "authors": authors,
                "author_ids": [],  # arXiv 没有作者ID
                "url": f"https://arxiv.org/abs/{arxiv_id}",
                "arxiv_url": f"https://arxiv.org/abs/{arxiv_id}",
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "year": pub_year,
                "venue": "arXiv preprint",
                "citation_count": 0,  # arXiv API 不提供引用数
                "match_score": overlap if 'overlap' in locals() else 1.0,
                "data_source": "arxiv"
            }
            print(f"[arXiv Fallback] Found: {arxiv_title}")
            print(f"  - arXiv ID: {arxiv_id}")
            print(f"  - Abstract: {len(abstract)} chars")
            print(f"  - Authors: {len(authors)}")
            print(f"  - Year: {pub_year or 'N/A'}")
            return result
        except requests.RequestException as e:
            print(f"[arXiv Fallback] Request error: {e}")
            return None
        except ET.ParseError as e:
            print(f"[arXiv Fallback] XML parse error: {e}")
            return None
        except Exception as e:
            print(f"[arXiv Fallback] Unexpected error: {e}")
            import traceback
            traceback.print_exc()
            return None