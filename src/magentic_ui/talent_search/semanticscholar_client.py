"""Lightweight Semantic Scholar client for `/paper/search` with retries.

This module is intentionally minimal so it can be reused in fallbacks where
Google Scholar results are missing.
"""
from __future__ import annotations
import re
import threading
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
import requests

BASE_URL = "https://api.semanticscholar.org/graph/v1"

@dataclass
class SemanticScholarAuthor:
    """Author snippet returned by the paper search endpoint."""

    authorId: Optional[str]
    name: str
    affiliations: Optional[List[str]] = None  # Add affiliations support

@dataclass
class SemanticScholarPaper:
    """Paper snippet returned by the paper search endpoint."""
    paperId: str
    title: str
    venue: str
    year: Optional[int]
    authors: List[SemanticScholarAuthor]
    abstract: Optional[str] = None
    externalIds: Optional[Dict] = None
    openAccessPdf: Optional[Dict] = None  # {"url": "...", "status": "..."}


class SemanticScholarSearchClient:
    """Semantic Scholar `/paper/search` client with retry and throttling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 8.0,
        max_retries: int = 3,
        requests_per_second: float = 1.0,
        user_agent: str = "TalentSearch/semantic-fallback"
    ) -> None:
        self.session = requests.Session()
        self.timeout = timeout
        self.max_retries = max_retries
        self._rps = max(0.01, requests_per_second)
        self._min_interval = 1.0 / self._rps
        self._last_call_ts = 0.0
        self._throttle_lock = threading.Lock()

        headers = {"User-Agent": user_agent}
        if api_key:
            headers["x-api-key"] = api_key
        self.session.headers.update(headers)

    def _throttle(self) -> None:
        with self._throttle_lock:
            now = time.monotonic()
            wait = self._min_interval - (now - self._last_call_ts)
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._last_call_ts = now

    def _get(self, path: str, params: Dict) -> Dict:
        url = f"{BASE_URL}{path}"
        backoff = 1.0
        last_response: Optional[requests.Response] = None

        for attempt in range(self.max_retries):
            try:
                self._throttle()
                last_response = self.session.get(url, params=params, timeout=self.timeout)
                if last_response.status_code in (429, 500, 502, 503, 504):
                    time.sleep(backoff)
                    backoff = min(backoff * 2, 8.0)
                    continue
                last_response.raise_for_status()
                return last_response.json() if last_response.content else {}
            except requests.HTTPError:
                if last_response is not None and last_response.status_code in (400, 404):
                    return {}
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(backoff)
                backoff = min(backoff * 2, 8.0)
        return {}

    @staticmethod
    def _normalize_query(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def search_papers(self, query: str, limit: int = 5, offset: int = 0) -> List[SemanticScholarPaper]:
        """Call `/paper/search` to retrieve paper candidates.
        
        Args:
            query: Search query string
            limit: Number of results per page (max 100)
            offset: Starting offset for pagination
            
        Returns:
            List of SemanticScholarPaper objects
        """
        if not query:
            return []

        params = {
            "query": self._normalize_query(query),
            "limit": min(limit, 100),  # S2 API max is 100
            "offset": offset,
            "fields": "paperId,title,authors.authorId,authors.name,venue,year,abstract,citationCount,externalIds,openAccessPdf",
        }

        payload = self._get("/paper/search", params)
        data = payload.get("data", []) if isinstance(payload, dict) else []

        papers: List[SemanticScholarPaper] = []
        for item in data:
            authors = [
                SemanticScholarAuthor(authorId=a.get("authorId"), name=a.get("name", ""))
                for a in item.get("authors", [])
            ]
            papers.append(
                SemanticScholarPaper(
                    paperId=item.get("paperId", ""),
                    title=item.get("title", ""),
                    venue=item.get("venue", ""),
                    year=item.get("year"),
                    authors=authors,
                    abstract=item.get("abstract"),
                    externalIds=item.get("externalIds"),
                    openAccessPdf=item.get("openAccessPdf"),
                )
            )
        return papers
    
    def search_papers_paginated(
        self,
        query: str,
        page_size: int = 10,
        pages: int = 3,
        start_offset: int = 0,
        year_range: Optional[str] = None,
        venue_filter: Optional[str] = None
    ) -> List[SemanticScholarPaper]:
        """
        Paginated paper search with multiple pages and venue filtering.
        
        Args:
            query: Search query string (keywords)
            page_size: Number of results per page (max 100)
            pages: Number of pages to fetch
            start_offset: Starting offset for pagination
            year_range: Optional year filter (e.g., "2023-2025")
            
        Returns:
            List of SemanticScholarPaper objects from all pages
            
        Note:
        """
        venue_names = []
        
        all_papers: List[SemanticScholarPaper] = []
        seen_ids = set()
        
        for page in range(pages):
            offset = start_offset + (page * page_size)
            # 每轮固定获取page_size篇论文
            fetch_limit = page_size
            print(f"[S2 Search] Fetching page {page + 1}/{pages} (offset={offset}, limit={fetch_limit})")
            
            # 使用S2 API的venue参数（使用第一个可能的名称）
            params = {
                "query": self._normalize_query(query),
                "limit": min(fetch_limit, 100),
                "offset": offset,
                "fields": "paperId,title,authors.authorId,authors.name,venue,year,abstract,citationCount,externalIds,openAccessPdf",
            }
            
            if year_range:
                params["year"] = year_range
            
            try:
                payload = self._get("/paper/search", params)
                data = payload.get("data", []) if isinstance(payload, dict) else []
                
                if not data:
                    print(f"[S2 Search] No more results at offset {offset}")
                    break
                
                page_papers = []
                filtered_count = 0
                for item in data:
                    paper_id = item.get("paperId", "")
                    if paper_id in seen_ids:
                        continue
                    seen_ids.add(paper_id)
                    
                    paper_venue = item.get("venue", "")
                    
                    authors = [
                        SemanticScholarAuthor(authorId=a.get("authorId"), name=a.get("name", ""))
                        for a in item.get("authors", [])
                    ]
                    page_papers.append(
                        SemanticScholarPaper(
                            paperId=paper_id,
                            title=item.get("title", ""),
                            venue=paper_venue,
                            year=item.get("year"),
                            authors=authors,
                            abstract=item.get("abstract"),
                            externalIds=item.get("externalIds"),
                            openAccessPdf=item.get("openAccessPdf"),
                        )
                    )
                
                all_papers.extend(page_papers)
                print(f"[S2 Search] Page {page + 1}: Got {len(page_papers)} papers (total: {len(all_papers)})")
                
            except Exception as e:
                print(f"[S2 Search] Error fetching page {page + 1}: {e}")
                break
        
        print(f"[S2 Search] Total: {len(all_papers)} unique papers fetched")
        return all_papers
    
    def _venue_matches(self, paper_venue: str, venue_names: List[str]) -> bool:
        """
        检查论文的venue是否匹配目标venue的任一可能名称
        使用模糊匹配：检查venue_name是否是paper_venue的子串（不区分大小写）
        
        Args:
            paper_venue: 论文实际的venue字符串
            venue_names: 目标venue的可能名称列表
        Returns:
            True if matches, False otherwise
        """
        if not paper_venue:
            return False
        
        paper_venue_lower = paper_venue.lower()
        
        for name in venue_names:
            name_lower = name.lower()
            # 精确匹配或子串匹配
            if name_lower == paper_venue_lower:
                return True
            if name_lower in paper_venue_lower:
                return True
            if paper_venue_lower in name_lower:
                return True
        
        return False
    def fetch_author_papers(
        self,
        author_id: str,
        limit: int = 1000,
        year_start: Optional[int] = None,
        year_end: Optional[int] = None
    ) -> List[SemanticScholarPaper]:
        """
        通过 Semantic Scholar authorId 获取作者的全部论文列表
        Args:
            author_id: Semantic Scholar 的 authorId
            limit: 最大获取论文数量（默认1000，不限制）
            year_start: 起始年份过滤（可选）
            year_end: 结束年份过滤（可选）
        Returns:
            List of SemanticScholarPaper objects
        """
        if not author_id:
            print("[S2 Author Papers] No author_id provided")
            return []
        # 清理 authorId（去除可能的前后空格）
        author_id = author_id.strip()
        print(f"[S2 Author Papers] Fetching papers for author: {author_id}")
        all_papers: List[SemanticScholarPaper] = []
        offset = 0
        page_size = 100  # S2 API 单次最多返回 100
        while len(all_papers) < limit:
            params = {
                "fields": "paperId,title,authors.authorId,authors.name,venue,year,abstract,citationCount,externalIds,openAccessPdf",
                "limit": min(page_size, limit - len(all_papers)),
                "offset": offset
            }
            try:
                self._throttle()
                path = f"/author/{author_id}/papers"
                payload = self._get(path, params)
                data = payload.get("data", []) if isinstance(payload, dict) else []
                if not data:
                    print(f"[S2 Author Papers] No more papers at offset {offset}")
                    break
                for item in data:
                    paper_year = item.get("year")
                    # 年份过滤
                    if year_start and paper_year and paper_year < year_start:
                        continue
                    if year_end and paper_year and paper_year > year_end:
                        continue
                    authors = [
                        SemanticScholarAuthor(authorId=a.get("authorId"), name=a.get("name", ""))
                        for a in item.get("authors", [])
                    ]
                    all_papers.append(
                        SemanticScholarPaper(
                            paperId=item.get("paperId", ""),
                            title=item.get("title", ""),
                            venue=item.get("venue", ""),
                            year=paper_year,
                            authors=authors,
                            abstract=item.get("abstract"),
                            externalIds=item.get("externalIds"),
                            openAccessPdf=item.get("openAccessPdf"),
                        )
                    )
                print(f"[S2 Author Papers] Fetched {len(data)} papers (total: {len(all_papers)})")
                # 如果返回的数据少于请求的数量，说明已经没有更多数据
                if len(data) < page_size:
                    break
                offset += page_size
            except Exception as e:
                print(f"[S2 Author Papers] Error fetching papers: {e}")
                break
        print(f"[S2 Author Papers] Total: {len(all_papers)} papers for author {author_id}")
        return all_papers
    
    def get_paper_authors_with_affiliations(self, paper_id: str) -> List[SemanticScholarAuthor]:
        """
        Get detailed author information including affiliations for a specific paper.
        Args:
            paper_id: Semantic Scholar paper ID
        Returns:
            List of SemanticScholarAuthor with affiliations populated
        """
        if not paper_id:
            print("[S2 Paper Authors] No paper_id provided")
            return []
        
        try:
            # GET /paper/{paperId}/authors?fields=name,authorId,affiliations
            path = f"/paper/{paper_id}/authors"
            params = {
                "fields": "name,authorId,affiliations"
            }
            
            payload = self._get(path, params)
            
            if not payload or "data" not in payload:
                print(f"[S2 Paper Authors] No data returned for paper {paper_id}")
                return []
            
            authors = []
            for author_data in payload.get("data", []):
                # affiliations is a list of strings
                affiliations = author_data.get("affiliations", [])
                # Filter out None/empty affiliations
                valid_affiliations = [aff for aff in (affiliations or []) if aff and aff.strip()]
                
                authors.append(
                    SemanticScholarAuthor(
                        authorId=author_data.get("authorId"),
                        name=author_data.get("name", ""),
                        affiliations=valid_affiliations if valid_affiliations else None
                    )
                )
            
            total_authors = len(authors)
            with_affiliations = sum(1 for a in authors if a.affiliations)
            print(f"[S2 Paper Authors] Got {total_authors} authors, {with_affiliations} with affiliations")
            
            return authors
            
        except Exception as e:
            print(f"[S2 Paper Authors] Error fetching authors for paper {paper_id}: {e}")
            return []
    
    def convert_papers_to_dict_list(self, papers: List[SemanticScholarPaper], self_author_id: Optional[str] = None) -> List[Dict]:
        """
        将 SemanticScholarPaper 列表转换为标准 dict 格式，便于后续处理
        Args:
            papers: SemanticScholarPaper 列表
            self_author_id: 候选人的 Semantic Scholar authorId，用于计算候选人在每篇论文中的位置
        Returns:
            List of dict，每个 dict 包含 title, authors, author_ids, year, venue, abstract, url, pdf_url 等字段
            如果提供了 self_author_id，还会包含 candidate_author_position 和 total_authors
        """
        result = []
        for paper in papers:
            # 提取 URL
            paper_url = ""
            arxiv_url = ""
            pdf_url = ""
            external_ids = paper.externalIds or {}
            arxiv_id = external_ids.get("ArXiv", "")
            if arxiv_id:
                arxiv_url = f"https://arxiv.org/abs/{arxiv_id}"
                paper_url = arxiv_url
            elif paper.paperId:
                paper_url = f"https://www.semanticscholar.org/paper/{paper.paperId}"
            # 提取 PDF URL
            if paper.openAccessPdf and isinstance(paper.openAccessPdf, dict):
                pdf_url = paper.openAccessPdf.get("url", "")
            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            # 提取作者信息
            author_names = [a.name for a in paper.authors if a.name]
            author_ids = [a.authorId for a in paper.authors]
            
            # 通过 authorId 匹配计算候选人在这篇论文中的位置（1-based）
            candidate_author_position = None
            total_authors = len(author_ids) if author_ids else None
            if self_author_id and author_ids:
                for idx, aid in enumerate(author_ids, 1):
                    if aid and aid == self_author_id:
                        candidate_author_position = idx
                        break
            
            result.append({
                "title": paper.title,
                "authors": author_names,
                "author_ids": author_ids,
                "year": paper.year,
                "venue": paper.venue or "",
                "abstract": paper.abstract or "",
                "url": paper_url,
                "arxiv_url": arxiv_url,
                "pdf_url": pdf_url,
                "paper_id": paper.paperId,
                "citation_count": external_ids.get("citationCount", 0),
                "candidate_author_position": candidate_author_position,
                "total_authors": total_authors,
                "_source": "semantic_scholar_author"
            })
        return result
