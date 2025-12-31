"""
OpenReview API Client with Rate Limiting, Caching, and Single-Flight Pattern

This module provides centralized OpenReview API access with:
1. Global rate limiting to avoid 429 errors
2. Caching for paper info and author profiles
3. Single-flight pattern to avoid duplicate concurrent requests
4. Paper-level prefetching to reduce author-level API calls
"""

import time
import threading
import requests
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from functools import lru_cache
import re
from . import config


# ============================ RATE LIMITER ============================

class OpenReviewRateLimiter:
    """
    Token bucket rate limiter for OpenReview API.
    OpenReview has two limits:
    - 20 requests per 30 seconds
    - 100 requests per 2 minutes
    We use conservative settings: 1 request per 2 seconds
    """
    
    def __init__(self, requests_per_second: float = 0.5, burst_size: int = 3):
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size
        self.tokens = burst_size
        self.last_refill = time.time()
        self.lock = threading.Lock()
        
        # Track 429 responses for adaptive backoff
        self._last_429_time: Optional[float] = None
        self._reset_time: Optional[float] = None
    
    def acquire(self, timeout: float = 60.0) -> bool:
        """
        Acquire a token to make a request.
        Returns True if acquired, False if timeout.
        """
        deadline = time.time() + timeout
        
        while time.time() < deadline:
            with self.lock:
                # Check if we need to wait for 429 reset
                if self._reset_time and time.time() < self._reset_time:
                    wait_time = self._reset_time - time.time()
                    print(f"[OpenReview Rate Limiter] Waiting {wait_time:.1f}s for 429 reset")
                    if wait_time > 0:
                        time.sleep(min(wait_time, 5.0))
                    continue
                
                # Refill tokens
                now = time.time()
                elapsed = now - self.last_refill
                new_tokens = elapsed * self.requests_per_second
                self.tokens = min(self.burst_size, self.tokens + new_tokens)
                self.last_refill = now
                
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            # Wait a bit before retry
            time.sleep(0.5)
        
        return False
    
    def report_429(self, reset_time: Optional[float] = None):
        """Report a 429 response for adaptive backoff"""
        with self.lock:
            self._last_429_time = time.time()
            if reset_time:
                self._reset_time = reset_time
            else:
                # Default: wait 60 seconds
                self._reset_time = time.time() + 60


# ============================ SINGLE-FLIGHT ============================

class SingleFlight:
    """
    Single-flight pattern: ensures only one request is in-flight for a given key.
    Other callers with the same key wait for the result.
    """
    
    def __init__(self):
        self._in_flight: Dict[str, threading.Event] = {}
        self._results: Dict[str, Any] = {}
        self._lock = threading.Lock()
    
    def do(self, key: str, fn) -> Any:
        """
        Execute fn() only once for a given key.
        Other callers with the same key wait for and share the result.
        """
        with self._lock:
            if key in self._in_flight:
                # Someone else is already fetching, wait for them
                event = self._in_flight[key]
            else:
                # We are the first, create event and proceed
                event = threading.Event()
                self._in_flight[key] = event
                event = None  # Signal that we should do the work
        
        if event is not None:
            # Wait for result
            event.wait(timeout=120)
            with self._lock:
                return self._results.get(key)
        
        # We are the leader, do the work
        try:
            result = fn()
            with self._lock:
                self._results[key] = result
        finally:
            with self._lock:
                if key in self._in_flight:
                    self._in_flight[key].set()
                    del self._in_flight[key]
        
        return result


# ============================ CACHE ============================

@dataclass
class PaperOpenReviewInfo:
    """Cached OpenReview info for a paper"""
    paper_title: str
    note_id: Optional[str] = None
    authors: List[str] = field(default_factory=list)
    authorids: List[str] = field(default_factory=list)
    venue: str = ""
    year: Optional[int] = None
    fetched_at: float = field(default_factory=time.time)
    

class OpenReviewCache:
    """Thread-safe cache for OpenReview data"""
    
    def __init__(self, ttl_seconds: float = 3600):
        self.ttl = ttl_seconds
        self._paper_cache: Dict[str, PaperOpenReviewInfo] = {}
        self._profile_cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
    
    def _normalize_title(self, title: str) -> str:
        """Normalize paper title for cache key"""
        return re.sub(r'\s+', ' ', title.lower().strip())
    
    def _normalize_name(self, name: str) -> str:
        """Normalize author name for cache key"""
        return re.sub(r'\s+', ' ', name.lower().strip())
    
    def get_paper(self, title: str) -> Optional[PaperOpenReviewInfo]:
        key = self._normalize_title(title)
        with self._lock:
            info = self._paper_cache.get(key)
            if info and (time.time() - info.fetched_at) < self.ttl:
                return info
            return None
    
    def set_paper(self, title: str, info: PaperOpenReviewInfo):
        key = self._normalize_title(title)
        with self._lock:
            self._paper_cache[key] = info
    
    def get_profile(self, name: str) -> Optional[Dict[str, Any]]:
        key = self._normalize_name(name)
        with self._lock:
            data = self._profile_cache.get(key)
            if data and (time.time() - data.get('_cached_at', 0)) < self.ttl:
                return data
            return None
    
    def set_profile(self, name: str, profile: Dict[str, Any]):
        key = self._normalize_name(name)
        with self._lock:
            profile['_cached_at'] = time.time()
            self._profile_cache[key] = profile
    
    def get_stats(self) -> Dict[str, int]:
        with self._lock:
            return {
                'paper_cache_size': len(self._paper_cache),
                'profile_cache_size': len(self._profile_cache)
            }


# ============================ CLIENT ============================

class OpenReviewClient:
    """
    Centralized OpenReview API client with rate limiting and caching.
    
    Usage:
        client = get_openreview_client()
        
        # Prefetch paper info for all papers in a batch
        client.prefetch_papers(paper_titles, authors_list)
        
        # Get cached paper info
        info = client.get_paper_info(paper_title)
        
        # Get author profile (uses cache and single-flight)
        profile = client.get_author_profile(author_name)
    """
    
    def __init__(self):
        self.rate_limiter = OpenReviewRateLimiter(
            requests_per_second=0.5,  # Conservative: 1 request per 2 seconds
            burst_size=3
        )
        self.cache = OpenReviewCache(ttl_seconds=3600)  # 1 hour TTL
        self.single_flight = SingleFlight()
        self.base_url = "https://api2.openreview.net"
        
        # Create a persistent session with connection pooling and SSL retry
        self.session = requests.Session()
        # Configure retry strategy for SSL errors
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,  # 2s, 4s, 8s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=20
        )
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)
    def _request(self, method: str, endpoint: str, params: dict = None, 
                 timeout: float = 30.0, max_retries: int = 3) -> Optional[dict]:
        """Make a rate-limited request to OpenReview API"""
        
        for attempt in range(max_retries):
            if not self.rate_limiter.acquire(timeout=30):
                print(f"[OpenReview Client] Rate limit timeout, attempt {attempt + 1}/{max_retries}")
                continue
            
            try:
                url = f"{self.base_url}/{endpoint}"
                # Use session instead of direct requests for SSL retry and connection pooling
                response = self.session.request(
                    method,
                    url,
                    params=params,
                    timeout=timeout,
                    headers={'User-Agent': config.UA.get('User-Agent', 'Mozilla/5.0')}
                )
                
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    # Parse reset time from response
                    try:
                        error_data = response.json()
                        error_msg = error_data.get('message', '')
                        # Parse resetTime from error message
                        import re
                        reset_match = re.search(r'resetTime:\s*(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+Z)', str(error_data))
                        if reset_match:
                            from datetime import datetime
                            reset_time_str = reset_match.group(1)
                            reset_dt = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))
                            reset_timestamp = reset_dt.timestamp()
                            self.rate_limiter.report_429(reset_timestamp)
                            wait_time = max(0, reset_timestamp - time.time())
                            print(f"[OpenReview Client] HTTP 429, waiting {wait_time:.1f}s until reset")
                        else:
                            self.rate_limiter.report_429()
                    except Exception as e:
                        self.rate_limiter.report_429()
                    
                    # Wait before retry
                    time.sleep(min(60, 10 * (attempt + 1)))
                else:
                    print(f"[OpenReview Client] HTTP {response.status_code}: {endpoint}")
                    
            except (requests.exceptions.ConnectionError, 
                    requests.exceptions.SSLError, 
                    ConnectionResetError) as e:
                if attempt < max_retries - 1:
                    retry_delay = 3 * (attempt + 1)
                    error_type = "SSL error" if "SSL" in str(e) else "Connection reset"
                    print(f"[OpenReview Client] {error_type} (attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                else:
                    error_type = "SSL/Connection"
                    print(f"[OpenReview Client] {error_type} failed after {max_retries} attempts: {str(e)}")
                    
            except requests.RequestException as e:
                if attempt < max_retries - 1:
                    print(f"[OpenReview Client] Request error (attempt {attempt + 1}/{max_retries}): {str(e)}")
                    time.sleep(2 * (attempt + 1))
                else:
                    print(f"[OpenReview Client] Request failed after {max_retries} attempts: {str(e)}")
        
        return None
    
    def get_paper_note_by_title(self, paper_title: str, authors: List[str]) -> Optional[PaperOpenReviewInfo]:
        """
        Get paper info from OpenReview by title and authors.
        Uses cache and single-flight pattern.
        """
        # Check cache first
        cached = self.cache.get_paper(paper_title)
        if cached:
            print(f"[OpenReview Client] Cache hit for paper: {paper_title[:50]}...")
            return cached
        
        # Use single-flight to avoid duplicate requests
        cache_key = f"paper:{self.cache._normalize_title(paper_title)}"
        
        def fetch():
            return self._fetch_paper_note(paper_title, authors)
        
        result = self.single_flight.do(cache_key, fetch)
        return result
    
    def _fetch_paper_note(self, paper_title: str, authors: List[str]) -> Optional[PaperOpenReviewInfo]:
        """Internal: fetch paper note from API"""
        if not authors:
            return None
        
        # Generate paperhash
        paperhash = self._generate_paperhash(paper_title, authors)
        if not paperhash:
            return None
        
        # Query OpenReview
        response = self._request('GET', 'notes', {
            'paperhash': paperhash,
            'limit': 5
        })
        
        if not response or not response.get('notes'):
            return None
        
        note = response['notes'][0]
        content = note.get('content', {})
        
        # Extract data
        def get_value(data, default=''):
            if isinstance(data, dict):
                return data.get('value', default)
            return data if data is not None else default
        
        info = PaperOpenReviewInfo(
            paper_title=paper_title,
            note_id=note.get('id'),
            authors=get_value(content.get('authors'), []) or [],
            authorids=get_value(content.get('authorids'), []) or [],
            venue=get_value(content.get('venue'), ''),
            year=note.get('tcdate')  # Could extract year from tcdate
        )
        
        # Cache the result
        self.cache.set_paper(paper_title, info)
        print(f"[OpenReview Client] Fetched and cached paper: {paper_title[:50]}... ({len(info.authorids)} authors)")
        
        return info
    
    def _generate_paperhash(self, title: str, authors: List[str]) -> Optional[str]:
        """
        Generate OpenReview paperhash from title and authors.
        注意：必须安装 openreview 库才能使用此功能。
        安装命令：pip install openreview-py
        Args:
            title: 论文标题
            authors: 作者列表
        Returns:
            paperhash 字符串，如果失败返回 None
        Raises:
            ImportError: 如果未安装 openreview 库
        """
        if not authors or len(authors) == 0:
            print("[OpenReview Client] 作者列表为空，无法生成 paperhash")
            return None
        try:
            from openreview import tools as or_tools
        except ImportError:
            print(
                "[OpenReview Client] openreview library not installed; skipping paperhash."
            )
            return None
        try:
            first_author = authors[0] if authors else ""
            return or_tools.get_paperhash(first_author=first_author, title=title)
        except Exception as e:
            print(f"[OpenReview Client] Paperhash generation failed: {e}")
            return None
    
    def prefetch_papers(self, papers: List[Dict[str, Any]], max_concurrent: int = 2):
        """
        Prefetch OpenReview info for multiple papers.
        This should be called ONCE per search round, before processing authors.
        
        Args:
            papers: List of paper dicts with 'title' and 'authors' keys
            max_concurrent: Max concurrent fetches (low to avoid 429)
        """
        print(f"[OpenReview Client] Prefetching {len(papers)} papers...")
        
        # Filter out already cached papers
        to_fetch = []
        for paper in papers:
            title = paper.get('title', '')
            if title and not self.cache.get_paper(title):
                to_fetch.append(paper)
        
        if not to_fetch:
            print(f"[OpenReview Client] All {len(papers)} papers already cached")
            return
        
        print(f"[OpenReview Client] {len(to_fetch)} papers need fetching")
        
        # Fetch sequentially with rate limiting (safer than concurrent)
        for i, paper in enumerate(to_fetch):
            title = paper.get('title', '')
            authors = paper.get('authors', [])
            
            if not title or not authors:
                continue
            
            try:
                self.get_paper_note_by_title(title, authors)
            except Exception as e:
                print(f"[OpenReview Client] Failed to prefetch paper {i+1}/{len(to_fetch)}: {e}")
            
            # Progress log every 5 papers
            if (i + 1) % 5 == 0:
                print(f"[OpenReview Client] Prefetch progress: {i+1}/{len(to_fetch)}")
        
        stats = self.cache.get_stats()
        print(f"[OpenReview Client] Prefetch complete. Cache stats: {stats}")
    
    def get_author_profile(self, author_name: str) -> Optional[Dict[str, Any]]:
        """
        Get author profile from OpenReview.
        Uses cache and single-flight pattern.
        """
        # Check cache first
        cached = self.cache.get_profile(author_name)
        if cached:
            print(f"[OpenReview Client] Cache hit for profile: {author_name}")
            return cached
        
        # Use single-flight
        cache_key = f"profile:{self.cache._normalize_name(author_name)}"
        
        def fetch():
            return self._fetch_author_profile(author_name)
        
        result = self.single_flight.do(cache_key, fetch)
        return result
    
    def _fetch_author_profile(self, author_name: str) -> Optional[Dict[str, Any]]:
        """Internal: fetch author profile from API"""
        # Search by name
        response = self._request('GET', 'profiles/search', {
            'fullname': author_name,
            'limit': 5
        })
        
        if not response or not response.get('profiles'):
            return None
        
        profile = response['profiles'][0]
        profile_id = profile.get('id', '')
        
        # Validate profile ID
        if not profile_id or not profile_id.startswith('~'):
            return None
        
        # Fetch full profile
        full_response = self._request('GET', 'profiles', {
            'id': profile_id
        })
        
        if not full_response or not full_response.get('profiles'):
            return None
        
        full_profile = full_response['profiles'][0]
        
        # Build result
        result = {
            'openreview_id': profile_id,
            'openreview_url': f"https://openreview.net/profile?id={profile_id}",
            'names': full_profile.get('content', {}).get('names', []),
            'emails': full_profile.get('content', {}).get('emailsConfirmed', []),
            'education': full_profile.get('content', {}).get('history', []),
            'personal_links': {},
            'publications': []
        }
        
        # Extract links
        content = full_profile.get('content', {})
        if content.get('homepage'):
            result['personal_links']['homepage'] = content['homepage']
        if content.get('gscholar'):
            result['personal_links']['google_scholar'] = content['gscholar']
        if content.get('linkedin'):
            result['personal_links']['linkedin'] = content['linkedin']
        if content.get('orcid'):
            result['personal_links']['orcid'] = content['orcid']
        
        # Cache the result
        self.cache.set_profile(author_name, result)
        print(f"[OpenReview Client] Fetched and cached profile: {author_name}")
        
        return result
    
    def get_author_id_from_paper(self, paper_title: str, authors: List[str], 
                                  target_author: str) -> Optional[str]:
        """
        Get OpenReview author ID for a specific author from a paper.
        Uses cached paper info if available.
        
        Args:
            paper_title: The paper title
            authors: List of all authors
            target_author: The author we want the ID for
            
        Returns:
            OpenReview author ID (starting with ~) or None
        """
        paper_info = self.get_paper_note_by_title(paper_title, authors)
        if not paper_info:
            return None
        
        # Find target author in the authorids list
        target_lower = target_author.lower().strip()
        
        for i, author_name in enumerate(paper_info.authors):
            if i >= len(paper_info.authorids):
                break
            
            if author_name.lower().strip() == target_lower:
                author_id = paper_info.authorids[i]
                # Validate it's a proper OpenReview ID
                if author_id and isinstance(author_id, str) and author_id.startswith('~'):
                    return author_id
        
        return None


# ============================ SINGLETON ============================

_client_instance: Optional[OpenReviewClient] = None
_client_lock = threading.Lock()


def get_openreview_client() -> OpenReviewClient:
    """Get the global OpenReview client instance"""
    global _client_instance
    
    if _client_instance is None:
        with _client_lock:
            if _client_instance is None:
                _client_instance = OpenReviewClient()
    
    return _client_instance


def reset_openreview_client():
    """Reset the global client (for testing)"""
    global _client_instance
    with _client_lock:
        _client_instance = None
