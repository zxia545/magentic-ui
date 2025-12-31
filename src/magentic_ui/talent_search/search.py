"""
Search functionality for Talent Search System
Handles SearXNG searches, URL fetching, and content extraction
"""
import re, json, io
import asyncio
from typing import Optional, Tuple, List, Dict, Any, Union
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode, quote_plus

import requests
from bs4 import BeautifulSoup
from . import config
from .utils import normalize_url, domain_of, safe_sleep, clean_text, looks_like_profile_url
import os
from magentic_ui.tools.bing_search import get_bing_search_results
from magentic_ui.tools.playwright import PlaywrightBrowser, PlaywrightController

try:
    # 可选：若未安装 readability-lxml，会自动回退
    from readability import Document  # type: ignore
    HAS_READABILITY = True
except Exception:
    HAS_READABILITY = False

# ============================ BING SEARCH FUNCTIONS ============================


def _run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


_BROWSER_RESOURCE_CONFIG: Any | None = None
_WEB_AGENT_LOOP: asyncio.AbstractEventLoop | None = None
_WEB_AGENT_BROWSER: PlaywrightBrowser | None = None
_WEB_AGENT_CONTEXT: Any | None = None


def set_browser_resource_config(resource: Any | None) -> None:
    global _BROWSER_RESOURCE_CONFIG
    _BROWSER_RESOURCE_CONFIG = resource


def _run_web_agent(coro):
    global _WEB_AGENT_LOOP
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        if _WEB_AGENT_LOOP is None or _WEB_AGENT_LOOP.is_closed():
            _WEB_AGENT_LOOP = asyncio.new_event_loop()
            asyncio.set_event_loop(_WEB_AGENT_LOOP)
        return _WEB_AGENT_LOOP.run_until_complete(coro)
    return asyncio.run_coroutine_threadsafe(coro, loop).result()


async def _ensure_web_agent_context() -> Any | None:
    global _WEB_AGENT_BROWSER, _WEB_AGENT_CONTEXT
    if _WEB_AGENT_CONTEXT is not None:
        return _WEB_AGENT_CONTEXT
    if _BROWSER_RESOURCE_CONFIG is None:
        return None
    try:
        browser = PlaywrightBrowser.load_component(_BROWSER_RESOURCE_CONFIG)
        await browser.__aenter__()
        _WEB_AGENT_BROWSER = browser
        _WEB_AGENT_CONTEXT = browser.browser_context
        return _WEB_AGENT_CONTEXT
    except Exception as exc:
        if config.VERBOSE:
            print(f"[web_agent_search] failed to start browser: {exc!r}")
        return None


def _parse_bing_html_from_page(html: str, k: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    url_set = set()
    for li in soup.select("li.b_algo"):
        a = li.find("a", href=True)
        if not a:
            continue
        url = a["href"].strip()
        if not url.startswith("http") or url in url_set:
            continue
        title = ""
        h2 = li.find("h2")
        if h2:
            title = h2.get_text(" ", strip=True)
        if not title:
            title = a.get_text(" ", strip=True)
        snippet = ""
        p = li.find("p")
        if p:
            snippet = p.get_text(" ", strip=True)
        url_set.add(url)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "engine": "bing_web_agent",
                "authors": [],
            }
        )
        if len(results) >= k:
            break
    return results


def _parse_scholar_html_from_page(html: str, k: int) -> List[Dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: List[Dict[str, str]] = []
    url_set = set()
    for entry in soup.select("div.gs_ri"):
        h3 = entry.select_one("h3.gs_rt")
        if not h3:
            continue
        link = h3.find("a", href=True)
        if not link:
            continue
        url = link["href"].strip()
        if not url.startswith("http") or url in url_set:
            continue
        title = h3.get_text(" ", strip=True)
        snippet = ""
        abstract = entry.select_one("div.gs_rs")
        if abstract:
            snippet = abstract.get_text(" ", strip=True)
        url_set.add(url)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "engine": "scholar_web_agent",
                "authors": [],
            }
        )
        if len(results) >= k:
            break
    return results


def _parse_markdown_links(markdown_text: str, k: int, engine: str) -> List[Dict[str, str]]:
    results: List[Dict[str, str]] = []
    url_set = set()
    for line in markdown_text.split("\n"):
        if (
            line.count("[") == 1
            and line.count("]") == 1
            and line.count("(") == 1
            and line.count(")") == 1
        ):
            display_start = line.find("[") + 1
            display_end = line.find("]")
            url_start = line.find("(") + 1
            url_end = line.find(")")
            display_text = line[display_start:display_end].strip()
            url = line[url_start:url_end].strip()
            if not url.startswith("http") or url in url_set:
                continue
            url_set.add(url)
            results.append(
                {
                    "title": display_text,
                    "url": url,
                    "snippet": "",
                    "engine": engine,
                    "authors": [],
                }
            )
            if len(results) >= k:
                break
    return results


async def _web_agent_search_async(query: str, k: int) -> List[Dict[str, str]]:
    context = await _ensure_web_agent_context()
    if context is None:
        return []
    controller = PlaywrightController()
    engines = config.WEB_AGENT_SEARCH_ENGINES or ["bing"]
    for engine in engines:
        if engine.lower() == "scholar":
            search_url = f"https://scholar.google.com/scholar?q={quote_plus(query)}"
            parser = _parse_scholar_html_from_page
            engine_name = "scholar_web_agent"
            ready_selector = "div.gs_ri"
        else:
            search_url = f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
            parser = _parse_bing_html_from_page
            engine_name = "bing_web_agent"
            ready_selector = "li.b_algo"
        page = await context.new_page()
        try:
            page.set_default_timeout(config.WEB_AGENT_TIMEOUT_SEC * 1000)
            await page.goto(search_url, wait_until="domcontentloaded")
            try:
                await page.wait_for_selector(ready_selector, timeout=5000)
            except Exception:
                pass
            await asyncio.sleep(1)
            html = await page.content()
            results = parser(html, k)
            if results:
                return results
            markdown = ""
            try:
                markdown = await controller.get_page_markdown(
                    page, max_tokens=config.WEB_AGENT_MAX_TOKENS
                )
            except Exception as exc:
                if config.VERBOSE:
                    print(f"[web_agent_search] markdown error: {exc!r} for query: {query}")
            if markdown:
                results = _parse_markdown_links(markdown, k, engine_name)
                if results:
                    return results
        except Exception as exc:
            if config.VERBOSE:
                print(f"[web_agent_search] error: {exc!r} for query: {query}")
        finally:
            await page.close()
    return []


def searxng_search(
    query: str,
    engines: List[str] = config.SEARXNG_ENGINES,
    pages: int = config.SEARXNG_PAGES,
    k_per_query: int = config.SEARCH_K,
) -> List[Dict[str, str]]:
    """Search via Magentic-UI Bing tool (compat shim for legacy searxng_search)."""
    if config.PAPER_SEARCH_BACKEND.lower() == "web_agent":
        results = _run_web_agent(_web_agent_search_async(query, k_per_query))
        if results:
            return results
        if config.VERBOSE:
            print(f"[web_agent_search] no results for query: {query}, fallback to HTML")
        return _bing_html_search(query, k_per_query)
    results = None
    allow_playwright = getattr(config, "BING_USE_PLAYWRIGHT", True)
    try:
        if hasattr(os, "geteuid") and os.geteuid() == 0:
            allow_playwright = False
    except Exception:
        pass
    if os.getenv("MAGENTIC_UI_BING_USE_PLAYWRIGHT") in {"0", "false", "False"}:
        allow_playwright = False
    if allow_playwright:
        try:
            results = _run_async(
                get_bing_search_results(
                    query,
                    max_pages=max(1, min(pages, 3)),
                    timeout_seconds=12,
                )
            )
        except Exception as exc:
            if config.VERBOSE:
                print(f"[bing_search] error: {exc!r} for query: {query}")
    else:
        if config.VERBOSE:
            print(f"[bing_search] skipping Playwright for query: {query}")

    out: List[Dict[str, str]] = []
    if results and getattr(results, "links", None):
        url_set = set()
        for link in results.links[:k_per_query]:
            url = link.get("url", "")
            if not url.startswith("http") or url in url_set:
                continue
            url_set.add(url)
            out.append(
                {
                    "title": (link.get("display_text") or "").strip(),
                    "url": url,
                    "snippet": "",
                    "engine": "bing",
                    "authors": [],
                }
            )
    if out:
        return out
    if config.VERBOSE:
        print(f"[bing_search] fallback to HTML scrape for query: {query}")
    return _bing_html_search(query, k_per_query)


def _bing_html_search(query: str, k: int = 10) -> List[Dict[str, str]]:
    url = f"https://www.bing.com/search?q={quote_plus(query)}&FORM=QBLH"
    headers = dict(config.UA or {})
    headers.setdefault("Accept", "text/html,application/xhtml+xml")
    headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    try:
        resp = requests.get(url, headers=headers, timeout=12)
    except Exception as exc:
        if config.VERBOSE:
            print(f"[bing_search] HTML fallback error: {exc!r} for query: {query}")
        return []
    if resp.status_code != 200:
        if config.VERBOSE:
            print(f"[bing_search] HTML fallback status {resp.status_code} for query: {query}")
        return []
    soup = BeautifulSoup(resp.text, "html.parser")
    results: List[Dict[str, str]] = []
    url_set = set()
    for li in soup.select("li.b_algo"):
        a = li.find("a", href=True)
        if not a:
            continue
        url = a["href"].strip()
        if not url.startswith("http") or url in url_set:
            continue
        title = ""
        h2 = li.find("h2")
        if h2:
            title = h2.get_text(" ", strip=True)
        if not title:
            title = a.get_text(" ", strip=True)
        snippet = ""
        p = li.find("p")
        if p:
            snippet = p.get_text(" ", strip=True)
        url_set.add(url)
        results.append(
            {
                "title": title,
                "url": url,
                "snippet": snippet,
                "engine": "bing_html",
                "authors": [],
            }
        )
        if len(results) >= k:
            break
    return results

# ============================ CONTENT FETCHING FUNCTIONS ============================


# ---- 通用：将 engines 既支持 str 也支持 list/tuple（修复你代码里传 ["bing"] 的用法）----
def _normalize_engines(engines: Union[str, List[str], Tuple[str, ...]]) -> str:
    if isinstance(engines, (list, tuple)):
        return ",".join(engines)
    return engines

# ---- URL 规范化：去除追踪参数，保留结构一致性 ----
_TRACKING_KEYS = {"utm_source","utm_medium","utm_campaign","utm_term","utm_content",
                  "gclid","fbclid","mc_cid","mc_eid","oly_anon_id","oly_enc_id"}

def canonicalize_url(u: str) -> str:
    try:
        p = urlparse(u)
        q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=True) if k.lower() not in _TRACKING_KEYS]
        p2 = p._replace(query=urlencode(q, doseq=True))
        p2 = p2._replace(fragment="")
        return urlunparse(p2)
    except Exception:
        return u

# ---- HTTP 获取：带重试、合理头、编码处理 ----
def _http_get(url: str, timeout: int = 15) -> requests.Response:
    sess = requests.Session()
    # 比默认更像浏览器，提升可达性
    headers = dict(config.UA or {})
    headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
    headers.setdefault("Accept-Language", "en-US,en;q=0.9")
    headers.setdefault("Cache-Control", "no-cache")
    r = sess.get(url, headers=headers, timeout=timeout, allow_redirects=True)
    # 让 requests 自动以 apparent_encoding 回填（对 text/html 有帮助）
    if not r.encoding or r.encoding.lower() == "iso-8859-1":
        r.encoding = r.apparent_encoding or r.encoding
    return r

# ---- JSON-LD 标题提取（优先级最高，常见于新闻/学术/博客）----
def _title_from_jsonld(soup: BeautifulSoup) -> Optional[str]:
    for tag in soup.find_all("script", attrs={"type": "application/ld+json"}):
        try:
            data = tag.string or tag.get_text() or ""
            if not data.strip():
                continue
            obj = json.loads(data)
            items = obj if isinstance(obj, list) else [obj]
            for it in items:
                if not isinstance(it, dict):
                    continue
                # 常见类型
                typ = it.get("@type")
                if isinstance(typ, list):
                    types = [t.lower() for t in typ if isinstance(t, str)]
                else:
                    types = [typ.lower()] if isinstance(typ, str) else []
                if any(t in ("article","newsarticle","blogposting","webpage","scholarlyarticle","report") for t in types):
                    t1 = it.get("headline") or it.get("name") or it.get("alternativeHeadline")
                    if isinstance(t1, str) and t1.strip():
                        return t1.strip()
                # 某些站点把主体包到 mainEntity
                main = it.get("mainEntity")
                if isinstance(main, dict):
                    t2 = main.get("headline") or main.get("name")
                    if isinstance(t2, str) and t2.strip():
                        return t2.strip()
        except Exception:
            continue
    return None

# ---- Meta 标题：OpenGraph / Twitter ----
def _title_from_meta(soup: BeautifulSoup) -> Optional[str]:
    # og:title
    og = soup.find("meta", attrs={"property": "og:title"}) or soup.find("meta", attrs={"name": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()
    # twitter:title
    tw = soup.find("meta", attrs={"name": "twitter:title"})
    if tw and tw.get("content"):
        return tw["content"].strip()
    # dc.title
    dc = soup.find("meta", attrs={"name": "dc.title"})
    if dc and dc.get("content"):
        return dc["content"].strip()
    return None

# ---- 可见 <h1> 回退（过滤导航/登录等噪声）----
_NAV_WORDS = {"menu","navigation","nav","search","login","sign","home","about","contact","subscribe","cookie"}

def _title_from_headings(soup: BeautifulSoup) -> Optional[str]:
    # 优先找“像文章标题”的 h1
    for h in soup.find_all("h1"):
        txt = h.get_text(" ", strip=True)
        if 10 <= len(txt) <= 200 and not any(w in txt.lower() for w in _NAV_WORDS):
            return txt
    # 再尝试 h2（有些站标题在 h2）
    for h in soup.find_all("h2")[:3]:
        txt = h.get_text(" ", strip=True)
        if 10 <= len(txt) <= 200:
            return txt
    return None

# ---- <title> 标签兜底 ----
def _title_from_title_tag(soup: BeautifulSoup) -> Optional[str]:
    if soup.title and soup.title.string:
        t = soup.title.string.strip()
        # 去掉网站名常用分隔
        t = re.split(r"\s+[|-]\s+|\s+·\s+|\s+–\s+", t)[0].strip() or t
        return t
    return None

def extract_title_unified(html_doc: str) -> str:
    soup = BeautifulSoup(html_doc, "html.parser")
    for fn in (_title_from_jsonld, _title_from_meta, _title_from_headings, _title_from_title_tag):
        t = fn(soup)
        if t:
            return t
    return ""

# ---- 正文抽取：trafilatura → readability → 轻量回退 ----
def extract_main_text(html_doc: str, base_url: Optional[str] = None) -> str:
    from trafilatura import extract as t_extract
    # 先尝试多种提取方式，但不提前返回；最后进行合并去重，确保覆盖完整
    extracted_parts: List[str] = []
    try:
        # favor_recall=True 能从结构复杂页多拿点正文；不需要注释/表格
        t_text = t_extract(html_doc, include_comments=False, favor_recall=True, url=base_url) or ""
        if t_text.strip():
            extracted_parts.extend([s for s in t_text.split("\n\n") if s.strip()])
    except Exception:
        pass

    if HAS_READABILITY:
        try:
            doc = Document(html_doc)
            summary_html = doc.summary(html_partial=True)
            r_text = BeautifulSoup(summary_html, "html.parser").get_text("\n", strip=True)
            if r_text.strip():
                extracted_parts.extend([s for s in r_text.split("\n\n") if s.strip()])
        except Exception:
            pass

    # 结构化回退：遍历文档重要标签，过滤导航/页脚，尽可能保留各 section 的内容
    soup = BeautifulSoup(html_doc, "html.parser")

    # 移除明显无关的标签
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    def is_noise(node) -> bool:
        # 检查自身及祖先是否属于噪声区域
        noise_names = {"nav", "header", "footer", "aside"}
        noise_keys = [
            "nav", "menu", "breadcrumb", "sidebar", "footer", "cookie", "subscribe",
            "advert", "ads", "toc", "pagination"
        ]
        cur = node
        while cur and getattr(cur, "name", None) not in (None, "body"):
            name = getattr(cur, "name", "") or ""
            if name in noise_names:
                return True
            cls = " ".join(cur.get("class", [])).lower()
            idv = (cur.get("id", "") or "").lower()
            if any(k in cls for k in noise_keys) or any(k in idv for k in noise_keys):
                return True
            cur = cur.parent
        return False

    # 选择重要标签：标题、段落、列表项、定义列表、引用、表格单元、span（用于新闻条目）、a（带较长文本的链接）
    tag_order = ["h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "dt", "dd", "blockquote", "td", "th", "span", "a"]
    parts: List[str] = []

    # 文档标题
    if soup.title and soup.title.string:
        title_text = soup.title.string.strip()
        if title_text:
            parts.append(title_text)

    # 顺序遍历并采集可见文本
    for tag in soup.find_all(tag_order):
        if is_noise(tag):
            continue
        txt = tag.get_text(" ", strip=True)
        if not txt:
            continue
        # 简单过滤：跳过过短/无信息片段
        if len(txt) < 2:
            continue
        # 列表项前缀
        if tag.name == "li" and not txt.startswith("-"):
            txt = f"- {txt}"
        parts.append(txt)

    # 去重并保序
    seen = set()
    uniq_parts: List[str] = []
    for s in parts:
        if s not in seen:
            seen.add(s)
            uniq_parts.append(s)

    # 合并不同提取器与结构化内容，做最终去重
    combined = []
    seen = set()
    for chunk in extracted_parts + uniq_parts:
        c = chunk.strip()
        if not c:
            continue
        if c in seen:
            continue
        seen.add(c)
        combined.append(c)

    text = "\n\n".join(combined).strip()
    return text or "[Empty after parse]"

# ---- 受限/动态站点识别（不给你突破登录，只做优雅退化）----
_BLOCK_HINTS = (
    "please enable javascript", "sign in", "log in", "subscribe", "are you a robot",
    "access denied", "verify you are human", "captcha"
)

def looks_likely_blocked(text: str) -> bool:
    t = text.lower()
    return any(k in t for k in _BLOCK_HINTS) or len(text.strip()) < 300

def _pick_snippet_for_url(url: str, snippet: str = "", prefer_same_domain: bool = True) -> str:
    """
    返回用于该 URL 的 snippet（优先使用调用方传入的 snippet；否则用 SearXNG 搜索结果里的一段预览文字）。
    - snippet（参数）通常来自你在 SERP 阶段拿到的 result["content"]。
    - 若未传入，则对 URL 本身做一次搜索，取同域命中或首条结果的 content 作为兜底。
    """
    if snippet:
        return snippet.strip()
    try:
        engines = config.SEARXNG_ENGINES_SNIPPET
        rows = searxng_search(url, engines=engines, pages=1, k_per_query=3) or []
        if not rows:
            return ""
        dom = domain_of(url)
        if prefer_same_domain:
            for it in rows:
                u = normalize_url(it.get("url") or "")
                if domain_of(u) == dom:
                    return (it.get("snippet") or "").strip()
        return (rows[0].get("snippet") or "").strip()
    except Exception:
        return ""


def fetch_text(url: str, max_chars: int = config.FETCH_MAX_CHARS, snippet: str = "") -> str:
    """
    Fetch & extract 主内容（HTML/PDF）。
    统一输出结构，**始终把 SNIPPET 放在最前**：
        SNIPPET: <来自搜索引擎的可见预览文字>
        TITLE:   <页面标题/推断标题>
        BODY:    <抽取到的正文，可能为空；若为受限/被拦截站点，此段可能缺失或极短>
        SOURCE:  <原始 URL>

    说明：
    - SNIPPET = search engine 结果页对该链接的简短预览文本（通常是标题+摘要片段），
      代表“用户在不点开网页时最能看到/认到的信息”。我们用它在被 403/登录墙 时兜底。
    """

    url_l = url.lower()
    
    # ---------- 0) arXiv 论文优化：跳过 snippet 获取（节省时间） ----------
    # arXiv 论文几乎总能成功下载完整内容，不需要 snippet 兜底
    is_arxiv = "arxiv.org/abs/" in url_l or "arxiv.org/pdf/" in url_l
    if is_arxiv:
        snippet = ""  # 强制清空，避免触发 _pick_snippet_for_url

    # ---------- 1) 明确受限域：ResearchGate / X(Twitter) 等，直接走 snippet 预览 ----------
    if "researchgate.net/publication/" in url_l:
        # 只做“可公开识别”的摘要拼装：从 slug 推断标题 + snippet 置顶 + 源地址
        try:
            slug = url.split("/publication/")[1].split("/")[0]
        except Exception:
            slug = url.split("/publication/")[-1]
        # 从 slug 推断一个人类可读标题
        guessed_title = slug.replace("_", " ").strip()
        sn = _pick_snippet_for_url(url, snippet)

        parts = []
        if sn:
            parts.append(f"SNIPPET: {sn}")
        if guessed_title:
            parts.append(f"TITLE: {guessed_title}")
        parts.append(f"SOURCE: {url}")
        return clean_text("\n\n".join(parts), max_chars)

    if "x.com/" in url_l or "twitter.com/" in url_l or "researchgate.net/" in url_l or "scholar.google.com" in url_l:
        # 其它 RG/X 页面：同样不抓正文，直接走 snippet 兜底
        sn = _pick_snippet_for_url(url, snippet)
        parts = []
        if sn:
            parts.append(f"SNIPPET: {sn}")
        parts.append(f"SOURCE: {url}")
        return clean_text("\n\n".join(parts), max_chars)

    # ---------- 2) Scholar citations 页面直接跳过 ----------
    if "scholar.google.com/citations" in url_l:
        return "[Skip] Google Scholar citations page (JS-heavy)"

    # ---------- 3) 常规抓取（HTML/PDF），若 40x/429 则回退 snippet ----------
    try:
        r = requests.get(url, timeout=5, headers=config.UA)
        if not r.ok:
            # 403/401/429 等都走 snippet 兜底（但 arXiv 几乎不会失败）
            parts = []
            if not is_arxiv:
                sn = _pick_snippet_for_url(url, snippet)
                if sn:
                    parts.append(f"SNIPPET: {sn}")
            parts.append(f"[FetchError] HTTP {r.status_code} for {url}")
            parts.append(f"SOURCE: {url}")
            return clean_text("\n\n".join(parts), max_chars)

        ct = (r.headers.get("content-type") or "").lower()
        is_pdf = ("application/pdf" in ct) or url_l.endswith(".pdf")

        if is_pdf:
            try:
                from pdfminer.high_level import extract_text as pdf_extract
                text = pdf_extract(io.BytesIO(r.content)) or ""
                
                # Only get snippet for non-arXiv URLs (arXiv has full content)
                parts = []
                if not is_arxiv:
                    sn = _pick_snippet_for_url(url, snippet)
                    if sn:
                        parts.append(f"SNIPPET: {sn}")
                
                if text.strip():
                    parts.append("TITLE: (from PDF)")
                    parts.append("BODY:\n" + text.strip())
                parts.append(f"SOURCE: {url}")
                return clean_text("\n\n".join(parts), max_chars)
            except Exception as e:
                # Only get snippet for non-arXiv URLs (arXiv has full content)
                parts = []
                if not is_arxiv:
                    sn = _pick_snippet_for_url(url, snippet)
                    if sn:
                        parts.append(f"SNIPPET: {sn}")
                parts.append(f"[Skip] PDF extract failed: {e!r}")
                parts.append(f"SOURCE: {url}")
                return clean_text("\n\n".join(parts), max_chars)

        # HTML
        if ("text/html" not in ct) and ("application/xhtml" not in ct):
            parts = []
            if not is_arxiv:
                sn = _pick_snippet_for_url(url, snippet)
                if sn:
                    parts.append(f"SNIPPET: {sn}")
            parts.append(f"[Skip] Content-Type not HTML/PDF: {ct}")
            parts.append(f"SOURCE: {url}")
            return clean_text("\n\n".join(parts), max_chars)

        html_doc = r.text
        title = extract_title_unified(html_doc)  # 使用统一的标题提取函数
        body  = extract(html_doc) or ""  # trafilatura 主体抽取

        if not body.strip():
            # 轻量回退：取 <title> 与 h1/h2
            soup = BeautifulSoup(html_doc, "html.parser")
            heads = []
            if soup.title and soup.title.string:
                heads.append(soup.title.string.strip())
            for h in soup.find_all(["h1", "h2"])[:2]:
                heads.append(h.get_text(" ", strip=True))
            body = "\n".join(heads) or "[Empty after parse]"

        # ---- 统一拼装，**SNIPPET 始终放最前** ----
        parts = []
        if not is_arxiv:
            sn = _pick_snippet_for_url(url, snippet)
            if sn:
                parts.append(f"SNIPPET: {sn}")
        if title:
            parts.append(f"TITLE: {title}")
        parts.append("BODY:\n" + body.strip())
        parts.append(f"SOURCE: {url}")

        return clean_text("\n\n".join(parts), max_chars)

    except Exception as e:
        parts = []
        if not is_arxiv:
            sn = _pick_snippet_for_url(url, snippet)
            if sn:
                parts.append(f"SNIPPET: {sn}")
        parts.append(f"[FetchError] {e!r}")
        parts.append(f"SOURCE: {url}")
        return clean_text("\n\n".join(parts), max_chars)


# ============ MULTI-DIMENSIONAL SCORING PROMPTS ============
PROMPT_PROBLEM_OBJECTIVE = """You are an expert meta-reviewer. Judge ONLY the paper's Problem & Objective Alignment to the user's query. Anchor every judgment to the query. If information is missing, say "Insufficient information" and score accordingly. Output ONLY the JSON object described below—no extra text.
DIMENSION DESCRIPTION:
Assess the high-level match between the paper's stated research problem and objectives and the user's query. Focus on whether the paper aims at the same core questions and outcomes that the query targets.

Scoring rubric (1–8):
1 = Completely unrelated (objectives do not address the query at all).
2 = Barely related (superficial mention; accidental overlap).
3 = Weakly related (tangential objectives; limited bearing).
4 = Somewhat related (peripheral alignment; not central).
5 = Moderately related (meaningful overlap but not the main objective).
6 = Clearly relevant (core objectives align with the query).
7 = Highly relevant (objectives are designed to address the query).
8 = Perfect match (objectives directly and centrally target the query).

Output JSON schema:
{
  "dimension": "Problem & Objective Alignment",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining alignment/misalignment to the query.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_CONCEPTUAL_METHOD = """You are an expert meta-reviewer. Judge ONLY the paper's Conceptual & Methodological Match to the user's query (core theories/constructs/definitions AND task/method/setup). Do NOT rate scientific rigor; focus on FIT to the query. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Compare the paper's key concepts and its chosen methods (task formulation, algorithms, procedures, evaluation style) with what the query implies or requires. Look for overlap in constructs AND suitability of methods for the query.

Scoring rubric (1–8):
1 = Concepts and methods unrelated to the query.
2 = Barely related; superficial term/procedure overlap.
3 = Weak; tangential concepts or methods with limited applicability.
4 = Some; peripheral linkage; not directly addressing the query.
5 = Moderate; meaningful overlaps but not central or only one of the two (concepts/methods) fits.
6 = Clear; both conceptual core and methods are applicable to the query.
7 = High; strong integration of concepts and methods tailored to the query.
8 = Perfect; conceptual core and methodology are purpose-built for the query.

Output JSON schema:
{
  "dimension": "Conceptual & Methodological Match",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences covering both conceptual and methodological fit to the query.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_LITERATURE_COVERAGE = """You are an expert meta-reviewer. Judge ONLY the paper's Literature & Field Coverage relevant to the user's query (coverage of key works and correct positioning in the right subfield). Do not judge citation style quality; focus on query-relevant coverage/positioning. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Check whether the related work adequately covers the important query-relevant papers and correctly positions the study within the appropriate subfield/domain implied by the query.

Scoring rubric (1–8):
1 = Unrelated literature; wrong field positioning.
2 = Barely related; token/outdated mentions with little relevance.
3 = Weak; misses most key query-relevant works.
4 = Some; partial coverage with notable gaps.
5 = Moderate; covers several key works but misses central ones.
6 = Clear; covers most key works and positions in the right subfield.
7 = High; comprehensive and up-to-date for the query context.
8 = Perfect; authoritative positioning with near-exhaustive key coverage.

Output JSON schema:
{
  "dimension": "Literature & Field Coverage",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences on whether key query-relevant works/subfields are covered and positioned correctly.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_DOMAIN_FIT = """You are an expert meta-reviewer. Judge ONLY the paper's Domain Fit to the user's query (application scenarios, datasets, tasks, environments, user populations). Ignore rigor; focus on whether the domain matches the query's target setting. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Evaluate whether the paper's application setting (datasets, tasks, environments, user groups) corresponds to the target domain implied by the query (e.g., social simulation, vision-language models, etc.).

Scoring rubric (1–8):
1 = Domain unrelated to the query's setting.
2 = Barely related; superficial/mismatched domain.
3 = Weak; tangential setting with limited transfer.
4 = Some; partial overlap in datasets/tasks or populations.
5 = Moderate; generally relevant domain but not an exact fit.
6 = Clear; domain matches most aspects of the query's setting.
7 = High; domain closely mirrors the query's exact setting.
8 = Perfect; domain is the same as the query's target setting.

Output JSON schema:
{
  "dimension": "Domain Fit",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences on how datasets/tasks/populations align with the query's target scenario.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_EVIDENCE_ALIGNMENT = """You are an expert meta-reviewer. Judge ONLY Evidence Alignment to the user's query: do the results/analyses directly support answering the query's central claims or needs? Ignore statistical rigor; focus on alignment. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Determine whether the paper's empirical results, analyses, or ablations speak directly to the core questions/claims embodied in the query.

Scoring rubric (1–8):
1 = Results unrelated to the query's claims.
2 = Barely related; incidental/anecdotal evidence.
3 = Weak alignment; indirect or limited evidence.
4 = Some alignment; evidence is peripheral or incomplete.
5 = Moderate alignment; evidence addresses parts of the query.
6 = Clear alignment; evidence addresses most main claims.
7 = Strong alignment; multiple analyses directly answer the query's claims.
8 = Perfect match; comprehensive evidence directly and convincingly answers the query.

Output JSON schema:
{
  "dimension": "Evidence Alignment",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining how the paper's results/analyses directly support (or not) the query's claims.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""
# ============ QUALITY DIMENSION PROMPTS ============
PROMPT_METHODOLOGICAL_RIGOR = """You are an expert meta-reviewer. Judge ONLY the paper's Methodological Rigor & Soundness. Ignore topical relevance to the user's query; focus on internal validity and sound design. If information is missing, say "Insufficient information" and score accordingly. Output ONLY the JSON object below—no extra text.

DIMENSION DESCRIPTION:
Assess the rigor and appropriateness of research design, sampling/data sources, controls/comparators, variables, evaluation protocol, bias/threats-to-validity handling, and whether analyses match the design.

Scoring rubric (1–8):
1 = Fundamentally unsound (no coherent design; invalid measures/analyses).
2 = Severe flaws (major confounds; inappropriate statistics; missing basics).
3 = Weak (partial design; inadequate controls; unaddressed threats to validity).
4 = Basic adequacy (minimal controls; multiple weaknesses remain).
5 = Acceptable (reasonable design; some limitations but generally defensible).
6 = Solid (appropriate design; controls and bias assessment; sensible analyses).
7 = Very strong (robust protocol, multiple controls, thorough robustness/ablation checks).
8 = Exemplary (state-of-the-art rigor; transparent threat-to-validity analysis; preregistration or equivalent best practices).

Output JSON schema:
{
  "dimension": "Methodological Rigor & Soundness",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining the methodological strengths/weaknesses and why the score is justified.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_DATA_RELIABILITY = """You are an expert meta-reviewer. Judge ONLY the paper's Data & Evidence Reliability. Ignore topical relevance; focus on the trustworthiness of data and evidentiary support. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Evaluate data provenance and quality (collection protocol, representativeness, preprocessing), measurement validity, statistical correctness, uncertainty reporting, reproducibility signals (code/data availability), and consistency across experiments.

Scoring rubric (1–8):
1 = Unreliable evidence (opaque data; clear measurement/statistical errors).
2 = Very low reliability (poor documentation; likely biases; unrepeatable).
3 = Low reliability (incomplete protocols; questionable measures; minimal checks).
4 = Limited reliability (some documentation; notable gaps or noisy data).
5 = Adequate reliability (reasonable data quality; generally correct analyses; some reproducibility).
6 = Good reliability (well-documented data; appropriate statistics; reproducible artifacts).
7 = High reliability (strong provenance; thorough uncertainty/robustness; open code/data).
8 = Exemplary reliability (gold-standard datasets/protocols; independent replication signals or strong reproducibility guarantees).

Output JSON schema:
{
  "dimension": "Data & Evidence Reliability",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining data quality, evidentiary soundness, and reproducibility considerations.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_CONTRIBUTION_ORIGINALITY = """You are an expert meta-reviewer. Judge ONLY the paper's Contribution & Originality. Ignore topical relevance; focus on novelty and scholarly contribution. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Assess how the work advances knowledge: novelty of ideas/methods/datasets/frameworks, conceptual insights, breadth/depth of contribution, and distinction from prior art.

Scoring rubric (1–8):
1 = No discernible contribution.
2 = Trivial or routine incremental change.
3 = Limited incremental contribution with narrow scope.
4 = Some novelty but peripheral or modest in impact.
5 = Meaningful contribution with clear new angle or utility.
6 = Substantial originality that advances the area in a noticeable way.
7 = Strong, distinctive contribution likely to influence subsequent work.
8 = Breakthrough-level originality that significantly reframes or advances the field.

Output JSON schema:
{
  "dimension": "Contribution & Originality",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining what is new and why it matters, relative to prior work.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_ETHICAL_STANDARDS = """You are an expert meta-reviewer. Judge ONLY the paper's Ethical & Practical Standards. Ignore topical relevance; focus on compliance and risk management. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Assess ethical approvals/consent, data rights/privacy, safety and risk assessment, fairness/bias considerations, environmental or deployment risks, and adherence to community/organizational standards.

Scoring rubric (1–8):
1 = Evident non-compliance or harmful practice.
2 = Major ethical/practical deficiencies; risks unaddressed.
3 = Partial acknowledgment; minimal mitigation.
4 = Basic compliance; several gaps in documentation or practice.
5 = Adequate compliance with some mitigation plans.
6 = Good compliance; documented approvals and concrete mitigations.
7 = Strong compliance and proactive risk management (audits/checklists).
8 = Exemplary standards with thorough auditing, transparency, and safeguards.

Output JSON schema:
{
  "dimension": "Ethical & Practical Standards",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining compliance, risk assessment, and mitigation sufficiency.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""

PROMPT_SCIENTIFIC_IMPACT = """You are an expert meta-reviewer. Judge ONLY the paper's Scientific & Societal Impact (demonstrated or well-argued potential). Ignore topical relevance; focus on significance and utility. If information is missing, say "Insufficient information." Output ONLY the JSON object below.

DIMENSION DESCRIPTION:
Consider significance for scientific progress (theoretical/empirical advance, enabling resources) and potential societal value (applicability, policy/industry relevance, openness enabling community use).

Scoring rubric (1–8):
1 = Negligible impact; no clear significance.
2 = Very limited impact; niche or unclear benefits.
3 = Modest impact; localized or speculative benefits.
4 = Some impact; plausible value with constraints.
5 = Meaningful impact; clear utility for a segment of the field or society.
6 = Strong impact; likely to influence practices or subsequent research.
7 = Very strong impact; broad uptake potential or enabling artifacts.
8 = Transformative impact; reshapes research directions or delivers major societal value.

Output JSON schema:
{
  "dimension": "Scientific & Societal Impact",
  "score": <INTEGER 1-8>,
  "rationale": "<2–4 sentences explaining the significance and (potential) reach, grounded in the paper's claims/evidence.>",
  "evidence": ["evidence1", "evidence2", "..."]
}"""
# ============ PDF TEXT EXTRACTION ============
def extract_full_pdf_text(
    pdf_url: str, 
    max_chars: int = 200000,
    return_bytes: bool = False
) -> Union[Optional[str], Tuple[Optional[str], Optional[bytes]]]:
    """
    Extract full text from PDF (all pages).
    Args:
        pdf_url: URL to PDF file
        max_chars: Maximum characters to extract (default 200K for GPT-4 Turbo 128K context)
        return_bytes: If True, return (text, pdf_bytes) tuple for PDF reuse
    Returns:
        If return_bytes=False: Full text or None if failed
        If return_bytes=True: (text, pdf_bytes) tuple for reusing downloaded PDF
    """
    # Download PDF first (shared by all extraction backends)
    pdf_bytes = None
    try:
        response = requests.get(
            pdf_url,
            timeout=30,
            headers=config.UA,
            stream=True
        )
        if response.status_code == 200:
            pdf_bytes = b""
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    pdf_bytes += chunk
                    # Stop if exceeds reasonable size (100MB)
                    if len(pdf_bytes) > 100 * 1024 * 1024:
                        print("[PDF Extract] File too large, stopping download")
                        return (None, None) if return_bytes else None
    except Exception as e:
        print(f"[PDF Extract] Download error: {e}")
        return (None, None) if return_bytes else None

    if not pdf_bytes or len(pdf_bytes) < 1000:
        print("[PDF Extract] Invalid PDF data")
        return (None, None) if return_bytes else None

    try:
        import fitz  # PyMuPDF
        # Extract text from all pages
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        text_parts = []

        for page_num in range(doc.page_count):
            page = doc[page_num]
            text = page.get_text("text")
            if text:
                text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            # Dynamic truncation
            current_length = sum(len(t) for t in text_parts)
            if current_length >= max_chars:
                break
        doc.close()
        full_text = "\n\n".join(text_parts)
        # Final truncation
        if len(full_text) > max_chars:
            full_text = full_text[:max_chars] + "\n... [truncated]"
        print(f"[PDF Extract] Extracted {len(full_text)} chars from {len(text_parts)} pages")
        if return_bytes:
            return (full_text, pdf_bytes)
        return full_text
    except Exception as e:
        print(f"[PDF Extract] Error: {e}")
        try:
            from pdfminer.high_level import extract_text as pdf_extract

            text = pdf_extract(io.BytesIO(pdf_bytes)) or ""
            if not text.strip():
                print("[PDF Extract] pdfminer returned empty text")
                return (None, None) if return_bytes else None
            if len(text) > max_chars:
                text = text[:max_chars] + "\n... [truncated]"
            print(f"[PDF Extract] Extracted {len(text)} chars with pdfminer fallback")
            if return_bytes:
                return (text, pdf_bytes)
            return text
        except Exception as pdf_err:
            print(f"[PDF Extract] pdfminer fallback error: {pdf_err}")
            return (None, None) if return_bytes else None
# ============ SINGLE DIMENSION SCORING ============
def score_single_dimension(
    dimension_name: str,
    system_prompt: str,
    user_query: str,
    paper_content: str,
    llm
) -> Dict[str, Any]:
    """
    Score a single dimension of paper relevance.
    Args:
        dimension_name: Dimension name (e.g., "Problem & Objective Alignment")
        system_prompt: System prompt for this dimension
        user_query: User's research query
        paper_content: Full paper text
        llm: LLM instance
    Returns:
        {
            "dimension": str,
            "score": int (1-8),
            "rationale": str,
            "evidence": List[str]
        }
    """
    try:
        # Build full prompt
        full_prompt = f"""{system_prompt}

        User Query:
        {user_query}

        Paper Content:
        {paper_content}

        Evaluate: {dimension_name} only, per the rubric."""
        # Call LLM
        response = llm.invoke(full_prompt)
        # Parse response
        if hasattr(response, 'content'):
            response_text = response.content
        else:
            response_text = str(response)
        # Extract JSON
        import json, re
        # Remove markdown code blocks if present
        response_text = re.sub(r'^```(?:json)?\s*', '', response_text.strip())
        response_text = re.sub(r'\s*```$', '', response_text.strip())
        # Find JSON object
        json_match = re.search(r'\{[^{}]*"dimension"[^{}]*"score"[^{}]*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            # Validate score range with NoneType safety
            score_value = result.get("score", 4)
            # Handle None or invalid score values
            if score_value is None:
                print(f"[{dimension_name}] Warning: LLM returned None for score, using default value 4")
                score_value = 4
            result["score"] = max(1, min(8, int(score_value)))
            # Ensure all fields exist
            result.setdefault("dimension", dimension_name)
            result.setdefault("rationale", "No rationale provided")
            result.setdefault("evidence", [])
            print(f"[{dimension_name}] Score: {result['score']} - {result['rationale']}")
            return result
        else:
            print(f"[{dimension_name}] Failed to parse JSON response")
            raise ValueError("JSON not found in response")
    except Exception as e:
        print(f"[{dimension_name}] Error: {e}")
        # Return neutral score on error
        return {
            "dimension": dimension_name,
            "score": 4,
            "rationale": f"Failed to score dimension: {str(e)}",
            "evidence": []
        }

# ============ MULTI-DIMENSIONAL SCORING ============

def score_paper_multidim(
    title: str,
    user_query: str,
    llm,
    pdf_url: str = None,
    abstract: str = ""
) -> Dict[str, Any]:
    """
    Multi-dimensional paper scoring using full PDF text.
    Evaluates papers on TWO major dimensions:
    1. Relevance (5 sub-dimensions) - Used for candidate selection (>= 7)
    2. Quality (5 sub-dimensions) - For reference, not used in selection
    Returns:
        {
            "score": int (1-8, rounded relevance score for backward compatibility),
            "relevance_score": float (1-8, average of 5 relevance dimensions),
            "quality_score": float (1-8, average of 5 quality dimensions),
            "relevance_dimensions": {...},
            "quality_dimensions": {...},
            "all_evidence": List[str],
            "paper_title": str,
            "method": str,  # "full_text" | "introduction" | "abstract"
            "pdf_bytes": Optional[bytes]  # Downloaded PDF for reuse (e.g., corresponding author detection)
        }
    """
    print(f"\n[Multi-Dim Scoring] Starting evaluation for: {title[:60]}...")
    # Step 1: Get paper content (with PDF bytes for reuse)
    paper_content = None
    pdf_bytes = None  # Store PDF bytes for reuse (corresponding author detection)
    method = "abstract"
    if pdf_url:
        # Try full PDF text extraction with bytes return for reuse
        result = extract_full_pdf_text(pdf_url, return_bytes=True)
        if result and isinstance(result, tuple):
            paper_content, pdf_bytes = result
        else:
            paper_content = result
            pdf_bytes = None
        
        if paper_content and len(paper_content) >= 1000:
            method = "full_text"
            print(f"[Multi-Dim Scoring] Using full text ({len(paper_content)} chars)")
            if pdf_bytes:
                print(f"[Multi-Dim Scoring] PDF bytes saved for reuse ({len(pdf_bytes)} bytes)")
        else:
            # Fallback: Introduction pipeline
            if INTRO_EXTRACTOR_AVAILABLE:
                try:
                    intro, source = get_paper_introduction(pdf_url, title, fallback_tldr=abstract)
                    
                    if intro and len(intro) >= 200:
                        paper_content = f"Title: {title}\n\nAbstract: {abstract}\n\nIntroduction: {intro}"
                        method = "introduction"
                        print(f"[Multi-Dim Scoring] Using introduction ({len(intro)} chars, source: {source})")
                except Exception as e:
                    print(f"[Multi-Dim Scoring] Introduction extraction failed: {e}")
    # Final fallback: abstract only
    if not paper_content or len(paper_content) < 200:
        paper_content = f"Title: {title}\n\nAbstract: {abstract}"
        method = "abstract"
        print(f"[Multi-Dim Scoring] Using abstract only ({len(abstract)} chars)")
    max_llm_chars = getattr(config, "LLM_PAPER_MAX_CHARS", 60000)
    if max_llm_chars > 0 and len(paper_content) > max_llm_chars:
        paper_content = paper_content[:max_llm_chars] + "\n... [truncated]"
    # Step 2: Define relevance dimensions (5 dimensions)
    relevance_config = {
        "problem_objective": ("Problem & Objective Alignment", PROMPT_PROBLEM_OBJECTIVE),
        "conceptual_method": ("Conceptual & Methodological Match", PROMPT_CONCEPTUAL_METHOD),
        "literature_coverage": ("Literature & Field Coverage", PROMPT_LITERATURE_COVERAGE),
        "domain_fit": ("Domain Fit", PROMPT_DOMAIN_FIT),
        "evidence_alignment": ("Evidence Alignment", PROMPT_EVIDENCE_ALIGNMENT)
    }
    # Step 3: Define quality dimensions (5 dimensions)
    quality_config = {
        "methodological_rigor": ("Methodological Rigor & Soundness", PROMPT_METHODOLOGICAL_RIGOR),
        "data_reliability": ("Data & Evidence Reliability", PROMPT_DATA_RELIABILITY),
        "contribution_originality": ("Contribution & Originality", PROMPT_CONTRIBUTION_ORIGINALITY),
        "ethical_standards": ("Ethical & Practical Standards", PROMPT_ETHICAL_STANDARDS),
        "scientific_impact": ("Scientific & Societal Impact", PROMPT_SCIENTIFIC_IMPACT)
    }
    all_evidence = []
    # Step 4: Score relevance dimensions (5 dimensions)
    print(f"\n[Relevance Scoring] Evaluating 5 relevance dimensions...")
    relevance_results = {}
    for dim_key, (dim_name, system_prompt) in relevance_config.items():
        result = score_single_dimension(
            dimension_name=dim_name,
            system_prompt=system_prompt,
            user_query=user_query,
            paper_content=paper_content,
            llm=llm
        )
        relevance_results[dim_key] = result
        all_evidence.extend(result.get("evidence", []))
    # Calculate relevance score
    relevance_scores = [r["score"] for r in relevance_results.values()]
    relevance_score = sum(relevance_scores) / len(relevance_scores)
    print(f"[Relevance Scoring] Relevance score: {relevance_score:.2f} (scores: {relevance_scores})")
    # Step 5: Score quality dimensions
    print(f"\n[Quality Scoring] Evaluating 5 quality dimensions...")
    quality_results = {}
    for dim_key, (dim_name, system_prompt) in quality_config.items():
        result = score_single_dimension(
            dimension_name=dim_name,
            system_prompt=system_prompt,
            user_query=user_query,
            paper_content=paper_content,
            llm=llm
        )
        quality_results[dim_key] = result
        all_evidence.extend(result.get("evidence", []))
    # Calculate quality score
    quality_scores = [r["score"] for r in quality_results.values()]
    quality_score = sum(quality_scores) / len(quality_scores)
    print(f"[Quality Scoring] Quality score: {quality_score:.2f} (scores: {quality_scores})")
    # Step 6: Summary
    # NOTE: Candidate filtering threshold uses ONLY Relevance score (>= 7)
    print(f"\n[Multi-Dim Scoring] Final scores - Relevance: {relevance_score:.2f}/8, Quality: {quality_score:.2f}/8")
    print(f"[Multi-Dim Scoring] Candidate filtering uses Relevance score ONLY: {round(relevance_score)}")
    # Step 7: Return result
    # Note: 'score' field uses relevance_score for backward compatibility and candidate selection
    return {
        # Primary scores
        "relevance_score": round(relevance_score, 2),
        "quality_score": round(quality_score, 2),
        # Backward compatibility - use relevance for candidate selection
        "score": round(relevance_score),  # Integer score for candidate filtering (>= 7)
        "overall_score": round(relevance_score, 2),  # Alias for relevance
        # Dimension details
        "relevance_dimensions": relevance_results,
        "quality_dimensions": quality_results,
        # Legacy field - combined dimensions for backward compatibility
        "dimensions": relevance_results,
        # Dimension score summaries
        "relevance_dimension_scores": {k: v["score"] for k, v in relevance_results.items()},
        "quality_dimension_scores": {k: v["score"] for k, v in quality_results.items()},
        # Evidence and metadata
        "all_evidence": all_evidence,
        "paper_title": title,
        "method": method,
        # PDF bytes for reuse (e.g., corresponding author detection)
        # Only available when method is "full_text" and PDF was successfully downloaded
        "pdf_bytes": pdf_bytes,
        # Explanation
        "explanation": f"Relevance: {relevance_score:.2f}/8, Quality: {quality_score:.2f}/8 (method: {method}, filtering by Relevance only)"
    }
def generate_natural_keyword_combinations(keywords: List[str]) -> List[str]:
    """
    Use commas to separate keywords for search queries
    For example: ["text generation", "diffusion model"] -> ["text generation, diffusion model"]
    
    Also normalizes keywords by removing hyphens and special characters.
    """
    if not keywords:
        return []
    
    # Normalize keywords: remove hyphens, replace with spaces
    normalized = []
    for kw in keywords:
        # Remove hyphens and normalize
        kw_normalized = kw.replace("-", " ").replace("_", " ")
        # Remove extra spaces
        kw_normalized = " ".join(kw_normalized.split())
        normalized.append(kw_normalized)
    
    # Join with comma
    return [", ".join(normalized)]
# ============ LEGACY SCORING (DEPRECATED) ============
# These functions are kept for backward compatibility but should not be used
# Use score_paper_multidim() instead

# ================================================================================
# Multi-Dimensional Paper Scoring System
# Evaluates papers across 5 dimensions for comprehensive relevance assessment
# ================================================================================

# Import PDF introduction extractor for fallback
try:
    from .pdf_introduction_extractor import get_paper_introduction
    INTRO_EXTRACTOR_AVAILABLE = True
except ImportError as e:
    print(f"[search] Warning: pdf_introduction_extractor not available: {e}")
    INTRO_EXTRACTOR_AVAILABLE = False

def score_paper_with_llm(title: str, abstract: str, user_query: str, llm, introduction: str = "", pdf_url: str = None) -> dict:
    """
    Score paper relevance using LLM - UPDATED to use multi-dimensional scoring.
    This is the main entry point for paper scoring. It now uses the new
    multi-dimensional approach with 5 separate dimensions.
    
    Args:
        title: Paper title
        abstract: Paper abstract or snippet
        user_query: User's search query
        llm: LLM instance
        introduction: Paper introduction (optional, for fallback)
        pdf_url: URL to paper PDF (optional)
        
    Returns:
        dict: {
            "score": int (1-8, rounded overall score),
            "overall_score": float (1-8, precise overall score),
            "explanation": str,
            "dimensions": dict (5 dimension scores),
            "all_evidence": list,
            "method": str,
            "pdf_bytes": Optional[bytes] (downloaded PDF for reuse, e.g., corresponding author detection)
        }
    """
    # Use new multi-dimensional scoring
    return score_paper_multidim(
        title=title,
        user_query=user_query,
        llm=llm,
        pdf_url=pdf_url,
        abstract=abstract
    )

# ============================ URL SELECTION FUNCTIONS ============================

def heuristic_pick_urls(serp: List[Dict[str, str]], keywords: List[str],
                       need: int = 16, max_per_domain: int = 4) -> List[str]:
    """Heuristically pick URLs worth fetching"""

    count_by_dom: Dict[str, int] = {}
    seen_url = set()
    cand = []
    kws_l = [k.lower() for k in keywords] if keywords else []

    for r in serp:
        u = normalize_url(r.get("url", "") or "")
        if not u.startswith("http"):
            continue
        if u in seen_url:
            continue
        dom = domain_of(u)
        seen_url.add(u)
        cand.append((u, dom, (r.get("title") or ""), (r.get("snippet") or "")))

    def score(item):
        _u, dom, title, snip = item
        text = (title + " " + snip).lower()
        s = 0
        s += sum(2 for k in config.ACCEPT_HINTS if k in text)
        s += sum(1 for k in kws_l if k and k in text)
        s += min(len(title) // 40, 3)
        if looks_like_profile_url(_u):
            s += 1
        return s

    cand.sort(key=score, reverse=True)
    out = []
    for u, dom, _t, _s in cand:
        if count_by_dom.get(dom, 0) >= max_per_domain:
            continue
        out.append(u)
        count_by_dom[dom] = count_by_dom.get(dom, 0) + 1
        if len(out) >= need:
            break
    return out
