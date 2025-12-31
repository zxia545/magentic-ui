from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ============================ OUTPUT & STORAGE CONFIG ============================
SAVE_DIR = str(DATA_DIR / "results")

# ============================ SEARCH CONFIG (Bing-backed) ============================
SEARXNG_BASE_URL = os.getenv("SEARXNG_BASE_URL", "")
SEARXNG_PAGES = 1
SEARXNG_ENGINES_PAPER_SEARCH = ["bing"]
SEARXNG_ENGINES_HOMEPAGE = ["bing"]
SEARXNG_ENGINES_ARXIV = ["bing"]
SEARXNG_ENGINES_SNIPPET = ["bing"]
SEARXNG_ENGINES = SEARXNG_ENGINES_PAPER_SEARCH
PAPER_SEARCH_BACKEND = os.getenv("PAPER_SEARCH_BACKEND", "web_agent")
BING_PAPER_MAX_RESULTS = 30
BING_PAPER_MIN_MATCH_SCORE = 0.7
BING_USE_PLAYWRIGHT = os.getenv("MAGENTIC_UI_BING_USE_PLAYWRIGHT", "true").lower() in {
    "1",
    "true",
    "yes",
}
WEB_AGENT_SEARCH_ENGINES = [
    engine.strip()
    for engine in os.getenv("WEB_AGENT_SEARCH_ENGINES", "bing,scholar").split(",")
    if engine.strip()
]
WEB_AGENT_MAX_TOKENS = int(os.getenv("WEB_AGENT_MAX_TOKENS", "4000"))
WEB_AGENT_TIMEOUT_SEC = int(os.getenv("WEB_AGENT_TIMEOUT_SEC", "15"))

# ============================ API KEYS ============================
USE_SEMANTIC_SCHOLAR_FALLBACK = os.getenv(
    "USE_SEMANTIC_SCHOLAR_FALLBACK", "True"
).lower() in {"1", "true", "yes"}
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")

# ============================ SEARCH & PROCESSING PARAMETERS ============================
MAX_ROUNDS = 3
SEARCH_K = 10
SELECT_K = 16
FETCH_MAX_CHARS = 15000
VERBOSE = True
DEFAULT_TOP_N = 10

# User Agent
UA = {"User-Agent": "Mozilla/5.0 (Magentic-UI TalentSearch)"}

# ============================ LLM TOKEN LIMITS ============================
LLM_OUT_TOKENS = {
    "parse": 2048,
    "plan": 2048,
    "select": 2048,
    "authors": 2048,
    "synthesize": 3072,
    "paper_name": 2048,
    "degree_matcher": 50,
    "venue_verify": 512,
    "extract": 2048,
    "score": 2048,
    "evaluate": 2048,
    "translate": 2048,
}

USE_LLM_PAPER_SCORING = True
ENABLE_LLM_DEGREE_MATCHING = True

# ============================ DEFAULT CONFERENCES & YEARS ============================
DEFAULT_CONFERENCES: Dict[str, List[str]] = {
    "NeurIPS": ["NeurIPS"],
    "AAAI": ["AAAI"],
    "IJCAI": ["IJCAI"],
    "ICML": ["ICML"],
    "ICLR": ["ICLR"],
    "KDD": ["KDD"],
    "ACL": ["ACL"],
    "EMNLP": ["EMNLP"],
    "NAACL": ["NAACL"],
    "AE": ["AE"],
    "IJCAR": ["IJCAR"],
    "CVPR": ["CVPR"],
    "ICCV": ["ICCV"],
    "ECCV": ["ECCV"],
    "COLING": ["COLING"],
    "SIGGRAPH": ["SIGGRAPH"],
    "ACM-MM": ["ACM MM"],
    "VIS": ["VIS"],
    "SIGMOD": ["SIGMOD"],
    "VLDB": ["VLDB"],
    "SIGIR": ["SIGIR"],
    "PODC": ["PODC"],
    "SIGCOMM": ["SIGCOMM"],
    "INFOCOM": ["INFOCOM"],
    "NSDI": ["NSDI"],
    "CHI": ["CHI"],
}

_current_year = datetime.now().year
DEFAULT_YEARS = [_current_year, _current_year - 1, _current_year - 2]

CORE_CONFERENCES = ["ICLR", "ICML", "NeurIPS"]

TOP_TIER_CONFERENCES = [
    "ACL",
    "EMNLP",
    "NAACL",
    "CVPR",
    "ICCV",
    "ECCV",
    "KDD",
    "WWW",
    "SIGIR",
    "AAAI",
    "IJCAI",
    "CHI",
]

CS_TOP_CONFERENCES = {
    "Artificial Intelligence": ["AAAI", "IJCAI"],
    "Machine Learning": ["NeurIPS", "ICML", "ICLR"],
    "Computer Vision": ["CVPR", "ICCV", "ECCV"],
    "Natural Language Processing": ["ACL", "EMNLP", "NAACL"],
    "Data Mining": ["KDD", "WWW", "WSDM"],
    "Robotics": ["ICRA", "IROS", "RSS", "CoRL"],
    "Computer Security": ["S&P", "USENIX Security", "CCS", "NDSS"],
    "Databases": ["SIGMOD", "VLDB", "ICDE"],
    "Systems": ["OSDI", "SOSP", "NSDI"],
    "HCI": ["CHI", "UIST"],
}

# ============================ VALIDATION CONSTANTS ============================
ACCEPT_HINTS = [
    "accepted papers",
    "accept",
    "acceptance",
    "program",
    "proceedings",
    "schedule",
    "paper list",
    "main conference",
    "research track",
]

MAX_AUTHORS = 25
MAX_KEYWORDS = 32
MAX_VENUES = 32
MAX_DEGREE_LEVELS = 32
MAX_AUTHOR_PRIORITY = 32
MAX_EXTRA_CONSTRAINTS = 32
MAX_SEARCH_TERMS = 120
MAX_URLS = 16

ENABLE_NAME_FILTER = True
MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 50

MIN_PAPER_SCORE = 6

LEAD_TOP_K = 3
SUPPORT_TOP_K = 5
ALPHA_NO_LEAD = 0.15
BETA_SUPPORT = 0.3

MIN_TEXT_LENGTH = 50
MIN_AUTHOR_NAME_LENGTH = 2
MAX_AUTHOR_NAME_LENGTH = 80

# ============================ DATABASE CONFIG ============================
ENABLE_DATABASE_STORAGE = False
DB_TYPE = os.getenv("TALENT_DB_TYPE", "sqlite")
SQLITE_DB_PATH = os.getenv("TALENT_SQLITE_DB_PATH", str(DATA_DIR / "talent_search.db"))

# ============================ QUERY POOL CONFIG ============================
GENERIC_KEYWORDS = {
    "llm",
    "ai",
    "ml",
    "machine learning",
    "deep learning",
    "neural network",
    "transformer",
    "model",
    "learning",
    "algorithm",
    "method",
    "approach",
    "framework",
    "system",
    "data",
    "analysis",
    "optimization",
    "training",
    "network",
    "computation",
    "computing",
    "intelligence",
    "neural",
    "generative",
    "foundation model",
    "language model",
    "representation",
    "embedding",
    "attention",
    "encoder",
    "decoder",
}

QUERY_POOL_CONFIG = {
    "max_variants": 5,
    "empty_batches_threshold": 2,
    "min_papers_per_batch": 5,
}

# ============================ BATCH SEARCH PARAMETERS ============================
SEARCH_BATCH_CHUNK = 1
