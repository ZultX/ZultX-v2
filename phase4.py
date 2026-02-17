# phase4.py — Hardened / optimized (drop-in)
"""
ZULTX Phase 4 — Hardened memory orchestration (patched)
- Compatible with Postgres 'conversations' schema (session_id,user_id,owner,role,content,created_at,ts)
  and older/simple schema (session_id,owner,role,content,ts).
- Loads recent messages by session_id when present, else by owner (user_id or guest:...).
- Keeps in-memory buffer keyed by session_id if present, otherwise owner.
- Defensive SQL and row-access logic for both psycopg2 and sqlite3 rows.
"""
import os
import re
import json
import time
import uuid
import math
import threading
import traceback
import sqlite3
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# optional fast vector recall
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# DB adapters / env
DB_URL = os.getenv("DATABASE_URL")  # Postgres URL if provided
USE_POSTGRES = bool(DB_URL)
PG_AVAILABLE = False
PG_POOL = None
if USE_POSTGRES:
    try:
        import psycopg2
        import psycopg2.extras
        from psycopg2.pool import SimpleConnectionPool
        PG_AVAILABLE = True
        min_conn = int(os.getenv("PG_POOL_MIN", "1"))
        max_conn = int(os.getenv("PG_POOL_MAX", "6"))
        try:
            PG_POOL = SimpleConnectionPool(min_conn, max_conn, DB_URL)
        except Exception as e:
            print("[phase_4] PG pool init failed:", e)
            PG_POOL = None
            PG_AVAILABLE = False
    except Exception as e:
        print("[phase_4] psycopg2 not available:", e)
        PG_AVAILABLE = False

# sqlite fallback
SQLITE_PATH = os.getenv("ZULTX_MEMORY_DB", "zultx_memory.db")
USE_SQLITE = not PG_AVAILABLE

# (skip Pinecone/OpenAI parts — keep as in your file)
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENV = os.getenv("PINECONE_ENV")
PINECONE_INDEX = os.getenv("PINECONE_INDEX")
USE_PINECONE = bool(PINECONE_API_KEY and PINECONE_INDEX)
PINECONE_CLIENT = None
_pine_index = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EMBED = False
if OPENAI_API_KEY:
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
        USE_OPENAI_EMBED = True
    except Exception as e:
        print("[phase_4] openai import failed:", e)
        USE_OPENAI_EMBED = False

# reasoning core import (phase_3)
try:
    from phase_3 import ask as phase3_ask
except Exception as e:
    phase3_ask = None
    print("[phase_4] WARNING: phase_3.ask not importable:", e)

# CONFIG knobs
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1200"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "6"))
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.60"))
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.85"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.80"))
STM_EXPIRE_DAYS = int(os.getenv("ZULTX_STM_DAYS", "1"))
CM_EXPIRE_DAYS = int(os.getenv("ZULTX_CM_DAYS", "365"))
EM_DEFAULT_DAYS = int(os.getenv("ZULTX_EM_DAYS", "7"))

# Sensitive patterns (same)
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    re.compile(r"\b(?:card(?:-|\s)?num|credit card|visa|mastercard)\b", re.I),
]

# Locks
_db_lock = threading.RLock()
_RECALL_BUILD_LOCK = threading.Lock()
_CONV_LOCK = threading.Lock()

# debounce
_RECALL_DEBOUNCE_SECONDS = int(os.getenv("RECALL_DEBOUNCE_SECONDS", "60"))
_last_recall_build = 0.0
_recall_build_scheduled = False

# DB init SQL (Postgres + SQLite compatibility)
# Keep memories and other tables identical to your original; conversations is flexible (we create both forms)
_POSTGRES_INIT_SQL = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    raw_snippet TEXT,
    created_at TIMESTAMP,
    last_used TIMESTAMP,
    expires_at TIMESTAMP,
    confidence DOUBLE PRECISION,
    importance DOUBLE PRECISION,
    frequency INTEGER,
    memory_score DOUBLE PRECISION,
    tags JSONB,
    consent BOOLEAN,
    metadata JSONB
);
CREATE INDEX IF NOT EXISTS idx_owner_type_score ON memories (owner, type, memory_score DESC NULLS LAST, last_used DESC NULLS LAST);
CREATE INDEX IF NOT EXISTS idx_expires_at ON memories (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    action TEXT,
    mem_id TEXT,
    payload JSONB
);

CREATE TABLE IF NOT EXISTS queue_pending (
    id TEXT PRIMARY KEY,
    ts TIMESTAMP,
    payload JSONB
);

-- Conversations: allow both owner and user_id plus created_at and ts
CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    user_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    created_at TIMESTAMP,
    ts TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session_created_at ON conversations (session_id, created_at DESC);
"""

_SQLITE_INIT_SQL = f"""
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    owner TEXT,
    type TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT,
    raw_snippet TEXT,
    created_at TEXT,
    last_used TEXT,
    expires_at TEXT,
    confidence REAL,
    importance REAL,
    frequency INTEGER,
    memory_score REAL,
    tags TEXT,
    consent INTEGER,
    metadata TEXT
);
CREATE INDEX IF NOT EXISTS idx_owner_type_score ON memories (owner, type, memory_score DESC, last_used DESC);
CREATE INDEX IF NOT EXISTS idx_expires_at ON memories (expires_at);

CREATE TABLE IF NOT EXISTS audit_log (
    id TEXT PRIMARY KEY,
    ts TEXT,
    action TEXT,
    mem_id TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS queue_pending (
    id TEXT PRIMARY KEY,
    ts TEXT,
    payload TEXT
);

CREATE TABLE IF NOT EXISTS conversations (
    session_id TEXT,
    user_id TEXT,
    owner TEXT,
    role TEXT,
    content TEXT,
    created_at TEXT,
    ts TEXT
);
CREATE INDEX IF NOT EXISTS idx_conversations_session_ts ON conversations (session_id, ts DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_session_created_at ON conversations (session_id, created_at DESC);
"""

# Utilities
def now_ts() -> str:
    return datetime.utcnow().isoformat()

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
        except Exception:
            return None

def uuid4() -> str:
    return str(uuid.uuid4())

def clamp(v, a=0.0, b=1.0):
    try:
        v = float(v)
    except Exception:
        v = a
    return max(a, min(b, v))

# DB connection helpers (same as your original)
def _pg_getconn():
    if PG_POOL:
        try:
            conn = PG_POOL.getconn()
            conn.autocommit = False
            return conn
        except Exception:
            pass
    conn = psycopg2.connect(DB_URL)
    conn.autocommit = False
    return conn

def _pg_putconn(conn):
    try:
        if PG_POOL and hasattr(psycopg2, "extensions") and isinstance(conn, psycopg2.extensions.connection):
            PG_POOL.putconn(conn)
            return
    except Exception:
        pass
    try:
        conn.close()
    except Exception:
        pass

def get_db_conn():
    if PG_AVAILABLE and DB_URL:
        return _pg_getconn()
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

# initialize DB
def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        try:
            if PG_AVAILABLE and DB_URL:
                cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                cur.execute(_POSTGRES_INIT_SQL)
                conn.commit()
                cur.close()
                _pg_putconn(conn)
            else:
                cur = conn.cursor()
                cur.executescript(_SQLITE_INIT_SQL)
                conn.commit()
                cur.close()
                conn.close()
        except Exception as e:
            print("[phase_4] initialize_db error:", e)
            try:
                if PG_AVAILABLE and DB_URL:
                    _pg_putconn(conn)
                else:
                    conn.close()
            except Exception:
                pass

initialize_db()

# defensive row accessor to avoid tuple/dict mismatch errors
def _col(row: Any, name: str, idx: int):
    try:
        if row is None:
            return None
        if isinstance(row, dict):
            return row.get(name)
        if hasattr(row, '__getitem__') and callable(getattr(row, "keys", None)) and name in row.keys():
            try:
                return row[name]
            except Exception:
                pass
        try:
            return row[idx]
        except Exception:
            return getattr(row, name, None)
    except Exception:
        return None

# SimpleRecall, other memory helpers kept as-is (use your original implementations)
class SimpleRecall:
    def __init__(self):
        self.vectorizer = None
        self.corpus = []
        self.matrix = None
        self.last_build = 0

    def build_from_db(self):
        global _last_recall_build
        with _RECALL_BUILD_LOCK:
            rows = []
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')")
                    rows = cur.fetchall()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    rows = cur.execute("SELECT id, owner, content FROM memories WHERE type IN ('CM','LTM')").fetchall()
                    cur.close()
                    conn.close()
                corpus = []
                for r in rows:
                    try:
                        if isinstance(r, dict):
                            mem_id = r.get("id")
                            owner = r.get("owner")
                            content = r.get("content")
                        else:
                            mem_id = _col(r, "id", 0)
                            owner = _col(r, "owner", 1)
                            content = _col(r, "content", 2)
                    except Exception:
                        continue
                    if mem_id is None:
                        continue
                    corpus.append((str(mem_id), owner, content or ""))
                self.corpus = corpus
                texts = [t for (_id, _owner, t) in self.corpus]
                if SKLEARN_AVAIL and texts and len(texts) > 1:
                    try:
                        self.vectorizer = TfidfVectorizer(max_features=20000)
                        self.matrix = self.vectorizer.fit_transform(texts)
                    except Exception as e:
                        print("[phase_4][recall] tfidf build failed:", e)
                        self.vectorizer = None
                        self.matrix = None
                else:
                    self.vectorizer = None
                    self.matrix = None
                self.last_build = time.time()
                _last_recall_build = self.last_build
            except Exception as e:
                print("[phase_4][recall] build error:", e)

    def retrieve(self, query: str, k: int = TFIDF_TOP_K, owner: Optional[str] = None) -> List[Tuple[str, str, float]]:
        if not self.corpus:
            return []
        filtered = []
        for mem_id, mem_owner, content in self.corpus:
            if owner is None:
                if mem_owner is None:
                    filtered.append((mem_id, content))
            else:
                if mem_owner is None or mem_owner == owner:
                    filtered.append((mem_id, content))
        if not filtered:
            return []
        ids, texts = zip(*filtered)
        if self.vectorizer is not None and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([query])
                fm = self.vectorizer.transform(list(texts))
                sims = cosine_similarity(qv, fm)[0]
                idxs = sims.argsort()[::-1][:k]
                out = []
                for i in idxs:
                    out.append((ids[int(i)], texts[int(i)], float(sims[int(i)])))
                return out
            except Exception:
                pass
        q = (query or "").lower()
        scored = []
        for mem_id, t in zip(ids, texts):
            txt = (t or "").lower()
            score = 0.0
            if q and q in txt:
                score += 1.0
            for w in q.split()[:8]:
                if w and w in txt:
                    score += 0.01
            scored.append((mem_id, t, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

_recall_instance = SimpleRecall()

def _schedule_recall_rebuild(debounce_seconds: int = _RECALL_DEBOUNCE_SECONDS):
    global _recall_build_scheduled, _last_recall_build
    with _RECALL_BUILD_LOCK:
        now = time.time()
        if now - _last_recall_build < debounce_seconds:
            return
        if _recall_build_scheduled:
            return
        _recall_build_scheduled = True

    def _worker():
        global _recall_build_scheduled, _last_recall_build
        try:
            _recall_instance.build_from_db()
        except Exception as e:
            print("[phase_4] recall rebuild worker error:", e)
        finally:
            _recall_build_scheduled = False
            _last_recall_build = time.time()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

try:
    _schedule_recall_rebuild(0)
except Exception:
    pass

# Memory scoring & policy & anonymize: keep your existing implementations
def compute_memory_score(importance: float, frequency: int, created_at: Optional[str], confidence: float) -> float:
    importance = clamp(importance)
    confidence = clamp(confidence)
    norm_frequency = min(1.0, math.log2(1 + max(0, frequency)) / 6.0)
    if created_at:
        created_dt = parse_ts(created_at)
        if created_dt:
            age_days = (datetime.utcnow() - created_dt).total_seconds() / 86400.0
            recency = 1.0 / (1.0 + (age_days / 7.0))
        else:
            recency = 0.5
    else:
        recency = 0.5
    score = (0.40 * importance) + (0.30 * norm_frequency) + (0.20 * recency) + (0.10 * confidence)
    return clamp(score)

def policy_allow_store(content: str) -> Tuple[bool, Optional[str]]:
    c = content or ""
    for p in SENSITIVE_PATTERNS:
        if p.search(c):
            return False, "sensitive_pattern"
    return True, None

def anonymize_if_needed(content: str) -> str:
    s = content
    s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[email]", s)
    s = re.sub(r"\b\d{10}\b", "[phone]", s)
    return s

# Audit + queue: keep as before (use existing functions)
def audit(action: str, mem_id: Optional[str], payload: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (%s, %s, %s, %s, %s)",
                                (uuid4(), datetime.utcnow(), action, mem_id, psycopg2.extras.Json(payload)))
                    conn.commit()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?, ?, ?, ?, ?)",
                                (uuid4(), now_ts(), action, mem_id, json.dumps(payload)))
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as e:
                print("[phase_4][audit_error]", e)
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
    except Exception:
        print("[phase_4] audit top-level exception")
        traceback.print_exc()

def enqueue_retry(item: Dict[str, Any]):
    try:
        with _db_lock:
            conn = get_db_conn()
            try:
                if PG_AVAILABLE and DB_URL:
                    cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
                    cur.execute("INSERT INTO queue_pending (id, ts, payload) VALUES (%s, %s, %s)",
                                (uuid4(), datetime.utcnow(), psycopg2.extras.Json(item)))
                    conn.commit()
                    cur.close()
                    _pg_putconn(conn)
                else:
                    cur = conn.cursor()
                    cur.execute("INSERT OR REPLACE INTO queue_pending (id, ts, payload) VALUES (?,?,?)",
                                (uuid4(), now_ts(), json.dumps(item)))
                    conn.commit()
                    cur.close()
                    conn.close()
            except Exception as e:
                print("[phase_4][enqueue_error]", e)
                try:
                    if PG_AVAILABLE and DB_URL:
                        _pg_putconn(conn)
                    else:
                        conn.close()
                except Exception:
                    pass
        audit("queue_enqueue", None, {"item": str(item)[:200]})
    except Exception:
        pass

# Conversations persistence (robust)
# Key design: support both schemas and allow session-less owner writes.
def persist_conversation(session_id: Optional[str], owner: Optional[str], role: str, content: str, ts_iso: Optional[str] = None):
    ts = ts_iso or now_ts()
    # compute user_id column if owner is not guest:...
    user_id_val = None
    if owner and not str(owner).startswith("guest:"):
        user_id_val = owner
    try:
        with _db_lock:
            conn = get_db_conn()
            try
