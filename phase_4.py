# phase_4.py
"""
ZULTX Phase 4 — Memory orchestration (single-file)
- phase_4.ask(...) uses phase_3.ask(...) for core reasoning
- Maintains a SQLite memory store (metadata) + TF-IDF semantic recall
- Extractor: lightweight rules + (optional) LLM extraction stub
- Promotion rules, scoring, expiry, audit logs, queue for failed vector syncs
- Safe defaults: conservative write thresholds, policy filters
- Designed to run on Railway / local (no required external services)
"""

import os
import re
import json
import time
import sqlite3
import uuid
import math
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple

# Optional dependencies (guarded)
SKLEARN_AVAIL = False
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAIL = True
except Exception:
    SKLEARN_AVAIL = False

# Phase 3 (RAG / reasoning wrapper) import
try:
    from phase_3 import ask as phase3_ask
except Exception as e:
    phase3_ask = None
    print("[phase_4] WARNING: phase_3.ask not importable:", e)

# CONFIG
DB_PATH = os.getenv("ZULTX_MEMORY_DB", "zultx_memory.db")
VECTOR_CACHE_PATH = os.getenv("ZULTX_VECTOR_CACHE", ".zultx_vector_cache.pkl")
MAX_INJECT_TOKENS = int(os.getenv("ZULTX_MAX_INJECT_TOKENS", "1600"))
TFIDF_TOP_K = int(os.getenv("ZULTX_TFIDF_K", "12"))
PROMOTE_TO_CM_SCORE = float(os.getenv("ZULTX_PROMOTE_CM", "0.45"))
PROMOTE_TO_LTM_SCORE = float(os.getenv("ZULTX_PROMOTE_LTM", "0.80"))
CONFIDENCE_PROMOTE_THRESHOLD = float(os.getenv("ZULTX_CONF_PROMOTE", "0.75"))

USE_OPENAI_EMBEDDINGS = False  # keep false by default; can be toggled if you add env and code
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("OPENAI_KEY")

# Basic policy: regex patterns for sensitive items we should not store raw
SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{10}\b"),  # simple phone number
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # US SSN-like
    re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),  # email
    re.compile(r"\b(?:card(?:-|\s)?num|credit card|visa|mastercard)\b", re.I),
    # add more patterns as needed
]

# ---------------------------
# SQLite memory store helpers
# ---------------------------
_INIT_SQL = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    type TEXT NOT NULL, -- STM|CM|LTM|EM
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

CREATE INDEX IF NOT EXISTS idx_type_score ON memories (type, memory_score DESC, last_used DESC);
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
"""

def get_db_conn(path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False, timeout=30)
    conn.row_factory = sqlite3.Row
    return conn

_db_lock = threading.Lock()
def initialize_db():
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.executescript(_INIT_SQL)
        conn.commit()
        conn.close()

initialize_db()

# ---------------------------
# Utility functions
# ---------------------------
def now_ts() -> str:
    return datetime.utcnow().isoformat()

def parse_ts(ts: Optional[str]) -> Optional[datetime]:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None

def uuid4() -> str:
    return str(uuid.uuid4())

def clamp(v, a=0.0, b=1.0):
    return max(a, min(b, v))

# ---------------------------
# Memory score computation
# ---------------------------
def compute_memory_score(importance: float, frequency: int, created_at: Optional[str], confidence: float) -> float:
    # normalize inputs
    importance = clamp(importance)
    confidence = clamp(confidence)
    # frequency normalization: log2 scale
    norm_frequency = min(1.0, math.log2(1 + max(0, frequency)) / 6.0)
    # recency: 1 / (1 + age_days/7)
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

# ---------------------------
# Simple policy engine
# ---------------------------
def policy_allow_store(content: str) -> Tuple[bool, Optional[str]]:
    """
    Return (allowed, reason_if_blocked)
    Conservative: block if any sensitive pattern matches
    """
    c = content or ""
    for p in SENSITIVE_PATTERNS:
        if p.search(c):
            return False, "sensitive_pattern"
    # additional heuristics (e.g., profanity) can go here
    return True, None

def anonymize_if_needed(content: str) -> str:
    # Very conservative: strip emails and phone numbers to placeholders
    s = content
    s = re.sub(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", "[email]", s)
    s = re.sub(r"\b\d{10}\b", "[phone]", s)
    return s

# ---------------------------
# TF-IDF semantic recall helper (fallback)
# ---------------------------
class SimpleRecall:
    """
    If sklearn available, uses TF-IDF vectorizer for semantic search.
    Else falls back to simple substring overlap ranking.
    The corpus is built from memories' distilled `content` field (CM+LTM).
    """
    def __init__(self):
        self.vectorizer = None
        self.corpus = []  # list of (mem_id, content)
        self.matrix = None

    def build_from_db(self):
        with get_db_conn() as conn:
            cur = conn.cursor()
            cur.execute("SELECT id, content, memory_score FROM memories WHERE type IN ('CM','LTM')")
            rows = cur.fetchall()
            self.corpus = [(r["id"], r["content"]) for r in rows]
            texts = [t for (_id, t) in self.corpus]
            if SKLEARN_AVAIL and texts:
                try:
                    self.vectorizer = TfidfVectorizer(max_features=50000)
                    self.matrix = self.vectorizer.fit_transform(texts)
                except Exception:
                    self.vectorizer = None
                    self.matrix = None
            else:
                self.vectorizer = None
                self.matrix = None

    def retrieve(self, query: str, k: int = TFIDF_TOP_K) -> List[Tuple[str, str, float]]:
        # returns list of (mem_id, content, score)
        if not self.corpus:
            return []
        if self.vectorizer is not None and self.matrix is not None:
            try:
                qv = self.vectorizer.transform([query])
                sims = cosine_similarity(qv, self.matrix)[0]
                idxs = sims.argsort()[::-1][:k]
                out = []
                for i in idxs:
                    mem_id, content = self.corpus[int(i)]
                    out.append((mem_id, content, float(sims[int(i)])))
                return out
            except Exception:
                pass
        # substring fallback scoring
        q = query.lower()
        scored = []
        for mem_id, content in self.corpus:
            t = content.lower()
            score = 0.0
            if q in t:
                score += 1.0
            for w in q.split()[:10]:
                if w and w in t:
                    score += 0.01
            scored.append((mem_id, content, score))
        scored.sort(key=lambda x: x[2], reverse=True)
        return scored[:k]

_recall_instance = SimpleRecall()
def refresh_recall_index():
    _recall_instance.build_from_db()

# ---------------------------
# DB operations: CRUD + audit
# ---------------------------
def insert_memory(mem: Dict[str, Any]) -> str:
    """
    mem must contain keys:
    id, type, content, source, raw_snippet, created_at, last_used, expires_at,
    confidence, importance, frequency, memory_score, tags, consent, metadata
    """
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("""
            INSERT OR REPLACE INTO memories
            (id,type,content,source,raw_snippet,created_at,last_used,expires_at,confidence,importance,frequency,memory_score,tags,consent,metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            mem.get("id"),
            mem.get("type"),
            mem.get("content"),
            mem.get("source"),
            mem.get("raw_snippet"),
            mem.get("created_at"),
            mem.get("last_used"),
            mem.get("expires_at"),
            float(mem.get("confidence", 0.0)),
            float(mem.get("importance", 0.0)),
            int(mem.get("frequency", 1)),
            float(mem.get("memory_score", 0.0)),
            json.dumps(mem.get("tags") or []),
            1 if mem.get("consent") else 0,
            json.dumps(mem.get("metadata") or {})
        ))
        conn.commit()
        conn.close()
    audit("create_memory", mem.get("id"), {"summary": mem.get("content")[:140]})
    # refresh recall index asynchronously (cheap)
    try:
        threading.Thread(target=refresh_recall_index, daemon=True).start()
    except Exception:
        pass
    return mem.get("id")

def update_memory_last_used(mem_id: str):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("UPDATE memories SET last_used = ?, frequency = frequency + 1 WHERE id = ?", (now_ts(), mem_id))
        conn.commit()
        conn.close()
    audit("update_last_used", mem_id, {"ts": now_ts()})

def get_memory(mem_id: str) -> Optional[Dict[str, Any]]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM memories WHERE id = ?", (mem_id,))
        row = cur.fetchone()
        if not row:
            return None
        r = dict(row)
        r["tags"] = json.loads(r["tags"]) if r["tags"] else []
        r["metadata"] = json.loads(r["metadata"]) if r["metadata"] else {}
        return r

def list_memories(limit: int = 50) -> List[Dict[str, Any]]:
    with get_db_conn() as conn:
        cur = conn.cursor()
        cur.execute("SELECT * FROM memories ORDER BY memory_score DESC, last_used DESC LIMIT ?", (limit,))
        rows = cur.fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["tags"] = json.loads(d["tags"]) if d["tags"] else []
            d["metadata"] = json.loads(d["metadata"]) if d["metadata"] else {}
            out.append(d)
        return out

def audit(action: str, mem_id: Optional[str], payload: Dict[str, Any]):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("INSERT INTO audit_log (id, ts, action, mem_id, payload) VALUES (?,?,?,?,?)", (
            uuid4(), now_ts(), action, mem_id, json.dumps(payload)
        ))
        conn.commit()
        conn.close()

def enqueue_retry(item: Dict[str, Any]):
    with _db_lock:
        conn = get_db_conn()
        cur = conn.cursor()
        cur.execute("INSERT OR REPLACE INTO queue_pending (id, ts, payload) VALUES (?,?,?)", (uuid4(), now_ts(), json.dumps(item)))
        conn.commit()
        conn.close()
    audit("queue_enqueue", None, {"item": str(item)[:200]})

# ---------------------------
# Extractor (rules + LLM-assist stub)
# ---------------------------
def rule_based_extract(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    """
    Very conservative rule-based extraction:
    - "remember: X" or "remember that X" explicit commands
    - "my name is X" patterns (lower priority)
    - "I prefer X" patterns
    Returns candidate memories with suggested importance/confidence.
    """
    candidates = []
    text = (user_input or "") + "\n\n" + (assistant_text or "")
    # explicit remember
    for m in re.finditer(r"\bremember(?: that)?\s*[:\-]?\s*(.+?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content:
            candidates.append({
                "content": content,
                "suggested_type": "CM",
                "importance": 0.9,
                "confidence": 0.9,
                "reason": "explicit_remember"
            })
    # preferences
    for m in re.finditer(r"\bI (?:prefer|like|love|hate|want)\s+(.*?)(?:\.|$|\n)", text, flags=re.I):
        content = m.group(1).strip()
        if content and len(content) < 200:
            candidates.append({
                "content": "pref:" + content,
                "suggested_type": "CM",
                "importance": 0.7,
                "confidence": 0.7,
                "reason": "preference_statement"
            })
    # name
    m = re.search(r"\bmy name is\s+([A-Z][a-z]{0,30})", text)
    if m:
        name = m.group(1).strip()
        candidates.append({
            "content": f"name:{name}",
            "suggested_type": "CM",
            "importance": 0.8,
            "confidence": 0.8,
            "reason": "self_identify"
        })
    return candidates

def llm_assisted_extract_stub(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    """
    Placeholder for LLM-assisted extraction.
    For Railway-safe default, this is a stub that returns nothing.
    If you integrate OpenAI or another LLM, implement a safe prompt here and return
    candidate memory items with confidence and suggested tags.
    """
    # conservative: return empty list by default
    return []

def extract_candidates(user_input: str, assistant_text: str) -> List[Dict[str, Any]]:
    # merge rule-based + llm-assisted (llm stub optional)
    cands = rule_based_extract(user_input, assistant_text)
    cands += llm_assisted_extract_stub(user_input, assistant_text)
    # normalize candidates
    out = []
    for c in cands:
        content = c.get("content", "").strip()
        if not content:
            continue
        allowed, reason = policy_allow_store(content)
        if not allowed:
            # try anonymize
            content2 = anonymize_if_needed(content)
            # if anonymization changed it, allow with low confidence
            if content2 != content:
                content = content2
                c["confidence"] = min(0.6, c.get("confidence", 0.6))
            else:
                continue
        out.append({
            "content": content,
            "suggested_type": c.get("suggested_type", "CM"),
            "importance": float(c.get("importance", 0.5)),
            "confidence": float(c.get("confidence", 0.5)),
            "reason": c.get("reason", "extracted")
        })
    return out

# ---------------------------
# Promotion decision (STM->CM->LTM)
# ---------------------------
def decide_storage_for_candidate(candidate: Dict[str, Any]) -> str:
    """
    Candidate contains: content, suggested_type, importance, confidence
    Return: target_type in {STM, CM, LTM, EM}
    """
    suggested = candidate.get("suggested_type", "CM")
    importance = clamp(float(candidate.get("importance", 0.5)))
    confidence = clamp(float(candidate.get("confidence", 0.5)))
    # compute initial memory_score with created_at = now and frequency=1
    mem_score = compute_memory_score(importance, 1, now_ts(), confidence)
    candidate["memory_score"] = mem_score
    # explicit time sensitive detection (very naive)
    if re.search(r"\btoday\b|\btomorrow\b|\bon\b\s+\w+\s+\d{1,2}\b", candidate["content"], flags=re.I):
        return "EM"
    # if explicit suggestion LTM
    if suggested == "LTM" and mem_score >= PROMOTE_TO_LTM_SCORE and confidence >= 0.8:
        return "LTM"
    if mem_score >= PROMOTE_TO_CM_SCORE or importance >= 0.8 or confidence >= CONFIDENCE_PROMOTE_THRESHOLD:
        return "CM"
    # fallback to STM
    return "STM"

# ---------------------------
# Inject memories into prompt context
# ---------------------------
def retrieve_relevant_memories(user_input: str, max_tokens_budget: int = MAX_INJECT_TOKENS) -> List[Dict[str, Any]]:
    """
    1) Query TF-IDF recall to get top candidate mem_ids
    2) Fetch from DB, filter by memory_score threshold
    3) Rank by memory_score * similarity (approx)
    4) Return list of memory dicts (id, content, why_matched)
    """
    refresh_recall_index()  # keep simple: rebuild index (cheap for small corpuses)
    recs = _recall_instance.retrieve(user_input, k=TFIDF_TOP_K)
    out = []
    token_budget = max_tokens_budget
    for mem_id, content, sim in recs:
        mem = get_memory(mem_id)
        if not mem:
            continue
        # discard expired EM
        if mem["type"] == "EM" and mem.get("expires_at"):
            exp = parse_ts(mem.get("expires_at"))
            if exp and exp < datetime.utcnow():
                continue
        # simple filter: memory_score threshold
        if (mem.get("memory_score") or 0.0) < 0.2:
            continue
        snippet = mem.get("content", "")
        # approximate token count by characters/4
        approx_tokens = max(1, int(len(snippet) / 4))
        if token_budget - approx_tokens < 0:
            break
        token_budget -= approx_tokens
        out.append({
            "id": mem_id,
            "content": snippet,
            "type": mem.get("type"),
            "memory_score": mem.get("memory_score"),
            "why_matched": f"tfidf_sim={sim:.3f}"
        })
    return out

def build_memory_injection_block(memories: List[Dict[str, Any]]) -> str:
    if not memories:
        return ""
    parts = ["-- INJECTED MEMORIES (distilled) --"]
    for m in memories:
        # keep injection compact
        parts.append(f"[{m['id']}] ({m['type']}) {m['content']}  -- reason: {m['why_matched']}")
    parts.append("-- END INJECTED MEMORIES --\n")
    return "\n".join(parts)

# ---------------------------
# phase_4.ask orchestration
# ---------------------------
def phase4_ask(user_input: str,
               session_id: Optional[str] = None,
               session_stm: Optional[Dict[str, Any]] = None,
               *,
               persona: Optional[str] = None,
               mode: Optional[str] = None,
               temperature: Optional[float] = None,
               max_tokens: Optional[int] = None,
               stream: bool = False,
               timeout: int = 30,
               **_kwargs) -> Dict[str, Any]:
    """
    Main entrypoint. Returns a dict:
    {
      "answer": str,
      "explain": [ {id, content, why} ],
      "memory_actions": [ {id, action, detail} ],
      "meta": { latency_ms, fallback, notes }
    }
    """
    start = time.time()
    if phase3_ask is None:
        return {"answer": "[phase_3 missing] Core unavailable.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True}}

    # 0. session STM injection (small)
    stm_block = ""
    if session_stm:
        # distilled STM bullets
        stm_bullets = []
        for k, v in session_stm.items():
            stm_bullets.append(f"{k}: {v}")
        stm_block = "-- STM --\n" + "\n".join(stm_bullets) + "\n-- END STM --\n\n"

    # 1. Retrieve relevant CM/LTM memories
    top_memories = retrieve_relevant_memories(user_input, max_tokens_budget=MAX_INJECT_TOKENS)
    injection = build_memory_injection_block(top_memories)

    # 2. Build final prompt context
    prompt_parts = []
    if persona:
        prompt_parts.append(f"[Persona: {persona}]")
    prompt_parts.append(stm_block)
    prompt_parts.append(injection)
    # safety/policy system instruction
    prompt_parts.append("[System: Use user preferences and memory to answer. Do not invent personal data.]")
    prompt_parts.append("\nUser: " + user_input)
    final_prompt = "\n\n".join([p for p in prompt_parts if p])

    # 3. Call phase_3.ask (reasoning + RAG wrapper)
    try:
        phase3_result = phase3_ask(final_prompt, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
    except Exception as e:
        # fallback: call phase_3 with raw user_input
        try:
            phase3_result = phase3_ask(user_input, persona=persona, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, timeout=timeout)
            fallback = True
        except Exception as e2:
            return {"answer": "ZULTX error: reasoning core failed.", "explain": [], "memory_actions": [], "meta": {"latency_ms": int((time.time()-start)*1000), "fallback": True, "error": str(e2)}}
    else:
        fallback = False

    # phase3_result may be string or structured; normalize
    answer_text = phase3_result if isinstance(phase3_result, str) else phase3_result.get("answer") or str(phase3_result)

    # 4. Mark used memories (increase frequency + last_used)
    used_ids = []
    for m in top_memories:
        try:
            update_memory_last_used(m["id"])
            used_ids.append(m["id"])
        except Exception:
            pass

    # 5. Extract candidate memories (rules + optional LLM)
    candidates = extract_candidates(user_input, answer_text)

    # 6. Score & decide writes
    memory_actions = []
    for cand in candidates:
        # initial fields
        cand_content = cand["content"]
        allowed, block_reason = policy_allow_store(cand_content)
        if not allowed:
            # if blocked, skip and log
            audit("blocked_candidate", None, {"content": cand_content[:200], "reason": block_reason})
            continue
        target = decide_storage_for_candidate(cand)
        mem_id = uuid4()
        created_at = now_ts()
        mem_obj = {
            "id": mem_id,
            "type": target,
            "content": cand_content,
            "source": "extractor",
            "raw_snippet": cand_content[:800],
            "created_at": created_at,
            "last_used": created_at,
            "expires_at": None,
            "confidence": float(cand.get("confidence", 0.5)),
            "importance": float(cand.get("importance", 0.5)),
            "frequency": 1,
            "memory_score": float(cand.get("memory_score", compute_memory_score(float(cand.get("importance", 0.5)), 1, created_at, float(cand.get("confidence", 0.5))))),
            "tags": [cand.get("reason", "auto")],
            "consent": True,
            "metadata": {"origin": "phase_4_extractor"}
        }
        # If EM, set a default expiry (e.g., 7 days) unless time explicitly longer
        if target == "EM":
            expires = datetime.utcnow() + timedelta(days=7)
            mem_obj["expires_at"] = expires.isoformat()
        # transactional write (SQLite single op is atomic)
        try:
            insert_memory(mem_obj)
            memory_actions.append({"id": mem_id, "action": "created", "type": target, "summary": mem_obj["content"][:140]})
        except Exception as e:
            # queue for retry
            enqueue_retry({"op": "insert_memory", "candidate": mem_obj})
            memory_actions.append({"id": None, "action": "queued", "detail": str(e)})
            audit("write_failed", None, {"error": str(e)})
    # 7. Return structured response
    meta = {
        "latency_ms": int((time.time() - start) * 1000),
        "fallback": fallback,
        "used_memory_count": len(used_ids),
        "candidate_count": len(candidates)
    }
    explain = []
    for m in top_memories:
        explain.append({"id": m["id"], "content": m["content"], "why": m["why_matched"], "type": m["type"], "score": m.get("memory_score")})
    return {
        "answer": answer_text,
        "explain": explain,
        "memory_actions": memory_actions,
        "meta": meta
    }

# ---------------------------
# Small CLI for manual testing
# ---------------------------
if __name__ == "__main__":
    print("ZULTX phase_4 local tester")
    print("Initializing recall index...")
    refresh_recall_index()
    while True:
        try:
            ui = input("\nYou: ").strip()
            if ui.lower() in ("exit", "quit"):
                break
            if ui.lower().startswith("listmem"):
                rows = list_memories(50)
                for r in rows:
                    print(f"{r['id'][:8]} {r['type']} score={r['memory_score']:.3f} freq={r['frequency']} content={r['content'][:80]}")
                continue
            if ui.lower().startswith("forget "):
                # simple forget by tag or id
                target = ui.split(" ", 1)[1].strip()
                # if id exists
                mem = get_memory(target)
                if mem:
                    with _db_lock:
                        c = get_db_conn().cursor()
                        c.execute("DELETE FROM memories WHERE id = ?", (target,))
                        get_db_conn().commit()
                    print("Deleted memory", target)
                    continue
                print("No memory by that id; use listmem to inspect")
                continue

            res = phase4_ask(ui, session_id="cli_test")
            print("\nZultX:", res["answer"])
            if res.get("explain"):
                print("\nUsed memories:")
                for e in res["explain"]:
                    print(f"- {e['id'][:8]} [{e.get('type')}] {e['content'][:120]} (why:{e['why']})")
            if res.get("memory_actions"):
                print("\nMemory actions:")
                for a in res["memory_actions"]:
                    print("-", a)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print("Error:", e)
