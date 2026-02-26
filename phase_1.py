# phase_1.py (GOD-GOD-TIER focused, mistral-first router)
from __future__ import annotations
import os
import time
import json
import threading
import traceback
from typing import Generator, Optional, List, Union, Callable, Dict, Any
import requests
import base64

# --------------------
# Config (env-friendly)
# --------------------
MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
OPENAI_KEY = os.getenv("OPENAI_KEY")    # optional fallback
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")  # optional (Gemini) - placeholder

DEFAULT_TIMEOUT = int(os.getenv("PHASE1_DEFAULT_TIMEOUT", "30"))
MAX_ATTEMPTS = int(os.getenv("PHASE1_MAX_ATTEMPTS", "2"))
BACKOFF_BASE = float(os.getenv("PHASE1_BACKOFF_BASE", "0"))
RATE_PER_SEC = float(os.getenv("PHASE1_RATE_PER_SEC", "1.0"))
RATE_CAPACITY = float(os.getenv("PHASE1_RATE_CAPACITY", "5.0"))

# Observability hooks
metrics_hook: Optional[Callable[[str, Dict], None]] = None
health_check_hook: Optional[Callable[[], Dict[str, bool]]] = None

def _emit_metric(event: str, meta: Dict = None):
    try:
        if metrics_hook:
            metrics_hook(event, meta or {})
        else:
            print(f"[phase1][metric] {event} {json.dumps(meta or {}, default=str)}")
    except Exception:
        pass

# --------------------
# Exceptions
# --------------------
class ModelFailure(Exception):
    pass

# --------------------
# Utilities
# --------------------
def safe_json_load(s: str):
    try:
        return json.loads(s)
    except Exception:
        return None

def now_s():
    return time.time()

# --------------------
# Intent & Complexity Detection
# --------------------
def detect_intent(prompt: str) -> str:
    txt = (prompt or "").strip().lower()
    if not txt:
        return "small"

    if any(w in txt for w in ["embed", "embedding", "vectorize", "vector"]):
        return "embed"
    if any(w in txt for w in ["image", "photo", "vision", ".png", ".jpg"]):
        return "multimodal"

    heavy_keywords = [
        "design", "architecture", "implement", "optimize",
        "debug", "proof", "analysis", "explain",
        "why", "how", "step by step", "derive",
        "prove", "does", "can", "do", "what", "will"
    ]
    if any(word in txt for word in heavy_keywords):
        return "heavy"
    if len(txt) > 400:
        return "long"
    return "small"

def detect_complexity(prompt: str) -> str:
    intent = detect_intent(prompt)
    if intent == "heavy":
        return "heavy"
    if intent == "long":
        return "normal"
    if intent in ("multimodal", "embed"):
        return intent
    return "fast"

# --------------------
# Simple Token Bucket (per-adapter)
# --------------------
class TokenBucket:
    def __init__(self, rate_per_sec: float = RATE_PER_SEC, capacity: float = RATE_CAPACITY):
        self.rate = float(rate_per_sec)
        self.capacity = float(capacity)
        self._tokens = float(capacity)
        self._last = now_s()
        self._lock = threading.Lock()

    def consume(self, amount: float = 1.0) -> bool:
        with self._lock:
            now = now_s()
            delta = now - self._last
            self._last = now
            self._tokens = min(self.capacity, self._tokens + delta * self.rate)
            if self._tokens >= amount:
                self._tokens -= amount
                return True
            return False

# --------------------
# Adapter Base
# --------------------
class ModelAdapter:
    name = "adapter"
    supports_stream = False

    def __init__(self):
        self.bucket = TokenBucket()
        self.last_health = 0.0
        self._healthy = True

    def check_ready(self) -> bool:
        return True

    def health(self) -> bool:
        return self._healthy

    def mark_unhealthy(self):
        self._healthy = False
        self.last_health = now_s()

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
        raise NotImplementedError

# --------------------
# Mistral Direct Adapter (primary streaming & heavy)
# --------------------
class MistralAdapter(ModelAdapter):
    name = "mistral-direct"
    supports_stream = True

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.key = api_key or MISTRAL_KEY
        self.endpoint = "https://api.mistral.ai/v1/chat/completions"

    def check_ready(self):
        return bool(self.key)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("mistral-key-missing")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")

        payload = {"model": "mistral-large-latest", "messages": [{"role": "user", "content": prompt}], "stream": bool(stream)}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            resp.raise_for_status()
            if not stream:
                j = resp.json()
                return j.get("choices", [{}])[0].get("message", {}).get("content") or j.get("output", {}).get("text", "") or ""
            # streaming generator
            def gen():
                for line in resp.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    line = line.strip()
                    # Handle SSE format
                    if line.startswith("data: "):
                        line = line[len("data: "):]

                    if line == "[DONE]":
                        break

                    try:
                       j = json.loads(line)
                       delta = j.get("choices", [{}])[0].get("delta", {}).get("content")
                       if delta:
                           yield delta
                    except Exception:
                       continue
            return gen()
        except Exception as e:
            self.mark_unhealthy()
            raise ModelFailure(f"mistral-error: {e}")
# --------------------
# Optional OpenAI Adapter (fallback)
# --------------------
class OpenAIAdapter(ModelAdapter):
    name = "openai"
    supports_stream = True

    def __init__(self, key: Optional[str]):
        super().__init__()
        self.key = key
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def check_ready(self):
        return bool(self.key)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("openai-missing")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")
        payload = {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "messages":[{"role":"user","content":prompt}], "stream": bool(stream)}
        headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
        try:
            r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout, stream=bool(stream))
            r.raise_for_status()
            if not stream:
                return r.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            def gen():
                for line in r.iter_lines(decode_unicode=True):
                    if not line: continue
                    s = line.strip()
                    if s.startswith("data: "): s = s[len("data: "):]
                    if s == "[DONE]": break
                    try:
                        obj = json.loads(s)
                        delta = obj.get("choices",[{}])[0].get("delta", {}).get("content")
                        if delta: yield delta
                    except Exception:
                        yield s
            return gen()
        except Exception as e:
            self.mark_unhealthy()
            raise ModelFailure(f"openai-error: {e}")

# --------------------
# Gemini Adapter placeholder (non-streaming) — uses GOOGLE_API_KEY
# --------------------
class GeminiAdapter(ModelAdapter):
    name = "gemini"
    supports_stream = False

    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.key = api_key or GOOGLE_API_KEY
        # Placeholder endpoint; integrate real Gemini API calls here.
        self.endpoint = "https://api.google.com/gemini-placeholder"

    def check_ready(self):
        return bool(self.key)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("gemini-missing")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")
        # Basic stub: implement Gemini call as you like
        # Return string (no streaming)
        return f"(gemini placeholder reply) {prompt[:240]}"

# --------------------
# Multimodal Adapters (image/whisper/tts)
# ImageGenAdapter keeps provider option (stability|openrouter|openai)
# --------------------
class ImageGenAdapter(ModelAdapter):
    name = "imagegen"
    supports_stream = False

    def __init__(self, provider: str = None, api_key: Optional[str] = None, model: Optional[str] = None):
        super().__init__()
        self.provider = (provider or os.getenv("IMAGE_PROVIDER", "openai")).lower()
        self.key = api_key
        self.model = model
        self._init_provider_settings()

    def _init_provider_settings(self):
        prov = self.provider
        if prov == "stability":
            self.key = self.key or os.getenv("STABILITY_KEY")
            self.endpoint = "https://api.stability.ai/v1/generation"
            self.engine = os.getenv("STABILITY_ENGINE", "stable-diffusion-v1-5")
        elif prov == "openrouter":
            # Keep openrouter here only for image generation if you want that path
            self.key = self.key or os.getenv("OPENROUTER_API_KEY")
            self.endpoint = os.getenv("OPENROUTER_IMAGE_ENDPOINT", "https://openrouter.ai/api/v1/images/generate")
            self.model = self.model or os.getenv("OPENROUTER_IMAGE_MODEL", "stability/stable-diffusion-v1")
        else:
            # default to openai images (or other provider)
            self.key = self.key or os.getenv("OPENAI_KEY")
            self.endpoint = "https://api.openai.com/v1/images/generations"
            self.model = self.model or os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1")

    def check_ready(self):
        return bool(self.key)

    def generate(self, prompt: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready():
            raise ModelFailure("imagegen-missing-key")
        if not self.bucket.consume():
            raise ModelFailure("rate_limited")
        try:
            prov = self.provider
            if prov == "stability":
                url = f"{self.endpoint}/{self.engine}/text-to-image"
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                payload = {"text_prompts":[{"text": prompt}], "width":512, "height":512}
                r = requests.post(url, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                b64 = j.get("artifacts",[{}])[0].get("base64")
                if b64:
                    return base64.b64decode(b64)
                return str(j)
            elif prov == "openrouter":
                headers = {
                  "Authorization": f"Bearer {self.key}",
                  "Content-Type": "application/json",
                }

                payload = {
                   "model": self.model,
                   "messages": [
                     {
                      "role": "user",
                      "content": prompt
                     }
                    ]
                }
                r = requests.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json=payload,
                    headers=headers,
                    timeout=timeout
                )
                r.raise_for_status()
                j = r.json()
                # Extract base64 image
                b64 = (
                    j.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", [{}])[0]
                    .get("image_url", {})
                    .get("url")
                )
                if b64 and b64.startswith("data:image"):
                    import re
                    encoded = re.sub("^data:image/.+;base64,", "", b64)
                    return base64.b64decode(encoded)
                return str(j)

            else:
                headers = {"Authorization": f"Bearer {self.key}", "Content-Type":"application/json"}
                payload = {"model": self.model, "prompt": prompt, "size":"1024x1024"}
                r = requests.post(self.endpoint, json=payload, headers=headers, timeout=timeout)
                r.raise_for_status()
                j = r.json()
                b64 = j.get("data",[{}])[0].get("b64_json") or j.get("data",[{}])[0].get("b64")
                if b64:
                    return base64.b64decode(b64)
                url = j.get("data",[{}])[0].get("url")
                if url:
                    return requests.get(url, timeout=timeout).content
                return str(j)
        except Exception as e:
            self.mark_unhealthy()
            raise ModelFailure(f"imagegen-error: {e}")

class WhisperASRAdapter(ModelAdapter):
    name = "whisper-asr"
    supports_stream = False
    def __init__(self, api_key: Optional[str] = None):
        super().__init__()
        self.key = api_key or os.getenv("OPENAI_KEY")
        self.endpoint = "https://api.openai.com/v1/audio/transcriptions"
    def check_ready(self): return bool(self.key)
    def generate(self, audio_bytes: bytes, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready(): raise ModelFailure("whisper-missing")
        if not self.bucket.consume(): raise ModelFailure("rate_limited")
        files = {"file": ("audio.wav", audio_bytes)}
        data = {"model": os.getenv("WHISPER_MODEL","whisper-1")}
        headers = {"Authorization": f"Bearer {self.key}"}
        r = requests.post(self.endpoint, headers=headers, files=files, data=data, timeout=timeout)
        r.raise_for_status()
        return r.json().get("text","")

class ElevenLabsTTSAdapter(ModelAdapter):
    name = "elevenlabs-tts"
    supports_stream = False
    def __init__(self, api_key: Optional[str] = None, voice: Optional[str] = None):
        super().__init__()
        self.key = api_key or os.getenv("ELEVENLABS_KEY")
        self.voice = voice or os.getenv("ELEVENLABS_VOICE", "alloy")
        self.endpoint_base = "https://api.elevenlabs.io/v1"
    def check_ready(self): return bool(self.key)
    def generate(self, text: str, stream: bool = False, timeout: int = DEFAULT_TIMEOUT):
        if not self.check_ready(): raise ModelFailure("elevenlabs-key-missing")
        if not self.bucket.consume(): raise ModelFailure("rate_limited")
        url = f"{self.endpoint_base}/text-to-speech/{self.voice}"
        headers = {"xi-api-key": self.key, "Content-Type":"application/json"}
        payload = {"text": text, "voice": self.voice, "model":"eleven_monolingual_v1"}
        r = requests.post(url, json=payload, headers=headers, timeout=timeout)
        r.raise_for_status()
        return r.content

# --------------------
# Router / Orchestration (mistral-first, stream-aware)
# --------------------
class ModelRouter:
    def __init__(self, adapters: List[ModelAdapter]):
        self.adapters = adapters

    def _candidates_for_intent(self, intent: str, complexity: str, stream: bool = False) -> List[ModelAdapter]:
        # Heavy -> strictly prefer Mistral (only Mistral if ready)
        mistral_adapters = [a for a in self.adapters if getattr(a, "name", "") == "mistral-direct"]
        stream_friendly = [a for a in self.adapters if getattr(a, "supports_stream", False)]
        if intent == "heavy":
            ready = [a for a in mistral_adapters if a.check_ready()]
            if ready:
                return ready
            # fallback to other stream-capable adapters if streaming, else other adapters
            if stream:
                return [a for a in stream_friendly if a.check_ready()]
            return [a for a in self.adapters if a.check_ready()]

        # multimodal: prefer imagegen / whisper route for multimodal work (non-chat)
        if intent == "multimodal":
            imgs = [a for a in self.adapters if getattr(a, "name", "") == "imagegen" and a.check_ready()]
            if imgs:
                return imgs

        # streaming requests: prefer stream-capable adapters, with mistral first if present and ready
        if stream:
            ordered = []
            if any(a.check_ready() and getattr(a, "name", "") == "mistral-direct" for a in self.adapters):
                ordered.append([a for a in self.adapters if getattr(a, "name","") == "mistral-direct"][0])
            # then other stream-capable adapters in deterministic order
            for a in stream_friendly:
                if a not in ordered and a.check_ready():
                    ordered.append(a)
            if ordered:
                return ordered

        # default routing for non-stream chat
        # prefer LLMs in order: mistral, openai (if present), gemini (if present), others
        priority_names = ["mistral-direct", "openai", "gemini"]
        ordered = []
        for pname in priority_names:
            for a in self.adapters:
                if pname in getattr(a, "name", "") and a.check_ready() and a not in ordered:
                    ordered.append(a)
        for a in self.adapters:
            if a.check_ready() and a not in ordered:
                ordered.append(a)
        return ordered

    def ask(self, prompt: Union[str, bytes], stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
        complexity = detect_complexity(prompt if isinstance(prompt, str) else str(prompt))
        intent = detect_intent(prompt if isinstance(prompt, str) else str(prompt))
        _emit_metric("route_selected", {"intent": intent, "complexity": complexity, "stream": stream, "time": now_s()})

        candidates = self._candidates_for_intent(intent, complexity, stream=stream)
        last_err = None
        for adapter in candidates:
            try:
                if not adapter.check_ready():
                    _emit_metric("adapter_skipped_not_ready", {"adapter": adapter.name})
                    continue
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    try:
                        _emit_metric("adapter_attempt", {"adapter": adapter.name, "attempt": attempt})
                        out = adapter.generate(prompt, stream=stream, timeout=timeout)
                        _emit_metric("adapter_success", {"adapter": adapter.name, "attempt": attempt})
                        return out
                    except ModelFailure as mf:
                        last_err = mf
                        if "rate_limited" in str(mf).lower():
                            _emit_metric("adapter_rate_limited", {"adapter": adapter.name})
                            break
                        backoff = BACKOFF_BASE * (2 ** (attempt - 1))
                        time.sleep(backoff)
                        continue
            except Exception as e:
                last_err = e
                _emit_metric("adapter_unexpected_error", {"adapter": getattr(adapter, "name", "unknown"), "err": str(e)})
                traceback.print_exc()
                continue

        msg = "ZULTX brain temporarily unavailable. Try again in a moment."
        _emit_metric("router_all_failed", {"err": str(last_err)})
        if stream:
            def g():
                for ch in msg:
                    yield ch
                    time.sleep(0.003)
            return g()
        return msg

# --------------------
# Bootstrap default router (mistral-first)
# --------------------
def build_default_router() -> ModelRouter:
    adapters: List[ModelAdapter] = []
    # 1) Primary: Mistral (stream-capable heavy reasoning)
    adapters.append(MistralAdapter(api_key=MISTRAL_KEY))
    # 2) Optional fallbacks: OpenAI, Gemini
    if OPENAI_KEY:
        adapters.append(OpenAIAdapter(OPENAI_KEY))
    if GOOGLE_API_KEY:
        adapters.append(GeminiAdapter(GOOGLE_API_KEY))
    # 3) Multimodal: image, asr, tts
    adapters.append(ImageGenAdapter(provider=os.getenv("IMAGE_PROVIDER", "openai")))
    adapters.append(WhisperASRAdapter())
    adapters.append(ElevenLabsTTSAdapter())
    return ModelRouter(adapters)

# global singleton
_router = build_default_router()

# Public API: ask()
def ask(prompt: Union[str, bytes], *, stream: bool = False, timeout: int = DEFAULT_TIMEOUT) -> Union[str, Generator[str, None, None]]:
    return _router.ask(prompt, stream=stream, timeout=timeout)

# Helpers & hooks
def get_adapters_status() -> List[Dict]:
    out = []
    for a in _router.adapters:
        out.append({
            "name": getattr(a, "name", "unknown"),
            "ready": a.check_ready(),
            "healthy": a.health(),
            "last_health_ts": getattr(a, "last_health", None)
        })
    return out

def set_metrics_hook(fn: Optional[Callable[[str, Dict], None]]):
    global metrics_hook
    metrics_hook = fn

def set_health_hook(fn: Optional[Callable[[], Dict[str, bool]]]):
    global health_check_hook
    health_check_hook = fn
