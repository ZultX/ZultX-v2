# app.py
import os
import glob
import json
import time
import traceback
import urllib.parse
import asyncio
from typing import Generator, Union, Optional, AsyncGenerator

from fastapi import FastAPI, Query, Body, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse, HTMLResponse, FileResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Try optional external ask functions (phase_2 / phase_1).
ASK_FUNC = None
try:
    from phase_2 import ask as ask_func
    ASK_FUNC = ask_func
    print("[ZULTX] Using phase_2.ask")
except Exception:
    try:
        from phase_1 import ask as ask_func
        ASK_FUNC = ask_func
        print("[ZULTX] Using phase_1.ask")
    except Exception as e:
        print("[ZULTX] No phase_1/phase_2 ask() found, using internal fallback. Error:", e)
        ASK_FUNC = None

# ensure directories
LETTERS_DIR = os.getenv("ZULTX_LETTERS_DIR", "letters")
os.makedirs(LETTERS_DIR, exist_ok=True)
os.makedirs("feedback", exist_ok=True)
os.makedirs("tips", exist_ok=True)

# Default UPI (safe placeholder)
UPI_ID = os.getenv("UPI_ID", "9358588509@fam")

app = FastAPI(title="ZULTX — v1.1")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

if os.path.isdir("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists("index.html"):
        return FileResponse("index.html")
    return HTMLResponse("<html><body><h1>ZultX</h1><p>UI missing</p></body></html>")


# SSE helper: always output `data: {...}\n\n` strings
def sse_format(obj: dict) -> str:
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n"


# Internal fallback generator (async) to stream a string answer in chunks
async def async_chunk_text_as_sse(text: str, chunk_size: int = 24, delay: float = 0.03) -> AsyncGenerator[str, None]:
    # start role event (optional)
    yield sse_format({"choices": [{"delta": {"role": "assistant", "content": ""}}]})
    i = 0
    n = len(text)
    while i < n:
        chunk = text[i: i + chunk_size]
        payload = {"choices": [{"delta": {"content": chunk}}]}
        yield sse_format(payload)
        await asyncio.sleep(delay)
        i += chunk_size
    # finalization event
    yield sse_format({"choices": [{"delta": {}},], "finish_reason": "stop"})
    yield "data: [DONE]\n\n"


# If external ASK_FUNC is missing, provide a simple fallback ask() that returns helpful text
def local_fallback_ask(user_input: str, mode: str = "friend", temperature: Optional[float] = None,
                       max_tokens: int = 512, stream: bool = False, speed: float = 0.02):
    # A friendly response template (improves "I m sad" example)
    base = user_input.strip().lower()
    if "sad" in base or "depressed" in base or "unhappy" in base:
        answer = ("Hey — I'm really sorry you're feeling sad. "
                  "You're not alone. If you'd like, tell me what's going on and I will listen. "
                  "Here are three quick things you can try right now:\n\n"
                  "1) Take three slow deep breaths (inhale 4s, hold 2s, exhale 6s).\n"
                  "2) Stand up and stretch for 30 seconds.\n"
                  "3) Write one small thing you're grateful for.\n\n"
                  "If you want, share more — I'm here to help you think through it.")
    else:
        answer = f"Hey. I heard: \"{user_input}\". I'd say: be kind to yourself — tell me more and I'll help."

    # If streaming requested, return an async generator
    if stream:
        async def gen():
            async for s in async_chunk_text_as_sse(answer, chunk_size=24, delay=speed):
                yield s
        return gen()
    return answer


# Helper that converts result (string / generator / async generator) into an SSE async generator of strings
async def to_sse_async_generator(result: Union[str, Generator[str, None, None], AsyncGenerator[str, None], None],
                                 stream_speed: float = 0.02) -> AsyncGenerator[str, None]:
    # If result is None, use fallback
    if result is None:
        result = local_fallback_ask("I have nothing to say", stream=False)

    # Async generator returned directly (assume it yields plain text fragments OR already sse-formatted strings)
    if hasattr(result, "__aiter__"):
        async for part in result:  # parts may be plain text fragments
            # If already looks like sse-formatted (starts with "data:") pass through; else wrap it.
            if isinstance(part, str) and part.strip().startswith("data:"):
                yield part
            else:
                yield sse_format({"choices": [{"delta": {"content": str(part)}}]})
        yield "data: [DONE]\n\n"
        return

    # Sync iterator
    if hasattr(result, "__iter__") and not isinstance(result, str):
        for part in result:
            yield sse_format({"choices": [{"delta": {"content": str(part)}}]})
            await asyncio.sleep(stream_speed)
        yield "data: [DONE]\n\n"
        return

    # Plain string
    text = str(result or "")
    async for s in async_chunk_text_as_sse(text, chunk_size=24, delay=stream_speed):
        yield s


@app.get("/ask")
async def ask_get(
    q: str = Query(..., alias="q"),
    mode: str = Query("friend"),
    stream: bool = Query(True),
    speed: float = Query(0.02),
    temperature: Optional[float] = Query(None),
    max_tokens: int = Query(512)
):
    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Missing query")

    # Try to call imported ASK_FUNC if available
    if ASK_FUNC:
        try:
            result = ASK_FUNC(user_input=q, mode=mode, temperature=temperature, max_tokens=max_tokens, stream=stream, speed=speed)
        except Exception as e:
            traceback.print_exc()
            return JSONResponse({"error": "ZULTX processing failed", "detail": str(e)}, status_code=500)

        # If streaming requested, return SSE stream
        if stream:
            # If ASK_FUNC already returned an async generator, wrap it (safely)
            try:
                return StreamingResponse(to_sse_async_generator(result, stream_speed=speed), media_type="text/event-stream")
            except Exception:
                # fall back to fake stream
                return StreamingResponse(to_sse_async_generator(local_fallback_ask(q, stream=True, speed=speed), stream_speed=speed),
                                         media_type="text/event-stream")
        else:
            # Non-streaming response: unify to JSON { answer: text }
            try:
                if hasattr(result, "__aiter__"):
                    # consume async generator
                    parts = []
                    async for p in result:
                        parts.append(str(p))
                    text = "".join(parts)
                elif hasattr(result, "__iter__") and not isinstance(result, str):
                    text = "".join([str(p) for p in result])
                else:
                    text = str(result)
            except Exception:
                text = str(result)
            return JSONResponse({"answer": text})

    # If ASK_FUNC not present — use local fallback
    fallback_text = local_fallback_ask(q, stream=False)
    if stream:
        return StreamingResponse(to_sse_async_generator(local_fallback_ask(q, stream=True, speed=speed), stream_speed=speed),
                                 media_type="text/event-stream")
    else:
        return JSONResponse({"answer": fallback_text})


@app.get("/letters")
def list_letters():
    files = sorted([os.path.basename(p) for p in glob.glob(os.path.join(LETTERS_DIR, "*.txt"))])
    return JSONResponse({"letters": files})


@app.get("/letters/{name}")
def get_letter(name: str):
    if ".." in name or not name.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    path = os.path.join(LETTERS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Letter not found")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    return PlainTextResponse(content)


@app.get("/profile")
def profile():
    return JSONResponse({
        "username": "Guest",
        "display_name": "Guest",
        "email": None,
        "avatar": None,
        "can_logout": True
    })


@app.post("/feedback")
def feedback(payload: dict = Body(...)):
    ts = int(time.time() * 1000)
    fname = f"feedback/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})


@app.post("/tip")
def tip(payload: dict = Body(...)):
    try:
        amount = int(payload.get("amount", 10))
        if amount <= 0:
            raise ValueError()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid amount")

    ts = int(time.time())
    tn = urllib.parse.quote_plus("Tip ZULTX")
    upi_uri = f"upi://pay?pa={urllib.parse.quote_plus(UPI_ID)}&pn=ZULTX&tn={tn}&am={amount}&cu=INR"
    qr_payload = urllib.parse.quote_plus(upi_uri)
    qr_url = f"https://chart.googleapis.com/chart?cht=qr&chs=360x360&chl={qr_payload}"

    order = {"id": f"upi_{ts}", "amount": amount * 100, "currency": "INR"}
    try:
        with open(f"tips/{ts}.json", "w", encoding="utf-8") as f:
            json.dump({"order": order, "upi": UPI_ID, "upi_uri": upi_uri, "created_at": ts}, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    return JSONResponse({"ok": True, "order": order, "upi_link": upi_uri, "qr": qr_url, "note": "Use UPI link or scan QR to pay."})


@app.post("/tip/confirm")
def tip_confirm(payload: dict = Body(...)):
    ts = int(time.time() * 1000)
    fname = f"tips/{ts}.json"
    with open(fname, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return JSONResponse({"ok": True})


@app.get("/health")
def health():
    return JSONResponse({"status": "ok"})
