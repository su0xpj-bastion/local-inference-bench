"""A local inference engine stand-in that speaks the OpenAI Chat Completions wire format.

It models the things that matter when you serve an open-weight model on one machine:
prefill cost that grows with prompt length, a per-token decode cost that grows with the
number of active sequences, a fixed number of parallel slots, a prefix cache, a cold load
on the first request per model, a context window, and two quantizations of one model.

Faults (comma-separated, via --faults or MOCK_FAULTS) break one of those on purpose:

  slow_prefill        prefill 4x slower
  decode_throttle     decode slows down as the engine heats up (thermal throttling)
  no_prefix_cache     every prompt is prefilled from scratch
  serialized          one slot: concurrent requests queue (an engine started with parallel=1)
  quant_degraded      the q4 variant answers ~25% of quality items wrong
  nondeterministic    T=0 + seed output varies between calls
  context_truncation  prompts beyond 2,048 tokens are silently cut (wrong context-length setting)
  keep_alive_zero     the model is unloaded after every request, so every request is a cold start
  usage_missing       no usage chunk even with include_usage
  ignore_max_tokens   generation ignores max_tokens

Conditions (not faults: a correct harness must still pass with them on):

  dense_tokenizer     ~5.5 characters per token instead of 4, like Qwen 2.5 on English filler
                      (reproduces INC-002 in docs/ops: a fixed 4-chars/token estimate false-alarms)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from localbench.quality import ANSWERS

MODELS = {"gemma-mock-1b-q8": {"quant": "q8"}, "gemma-mock-1b-q4": {"quant": "q4"}}
CONTEXT_WINDOW = 8192
TRUNCATE_AT = 2048
SLOTS = 4
PREFILL_S_PER_TOKEN = 0.00005      # 20,000 tok/s prefill
DECODE_S_PER_TOKEN = 0.004         # 250 tok/s decode, one sequence
BATCH_PENALTY = 0.15               # each extra active sequence slows every step by 15%
COLD_LOAD_S = 0.25
WORDS = ("the", "model", "runs", "locally", "on", "one", "machine", "and", "streams", "tokens",
         "quickly", "while", "the", "cache", "keeps", "the", "prompt", "warm")


DENSE_CHARS_PER_TOKEN = 5.5


def ntokens(text: str, chars_per_token: float = 4) -> int:
    return max(1, int(len(text) // chars_per_token))


class Engine:
    def __init__(self, faults: set[str]):
        self.faults = faults
        self.cpt = DENSE_CHARS_PER_TOKEN if "dense_tokenizer" in faults else 4
        self.lock = threading.Lock()
        self.slots = threading.Semaphore(1 if "serialized" in faults else SLOTS)
        self.active = 0
        self.loaded: set[str] = set()
        self.cache: dict[str, set[str]] = {m: set() for m in MODELS}
        self.decoded = 0                     # total tokens decoded, drives throttling
        self.calls = 0

    # ---- timing model -------------------------------------------------------------
    def prefill_seconds(self, model: str, messages: list[dict]) -> tuple[float, int, int]:
        """Return (seconds, prompt_tokens, cached_tokens). Cache unit = one message prefix."""
        total, cached, key = 0, 0, ""
        cache = self.cache[model]
        hit_run = "no_prefix_cache" not in self.faults
        for m in messages:
            t = ntokens(m.get("content") or "", self.cpt)
            key = hashlib.sha1((key + m["role"] + (m.get("content") or "")).encode()).hexdigest()
            if hit_run and key in cache:
                cached += t
            else:
                hit_run = False
            cache.add(key)
            total += t
        rate = PREFILL_S_PER_TOKEN * (4 if "slow_prefill" in self.faults else 1)
        return (total - cached) * rate, total, cached

    def step_seconds(self) -> float:
        s = DECODE_S_PER_TOKEN * (1 + BATCH_PENALTY * max(0, self.active - 1))
        if "decode_throttle" in self.faults:
            s *= 1 + min(3.0, self.decoded / 600)
        return s

    # ---- content ------------------------------------------------------------------
    def answer(self, model: str, prompt: str, seed: int | None, temperature: float) -> str | None:
        for q, a in ANSWERS.items():
            if q in prompt:
                wrong = ("quant_degraded" in self.faults and MODELS[model]["quant"] == "q4"
                         and int(hashlib.sha1(q.encode()).hexdigest(), 16) % 4 == 0)
                return f"{a}1" if wrong else a
        if "NEEDLE" in prompt and "secret code" in prompt:
            code = prompt.split("The secret code is ")[1].split(".")[0]
            return code
        return None

    def filler(self, prompt: str, n: int, seed: int | None, temperature: float) -> list[str]:
        base = int(hashlib.sha1(prompt.encode()).hexdigest(), 16)
        if temperature > 0 or seed is None or "nondeterministic" in self.faults:
            base ^= random.getrandbits(32)
        rng = random.Random(base)
        return [rng.choice(WORDS) + " " for _ in range(n)]


class Handler(BaseHTTPRequestHandler):
    engine: Engine
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code: int, body: dict) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path.rstrip("/") == "/v1/models":
            return self._json(200, {"object": "list", "data": [
                {"id": m, "object": "model", "owned_by": "mock"} for m in MODELS]})
        if self.path == "/health":
            return self._json(200, {"status": "ok"})
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self):
        if self.path.rstrip("/") != "/v1/chat/completions":
            return self._json(404, {"error": {"message": "not found"}})
        req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        eng = self.engine
        model = req.get("model")
        if model not in MODELS:
            return self._json(404, {"error": {"message": f"model '{model}' not found", "type": "not_found"}})
        messages = req.get("messages") or []
        if "context_truncation" in eng.faults:
            messages = _truncate(messages, TRUNCATE_AT, eng.cpt)
        prompt = "\n".join(m.get("content") or "" for m in messages)
        if ntokens(prompt, eng.cpt) > CONTEXT_WINDOW:
            return self._json(400, {"error": {"message": "prompt exceeds context window", "type": "invalid_request_error"}})
        max_tokens = req.get("max_tokens") or req.get("max_completion_tokens") or 64
        temperature = float(req.get("temperature", 1.0))
        seed = req.get("seed")
        stream = bool(req.get("stream"))
        include_usage = bool((req.get("stream_options") or {}).get("include_usage"))

        with eng.slots:
            with eng.lock:
                eng.calls += 1
                cold = model not in eng.loaded
                eng.loaded.add(model)
                eng.active += 1
                prefill, prompt_tokens, cached = eng.prefill_seconds(model, messages)
            try:
                time.sleep((COLD_LOAD_S if cold else 0) + prefill)
                fixed = eng.answer(model, prompt, seed, temperature)
                n = max_tokens if fixed is None else 1
                if "ignore_max_tokens" in eng.faults:
                    n = max_tokens * 3
                pieces = [fixed] if fixed is not None else eng.filler(prompt, n, seed, temperature)
                finish = "length" if fixed is None and len(pieces) >= max_tokens else "stop"
                if "ignore_max_tokens" in eng.faults:
                    finish = "stop"
                usage = {"prompt_tokens": prompt_tokens, "completion_tokens": len(pieces),
                         "total_tokens": prompt_tokens + len(pieces),
                         "prompt_tokens_details": {"cached_tokens": cached}}
                cid = f"chatcmpl-{eng.calls}"
                if not stream:
                    for _ in pieces:
                        time.sleep(eng.step_seconds())
                    self._json(200, {"id": cid, "object": "chat.completion", "created": int(time.time()),
                                     "model": model, "choices": [{"index": 0, "finish_reason": finish,
                                     "message": {"role": "assistant", "content": "".join(pieces)}}],
                                     "usage": usage})
                else:
                    self._stream(cid, model, pieces, finish, usage, include_usage)
            finally:
                with eng.lock:
                    eng.active -= 1
                    if "keep_alive_zero" in eng.faults:
                        eng.loaded.discard(model)

    def _stream(self, cid, model, pieces, finish, usage, include_usage):
        eng = self.engine
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            data = f"data: {json.dumps(obj) if isinstance(obj, dict) else obj}\n\n".encode()
            self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        base = {"id": cid, "object": "chat.completion.chunk", "created": int(time.time()), "model": model}
        send({**base, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}, "finish_reason": None}]})
        for p in pieces:
            time.sleep(eng.step_seconds())
            with eng.lock:
                eng.decoded += 1
            send({**base, "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]})
        send({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        if include_usage and "usage_missing" not in eng.faults:
            send({**base, "choices": [], "usage": usage})
        send("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def _truncate(messages: list[dict], limit: int, chars_per_token: float = 4) -> list[dict]:
    """Keep the LAST `limit` tokens, the way a too-small context setting silently drops the start."""
    out, budget = [], limit
    for m in reversed(messages):
        c = m.get("content") or ""
        if ntokens(c, chars_per_token) <= budget:
            out.append(m)
            budget -= ntokens(c, chars_per_token)
        else:
            out.append({**m, "content": c[-int(budget * chars_per_token):]})
            break
    return list(reversed(out))


class QuietServer(ThreadingHTTPServer):
    def handle_error(self, request, client_address):
        pass                               # clients hanging up mid-stream are expected


def serve(port: int = 0, faults: set[str] | None = None) -> tuple[ThreadingHTTPServer, Engine]:
    engine = Engine(faults or set())
    handler = type("BoundHandler", (Handler,), {"engine": engine})
    server = QuietServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, engine


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--faults", default=os.environ.get("MOCK_FAULTS", ""))
    a = ap.parse_args()
    server, _ = serve(a.port, {f for f in a.faults.split(",") if f})
    print(f"mock engine on http://127.0.0.1:{server.server_port}/v1  faults={a.faults or 'none'}")
    try:
        threading.Event().wait()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
