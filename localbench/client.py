"""One streamed chat completion, timed per content chunk, against any OpenAI-compatible server.

Ollama (`/v1`), llama.cpp's `llama-server`, `mlx_lm.server` and vLLM all expose the same
Chat Completions route, so one client covers every local engine.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field

import openai

MAX_ATTEMPTS = 3


@dataclass
class Call:
    content: str = ""
    finish_reason: str | None = None
    ttft: float | None = None              # request start -> first content chunk (s)
    wall: float = 0.0                      # request start -> end of stream (s)
    gaps: list[float] = field(default_factory=list)   # time between content chunks (s)
    chunks: int = 0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int = 0
    usage_seen: bool = False

    @property
    def decode_tps(self) -> float | None:
        """Tokens per second after the first token. Uses usage when present, else chunk count."""
        n = (self.completion_tokens or self.chunks) - 1
        span = self.wall - (self.ttft or 0)
        return n / span if n > 0 and span > 0 else None


def make_client(base_url: str, api_key: str = "local", timeout: float = 300.0) -> openai.OpenAI:
    # Local engines ignore the key, but the SDK requires one. Retries are ours, not the SDK's,
    # so that a retried request never hides inside a latency number.
    return openai.OpenAI(base_url=base_url, api_key=api_key, max_retries=0, timeout=timeout)


def chat(client: openai.OpenAI, model: str, messages: list[dict] | str, *, max_tokens: int = 64,
         temperature: float = 0.0, seed: int | None = 7) -> Call:
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    for attempt in range(MAX_ATTEMPTS):
        call = Call()
        last = start = time.perf_counter()
        try:
            with client.chat.completions.create(
                model=model, messages=messages, max_tokens=max_tokens, temperature=temperature,
                seed=seed, stream=True, stream_options={"include_usage": True},
            ) as stream:
                for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        now = time.perf_counter()
                        if call.ttft is None:
                            call.ttft = now - start
                        else:
                            call.gaps.append(now - last)
                        last = now
                        call.chunks += 1
                        call.content += chunk.choices[0].delta.content
                    if chunk.choices and chunk.choices[0].finish_reason:
                        call.finish_reason = chunk.choices[0].finish_reason
                    if chunk.usage:
                        call.usage_seen = True
                        call.prompt_tokens = chunk.usage.prompt_tokens
                        call.completion_tokens = chunk.usage.completion_tokens
                        details = chunk.usage.prompt_tokens_details
                        call.cached_tokens = (details.cached_tokens or 0) if details else 0
            call.wall = time.perf_counter() - start
            return call
        except openai.RateLimitError:
            if attempt == MAX_ATTEMPTS - 1:
                raise
            time.sleep(2 ** attempt + random.random())
        except openai.APIStatusError:
            raise
        except openai.APIError:
            raise
    raise RuntimeError("unreachable")
