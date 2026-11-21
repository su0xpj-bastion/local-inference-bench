"""The checks. Each returns a Check; each has a mock fault it must catch (tests/test_faults.py).

Order matters a little: decode_sustained runs before the long concurrency sweep so that a
thermally throttling engine is measured from a cool start.
"""
from __future__ import annotations

import concurrent.futures as cf
import statistics
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

import openai

from .client import Call, chat
from .quality import ITEMS, SYSTEM, correct

FILLER = ("Local inference keeps data on the machine, trades peak throughput for privacy, and depends "
          "on memory bandwidth more than on raw compute. ")


@dataclass
class Check:
    area: str
    name: str
    status: str                       # pass | fail | warn | skip | info
    detail: str
    gated: bool = True
    metrics: dict = field(default_factory=dict)


@dataclass
class Ctx:
    client: openai.OpenAI
    model: str
    model_b: str | None
    slo: dict
    calls: int = 0

    def chat(self, messages, **kw) -> Call:
        self.calls += 1
        return chat(self.client, self.model if "model" not in kw else kw.pop("model"), messages, **kw)


def pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p / 100 * (len(xs) - 1))))]


def text_of(tokens: int) -> str:
    """About `tokens` tokens of filler, prefixed with a nonce so no prefix cache can serve it."""
    reps = max(1, tokens * 4 // len(FILLER))
    return f"[{uuid.uuid4().hex}] " + FILLER * reps


def gate(ctx: Ctx, key: str) -> bool:
    return key not in ctx.slo.get("ungated", [])


def status(ok: bool, gated: bool) -> str:
    return "pass" if ok else ("fail" if gated else "warn")


# ---------------- contract ----------------

def usage_reported(ctx: Ctx) -> Check:
    c = ctx.chat("Write two sentences about local inference.", max_tokens=32)
    ok = c.usage_seen and (c.completion_tokens or 0) >= 1 and (c.prompt_tokens or 0) >= 1
    return Check("contract", "usage_reported", "pass" if ok else "fail",
                 f"usage chunk seen={c.usage_seen}, prompt={c.prompt_tokens}, completion={c.completion_tokens}, "
                 f"content chunks={c.chunks}")


def max_tokens_honoured(ctx: Ctx) -> Check:
    c = ctx.chat("Write a long story about a lighthouse.", max_tokens=16)
    n = c.completion_tokens if c.completion_tokens is not None else c.chunks
    ok = n <= 16 and c.finish_reason == "length"
    return Check("contract", "max_tokens_honoured", "pass" if ok else "fail",
                 f"max_tokens=16 -> {n} tokens, finish_reason={c.finish_reason}")


# ---------------- performance ----------------

def warm_ttft(ctx: Ctx) -> Check:
    calls = [ctx.chat("Say hello in five words.", max_tokens=8) for _ in range(ctx.slo["warm_samples"] + 1)]
    first, warm = calls[0].ttft or 0, [c.ttft or 0 for c in calls[1:]]
    p50, p95 = statistics.median(warm), pct(warm, 95)
    cap = ctx.slo["warm_ttft_p50_max_s"]
    ok = p50 <= cap
    return Check("performance", "warm_ttft", status(ok, gate(ctx, "warm_ttft")),
                 f"first request {first * 1e3:.0f} ms; warm p50 {p50 * 1e3:.0f} ms, p95 {p95 * 1e3:.0f} ms "
                 f"(cap p50 {cap * 1e3:.0f} ms)" + ("" if ok else "; every request pays a load: is keep-alive 0?"),
                 gate(ctx, "warm_ttft"), {"first_ms": first * 1e3, "warm_p50_ms": p50 * 1e3, "warm_p95_ms": p95 * 1e3})


def prefill_scaling(ctx: Ctx) -> Check:
    pts = []
    for n in ctx.slo["prefill_lengths"]:
        c = ctx.chat(text_of(n) + "\nReply with one word.", max_tokens=1)
        pts.append(((c.prompt_tokens or n), c.ttft or 0))
    (n0, t0), (n1, t1) = pts[0], pts[-1]
    rate = (n1 - n0) / (t1 - t0) if t1 > t0 else float("inf")
    monotonic = all(b[1] >= a[1] * 0.9 for a, b in zip(pts, pts[1:]))
    floor = ctx.slo["min_prefill_tps"]
    ok = rate >= floor and monotonic
    return Check("performance", "prefill_scaling", status(ok, gate(ctx, "prefill_scaling")),
                 f"TTFT by prompt tokens {[(n, round(t * 1e3)) for n, t in pts]} ms -> prefill ≈ {rate:,.0f} tok/s "
                 f"(floor {floor:,})", gate(ctx, "prefill_scaling"), {"prefill_tps": rate, "points": pts})


def decode_rate(ctx: Ctx) -> Check:
    calls = [ctx.chat(f"[{uuid.uuid4().hex[:6]}] Describe a city at night in detail.",
                      max_tokens=ctx.slo["decode_tokens"], temperature=0.7, seed=None) for _ in range(3)]
    tps = [c.decode_tps for c in calls if c.decode_tps]
    gaps = [g for c in calls for g in c.gaps]
    med = statistics.median(tps) if tps else 0.0
    floor = ctx.slo["min_decode_tps"]
    ok = med >= floor
    tbt50, tbt95 = (statistics.median(gaps), pct(gaps, 95)) if gaps else (0, 0)
    return Check("performance", "decode_rate", status(ok, gate(ctx, "decode_rate")),
                 f"decode {med:.0f} tok/s median of {len(tps)} (floor {floor}); TBT p50 {tbt50 * 1e3:.1f} ms, "
                 f"p95 {tbt95 * 1e3:.1f} ms", gate(ctx, "decode_rate"),
                 {"decode_tps": med, "tbt_p50_ms": tbt50 * 1e3, "tbt_p95_ms": tbt95 * 1e3})


def decode_sustained(ctx: Ctx) -> Check:
    runs = ctx.slo["sustained_runs"]
    tps = []
    for i in range(runs):
        c = ctx.chat(f"[{i}-{uuid.uuid4().hex[:6]}] Write a detailed travel diary.",
                     max_tokens=ctx.slo["decode_tokens"], temperature=0.7, seed=None)
        tps.append(c.decode_tps or 0.0)
    drop = 1 - (statistics.mean(tps[-2:]) / statistics.mean(tps[:2])) if tps[0] else 1.0
    cap = ctx.slo["max_sustained_drop"]
    ok = drop <= cap
    return Check("performance", "decode_sustained", status(ok, gate(ctx, "decode_sustained")),
                 f"tok/s over {runs} back-to-back generations {[round(t) for t in tps]}: "
                 f"{abs(drop):.0%} {'slower' if drop > 0 else 'faster'} from first to last (cap {cap:.0%} slower)" + ("" if ok else "; thermal throttling?"),
                 gate(ctx, "decode_sustained"), {"tps": tps, "drop": drop})


def prefix_cache(ctx: Ctx) -> Check:
    system = text_of(ctx.slo["prefix_tokens"])
    a = ctx.chat([{"role": "system", "content": system}, {"role": "user", "content": "Reply with: one"}], max_tokens=1)
    b = ctx.chat([{"role": "system", "content": system}, {"role": "user", "content": "Reply with: two"}], max_tokens=1)
    ratio = (b.ttft or 0) / (a.ttft or 1e-9)
    ok = ratio <= ctx.slo["max_cached_ttft_ratio"]
    g = gate(ctx, "prefix_cache")
    return Check("performance", "prefix_cache", status(ok, g),
                 f"same {a.prompt_tokens}-token system prompt: TTFT {a.ttft * 1e3:.0f} ms -> {b.ttft * 1e3:.0f} ms "
                 f"({ratio:.2f}x, cap {ctx.slo['max_cached_ttft_ratio']}x); cached_tokens reported {b.cached_tokens}",
                 g, {"ratio": ratio, "cached_tokens": b.cached_tokens})


def concurrency(ctx: Ctx) -> Check:
    tokens = ctx.slo["concurrency_tokens"]
    ctx.chat("warm-up", max_tokens=2)     # an unmeasured request, so a cold load can't pose as c=1 latency
    result = {}
    for level in ctx.slo["concurrency_levels"]:
        def one(i: int) -> Call:
            return ctx.chat(f"[{level}-{i}-{uuid.uuid4().hex[:6]}] List facts about rivers.",
                            max_tokens=tokens, temperature=0.7, seed=None)
        start = time.perf_counter()
        with cf.ThreadPoolExecutor(level) as pool:
            calls = list(pool.map(one, range(level)))
        wall = time.perf_counter() - start
        produced = sum((c.completion_tokens or c.chunks) for c in calls)
        result[level] = {"agg_tps": produced / wall, "ttft_p50_ms": statistics.median(c.ttft or 0 for c in calls) * 1e3}
    lo, hi = min(result), max(result)
    gain = result[hi]["agg_tps"] / result[lo]["agg_tps"]
    ok = gain >= ctx.slo["min_parallel_gain"]
    g = gate(ctx, "concurrency")
    summary = ", ".join(f"c={k}: {v['agg_tps']:.0f} tok/s, TTFT p50 {v['ttft_p50_ms']:.0f} ms" for k, v in result.items())
    return Check("performance", "concurrency", status(ok, g),
                 f"{summary}; c={hi} gives {gain:.2f}x aggregate throughput (floor {ctx.slo['min_parallel_gain']}x)"
                 + ("" if ok else "; requests are being served one at a time"), g, {"levels": result, "gain": gain})


# ---------------- context and quality ----------------

def _expected_tokens(ctx: Ctx, prompt: str) -> tuple[float, str]:
    """Expected prompt_tokens for `prompt`, measured on this model's own tokenizer.

    Two short calls of known length give the tokenizer's characters-per-token on this filler
    (the slope) and the chat template's fixed overhead (the intercept), so no guess about the
    tokenizer is needed. Falls back to ~4 characters per token if usage isn't reported.
    """
    short, longer = text_of(150), text_of(600)
    a = ctx.chat(short + "\nReply with one word.", max_tokens=1).prompt_tokens
    b = ctx.chat(longer + "\nReply with one word.", max_tokens=1).prompt_tokens
    if not a or not b or b <= a:
        return len(prompt) / 4, "assumed 4 chars/token"
    per_char = (b - a) / (len(longer) - len(short))
    overhead = a - per_char * len(short + "\nReply with one word.")
    return overhead + per_char * len(prompt), f"calibrated {1 / per_char:.1f} chars/token"


def context_integrity(ctx: Ctx) -> Check:
    n = ctx.slo["context_probe_tokens"]
    code = uuid.uuid4().hex[:6].upper()
    prompt = (f"NEEDLE: The secret code is {code}. Remember it.\n" + text_of(n)
              + "\nWhat is the secret code? Reply with only the code.")
    c = ctx.chat(prompt, max_tokens=12)
    expected, how = _expected_tokens(ctx, prompt)
    seen = c.prompt_tokens
    recalled = code.lower() in c.content.lower()
    if seen is None:                       # no usage: only the needle can tell
        tokens_ok, count = True, "prompt_tokens not reported"
    else:
        tokens_ok = seen >= ctx.slo["min_prompt_token_ratio"] * expected
        count = f"prompt_tokens {seen:,} ({seen / max(expected, 1):.0%})"
    ok = tokens_ok and recalled
    return Check("context", "context_integrity", "pass" if ok else "fail",
                 f"~{expected:,.0f}-token prompt ({how}) -> {count}; needle at the start recalled: {recalled}"
                 + ("" if ok else "; prompt was cut: check the engine's context-length setting"),
                 metrics={"expected": expected, "prompt_tokens": seen, "recalled": recalled,
                          "prompt_chars": len(prompt)})


def determinism(ctx: Ctx) -> Check:
    outs = [ctx.chat("Write one sentence about the sea.", max_tokens=24, temperature=0, seed=11).content
            for _ in range(3)]
    ok = len(set(outs)) == 1
    g = gate(ctx, "determinism")
    return Check("quality", "determinism", status(ok, g),
                 f"T=0, seed=11, 3 calls: {len(set(outs))} distinct output(s)", g)


def quant_delta(ctx: Ctx) -> Check:
    if not ctx.model_b:
        return Check("quality", "quant_delta", "skip", "no --model-b to compare", gated=False)

    def score(model: str) -> list[bool]:
        res = []
        for q, a in ITEMS:
            c = ctx.chat([{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}],
                         model=model, max_tokens=12, temperature=0, seed=3)
            res.append(correct(c.content, a))
        return res

    ra, rb = score(ctx.model), score(ctx.model_b)
    acc_a, acc_b = sum(ra) / len(ra), sum(rb) / len(rb)
    drop_pts = (acc_a - acc_b) * 100
    flipped = sum(1 for x, y in zip(ra, rb) if x and not y)
    cap = ctx.slo["max_quant_drop_pts"]
    ok = drop_pts <= cap
    return Check("quality", "quant_delta", "pass" if ok else "fail",
                 f"{ctx.model}: {acc_a:.0%} vs {ctx.model_b}: {acc_b:.0%} on {len(ITEMS)} exact-answer items "
                 f"({drop_pts:+.1f} pts drop, cap {cap}); {flipped} items right->wrong",
                 metrics={"acc_a": acc_a, "acc_b": acc_b, "drop_pts": drop_pts, "flipped": flipped})


CHECKS: list[Callable[[Ctx], Check]] = [
    usage_reported, max_tokens_honoured, decode_sustained, warm_ttft, prefill_scaling, decode_rate,
    prefix_cache, concurrency, context_integrity, determinism, quant_delta,
]


def run_all(ctx: Ctx, only: set[str] | None = None) -> list[Check]:
    out = []
    for fn in CHECKS:
        if only and fn.__name__ not in only:
            continue
        try:
            out.append(fn(ctx))
        except (openai.APIError, ValueError, KeyError, ZeroDivisionError, statistics.StatisticsError) as exc:
            out.append(Check("error", fn.__name__, "fail", f"{type(exc).__name__}: {exc}"))
    return out
