# local-inference-bench

Contract, performance, context and quantization checks for an open-weight model served **on your own machine** through any OpenAI-compatible engine: Ollama, llama.cpp `llama-server`, MLX `mlx_lm.server`, or vLLM. Eleven checks, each with a fault it must catch. The same run gives you the numbers people ask about first (TTFT, time between tokens, prefill and decode tokens per second, concurrency scaling, prefix-cache effect) and catches the misconfigurations that quietly ruin them.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m localbench run --target mock                       # built-in mock engine, ~7 s
.venv/bin/python -m localbench run --target mock --faults serialized   # exit 1: concurrency fails
.venv/bin/python -m localbench run --target live --base-url http://127.0.0.1:11434/v1 \
    --model qwen2.5:1.5b                                               # see docs/LIVE.md
```

> **Status:** CI runs everything against the mock engine. **Live runs are recorded** for `qwen2.5:1.5b` on Ollama 0.17 on an 8 GB M1 MacBook (below and in [`results/`](results/)). The "Healthy mock run" section further down is mock data.

## Live run: qwen2.5:1.5b on Ollama, M1 with 8 GB (2026-11-16)

Two full runs, about 70 s each, agreed within ~2%:

| Measure | Result |
|---|---|
| Decode | **37–38 tok/s**; time between tokens p50 26 ms, p95 29 ms |
| Prefill | **≈ 490 tok/s** (TTFT 0.5 s at 240 prompt tokens, 3.8 s at 1,865) |
| Warm TTFT | p50 **93–95 ms**, p95 105 ms; first request 148–169 ms |
| Prefix cache | same 1.5K-token system prompt twice: **3.1 s → 0.2 s** (0.07x) |
| Concurrency | 1 → 4 parallel requests: 35 → 65 tok/s aggregate (**1.87x**); TTFT p50 143 → 533 ms |
| Sustained decode | 6 back-to-back generations, no slowdown (thermal throttling not seen at this size) |
| Determinism | T=0 with a seed: identical output 3 of 3 |

### Finding: Ollama's default context silently drops the start of long prompts

The model supports a 32K context, but Ollama's default window here is **4,096 tokens**. A ~4.4K-token prompt returned HTTP 200 with `prompt_tokens` capped at exactly 4,096, and a code planted at the start of the prompt was gone from the answer. At ~8.9K tokens the same: capped at 4,096, 46% of the prompt seen. No error, no warning.

| Ollama setting | 4.4K-token prompt | 8.9K-token prompt |
|---|---|---|
| default | `prompt_tokens` 4,096 (92%), code lost ❌ | 4,096 (46%), code lost ❌ |
| `OLLAMA_CONTEXT_LENGTH=16384` | 4,465 (101%), code recalled ✅ | 8,839 (100%), code recalled ✅ |

`context_integrity` now probes at 6,000 tokens on the live profile so it catches this default.

Two smaller observations: the prefix cache clearly works (TTFT drops 15x) but Ollama reports `cached_tokens: 0`, so cache hits are invisible in usage data; and `gemma3:1b` could not be pulled at all on Ollama 0.17 (the registry requires a newer version), which is why this run uses Qwen.

### A harness bug the live run found

The first live run failed `context_integrity` on a prompt that was **not** truncated: the check estimated tokens at 4 characters per token, and Qwen's tokenizer averages 5.5 on this text. The estimate is now calibrated on the model itself (two short prompts give the tokenizer's characters-per-token and the chat template's fixed overhead); expected and actual counts now agree within 1%.

## The checks

| Area | Check | What it proves | Fault it catches |
|---|---|---|---|
| Contract | `usage_reported` | `stream_options.include_usage` returns a usage chunk with prompt and completion counts | `usage_missing` |
| Contract | `max_tokens_honoured` | a 16-token limit gives ≤ 16 tokens and `finish_reason: length` | `ignore_max_tokens` |
| Performance | `warm_ttft` | warm TTFT p50 / p95 under the cap; the first request is reported separately | `keep_alive_zero` (model unloaded after every request, so every request pays the load) |
| Performance | `prefill_scaling` | TTFT at 3 prompt lengths, each unique so no cache helps → prefill tok/s from the slope | `slow_prefill` |
| Performance | `decode_rate` | decode tok/s after the first token, TBT p50 / p95 | `decode_throttle` |
| Performance | `decode_sustained` | 6 back-to-back generations: the last ones are not more than 20% slower than the first | `decode_throttle` (thermal throttling on a fanless laptop) |
| Performance | `prefix_cache` | the same long system prompt a second time: TTFT drops to ≤ 0.5x | `no_prefix_cache`, and `keep_alive_zero` (unloading drops the cache) |
| Performance | `concurrency` | aggregate tok/s at c=4 vs c=1 ≥ 1.5x, after an unmeasured warm-up | `serialized` (an engine started with one slot) |
| Context | `context_integrity` | a long prompt (5K mock, 6K live) keeps its `prompt_tokens` against a count calibrated on the model's tokenizer, and a code planted at the very start is recalled | `context_truncation` (a context-length setting smaller than the prompt drops the start, with no error) |
| Quality | `determinism` | T=0 with a fixed seed gives the same output 3 times | `nondeterministic` |
| Quality | `quant_delta` | two quantizations of one model on 40 exact-answer items: accuracy drop ≤ 5 points, and how many items went right → wrong | `quant_degraded` |

`tests/test_faults.py` runs the full suite once per fault and pins the **exact** set of checks that fail, so a detector that stops working, or starts firing on the wrong fault, breaks the build. A healthy engine must pass all eleven.

## Healthy mock run

```
| ✅ | performance | prefill_scaling  | TTFT by prompt tokens [(493, 36), (1486, 90), (2993, 166)] ms -> prefill ≈ 19,152 tok/s |
| ✅ | performance | decode_rate      | decode 196 tok/s median of 3; TBT p50 5.1 ms, p95 5.6 ms |
| ✅ | performance | prefix_cache     | same 2991-token system prompt: TTFT 165 ms -> 11 ms (0.06x); cached_tokens 2988 |
| ✅ | performance | concurrency      | c=1: 188 tok/s, c=4: 530 tok/s; 2.82x aggregate throughput |
| ✅ | quality     | quant_delta      | q8: 100% vs q4: 100% on 40 exact-answer items |
```

And what the faults look like (from the same suite):

```
serialized          c=1: 193 tok/s, c=4: 194 tok/s, TTFT p50 376 ms; 1.01x; requests are being served one at a time
context_truncation  ~4,999-token prompt -> prompt_tokens 2,048 (41%); needle at the start recalled: False
decode_throttle     tok/s over 6 back-to-back generations [173, 158, 147, 137, 127, 120]: 26% slower
quant_degraded      q8: 100% vs q4: 62% (+37.5 pts drop); 15 items right->wrong
```

## The mock engine

`mockserve/server.py` speaks the Chat Completions wire format (SSE chunks, the `include_usage` chunk with `choices: []`, `prompt_tokens_details.cached_tokens`, 404 for an unknown model, 400 over the context window) and models what matters on one machine: prefill cost per prompt token, decode cost per token that grows with the number of active sequences, 4 parallel slots, a message-level prefix cache, a cold load on the first request per model, an 8K context window, and q8 / q4 variants of one model. It is a timing fixture with the right *shape*, not a performance model of any real engine.

## Layout

```
localbench/client.py    one streamed request, timed per content chunk; 429-only retry
localbench/checks.py    the 11 checks
localbench/quality.py   40 exact-answer items + grading (no LLM judge)
localbench/cli.py       run, report.md + report.json, CI job summary
mockserve/server.py     mock engine + 10 injectable faults
slo.json                mock profile (gated) and ollama-m1-8gb profile (performance not gated yet)
docs/LIVE.md            running against Ollama, llama.cpp, MLX or vLLM
```

## Where this stops

- **One live model so far.** The `ollama-m1-8gb` numbers come from two runs of one 1.5B model; performance is reported, not gated, on live until more runs set thresholds. `quant_delta` has no live result yet (it needs a second quantization of the same model).
- **One machine, small load.** Concurrency goes to 4. This measures the shape of a laptop or workstation engine, not datacenter capacity; for that see [llm-serving-validation](https://github.com/su0xpj-bastion/llm-serving-validation) and [inference-endpoint-scenarios](https://github.com/su0xpj-bastion/inference-endpoint-scenarios).
- **Client-side timing.** TTFT includes the local HTTP hop, which on localhost is well under a millisecond.
- **Token counts on live depend on the model's tokenizer.** `prefill_scaling` uses the engine's own `prompt_tokens` for its x-axis; `context_integrity` estimates the expected count at ~4 characters per token (set `chars_per_token` per model in `slo.json`).
- **The quality set measures a difference, not a score.** 40 short exact-answer items show whether a 4-bit build lost accuracy against an 8-bit build of the same model. They are not a model evaluation.
