# local-inference-bench

Contract, performance, context and quantization checks for an open-weight model served **on your own machine** through any OpenAI-compatible engine: Ollama, llama.cpp `llama-server`, MLX `mlx_lm.server`, or vLLM. Eleven checks, each with a fault it must catch. The same run gives you the numbers people ask about first (TTFT, time between tokens, prefill and decode tokens per second, concurrency scaling, prefix-cache effect) and catches the misconfigurations that quietly ruin them.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m localbench run --target mock                       # built-in mock engine, ~7 s
.venv/bin/python -m localbench run --target mock --faults serialized   # exit 1: concurrency fails
.venv/bin/python -m localbench run --target live --base-url http://127.0.0.1:11434/v1 \
    --model gemma3:1b-it-q8_0 --model-b gemma3:1b                      # see docs/LIVE.md
```

> **Status:** CI runs everything against the mock engine. The live path is implemented but no live run has been recorded yet, so every number on this page comes from the mock, not from a real model.

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
| Context | `context_integrity` | a 5K-token prompt keeps its `prompt_tokens`, and a code planted at the very start is recalled | `context_truncation` (a context-length setting smaller than the prompt drops the start, with no error) |
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

- **Not yet run on a real engine.** The first live run sets the `ollama-m1-8gb` thresholds; until then performance is reported, not gated, on live.
- **One machine, small load.** Concurrency goes to 4. This measures the shape of a laptop or workstation engine, not datacenter capacity; for that see [llm-serving-validation](https://github.com/su0xpj-bastion/llm-serving-validation) and [inference-endpoint-scenarios](https://github.com/su0xpj-bastion/inference-endpoint-scenarios).
- **Client-side timing.** TTFT includes the local HTTP hop, which on localhost is well under a millisecond.
- **Token counts on live depend on the model's tokenizer.** `prefill_scaling` uses the engine's own `prompt_tokens` for its x-axis; `context_integrity` estimates the expected count at ~4 characters per token (set `chars_per_token` per model in `slo.json`).
- **The quality set measures a difference, not a score.** 40 short exact-answer items show whether a 4-bit build lost accuracy against an 8-bit build of the same model. They are not a model evaluation.
