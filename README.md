# local-inference-bench

Contract, performance, context and quantization checks for an open-weight model served **on your own machine** through any OpenAI-compatible engine: Ollama, llama.cpp `llama-server`, MLX `mlx_lm.server`, or vLLM. Eleven checks, each with a fault it must catch. The same run gives you the numbers people ask about first (TTFT, time between tokens, prefill and decode tokens per second, concurrency scaling, prefix-cache effect) and catches the misconfigurations that quietly ruin them.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m localbench run --target mock                       # built-in mock engine, ~7 s
.venv/bin/python -m localbench run --target mock --faults serialized   # exit 1: concurrency fails
.venv/bin/python -m localbench run --target live --base-url http://127.0.0.1:11434/v1 \
    --model qwen2.5:1.5b                                               # see docs/LIVE.md
```

> **Status:** CI runs everything against the mock engine. **Live runs are recorded** on an 8 GB M1 MacBook with Ollama 0.17 and 0.40 (below, raw reports in [`results/`](results/)). The "Healthy mock run" section further down is mock data.

## Live runs on an M1 with 8 GB (2026-11-16)

All with `OLLAMA_NUM_PARALLEL=4` and `OLLAMA_CONTEXT_LENGTH=16384`. Each full run takes about 70 s; every check passed.

| Measure | qwen2.5:1.5b, Ollama 0.17 | qwen2.5:1.5b, Ollama 0.40 | gemma3:1b Q8_0, Ollama 0.40 |
|---|---|---|---|
| Decode tok/s (median) | 37–38 (2 runs) | 41 | **49** |
| Time between tokens p50 / p95 | 26 / 29 ms | 23 / 43 ms | 20 / 23 ms |
| Prefill tok/s | ≈ 490 | ≈ 483 | **≈ 1,009** |
| Warm TTFT p50 | 95 ms | 36 ms | 43 ms |
| Prefix cache, same 1.5K-token prompt twice | 3.1 s → 0.2 s, `cached_tokens` **0** | 2.8 s → 0.36 s, `cached_tokens` 1,491 | 1.3 s → 0.07 s, `cached_tokens` 1,493 |
| Aggregate throughput, 4 parallel vs 1 | 1.87x | 2.14x | 2.44x |
| T=0 + seed, 3 calls | identical | identical | identical |

**Version upgrade, same model:** going from Ollama 0.17 to 0.40, warm TTFT dropped from 95 to 36 ms, parallel scaling rose from 1.87x to 2.14x, and the prefix cache became visible in usage data (0.17 reported `cached_tokens: 0` even when the cache cut TTFT 15x). Decode got slightly faster but noisier (TBT p95 29 → 43 ms; one dip to 36 tok/s in the sustained run).

**4-bit vs 8-bit Gemma 3 1B** (`quant_delta`, 40 exact-answer items): Q8_0 29/40, Q4_K_M 29/40, no drop; two items went right → wrong and two wrong → right, so the difference is noise at this size. The 4-bit build is 806 MB against 1.1 GB.

### Finding: Ollama's default context silently drops the start of long prompts

Qwen 2.5 1.5B and Gemma 3 1B both support a 32K context, but Ollama's default window on this machine is much smaller, and a longer prompt is cut from the **front** with an HTTP 200 and no error in the API response. A code planted at the start of the prompt is gone from the answer.

| Ollama | Setting | ~4.4K-token prompt | ~8.9K-token prompt |
|---|---|---|---|
| 0.17 | default | `prompt_tokens` 4,096 (92%), code lost ❌ | 4,096 (46%), code lost ❌ |
| 0.40 | default | `prompt_tokens` **2,050** (46%), code lost ❌ | |
| 0.17 / 0.40 | `OLLAMA_CONTEXT_LENGTH=16384` | 100–102%, code recalled ✅ | 8,839 (100%), code recalled ✅ |

The upgrade made the default **stricter**, not looser, on this 8 GB machine. The only trace is in the server's own log: `WARN msg="truncating input prompt" limit=2050 prompt=4466`. A client calling the OpenAI-compatible API never sees it, which is exactly why `context_integrity` checks `prompt_tokens` against a calibrated expectation and plants a code at the start. The live profile probes at 6,000 tokens so the check catches the default.

Also: `gemma3:1b` couldn't be pulled at all on Ollama 0.17 (the registry requires a newer version), which is why the first runs used Qwen.

### A harness bug the live run found

The first live run failed `context_integrity` on a prompt that was **not** truncated: the check estimated tokens at 4 characters per token, and Qwen's tokenizer averages 5.5 on this text. The estimate is now calibrated on the model itself (two short prompts give the tokenizer's characters-per-token and the chat template's fixed overhead); expected and actual counts now agree within 1%.

## Operations: incidents, SLOs, and action items that stay closed

The two live findings above are written up as incidents in [`docs/ops/`](docs/ops/), run the way a production reliability team would:

- **Post-mortems:** [INC-001](docs/ops/INC-001-silent-context-truncation.md) (Ollama silently drops the start of long prompts) and [INC-002](docs/ops/INC-002-false-truncation-alarm.md) (our own detector raised a false alarm on a denser tokenizer, fixed without loosening the threshold).
- **[SLOs](docs/ops/SLO.md)** for the detector itself: 100% fault coverage, zero false alarms, zero silent truncations.
- **[Runbook](docs/ops/RUNBOOK.md)**: triage a `context_integrity` failure in five minutes, fix, verify.
- **[Action tracker](docs/ops/actions.json)**: every closed action names the regression test that proves it. `tools/ops_review.py` runs those tests on every push; if a fix is undone, the action shows as **REOPENED** and CI fails. Open actions need an owner and a due date and are flagged when overdue. Latest review: [`docs/ops/REVIEW.md`](docs/ops/REVIEW.md).

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
mockserve/server.py     mock engine + 10 injectable faults + dense_tokenizer condition
tools/ops_review.py     ops review: incidents, actions proven by tests, live evidence
docs/ops/               post-mortems, SLOs, runbook, action tracker
slo.json                mock profile (gated) and ollama-m1-8gb profile (performance not gated yet)
docs/LIVE.md            running against Ollama, llama.cpp, MLX or vLLM
```

## Where this stops

- **Two small models, one machine.** Live numbers come from 1–1.5B models on one 8 GB M1; performance is reported, not gated, on live until more runs set thresholds. The 40-item quality set shows a large quantization regression, not a 1–2 point one.
- **One machine, small load.** Concurrency goes to 4. This measures the shape of a laptop or workstation engine, not datacenter capacity; for that see [llm-serving-validation](https://github.com/su0xpj-bastion/llm-serving-validation) and [inference-endpoint-scenarios](https://github.com/su0xpj-bastion/inference-endpoint-scenarios).
- **Client-side timing.** TTFT includes the local HTTP hop, which on localhost is well under a millisecond.
- **Token counts on live depend on the model's tokenizer.** `prefill_scaling` uses the engine's own `prompt_tokens` for its x-axis; `context_integrity` estimates the expected count at ~4 characters per token (set `chars_per_token` per model in `slo.json`).
- **The quality set measures a difference, not a score.** 40 short exact-answer items show whether a 4-bit build lost accuracy against an 8-bit build of the same model. They are not a model evaluation.
