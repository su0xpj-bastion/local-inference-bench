# local-inference-bench: live · qwen2.5:1.5b

Engine `http://127.0.0.1:11434/v1` · profile `ollama-m1-8gb` · 36 requests · 42.3 s · host Apple M1, 8 GB

| | Area | Check | Result |
|---|---|---|---|
| ✅ | contract | usage_reported | usage chunk seen=True, prompt=36, completion=32, content chunks=32 |
| ✅ | contract | max_tokens_honoured | max_tokens=16 -> 16 tokens, finish_reason=length |
| ✅ | performance | decode_sustained | tok/s over 6 back-to-back generations [42, 43, 36, 46, 47, 48]: 14% faster from first to last (cap 25% slower) *(not gated)* |
| ✅ | performance | warm_ttft | first request 75 ms; warm p50 36 ms, p95 43 ms (cap p50 1000 ms) *(not gated)* |
| ✅ | performance | prefill_scaling | TTFT by prompt tokens [(241, 429), (792, 1681), (1865, 3789)] ms -> prefill ≈ 483 tok/s (floor 200) *(not gated)* |
| ✅ | performance | decode_rate | decode 41 tok/s median of 3 (floor 15); TBT p50 23.3 ms, p95 43.1 ms *(not gated)* |
| ✅ | performance | prefix_cache | same 1497-token system prompt: TTFT 2815 ms -> 355 ms (0.13x, cap 0.6x); cached_tokens reported 1491 *(not gated)* |
| ✅ | performance | concurrency | c=1: 46 tok/s, TTFT p50 82 ms, c=2: 58 tok/s, TTFT p50 183 ms, c=4: 99 tok/s, TTFT p50 267 ms; c=4 gives 2.14x aggregate throughput (floor 1.3x) *(not gated)* |
| ✅ | context | context_integrity | ~4,482-token prompt (calibrated 5.4 chars/token) -> prompt_tokens 4,466 (100%); needle at the start recalled: True |
| ✅ | quality | determinism | T=0, seed=11, 3 calls: 1 distinct output(s) *(not gated)* |
| ⏭️ | quality | quant_delta | no --model-b to compare *(not gated)* |

**Verdict: PASS**
