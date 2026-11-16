# local-inference-bench: live · qwen2.5:1.5b

Engine `http://127.0.0.1:11434/v1` · profile `ollama-m1-8gb` · 36 requests · 54.7 s · host Apple M1, 8 GB

| | Area | Check | Result |
|---|---|---|---|
| ✅ | contract | usage_reported | usage chunk seen=True, prompt=36, completion=32, content chunks=32 |
| ✅ | contract | max_tokens_honoured | max_tokens=16 -> 16 tokens, finish_reason=length |
| ✅ | performance | decode_sustained | tok/s over 6 back-to-back generations [37, 38, 37, 37, 37, 39]: 2% faster from first to last (cap 25% slower) *(not gated)* |
| ✅ | performance | warm_ttft | first request 148 ms; warm p50 95 ms, p95 106 ms (cap p50 1000 ms) *(not gated)* |
| ✅ | performance | prefill_scaling | TTFT by prompt tokens [(239, 502), (792, 1577), (1865, 3833)] ms -> prefill ≈ 488 tok/s (floor 200) *(not gated)* |
| ✅ | performance | decode_rate | decode 38 tok/s median of 3 (floor 15); TBT p50 26.0 ms, p95 28.6 ms *(not gated)* |
| ✅ | performance | prefix_cache | same 1498-token system prompt: TTFT 3081 ms -> 205 ms (0.07x, cap 0.6x); cached_tokens reported 0 *(not gated)* |
| ✅ | performance | concurrency | c=1: 35 tok/s, TTFT p50 143 ms, c=2: 36 tok/s, TTFT p50 278 ms, c=4: 65 tok/s, TTFT p50 533 ms; c=4 gives 1.87x aggregate throughput (floor 1.3x) *(not gated)* |
| ✅ | context | context_integrity | ~4,455-token prompt (calibrated 5.5 chars/token) -> prompt_tokens 4,467 (100%); needle at the start recalled: True |
| ✅ | quality | determinism | T=0, seed=11, 3 calls: 1 distinct output(s) *(not gated)* |
| ⏭️ | quality | quant_delta | no --model-b to compare *(not gated)* |

**Verdict: PASS**
