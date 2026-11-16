# local-inference-bench: live · gemma3:1b-it-q8_0 vs gemma3:1b

Engine `http://127.0.0.1:11434/v1` · profile `ollama-m1-8gb` · 116 requests · 53.2 s · host Apple M1, 8 GB

| | Area | Check | Result |
|---|---|---|---|
| ✅ | contract | usage_reported | usage chunk seen=True, prompt=16, completion=32, content chunks=32 |
| ✅ | contract | max_tokens_honoured | max_tokens=16 -> 16 tokens, finish_reason=length |
| ✅ | performance | decode_sustained | tok/s over 6 back-to-back generations [50, 51, 52, 52, 51, 51]: 1% faster from first to last (cap 25% slower) *(not gated)* |
| ✅ | performance | warm_ttft | first request 54 ms; warm p50 43 ms, p95 45 ms (cap p50 1000 ms) *(not gated)* |
| ✅ | performance | prefill_scaling | TTFT by prompt tokens [(222, 219), (772, 720), (1846, 1829)] ms -> prefill ≈ 1,009 tok/s (floor 200) *(not gated)* |
| ✅ | performance | decode_rate | decode 49 tok/s median of 3 (floor 15); TBT p50 19.9 ms, p95 22.7 ms *(not gated)* |
| ✅ | performance | prefix_cache | same 1499-token system prompt: TTFT 1339 ms -> 73 ms (0.05x, cap 0.6x); cached_tokens reported 1493 *(not gated)* |
| ✅ | performance | concurrency | c=1: 44 tok/s, TTFT p50 62 ms, c=2: 80 tok/s, TTFT p50 95 ms, c=4: 108 tok/s, TTFT p50 169 ms; c=4 gives 2.44x aggregate throughput (floor 1.3x) *(not gated)* |
| ✅ | context | context_integrity | ~4,371-token prompt (calibrated 5.6 chars/token) -> prompt_tokens 4,448 (102%); needle at the start recalled: True |
| ✅ | quality | determinism | T=0, seed=11, 3 calls: 1 distinct output(s) *(not gated)* |
| ✅ | quality | quant_delta | gemma3:1b-it-q8_0: 72% vs gemma3:1b: 72% on 40 exact-answer items (+0.0 pts drop, cap 7.5); 2 items right->wrong |

**Verdict: PASS**
