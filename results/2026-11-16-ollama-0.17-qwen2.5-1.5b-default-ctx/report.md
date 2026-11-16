# local-inference-bench: live · qwen2.5:1.5b

Engine `http://127.0.0.1:11434/v1` · profile `ollama-m1-8gb` · 4 requests · 12.6 s · host Apple M1, 8 GB

| | Area | Check | Result |
|---|---|---|---|
| ✅ | contract | usage_reported | usage chunk seen=True, prompt=36, completion=32, content chunks=32 |
| ❌ | context | context_integrity | ~4,456-token prompt (calibrated 5.5 chars/token) -> prompt_tokens 4,096 (92%); needle at the start recalled: False; prompt was cut: check the engine's context-length setting |

**Verdict: FAIL**
