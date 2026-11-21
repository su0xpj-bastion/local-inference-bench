# Service-level objectives

The bench is a detection pipeline: its job is to notice when a local engine is quietly wrong. So the
objectives are about the detector, not about the engine's speed.

| SLO | Target | Measured by | Where |
|---|---|---|---|
| Detection coverage | 100% of injected faults caught, by exactly the checks expected | `tests/test_faults.py` (10 faults pinned) | CI, every push |
| False alarms | 0 failures on a healthy engine, including the `dense_tokenizer` condition | `test_healthy_engine_passes_every_check`, `test_inc002_*` | CI, every push |
| Silent truncation | 0 truncated prompts pass undetected | `test_inc001_*` | CI, every push |
| Closed means closed | every closed action item has a passing regression test | `tools/ops_review.py` | CI, every push |
| Live context integrity | `context_integrity` passes on every recorded live run with the documented config | `results/*/report.json` | ops review (informational) |

Error budget: none. A miss on any CI-gated SLO fails the build and reopens the related action item.

Performance numbers (TTFT, decode, concurrency) are reported but not gated on live runs yet; see
`slo.json` → `ollama-m1-8gb.ungated`. They become SLOs once repeated runs set stable thresholds.
