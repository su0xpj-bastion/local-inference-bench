# Ops review (2026-11-20)

## Incidents

| ID | Severity | Title | Open actions | Post-mortem |
|---|---|---|---|---|
| INC-001 | high | Ollama silently drops the start of long prompts | 2 | [INC-001-silent-context-truncation.md](INC-001-silent-context-truncation.md) |
| INC-002 | medium | Truncation detector raised a false alarm on a dense tokenizer | 0 | [INC-002-false-truncation-alarm.md](INC-002-false-truncation-alarm.md) |

## Action items

| ID | Incident | Type | Action | State | Proof |
|---|---|---|---|---|---|
| A-001 | INC-001 | detect | Fail on short prompt_tokens or a lost start-of-prompt needle | closed | `test_inc001_silent_truncation_is_caught` passed |
| A-002 | INC-001 | detect | Live profile probes past Ollama's 4,096-token default | closed | `test_inc001_live_probe_exceeds_ollama_default_context` passed |
| A-003 | INC-001 | mitigate | Runbook carries the OLLAMA_CONTEXT_LENGTH fix and a verification command | closed | `test_inc001_runbook_has_fix_and_verification` passed |
| A-004 | INC-001 | prevent | Ask Ollama to surface truncation to API callers (response field or header) | open, due 2026-12-05 | Not filed yet. Check existing Ollama issues first. |
| A-005 | INC-001 | detect | Measure overflow behavior on llama.cpp, MLX and vLLM | open, due 2026-12-15 | Needs each engine installed locally; vLLM needs a GPU host. |
| A-006 | INC-002 | fix | Calibrate expected tokens on the model's own tokenizer | closed | `test_inc002_dense_tokenizer_does_not_false_alarm` passed |
| A-007 | INC-002 | detect | Reproduce the original false alarm in the mock (dense_tokenizer) | closed | `test_inc002_fixed_estimate_would_have_false_alarmed` passed |
| A-008 | INC-002 | detect | Fix must not blind the detector: truncation still caught on a dense tokenizer | closed | `test_inc001_truncation_caught_with_dense_tokenizer_too` passed |

## Live runs (evidence)

| Run | Model | Verdict | context_integrity |
|---|---|---|---|
| 2026-11-16-ollama-0.17-qwen2.5-1.5b-ctx16k | qwen2.5:1.5b | PASS | pass |
| 2026-11-16-ollama-0.17-qwen2.5-1.5b-default-ctx | qwen2.5:1.5b | FAIL | fail |
| 2026-11-16-ollama-0.40-gemma3-1b-q8-vs-q4-ctx16k | gemma3:1b-it-q8_0 | PASS | pass |
| 2026-11-16-ollama-0.40-qwen2.5-1.5b-ctx16k | qwen2.5:1.5b | PASS | pass |
| 2026-11-16-ollama-0.40-qwen2.5-1.5b-default-ctx | qwen2.5:1.5b | FAIL | fail |

With the documented config (`OLLAMA_CONTEXT_LENGTH=16384`): 3/3 live runs pass context_integrity. On defaults: 2 runs caught truncation.

## Summary

- Actions: 6 closed and proven, 2 open, 0 reopened
- Problems: none
