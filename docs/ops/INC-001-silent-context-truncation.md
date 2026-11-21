# INC-001: Ollama silently drops the start of long prompts

| | |
|---|---|
| Detected | 2026-11-16, first live run, by `context_integrity` |
| Severity | High: wrong answers with no error anywhere the caller can see |
| Affected | Ollama 0.17 and 0.40 on default settings, 8 GB M1, qwen2.5:1.5b (measured); any model served with a default context below its prompt length |
| Status | Mitigated (config + detection). Upstream visibility gap still open (A-004) |

## What happened

Both models support a 32K context. Ollama's default window on this machine is much smaller, and a longer
prompt is cut from the **front**. The API returns HTTP 200 with a normal answer. A code planted at the
start of the prompt was gone from the answer.

| Ollama | Prompt (calibrated) | `prompt_tokens` | Needle recalled | Evidence |
|---|---|---|---|---|
| 0.17 default | ~4,456 | 4,096 (92%) | no | `results/2026-11-16-ollama-0.17-qwen2.5-1.5b-default-ctx/` |
| 0.40 default | ~4,441 | 2,050 (46%) | no | `results/2026-11-16-ollama-0.40-qwen2.5-1.5b-default-ctx/` |
| either, `OLLAMA_CONTEXT_LENGTH=16384` | ~4.4K and ~8.9K | 100–102% | yes | `results/*-ctx16k/` |

## Why it was invisible

- No error, no response field, no header. The only trace is the server log:
  `WARN msg="truncating input prompt" limit=2050 prompt=4466`.
- The upgrade from 0.17 to 0.40 made the default **stricter** (4,096 → 2,050), so an upgrade can
  silently break callers that worked before.
- A latency or availability dashboard stays green: shorter prompts are faster.

## Detection

`context_integrity` checks two independent signals: `prompt_tokens` against a count calibrated on the
model's own tokenizer, and recall of a code planted at the very start. Either one failing fails the run.

## Root cause

Default context length is a server-side setting smaller than the models' supported window, and
overflow is handled by front truncation with a log-only warning.

## Action items

Tracked in [`actions.json`](actions.json). An action closes only when its named regression test passes
(`python tools/ops_review.py`).

| ID | Type | Action | Status |
|---|---|---|---|
| A-001 | detect | Fail on short `prompt_tokens` or a lost start-of-prompt needle | closed |
| A-002 | detect | Live profile probes past the 4,096 default (6,000 tokens) | closed |
| A-003 | mitigate | Runbook: fix with `OLLAMA_CONTEXT_LENGTH`, then verify | closed |
| A-004 | prevent | Ask Ollama to surface truncation to API callers (field or header) | open |
| A-005 | detect | Measure overflow behavior on llama.cpp, MLX and vLLM | open |
