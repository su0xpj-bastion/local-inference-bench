# Runbook: `context_integrity` failed

## Symptoms

- `context_integrity` fails with `prompt_tokens` well under 100% and/or `needle at the start recalled: False`
- Callers report answers that ignore instructions or data near the start of long prompts
- Nothing in the API response looks wrong (HTTP 200)

## Triage (5 minutes)

1. Read the check's detail line in `report.md`.
   - `prompt_tokens` short **and** needle lost → real truncation. Go to *Fix*.
   - `prompt_tokens` short but needle recalled → suspect the estimate (INC-002 pattern). Check the
     detail says `calibrated`, not `assumed 4 chars/token`. If it says `assumed`, the engine isn't
     reporting usage on short prompts; treat the result as unproven and verify by hand.
   - needle lost but `prompt_tokens` ~100% → the model ignored the start. Not truncation: a quality
     issue; compare with a larger model.
2. Check the server log for a truncation warning (Ollama: `truncating input prompt`).

## Fix (Ollama)

Restart the server with an explicit context length at least as large as your longest prompt:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=16384 ollama serve
```

Memory grows with context length × parallel slots; on 8 GB, 16384 × 4 worked for 1–1.5B models.

## Verify

```bash
.venv/bin/python -m localbench run --target live --base-url http://127.0.0.1:11434/v1 \
    --model qwen2.5:1.5b --profile ollama-m1-8gb --only context_integrity
```

Pass = `prompt_tokens` within 80% of expected (profile threshold) and the needle recalled.

## Escalate / record

Record the run under `results/`, add the incident to `actions.json` if it is new, and link it from the
ops review.
