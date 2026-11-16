# Running it against a real local engine

Everything in CI runs against the mock engine. This is the procedure for a real run on a laptop.
Nothing here needs an API key or costs money; it needs disk space for the model weights.

## 1. Serve a model (Ollama shown; llama.cpp and MLX work the same way)

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=16384 ollama serve &   # OpenAI-compatible API on :11434/v1
ollama pull gemma3:1b                           # Q4_K_M, 806 MB (needs a recent Ollama)
ollama pull gemma3:1b-it-q8_0                   # Q8_0, 1.1 GB: the second build for quant_delta
```

On an 8 GB machine stay at 1-4B parameters; a model that doesn't fit in memory swaps, and
every latency number becomes a disk benchmark.

Other engines, same command below with a different `--base-url`:

| Engine | Start | Base URL |
|---|---|---|
| llama.cpp | `llama-server -m model.gguf --parallel 4 -c 8192` | `http://127.0.0.1:8080/v1` |
| MLX | `mlx_lm.server --model <hf-repo>` | `http://127.0.0.1:8080/v1` |
| vLLM (GPU) | `vllm serve <hf-repo>` | `http://127.0.0.1:8000/v1` |

## 2. Run

```bash
.venv/bin/python -m localbench run --target live \
    --base-url http://127.0.0.1:11434/v1 \
    --model gemma3:1b-it-q8_0 --model-b gemma3:1b \
    --out results/<date>-ollama-<version>-<model>
```

The `ollama-m1-8gb` profile in `slo.json` reports performance without gating it until the
thresholds are calibrated from repeated runs. Contract, context and quantization checks gate.

## 3. What to look at first

- **concurrency**: Ollama serves several requests at once only if `OLLAMA_NUM_PARALLEL` allows it.
  A gain near 1.0x means requests queue.
- **context_integrity**: if the engine's context length is smaller than the prompt, the start of
  the prompt is dropped without an error. On Ollama 0.17 the default window is 4,096 tokens even for
  models that support 32K, and 0.40 cut a prompt to about 2,050 tokens on an 8 GB machine; start the server with `OLLAMA_CONTEXT_LENGTH` set (see the README finding).
- **warm_ttft**: a first request that is seconds slower than the rest is the model loading.
  If every request is that slow, the model is being unloaded between requests (keep-alive).
- **quant_delta**: the 4-bit vs 8-bit accuracy difference on the same 40 items.

Commit the report directory with the run date, engine version and `ollama ps` output.
