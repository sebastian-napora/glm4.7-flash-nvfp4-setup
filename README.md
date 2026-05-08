# GLM-4.7-Flash-NVFP4 Serving Setup

Local vLLM + LiteLLM proxy stack for `GadflyII/GLM-4.7-Flash-NVFP4`.

## Architecture

```
GitHub Copilot (VS Code) -> LiteLLM (11111) -> vLLM (11112)
```

- **vLLM backend** (port 11112): loads model and handles inference
- **LiteLLM proxy** (port 11111): OpenAI-compatible HTTP API, handles auth/routing/callbacks
- **History sanitization**: strips stored reasoning blocks from assistant history before the next turn is forwarded

## Files

| File | Purpose |
|------|---------|
| `glm_server.py` | vLLM backend server |
| `glm_compress.py` | LiteLLM callback that strips stored reasoning blocks from assistant history |
| `server_compress.py` | LiteLLM proxy entrypoint |
| `lite_llm_config.yaml` | LiteLLM model routing config |
| `start.sh` | Convenience start script |
| `glm_benchmark.py` | Benchmarks direct vLLM, LiteLLM, or both |
| `benchmark.sh` | Starts needed local services, waits for health, then benchmarks |

## Quick Start

```bash
./start.sh both
```

Typical startup time: **2-3 min** (CUDA graph capture only, no torch.compile).
Peak RAM during load: **~75-80 GB** (down from ~121 GB with old opt-level 3).

### Memory / context tuning (env vars)

| Env var | Default | Notes |
|---|---|---|
| `VLLM_MAX_MODEL_LEN` | `65536` | Max context tokens. Reduce to `32768` to save ~4 GB more KV-cache |
| `VLLM_OPT_LEVEL` | `1` | `1`=CUDA graphs only (recommended). `0`=eager (least RAM). `3`=torch.compile (**unsupported** by this model — causes ~121 GB peak with no benefit) |
| `VLLM_GPU_MEM_UTIL` | `0.75` | Fraction of 128 GB reserved for vLLM pool |

```bash
# Minimal RAM (eager, 32K context) — ~60-65 GB peak
VLLM_OPT_LEVEL=0 VLLM_MAX_MODEL_LEN=32768 ./start.sh both

# Full context (200K) — ~100 GB peak, for very long prompts
VLLM_OPT_LEVEL=1 VLLM_MAX_MODEL_LEN=202752 ./start.sh both
```

<details>
<summary>Manual start (two terminals)</summary>

```bash
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python glm_server.py        # terminal 1
python server_compress.py   # terminal 2
```
</details>

## Speculative Decoding (MTP) — currently BROKEN with this checkpoint ⚠️

GLM-4.7-Flash ships built-in **Multi-Token Prediction** layers
(`num_nextn_predict_layers: 1` in `config.json`) that vLLM can normally use
as a zero-overhead draft head. **However**, the `GadflyII/GLM-4.7-Flash-NVFP4`
checkpoint is **incompatible** with vLLM's MTP path:

- The checkpoint NVFP4-quantizes the MTP head's `eh_proj` linear
  (its `quantization_config.ignore` regex omits `eh_proj`), so it ships
  `eh_proj.weight_packed`, `eh_proj.input_global_scale`,
  `eh_proj.weight_scale` tensors.
- vLLM (`vllm/model_executor/models/glm4_moe_lite_mtp.py`, ≤ 0.19.x and
  current `main`) hard-codes `self.eh_proj = nn.Linear(...)` (unquantized),
  so loading crashes with:
  ```
  KeyError: 'model.layers.47.eh_proj.input_global_scale'
  ```

Either the checkpoint must add `eh_proj` to `quantization_config.ignore`,
or vLLM must wrap `eh_proj` in a quant-aware Linear. Both fixes are upstream.

**Until then, run without MTP:**

```bash
./start.sh both
```

The `start_mtp.sh` script and the `VLLM_SPEC_MTP=1` env var will refuse to
start with a clear error message. To bypass the guard and reproduce the
crash for upstream bug reports, set `VLLM_SPEC_MTP_FORCE=1`.

| Env var | Default | Purpose |
|---|---|---|
| `VLLM_SPEC_MTP` | `0` | Toggle MTP speculative decoding (currently guarded) |
| `VLLM_SPEC_MTP_FORCE` | `0` | Bypass guard — will crash on this checkpoint |
| `VLLM_SPEC_NUM_TOKENS` | `1` | Draft tokens per step (vLLM recipe recommends 1) |
| `VLLM_GPU_MEM_UTIL` | `0.75` (or `0.70` w/ MTP) | KV-cache budget |

## Inference Benchmark

Use the benchmark helper to start the local services it needs, wait for
readiness, and then compare LiteLLM against direct vLLM:

```bash
# One-shot local benchmark runner
./benchmark.sh --target both

# Fresh local start first
./benchmark.sh --restart --target both

# Single direct-vLLM run and stop what the script started
./benchmark.sh --target vllm --runs 1 --warmup-runs 0 --stop-after

# Or start the stack yourself, then run the benchmark directly
./start.sh both
python glm_benchmark.py --target both
```

`./benchmark.sh`:

- starts the local vLLM backend for `vllm`, `litellm`, or `both`
- starts the LiteLLM proxy when the target includes `litellm`
- waits for `/health` before benchmarking
- reuses already-running healthy local services
- supports `--restart` and `--stop-after`

What the benchmark reports:

- average, min, and max end-to-end latency
- average output tokens/sec
- average total tokens/sec
- average prompt and completion token counts

Default endpoints:

- LiteLLM proxy: `http://127.0.0.1:11111/v1`
- direct vLLM: `http://127.0.0.1:11112/v1`

## API Endpoints

| Endpoint | Port | Purpose |
|----------|------|---------|
| `POST /v1/chat/completions` | 11111/11112 | Chat completion |
| `GET /health` | 11111/11112 | Health check |

## Copilot Integration

In VS Code with GitHub Copilot, reference these files:
```
@lunch-model/glm_4_7_flash_setup/glm_server.py
@lunch-model/glm_4_7_flash_setup/server_compress.py
```

Configure Copilot to use `http://localhost:11111/v1/chat/completions` as the API endpoint.

## Model Details

- **Model**: `GadflyII/GLM-4.7-Flash-NVFP4`
- **Type**: 30B-A3B MoE (64 experts, 4 active)
- **Quantization**: NVFP4 (compressed-tensors, mixed precision)
- **Context**: 202,752 tokens max
- **Size**: ~20.4 GB (vs 62.4 GB BF16)
- **Accuracy loss**: ~1.3% vs BF16 on MMLU-Pro
