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

## Quick Start

```bash
# 1. Reuse existing venv (from lunch-model) or create new:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. Start backend + proxy (two terminals, or use background):
# Terminal 1: vLLM backend
python glm_server.py

# Terminal 2: LiteLLM proxy
python server_compress.py

# Or use the convenience script:
./start.sh both
```

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
