# GLM-4.7-Flash-NVFP4 — Build, Setup & Run

## Table of Contents

1. [Build the package](#1-build-the-package)
2. [Transfer to target machine](#2-transfer-to-target-machine)
3. [System requirements](#3-system-requirements)
4. [Install dependencies](#4-install-dependencies)
5. [Run the servers](#5-run-the-servers)
6. [Verify](#6-verify)
7. [Stop the servers](#7-stop-the-servers)
8. [VS Code Copilot integration](#8-vs-code-copilot-integration)
9. [Configuration](#9-configuration)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Build the package

```bash
cd ~/ai-projects/glm_4_7_flash_setup
python build.py -o dist/
```

Output is a self-contained directory — no model, no compiled venv:

```
dist/
├── benchmark.sh
├── glm_benchmark.py
├── glm_server.py
├── glm_compress.py
├── server_compress.py
├── lite_llm_config.yaml
├── requirements.txt
├── kill.sh
├── .env.example
├── README.md
└── scripts/
    ├── install.sh        # creates venv + installs deps
    ├── serve.sh          # starts both services
    ├── serve_backend.sh  # vLLM backend only
    └── serve_proxy.sh    # LiteLLM proxy only
```

---

## 2. Transfer to target machine

Copy the `dist/` directory to the target machine by any means:

```bash
# Example: rsync over SSH
rsync -avP dist/ user@target-machine:/path/to/glm-serving/

# Example: archive + scp
tar czf glm-serving.tar.gz dist/
scp glm-serving.tar.gz user@target-machine:/path/to/
ssh user@target-machine "tar xzf glm-serving.tar.gz"
```

The target machine only needs Python 3.12+, CUDA 12+, and an NVIDIA GPU.

---

## 3. System requirements

| Component | Requirement |
|-----------|-------------|
| Python | 3.12+ |
| CUDA | 12.x |
| GPU | Any NVIDIA GPU with ≥ 25 GB VRAM |
| Disk | ~25 GB for model + working space |
| OS | Linux (x86_64 / aarch64) |

Tested on: NVIDIA DGX Spark (GB10), H100, A100.

---

## 4. Install dependencies

```bash
cd /path/to/glm-serving/dist
bash scripts/install.sh
```

This:
1. Creates a Python 3.12 venv at `./venv`
2. Installs all pip dependencies from `requirements.txt`

Expect this to take several minutes on first run (pip downloads and compiles native dependencies).

> **Note:** The model is downloaded on first start, not during install. See [Run the servers](#5-run-the-servers).

---

## 5. Run the servers

### Start both services

```bash
bash scripts/serve.sh
```

Starts:
- **vLLM backend** on `http://0.0.0.0:11112` (model loading happens here)
- **LiteLLM proxy** on `http://0.0.0.0:11111` (OpenAI-compatible API)

The first start downloads the model from HuggingFace (~20 GB). The vLLM backend then loads it into GPU memory — allow 1–3 minutes depending on GPU.

### Start services separately

```bash
# Terminal 1 — backend only (must be running first)
bash scripts/serve_backend.sh

# Terminal 2 — proxy only (after backend is up)
bash scripts/serve_proxy.sh
```

### With custom ports or proxy settings

```bash
cp .env.example .env
# edit .env
bash scripts/serve.sh
```

---

## 6. Verify

```bash
# Check vLLM backend
curl http://localhost:11112/health
# → {"status": "ok", "model": "GadflyII/GLM-4.7-Flash-NVFP4"}

# Check LiteLLM proxy
curl http://localhost:11111/health
# → {"healthy": true}

# Quick chat completion test
curl http://localhost:11111/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-4.7-flash-nvfp4",
    "messages": [{"role": "user", "content": "Hello"}],
     "max_tokens": 50
   }'
```

## Benchmark local inference

```bash
# One-shot runner: starts what it needs, waits for health, then benchmarks
./benchmark.sh --target both

# Fresh local restart before benchmarking
./benchmark.sh --restart --target both

# Or benchmark directly if services are already up
python glm_benchmark.py --target both
```

This reports end-to-end latency plus output/total tokens per second for
LiteLLM, direct vLLM, or both.

---

## 7. Stop the servers

```bash
bash kill.sh
```

This kills all server processes by port and by name pattern, then verifies ports are free.

---

## 8. VS Code Copilot integration

1. Install the **[Copilot LLM Gateway](https://marketplace.visualstudio.com/items?itemName=andrewbutson.github-copilot-llm-gateway)** extension in VS Code.
2. Open extension settings.
3. Set the API endpoint to:
   ```
   http://localhost:11111/v1/chat/completions
   ```
4. Set the model name to:
   ```
   glm-4.7-flash-nvfp4
   ```
5. Start chatting with Copilot. All requests route through the LiteLLM proxy → vLLM backend → GLM-4.7-Flash-NVFP4.

---

## 9. Configuration

Key environment variables (set in `.env` before running):

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_PORT` | `11112` | vLLM backend port |
| `VLLM_HOST` | `0.0.0.0` | vLLM bind address |
| `LITE_LLM_PROXY_PORT` | `11111` | LiteLLM proxy port |
| `LITE_LLM_PROXY_HOST` | `0.0.0.0` | LiteLLM bind address |
| `HF_ENDPOINT` | _(unset)_ | HuggingFace mirror (e.g. `https://hf-mirror.com`) |

To adjust ports or other settings:

```bash
cp .env.example .env
nano .env
bash scripts/serve.sh  # restart to apply
```

---

## 10. Troubleshooting

### "address already in use" when starting

```bash
bash kill.sh
bash scripts/serve.sh
```

### Model download is very slow

Set a HuggingFace mirror:

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/serve.sh
```

Or pre-download the model on a fast connection:

```bash
source venv/bin/activate
python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
    AutoTokenizer.from_pretrained('GadflyII/GLM-4.7-Flash-NVFP4', trust_remote_code=True); \
    AutoModelForCausalLM.from_pretrained('GadflyII/GLM-4.7-Flash-NVFP4', trust_remote_code=True)"
```

### Low throughput / out of memory

Increase GPU memory utilization in `glm_server.py`:

```python
"--gpu-memory-utilization", "0.80",   # increase from default 0.50
```

Restart after editing.

### "auto tool choice" error

Ensure vLLM is started with these flags (already set in `glm_server.py`):

```
--enable-auto-tool-choice --tool-call-parser hermes
```

Restart the backend if you see:
```
"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set
```

### Check server logs

```bash
# Backend (vLLM)
tail -f logs/backend.log

# Proxy (LiteLLM)
tail -f logs/proxy.log
```

### Full restart from clean state

```bash
bash kill.sh
# wait 2 seconds
bash scripts/serve.sh
```
