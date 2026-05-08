#!/usr/bin/env python3
"""
Build script — creates a distributable package of the GLM-4.7-Flash-NVFP4 serving stack.

Package contents (no model, no compiled venv — those are machine-specific):
  dist/
  ├── benchmark.sh           — start needed services, then benchmark them
  ├── glm_benchmark.py       — benchmark direct vLLM and/or LiteLLM
  ├── kill.sh                 — stop all servers
  ├── requirements.txt        — pip dependencies
  ├── glm_server.py           — vLLM backend
  ├── glm_compress.py         — LiteLLM history sanitization callback
  ├── server_compress.py      — LiteLLM proxy entrypoint
  ├── lite_llm_config.yaml    — LiteLLM routing config
  ├── README.md                — deployment + usage guide
  ├── .env.example            — environment variable template
  └── scripts/
      ├── install.sh          — install deps on target machine
      ├── serve.sh            — start both servers
      ├── serve_backend.sh    — start vLLM backend only
      └── serve_proxy.sh      — start LiteLLM proxy only

Usage:
  python build.py --output dist/

Target machine needs:
  • Python 3.12+
  • CUDA 12+ (for vLLM / GPU inference)
  • NVIDIA GPU (GB10 / H100 / etc.)
  • ~25 GB VRAM for this model
  • Internet (to download model from HuggingFace on first run)
"""

import argparse
import os
import shutil
import stat
import sys
from pathlib import Path

# ── Colour output ─────────────────────────────────────────────────────────────
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
BOLD = "\033[1m"
RESET = "\033[0m"


def info(msg: str):
    print(f"{GREEN}✔{RESET} {msg}")


def warn(msg: str):
    print(f"{YELLOW}⚠ {msg}{RESET}")


def fatal(msg: str):
    print(f"{RED}✖ {msg}{RESET}", file=sys.stderr)
    sys.exit(1)


def make_executable(path: Path):
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


# ── Shell script templates (written with raw strings to avoid escaping hell) ──

INSTALL_SH = r"""#!/bin/bash
# Install GLM-4.7-Flash-NVFP4 serving dependencies.
# Run once on a fresh machine. Creates a venv at ./venv.
# Requires: Python 3.12+, CUDA 12+, NVIDIA GPU.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

echo "Creating Python 3.12 venv..."
python3 -m venv venv

echo "Activating venv..."
source venv/bin/activate

echo "Installing requirements (this may take several minutes)..."
pip install --upgrade pip
pip install -r requirements.txt

echo ""
echo "✅ Installation complete."
echo "   Next: bash scripts/serve.sh"
echo "   Or benchmark locally: ./benchmark.sh --target both"
"""

SERVE_SH = r"""#!/bin/bash
# Start both vLLM backend (11112) and LiteLLM proxy (11111).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

[ -f .env ] && source .env
mkdir -p logs

echo "🚀 Starting vLLM backend (port 11112)..."
source venv/bin/activate
python glm_server.py > logs/backend.log 2>&1 &
BACKEND_PID=$!
echo "Backend PID: $BACKEND_PID"

echo "Waiting 8s for backend to load model..."
sleep 8

echo "🚀 Starting LiteLLM proxy (port 11111)..."
python server_compress.py > logs/proxy.log 2>&1 &
PROXY_PID=$!
echo "Proxy PID: $PROXY_PID"

echo ""
echo "✅ Both services started:"
echo "   vLLM backend:  http://0.0.0.0:11112"
echo "   LiteLLM proxy: http://0.0.0.0:11111"
echo ""
echo "Logs: logs/backend.log | logs/proxy.log"
"""

SERVE_BACKEND_SH = r"""#!/bin/bash
# Start vLLM backend only (port 11112).

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

[ -f .env ] && source .env
mkdir -p logs
source venv/bin/activate
python glm_server.py > logs/backend.log 2>&1 &
echo "Backend PID: $!"
echo "Wait ~10s for model to load, then: curl http://localhost:11112/health"
"""

SERVE_PROXY_SH = r"""#!/bin/bash
# Start LiteLLM proxy only (port 11111).
# Requires the vLLM backend already running on 11112.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PACKAGE_DIR"

[ -f .env ] && source .env
mkdir -p logs
source venv/bin/activate
python server_compress.py > logs/proxy.log 2>&1 &
echo "Proxy PID: $!"
echo "Then: curl http://localhost:11111/health"
"""

ENV_EXAMPLE = (
    "# Environment variables for GLM-4.7-Flash-NVFP4 serving stack.\n"
    "# Copy to .env and adjust as needed.\n"
    "\n"
    "# ── vLLM backend ──────────────────────────────────────────────────\n"
    "VLLM_HOST=0.0.0.0\n"
    "VLLM_PORT=11112\n"
    "\n"
    "# ── LiteLLM proxy ───────────────────────────────────────────────\n"
    "LITE_LLM_PROXY_HOST=0.0.0.0\n"
    "LITE_LLM_PROXY_PORT=11111\n"
    "\n"
    "# ── Model cache ─────────────────────────────────────────────────\n"
    "# Optional: use a HuggingFace mirror for faster downloads:\n"
    "# HF_ENDPOINT=https://hf-mirror.com\n"
)

README_PACKAGE = """\
# GLM-4.7-Flash-NVFP4 Serving Package

Self-contained serving stack for `GadflyII/GLM-4.7-Flash-NVFP4`.

## What Gets Downloaded

On first run the model is downloaded automatically from HuggingFace:
  `GadflyII/GLM-4.7-Flash-NVFP4`

Cached at: `~/.cache/huggingface/hub/models--GadflyII--GLM-4.7-Flash-NVFP4/`

## System Requirements

| | |
|---|---|
| Python | 3.12+ |
| CUDA | 12.x |
| GPU | Any NVIDIA GPU with ≥ 25 GB VRAM |
| Disk | ~25 GB for model + working space |

Tested on: NVIDIA DGX Spark (GB10, 128 GB), H100, A100.

## Quick Start

```bash
# 1 — Install dependencies (one-time, creates a fresh venv)
bash scripts/install.sh

# 2 — Download model, start services if needed, and benchmark
bash benchmark.sh --target both

# 3 — Or start servers manually (first run downloads ~20 GB)
bash scripts/serve.sh

# 4 — Verify health
curl http://localhost:11112/health
curl http://localhost:11111/health
```

## Services

| Service | Port | Purpose |
|---------|------|---------|
| vLLM backend | 11112 | Inference |
| LiteLLM proxy | 11111 | OpenAI-compatible API for Copilot |

## Configuration

Copy `.env.example` to `.env` and adjust if needed:

```bash
cp .env.example .env
# edit .env
```

Key variables:
  `VLLM_PORT` / `VLLM_HOST` — vLLM backend address
  `LITE_LLM_PROXY_PORT` / `LITE_LLM_PROXY_HOST` — proxy address

## Stopping

```bash
bash kill.sh
```

## Troubleshooting

**Port already in use**
```bash
bash kill.sh
bash scripts/serve.sh
```

**Model download slow**
Set `HF_ENDPOINT` to a faster mirror:
```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/serve.sh
```

**Low throughput**
Increase `--gpu-memory-utilization` in `glm_server.py` (default: 0.50):
```python
"--gpu-memory-utilization", "0.80",
```

## Integrate with VS Code Copilot

Use the [Copilot LLM Gateway](https://marketplace.visualstudio.com/items?itemName=andrewbutson.github-copilot-llm-gateway)
extension. Point it at `http://localhost:11111/v1/chat/completions`.

## File Overview

| File | Role |
|------|------|
| `glm_server.py` | vLLM backend; loads model and handles inference |
| `glm_compress.py` | LiteLLM hook; strips stored reasoning blocks from assistant history |
| `server_compress.py` | LiteLLM proxy entrypoint |
| `glm_benchmark.py` | Benchmark direct vLLM, LiteLLM, or both |
| `benchmark.sh` | Start needed services, wait for health, then run the benchmark |
| `lite_llm_config.yaml` | Model routing: `glm-4.7-flash-nvfp4` → `localhost:11112` |
| `scripts/serve.sh` | Start both services |
| `scripts/install.sh` | Create venv + install `requirements.txt` |
| `kill.sh` | Stop all servers |
"""

# ── Files to include in the package ────────────────────────────────────────────
SRC_ROOT = Path(__file__).parent

PACKAGE_FILES = [
    "benchmark.sh",
    "glm_benchmark.py",
    "glm_server.py",
    "glm_compress.py",
    "server_compress.py",
    "lite_llm_config.yaml",
    "requirements.txt",
    "kill.sh",
    "build.md",
]

# ── Build logic ────────────────────────────────────────────────────────────────

def collect_files(output_dir: Path, verbose: bool = True):
    """Copy source files into output_dir."""
    for rel_path in PACKAGE_FILES:
        src = SRC_ROOT / rel_path
        dst = output_dir / rel_path
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            if verbose:
                info(f"copied: {rel_path}")
        else:
            if verbose:
                warn(f"missing (skipped): {rel_path}")


def write_scripts(output_dir: Path):
    """Write shell scripts into output_dir/scripts/."""
    scripts_dir = output_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    (scripts_dir / "install.sh").write_text(INSTALL_SH)
    make_executable(scripts_dir / "install.sh")

    (scripts_dir / "serve.sh").write_text(SERVE_SH)
    make_executable(scripts_dir / "serve.sh")

    (scripts_dir / "serve_backend.sh").write_text(SERVE_BACKEND_SH)
    make_executable(scripts_dir / "serve_backend.sh")

    (scripts_dir / "serve_proxy.sh").write_text(SERVE_PROXY_SH)
    make_executable(scripts_dir / "serve_proxy.sh")

    info("wrote: scripts/")


def write_readme(output_dir: Path):
    readme = output_dir / "README.md"
    readme.write_text(README_PACKAGE)
    info("wrote: README.md")


def write_manifest(output_dir: Path):
    import datetime
    manifest = output_dir / "MANIFEST.txt"
    with manifest.open("w") as f:
        f.write(f"# GLM-4.7-Flash-NVFP4 Serving Package\n")
        f.write(f"# Built from: {SRC_ROOT}\n")
        f.write(f"# Built at:  {datetime.datetime.now().isoformat()}\n")
        f.write("\n# Contents:\n")
        for rel_path in PACKAGE_FILES:
            status = "ok" if (SRC_ROOT / rel_path).is_file() else "MISSING"
            f.write(f"{status}: {rel_path}\n")
    info("wrote: MANIFEST.txt")


def write_env_example(output_dir: Path):
    (output_dir / ".env.example").write_text(ENV_EXAMPLE)
    info("wrote: .env.example")


def build(output_dir: Path, *, verbose: bool = True):
    if output_dir.exists():
        if verbose:
            warn(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    collect_files(output_dir, verbose=verbose)
    write_scripts(output_dir)
    write_readme(output_dir)
    write_manifest(output_dir)
    write_env_example(output_dir)

    print()
    info(f"Package written to: {output_dir}")
    size = sum(f.stat().st_size for f in output_dir.rglob("*") if f.is_file())
    info(f"Package size: {size / 1024 / 1024:.1f} MB (model not included)")


def main():
    parser = argparse.ArgumentParser(
        description="Build GLM-4.7-Flash-NVFP4 serving package.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output", "-o", default="dist",
        help="Output directory (default: dist/)",
    )
    args = parser.parse_args()

    print(f"{BOLD}GLM-4.7-Flash-NVFP4 Package Builder{RESET}\n")
    build(Path(args.output))


if __name__ == "__main__":
    main()
