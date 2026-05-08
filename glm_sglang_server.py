#!/usr/bin/env python3
"""
SGLang API Server for GLM-4.7-Flash — EAGLE Speculative Decoding.

Uses SGLang's built-in EAGLE speculative algorithm which predicts multiple
draft tokens using the model's own hidden states.  The official recipe
from zai-org/GLM-4.7-Flash uses 3 EAGLE steps × 4 draft tokens, yielding
up to 3–4× throughput on decode-heavy workloads.

Architecture:
  Copilot -> LiteLLM (11111) -> SGLang (11112)

Model: GadflyII/GLM-4.7-Flash-NVFP4 (compressed-tensors w4a4 NVFP4, ~20 GB)
       SGLang v0.5.11 supports NVFP4 via compressed_tensors_w4a4_nvfp4_moe.
       The checkpoint includes num_nextn_predict_layers=1 for embedded EAGLE.

EAGLE vs MTP (vLLM) comparison:
  ┌──────────────┬──────────────────────────────────────────────────┐
  │ Feature      │ EAGLE (SGLang)       │ MTP (vLLM)               │
  ├──────────────┼──────────────────────┼──────────────────────────┤
  │ Draft tokens │ 4 per cycle          │ 1 per cycle              │
  │ Steps        │ 3 EAGLE steps        │ 1 step                   │
  │ Model        │ NVFP4 (~20 GB)       │ NVFP4 (~20 GB)           │
  │ Status       │ Works now ✅         │ Slower until compile fix  │
  │ Framework    │ SGLang               │ vLLM                     │
  └──────────────┴──────────────────────┴──────────────────────────┘

Requirements
────────────
  SGLang (stable, GLM-4.7-Flash supported since v0.4.x, mainlined Jan 2026):
    pip install "sglang[all]==0.5.11" \\
        --extra-index-url https://flashinfer.ai/whl/cu130/torch2.9/
    pip install git+https://github.com/huggingface/transformers.git

  For Blackwell (GB10 CC 12.1), the following flags are automatically added:
    --attention-backend triton
    --speculative-draft-attention-backend triton

  Model is already downloaded:
    GadflyII/GLM-4.7-Flash-NVFP4  (~20 GB, in HuggingFace cache)

Env knobs
─────────
  SGLANG_MODEL               Model path (default GadflyII/GLM-4.7-Flash-NVFP4)
  SGLANG_PORT                Server port (default 11112)
  SGLANG_HOST                Bind address (default 0.0.0.0)
  SGLANG_MEM_FRACTION        Static memory fraction (default 0.45)
  SGLANG_SPEC_NUM_STEPS      EAGLE steps (default 3)
  SGLANG_SPEC_EAGLE_TOPK     Top-k per step (default 1)
  SGLANG_SPEC_DRAFT_TOKENS   Draft tokens per cycle (default 4)
  SGLANG_TP_SIZE             Tensor parallel size (default 1 for single GB10)
  SGLANG_MAX_MODEL_LEN       Context window (default 180000)
  SGLANG_DISABLE_CUDA_GRAPH  Disable all CUDA graphs (default 1=disabled).
                             NVFP4+EAGLE has an irrecoverable KV-cache head-dim
                             shape mismatch during graph capture ([N,20,64] vs
                             [N,20,256]).  Set to 0 to re-enable when upstream
                             fixes the incompatibility.
"""

import sys
import os
import subprocess

_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if os.path.exists(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if os.path.exists(_venv_lib) and _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)

# ── Check SGLang is installed ────────────────────────────────────────────────
try:
    import sglang  # noqa: F401
except ImportError:
    print("❌  SGLang is not installed.")
    print()
    print("   Install SGLang v0.5.11 (stable, GLM-4.7-Flash mainlined Jan 2026):")
    print()
    print('     pip install "sglang[all]==0.5.11" \\')
    print("         --extra-index-url https://flashinfer.ai/whl/cu130/torch2.9/")
    print("     pip install git+https://github.com/huggingface/transformers.git")
    print()
    print("   Then re-run:  ./start_sglang.sh")
    sys.exit(1)


def main():
    MODEL             = os.environ.get("SGLANG_MODEL",            "GadflyII/GLM-4.7-Flash-NVFP4")
    PORT              = os.environ.get("SGLANG_PORT",             "11112")
    HOST              = os.environ.get("SGLANG_HOST",             "0.0.0.0")
    MEM_FRACTION      = os.environ.get("SGLANG_MEM_FRACTION",     "0.45")
    SPEC_NUM_STEPS    = os.environ.get("SGLANG_SPEC_NUM_STEPS",   "3")
    SPEC_EAGLE_TOPK   = os.environ.get("SGLANG_SPEC_EAGLE_TOPK",  "1")
    SPEC_DRAFT_TOKENS = os.environ.get("SGLANG_SPEC_DRAFT_TOKENS","4")
    TP_SIZE           = os.environ.get("SGLANG_TP_SIZE",          "1")
    MAX_MODEL_LEN     = os.environ.get("SGLANG_MAX_MODEL_LEN",    "180000")
    # NVFP4 + EAGLE has an irrecoverable head-dim shape mismatch during CUDA
    # graph capture ([N,20,64] vs [N,20,256]) that --cuda-graph-max-bs cannot
    # fix (it's a KV dtype/packing issue, not a batch-size issue).
    # Disabling CUDA graphs avoids the crash; EAGLE multi-token speculative
    # decoding still accelerates decode — only graph kernel-fusion is lost.
    # Set SGLANG_DISABLE_CUDA_GRAPH=0 to re-enable when upstream fixes this.
    DISABLE_CUDA_GRAPH = os.environ.get("SGLANG_DISABLE_CUDA_GRAPH", "1")

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path",                  MODEL,
        "--tp-size",                     TP_SIZE,
        "--tool-call-parser",            "glm47",
        "--reasoning-parser",            "glm45",
        "--speculative-algorithm",       "EAGLE",
        "--speculative-num-steps",       SPEC_NUM_STEPS,
        "--speculative-eagle-topk",      SPEC_EAGLE_TOPK,
        "--speculative-num-draft-tokens",SPEC_DRAFT_TOKENS,
        "--mem-fraction-static",         MEM_FRACTION,
        "--context-length",              MAX_MODEL_LEN,
        "--served-model-name",           "glm-4.7-flash",
        "--host",                        HOST,
        "--port",                        PORT,
        # Blackwell (GB10, CC 12.1) requires Triton attention backends
        "--attention-backend",           "triton",
        "--speculative-draft-attention-backend", "triton",
        # Piecewise CUDA graph breaks NVFP4 fp4_quantize JIT on Blackwell
        "--disable-piecewise-cuda-graph",
    ]

    # Disable CUDA graphs to work around the NVFP4 + EAGLE KV-cache
    # head-dim mismatch that crashes during graph capture.
    if DISABLE_CUDA_GRAPH == "1":
        cmd.append("--disable-cuda-graph")

    print(f"🚀 SGLang EAGLE speculative decoding")
    print(f"   Model:  {MODEL}  (NVFP4, ~20 GB)")
    print(f"   EAGLE:  steps={SPEC_NUM_STEPS}  topk={SPEC_EAGLE_TOPK}  draft_tokens={SPEC_DRAFT_TOKENS}")
    print(f"   TP:     {TP_SIZE}   ctx={MAX_MODEL_LEN}   mem={MEM_FRACTION}")
    print(f"   URL:    http://{HOST}:{PORT}")
    print()
    print("⏳ Loading model — NVFP4 ~20 GB, allow 1–2 min on GB10…")
    print()

    os.execvp(sys.executable, cmd)


if __name__ == "__main__":
    main()
