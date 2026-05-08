#!/bin/bash
#
# start_sglang.sh — GLM-4.7-Flash (BF16) with SGLang + EAGLE speculative decoding.
#
# Uses the original zai-org/GLM-4.7-Flash (BF16) model via SGLang's EAGLE
# algorithm: 3 speculative steps × 4 draft tokens per cycle.
# Typical throughput gain: 2–4× vs standard autoregressive decoding.
#
# ⚠️  Requirements (different from the vLLM stack):
#   1. SGLang special build for GLM-4.7-Flash support (see below)
#   2. zai-org/GLM-4.7-Flash model downloaded (~62 GB BF16)
#
# Install SGLang:
#   source venv/bin/activate
#   pip install "sglang[all]==0.5.11" \
#       --extra-index-url https://flashinfer.ai/whl/cu130/torch2.9/
#   pip install git+https://github.com/huggingface/transformers.git
#
# Model is already in HuggingFace cache:
#   GadflyII/GLM-4.7-Flash-NVFP4  (~20 GB, compressed-tensors NVFP4)
#
# Usage:
#   ./start_sglang.sh           # SGLang backend + token-stats + LiteLLM proxy
#   ./start_sglang.sh backend   # SGLang only
#   ./start_sglang.sh proxy     # LiteLLM proxy only (connects to running SGLang)
#
# Tuning knobs (env):
#   SGLANG_MODEL               Model path (default zai-org/GLM-4.7-Flash)
#   SGLANG_SPEC_NUM_STEPS      EAGLE steps (default 3)
#   SGLANG_SPEC_EAGLE_TOPK     Top-k per step (default 1)
#   SGLANG_SPEC_DRAFT_TOKENS   Draft tokens per cycle (default 4)
#   SGLANG_MEM_FRACTION        Static memory fraction (default 0.45)
#   SGLANG_MAX_MODEL_LEN       Context window (default 180000)
#   SGLANG_TP_SIZE             Tensor parallel size (default 1)
#
# Logs:  logs/sglang_server.log, logs/litellm_proxy.log
# Ports: 11111 LiteLLM proxy · 11112 SGLang backend · 11113 token stats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect venv
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

mkdir -p "$SCRIPT_DIR/logs"

export SGLANG_MODEL="${SGLANG_MODEL:-GadflyII/GLM-4.7-Flash-NVFP4}"
export SGLANG_SPEC_NUM_STEPS="${SGLANG_SPEC_NUM_STEPS:-3}"
export SGLANG_SPEC_EAGLE_TOPK="${SGLANG_SPEC_EAGLE_TOPK:-1}"
export SGLANG_SPEC_DRAFT_TOKENS="${SGLANG_SPEC_DRAFT_TOKENS:-4}"
export SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-0.45}"
export SGLANG_MAX_MODEL_LEN="${SGLANG_MAX_MODEL_LEN:-180000}"
export SGLANG_TP_SIZE="${SGLANG_TP_SIZE:-1}"

echo "🦅 SGLang EAGLE config:"
echo "   model=${SGLANG_MODEL}"
echo "   steps=${SGLANG_SPEC_NUM_STEPS}  topk=${SGLANG_SPEC_EAGLE_TOPK}  draft_tokens=${SGLANG_SPEC_DRAFT_TOKENS}"
echo "   mem=${SGLANG_MEM_FRACTION}  ctx=${SGLANG_MAX_MODEL_LEN}  tp=${SGLANG_TP_SIZE}"
echo ""

start_backend() {
    echo "🚀 Starting SGLang backend (port 11112)..."
    $VENV_PYTHON "$SCRIPT_DIR/glm_sglang_server.py" \
        >> "$SCRIPT_DIR/logs/sglang_server.log" 2>&1 &
    echo "Backend PID: $!"
}

start_stats() {
    echo "📊 Starting token stats server (port 11113)..."
    $VENV_PYTHON "$SCRIPT_DIR/token_stats_server.py" &
    echo "Stats PID: $!"
}

start_proxy() {
    echo "🔀 Starting LiteLLM proxy (port 11111)..."
    $VENV_PYTHON "$SCRIPT_DIR/server_compress.py" &
    echo "Proxy PID: $!"
}

case "${1:-both}" in
    both)
        start_backend
        sleep 5
        start_stats
        sleep 1
        start_proxy
        echo ""
        echo "✅ All services started (SGLang EAGLE):"
        echo "   SGLang backend: http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Token stats:    http://0.0.0.0:11113"
        echo ""
        echo "Tail logs with:  tail -f logs/sglang_server.log"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy|stats]"
        exit 1
        ;;
esac
