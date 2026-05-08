#!/bin/bash
#
# start_draft.sh — GLM-4.7-Flash-NVFP4 with draft-model speculative decoding.
#
# ⚠️  CURRENTLY BLOCKED — no compatible public draft model exists yet.
#
# Draft-model speculative decoding works by running a small "draft" model
# to cheaply propose N tokens, then verifying them all in a single forward
# pass through the large "target" model.  Typical gains: 2–3× throughput.
#
# Requirement: the draft model MUST share the same tokenizer vocabulary as
# GadflyII/GLM-4.7-Flash-NVFP4 (tiktoken BPE, vocab_size=151,643).
#
# No compatible draft model is publicly available at the time of writing.
# Watch: https://huggingface.co/zai-org  for  zai-org/GLM-4.7-Flash-Draft
#
# Usage (once DRAFT_MODEL is available):
#   DRAFT_MODEL=zai-org/GLM-4.7-Flash-Draft ./start_draft.sh
#
# Tuning knobs (env):
#   DRAFT_MODEL           HuggingFace ID or local path  (REQUIRED)
#   VLLM_SPEC_NUM_TOKENS  Draft tokens per step          (default 3)
#   VLLM_GPU_MEM_UTIL     GPU memory fraction            (default 0.80)
#   VLLM_MAX_MODEL_LEN    Context window                 (default 65536)
#
# Logs:  logs/vllm_draft_server.log
# Ports: 11111 LiteLLM proxy · 11112 vLLM backend · 11113 token stats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -z "$DRAFT_MODEL" ]; then
    echo ""
    echo "❌  DRAFT_MODEL is not set."
    echo ""
    echo "    Draft-model speculative decoding requires a small model that"
    echo "    shares the same tokenizer as GLM-4.7-Flash-NVFP4."
    echo ""
    echo "    No compatible public draft model exists yet."
    echo "    Watch: https://huggingface.co/zai-org"
    echo ""
    echo "    Once available, run with:"
    echo "      DRAFT_MODEL=zai-org/GLM-4.7-Flash-Draft ./start_draft.sh"
    echo ""
    exit 1
fi

export DRAFT_MODEL
export VLLM_SPEC_NUM_TOKENS="${VLLM_SPEC_NUM_TOKENS:-3}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.50}"

echo "🚀 Draft config:  model=${DRAFT_MODEL}  k=${VLLM_SPEC_NUM_TOKENS}  gpu_mem=${VLLM_GPU_MEM_UTIL}"

# Detect venv
if [ -d "$SCRIPT_DIR/venv" ]; then
    VENV_PYTHON="$SCRIPT_DIR/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

mkdir -p "$SCRIPT_DIR/logs"

case "${1:-both}" in
    both)
        $VENV_PYTHON "$SCRIPT_DIR/glm_server_draft.py" &
        echo "Backend PID: $!"
        sleep 5
        $VENV_PYTHON "$SCRIPT_DIR/token_stats_server.py" &
        echo "Stats PID: $!"
        sleep 1
        $VENV_PYTHON "$SCRIPT_DIR/server_compress.py" &
        echo "Proxy PID: $!"
        echo ""
        echo "✅ All services started (draft-model speculative decoding):"
        echo "   vLLM backend:   http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Token stats:    http://0.0.0.0:11113"
        ;;
    backend)
        $VENV_PYTHON "$SCRIPT_DIR/glm_server_draft.py" &
        echo "Backend PID: $!"
        ;;
    *)
        echo "Usage: $0 [both|backend]"
        exit 1
        ;;
esac
