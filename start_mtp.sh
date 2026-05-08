#!/bin/bash
#
# start_mtp.sh — MTP speculative decoding for GLM-4.7-Flash-NVFP4.
#
# ⚠️  CURRENTLY BLOCKED — prints error and exits immediately.
#
# The GadflyII/GLM-4.7-Flash-NVFP4 checkpoint NVFP4-quantizes the MTP
# head's eh_proj linear, but vLLM hard-codes it as a plain nn.Linear.
# Loading crashes with:
#   KeyError: 'model.layers.47.eh_proj.input_global_scale'
#
# Fix needed upstream (checkpoint or vLLM).  Track progress in README.md.
# To bypass the guard and reproduce the crash: VLLM_SPEC_MTP_FORCE=1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo ""
echo "❌  start_mtp.sh is currently unavailable."
echo ""
echo "    MTP speculative decoding is INCOMPATIBLE with the"
echo "    GadflyII/GLM-4.7-Flash-NVFP4 checkpoint (vLLM ≤ 0.19.x)."
echo ""
echo "    Cause:  eh_proj is NVFP4-quantized in the checkpoint but"
echo "            vllm/model_executor/models/glm4_moe_lite_mtp.py"
echo "            hard-codes  self.eh_proj = nn.Linear  (unquantized)."
echo "    Error:  KeyError: 'model.layers.47.eh_proj.input_global_scale'"
echo ""
echo "    Use the normal stack instead:"
echo "      ./start.sh both"
echo ""
echo "    To force anyway (will crash after ~3 min of loading):"
echo "      VLLM_SPEC_MTP_FORCE=1 VLLM_SPEC_MTP=1 ./start.sh both"
echo ""
exit 1
