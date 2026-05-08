#!/bin/bash
#
# start_ngram.sh — GLM-4.7-Flash-NVFP4 with N-gram speculative decoding.
#
# Drafts tokens by matching n-grams already present in the input prompt.
# No separate draft model — zero extra memory, zero compatibility issues.
# Works with the current NVFP4 checkpoint out of the box.
#
# Best for workloads where the output repeats content from the input:
#   • RAG / document Q&A          (answer cites source text)
#   • Code completion / editing   (output echoes variable/function names)
#   • Summarisation               (output paraphrases input phrases)
#   • Long-context chat           (model refers back to earlier turns)
#
# Less effective for:
#   • Pure creative generation    (novel text with no prompt overlap)
#   • Short prompts               (little source material to match against)
#
# Tuning knobs (env):
#   VLLM_SPEC_NGRAM_K    — speculative tokens per step (default 5)
#   VLLM_SPEC_NGRAM_MIN  — min n-gram size to match    (default 3)
#   VLLM_SPEC_NGRAM_MAX  — max n-gram size to match    (default 5)
#
# Usage:
#   ./start_ngram.sh           # vLLM (n-gram) + token-stats + LiteLLM proxy
#   ./start_ngram.sh backend   # vLLM only
#   ./start_ngram.sh proxy     # LiteLLM proxy only
#
# Logs:  logs/vllm_server.log, logs/litellm_proxy.log
# Ports: 11111 LiteLLM proxy · 11112 vLLM backend · 11113 token stats

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export VLLM_SPEC_NGRAM=1
export VLLM_SPEC_NGRAM_K="${VLLM_SPEC_NGRAM_K:-5}"
export VLLM_SPEC_NGRAM_MIN="${VLLM_SPEC_NGRAM_MIN:-3}"
export VLLM_SPEC_NGRAM_MAX="${VLLM_SPEC_NGRAM_MAX:-5}"

echo "🔍 N-gram config:  k=${VLLM_SPEC_NGRAM_K}  ngram=[${VLLM_SPEC_NGRAM_MIN}..${VLLM_SPEC_NGRAM_MAX}]"

exec "$SCRIPT_DIR/start.sh" "${1:-both}"
