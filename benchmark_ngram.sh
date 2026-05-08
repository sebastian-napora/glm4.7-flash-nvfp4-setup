#!/usr/bin/env bash
#
# benchmark_ngram.sh — Compare normal vs n-gram speculative decoding.
#
# Restarts the vLLM backend twice (once without spec, once with n-gram),
# runs the same prompts through both, then prints a side-by-side comparison.
#
# Usage:
#   ./benchmark_ngram.sh                     # default: 5 runs, 512 tokens
#   ./benchmark_ngram.sh --runs 10           # more runs for stable average
#   ./benchmark_ngram.sh --max-tokens 1024   # longer outputs
#   ./benchmark_ngram.sh --test rag          # use long RAG-style prompt
#   ./benchmark_ngram.sh --test code         # use code-completion prompt
#   ./benchmark_ngram.sh --test short        # use short creative prompt
#
# Options:
#   --runs N          Measured runs per mode    (default 5)
#   --warmup N        Warmup runs per mode      (default 2)
#   --max-tokens N    Output tokens per request (default 512)
#   --test TYPE       Prompt type: short|rag|code|all (default rag)
#   --startup-timeout S  Seconds to wait for server health (default 600)
#   -h, --help        Show this help
#
# N-gram works best when output tokens appear in the prompt (RAG, code, docs).
# Results are saved to logs/benchmark_ngram_<timestamp>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=5
WARMUP=2
MAX_TOKENS=512
TEST_TYPE="rag"
STARTUP_TIMEOUT=600
POLL_INTERVAL=3

usage() {
    sed -n '3,25p' "$0" | sed 's/^# \?//'
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)           RUNS="$2";            shift 2 ;;
        --warmup)         WARMUP="$2";          shift 2 ;;
        --max-tokens)     MAX_TOKENS="$2";      shift 2 ;;
        --test)           TEST_TYPE="$2";       shift 2 ;;
        --startup-timeout) STARTUP_TIMEOUT="$2"; shift 2 ;;
        -h|--help)        usage ;;
        *) echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -x "$SCRIPT_DIR/venv/bin/python3" ]]; then
    PY="$SCRIPT_DIR/venv/bin/python3"
else
    PY="python3"
fi

VLLM_HEALTH="http://127.0.0.1:11112/health"
VLLM_URL="http://127.0.0.1:11112/v1"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="$SCRIPT_DIR/logs"
mkdir -p "$RESULTS_DIR"
RESULTS_JSON="$RESULTS_DIR/benchmark_ngram_${TIMESTAMP}.json"

# ── Prompt definitions ───────────────────────────────────────────────────────

SHORT_PROMPT="Explain what prefix caching is and why it matters for LLM inference. Be concise."

RAG_PROMPT='You are given the following document extract. Answer the question based only on this text.

DOCUMENT:
Speculative decoding is an inference optimization technique that accelerates large language model
generation without changing output quality. The technique works by using a smaller, faster draft
model to propose multiple candidate tokens in advance. The target (large) model then verifies
all proposed tokens in a single forward pass, accepting correct tokens and regenerating from the
first mismatch. Because the target model processes multiple tokens at once during verification,
the effective throughput is higher than standard autoregressive decoding. The speedup depends on
the acceptance rate — how often the draft tokens match what the target model would have generated.
N-gram speculative decoding is a special case where, instead of a separate draft model, candidate
tokens are looked up directly in the input prompt using n-gram matching. If the model is likely to
repeat or paraphrase text from the prompt (as in RAG, summarisation, or code editing), n-gram
speculative decoding can achieve significant speedups with zero additional memory overhead and no
compatibility issues. The n parameter controls the minimum length of the matched sequence.

QUESTION: What is n-gram speculative decoding and what are its main advantages compared to using
a separate draft model?'

CODE_PROMPT='Complete the following Python function. Return only the completed function with no explanation.

def compute_token_stats(completions: list[dict]) -> dict:
    """
    Given a list of OpenAI-style completion response dicts, compute:
    - total_requests: number of completions
    - total_prompt_tokens: sum of usage.prompt_tokens
    - total_completion_tokens: sum of usage.completion_tokens
    - avg_completion_tokens: average completion tokens per request
    - avg_tokens_per_second: average completion_tokens / latency_seconds
      (each dict may have a latency_seconds key)
    Returns a dict with the above keys.
    """'

declare -A PROMPTS
PROMPTS["short"]="$SHORT_PROMPT"
PROMPTS["rag"]="$RAG_PROMPT"
PROMPTS["code"]="$CODE_PROMPT"

if [[ "$TEST_TYPE" == "all" ]]; then
    TEST_TYPES=("short" "rag" "code")
else
    TEST_TYPES=("$TEST_TYPE")
fi

# ── Helper: wait for vLLM health ─────────────────────────────────────────────

wait_for_health() {
    local label="$1"
    local deadline=$(( SECONDS + STARTUP_TIMEOUT ))
    echo -n "   ⏳ Waiting for $label"
    while ! curl -fsS --max-time 5 "$VLLM_HEALTH" >/dev/null 2>&1; do
        if (( SECONDS >= deadline )); then
            echo ""
            echo "   ❌ Timed out waiting for $label after ${STARTUP_TIMEOUT}s"
            exit 1
        fi
        echo -n "."
        sleep "$POLL_INTERVAL"
    done
    echo " ready ✅"
}

# ── Helper: run benchmark and return JSON ────────────────────────────────────

run_bench() {
    local label="$1"
    local prompt="$2"
    local out_file="$3"

    $PY "$SCRIPT_DIR/glm_benchmark.py" \
        --target vllm \
        --vllm-url "$VLLM_URL" \
        --runs "$RUNS" \
        --warmup-runs "$WARMUP" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.0 \
        --prompt "$prompt" \
        --format json \
        > "$out_file" 2>/dev/null
}

# ── Helper: kill vLLM backend ────────────────────────────────────────────────

kill_backend() {
    local pid=""
    if command -v ss >/dev/null 2>&1; then
        pid=$(ss -ltnp "( sport = :11112 )" 2>/dev/null | grep -o 'pid=[0-9]\+' | head -1 | cut -d= -f2 || true)
    fi
    if [[ -z "$pid" ]] && command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti tcp:11112 -sTCP:LISTEN 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$pid" ]]; then
        echo "   🛑 Stopping vLLM backend (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        local deadline=$(( SECONDS + 30 ))
        while kill -0 "$pid" 2>/dev/null; do
            if (( SECONDS >= deadline )); then
                kill -9 "$pid" 2>/dev/null || true
                break
            fi
            sleep 1
        done
    fi
}

# ── Main ─────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║       GLM-4.7-Flash: Normal vs N-gram Speculative Benchmark  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Config:  runs=${RUNS}  warmup=${WARMUP}  max_tokens=${MAX_TOKENS}  test=${TEST_TYPE}"
echo "  Results: ${RESULTS_JSON}"
echo ""

declare -A NORMAL_RESULTS
declare -A NGRAM_RESULTS

for TEST in "${TEST_TYPES[@]}"; do
    PROMPT="${PROMPTS[$TEST]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Test: ${TEST^^}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Phase 1: Normal (no spec decoding) ──────────────────────────────────
    echo ""
    echo "  [1/2] NORMAL mode (no speculative decoding)"
    kill_backend
    echo "   🚀 Starting vLLM in normal mode..."
    $PY "$SCRIPT_DIR/glm_server.py" >> "$RESULTS_DIR/benchmark_vllm_normal.log" 2>&1 &
    NORMAL_PID=$!
    wait_for_health "vLLM normal"
    echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
    NORMAL_JSON="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_normal.json"
    run_bench "normal" "$PROMPT" "$NORMAL_JSON"
    echo "   ✅ Normal mode done."

    # ── Phase 2: N-gram speculative decoding ─────────────────────────────────
    echo ""
    echo "  [2/2] N-GRAM mode (speculative decoding)"
    kill_backend
    echo "   🚀 Starting vLLM with n-gram speculative decoding..."
    VLLM_SPEC_NGRAM=1 $PY "$SCRIPT_DIR/glm_server.py" >> "$RESULTS_DIR/benchmark_vllm_ngram.log" 2>&1 &
    NGRAM_PID=$!
    wait_for_health "vLLM ngram"
    echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
    NGRAM_JSON="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_ngram.json"
    run_bench "ngram" "$PROMPT" "$NGRAM_JSON"
    echo "   ✅ N-gram mode done."

    NORMAL_RESULTS[$TEST]="$NORMAL_JSON"
    NGRAM_RESULTS[$TEST]="$NGRAM_JSON"
done

# ── Comparison report ─────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                   COMPARISON RESULTS                        ║"
echo "╚══════════════════════════════════════════════════════════════╝"

COMBINED_JSON="${RESULTS_JSON}"

# Write a manifest so Python can find the per-test result files
MANIFEST="${RESULTS_DIR}/bench_${TIMESTAMP}_manifest.json"
$PY -c "
import json, sys
tests = sys.argv[1].split()
ts    = sys.argv[2]
rd    = sys.argv[3]
m = {}
for t in tests:
    m[t] = {
        'normal': f'{rd}/bench_{ts}_{t}_normal.json',
        'ngram':  f'{rd}/bench_{ts}_{t}_ngram.json',
    }
print(json.dumps(m))
" "${TEST_TYPES[*]}" "$TIMESTAMP" "$RESULTS_DIR" > "$MANIFEST"

$PY - "$MANIFEST" "$COMBINED_JSON" <<'PYEOF'
import json, sys
from pathlib import Path

manifest_path = sys.argv[1]
combined_path = sys.argv[2]

manifest = json.loads(Path(manifest_path).read_text())
test_types = list(manifest.keys())

records = []
print()
print(f"  {'Test':<8}  {'Metric':<28}  {'Normal':>10}  {'N-gram':>10}  {'Δ':>8}  {'Speedup':>8}")
print(f"  {'-'*8}  {'-'*28}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")

all_results = {}

for t in test_types:
    nfile = Path(manifest[t]["normal"])
    gfile = Path(manifest[t]["ngram"])

    try:
        nd = json.loads(nfile.read_text())[0]
        gd = json.loads(gfile.read_text())[0]
    except Exception as e:
        print(f"  ⚠️  Could not load results for {t}: {e}")
        continue

    def fmt(v, unit=""):
        if v is None: return "n/a"
        return f"{v:.2f}{unit}"

    ntps = nd.get("avg_completion_tokens_per_second")
    gtps = gd.get("avg_completion_tokens_per_second")
    nlat = nd.get("avg_latency_seconds")
    glat = gd.get("avg_latency_seconds")

    if ntps and gtps:
        delta_tps = gtps - ntps
        speedup   = gtps / ntps if ntps > 0 else None
        sign      = "+" if delta_tps >= 0 else ""
        sp_str    = f"{speedup:.2f}x" if speedup else "n/a"
        print(f"  {t:<8}  {'Avg completion tok/s':<28}  {fmt(ntps):>10}  {fmt(gtps):>10}  {sign+fmt(delta_tps):>8}  {sp_str:>8}")

    if nlat and glat:
        delta_lat = glat - nlat
        sign      = "+" if delta_lat >= 0 else ""
        lat_sp    = nlat / glat if glat > 0 else None
        lat_sp_str= f"{lat_sp:.2f}x" if lat_sp else "n/a"
        print(f"  {t:<8}  {'Avg latency (s)':<28}  {fmt(nlat,'s'):>10}  {fmt(glat,'s'):>10}  {sign+fmt(delta_lat,'s'):>8}  {lat_sp_str:>8}")

    print()

    all_results[t] = {"normal": nd, "ngram": gd}

# Save combined JSON
out = Path(combined_path)
out.write_text(json.dumps(all_results, indent=2))
print(f"  💾 Full results saved to: {out}")
print()

# Print interpretation
print("  Interpretation:")
print("  • Speedup > 1.0x = n-gram helped  (output tokens appeared in prompt)")
print("  • Speedup ≈ 1.0x = no effect       (short or purely creative prompts)")
print("  • Speedup < 1.0x = n-gram overhead  (very short prompts, low match rate)")
print()
PYEOF
