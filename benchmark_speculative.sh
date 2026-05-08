#!/usr/bin/env bash
#
# benchmark_speculative.sh — 3-way speculative decoding comparison:
#   1. vLLM  — normal autoregressive decoding (baseline)
#   2. vLLM  — N-gram speculative decoding
#   3. SGLang — EAGLE speculative decoding (embedded NextN draft head)
#
# Usage:
#   ./benchmark_speculative.sh                    # default: rag test, 5 runs
#   ./benchmark_speculative.sh --runs 10          # more runs
#   ./benchmark_speculative.sh --test code        # code-completion prompt
#   ./benchmark_speculative.sh --test all         # short + rag + code
#   ./benchmark_speculative.sh --skip-ngram       # only vLLM normal vs SGLang EAGLE
#   ./benchmark_speculative.sh --skip-sglang      # only vLLM normal vs N-gram
#   ./benchmark_speculative.sh --only-eagle       # only SGLang EAGLE
#
# Options:
#   --runs N            Measured runs per mode        (default 5)
#   --warmup N          Warmup runs per mode          (default 2)
#   --max-tokens N      Output tokens per request     (default 512)
#   --test TYPE         Prompt type: short|rag|code|all  (default rag)
#   --skip-ngram        Skip the N-gram phase
#   --skip-sglang       Skip the SGLang EAGLE phase
#   --skip-normal       Skip the vLLM normal baseline
#   --only-eagle        Run only the SGLang EAGLE phase
#   --startup-timeout S Seconds to wait for health   (default 600)
#   --eagle-mem F       SGLang static memory fraction (default 0.42)
#   --eagle-context N   SGLang context length         (default 32768)
#   -h, --help          Show this help
#
# Results saved to logs/bench_speculative_<timestamp>.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

RUNS=5
WARMUP=2
MAX_TOKENS=512
TEST_TYPE="rag"
STARTUP_TIMEOUT=900
POLL_INTERVAL=3
SKIP_NGRAM=0
SKIP_SGLANG=0
SKIP_NORMAL=0
EAGLE_MEM_FRACTION=0.42
EAGLE_MAX_MODEL_LEN=32768
EAGLE_SPEC_NUM_STEPS=3
EAGLE_SPEC_EAGLE_TOPK=1
EAGLE_SPEC_DRAFT_TOKENS=4

usage() { sed -n '3,30p' "$0" | sed 's/^# \?//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runs)            RUNS="$2";             shift 2 ;;
        --warmup)          WARMUP="$2";           shift 2 ;;
        --max-tokens)      MAX_TOKENS="$2";       shift 2 ;;
        --test)            TEST_TYPE="$2";        shift 2 ;;
        --skip-ngram)      SKIP_NGRAM=1;          shift ;;
        --skip-sglang)     SKIP_SGLANG=1;         shift ;;
        --skip-normal)     SKIP_NORMAL=1;         shift ;;
        --only-eagle)      SKIP_NORMAL=1; SKIP_NGRAM=1; SKIP_SGLANG=0; shift ;;
        --startup-timeout) STARTUP_TIMEOUT="$2";  shift 2 ;;
        --eagle-mem)       EAGLE_MEM_FRACTION="$2"; shift 2 ;;
        --eagle-context)   EAGLE_MAX_MODEL_LEN="$2"; shift 2 ;;
        -h|--help)         usage ;;
        *) echo "❌ Unknown argument: $1"; exit 1 ;;
    esac
done

PY="${SCRIPT_DIR}/venv/bin/python3"
BACKEND_URL="http://127.0.0.1:11112/v1"
BACKEND_HEALTH="http://127.0.0.1:11112/health"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$RESULTS_DIR"

# ── Prompts ───────────────────────────────────────────────────────────────────

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

# ── Helpers ───────────────────────────────────────────────────────────────────

kill_backend() {
    local pid=""
    if command -v ss >/dev/null 2>&1; then
        pid=$(ss -ltnp "( sport = :11112 )" 2>/dev/null \
              | grep -o 'pid=[0-9]\+' | head -1 | cut -d= -f2 || true)
    fi
    if [[ -z "$pid" ]] && command -v lsof >/dev/null 2>&1; then
        pid=$(lsof -ti tcp:11112 -sTCP:LISTEN 2>/dev/null | head -1 || true)
    fi
    if [[ -n "$pid" ]]; then
        echo "   🛑 Stopping backend (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        local deadline=$(( SECONDS + 30 ))
        while kill -0 "$pid" 2>/dev/null; do
            (( SECONDS >= deadline )) && { kill -9 "$pid" 2>/dev/null || true; break; }
            sleep 1
        done
    fi
}

wait_for_health() {
    local label="$1"
    local health_url="${2:-$BACKEND_HEALTH}"
    local monitor_pid="${3:-}"
    local deadline=$(( SECONDS + STARTUP_TIMEOUT ))
    echo -n "   ⏳ Waiting for $label"
    while ! curl -fsS --max-time 5 "$health_url" >/dev/null 2>&1; do
        if [[ -n "$monitor_pid" ]] && ! kill -0 "$monitor_pid" 2>/dev/null; then
            echo " ❌ process died (crash/OOM — check log)"
            return 1
        fi
        (( SECONDS >= deadline )) && { echo ""; echo "   ❌ Timed out after ${STARTUP_TIMEOUT}s"; return 1; }
        echo -n "."
        sleep "$POLL_INTERVAL"
    done
    echo " ready ✅"
    return 0
}

run_bench() {
    local label="$1"
    local prompt="$2"
    local out_file="$3"
    "$PY" "$SCRIPT_DIR/glm_benchmark.py" \
        --target vllm \
        --vllm-url "$BACKEND_URL" \
        --runs "$RUNS" \
        --warmup-runs "$WARMUP" \
        --max-tokens "$MAX_TOKENS" \
        --temperature 0.0 \
        --prompt "$prompt" \
        --format json \
        > "$out_file" 2>/dev/null
}

# ── Header ────────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   GLM-4.7-Flash: Normal vs N-gram vs SGLang EAGLE Benchmark  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  Config:  runs=${RUNS}  warmup=${WARMUP}  max_tokens=${MAX_TOKENS}  test=${TEST_TYPE}"
modes=""
[[ "$SKIP_NORMAL" == "0" ]] && modes+="vLLM-normal"
[[ "$SKIP_NGRAM"  == "0" ]] && modes+=" | vLLM-ngram"
[[ "$SKIP_SGLANG" == "0" ]] && modes+=" | SGLang-EAGLE"
modes="${modes# | }"
echo "  Modes:   ${modes}"
echo ""

# ── Per-test loop ─────────────────────────────────────────────────────────────

declare -A NORMAL_JSON_MAP
declare -A NGRAM_JSON_MAP
declare -A EAGLE_JSON_MAP

total_phases=$(( (SKIP_NORMAL == 0 ? 1 : 0) + (SKIP_NGRAM == 0 ? 1 : 0) + (SKIP_SGLANG == 0 ? 1 : 0) ))
if (( total_phases == 0 )); then
    echo "❌ Nothing to run: all benchmark modes are skipped."
    exit 1
fi

for TEST in "${TEST_TYPES[@]}"; do
    phase=0
    PROMPT="${PROMPTS[$TEST]}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Prompt type: ${TEST^^}"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # ── Phase 1: vLLM normal ────────────────────────────────────────────────
    if [[ "$SKIP_NORMAL" == "0" ]]; then
        phase=$(( phase + 1 ))
        echo ""
        echo "  [${phase}/${total_phases}] vLLM — normal autoregressive (baseline)"
        kill_backend
        echo "   🚀 Starting vLLM normal..."
        "$PY" "${SCRIPT_DIR}/glm_server.py" \
            >> "${RESULTS_DIR}/bench_spec_vllm_normal.log" 2>&1 &
        wait_for_health "vLLM normal"
        NORMAL_OUT="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_normal.json"
        echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
        run_bench "normal" "$PROMPT" "$NORMAL_OUT"
        NORMAL_JSON_MAP[$TEST]="$NORMAL_OUT"
        echo "   ✅ vLLM normal done."
    fi

    # ── Phase 2: vLLM N-gram ────────────────────────────────────────────────
    if [[ "$SKIP_NGRAM" == "0" ]]; then
        phase=$(( phase + 1 ))
        echo ""
        echo "  [${phase}/${total_phases}] vLLM — N-gram speculative decoding"
        kill_backend
        echo "   🚀 Starting vLLM with N-gram (k=5, min=3, max=5)..."
        VLLM_SPEC_NGRAM=1 "$PY" "${SCRIPT_DIR}/glm_server.py" \
            >> "${RESULTS_DIR}/bench_spec_vllm_ngram.log" 2>&1 &
        wait_for_health "vLLM ngram"
        NGRAM_OUT="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_ngram.json"
        echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
        run_bench "ngram" "$PROMPT" "$NGRAM_OUT"
        NGRAM_JSON_MAP[$TEST]="$NGRAM_OUT"
        echo "   ✅ vLLM N-gram done."
    fi

    # ── Phase 3: SGLang EAGLE ───────────────────────────────────────────────
    if [[ "$SKIP_SGLANG" == "0" ]]; then
        phase=$(( phase + 1 ))
        echo ""
        echo "  [${phase}/${total_phases}] SGLang — EAGLE speculative decoding (embedded NextN)"
        kill_backend
        echo "   🚀 Starting SGLang EAGLE..."
        echo "   ⚙️  EAGLE env: mem=${SGLANG_MEM_FRACTION:-$EAGLE_MEM_FRACTION} ctx=${SGLANG_MAX_MODEL_LEN:-$EAGLE_MAX_MODEL_LEN} cuda_graph=${SGLANG_DISABLE_CUDA_GRAPH:-1}"
        SGLANG_MEM_FRACTION="${SGLANG_MEM_FRACTION:-$EAGLE_MEM_FRACTION}" \
        SGLANG_MAX_MODEL_LEN="${SGLANG_MAX_MODEL_LEN:-$EAGLE_MAX_MODEL_LEN}" \
        SGLANG_SPEC_NUM_STEPS="${SGLANG_SPEC_NUM_STEPS:-$EAGLE_SPEC_NUM_STEPS}" \
        SGLANG_SPEC_EAGLE_TOPK="${SGLANG_SPEC_EAGLE_TOPK:-$EAGLE_SPEC_EAGLE_TOPK}" \
        SGLANG_SPEC_DRAFT_TOKENS="${SGLANG_SPEC_DRAFT_TOKENS:-$EAGLE_SPEC_DRAFT_TOKENS}" \
        SGLANG_DISABLE_CUDA_GRAPH="${SGLANG_DISABLE_CUDA_GRAPH:-1}" \
            "$PY" "${SCRIPT_DIR}/glm_sglang_server.py" \
            >> "${RESULTS_DIR}/bench_spec_sglang_eagle.log" 2>&1 &
        SGLANG_PID=$!
        # SGLang health: /health is basic readiness; /health_generate verifies
        # actual inference pipeline. Use /health_generate for a solid gate.
        if wait_for_health "SGLang EAGLE" "http://127.0.0.1:11112/health_generate" "$SGLANG_PID"; then
            EAGLE_OUT="${RESULTS_DIR}/bench_${TIMESTAMP}_${TEST}_eagle.json"
            echo "   📊 Running ${RUNS} measured requests (${WARMUP} warmup)..."
            run_bench "eagle" "$PROMPT" "$EAGLE_OUT"
            EAGLE_JSON_MAP[$TEST]="$EAGLE_OUT"
            echo "   ✅ SGLang EAGLE done."
        else
            echo "   ⚠️  SGLang EAGLE skipped — startup failed (OOM or crash)."
            echo "   💡 Check logs/bench_spec_sglang_eagle.log for details."
            tail -40 "${RESULTS_DIR}/bench_spec_sglang_eagle.log" 2>/dev/null || true
        fi
    fi

    # ── Per-test inline results ─────────────────────────────────────────────────
    echo ""
    echo "  ┌─ ${TEST^^} results ───────────────────────────────────────────────────────┐"
    "$PY" - "${NORMAL_JSON_MAP[$TEST]:-}" "${NGRAM_JSON_MAP[$TEST]:-}" "${EAGLE_JSON_MAP[$TEST]:-}" <<'PYEOF'
import json, sys
from pathlib import Path

def load(p):
    if not p: return None
    try:
        data = json.loads(Path(p).read_text())
        return data[0] if isinstance(data, list) else data
    except: return None

def tps(d): return d.get("avg_completion_tokens_per_second") if d else None
def lat(d): return d.get("avg_latency_seconds") if d else None
def fmt(v, u=""): return f"{v:.2f}{u}" if v is not None else "  n/a"
def spd(a, b): return f"{b/a:.2f}x" if a and b and a > 0 else "  n/a"

nd = load(sys.argv[1] if len(sys.argv) > 1 else "")
gd = load(sys.argv[2] if len(sys.argv) > 2 else "")
ed = load(sys.argv[3] if len(sys.argv) > 3 else "")
w = 12
print(f"  │  {'Metric':<30}  {'Normal':>{w}}  {'N-gram':>{w}}  {'EAGLE':>{w}}  │")
print(f"  │  {'─'*30}  {'─'*w}  {'─'*w}  {'─'*w}  │")
print(f"  │  {'tok/s':<30}  {fmt(tps(nd)):>{w}}  {fmt(tps(gd)):>{w}}  {fmt(tps(ed)):>{w}}  │")
print(f"  │  {'latency (s)':<30}  {fmt(lat(nd),'s'):>{w}}  {fmt(lat(gd),'s'):>{w}}  {fmt(lat(ed),'s'):>{w}}  │")
sp_ng = spd(tps(nd), tps(gd))
sp_ea = spd(tps(nd), tps(ed))
normal_sp = "  1.00x" if nd else "  n/a"
print(f"  │  {'speedup vs normal':<30}  {normal_sp:>{w}}  {sp_ng:>{w}}  {sp_ea:>{w}}  │")
print(f"  └{'─'*60}┘")
PYEOF
done

# ── Comparison report ─────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COMPARISON RESULTS                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

COMBINED_JSON="${RESULTS_DIR}/bench_speculative_${TIMESTAMP}.json"

# Build manifest for the Python comparison script
MANIFEST="${RESULTS_DIR}/bench_${TIMESTAMP}_manifest.json"
"$PY" - "${TEST_TYPES[*]}" "$TIMESTAMP" "$RESULTS_DIR" \
       "$SKIP_NGRAM" "$SKIP_SGLANG" "$SKIP_NORMAL" > "$MANIFEST" <<'PYEOF'
import json, sys
tests, ts, rd, skip_ngram, skip_sglang, skip_normal = \
    sys.argv[1].split(), sys.argv[2], sys.argv[3], sys.argv[4]=="1", sys.argv[5]=="1", sys.argv[6]=="1"
m = {}
for t in tests:
    m[t] = {}
    if not skip_normal:
        m[t]["normal"] = f"{rd}/bench_{ts}_{t}_normal.json"
    if not skip_ngram:
        m[t]["ngram"] = f"{rd}/bench_{ts}_{t}_ngram.json"
    if not skip_sglang:
        m[t]["eagle"] = f"{rd}/bench_{ts}_{t}_eagle.json"
print(json.dumps(m))
PYEOF

"$PY" - "$MANIFEST" "$COMBINED_JSON" <<'PYEOF'
import json, sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
combined_path = sys.argv[2]

def load(path):
    try:
        data = json.loads(Path(path).read_text())
        return data[0] if isinstance(data, list) else data
    except Exception as e:
        return None

def tps(d):  return d.get("avg_completion_tokens_per_second") if d else None
def lat(d):  return d.get("avg_latency_seconds") if d else None
def fmt(v, unit=""): return f"{v:.2f}{unit}" if v is not None else "  n/a"
def speedup(a, b):   return f"{b/a:.2f}x" if a and b and a > 0 else "  n/a"
def delta(a, b, unit=""):
    if a is None or b is None: return "  n/a"
    d = b - a
    return ("+" if d >= 0 else "") + f"{d:.2f}{unit}"

all_results = {}

for test, paths in manifest.items():
    nd = load(paths.get("normal"))
    gd = load(paths.get("ngram"))
    ed = load(paths.get("eagle"))

    modes = ["normal"]
    if gd: modes.append("ngram")
    if ed: modes.append("eagle")

    col_w = 12
    header_mode = f"{'Normal':>{col_w}}  {'N-gram':>{col_w}}  {'EAGLE':>{col_w}}"
    sep = "─" * (32 + col_w * 3 + 4)

    print()
    print(f"  ┌─ {test.upper()} ─{'─'*(len(sep)-5-len(test))}┐")
    print(f"  │  {'Metric':<30}  {header_mode}  │")
    print(f"  │  {'─'*30}  {'─'*col_w}  {'─'*col_w}  {'─'*col_w}  │")

    # tok/s row
    row_tps = f"  │  {'tok/s':<30}  {fmt(tps(nd)):>{col_w}}  {fmt(tps(gd)):>{col_w}}  {fmt(tps(ed)):>{col_w}}  │"
    print(row_tps)

    # latency row
    row_lat = f"  │  {'latency (s)':<30}  {fmt(lat(nd),'s'):>{col_w}}  {fmt(lat(gd),'s'):>{col_w}}  {fmt(lat(ed),'s'):>{col_w}}  │"
    print(row_lat)

    # speedup vs normal row
    sp_ng  = speedup(tps(nd), tps(gd))
    sp_ea  = speedup(tps(nd), tps(ed))
    normal_sp = "  1.00x" if nd else "  n/a"
    row_sp = f"  │  {'speedup vs normal':<30}  {normal_sp:>{col_w}}  {sp_ng:>{col_w}}  {sp_ea:>{col_w}}  │"
    print(row_sp)

    print(f"  └{'─'*(len(sep)-1)}┘")

    # Interpretation
    def interpret(sp_str, name):
        try:
            v = float(sp_str.strip().rstrip("x"))
            if v >= 1.3:   return f"  ✅ {name}: {v:.2f}× — meaningful speedup"
            elif v >= 1.05: return f"  ℹ️  {name}: {v:.2f}× — modest speedup"
            elif v >= 0.95: return f"  ➖ {name}: {v:.2f}× — no significant change"
            else:           return f"  ⚠️  {name}: {v:.2f}× — slower than baseline"
        except: return f"  ─  {name}: n/a"

    if gd and nd: print(interpret(sp_ng, "vLLM N-gram"))
    if ed and nd: print(interpret(sp_ea, "SGLang EAGLE"))

    all_results[test] = {
        "normal": nd, "ngram": gd, "eagle": ed,
        "speedup_ngram": sp_ng if gd else None,
        "speedup_eagle": sp_ea if ed else None,
    }

print()
combined = Path(combined_path)
combined.write_text(json.dumps(all_results, indent=2))
print(f"  💾 Results saved: {combined}")
print()
PYEOF

echo ""
echo "  Logs:"
[[ "$SKIP_NORMAL" == "0" ]] && echo "    vLLM normal : logs/bench_spec_vllm_normal.log"
[[ "$SKIP_NGRAM"  == "0" ]] && echo "    vLLM N-gram : logs/bench_spec_vllm_ngram.log"
[[ "$SKIP_SGLANG" == "0" ]] && echo "    SGLang EAGLE: logs/bench_spec_sglang_eagle.log"
echo ""
