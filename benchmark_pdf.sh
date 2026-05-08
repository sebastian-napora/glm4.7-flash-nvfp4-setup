#!/usr/bin/env bash
#
# benchmark_pdf.sh — benchmark normal vs n-gram using a real PDF as RAG context.
#
# Usage:
#   ./benchmark_pdf.sh                          # uses Zaproszenie.pdf with default questions
#   ./benchmark_pdf.sh --pdf /path/to/file.pdf  # custom PDF
#   ./benchmark_pdf.sh --runs 10 --warmup 3
#
# The entire PDF text is sent as context in each request (full-document RAG).
#

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PY="${SCRIPT_DIR}/venv/bin/python"
VLLM_URL="http://127.0.0.1:11112/v1"
HEALTH_URL="http://127.0.0.1:11112/health"
STARTUP_TIMEOUT=600

# ── Defaults ──────────────────────────────────────────────────────────────────
PDF_FILE="${SCRIPT_DIR}/Zaproszenie.pdf"
RUNS=5
WARMUP=2
MAX_TOKENS=512
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="${SCRIPT_DIR}/logs"
mkdir -p "$RESULTS_DIR"

# ── Argument parsing ───────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --pdf)       PDF_FILE="$2";    shift 2 ;;
        --runs)      RUNS="$2";        shift 2 ;;
        --warmup)    WARMUP="$2";      shift 2 ;;
        --max-tokens) MAX_TOKENS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

[[ -f "$PDF_FILE" ]] || { echo "❌ PDF not found: $PDF_FILE"; exit 1; }

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      GLM-4.7-Flash: Normal vs N-gram — PDF RAG Benchmark     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "  PDF:     $PDF_FILE"
echo "  Config:  runs=$RUNS  warmup=$WARMUP  max_tokens=$MAX_TOKENS"
echo ""

# ── Extract PDF text once ─────────────────────────────────────────────────────
PDF_TEXT_FILE="${RESULTS_DIR}/pdf_text_cache.txt"
echo "  📄 Extracting PDF text..."
"$PY" - "$PDF_FILE" "$PDF_TEXT_FILE" <<'PYEOF'
import sys
from pypdf import PdfReader

pdf_path, out_path = sys.argv[1], sys.argv[2]
reader = PdfReader(pdf_path)
pages = []
for i, page in enumerate(reader.pages):
    text = page.extract_text() or ""
    if text.strip():
        pages.append(f"[Page {i+1}]\n{text.strip()}")
full_text = "\n\n".join(pages)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(full_text)
words = len(full_text.split())
chars = len(full_text)
print(f"  ✅ Extracted {len(pages)} pages  |  ~{words} words  |  {chars} chars")
PYEOF

# ── Questions to ask about the document ───────────────────────────────────────
# These are generic enough to work with any business document
QUESTIONS=(
    "Summarize the main purpose and key points of this document."
    "What are the most important requirements or conditions mentioned?"
    "List all deadlines, dates, or timeframes mentioned in the document."
    "What parties are involved and what are their responsibilities?"
    "What financial amounts, prices, or costs are mentioned?"
)

# ── Helper functions ──────────────────────────────────────────────────────────
kill_backend() {
    local pid
    pid=$(lsof -ti tcp:11112 2>/dev/null | head -1) || true
    if [[ -n "$pid" ]]; then
        echo "   🛑 Stopping vLLM backend (PID $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 5
    fi
}

wait_for_health() {
    local label="$1"
    local elapsed=0
    printf "   ⏳ Waiting for %s" "$label"
    while ! curl -fsS --max-time 5 "$HEALTH_URL" > /dev/null 2>&1; do
        sleep 2
        elapsed=$((elapsed + 2))
        printf "."
        if [[ $elapsed -ge $STARTUP_TIMEOUT ]]; then
            echo " TIMEOUT ❌"
            exit 1
        fi
    done
    echo " ready ✅"
}

discover_model() {
    "$PY" - "$VLLM_URL" <<'PYEOF'
import json, sys
from urllib import request, error

url = sys.argv[1]
try:
    with request.urlopen(url + "/models", timeout=10) as resp:
        data = json.loads(resp.read())
    models = data.get("data", [])
    if models and isinstance(models[0].get("id"), str):
        print(models[0]["id"])
    else:
        print("")
except Exception as e:
    print("")
PYEOF
}

run_single_request() {
    local doc_text="$1"
    local question="$2"
    local max_tok="$3"
    local model_id="$4"

    local prompt="You are given a document. Answer the question based only on the document content.

DOCUMENT:
${doc_text}

QUESTION: ${question}"

    "$PY" - "$VLLM_URL" "$prompt" "$max_tok" "$model_id" <<'PYEOF'
import json, sys, time
from urllib import request, error

url, prompt, max_tokens, model_id = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
payload = json.dumps({
    "model": model_id,
    "messages": [{"role": "user", "content": prompt}],
    "max_tokens": max_tokens,
    "temperature": 0.0,
    "stream": False
}).encode()

start = time.monotonic()
try:
    req = request.Request(url + "/chat/completions",
                          data=payload,
                          headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    latency = time.monotonic() - start
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    tps = completion_tokens / latency if latency > 0 else 0
    print(json.dumps({"latency": latency, "completion_tokens": completion_tokens, "tps": tps}))
except Exception as e:
    print(json.dumps({"error": str(e), "latency": 0, "completion_tokens": 0, "tps": 0}), file=sys.stderr)
    print(json.dumps({"latency": 0, "completion_tokens": 0, "tps": 0}))
PYEOF
}

run_benchmark() {
    local mode="$1"
    local doc_text="$2"
    local out_file="$3"
    local model_id="$4"

    local total_latency=0
    local total_tps=0
    local count=0
    local all_warmup=$((RUNS + WARMUP))

    for ((i=1; i<=all_warmup; i++)); do
        local q_idx=$(( (i - 1) % ${#QUESTIONS[@]} ))
        local question="${QUESTIONS[$q_idx]}"
        local result err_msg
        result=$(run_single_request "$doc_text" "$question" "$MAX_TOKENS" "$model_id" 2>/tmp/pdf_bench_req_err.txt)
        err_msg=$(cat /tmp/pdf_bench_req_err.txt 2>/dev/null || true)
        if [[ -n "$err_msg" ]]; then
            echo "   ⚠️  Request error: $err_msg" >&2
        fi

        if [[ $i -le $WARMUP ]]; then
            printf "     warm %d/%d\r" "$i" "$WARMUP"
            continue
        fi

        local lat tps
        lat=$(echo "$result" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['latency'])")
        tps=$(echo "$result" | "$PY" -c "import json,sys; d=json.load(sys.stdin); print(d['tps'])")
        total_latency=$(echo "$total_latency + $lat" | bc)
        total_tps=$(echo "$total_tps + $tps" | bc)
        count=$((count + 1))
        printf "     run %d/%d  %.1f tok/s  %.2fs latency\n" "$count" "$RUNS" "$tps" "$lat"
    done

    local avg_lat avg_tps
    avg_lat=$(echo "scale=3; $total_latency / $count" | bc)
    avg_tps=$(echo "scale=2; $total_tps / $count" | bc)

    "$PY" -c "
import json
data = {'mode': '$mode', 'runs': $count, 'avg_tps': $avg_tps, 'avg_latency': $avg_lat}
with open('$out_file', 'w') as f:
    json.dump(data, f, indent=2)
print(f'   avg: {$avg_tps:.2f} tok/s  |  {$avg_lat:.2f}s latency')
"
}

# ── Load extracted text ────────────────────────────────────────────────────────
DOC_TEXT=$(cat "$PDF_TEXT_FILE")

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 1/2: NORMAL mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
kill_backend
echo "   🚀 Starting vLLM (normal)..."
"$PY" "${SCRIPT_DIR}/glm_server.py" >> "${RESULTS_DIR}/pdf_bench_normal.log" 2>&1 &
wait_for_health "vLLM normal"
NORMAL_JSON="${RESULTS_DIR}/pdf_bench_${TIMESTAMP}_normal.json"
NORMAL_MODEL=$(discover_model)
echo "   🤖 Model: ${NORMAL_MODEL:-<unknown>}"
run_benchmark "normal" "$DOC_TEXT" "$NORMAL_JSON" "$NORMAL_MODEL"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Phase 2/2: N-GRAM mode"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
kill_backend
echo "   🚀 Starting vLLM (n-gram k=5, min=3, max=5)..."
VLLM_SPEC_NGRAM=1 "$PY" "${SCRIPT_DIR}/glm_server.py" >> "${RESULTS_DIR}/pdf_bench_ngram.log" 2>&1 &
wait_for_health "vLLM ngram"
NGRAM_JSON="${RESULTS_DIR}/pdf_bench_${TIMESTAMP}_ngram.json"
NGRAM_MODEL=$(discover_model)
echo "   🤖 Model: ${NGRAM_MODEL:-<unknown>}"
run_benchmark "ngram" "$DOC_TEXT" "$NGRAM_JSON" "$NGRAM_MODEL"

# ── Final comparison ──────────────────────────────────────────────────────────
echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    COMPARISON RESULTS                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"

"$PY" - "$NORMAL_JSON" "$NGRAM_JSON" "${RESULTS_DIR}/pdf_bench_${TIMESTAMP}_combined.json" <<'PYEOF'
import json, sys
from pathlib import Path

nf, gf, out = sys.argv[1], sys.argv[2], sys.argv[3]
nd = json.loads(Path(nf).read_text())
gd = json.loads(Path(gf).read_text())

ntps, gtps = nd["avg_tps"], gd["avg_tps"]
nlat, glat = nd["avg_latency"], gd["avg_latency"]
speedup = gtps / ntps if ntps > 0 else 0
delta_tps = gtps - ntps
sign = "+" if delta_tps >= 0 else ""

print()
print(f"  {'Metric':<30}  {'Normal':>10}  {'N-gram':>10}  {'Δ':>8}  {'Speedup':>8}")
print(f"  {'-'*30}  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}")
print(f"  {'Avg completion tok/s':<30}  {ntps:>10.2f}  {gtps:>10.2f}  {sign+f'{delta_tps:.2f}':>8}  {speedup:>7.2f}x")

lat_sp = nlat / glat if glat > 0 else 0
dlat = glat - nlat
sign2 = "+" if dlat >= 0 else ""
print(f"  {'Avg latency (s)':<30}  {nlat:>9.2f}s  {glat:>9.2f}s  {sign2+f'{dlat:.2f}s':>8}  {lat_sp:>7.2f}x")
print()

combined = {"normal": nd, "ngram": gd, "speedup": speedup, "delta_tps": delta_tps}
Path(out).write_text(json.dumps(combined, indent=2))
print(f"  💾 Saved: {out}")
print()
if speedup >= 1.2:
    print("  ✅ N-gram delivers meaningful speedup on this document.")
elif speedup >= 1.05:
    print("  ℹ️  Modest speedup — model partially quotes the document.")
else:
    print("  ⚠️  Minimal speedup — model paraphrases rather than quotes.")
print()
PYEOF
