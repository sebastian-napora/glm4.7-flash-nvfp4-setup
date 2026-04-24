#!/bin/bash
#
# Start GLM-4.7-Flash-NVFP4 serving stack
#
# Usage:
#   ./start.sh          # start both vLLM backend and LiteLLM proxy
#   ./start.sh backend  # start only vLLM backend (11112)
#   ./start.sh proxy    # start only LiteLLM proxy (11111)
#
# Architecture:
#   Copilot -> LiteLLM (11111) -> vLLM (11112)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Detect or create venv
if [ -d "venv" ]; then
    VENV_PYTHON="venv/bin/python3"
elif [ -d "../lunch-model/venv" ]; then
    VENV_PYTHON="../lunch-model/venv/bin/python3"
else
    VENV_PYTHON="python3"
fi

mkdir -p logs

start_backend() {
    echo "🚀 Starting vLLM backend (port 11112)..."
    $VENV_PYTHON glm_server.py &
    echo "Backend PID: $!"
}

new_token_session() {
    # Create a fresh session ID and write it to the shared session file.
    # Both the proxy and the stats server read this file at startup, so
    # every ./start.sh run begins a clean session automatically.
    NEW_SID=$($VENV_PYTHON -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
import glm_token_tracker
sid = glm_token_tracker.new_session()
print(sid, end='')
" 2>/dev/null)
    echo "📊 New token session: ${NEW_SID:-unknown}"
}

start_stats() {
    echo "🚀 Starting token stats server (port 11113)..."
    $VENV_PYTHON token_stats_server.py &
    echo "Stats PID: $!"
}

start_proxy() {
    echo "🚀 Starting LiteLLM proxy (port 11111)..."
    $VENV_PYTHON server_compress.py &
    echo "Proxy PID: $!"
}

case "${1:-both}" in
    both)
        start_backend
        new_token_session
        echo "Waiting 5s for backend to initialize..."
        sleep 5
        start_stats
        sleep 1
        start_proxy
        echo ""
        echo "✅ All services started:"
        echo "   vLLM backend:   http://0.0.0.0:11112"
        echo "   LiteLLM proxy:  http://0.0.0.0:11111"
        echo "   Token stats:    http://0.0.0.0:11113"
        echo ""
        echo "Test with:"
        echo "  curl http://localhost:11112/health"
        echo "  curl http://localhost:11111/health"
        ;;
    backend)
        start_backend
        ;;
    proxy)
        new_token_session
        start_proxy
        ;;
    stats)
        start_stats
        ;;
    *)
        echo "Usage: $0 [both|backend|proxy]"
        exit 1
        ;;
esac
