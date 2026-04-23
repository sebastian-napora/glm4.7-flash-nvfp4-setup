#!/bin/bash
# Kill GLM-4.7-Flash-NVFP4 serving stack

# Kill by process name
pkill -f "glm_server.py" 2>/dev/null
pkill -f "server_compress.py" 2>/dev/null
pkill -f "token_stats_server" 2>/dev/null

# Kill orphaned vLLM engine cores (can survive pkill by name)
pkill -9 -f "VLLM::EngineCore" 2>/dev/null

# Kill any uvicorn on our ports (orphaned or missed)
fuser -k 11111/tcp 2>/dev/null
fuser -k 11112/tcp 2>/dev/null
fuser -k 11113/tcp 2>/dev/null

# Final sweep — kill any lingering python on our ports
for port in 11111 11112 11113; do
    pid=$(ss -tlnp 2>/dev/null | grep ":$port " | grep -oP 'pid=\K[0-9]+')
    if [ -n "$pid" ]; then
        kill -9 "$pid" 2>/dev/null
        echo "Killed PID $pid on port $port"
    fi
done

echo "✅ GLM serving processes killed"
ss -tlnp 2>/dev/null | grep -E '11111|11112' || echo "  (ports are free)"
ps aux | grep -E 'glm|VLLM|server_compress' | grep -v grep | grep -v pkill || echo "  (no remaining processes)"
