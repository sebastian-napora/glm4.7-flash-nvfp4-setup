#!/bin/bash
cd /home/sna/ai-projects/glm_4_7_flash_setup
exec ./venv/bin/python glm_server.py 2>&1 | tee -a logs/vllm_backend.log