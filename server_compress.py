#!/usr/bin/env python3
"""
LiteLLM proxy for GLM-4.7-Flash-NVFP4.

Architecture:
    Copilot -> LiteLLM (11111) -> vLLM (11112)

Usage:
    # Terminal 1: start vLLM backend on 11112
    python3 glm_server.py

    # Terminal 2: start LiteLLM proxy on 11111
    python3 server_compress.py
"""

import os
import logging
from pathlib import Path

import litellm

import glm_compress       # noqa: F401 — strips thinking tokens from history
import glm_token_tracker  # noqa: F401 — records per-request token usage
glm_compress.register()
glm_token_tracker.register()

# Setup detailed logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

# Create dedicated logger for image requests
litellm_logger = logging.getLogger("litellm.image_request")
litellm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "litellm_image_requests.log"))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(fh)

# Also log to console
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(logging.Formatter("%(asctime)s %(name)-25s %(levelname)-8s %(message)s"))
litellm_logger.addHandler(ch)

litellm_logger.info("=" * 60)
litellm_logger.info("LiteLLM GLM-4.7-Flash-NVFP4 Proxy Started")

# Enable litellm verbose mode for debugging
os.environ["LITELLM_LOG"] = "DEBUG"
os.environ["LITELLM_REQUEST_LOGGING"] = "true"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(name)-25s %(levelname)-8s %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "litellm_detailed.log")),
        logging.StreamHandler()
    ]
)

# Get the litellm logger and set to DEBUG
litellm_main_logger = logging.getLogger("litellm")
litellm_main_logger.setLevel(logging.DEBUG)

logger = logging.getLogger("server_compress")

LITELLM_PORT = os.environ.get("LITE_LLM_PROXY_PORT", "11111")
LITELLM_HOST = os.environ.get("LITE_LLM_PROXY_HOST", "0.0.0.0")
CONFIG_PATH = Path(__file__).parent / "lite_llm_config.yaml"

logger.info("Starting LiteLLM proxy on %s:%s", LITELLM_HOST, LITELLM_PORT)
logger.info("Config: %s", CONFIG_PATH)
logger.info("Assistant history sanitization enabled")

os.environ.pop("LITELLM_MASTER_KEY", None)
os.environ.pop("LITELLM_SALT_KEY", None)
os.environ["CONFIG_FILE_PATH"] = str(CONFIG_PATH)

logger.info("=" * 60)
logger.info("LiteLLM proxy starting in-process with GLM history sanitizer")
logger.info("=" * 60)

# Verify callback is registered before starting
from litellm.integrations.custom_logger import CustomLogger
registered_callbacks = [cb for cb in litellm.callbacks if isinstance(cb, CustomLogger)]
logger.info("Registered custom callbacks: %d", len(registered_callbacks))
for cb in registered_callbacks:
    logger.info("  - %s", type(cb).__name__)

# Run uvicorn in the same process (NOT via exec — preserving callbacks)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "litellm.proxy.proxy_server:app",
        host=LITELLM_HOST,
        port=int(LITELLM_PORT),
        reload=False,
        log_level="debug",
    )
