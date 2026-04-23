#!/usr/bin/env python3
"""
Local vLLM API Server for GLM-4.7-Flash-NVFP4 on NVIDIA GB10
with a plain OpenAI-compatible chat endpoint.

Architecture:
  Copilot -> LiteLLM (11111) -> vLLM (11112)

Model: GadflyII/GLM-4.7-Flash-NVFP4 (30B-A3B MoE, NVFP4 quantized)
Context: 202,752 tokens max
Requires: vLLM 0.14.0+, transformers 5.0.0+
"""

import sys
import os
import logging
import traceback

# Allow long max_model_len (model's native limit is 202752)
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"

# Enable VLLM request logging
os.environ["VLLM_WORKER_LOGGING_LEVEL"] = "DEBUG"

# Setup logging
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
vllm_logger = logging.getLogger("vllm.image_request")
vllm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "vllm_image_requests.log"))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
vllm_logger.addHandler(fh)
vllm_logger.info("=" * 60)
vllm_logger.info("vLLM GLM-4.7-Flash-NVFP4 Server Started")

# Ensure venv packages take priority
_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if os.path.exists(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if os.path.exists(_venv_lib) and _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)

# ─── Main ─────────────────────────────────────────────────────────────────────
async def main():
    from vllm.entrypoints.openai.api_server import (
        build_async_engine_client,
        build_app,
        init_app_state,
        setup_server,
        serve_http,
    )
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    # ── Build vLLM args ─────────────────────────────────────────────────────────
    parser = FlexibleArgumentParser(prog="glm-4.7-flash-server")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve")
    serve_parser = make_arg_parser(serve_parser)

    argv = [
        "GadflyII/GLM-4.7-Flash-NVFP4",
        "--trust-remote-code",
        "--dtype", "auto",
        "--max-model-len", "202752",       # leave headroom for system prompt
        "--gpu-memory-utilization", "0.70",
        "--enforce-eager",
        "--disable-log-stats",
        "--port", "11112",
        "--host", "0.0.0.0",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
    ]

    args = serve_parser.parse_args(argv)
    args.command = "serve"
    args.model_tag = argv[0]
    args.model = args.model_tag
    validate_parsed_serve_args(args)

    # ── Start engine (blocking until model is loaded) ──────────────────────────
    print("⏳ Loading GLM-4.7-Flash-NVFP4… this may take a few minutes.")
    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        # ── Build FastAPI app ───────────────────────────────────────────────────
        app = build_app(args, supported_tasks, model_config)
        await init_app_state(engine_client, app.state, args, supported_tasks)

        from fastapi import Request
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
        )

        serving_chat: OpenAIServingChat = app.state.openai_serving_chat

        # Wrap the original method to log all requests
        original_create = serving_chat.create_chat_completion

        async def logged_create_chat_completion(request: ChatCompletionRequest, raw_request: Request = None, **kwargs):
            vllm_logger.info("=" * 60)
            vllm_logger.info("/v1/chat/completions request received")
            vllm_logger.info("Model: %s", request.model)
            vllm_logger.info("Stream: %s", request.stream)
            vllm_logger.info("Message count: %d", len(request.messages))

            for i, msg in enumerate(request.messages):
                if isinstance(msg, dict):
                    msg_role = msg.get("role", "unknown")
                    content = msg.get("content", "")
                else:
                    msg_role = getattr(msg, "role", "unknown")
                    content = getattr(msg, "content", "")

                if isinstance(content, list):
                    image_types = [c for c in content if c.get("type") == "image_url"]
                    text_parts = [c for c in content if c.get("type") == "text"]
                    vllm_logger.info(
                        "  msg[%d] role=%s: %d image_url items, %d text items",
                        i, msg_role, len(image_types), len(text_parts)
                    )
                    for j, part in enumerate(content):
                        if part.get("type") == "image_url":
                            img_url = part.get("image_url", {})
                            if isinstance(img_url, dict):
                                url = img_url.get("url", "")[:100]
                                detail = img_url.get("detail", "not_set")
                            else:
                                url = str(img_url)[:100]
                                detail = "not_set"
                            vllm_logger.info(
                                "    image_url[%d]: url_len=%d, detail=%s, url=%s...",
                                j, len(img_url.get("url", "")) if isinstance(img_url, dict) else len(str(img_url)), detail, url
                            )
                        elif part.get("type") == "text":
                            vllm_logger.info("    text[%d]: %s", j, part.get("text", "")[:300])
                elif isinstance(content, str):
                    vllm_logger.info("  msg[%d] role=%s: %s", i, msg_role, content[:300])

            if hasattr(request, "extra_body") and request.extra_body:
                vllm_logger.info("Extra body: %s", request.extra_body)

            vllm_logger.info("=" * 60)

            try:
                result = await original_create(request, raw_request, **kwargs)
                vllm_logger.info("Request completed successfully")
                return result
            except Exception as e:
                vllm_logger.error("Request failed: %s", str(e))
                vllm_logger.error(traceback.format_exc())
                raise

        serving_chat.create_chat_completion = logged_create_chat_completion

        # ─── Health check ──────────────────────────────────────────────────────
        @app.get("/health")
        async def health():
            return {"status": "ok", "model": "GadflyII/GLM-4.7-Flash-NVFP4"}

        # ── Serve ───────────────────────────────────────────────────────────────
        listen_address, sock = setup_server(args)
        print(f"\n🚀 GLM-4.7-Flash-NVFP4 Server @ 131K Context")
        print(f"📡 Chat API:    http://0.0.0.0:11112/v1/chat/completions")
        print(f"❤️  Health:     http://0.0.0.0:11112/health")
        print()
        await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level=args.uvicorn_log_level,
            timeout_keep_alive=30,
        )


if __name__ == "__main__":
    import uvloop
    uvloop.run(main())


def run():
    """Synchronous entry point for console_scripts."""
    import uvloop
    uvloop.run(main())
