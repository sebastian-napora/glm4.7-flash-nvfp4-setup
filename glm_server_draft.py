#!/usr/bin/env python3
"""
vLLM API Server for GLM-4.7-Flash-NVFP4 — Draft-Model Speculative Decoding.

Uses a small compatible draft model to propose tokens that the main model
then verifies in a single forward pass.  Typically gives 2–3× throughput
improvement when draft acceptance rate is high.

Architecture:
  Copilot -> LiteLLM (11111) -> vLLM (11112)
                                ├── Target: GadflyII/GLM-4.7-Flash-NVFP4
                                └── Draft:  $DRAFT_MODEL  (small, same vocab)

Draft model requirements
─────────────────────────
The draft model MUST share the same tokenizer vocabulary as
GadflyII/GLM-4.7-Flash-NVFP4 (tiktoken BPE, vocab_size=151,643).
A model from a different family with a different tokenizer will produce
garbage tokens and achieve 0% acceptance rate.

No public GLM-4.7-Flash draft model exists at the time of writing.
When one becomes available, set DRAFT_MODEL to its HuggingFace path.

Candidate future options to watch:
  • zai-org/GLM-4.7-Flash-Draft          (official; not yet published)
  • Any small (1B–4B) GLM model sharing GLM-4.7's tokenizer

Env knobs
─────────
  DRAFT_MODEL           HuggingFace model ID or local path (required)
  VLLM_SPEC_NUM_TOKENS  Draft tokens per step (default 3)
  VLLM_GPU_MEM_UTIL     GPU memory fraction  (default 0.80; draft needs headroom)
  VLLM_MAX_MODEL_LEN    Context window        (default 65536; reduce if OOM)
  VLLM_OPT_LEVEL        Optimisation level    (default 1; 3 = torch.compile)

Ports: 11112 vLLM backend · 11111 LiteLLM proxy · 11113 token stats
"""

import sys
import os
import logging
import traceback

os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"
os.environ["VLLM_WORKER_LOGGING_LEVEL"] = "DEBUG"

LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
vllm_logger = logging.getLogger("vllm.draft_server")
vllm_logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(os.path.join(LOG_DIR, "vllm_draft_server.log"))
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(message)s"))
vllm_logger.addHandler(fh)

_venv_bin = os.path.join(os.path.dirname(__file__), "venv", "bin")
if os.path.exists(_venv_bin) and _venv_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = _venv_bin + os.pathsep + os.environ.get("PATH", "")

_venv_lib = os.path.join(os.path.dirname(__file__), "venv", "lib", "python3.12", "site-packages")
if os.path.exists(_venv_lib) and _venv_lib not in sys.path:
    sys.path.insert(0, _venv_lib)


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
    import json

    DRAFT_MODEL     = os.environ.get("DRAFT_MODEL", "")
    SPEC_NUM_TOKENS = os.environ.get("VLLM_SPEC_NUM_TOKENS", "3")
    GPU_MEM_UTIL    = os.environ.get("VLLM_GPU_MEM_UTIL",    "0.50")
    MAX_MODEL_LEN   = os.environ.get("VLLM_MAX_MODEL_LEN",   "65536")
    OPT_LEVEL       = os.environ.get("VLLM_OPT_LEVEL",       "1")

    if not DRAFT_MODEL:
        raise SystemExit(
            "❌  DRAFT_MODEL is not set.\n\n"
            "    Draft-model speculative decoding requires a small model that\n"
            "    shares the same tokenizer vocabulary as GLM-4.7-Flash-NVFP4\n"
            "    (tiktoken BPE, vocab_size=151,643).\n\n"
            "    No compatible public draft model exists yet.  Check back for:\n"
            "      zai-org/GLM-4.7-Flash-Draft\n\n"
            "    Once available, run with:\n"
            "      DRAFT_MODEL=zai-org/GLM-4.7-Flash-Draft ./start_draft.sh\n"
        )

    parser = FlexibleArgumentParser(prog="glm-4.7-flash-draft-server")
    subparsers = parser.add_subparsers(dest="command")
    serve_parser = subparsers.add_parser("serve")
    serve_parser = make_arg_parser(serve_parser)

    spec_cfg = json.dumps({
        "method": "draft_model",
        "model": DRAFT_MODEL,
        "num_speculative_tokens": int(SPEC_NUM_TOKENS),
    })

    argv = [
        "GadflyII/GLM-4.7-Flash-NVFP4",
        "--trust-remote-code",
        "--dtype", "auto",
        "--load-format", "safetensors",
        "--max-model-len", MAX_MODEL_LEN,
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        "--max-num-batched-tokens", "65536",
        "--optimization-level", OPT_LEVEL,
        "--port", "11112",
        "--host", "0.0.0.0",
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
        "--speculative-config", spec_cfg,
    ]

    print(f"🚀 Draft-model speculative decoding ENABLED")
    print(f"   Target: GadflyII/GLM-4.7-Flash-NVFP4")
    print(f"   Draft:  {DRAFT_MODEL}")
    print(f"   k={SPEC_NUM_TOKENS}  gpu_mem={GPU_MEM_UTIL}  ctx={MAX_MODEL_LEN}")

    args = serve_parser.parse_args(argv)
    args.command = "serve"
    args.model_tag = argv[0]
    args.model = args.model_tag
    validate_parsed_serve_args(args)

    print("⏳ Loading GLM-4.7-Flash-NVFP4 + draft model…")
    async with build_async_engine_client(args) as engine_client:
        supported_tasks = await engine_client.get_supported_tasks()
        model_config = engine_client.model_config

        app = build_app(args, supported_tasks, model_config)
        await init_app_state(engine_client, app.state, args, supported_tasks)

        from fastapi import Request
        from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest

        serving_chat: OpenAIServingChat = app.state.openai_serving_chat
        original_create = serving_chat.create_chat_completion

        async def logged_create(request: ChatCompletionRequest, raw_request: Request = None, **kwargs):
            vllm_logger.info("Request: model=%s stream=%s msgs=%d", request.model, request.stream, len(request.messages))
            try:
                result = await original_create(request, raw_request, **kwargs)
                vllm_logger.info("Request completed")
                return result
            except Exception as e:
                vllm_logger.error("Request failed: %s\n%s", e, traceback.format_exc())
                raise

        serving_chat.create_chat_completion = logged_create

        @app.get("/health")
        async def health():
            return {"status": "ok", "model": "GadflyII/GLM-4.7-Flash-NVFP4", "speculative": "draft_model", "draft": DRAFT_MODEL}

        listen_address, sock = setup_server(args)
        print(f"\n🚀 GLM-4.7-Flash (Draft-Model Spec) @ {MAX_MODEL_LEN} tokens")
        print(f"📡 Chat API: http://0.0.0.0:11112/v1/chat/completions")
        print(f"❤️  Health:  http://0.0.0.0:11112/health")
        await serve_http(
            app,
            sock=sock,
            host=args.host,
            port=args.port,
            log_level="info",
            timeout_keep_alive=30,
        )


if __name__ == "__main__":
    import uvloop
    uvloop.run(main())


def run():
    import uvloop
    uvloop.run(main())
