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

# FlashInfer MoE FP4 is disabled — it triggers CUDA misaligned address errors on
# GB10 (compute capability 12.1, beyond PyTorch's supported range 8.0–12.0).
# Falls back to VLLM_CUTLASS MoE backend which is stable on this hardware.
# NOTE: --moe-backend flashinfer would override this; keep it out of argv until
#       upstream FlashInfer ships CC 12.1 kernels.
os.environ["VLLM_USE_FLASHINFER_MOE_FP4"] = "0"

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

    # ── Speculative decoding (MTP) ──────────────────────────────────────────
    # GLM-4.7-Flash ships built-in Multi-Token Prediction layers
    # (config.json: "num_nextn_predict_layers": 1). vLLM can use them as a
    # zero-overhead draft head — no second model required.
    #
    # Enable:   VLLM_SPEC_MTP=1 ./start.sh   (or use start_mtp.sh)
    # Tuning:   VLLM_SPEC_NUM_TOKENS=1       (recommended; vLLM GLM recipe)
    #
    # Per the vLLM GLM recipe, k=1 keeps acceptance >90 % and gives the best
    # net throughput; k=3 increases accept length but lowers acceptance rate
    # and hurts overall tok/s.
    #
    # ⚠️  KNOWN INCOMPATIBILITY with `GadflyII/GLM-4.7-Flash-NVFP4` ⚠️
    # That checkpoint NVFP4-quantizes the MTP head's `eh_proj` linear, but
    # vLLM (≤ 0.19.x and current main) hard-codes `self.eh_proj = nn.Linear`
    # in `glm4_moe_lite_mtp.py` (unquantized). Loading then crashes with:
    #   KeyError: 'model.layers.47.eh_proj.input_global_scale'
    # Fix is upstream: either the checkpoint's quantization_config.ignore
    # must list `eh_proj`, or vLLM must wrap it in a quant-aware Linear.
    # Until then, MTP is disabled by default and a hard-stop is raised if
    # someone forces it on with this model. Override at your own risk with
    # VLLM_SPEC_MTP_FORCE=1.
    SPEC_MTP        = os.environ.get("VLLM_SPEC_MTP",        "0") == "1"
    SPEC_MTP_FORCE  = os.environ.get("VLLM_SPEC_MTP_FORCE",  "0") == "1"
    SPEC_NUM_TOKENS = os.environ.get("VLLM_SPEC_NUM_TOKENS", "1")

    # ── Speculative decoding (N-gram / Prompt Lookup) ───────────────────────
    # Drafts tokens by matching n-grams from the input prompt.  No separate
    # model required — zero extra memory, zero compatibility issues.
    # Best for: RAG, summarisation, code completion (output repeats input).
    # Enable:  VLLM_SPEC_NGRAM=1 ./start.sh   (or use start_ngram.sh)
    # Tuning:  VLLM_SPEC_NGRAM_K   — draft tokens per step (default 5)
    #          VLLM_SPEC_NGRAM_MIN  — min n-gram length to match (default 3)
    #          VLLM_SPEC_NGRAM_MAX  — max n-gram length to match (default 5)
    SPEC_NGRAM     = os.environ.get("VLLM_SPEC_NGRAM",     "0") == "1"
    SPEC_NGRAM_K   = os.environ.get("VLLM_SPEC_NGRAM_K",   "5")
    SPEC_NGRAM_MIN = os.environ.get("VLLM_SPEC_NGRAM_MIN", "3")
    SPEC_NGRAM_MAX = os.environ.get("VLLM_SPEC_NGRAM_MAX", "5")

    # MTP runs an extra forward per step → reduce KV-cache budget a touch on
    # the 128 GB unified GB10 to leave headroom for the draft activations.
    GPU_MEM_UTIL    = os.environ.get(
        "VLLM_GPU_MEM_UTIL",
        "0.50" if SPEC_MTP else "0.50",
    )

    # ── Env-configurable knobs ──────────────────────────────────────────────
    # MAX_MODEL_LEN: 202752 is the model's native limit, but for Copilot use
    # 32K–65K is more than enough and reserves far less KV-cache RAM.
    # Rule of thumb on GB10: each 65536 tokens of context ≈ 4-8 GB KV cache.
    MAX_MODEL_LEN = os.environ.get("VLLM_MAX_MODEL_LEN", "202752")

    # OPTIMIZATION_LEVEL:
    #   ⚠️  GLM-4.7-Flash-NVFP4 does NOT support torch.compile (vLLM warns:
    #       "torch.compile is turned on, but the model does not support it").
    #   Level 3 = torch.compile + Inductor + CUDA graphs → burns 30-50 GB of
    #             unified RAM during startup for zero benefit on this model.
    #   Level 1 = CUDA graphs only (no Inductor) → same runtime speed,
    #             ~40 GB less peak RAM, much faster cold start.  ← DEFAULT
    #   Level 0 = eager only → fastest cold start, lowest memory,
    #             ~10-15% slower decode.
    OPT_LEVEL = os.environ.get("VLLM_OPT_LEVEL", "1")

    argv = [
        "GadflyII/GLM-4.7-Flash-NVFP4",
        "--trust-remote-code",
        "--dtype", "auto",
        "--load-format", "safetensors",
        "--max-model-len", MAX_MODEL_LEN,
        # ── Memory ──────────────────────────────────────────────────────────────
        # GB10 has 128 GB unified RAM (CPU+GPU shared).
        # 0.75 × 128 ≈ 96 GB reserved for vLLM (weights + KV cache pool).
        # With OPT_LEVEL=1 (no Inductor), peak load stays well under 100 GB.
        "--gpu-memory-utilization", GPU_MEM_UTIL,
        # ── Attention & decode ───────────────────────────────────────────────────
        # NOTE: --attention-backend flashinfer is NOT compatible with GLM-4.7
        # ("head_size not supported", "MLA not supported"). Stick with the
        # default Flash-Attention backend chosen by vLLM for this model.
        # ── Batching ────────────────────────────────────────────────────────────
        "--max-num-batched-tokens", "65536",
        # ── Optimization level ───────────────────────────────────────────────────
        # Default 1 = CUDA graphs only; torch.compile unsupported on this model.
        # Set VLLM_OPT_LEVEL=3 to try full compile (expect ~121 GB peak RAM).
        "--optimization-level", OPT_LEVEL,
        "--port", "11112",
        "--host", "0.0.0.0",
        "--enable-auto-tool-choice",
        "--enable-prefix-caching",         # reuse KV blocks for shared prefixes (system prompt, tool schemas)
        "--tool-call-parser", "glm47",
        "--reasoning-parser", "glm45",
    ]

    # ── Append speculative-decoding flags if enabled ────────────────────────
    import json as _json
    if SPEC_NGRAM:
        spec_cfg = _json.dumps({
            "method": "ngram",
            "num_speculative_tokens": int(SPEC_NGRAM_K),
            "prompt_lookup_min": int(SPEC_NGRAM_MIN),
            "prompt_lookup_max": int(SPEC_NGRAM_MAX),
        })
        argv += ["--speculative-config", spec_cfg]
        print(f"🔍 N-gram speculative decoding ENABLED  (k={SPEC_NGRAM_K}, ngram_min={SPEC_NGRAM_MIN}, ngram_max={SPEC_NGRAM_MAX})")
        print("   Best for long-context tasks where output tokens appear in the prompt.")
    elif SPEC_MTP:
        if not SPEC_MTP_FORCE:
            raise SystemExit(
                "❌ MTP speculative decoding cannot be enabled with the\n"
                "   GadflyII/GLM-4.7-Flash-NVFP4 checkpoint due to a known\n"
                "   upstream incompatibility:\n\n"
                "     • The checkpoint NVFP4-quantizes the MTP head's eh_proj\n"
                "       (`model.layers.47.eh_proj.{weight_packed,*_scale}`).\n"
                "     • vLLM hard-codes `self.eh_proj = nn.Linear(...)` in\n"
                "       `vllm/model_executor/models/glm4_moe_lite_mtp.py`,\n"
                "       so the FP4 scale tensors have no parameter slot →\n"
                "       KeyError: 'model.layers.47.eh_proj.input_global_scale'.\n\n"
                "   Until either the checkpoint adds `eh_proj` to its\n"
                "   quantization_config.ignore list, or vLLM wraps eh_proj in\n"
                "   a quant-aware Linear, MTP is disabled.\n\n"
                "   To proceed anyway (will crash on load), set:\n"
                "       VLLM_SPEC_MTP_FORCE=1\n"
            )
        spec_cfg = _json.dumps({
            "method": "mtp",
            "num_speculative_tokens": int(SPEC_NUM_TOKENS),
        })
        argv += ["--speculative-config", spec_cfg]
        print(f"🚀 MTP speculative decoding ENABLED  (k={SPEC_NUM_TOKENS}, gpu_mem={GPU_MEM_UTIL})")
        print("⚠️  VLLM_SPEC_MTP_FORCE=1 — known to crash on this NVFP4 checkpoint.")
    else:
        print("ℹ️  Speculative decoding disabled. Use start_ngram.sh for n-gram or start_mtp.sh (blocked).")

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
