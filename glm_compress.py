"""
GLM-4.7-Flash-NVFP4 Auto-Compression Callback for LiteLLM.

Implements CustomLogger.async_pre_call_hook — intercepts requests before
they are forwarded to the LLM and auto-compresses long conversations.

Triggered when estimated input tokens exceed COMPRESS_THRESHOLD_TOKENS (default: 50 000).
Compresses older messages via /compress endpoint on the same vLLM backend,
then replaces them with a compressed summary.

Architecture:
    Copilot → LiteLLM (11111) → vLLM (11112)
                                         ↑
                                  /compress @ 11112
"""

import asyncio
import json
import logging
import os
import urllib.request
from typing import Any, Optional, Union

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("glm_compress")

COMPRESS_THRESHOLD_TOKENS: int = int(
    os.environ.get("LITE_LLM_COMPRESS_THRESHOLD_TOKENS", "50000")
)
COMPRESS_TARGET_TOKENS: int = int(
    os.environ.get("LITE_LLM_COMPRESS_TARGET_TOKENS", "16384")
)
COMPRESS_ENDPOINT: str = os.environ.get(
    "LITE_LLM_COMPRESS_ENDPOINT", "http://localhost:11112/compress"
)
COMPRESSED_MODELS: set[str] = set(
    m.strip()
    for m in os.environ.get("LITE_LLM_COMPRESS_MODELS", "glm-4.7-flash-nvfp4,glm47flash").split(",")
    if m.strip()
)
PRESERVE_RECENT_MESSAGES: int = int(
    os.environ.get("LITE_LLM_COMPRESS_PRESERVE_RECENT", "5")
)

_CACHED_TOKENIZER: Any = None


def _token_encode(text: str) -> list[int]:
    """Thread-safe token encoding via cached HuggingFace tokenizer."""
    global _CACHED_TOKENIZER
    if _CACHED_TOKENIZER is None:
        try:
            from transformers import AutoTokenizer
            _CACHED_TOKENIZER = AutoTokenizer.from_pretrained(
                "GadflyII/GLM-4.7-Flash-NVFP4",
                use_fast=True,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Failed to load tokenizer for token counting: %s", exc)
            return []
    try:
        return _CACHED_TOKENIZER.encode(text, add_special_tokens=False)
    except Exception as exc:
        logger.warning("Token encoding failed: %s", exc)
        return []


def _count_message_tokens(msg: dict) -> int:
    """Estimate tokens for a single message dict."""
    content = msg.get("content", "")
    if isinstance(content, list):
        text_parts = [
            p.get("text", "") or p.get("inline_data", {}).get("data", "")
            for p in content if isinstance(p, dict)
        ]
        content = " ".join(text_parts)
    elif not isinstance(content, str):
        content = str(content)
    role = msg.get("role", "")
    return len(_token_encode(f"{role}: {content}"))


def count_messages_tokens(messages: list[dict]) -> int:
    """Count total tokens in a messages list."""
    return sum(_count_message_tokens(msg) for msg in messages)


def _sync_compress(messages: list[dict], target_tokens: int) -> dict | None:
    """Synchronous /compress call — used as fallback in thread pool."""
    payload = json.dumps({
        "messages": messages,
        "target_tokens": target_tokens,
    }).encode("utf-8")

    req = urllib.request.Request(
        COMPRESS_ENDPOINT,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            return result.get("compressed")
    except Exception as exc:
        logger.error("Compression call failed: %s", exc)
        return None


async def _async_compress(messages: list[dict], target_tokens: int) -> dict | None:
    """Async /compress call using aiohttp."""
    payload = json.dumps({
        "messages": messages,
        "target_tokens": target_tokens,
    })

    try:
        import aiohttp
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=120)
        ) as session:
            async with session.post(
                COMPRESS_ENDPOINT,
                data=payload,
                headers={"Content-Type": "application/json"},
            ) as resp:
                result = await resp.json()
                return result.get("compressed")
    except ImportError:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, _sync_compress, messages, target_tokens)
    except Exception as exc:
        logger.error("Async compression call failed: %s", exc)
        return None


def _build_system_prompt(summary: dict, preserved: list[dict]) -> str:
    """Build a system message from the compression result."""
    summary_text = summary.get("summary", "")
    preserved_msgs = preserved[-PRESERVE_RECENT_MESSAGES:] if preserved else []

    lines = [
        "[CONTEXT COMPRESSED — START]",
        f"Summary: {summary_text}",
        "",
        "Preserved messages (recent context):",
    ]
    for msg in preserved_msgs:
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        lines.append(f"\n--- {role.upper()} ---")
        lines.append(content)

    lines.append("\n[CONTEXT COMPRESSED — END]")
    return "\n".join(lines)


class GLMCompressCallback(CustomLogger):
    """
    LiteLLM CustomLogger that auto-compresses long conversations before inference.

    Conditions to trigger compression:
      1. Model name matches LITE_LLM_COMPRESS_MODELS (default: glm-4.7-flash-nvfp4,glm47flash)
      2. Estimated input tokens > LITE_LLM_COMPRESS_THRESHOLD_TOKENS (default: 50 000)
      3. At least 10 messages to compress (don't compress short chats)

    Returns a modified ``data`` dict with compressed messages if compression was
    triggered, otherwise returns ``None`` to pass through unchanged.
    """

    def __init__(
        self,
        threshold_tokens: int = COMPRESS_THRESHOLD_TOKENS,
        target_tokens: int = COMPRESS_TARGET_TOKENS,
        models: set[str] | None = None,
        preserve_recent: int = PRESERVE_RECENT_MESSAGES,
    ):
        super().__init__()
        self.threshold_tokens = threshold_tokens
        self.target_tokens = target_tokens
        self.models = models or COMPRESSED_MODELS
        self.preserve_recent = preserve_recent
        logger.info(
            "GLMCompressCallback init: threshold=%d tokens, target=%d tokens, models=%s",
            self.threshold_tokens, self.target_tokens, self.models,
        )

    def _should_compress(self, model: str | None, messages: list[dict]) -> bool:
        if not model or not messages:
            return False
        model_lower = model.lower()
        if not any(
            m.lower() in model_lower or model_lower in m.lower()
            for m in self.models
        ):
            return False
        if len(messages) < 10:
            return False
        return count_messages_tokens(messages) > self.threshold_tokens

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Called by the LiteLLM proxy before each request is forwarded to the LLM.

        Returns:
          None        → pass through data unchanged
          dict        → replace data with returned dict
          str/Exception → rejection (not used here)
        """
        if call_type not in ("acompletion", "completion"):
            return None

        messages = data.get("messages", [])
        if isinstance(messages, str):
            return None

        model = data.get("model")
        if not self._should_compress(model, messages):
            return None

        total = count_messages_tokens(messages)
        logger.info(
            "Compressing %d messages (~%d tokens, threshold=%d) for model=%s",
            len(messages), total, self.threshold_tokens, model,
        )

        compressed = await _async_compress(messages, self.target_tokens)

        if compressed is None:
            logger.warning("Compression failed — forwarding original messages")
            return None

        summary = compressed.get("summary", "")
        preserved = compressed.get("preserved_messages", [])
        budget = compressed.get("token_budget_used")

        logger.info(
            "Compression done: summary=%d chars, preserved=%d msgs, budget=%.2f",
            len(summary), len(preserved), budget if budget else -1,
        )

        system_content = _build_system_prompt(
            {"summary": summary}, preserved
        )

        compressed_messages = [{"role": "system", "content": system_content}]
        compressed_messages.extend(messages[-self.preserve_recent:])

        new_total = count_messages_tokens(compressed_messages)
        logger.info(
            "Compressed: %d msgs (~%d tokens) → %d msgs (~%d tokens)",
            len(messages), total, len(compressed_messages), new_total,
        )

        # Return modified data dict — LiteLLM will use this instead
        data = dict(data)
        data["messages"] = compressed_messages
        return data


# ── Singleton for LiteLLM callback registration ────────────────────────────────
_callback_instance: GLMCompressCallback | None = None


def get_callback() -> GLMCompressCallback:
    global _callback_instance
    if _callback_instance is None:
        _callback_instance = GLMCompressCallback()
    return _callback_instance


def register():
    """Register the callback with LiteLLM's global callback system."""
    cb = get_callback()
    litellm.callbacks.append(cb)
    litellm.success_callback.append(cb)
    logger.info(
        "GLMCompressCallback registered to litellm.callbacks "
        "(threshold=%d tokens, models=%s)",
        cb.threshold_tokens, cb.models,
    )