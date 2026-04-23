"""
GLM request sanitization callback for LiteLLM.

This keeps the request flow simple: Copilot -> LiteLLM -> vLLM.
The callback:
  1. Removes stored reasoning blocks from assistant history to prevent
     the model re-entering thinking mode on every turn.
  2. Compresses old tool result messages to reduce context accumulation
     from large tool responses across turns.
"""

import json
import logging
import re
from typing import Any, Optional, Union

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("glm_compress")

# Matches both GLM <|begin_of_thought|>...<|end_of_thought|> and <think>...</think> blocks.
# These accumulate in conversation history and cause the model to re-enter thinking mode
# on every subsequent turn, progressively inflating context size.
_THINKING_RE = re.compile(
    r'(<\|begin_of_thought\|>.*?<\|end_of_thought\|>|<think>.*?</think>)',
    re.DOTALL | re.IGNORECASE,
)

# Maximum characters to keep when truncating a plain-text tool result from history.
TOOL_RESULT_MAX_CHARS = 400

# Detects previously compressed results — prevents double-compression on repeated calls.
_COMPRESSED_MARKER_RE = re.compile(r'\[… \d+ chars omitted\]')


def _strip_thinking_tokens(text: str) -> str:
    """Remove thinking token blocks from a message string."""
    return _THINKING_RE.sub('', text).strip()


def _compress_tool_result_text(text: str) -> str:
    """
    Compress a single tool result string for older history entries.

    - Already-compressed results are returned unchanged (idempotent).
    - JSON-structured results are replaced with a structural stub to avoid
      leaving the model with syntactically broken JSON.
    - Plain text is truncated with a clear omission marker.
    """
    if len(text) <= TOOL_RESULT_MAX_CHARS:
        return text
    if _COMPRESSED_MARKER_RE.search(text):
        return text

    original_len = len(text)
    stripped = text.strip()
    if stripped.startswith(('{', '[')):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                return (
                    f"[tool result omitted: JSON object, keys={list(obj.keys())}, "
                    f"{original_len} chars]"
                )
            if isinstance(obj, list):
                return (
                    f"[tool result omitted: JSON array, {len(obj)} items, "
                    f"{original_len} chars]"
                )
        except (json.JSONDecodeError, ValueError):
            pass

    return text[:TOOL_RESULT_MAX_CHARS] + f"\n[… {original_len - TOOL_RESULT_MAX_CHARS} chars omitted]"


def _compress_content(content: Any) -> tuple[Any, bool]:
    """
    Compress a tool message's content field, preserving its schema shape.

    Handles both plain string content and list-of-parts content.
    Multi-part content is only compressed when every part is a text part
    (never touch image/binary parts).

    Returns (new_content, changed).
    """
    if isinstance(content, str):
        compressed = _compress_tool_result_text(content)
        return compressed, compressed != content

    if isinstance(content, list):
        if not all(isinstance(p, dict) and p.get("type") == "text" for p in content):
            return content, False
        new_parts: list[dict] = []
        changed = False
        for part in content:
            text = part.get("text", "")
            compressed = _compress_tool_result_text(text)
            if compressed != text:
                new_parts.append({**part, "text": compressed})
                changed = True
            else:
                new_parts.append(part)
        return new_parts, changed

    return content, False


class GLMHistorySanitizer(CustomLogger):
    """Remove stored reasoning blocks and compress old tool results from history."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict,
        call_type: str,
    ) -> Optional[Union[Exception, str, dict]]:
        """
        Called by the LiteLLM proxy before each request is forwarded to the LLM.

        Tool results that appear before the current user turn (i.e. from prior
        conversation turns) are compressed.  All tool results after the last user
        message belong to the current turn and are kept verbatim.

        Returns:
          None  → pass through data unchanged
          dict  → replace data with returned dict
        """
        if call_type not in ("acompletion", "completion"):
            return None

        messages = data.get("messages", [])
        if isinstance(messages, str):
            return None

        # Boundary: all tool results after the last user message are in the
        # current turn and must not be touched.
        last_user_idx = -1
        for i, msg in enumerate(messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                last_user_idx = i

        sanitized: list[Any] = []
        changed = False

        for i, msg in enumerate(messages):
            if not isinstance(msg, dict):
                sanitized.append(msg)
                continue

            role = msg.get("role")

            if role == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    stripped = _strip_thinking_tokens(content)
                    if stripped != content:
                        msg = {**msg, "content": stripped}
                        changed = True

            elif role == "tool" and i < last_user_idx:
                content = msg.get("content")
                new_content, did_change = _compress_content(content)
                if did_change:
                    logger.debug(
                        "Compressed tool result [call_id=%s]",
                        msg.get("tool_call_id", "?"),
                    )
                    msg = {**msg, "content": new_content}
                    changed = True

            sanitized.append(msg)

        if changed:
            data = {**data, "messages": sanitized}
            return data
        return None


# ── Singleton for LiteLLM callback registration ────────────────────────────────
_callback_instance: GLMHistorySanitizer | None = None


def get_callback() -> GLMHistorySanitizer:
    global _callback_instance
    if _callback_instance is None:
        _callback_instance = GLMHistorySanitizer()
    return _callback_instance


def register():
    """Register the callback with LiteLLM's global callback system."""
    cb = get_callback()
    if cb not in litellm.callbacks:
        litellm.callbacks.append(cb)
    logger.info("GLMHistorySanitizer registered to litellm.callbacks")
