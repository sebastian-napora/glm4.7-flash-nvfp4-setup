"""
GLM request sanitization callback for LiteLLM.

This keeps the request flow simple: Copilot -> LiteLLM -> vLLM.
The callback only removes stored reasoning blocks from assistant history
before the next turn is forwarded to the model.
"""

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


def _strip_thinking_tokens(text: str) -> str:
    """Remove thinking token blocks from a message string."""
    return _THINKING_RE.sub('', text).strip()


class GLMHistorySanitizer(CustomLogger):
    """Remove stored reasoning blocks from assistant history before inference."""

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

        sanitized_messages: list[Any] = []
        stripped_any = False
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "assistant":
                content = msg.get("content")
                if isinstance(content, str):
                    stripped = _strip_thinking_tokens(content)
                    if stripped != content:
                        msg = dict(msg)
                        msg["content"] = stripped
                        stripped_any = True
            sanitized_messages.append(msg)
        if stripped_any:
            logger.debug("Stripped thinking tokens from assistant messages in history")
            data = dict(data)
            data["messages"] = sanitized_messages
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
