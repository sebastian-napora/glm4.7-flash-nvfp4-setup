"""
GLM request sanitization callback for LiteLLM.

This keeps the request flow simple: Copilot -> LiteLLM -> vLLM.
The callback:
  1. Removes stored reasoning blocks from assistant history to prevent
     the model re-entering thinking mode on every turn.
  2. Compresses old tool result messages to reduce context accumulation
     from large tool responses across turns.
  3. Trims oldest conversation exchanges when the estimated token count
     would exceed the model's context window, preventing hard vLLM errors.
"""

import hashlib
import json
import logging
import re
import time
from pathlib import Path
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

# ── Tool schema compression ────────────────────────────────────────────────────

# Max chars for a tool's top-level description before truncation.
TOOL_DESC_MAX_CHARS = 280

# Max chars for a per-parameter description before truncation.
TOOL_PARAM_DESC_MAX_CHARS = 120

# Fields inside a parameter's JSON schema that are purely documentary and safe
# to drop without affecting the model's ability to call the tool correctly.
_TOOL_SCHEMA_DROP_FIELDS = frozenset({"examples", "x-ms-docs", "deprecated", "additionalProperties"})


def _compress_param_schema(schema: Any) -> Any:
    """Strip verbose/documentary fields from a single parameter's JSON schema."""
    if not isinstance(schema, dict):
        return schema
    out: dict = {}
    for k, v in schema.items():
        if k in _TOOL_SCHEMA_DROP_FIELDS:
            continue
        if k == "description" and isinstance(v, str) and len(v) > TOOL_PARAM_DESC_MAX_CHARS:
            v = v[:TOOL_PARAM_DESC_MAX_CHARS] + "…"
        elif k == "properties" and isinstance(v, dict):
            v = {pk: _compress_param_schema(pv) for pk, pv in v.items()}
        elif k == "items" and isinstance(v, dict):
            v = _compress_param_schema(v)
        out[k] = v
    return out


def _compress_tool_schemas(tools: list) -> tuple[list, bool]:
    """
    Compress tool schema definitions to reduce context token usage.

    Safe transforms:
    - Truncate top-level function descriptions to TOOL_DESC_MAX_CHARS.
    - Truncate per-parameter descriptions to TOOL_PARAM_DESC_MAX_CHARS.
    - Remove purely documentary fields (examples, deprecated, x-ms-docs).

    The model can still call every tool correctly after compression; it only
    loses verbose explanation text it doesn't need at inference time.

    Returns (compressed_tools, was_changed).
    """
    if not tools:
        return tools, False

    compressed = []
    changed = False

    for tool in tools:
        if not isinstance(tool, dict):
            compressed.append(tool)
            continue

        fn = tool.get("function")
        if not isinstance(fn, dict):
            compressed.append(tool)
            continue

        new_fn = dict(fn)
        did_change = False

        # Truncate top-level description.
        desc = fn.get("description", "")
        if isinstance(desc, str) and len(desc) > TOOL_DESC_MAX_CHARS:
            new_fn["description"] = desc[:TOOL_DESC_MAX_CHARS] + "…"
            did_change = True

        # Compress parameter schemas.
        params = fn.get("parameters")
        if isinstance(params, dict):
            new_params = _compress_param_schema(params)
            if new_params != params:
                new_fn["parameters"] = new_params
                did_change = True

        if did_change:
            compressed.append({**tool, "function": new_fn})
            changed = True
        else:
            compressed.append(tool)

    return compressed, changed


# ── Context window management ──────────────────────────────────────────────────

# Hard context limit from glm_server.py --max-model-len.
CONTEXT_LIMIT_TOKENS = 202_752

# Reserved headroom to absorb tokenizer estimation error and protocol overhead.
CONTEXT_TRIM_BUFFER = 3_000

# Conservative estimate: 1 BPE token ≈ 2.5 chars (errs on the side of over-counting
# to avoid under-trimming — safer for a hard-limit safeguard).
_CHARS_PER_TOKEN = 2.5

# Fallback output reservation when the request omits max_tokens.
_DEFAULT_MAX_OUTPUT_TOKENS = 4_096

# Path where session state is saved when a context overflow cannot be recovered.
_OVERFLOW_SESSION_PATH = Path(__file__).parent / "logs" / "last_session.json"

# ── Preserved-thinking configuration ──────────────────────────────────────────

# When True, reasoning_content from recent assistant turns is re-injected into
# the conversation history before each request.  This lets the model see its own
# prior reasoning chain, improving continuity and increasing KV-cache prefix hits.
# Set to False to fall back to the original strip-only behaviour.
PRESERVE_THINKING = True

# How many of the most-recent assistant turns to carry reasoning for.
# Older turns are left as clean content to bound extra context growth.
_PRESERVE_TURNS = 3


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


# ── Token estimation ───────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """Conservative BPE token estimate: 1 token ≈ 2.5 chars."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


def _message_token_estimate(msg: dict) -> int:
    """Estimate tokens for one message, including content and tool_calls."""
    content = msg.get("content") or ""
    if isinstance(content, str):
        base = _estimate_tokens(content)
    elif isinstance(content, list):
        base = sum(
            _estimate_tokens(p.get("text", "") if isinstance(p, dict) else str(p))
            for p in content
        )
    else:
        base = 4
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        try:
            base += _estimate_tokens(json.dumps(tool_calls))
        except Exception:
            base += 50
    return base + 8  # per-message overhead (role, separators, etc.)


def _messages_token_estimate(messages: list) -> int:
    return sum(_message_token_estimate(m) for m in messages if isinstance(m, dict))


def _tools_token_estimate(tools: Any) -> int:
    """Estimate tokens consumed by the tools array (sent separately from messages)."""
    if not tools:
        return 0
    try:
        return _estimate_tokens(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0


# ── Context-window trimming ────────────────────────────────────────────────────

def _trim_to_context(messages: list, token_budget: int) -> tuple[list, bool]:
    """
    Drop oldest conversation exchanges from *messages* until the estimated
    token count fits within *token_budget*.

    Preservation rules (highest priority first):
      1. Leading system messages (contain tool schemas — never dropped).
      2. The last user message and everything after it (current turn).
      3. Most-recent history exchanges (dropped last).

    An "exchange" is the slice from one user message up to (but not
    including) the next user message.  This keeps tool_calls + tool
    results + follow-up assistant turns atomically grouped.

    Orphaned non-user messages at the start of the middle zone (rare but
    possible) are dropped as a leading fragment before any exchange.

    Returns (new_messages, was_changed).
    """
    total = _messages_token_estimate(messages)
    if total <= token_budget:
        return messages, False

    # ── Zone 1: leading system messages (always keep) ────────────────────────
    head: list = []
    idx = 0
    while (
        idx < len(messages)
        and isinstance(messages[idx], dict)
        and messages[idx].get("role") == "system"
    ):
        head.append(messages[idx])
        idx += 1

    # ── Zone 3: current turn (last user message onward, always keep) ─────────
    last_user_idx = -1
    for j in range(len(messages) - 1, idx - 1, -1):
        if isinstance(messages[j], dict) and messages[j].get("role") == "user":
            last_user_idx = j
            break

    if last_user_idx < idx:
        # No history user message after system head — nothing to trim.
        return messages, False

    current_turn = list(messages[last_user_idx:])
    middle = list(messages[idx:last_user_idx])

    head_tokens = _messages_token_estimate(head)
    current_tokens = _messages_token_estimate(current_turn)
    middle_tokens = total - head_tokens - current_tokens

    was_changed = False

    while middle and (head_tokens + middle_tokens + current_tokens) > token_budget:
        # Locate the first user message in middle to establish exchange boundary.
        first_user = next(
            (k for k, m in enumerate(middle) if isinstance(m, dict) and m.get("role") == "user"),
            -1,
        )

        if first_user == -1:
            # No more user messages — drop remaining middle entirely.
            middle_tokens = 0
            middle = []
            was_changed = True
            break

        if first_user > 0:
            # Orphaned non-user prefix — drop it first as a standalone fragment.
            chunk = middle[:first_user]
            middle_tokens -= _messages_token_estimate(chunk)
            middle = middle[first_user:]
            was_changed = True
            continue

        # Middle starts with a user message.  Find the end of this exchange
        # (the next user message boundary).
        drop_until = len(middle)
        for k in range(1, len(middle)):
            if isinstance(middle[k], dict) and middle[k].get("role") == "user":
                drop_until = k
                break

        chunk = middle[:drop_until]
        dropped = _messages_token_estimate(chunk)
        middle = middle[drop_until:]
        middle_tokens -= dropped
        was_changed = True

        logger.warning(
            "Context trim: dropped oldest exchange (%d msgs, ~%d tokens). "
            "Remaining estimate: ~%d tokens (budget=%d)",
            len(chunk), dropped,
            head_tokens + middle_tokens + current_tokens,
            token_budget,
        )

    if (head_tokens + middle_tokens + current_tokens) > token_budget:
        logger.warning(
            "Context trim: history fully cleared but system+current_turn "
            "(~%d tokens) still exceeds budget (~%d tokens). "
            "Request may be rejected by the backend.",
            head_tokens + current_tokens,
            token_budget,
        )

    return head + middle + current_turn, was_changed


# ── Preserved-thinking reasoning store ────────────────────────────────────────

class _ReasoningStore:
    """
    Bounded FIFO cache: md5(content) → reasoning_content.

    Captures the model's reasoning_content from each completed response so it
    can be injected back into the conversation history on the next request.
    This implements "preserved thinking": the model sees its own previous
    reasoning, which improves reasoning continuity and increases KV-cache
    prefix hit rates (the reasoning tokens are already cached from the prior
    turn and do not require recomputation).

    Memory cost: bounded to max_size entries (default 30).  Each entry holds
    one reasoning string — typically a few KB at most.  No extra GPU RAM is
    used; the KV cache is pre-allocated at startup.
    """

    def __init__(self, max_size: int = 30) -> None:
        self._max = max_size
        self._data: dict[str, str] = {}
        self._keys: list[str] = []   # FIFO eviction order

    def _key(self, content: str) -> str:
        return hashlib.md5(content.encode("utf-8", errors="replace")).hexdigest()

    def put(self, content: str, reasoning: str) -> None:
        if not content or not reasoning:
            return
        k = self._key(content)
        if k not in self._data:
            if len(self._keys) >= self._max:
                evicted = self._keys.pop(0)
                self._data.pop(evicted, None)
            self._keys.append(k)
        self._data[k] = reasoning

    def get(self, content: str) -> Optional[str]:
        return self._data.get(self._key(content))


_reasoning_store = _ReasoningStore()


# ── Context overflow recovery ──────────────────────────────────────────────────

def _extract_msg_text(msg: dict) -> str:
    """Plain text from a message dict (handles str and list-of-parts content)."""
    content = msg.get("content") or ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


def _save_overflow_session(messages: list, tokens: int) -> None:
    """Persist conversation state to JSON so the user can reference it later."""
    try:
        _OVERFLOW_SESSION_PATH.parent.mkdir(parents=True, exist_ok=True)
        user_turns = [
            _extract_msg_text(m)[:400]
            for m in messages if isinstance(m, dict) and m.get("role") == "user"
        ]
        asst_turns = [
            _extract_msg_text(m)[:400]
            for m in messages if isinstance(m, dict) and m.get("role") == "assistant"
        ]
        _OVERFLOW_SESSION_PATH.write_text(json.dumps({
            "timestamp":         time.strftime("%Y-%m-%dT%H:%M:%S"),
            "message_count":     len(messages),
            "estimated_tokens":  tokens,
            "user_turns":        user_turns,
            "assistant_turns_last3": asst_turns[-3:],
        }, ensure_ascii=False, indent=2))
        logger.info("Overflow session saved → %s", _OVERFLOW_SESSION_PATH)
    except Exception as exc:
        logger.warning("Failed to save overflow session: %s", exc)


def _make_overflow_response(messages: list, tokens: int) -> str:
    """
    Build a friendly Markdown assistant message for Copilot when the context
    window is exhausted and cannot be recovered by trimming.

    The message is returned via LiteLLM's mock_response mechanism so Copilot
    sees a normal 200 assistant turn instead of a cryptic 500 error.
    """
    user_msgs = [m for m in messages if isinstance(m, dict) and m.get("role") == "user"]
    first = _extract_msg_text(user_msgs[0])[:280].strip() if user_msgs else ""
    recent = [
        _extract_msg_text(m)[:130].strip()
        for m in user_msgs[-4:-1]   # last few before the current one
        if isinstance(m, dict)
    ]
    recent = [r for r in recent if r]

    lines = [
        f"⚠️ **Context window limit reached** (~{tokens:,} / {CONTEXT_LIMIT_TOKENS:,} tokens)\n\n",
        "All available history has been trimmed but the session was still too large. "
        f"A snapshot has been saved to `logs/last_session.json`.\n",
    ]
    if first:
        clip = "…" if len(_extract_msg_text(user_msgs[0]).strip()) > 280 else ""
        lines.append(f"\n**Session started with:**\n> {first}{clip}\n")
    if recent:
        lines.append("\n**Recent topics discussed:**")
        for r in recent:
            clip = "…" if len(r) == 130 else ""
            lines.append(f"\n> {r}{clip}")
        lines.append("\n")
    lines.append(
        "\n---\n"
        "**Fresh session started automatically.** "
        "Just continue — describe what you need and I'll pick up from here."
    )
    return "".join(lines)


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

        # Preserved thinking: collect indices of the most-recent assistant turns
        # before the current user message.  Only those turns get reasoning_content
        # re-injected; older turns are left clean to bound context growth.
        recent_asst_set: set[int] = set()
        if PRESERVE_THINKING:
            asst_before = [
                i for i, m in enumerate(messages)
                if isinstance(m, dict)
                and m.get("role") == "assistant"
                and i < last_user_idx
            ]
            recent_asst_set = set(asst_before[-_PRESERVE_TURNS:])

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
                    # Inject preserved reasoning for the most-recent turns.
                    if PRESERVE_THINKING and i in recent_asst_set and "reasoning_content" not in msg:
                        stored = _reasoning_store.get(content)
                        if stored:
                            msg = {**msg, "reasoning_content": stored}
                            changed = True
                            logger.debug(
                                "Preserved thinking: injected reasoning turn %d "
                                "(%d reasoning chars → %d content chars)",
                                i, len(stored), len(content),
                            )
                    # Strip any thinking tokens still embedded in content (safety net).
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

        # ── Tool schema compression ───────────────────────────────────────────
        # Compress tool descriptions before token counting so the budget
        # calculation reflects the actual (reduced) tool schema footprint.
        tools = data.get("tools") or []
        if tools:
            compressed_tools, tools_changed = _compress_tool_schemas(tools)
            if tools_changed:
                before_tool_tokens = _tools_token_estimate(tools)
                after_tool_tokens = _tools_token_estimate(compressed_tools)
                logger.debug(
                    "Tool schema compression: %d tools, ~%d→~%d tokens saved",
                    len(tools), before_tool_tokens, after_tool_tokens,
                )
                data = {**data, "tools": compressed_tools}
                tools = compressed_tools

        # ── Context window trim ───────────────────────────────────────────────
        # Account for tool schemas (sent outside messages[]) and the requested
        # output budget so that the total request stays within the model limit.
        tools_tokens = _tools_token_estimate(tools)
        # Use the actual requested max_tokens; cap at half the context to keep
        # the budget positive even for very large output requests.
        max_output = int(data.get("max_tokens") or _DEFAULT_MAX_OUTPUT_TOKENS)
        output_reserve = min(max_output, CONTEXT_LIMIT_TOKENS // 2)
        token_budget = CONTEXT_LIMIT_TOKENS - CONTEXT_TRIM_BUFFER - tools_tokens - output_reserve

        before_tokens = _messages_token_estimate(sanitized)
        if before_tokens > token_budget:
            trimmed, did_trim = _trim_to_context(sanitized, token_budget)
            if did_trim:
                after_tokens = _messages_token_estimate(trimmed)
                logger.warning(
                    "Context window trim: %d→%d messages, ~%d→~%d estimated tokens "
                    "(tool_schemas=~%d, output_reserve=%d, budget=%d)",
                    len(sanitized), len(trimmed),
                    before_tokens, after_tokens,
                    tools_tokens, output_reserve, token_budget,
                )
                sanitized = trimmed
                changed = True

                # If still over budget after exhausting all trimmable history,
                # return a friendly assistant message instead of forwarding to
                # vLLM (which would crash with a cryptic 500 error).
                if after_tokens > token_budget:
                    _save_overflow_session(data.get("messages", []), before_tokens)
                    overflow_msg = _make_overflow_response(
                        data.get("messages", []), before_tokens
                    )
                    logger.warning(
                        "Context overflow: history exhausted, returning graceful "
                        "recovery message to client (~%d tokens, budget=%d)",
                        after_tokens, token_budget,
                    )
                    # mock_response is a LiteLLM parameter: when set, acompletion
                    # short-circuits and returns this text as the assistant reply
                    # (works for both streaming and non-streaming requests).
                    data = {**data, "messages": sanitized, "mock_response": overflow_msg}
                    return data

        if changed:
            data = {**data, "messages": sanitized}
            return data
        return None

    async def async_log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """
        Capture reasoning_content from completed responses for preserved thinking.

        After each successful response from vLLM, stores the model's reasoning
        in _reasoning_store keyed by the assistant's content.  On the next
        request, async_pre_call_hook re-injects this reasoning into the
        historical assistant message so the model sees its prior chain of thought.
        """
        if not PRESERVE_THINKING:
            return
        try:
            choices = getattr(response_obj, "choices", None) or []
            if not choices:
                return
            choice = choices[0]
            msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if not msg:
                return
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content_raw = getattr(msg, "content", None) or ""
            content = (
                content_raw if isinstance(content_raw, str)
                else _extract_msg_text({"content": content_raw})
            )
            if reasoning and content:
                _reasoning_store.put(content, reasoning)
                logger.debug(
                    "Preserved thinking: stored %d-char reasoning for %d-char response",
                    len(reasoning), len(content),
                )
        except Exception as exc:
            logger.debug("Preserved thinking capture error: %s", exc)

    def log_success_event(
        self, kwargs: dict, response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        """Sync version of async_log_success_event for non-async contexts."""
        if not PRESERVE_THINKING:
            return
        try:
            choices = getattr(response_obj, "choices", None) or []
            if not choices:
                return
            choice = choices[0]
            msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
            if not msg:
                return
            reasoning = getattr(msg, "reasoning_content", None) or ""
            content_raw = getattr(msg, "content", None) or ""
            content = (
                content_raw if isinstance(content_raw, str)
                else _extract_msg_text({"content": content_raw})
            )
            if reasoning and content:
                _reasoning_store.put(content, reasoning)
        except Exception as exc:
            logger.debug("Preserved thinking capture error (sync): %s", exc)


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
