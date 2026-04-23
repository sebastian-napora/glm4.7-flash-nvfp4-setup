"""
Token usage tracker for GLM model requests.

Hooks into LiteLLM's CustomLogger system to record per-request token usage
broken down by input category (system / user / tool-schemas / tool-results /
history) and output category (reasoning / response).

Stats are written to logs/token_stats.db (SQLite WAL) and are read by
token_stats_server.py (port 11113).

Input breakdowns are *proportional estimates* — char counts per category are
used to distribute the actual prompt_tokens reported by the model API.
Output reasoning split uses reasoning_content length vs total output length.

Session sharing: the current session ID is persisted to
logs/.token_session so that token_stats_server.py (separate process) can
read the same session and show live stats.
"""

import json
import logging
import os
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger

logger = logging.getLogger("glm_token_tracker")

DB_PATH = Path(__file__).parent / "logs" / "token_stats.db"


SESSION_FILE = DB_PATH.parent / ".token_session"


def _load_session() -> str:
    """Load session ID from file, creating one if missing."""
    try:
        if SESSION_FILE.exists():
            sid = SESSION_FILE.read_text().strip()
            if sid:
                return sid
    except Exception:
        pass
    return _new_session()


def _save_session(sid: str) -> None:
    """Persist session ID to file."""
    try:
        SESSION_FILE.write_text(sid)
    except Exception as exc:
        logger.warning("Failed to write session file: %s", exc)


def _new_session() -> str:
    return str(uuid.uuid4().hex[:16])


def new_session() -> str:
    """Create a fresh session, persist it, and return the ID.

    Called by start.sh at startup so every service restart begins a clean
    session automatically — without needing a manual /api/reset.
    """
    sid = _new_session()
    _save_session(sid)
    logger.info("New token tracking session: %s", sid)
    return sid


# ── SQLite helpers ─────────────────────────────────────────────────────────────

def _open_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), check_same_thread=False, timeout=5.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=3000")
    conn.row_factory = sqlite3.Row
    return conn


def _init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _open_db(path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS token_events (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                ts                 REAL    NOT NULL,
                session_id         TEXT    NOT NULL,
                model              TEXT,
                -- Input categories (proportional estimates of prompt_tokens)
                system_tokens      INTEGER DEFAULT 0,
                user_tokens        INTEGER DEFAULT 0,
                tool_schema_tokens INTEGER DEFAULT 0,
                tool_result_tokens INTEGER DEFAULT 0,
                history_tokens     INTEGER DEFAULT 0,
                -- Output categories (estimated from content lengths)
                reasoning_tokens   INTEGER DEFAULT 0,
                response_tokens    INTEGER DEFAULT 0,
                -- Actuals from API usage object
                prompt_tokens      INTEGER DEFAULT 0,
                completion_tokens  INTEGER DEFAULT 0,
                cached_tokens      INTEGER DEFAULT 0,
                total_tokens       INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ts      ON token_events(ts)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_session ON token_events(session_id)")
        conn.commit()


# ── Text extraction ────────────────────────────────────────────────────────────

def _text_of(content: Any) -> str:
    """Extract plain text from message content (str, list-of-parts, or None)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            p.get("text", "") for p in content
            if isinstance(p, dict) and p.get("type") == "text"
        )
    return ""


# ── Request analysis ───────────────────────────────────────────────────────────

def _analyze_request(messages: list, tools: list) -> dict:
    """
    Return char-count breakdown of a request by input category.
    These counts are later used to proportionally distribute prompt_tokens.
    """
    system_chars = 0
    all_user_chars = 0
    last_user_chars = 0
    tool_result_chars = 0
    history_chars = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _text_of(msg.get("content"))
        # assistant tool_calls also consume tokens
        tc = msg.get("tool_calls")
        if tc:
            try:
                text += json.dumps(tc)
            except (TypeError, ValueError):
                pass

        if role == "system":
            system_chars += len(text)
        elif role == "user":
            all_user_chars += len(text)
        elif role == "tool":
            tool_result_chars += len(text)
        elif role == "assistant":
            history_chars += len(text)

    # Separate the current (last) user turn from history
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user":
            last_user_chars = len(_text_of(msg.get("content")))
            break
    history_chars += max(0, all_user_chars - last_user_chars)

    tool_schema_chars = 0
    if tools:
        try:
            tool_schema_chars = len(json.dumps(tools))
        except (TypeError, ValueError):
            pass

    total = system_chars + last_user_chars + tool_schema_chars + tool_result_chars + history_chars
    return {
        "system_chars":      system_chars,
        "user_chars":        last_user_chars,
        "tool_schema_chars": tool_schema_chars,
        "tool_result_chars": tool_result_chars,
        "history_chars":     history_chars,
        "total_chars":       max(1, total),
    }


def _distribute(prompt_tokens: int, analysis: dict) -> dict:
    """Proportionally distribute prompt_tokens across input categories."""
    total = analysis["total_chars"]

    def prop(chars: int) -> int:
        return max(0, round((chars / total) * prompt_tokens))

    return {
        "system_tokens":      prop(analysis["system_chars"]),
        "user_tokens":        prop(analysis["user_chars"]),
        "tool_schema_tokens": prop(analysis["tool_schema_chars"]),
        "tool_result_tokens": prop(analysis["tool_result_chars"]),
        "history_tokens":     prop(analysis["history_chars"]),
    }


# ── Response parsing ───────────────────────────────────────────────────────────

def _extract_reasoning_tokens(response_obj: Any, completion_tokens: int) -> int:
    """
    Extract or estimate reasoning token count.
    Prefers completion_tokens_details.reasoning_tokens if present,
    otherwise estimates from reasoning_content vs response content lengths.
    """
    if completion_tokens <= 0:
        return 0

    # Try structured details first
    usage = getattr(response_obj, "usage", None)
    if usage:
        details = getattr(usage, "completion_tokens_details", None)
        if details:
            rt = getattr(details, "reasoning_tokens", None)
            if rt:
                return int(rt)

    # Estimate from content lengths
    choices = getattr(response_obj, "choices", []) or []
    if not choices:
        return 0

    choice = choices[0]
    # Handles both streaming (delta) and non-streaming (message)
    msg = getattr(choice, "message", None) or getattr(choice, "delta", None)
    if not msg:
        return 0

    reasoning_text = getattr(msg, "reasoning_content", None) or ""
    response_text = _text_of(getattr(msg, "content", None))

    reasoning_chars = len(reasoning_text)
    total_chars = reasoning_chars + len(response_text)
    if total_chars <= 0:
        return 0

    return max(0, round((reasoning_chars / total_chars) * completion_tokens))


# ── Public query functions (used by token_stats_server) ───────────────────────

def query_summary(db_path: Path = DB_PATH, session_id: Optional[str] = None) -> dict:
    """Return aggregated token stats, optionally scoped to a session."""
    where = "WHERE session_id = ?" if session_id else "WHERE 1=1"
    params: tuple = (session_id,) if session_id else ()
    with _open_db(db_path) as conn:
        row = conn.execute(f"""
            SELECT
                COUNT(*)               AS requests,
                SUM(system_tokens)     AS system_tokens,
                SUM(user_tokens)       AS user_tokens,
                SUM(tool_schema_tokens)AS tool_schema_tokens,
                SUM(tool_result_tokens)AS tool_result_tokens,
                SUM(history_tokens)    AS history_tokens,
                SUM(reasoning_tokens)  AS reasoning_tokens,
                SUM(response_tokens)   AS response_tokens,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens) AS completion_tokens,
                SUM(cached_tokens)     AS cached_tokens,
                SUM(total_tokens)      AS total_tokens
            FROM token_events {where}
        """, params).fetchone()
        return dict(row)


def query_timeline(
    db_path: Path = DB_PATH,
    session_id: Optional[str] = None,
    limit: int = 50,
) -> list[dict]:
    """Return per-request rows, most recent first."""
    where = "WHERE session_id = ?" if session_id else "WHERE 1=1"
    params: tuple = (session_id, limit) if session_id else (limit,)
    with _open_db(db_path) as conn:
        rows = conn.execute(
            f"SELECT * FROM token_events {where} ORDER BY ts DESC LIMIT ?", params
        ).fetchall()
        return [dict(r) for r in rows]


def query_sessions(db_path: Path = DB_PATH) -> list[dict]:
    """Return all sessions with request count and first/last timestamp."""
    with _open_db(db_path) as conn:
        rows = conn.execute("""
            SELECT
                session_id,
                COUNT(*)         AS requests,
                MIN(ts)          AS ts_first,
                MAX(ts)          AS ts_last,
                SUM(prompt_tokens)     AS prompt_tokens,
                SUM(completion_tokens)  AS completion_tokens,
                SUM(reasoning_tokens)  AS reasoning_tokens,
                SUM(cached_tokens)     AS cached_tokens,
                SUM(total_tokens)      AS total_tokens
            FROM token_events
            GROUP BY session_id
            ORDER BY ts_first DESC
        """).fetchall()
        return [dict(r) for r in rows]


# ── CustomLogger ───────────────────────────────────────────────────────────────

class GLMTokenTracker(CustomLogger):
    """Record per-request token usage into SQLite, broken down by category."""

    def __init__(self, db_path: Path = DB_PATH):
        self._db_path = db_path
        self._session_id = _load_session()
        _save_session(self._session_id)  # ensure file is written so stats server sees it
        _init_db(db_path)
        logger.info("GLMTokenTracker session: %s", self._session_id)

    @property
    def session_id(self) -> str:
        return self._session_id

    def new_session(self) -> str:
        """Advance the session watermark — used by the /api/reset endpoint."""
        self._session_id = _new_session()
        _save_session(self._session_id)
        logger.info("Token tracker: new session %s", self._session_id)
        return self._session_id

    # ── Internal ────────────────────────────────────────────────────────────────

    def _record(self, event: dict) -> None:
        """Write event to DB — errors are logged but never propagate."""
        try:
            with _open_db(self._db_path) as conn:
                conn.execute("""
                    INSERT INTO token_events (
                        ts, session_id, model,
                        system_tokens, user_tokens, tool_schema_tokens,
                        tool_result_tokens, history_tokens,
                        reasoning_tokens, response_tokens,
                        prompt_tokens, completion_tokens, cached_tokens, total_tokens
                    ) VALUES (
                        :ts, :session_id, :model,
                        :system_tokens, :user_tokens, :tool_schema_tokens,
                        :tool_result_tokens, :history_tokens,
                        :reasoning_tokens, :response_tokens,
                        :prompt_tokens, :completion_tokens, :cached_tokens, :total_tokens
                    )
                """, event)
                conn.commit()
        except Exception as exc:
            logger.warning("Token DB write failed: %s", exc)

    def _build_event(self, kwargs: dict, response_obj: Any) -> Optional[dict]:
        usage = getattr(response_obj, "usage", None)
        if usage is None:
            return None

        prompt_tokens     = int(getattr(usage, "prompt_tokens",     0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens      = int(getattr(usage, "total_tokens",      0) or
                                (prompt_tokens + completion_tokens))

        cached_tokens = 0
        pt_details = getattr(usage, "prompt_tokens_details", None)
        if pt_details:
            # LiteLLM may expose it as a Pydantic model or a plain dict
            if isinstance(pt_details, dict):
                cached_tokens = int(pt_details.get("cached_tokens", 0) or 0)
            else:
                cached_tokens = int(getattr(pt_details, "cached_tokens", 0) or 0)
        # Fallback: some backends put it directly on the usage object
        if cached_tokens == 0:
            cached_tokens = int(getattr(usage, "cached_tokens", 0) or 0)

        reasoning_tokens = _extract_reasoning_tokens(response_obj, completion_tokens)
        response_tokens  = completion_tokens - reasoning_tokens

        messages = kwargs.get("messages") or []
        tools    = kwargs.get("tools")    or []
        analysis  = _analyze_request(messages, tools)
        breakdown = _distribute(prompt_tokens, analysis)

        event = {
            "ts":         time.time(),
            "session_id": self._session_id,
            "model":      kwargs.get("model", "unknown"),
            **breakdown,
            "reasoning_tokens":   reasoning_tokens,
            "response_tokens":    response_tokens,
            "prompt_tokens":      prompt_tokens,
            "completion_tokens":  completion_tokens,
            "cached_tokens":      cached_tokens,
            "total_tokens":       total_tokens,
        }

        kv_hit_pct = f"{100*cached_tokens/max(1,prompt_tokens):.1f}%" if prompt_tokens else "n/a"
        logger.debug(
            "Tokens — prompt=%d [user=%d sys=%d tools=%d results=%d hist=%d] "
            "kv_cache=%d (%s hit) | completion=%d [reasoning=%d response=%d]",
            prompt_tokens,
            breakdown["user_tokens"], breakdown["system_tokens"],
            breakdown["tool_schema_tokens"], breakdown["tool_result_tokens"],
            breakdown["history_tokens"], cached_tokens, kv_hit_pct,
            completion_tokens, reasoning_tokens, response_tokens,
        )
        return event

    # ── LiteLLM hooks ────────────────────────────────────────────────────────────

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            event = self._build_event(kwargs, response_obj)
            if event:
                self._record(event)
        except Exception as exc:
            logger.warning("Token tracking error: %s", exc, exc_info=True)

    def log_success_event(self, kwargs, response_obj, start_time, end_time):
        try:
            event = self._build_event(kwargs, response_obj)
            if event:
                self._record(event)
        except Exception as exc:
            logger.warning("Token tracking error: %s", exc, exc_info=True)


# ── Singleton + registration ───────────────────────────────────────────────────

_tracker_instance: GLMTokenTracker | None = None


def get_tracker() -> GLMTokenTracker:
    global _tracker_instance
    if _tracker_instance is None:
        _tracker_instance = GLMTokenTracker()
    return _tracker_instance


def register() -> None:
    """Register the tracker with LiteLLM's global callback system."""
    cb = get_tracker()
    if cb not in litellm.callbacks:
        litellm.callbacks.append(cb)
    logger.info("GLMTokenTracker registered to litellm.callbacks")
