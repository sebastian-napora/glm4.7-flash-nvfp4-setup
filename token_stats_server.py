#!/usr/bin/env python3
"""
Token statistics dashboard for GLM-4.7-Flash-NVFP4.

Runs on port 11113.  Reads from the same logs/token_stats.db that the
LiteLLM proxy tracker writes to.

Endpoints:
  GET  /              — HTML dashboard (auto-refreshes every 10 s)
  GET  /api/stats     — JSON summary for current session
  GET  /api/timeline  — JSON per-request history
  POST /api/reset     — Start a new session (old data is preserved, just hidden)

Usage:
  python token_stats_server.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Allow imports from parent directory (glm_token_tracker lives here)
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

import glm_token_tracker as tracker
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
import uvicorn

STATS_PORT = int(os.environ.get("GLM_STATS_PORT", "11113"))
STATS_HOST = os.environ.get("GLM_STATS_HOST", "0.0.0.0")
DB_PATH = tracker.DB_PATH

# Read shared session from the same file the token tracker uses.
# This ensures the server and tracker always see the same session ID.
SESSION_FILE = tracker.SESSION_FILE


def _read_session() -> str:
    """Read current session ID from the shared session file.
    If the file doesn't exist yet, bootstrap the tracker singleton first."""
    try:
        if SESSION_FILE.exists():
            sid = SESSION_FILE.read_text().strip()
            if sid:
                return sid
    except Exception:
        pass
    # Cold start: load tracker singleton to create the session file
    tracker.get_tracker()
    try:
        if SESSION_FILE.exists():
            sid = SESSION_FILE.read_text().strip()
            if sid:
                return sid
    except Exception:
        pass
    return "unknown"

app = FastAPI(title="GLM Token Stats", docs_url=None, redoc_url=None)


# ── API endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/stats")
def api_stats(all_sessions: bool = False):
    sid = None if all_sessions else _read_session()
    return tracker.query_summary(DB_PATH, session_id=sid)


@app.get("/api/timeline")
def api_timeline(
    limit: int = Query(50, ge=1, le=500),
    all_sessions: bool = False,
):
    sid = None if all_sessions else _read_session()
    return tracker.query_timeline(DB_PATH, session_id=sid, limit=limit)


@app.post("/api/reset")
def api_reset():
    new_sid = tracker.get_tracker().new_session()
    return {"ok": True, "new_session_id": new_sid}


@app.get("/api/session")
def api_session():
    """Return the current shared session ID."""
    return {"session_id": _read_session()}


@app.get("/api/sessions")
def api_sessions():
    """Return all sessions with summary stats."""
    sessions = tracker.query_sessions(DB_PATH)
    current = _read_session()
    return {"sessions": sessions, "current": current}


# ── Dashboard ──────────────────────────────────────────────────────────────────

def _fmt(n: Optional[int]) -> str:
    return f"{int(n):,}" if n else "0"


def _pct(part: Optional[int], total: Optional[int]) -> str:
    if not part or not total:
        return "0%"
    return f"{100 * int(part) / max(1, int(total)):.1f}%"


def _bar(part: Optional[int], total: Optional[int], colour: str = "#4a9eff") -> str:
    pct = 100 * int(part or 0) / max(1, int(total or 1))
    pct = min(pct, 100)
    return (
        f"<div class='bar'>"
        f"<div class='bar-fill' style='width:{pct:.1f}%;background:{colour}'></div>"
        f"</div>"
    )


@app.get("/", response_class=HTMLResponse)
def dashboard(session: Optional[str] = Query(None)):
    # session param from URL overrides the live tracker session
    active_sid = session if session else _read_session()
    sessions = tracker.query_sessions(DB_PATH)
    current_live = _read_session()

    # Use the session's own stats if it exists, else fall back to live session
    if active_sid and any(s["session_id"] == active_sid for s in sessions):
        stats = tracker.query_summary(DB_PATH, session_id=active_sid)
        timeline = tracker.query_timeline(DB_PATH, session_id=active_sid, limit=50)
    elif active_sid == current_live:
        stats = tracker.query_summary(DB_PATH, session_id=active_sid)
        timeline = tracker.query_timeline(DB_PATH, session_id=active_sid, limit=50)
    else:
        # Session not in DB — show all stats with a note
        stats = tracker.query_summary(DB_PATH, session_id=None)
        timeline = tracker.query_timeline(DB_PATH, session_id=None, limit=50)
        active_sid = None

    all_stats = tracker.query_summary(DB_PATH, session_id=None)

    total_in  = int(stats.get("prompt_tokens")     or 0)
    total_out = int(stats.get("completion_tokens") or 0)

    # ── Session selector buttons ───────────────────────────────────────────────
    session_btns = ""
    for s in sessions:
        sid = s["session_id"]
        label = sid[:8]
        badge = _fmt(s["requests"])
        first = datetime.fromtimestamp(s["ts_first"]).strftime("%m-%d %H:%M")
        is_active = "active" if sid == active_sid else ""
        is_live = "🔴" if sid == current_live else ""
        session_btns += (
            f"<a href='/?session={sid}' class='sbtn {is_active}' title='{sid}&#10;{badge} reqs&#10;First: {first}'>"
            f"{label} {is_live}<span class='sbadge'>{badge}</span></a>"
        )
    # Add "current live" button if it has no DB records yet
    if current_live not in [s["session_id"] for s in sessions]:
        session_btns += (
            f"<a href='/?session={current_live}' class='sbtn active' title='{current_live}'>"
            f"{current_live[:8]} 🔴<span class='sbadge'>0</span></a>"
        )
    # "All sessions" button
    all_active = "active" if active_sid is None else ""
    session_btns += (
        f"<a href='/' class='sbtn {all_active}'>All</a>"
    )

    # ── Timeline rows ──────────────────────────────────────────────────────────
    trows = ""
    for r in timeline:
        ts = datetime.fromtimestamp(r["ts"]).strftime("%H:%M:%S")
        trows += (
            f"<tr>"
            f"<td>{ts}</td>"
            f"<td>{_fmt(r['prompt_tokens'])}</td>"
            f"<td class='dim'>"
            f"  {_fmt(r['user_tokens'])} / "
            f"  {_fmt(r['tool_schema_tokens'])} / "
            f"  {_fmt(r['tool_result_tokens'])}"
            f"</td>"
            f"<td>{_fmt(r['completion_tokens'])}</td>"
            f"<td class='purple'>{_fmt(r['reasoning_tokens'])}</td>"
            f"<td class='green'>{_fmt(r['cached_tokens'])}</td>"
            f"</tr>"
        )
    if not trows:
        trows = "<tr><td colspan='6' class='dim' style='text-align:center'>No requests yet</td></tr>"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="refresh" content="10">
  <title>GLM Token Stats</title>
  <style>
    *   {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: 'Courier New', monospace; background: #0f0f1a; color: #d0d0e0;
           padding: 28px 32px; }}
    h1  {{ color: #7eb8f7; font-size: 1.35rem; margin-bottom: 4px; }}
    .sub {{ color: #555; font-size: 0.72rem; margin-bottom: 16px; }}
    .sbar {{ margin-bottom: 22px; }}
    .sbar-label {{ color: #444; font-size: 0.65rem; text-transform: uppercase;
                   letter-spacing: 1px; margin-bottom: 6px; }}
    .sbtn {{ display: inline-block; background: #161628; border: 1px solid #262650;
             color: #7a7a9a; padding: 5px 10px; border-radius: 5px;
             text-decoration: none; font-size: 0.72rem; margin-right: 5px;
             margin-bottom: 4px; white-space: nowrap; transition: all 0.15s; }}
    .sbtn:hover {{ background: #1e1e38; color: #d0d0e0; border-color: #40407a; }}
    .sbtn.active {{ background: #1e2a4a; color: #7eb8f7; border-color: #3a5aaa; }}
    .sbadge {{ font-size: 0.62rem; background: #222245; color: #666;
               padding: 1px 5px; border-radius: 8px; margin-left: 4px; }}
    .cards {{ display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }}
    .card {{ background: #161628; border: 1px solid #262650; border-radius: 8px;
             padding: 14px 22px; min-width: 155px; }}
    .card .label {{ color: #666; font-size: 0.7rem; text-transform: uppercase;
                    letter-spacing: 1px; }}
    .card .value {{ color: #7eb8f7; font-size: 1.5rem; font-weight: bold;
                    margin-top: 4px; }}
    .card .hint {{ color: #555; font-size: 0.72rem; margin-top: 3px; }}
    .tables {{ display: flex; gap: 22px; flex-wrap: wrap; margin-bottom: 28px; }}
    .tblock {{ flex: 1; min-width: 300px; }}
    h2  {{ color: #8ab4ff; font-size: 0.88rem; margin-bottom: 10px;
           border-bottom: 1px solid #222240; padding-bottom: 5px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.82rem; }}
    th  {{ color: #555; text-align: left; padding: 5px 8px;
           border-bottom: 1px solid #1e1e38; font-weight: normal;
           text-transform: uppercase; font-size: 0.68rem; letter-spacing: 0.5px; }}
    td  {{ padding: 6px 8px; border-bottom: 1px solid #16162a; vertical-align: middle; }}
    tr:last-child td {{ border-bottom: none; }}
    tr:hover td {{ background: #181830; }}
    .bar {{ height: 5px; background: #1e1e38; border-radius: 3px; margin-top: 5px;
            min-width: 60px; }}
    .bar-fill {{ height: 5px; border-radius: 3px; }}
    .green  {{ color: #4aef9a; }}
    .purple {{ color: #bf7aff; }}
    .orange {{ color: #f7b84a; }}
    .dim    {{ color: #555; }}
    .badge  {{ display: inline-block; font-size: 0.65rem; padding: 1px 6px;
               border-radius: 10px; background: #222245; color: #888;
               vertical-align: middle; margin-left: 6px; }}
    .reset-btn {{ background: #1e1e38; border: 1px solid #36366a; color: #bbb;
                  padding: 5px 12px; border-radius: 5px; cursor: pointer;
                  font-family: monospace; font-size: 0.78rem; }}
    .reset-btn:hover {{ background: #28284a; color: #eee; }}
  </style>
</head>
<body>
  <h1>GLM Token Statistics</h1>
  <div class="sub">
    auto-refresh every 10 s &nbsp;·&nbsp;
    <a href="/api/stats" style="color:#555">JSON</a>
    &nbsp;·&nbsp;
    <form action="/api/reset" method="post" style="display:inline">
      <button class="reset-btn">⟳ New session</button>
    </form>
  </div>

  <!-- Session selector -->
  <div class="sbar">
    <div class="sbar-label">Sessions</div>
    {session_btns}
  </div>

  <!-- Summary cards -->
  <div class="cards">
    <div class="card">
      <div class="label">Requests</div>
      <div class="value">{_fmt(stats.get('requests'))}</div>
      <div class="hint">all-time: {_fmt(all_stats.get('requests'))}</div>
    </div>
    <div class="card">
      <div class="label">Total tokens</div>
      <div class="value">{_fmt(stats.get('total_tokens'))}</div>
      <div class="hint">all-time: {_fmt(all_stats.get('total_tokens'))}</div>
    </div>
    <div class="card">
      <div class="label">Prompt tokens</div>
      <div class="value">{_fmt(total_in)}</div>
      <div class="hint green">cached {_fmt(stats.get('cached_tokens'))}
        ({_pct(stats.get('cached_tokens'), total_in)})</div>
    </div>
    <div class="card">
      <div class="label">Completion tokens</div>
      <div class="value">{_fmt(total_out)}</div>
      <div class="hint purple">reasoning {_fmt(stats.get('reasoning_tokens'))}
        ({_pct(stats.get('reasoning_tokens'), total_out)})</div>
    </div>
  </div>

  <!-- Breakdown tables -->
  <div class="tables">
    <div class="tblock">
      <h2>Input breakdown <span class="badge">estimated</span></h2>
      <table>
        <tr><th>Category</th><th>Tokens</th><th>%</th><th style="min-width:80px">Bar</th></tr>
        <tr>
          <td>User input</td>
          <td>{_fmt(stats.get('user_tokens'))}</td>
          <td>{_pct(stats.get('user_tokens'), total_in)}</td>
          <td>{_bar(stats.get('user_tokens'), total_in, '#4a9eff')}</td>
        </tr>
        <tr>
          <td>System prompt</td>
          <td>{_fmt(stats.get('system_tokens'))}</td>
          <td>{_pct(stats.get('system_tokens'), total_in)}</td>
          <td>{_bar(stats.get('system_tokens'), total_in, '#f7b84a')}</td>
        </tr>
        <tr>
          <td>Tool schemas</td>
          <td>{_fmt(stats.get('tool_schema_tokens'))}</td>
          <td>{_pct(stats.get('tool_schema_tokens'), total_in)}</td>
          <td>{_bar(stats.get('tool_schema_tokens'), total_in, '#4aef9a')}</td>
        </tr>
        <tr>
          <td>Tool results</td>
          <td>{_fmt(stats.get('tool_result_tokens'))}</td>
          <td>{_pct(stats.get('tool_result_tokens'), total_in)}</td>
          <td>{_bar(stats.get('tool_result_tokens'), total_in, '#f7b84a')}</td>
        </tr>
        <tr>
          <td>History</td>
          <td>{_fmt(stats.get('history_tokens'))}</td>
          <td>{_pct(stats.get('history_tokens'), total_in)}</td>
          <td>{_bar(stats.get('history_tokens'), total_in, '#6a7aef')}</td>
        </tr>
      </table>
    </div>

    <div class="tblock">
      <h2>Output breakdown <span class="badge">estimated</span></h2>
      <table>
        <tr><th>Category</th><th>Tokens</th><th>%</th><th style="min-width:80px">Bar</th></tr>
        <tr>
          <td class="purple">Reasoning (think)</td>
          <td class="purple">{_fmt(stats.get('reasoning_tokens'))}</td>
          <td>{_pct(stats.get('reasoning_tokens'), total_out)}</td>
          <td>{_bar(stats.get('reasoning_tokens'), total_out, '#bf7aff')}</td>
        </tr>
        <tr>
          <td>Response</td>
          <td>{_fmt(stats.get('response_tokens'))}</td>
          <td>{_pct(stats.get('response_tokens'), total_out)}</td>
          <td>{_bar(stats.get('response_tokens'), total_out, '#4a9eff')}</td>
        </tr>
        <tr>
          <td class="green">KV-cache hits</td>
          <td class="green">{_fmt(stats.get('cached_tokens'))}</td>
          <td>{_pct(stats.get('cached_tokens'), total_in)}</td>
          <td>{_bar(stats.get('cached_tokens'), total_in, '#4aef9a')}</td>
        </tr>
      </table>
    </div>
  </div>

  <!-- Per-request timeline -->
  <div class="tblock">
    <h2>Recent requests — this session</h2>
    <table>
      <tr>
        <th>Time</th>
        <th>Prompt</th>
        <th class="dim">User / Schemas / Results</th>
        <th>Completion</th>
        <th class="purple">Reasoning</th>
        <th class="green">Cached</th>
      </tr>
      {trows}
    </table>
  </div>
</body>
</html>"""
    return html


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(app, host=STATS_HOST, port=STATS_PORT, log_level="warning")
