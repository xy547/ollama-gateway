"""外挂三：会话记忆。

模型是无状态的，每一轮都要把完整历史重新发过去。这个模块负责：
- 落库（SQLite，单文件，零部署成本）；
- 下次同 ``session_id`` 进来时自动把最近 N 轮拼回 messages。

注意这跟"长期记忆"不是一回事：这里只是把对话历史持久化，
真正让模型"记住事实"还需要摘要压缩（超出本项目的演示范围）。
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from ..config import settings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_session ON messages(session_id, id);
"""


def _db_path() -> Path:
    return Path(settings.data_path) / "memory.db"


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(_db_path(), timeout=10)
    conn.executescript(_SCHEMA)
    return conn


def append_turn(session_id: str, user_text: str, assistant_text: str) -> None:
    """写一轮对话。空内容不写，避免污染历史。"""
    if not session_id:
        return
    now = time.time()
    with _conn() as conn:
        conn.executemany(
            "INSERT INTO messages(session_id, role, content, created_at) VALUES (?,?,?,?)",
            [
                (session_id, "user", user_text, now),
                (session_id, "assistant", assistant_text, now),
            ],
        )


def recent_messages(session_id: str, limit: int = 12) -> list[dict[str, Any]]:
    """取最近 limit 条历史（按时间正序返回）。"""
    if not session_id:
        return []
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    return [{"role": r[0], "content": r[1], "created_at": r[2]} for r in reversed(rows)]


def list_sessions(limit: int = 20) -> list[dict[str, Any]]:
    if not _db_path().exists():
        return []
    with _conn() as conn:
        rows = conn.execute(
            """SELECT session_id, COUNT(*), MAX(created_at) FROM messages
               GROUP BY session_id ORDER BY MAX(created_at) DESC LIMIT ?""",
            (limit,),
        ).fetchall()
    return [{"session_id": r[0], "messages": r[1], "last_at": r[2]} for r in rows]


def clear_session(session_id: str) -> dict[str, Any]:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
        return {"session_id": session_id, "deleted": cur.rowcount}
