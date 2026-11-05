"""Read-only access to Telegram via the fast-mcp-telegram MCP server (stdio).

Only the `get_messages` tool is ever called. Nothing here can send or edit.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tegerun import mcp_client

_REL = re.compile(r"^(\d+)\s*([mhdw])$")
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_since(value: str, now: datetime | None = None, flag: str = "--since") -> str:
    """'90m' / '24h' / '7d' / '2w' (ago) or an ISO date(-time) -> ISO-8601 UTC string."""
    now = now or datetime.now(timezone.utc)
    value = value.strip()
    m = _REL.match(value)
    if m:
        delta = timedelta(**{_UNITS[m.group(2)]: int(m.group(1))})
        return (now - delta).isoformat()
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Bad {flag} {value!r}: use 24h / 7d / 2w or an ISO date") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


@dataclass
class ChatResult:
    peer_id: str
    messages: list[dict[str, Any]] = field(default_factory=list)
    has_more: bool = False
    error: str | None = None


def extract_payload(result: Any) -> dict[str, Any]:
    """Pull the JSON object out of an MCP CallToolResult (structured or text)."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps non-object returns as {"result": ...}
        inner = structured.get("result")
        return inner if isinstance(inner, dict) and "messages" in inner else structured
    text = "".join(getattr(c, "text", "") for c in (result.content or []))
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"ok": False, "error": text.strip() or "empty response"}
    return data if isinstance(data, dict) else {"ok": False, "error": "unexpected response shape"}


def to_chat_result(peer_id: str, result: Any) -> ChatResult:
    payload = extract_payload(result)
    if getattr(result, "isError", False) or payload.get("ok") is False or "messages" not in payload:
        err = payload.get("error") or payload.get("message") or "unknown error"
        return ChatResult(peer_id, error=str(err))
    return ChatResult(peer_id, messages=payload["messages"], has_more=bool(payload.get("has_more")))


async def fetch_recent(
    peers: list[str],
    limit: int,
    since: str | None,
    verbose: bool = False,
    until: str | None = None,
) -> list[ChatResult]:
    results: list[ChatResult] = []
    async with mcp_client.open_session(verbose=verbose) as session:
        for peer in peers:
            args: dict[str, Any] = {"chat_id": peer, "limit": limit}
            if since:
                args["min_date"] = since
            if until:
                args["max_date"] = until
            try:
                res = await session.call_tool("get_messages", args)
                results.append(to_chat_result(peer, res))
            except Exception as exc:  # one bad peer must not kill the run
                results.append(ChatResult(peer, error=f"{type(exc).__name__}: {exc}"))
    return results
