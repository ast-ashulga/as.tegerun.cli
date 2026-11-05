"""Read-only access to Telegram via the fast-mcp-telegram MCP server (stdio).

Only the `get_messages` tool is ever called (the session is opened with READ_TOOLS, so nothing here
can send or edit), and only for chats the caller explicitly allows (the whitelist).
"""

from __future__ import annotations

import json
from collections.abc import Collection
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from tegerun import config, mcp_client

MAX_PAGES = 20  # pages of `limit` messages read per chat when paginating


def parse_since(value: str, now: datetime | None = None, flag: str = "--since") -> str:
    """'90m' / '24h' / '7d' / '2w' (ago) or an ISO date(-time) -> ISO-8601 UTC string."""
    now = now or datetime.now(timezone.utc)
    value = value.strip()
    delta = config.try_parse_duration(value)
    if delta is not None:
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
    has_more: bool = False  # the window holds messages that were not read
    error: str | None = None


def extract_payload(result: Any) -> dict[str, Any]:
    """Pull the JSON object out of an MCP CallToolResult (structured or text)."""
    structured = getattr(result, "structuredContent", None)
    if isinstance(structured, dict):
        # FastMCP wraps non-object returns as {"result": ...}
        inner = structured.get("result")
        return inner if set(structured) == {"result"} and isinstance(inner, dict) else structured
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
    messages = payload["messages"]
    if not isinstance(messages, list):
        return ChatResult(peer_id, error="unexpected response shape: 'messages' is not a list")
    return ChatResult(
        peer_id,
        messages=[m for m in messages if isinstance(m, dict)],
        has_more=bool(payload.get("has_more")),
    )


def _oldest_plus_a_second(messages: list[dict[str, Any]]) -> str | None:
    """Upper bound for the next (older) page. One second above the oldest message, because it is not
    documented whether the server's `max_date` is inclusive; the overlap is removed by message id."""
    try:
        dates = [datetime.fromisoformat(str(m["date"])) for m in messages]
    except (KeyError, ValueError):
        return None
    if not dates:
        return None
    # A date without an offset is taken as UTC, as everywhere else in this project.
    dates = [d if d.tzinfo else d.replace(tzinfo=timezone.utc) for d in dates]
    return (min(dates) + timedelta(seconds=1)).isoformat()


async def _fetch_chat(
    session: Any, peer: str, limit: int, since: str | None, until: str | None, paginate: bool
) -> ChatResult:
    """One chat. The server returns the *newest* `limit` messages of the window, so when the window
    holds more (`has_more`) and `paginate` is on, older pages are fetched with a moving `max_date`."""
    messages: list[dict[str, Any]] = []
    seen: set[Any] = set()
    page_until = until
    has_more = False
    for _ in range(MAX_PAGES if paginate else 1):
        args: dict[str, Any] = {"chat_id": peer, "limit": limit}
        if since:
            args["min_date"] = since
        if page_until:
            args["max_date"] = page_until
        page = to_chat_result(peer, await session.call_tool("get_messages", args))
        if page.error:
            return page
        fresh = [m for m in page.messages if m.get("id") is None or m["id"] not in seen]
        seen.update(m["id"] for m in fresh if m.get("id") is not None)
        messages = fresh + messages  # pages come newest first, each one ascending
        has_more = page.has_more
        page_until = _oldest_plus_a_second(fresh) if fresh else None
        if not has_more or page_until is None:
            break
    return ChatResult(peer, messages=messages, has_more=has_more)


async def fetch_recent(
    peers: list[str],
    limit: int,
    since: str | None,
    verbose: bool = False,
    until: str | None = None,
    *,
    allowed: Collection[str],
    paginate: bool = False,
) -> list[ChatResult]:
    """Read `peers` (each must be in `allowed`). `limit` is per page; with `paginate` the whole window
    is read (up to MAX_PAGES pages), otherwise only its newest `limit` messages and `has_more` says so."""
    outside = [p for p in peers if p not in allowed]
    if outside:
        raise config.ConfigError(f"Not in whitelist: {', '.join(outside)}.")
    results: list[ChatResult] = []
    async with mcp_client.open_session(verbose=verbose, allow=mcp_client.READ_TOOLS) as session:
        for peer in peers:
            try:
                results.append(await _fetch_chat(session, peer, limit, since, until, paginate))
            except Exception as exc:  # one bad peer must not kill the run
                results.append(ChatResult(peer, error=f"{type(exc).__name__}: {exc}"))
    return results
