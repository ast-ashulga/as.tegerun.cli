"""Send a notification to the chat configured in `.config.yaml` (`notify:`) via fast-mcp-telegram.

This is only the delivery tool. Deciding *what* is worth sending and *when* is up to the caller.
The destination always comes from the config, never from a function argument, and nothing is
sent unless `notify.enabled` is true. The message is sent from the logged-in Telegram account
(the one in ./.sessions), using the server's `send_message` tool.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from tegerun import config, mcp_client
from tegerun.reader import extract_payload

TELEGRAM_LIMIT = 4096


@dataclass
class NotifyResult:
    sent: bool
    skipped: bool = False  # notifications are disabled in the config; nothing was attempted
    error: str | None = None
    message_ids: list[int] = field(default_factory=list)


def split_text(text: str, limit: int = TELEGRAM_LIMIT) -> list[str]:
    """Cut text into parts of at most `limit` chars, preferring line breaks, then spaces."""
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit + 1)
        if cut <= 0:
            cut = rest.rfind(" ", 0, limit + 1)
        if cut <= 0:
            parts.append(rest[:limit])
            rest = rest[limit:]
        else:
            parts.append(rest[:cut])
            rest = rest[cut + 1 :]  # the break char itself is dropped
    parts.append(rest)
    return parts


def _send_error(result: Any, payload: dict[str, Any]) -> str | None:
    if getattr(result, "isError", False) or payload.get("ok") is False or "message_id" not in payload:
        return str(payload.get("error") or payload.get("message") or "unknown error")
    return None


async def deliver(
    session: Any,
    chat_id: str,
    text: str,
    reply_to: int | None = None,
    limit: int = TELEGRAM_LIMIT,
    topic_id: int | None = None,
) -> NotifyResult:
    """Send `text` (split if needed) to `chat_id` through an open MCP session. Never raises.

    `topic_id` (forum topic) applies to every part; `reply_to` only to the first one.
    """
    if not text.strip():
        return NotifyResult(sent=False, error="empty message")
    ids: list[int] = []
    for i, part in enumerate(split_text(text.strip(), limit)):
        args: dict[str, Any] = {"chat_id": chat_id, "message": part}
        target = reply_to if i == 0 and reply_to is not None else topic_id
        if target is not None:
            args["reply_to_id"] = target
        try:
            res = await session.call_tool("send_message", args)
        except Exception as exc:
            return NotifyResult(sent=False, message_ids=ids, error=f"{type(exc).__name__}: {exc}")
        payload = extract_payload(res)
        if "message_id" not in payload and isinstance(payload.get("result"), dict):
            payload = payload["result"]  # FastMCP may wrap the result object
        error = _send_error(res, payload)
        if error:
            return NotifyResult(sent=False, message_ids=ids, error=error)
        ids.append(payload["message_id"])
    return NotifyResult(sent=True, message_ids=ids)


async def send_notification(
    text: str,
    settings: config.NotifySettings | None = None,
    reply_to: int | None = None,
    verbose: bool = False,
) -> NotifyResult:
    """Send `text` to the configured chat. `reply_to` overrides `notify.reply_to`. Never raises."""
    try:
        settings = settings or config.load_notify_settings()
        if not settings.enabled:
            return NotifyResult(sent=False, skipped=True)
        async with mcp_client.open_session(verbose=verbose) as session:
            return await deliver(
                session,
                settings.chat_id,
                text,
                reply_to if reply_to is not None else settings.reply_to,
                topic_id=settings.topic_id,
            )
    except config.ConfigError as exc:
        return NotifyResult(sent=False, error=str(exc))
    except Exception as exc:  # MCP server failed to start, connection dropped, ...
        return NotifyResult(sent=False, error=f"{type(exc).__name__}: {exc}")
