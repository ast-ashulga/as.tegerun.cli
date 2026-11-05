"""Shared stdio connection to the fast-mcp-telegram server (used by reader and notifier).

The server exposes far more than this project needs (raw MTProto, edit, send-to-phone, ...) and the
session is a full user account. So every call goes through `GuardedSession`, which only lets the
tools the caller asked for through and gives each call a deadline.
"""

from __future__ import annotations

import asyncio
import os
import sys
from collections.abc import AsyncIterator, Collection
from contextlib import asynccontextmanager
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tegerun import config

READ_TOOLS = frozenset({"get_messages"})
SEND_TOOLS = frozenset({"send_message"})
INIT_TIMEOUT = 60.0  # seconds to start the server and finish the MCP handshake
CALL_TIMEOUT = 120.0  # seconds for one tool call (a Telegram flood-wait must not hang a run forever)


class ToolNotAllowed(Exception):
    """Raised before anything is sent to the server: this tool is outside the caller's allowlist."""


class GuardedSession:
    """Wraps an MCP ClientSession: allowlisted tool names only, and a timeout on every call."""

    def __init__(self, session: Any, allow: Collection[str], timeout: float = CALL_TIMEOUT) -> None:
        self._session = session
        self._allow = frozenset(allow)
        self._timeout = timeout

    async def call_tool(self, name: str, arguments: dict[str, Any] | None = None) -> Any:
        if name not in self._allow:
            raise ToolNotAllowed(f"MCP tool {name!r} is not allowed here (allowed: {', '.join(sorted(self._allow))}).")
        try:
            return await asyncio.wait_for(self._session.call_tool(name, arguments), self._timeout)
        except asyncio.TimeoutError:
            raise TimeoutError(f"MCP tool {name!r} did not answer within {self._timeout:.0f}s") from None


def _open_log() -> Any:
    """Append-only server log, readable by this user only (it can contain message text)."""
    config.LOG_DIR.mkdir(mode=0o700, exist_ok=True)
    fd = os.open(config.LOG_DIR / "server.log", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    return os.fdopen(fd, "a")


@asynccontextmanager
async def open_session(verbose: bool = False, *, allow: Collection[str]) -> AsyncIterator[GuardedSession]:
    """Start the MCP server subprocess with this project's Telegram session and yield a guarded client.

    `allow` names the tools the caller may use (READ_TOOLS / SEND_TOOLS): least privilege per caller.
    """
    creds = config.load_credentials()
    config.require_session()
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)

    params = StdioServerParameters(
        command=config.venv_script("fast-mcp-telegram"),
        args=[],
        env=config.server_env(creds),
        # cwd away from the project root so the server never auto-loads our .env
        cwd=str(config.SESSION_DIR),
    )
    log = None if verbose else _open_log()
    try:
        async with stdio_client(params, errlog=log or sys.stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await asyncio.wait_for(session.initialize(), INIT_TIMEOUT)
                yield GuardedSession(session, allow)
    finally:
        if log:
            log.close()
