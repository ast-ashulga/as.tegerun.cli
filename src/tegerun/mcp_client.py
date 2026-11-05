"""Shared stdio connection to the fast-mcp-telegram server (used by reader and notifier)."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from tegerun import config


@asynccontextmanager
async def open_session(verbose: bool = False) -> AsyncIterator[ClientSession]:
    """Start the MCP server subprocess with this project's Telegram session and yield an initialized client."""
    creds = config.load_credentials()
    config.require_session()
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    config.LOG_DIR.mkdir(exist_ok=True)

    params = StdioServerParameters(
        command=config.venv_script("fast-mcp-telegram"),
        args=[],
        env=config.server_env(creds),
        # cwd away from the project root so the server never auto-loads our .env
        cwd=str(config.SESSION_DIR),
    )
    log = None if verbose else open(config.LOG_DIR / "server.log", "a")
    try:
        async with stdio_client(params, errlog=log or sys.stderr) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
    finally:
        if log:
            log.close()
