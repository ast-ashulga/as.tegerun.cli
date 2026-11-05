"""reader.fetch_recent / mcp_client.GuardedSession driven through a fake MCP session.

These cover the path every Telegram read takes, which earlier tests never exercised (a missing
import in reader.py went unnoticed because of that).
"""

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tegerun import config, mcp_client, reader
from tegerun.textutil import printable

T0 = datetime(2026, 11, 4, 10, 0, tzinfo=timezone.utc)


def _res(payload):
    return SimpleNamespace(structuredContent=payload, content=[], isError=False)


class FakeServer:
    """Behaves like get_messages: the NEWEST `limit` messages of the window, oldest first."""

    def __init__(self, count, per_chat_error=None):
        self.messages = [{"id": i, "text": f"m{i}", "date": (T0 + timedelta(minutes=i)).isoformat()} for i in range(1, count + 1)]
        self.calls = []
        self.error = per_chat_error

    async def call_tool(self, name, args):
        self.calls.append((name, dict(args)))
        if self.error:
            raise RuntimeError(self.error)
        lo = datetime.fromisoformat(args["min_date"]) if "min_date" in args else None
        hi = datetime.fromisoformat(args["max_date"]) if "max_date" in args else None
        window = [
            m for m in self.messages
            if (lo is None or datetime.fromisoformat(m["date"]) >= lo) and (hi is None or datetime.fromisoformat(m["date"]) <= hi)
        ]
        page = window[-args["limit"]:]
        return _res({"messages": page, "has_more": len(window) > len(page)})


@pytest.fixture
def server(monkeypatch):
    def install(srv):
        opened = []

        class _Ctx:
            def __init__(self, allow):
                self.allow = allow

            async def __aenter__(self):
                opened.append(self.allow)
                return mcp_client.GuardedSession(srv, self.allow)

            async def __aexit__(self, *exc):
                return False

        monkeypatch.setattr(mcp_client, "open_session", lambda verbose=False, *, allow: _Ctx(allow))
        return opened

    return install


async def test_fetch_recent_reads_through_the_session_with_read_only_tools(server):
    srv = FakeServer(3)
    opened = server(srv)
    [r] = await reader.fetch_recent(["c1"], 10, None, allowed={"c1"})
    assert [m["id"] for m in r.messages] == [1, 2, 3] and not r.has_more and r.error is None
    assert opened == [mcp_client.READ_TOOLS]
    assert [name for name, _ in srv.calls] == ["get_messages"]


async def test_fetch_recent_refuses_chats_outside_the_whitelist_before_connecting(server):
    opened = server(FakeServer(1))
    with pytest.raises(config.ConfigError, match="other"):
        await reader.fetch_recent(["c1", "other"], 10, None, allowed={"c1"})
    assert opened == []


async def test_without_pagination_only_the_newest_page_is_read_and_has_more_says_so(server):
    server(FakeServer(10))
    [r] = await reader.fetch_recent(["c1"], 4, None, allowed={"c1"})
    assert [m["id"] for m in r.messages] == [7, 8, 9, 10] and r.has_more is True


async def test_pagination_reads_the_whole_window_oldest_first_without_duplicates(server):
    srv = FakeServer(10)
    server(srv)
    [r] = await reader.fetch_recent(["c1"], 4, None, allowed={"c1"}, paginate=True)
    assert [m["id"] for m in r.messages] == list(range(1, 11))
    assert r.has_more is False and len(srv.calls) >= 3
    assert "max_date" not in srv.calls[0][1] and "max_date" in srv.calls[1][1]


async def test_pagination_respects_since_and_until(server):
    server(FakeServer(10))
    since = (T0 + timedelta(minutes=3)).isoformat()
    until = (T0 + timedelta(minutes=8)).isoformat()
    [r] = await reader.fetch_recent(["c1"], 2, since, until=until, allowed={"c1"}, paginate=True)
    assert [m["id"] for m in r.messages] == [3, 4, 5, 6, 7, 8]


async def test_pagination_gives_up_visibly_after_max_pages(server, monkeypatch):
    monkeypatch.setattr(reader, "MAX_PAGES", 2)
    server(FakeServer(10))
    [r] = await reader.fetch_recent(["c1"], 2, None, allowed={"c1"}, paginate=True)
    assert [m["id"] for m in r.messages] == [8, 9, 10]  # page 2 starts one second above message 9: it overlaps
    assert r.has_more is True  # the caller must not treat this window as fully read


async def test_a_failing_chat_is_reported_and_the_others_still_read(server):
    class Mixed(FakeServer):
        async def call_tool(self, name, args):
            if args["chat_id"] == "bad":
                raise RuntimeError("FLOOD_WAIT")
            return await super().call_tool(name, args)

    server(Mixed(2))
    bad, good = await reader.fetch_recent(["bad", "good"], 10, None, allowed={"bad", "good"})
    assert "FLOOD_WAIT" in bad.error and len(good.messages) == 2


def test_chat_result_rejects_a_malformed_messages_field():
    assert reader.to_chat_result("p", _res({"messages": "oops"})).error
    ok = reader.to_chat_result("p", _res({"messages": [{"id": 1}, "junk", 3]}))
    assert ok.messages == [{"id": 1}]


def test_parse_since_and_config_share_one_duration_grammar():
    assert reader.parse_since("90 m", T0).startswith("2026-11-04T08:30")
    with pytest.raises(ValueError):
        reader.parse_since("0h", T0)  # a zero duration is not a window


# --- GuardedSession ------------------------------------------------------------


class _Session:
    def __init__(self, delay=0):
        self.calls, self.delay = [], delay

    async def call_tool(self, name, args=None):
        await asyncio.sleep(self.delay)
        self.calls.append(name)
        return "ok"


async def test_guarded_session_only_lets_allowlisted_tools_through():
    inner = _Session()
    guarded = mcp_client.GuardedSession(inner, mcp_client.READ_TOOLS)
    assert await guarded.call_tool("get_messages", {}) == "ok"
    for forbidden in ("send_message", "edit_message", "invoke_mtproto", "send_message_to_phone"):
        with pytest.raises(mcp_client.ToolNotAllowed):
            await guarded.call_tool(forbidden, {})
    assert inner.calls == ["get_messages"]  # the forbidden ones never reached the server


async def test_send_session_cannot_read():
    guarded = mcp_client.GuardedSession(_Session(), mcp_client.SEND_TOOLS)
    with pytest.raises(mcp_client.ToolNotAllowed):
        await guarded.call_tool("get_messages", {})


async def test_guarded_session_times_out_instead_of_hanging():
    guarded = mcp_client.GuardedSession(_Session(delay=5), mcp_client.READ_TOOLS, timeout=0.05)
    with pytest.raises(TimeoutError, match="did not answer"):
        await guarded.call_tool("get_messages", {})


# --- terminal safety -----------------------------------------------------------


def test_printable_removes_terminal_escape_sequences_but_keeps_text_and_newlines():
    evil = "hi\x1b]0;pwned\x07 there\r\x1b[2J\nnext\tline \x9b31m"
    clean = printable(evil)
    assert "\x1b" not in clean and "\x07" not in clean and "\r" not in clean and "\x9b" not in clean
    assert "hi" in clean and "there" in clean and "\nnext\tline" in clean
    assert printable("Привет, мир 🎯") == "Привет, мир 🎯"


async def test_pagination_also_works_when_the_server_sends_dates_without_an_offset(server):
    class Naive(FakeServer):
        async def call_tool(self, name, args):
            res = await super().call_tool(name, args)
            page = res.structuredContent["messages"]
            # copies, so the fake server's own (aware) dates stay intact; the client sees naive ones
            res.structuredContent["messages"] = [{**m, "date": m["date"].replace("+00:00", "")} for m in page]
            return res

    server(Naive(10))
    [r] = await reader.fetch_recent(["c1"], 4, None, allowed={"c1"}, paginate=True)
    assert [m["id"] for m in r.messages] == list(range(1, 11)) and r.has_more is False
