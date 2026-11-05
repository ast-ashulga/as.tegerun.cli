import json
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from tegerun import cli, config, reader

NOW = datetime(2026, 11, 3, 12, 0, tzinfo=timezone.utc)


def test_parse_since_relative():
    assert reader.parse_since("24h", NOW).startswith("2026-11-02T12:00")
    assert reader.parse_since("2w", NOW).startswith("2026-10-20T12:00")


def test_parse_since_iso_and_bad():
    assert reader.parse_since("2026-11-01").startswith("2026-11-01T00:00")
    with pytest.raises(ValueError):
        reader.parse_since("yesterday-ish")


def test_whitelist_and_credentials(tmp_path):
    cfg = tmp_path / "c.yaml"
    cfg.write_text(
        "whitelist:\n"
        "  - {id: 1, name: One, description: First chat}\n"
        "  - {id: '2', name: Two}\n"
        "  - {id: '2', name: Dup}\n"
    )
    assert config.load_whitelist(cfg) == [
        config.Chat("1", "One", "First chat"),
        config.Chat("2", "Two", ""),
    ]
    for bad in ("{}", "whitelist: [1, 2]", "whitelist: [{id: 1}]", "whitelist: [{name: x}]", "a: [unclosed"):
        cfg.write_text(bad)
        with pytest.raises(config.ConfigError):
            config.load_whitelist(cfg)
    env = tmp_path / ".env"
    env.write_text("tege_api_id=1\ntege_api_hash=abc\n")
    c = config.load_credentials(env)
    assert (c.api_id, c.api_hash) == ("1", "abc") and "abc" not in repr(c)
    env.write_text("tege_api_id=1\n")
    with pytest.raises(config.ConfigError):
        config.load_credentials(env)


def test_server_env_maps_and_hides_nothing_extra():
    env = config.server_env(config.Credentials("7", "h"))
    assert env["API_ID"] == "7" and env["API_HASH"] == "h" and env["DO_NOT_TRACK"] == "1"
    assert "tege_api_hash" not in env


def _res(structured=None, text=None, is_error=False):
    content = [SimpleNamespace(text=text)] if text is not None else []
    return SimpleNamespace(structuredContent=structured, content=content, isError=is_error)


def test_result_parsing_variants():
    ok = {"messages": [{"id": 1, "text": "hi"}], "has_more": True}
    for res in (_res(structured=ok), _res(structured={"result": ok}), _res(text=json.dumps(ok))):
        r = reader.to_chat_result("p", res)
        assert r.error is None and r.has_more and r.messages[0]["id"] == 1
    assert reader.to_chat_result("p", _res(text=json.dumps({"ok": False, "error": "nope"}))).error == "nope"
    assert reader.to_chat_result("p", _res(text="boom", is_error=True)).error == "boom"


def test_rendering():
    msg = {"date": "2026-11-03T10:05:00+00:00", "sender": {"first_name": "A", "last_name": "B"}, "text": "x\ny"}
    assert cli.format_message(msg) == "[2026-11-03 10:05] A B: x\n    y"
    chats = {"1": config.Chat("1", "Family", "Close relatives")}
    out = cli.render_text([reader.ChatResult("1", error="bad"), reader.ChatResult("2")], chats)
    assert "=== Family (1) ===" in out and "Close relatives" in out
    assert "ERROR: bad" in out and "(no messages)" in out
    data = json.loads(cli.render_json([reader.ChatResult("1")], chats))
    assert data[0]["name"] == "Family" and data[0]["description"] == "Close relatives"


def test_read_rejects_non_whitelisted_chat(capsys):
    assert cli.main(["read", "--chat", "999999"]) == 2
    assert "Not in whitelist" in capsys.readouterr().err
