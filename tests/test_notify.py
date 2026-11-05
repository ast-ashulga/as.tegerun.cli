import json
from types import SimpleNamespace

import pytest

from tegerun import cli, config, notifier

CHAT = "-1003842301085"


def _cfg(tmp_path, body: str):
    path = tmp_path / "c.yaml"
    path.write_text(body)
    return path


# --- config -----------------------------------------------------------------


def test_notify_settings_default_to_disabled_when_section_missing(tmp_path):
    s = config.load_notify_settings(_cfg(tmp_path, "whitelist: []\n"))
    assert s.enabled is False and s.chat_id == "" and s.reply_to is None


def test_notify_settings_read_all_fields(tmp_path):
    s = config.load_notify_settings(
        _cfg(tmp_path, f"notify:\n  enabled: true\n  chat_id: '{CHAT}'\n  reply_to: 273\n")
    )
    assert (s.enabled, s.chat_id, s.reply_to) == (True, CHAT, 273)


def test_notify_chat_id_may_be_a_plain_yaml_number(tmp_path):
    s = config.load_notify_settings(_cfg(tmp_path, f"notify:\n  enabled: true\n  chat_id: {CHAT}\n"))
    assert s.chat_id == CHAT


@pytest.mark.parametrize(
    "body",
    [
        "notify: [1]\n",  # not a mapping
        "notify:\n  enabled: 'false'\n  chat_id: x\n",  # a string 'false' is truthy: must be rejected
        "notify:\n  enabled: true\n",  # enabled without a destination
        f"notify:\n  enabled: true\n  chat_id: '{CHAT}'\n  reply_to: abc\n",
    ],
)
def test_notify_settings_reject_bad_values(tmp_path, body):
    with pytest.raises(config.ConfigError):
        config.load_notify_settings(_cfg(tmp_path, body))


def test_project_config_targets_ketchupclaw():
    # `enabled` is the user's live switch, so it is not pinned here; the default (False) is
    # covered by test_notify_settings_default_to_disabled_when_section_missing.
    assert config.load_notify_settings().chat_id == CHAT


# --- splitting --------------------------------------------------------------


def test_split_text_keeps_short_text_whole():
    assert notifier.split_text("hello") == ["hello"]


def test_split_text_prefers_line_breaks_and_respects_limit():
    text = "aaaa\nbbbb\ncccc"
    parts = notifier.split_text(text, limit=9)
    assert parts == ["aaaa\nbbbb", "cccc"]
    assert all(len(p) <= 9 for p in parts)


def test_split_text_hard_cuts_a_long_word_and_loses_nothing():
    text = "x" * 25
    parts = notifier.split_text(text, limit=10)
    assert [len(p) for p in parts] == [10, 10, 5]
    assert "".join(parts) == text


def test_split_text_default_limit_is_telegrams_4096():
    parts = notifier.split_text("y" * 4097)
    assert [len(p) for p in parts] == [4096, 1]


# --- delivery ---------------------------------------------------------------


class FakeSession:
    def __init__(self, responses=None, raises=None):
        self.calls = []
        self._responses = list(responses or [])
        self._raises = raises

    async def call_tool(self, name, args):
        self.calls.append((name, args))
        if self._raises:
            raise self._raises
        if self._responses:
            return self._responses.pop(0)
        return _ok(len(self.calls))


def _ok(message_id):
    return SimpleNamespace(
        structuredContent={"message_id": message_id, "status": "sent"}, content=[], isError=False
    )


def _fail(text):
    return SimpleNamespace(structuredContent=None, content=[SimpleNamespace(text=text)], isError=True)


async def test_deliver_sends_to_configured_chat_via_send_message():
    session = FakeSession()
    result = await notifier.deliver(session, CHAT, "hi")
    assert session.calls == [("send_message", {"chat_id": CHAT, "message": "hi"})]
    assert result.sent is True and result.error is None and result.message_ids == [1]


async def test_deliver_replies_only_on_the_first_part():
    session = FakeSession()
    await notifier.deliver(session, CHAT, "a\nb\nc", reply_to=273, limit=1)
    assert [c[1].get("reply_to_id") for c in session.calls] == [273, None, None]
    assert [c[1]["message"] for c in session.calls] == ["a", "b", "c"]


async def test_deliver_accepts_result_wrapped_in_a_result_key():
    wrapped = SimpleNamespace(structuredContent={"result": {"message_id": 7, "status": "sent"}}, content=[], isError=False)
    result = await notifier.deliver(FakeSession(responses=[wrapped]), CHAT, "hi")
    assert result.sent is True and result.message_ids == [7]


async def test_deliver_reports_tool_error_and_stops():
    session = FakeSession(responses=[_ok(1), _fail("CHAT_WRITE_FORBIDDEN")])
    result = await notifier.deliver(session, CHAT, "a\nb\nc", limit=1)
    assert result.sent is False and "CHAT_WRITE_FORBIDDEN" in result.error
    assert result.message_ids == [1]
    assert len(session.calls) == 2  # nothing is sent after the failure


async def test_deliver_reports_ok_false_payload_as_error():
    bad = SimpleNamespace(structuredContent={"ok": False, "error": "no access"}, content=[], isError=False)
    result = await notifier.deliver(FakeSession(responses=[bad]), CHAT, "hi")
    assert result.sent is False and result.error == "no access"


async def test_deliver_turns_exceptions_into_error_result():
    result = await notifier.deliver(FakeSession(raises=RuntimeError("boom")), CHAT, "hi")
    assert result.sent is False and "RuntimeError: boom" in result.error


async def test_deliver_rejects_empty_text_without_calling_the_server():
    session = FakeSession()
    result = await notifier.deliver(session, CHAT, "   \n ")
    assert result.sent is False and result.error and session.calls == []


# --- send_notification: the flag and the opening of the MCP session -----------


async def test_send_notification_is_a_noop_when_disabled(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("must not open an MCP session when disabled")

    monkeypatch.setattr(notifier.mcp_client, "open_session", boom)
    settings = config.NotifySettings(enabled=False, chat_id=CHAT, reply_to=None)
    result = await notifier.send_notification("hi", settings=settings)
    assert result.sent is False and result.skipped is True and result.error is None


async def test_send_notification_uses_settings_chat_and_reply_to(monkeypatch):
    session = FakeSession()

    class _Ctx:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(notifier.mcp_client, "open_session", lambda verbose=False: _Ctx())
    settings = config.NotifySettings(enabled=True, chat_id=CHAT, reply_to=273)
    result = await notifier.send_notification("hi", settings=settings)
    assert result.sent is True
    assert session.calls == [("send_message", {"chat_id": CHAT, "message": "hi", "reply_to_id": 273})]


async def test_send_notification_returns_error_when_session_cannot_open(monkeypatch):
    def boom(verbose=False):
        raise config.ConfigError("No Telegram session")

    monkeypatch.setattr(notifier.mcp_client, "open_session", boom)
    settings = config.NotifySettings(enabled=True, chat_id=CHAT, reply_to=None)
    result = await notifier.send_notification("hi", settings=settings)
    assert result.sent is False and "No Telegram session" in result.error


# --- CLI --------------------------------------------------------------------


def test_cli_notify_dry_run_prints_target_and_does_not_send(monkeypatch, capsys):
    monkeypatch.setattr(
        config, "load_notify_settings", lambda *a, **k: config.NotifySettings(True, CHAT, 273)
    )

    def boom(*a, **k):
        raise AssertionError("dry run must not send")

    monkeypatch.setattr(notifier, "send_notification", boom)
    assert cli.main(["notify", "--dry-run", "hello there"]) == 0
    out = capsys.readouterr().out
    assert CHAT in out and "hello there" in out and "273" in out


def test_cli_notify_when_disabled_sends_nothing_and_says_so(monkeypatch, capsys):
    monkeypatch.setattr(
        config, "load_notify_settings", lambda *a, **k: config.NotifySettings(False, CHAT, None)
    )

    def boom(*a, **k):
        raise AssertionError("disabled: must not send")

    monkeypatch.setattr(notifier, "send_notification", boom)
    assert cli.main(["notify", "hello"]) == 1
    assert "disabled" in capsys.readouterr().err


def test_cli_notify_sends_and_reports_ids(monkeypatch, capsys):
    monkeypatch.setattr(
        config, "load_notify_settings", lambda *a, **k: config.NotifySettings(True, CHAT, None)
    )
    seen = {}

    async def fake_send(text, settings=None, reply_to=None, verbose=False):
        seen["text"] = text
        return notifier.NotifyResult(sent=True, message_ids=[42])

    monkeypatch.setattr(notifier, "send_notification", fake_send)
    assert cli.main(["notify", "hello"]) == 0
    assert seen["text"] == "hello" and "42" in capsys.readouterr().out


def test_cli_notify_failure_returns_nonzero(monkeypatch, capsys):
    monkeypatch.setattr(
        config, "load_notify_settings", lambda *a, **k: config.NotifySettings(True, CHAT, None)
    )

    async def fake_send(text, settings=None, reply_to=None, verbose=False):
        return notifier.NotifyResult(sent=False, error="no access")

    monkeypatch.setattr(notifier, "send_notification", fake_send)
    assert cli.main(["notify", "hello"]) == 1
    assert "no access" in capsys.readouterr().err


def test_cli_notify_reads_text_from_stdin(monkeypatch, capsys):
    import io

    monkeypatch.setattr(
        config, "load_notify_settings", lambda *a, **k: config.NotifySettings(True, CHAT, None)
    )
    monkeypatch.setattr("sys.stdin", io.StringIO("from stdin\n"))
    assert cli.main(["notify", "--dry-run"]) == 0
    assert "from stdin" in capsys.readouterr().out


def test_cli_has_no_way_to_override_the_destination():
    help_text = cli.build_parser().format_help()
    notify_help = cli.build_parser().parse_args(["notify", "--dry-run", "x"])
    assert not hasattr(notify_help, "chat") and "--chat-id" not in help_text


# --- forum topic ----------------------------------------------------------------


def test_notify_topic_id_is_read_and_optional(tmp_path):
    assert config.load_notify_settings(_cfg(tmp_path, "notify:\n  chat_id: x\n")).topic_id is None
    s = config.load_notify_settings(_cfg(tmp_path, "notify:\n  enabled: true\n  chat_id: x\n  topic_id: 273\n"))
    assert s.topic_id == 273


@pytest.mark.parametrize("bad", ["abc", "true", "1.5"])
def test_notify_topic_id_must_be_an_integer(tmp_path, bad):
    with pytest.raises(config.ConfigError):
        config.load_notify_settings(_cfg(tmp_path, f"notify:\n  chat_id: x\n  topic_id: {bad}\n"))


def test_project_config_targets_the_sisyphus_topic():
    assert config.load_notify_settings().topic_id == 273


async def test_deliver_puts_every_part_into_the_topic():
    session = FakeSession()
    await notifier.deliver(session, CHAT, "a\nb\nc", limit=1, topic_id=273)
    assert [c[1]["reply_to_id"] for c in session.calls] == [273, 273, 273]


async def test_deliver_reply_inside_a_topic_replies_first_part_then_stays_in_topic():
    session = FakeSession()
    await notifier.deliver(session, CHAT, "a\nb\nc", reply_to=300, limit=1, topic_id=273)
    assert [c[1]["reply_to_id"] for c in session.calls] == [300, 273, 273]


async def test_send_notification_passes_the_configured_topic(monkeypatch):
    session = FakeSession()

    class _Ctx:
        async def __aenter__(self):
            return session

        async def __aexit__(self, *exc):
            return False

    monkeypatch.setattr(notifier.mcp_client, "open_session", lambda verbose=False: _Ctx())
    settings = config.NotifySettings(enabled=True, chat_id=CHAT, reply_to=None, topic_id=273)
    await notifier.send_notification("hi", settings=settings)
    assert session.calls == [("send_message", {"chat_id": CHAT, "message": "hi", "reply_to_id": 273})]
