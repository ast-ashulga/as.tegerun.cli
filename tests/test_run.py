import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from tegerun import classify, cli, config, notifier, reader, runner, runstate

CHAT = "-1003842301085"
NOW = datetime(2026, 11, 4, 12, 0, tzinfo=timezone.utc)


def _cfg(tmp_path, body: str):
    path = tmp_path / "c.yaml"
    path.write_text(body)
    return path


def _run_settings(**over):
    base = dict(
        every=timedelta(minutes=30),
        first_window=timedelta(hours=24),
        limit=50,
        send=True,
        max_per_run=10,
        thresholds={"is_vacancy": 0.8, "match": 3, "match_ok": 0.7},
    )
    base.update(over)
    return config.RunSettings(**base)


# --- config: durations --------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("30m", timedelta(minutes=30)),
        ("1h", timedelta(hours=1)),
        ("3h", timedelta(hours=3)),
        ("90m", timedelta(minutes=90)),
        ("2d", timedelta(days=2)),
    ],
)
def test_parse_duration(text, expected):
    assert config.parse_duration(text, "run.every") == expected


@pytest.mark.parametrize("bad", ["soon", "0m", "-5m", "1.5h", "", "h", "30"])
def test_parse_duration_rejects_garbage_and_zero(bad):
    with pytest.raises(config.ConfigError):
        config.parse_duration(bad, "run.every")


# --- config: run section ------------------------------------------------------


def test_run_settings_defaults_when_section_missing(tmp_path):
    s = config.load_run_settings(_cfg(tmp_path, "whitelist: []\n"))
    assert s.every == timedelta(minutes=30)
    assert s.first_window == timedelta(hours=24)
    assert s.limit == 50 and s.max_per_run == 10
    assert s.send is False  # feature flags are off by default
    assert s.thresholds == {}


def test_run_settings_read_all_fields(tmp_path):
    body = (
        "run:\n  every: 3h\n  first_window: 2d\n  limit: 100\n  send: true\n  max_per_run: 5\n"
        "  thresholds:\n    is_vacancy: 0.8\n    match: 3\n"
    )
    s = config.load_run_settings(_cfg(tmp_path, body))
    assert s.every == timedelta(hours=3) and s.first_window == timedelta(days=2)
    assert (s.limit, s.send, s.max_per_run) == (100, True, 5)
    assert s.thresholds == {"is_vacancy": 0.8, "match": 3}


@pytest.mark.parametrize(
    "body",
    [
        "run: [1]\n",
        "run:\n  every: soon\n",
        "run:\n  first_window: 0h\n",
        "run:\n  send: 'false'\n",  # a string 'false' is truthy: must be rejected
        "run:\n  send: 1\n",
        "run:\n  limit: 0\n",
        "run:\n  limit: true\n",
        "run:\n  max_per_run: 0\n",
        "run:\n  max_per_run: 2.5\n",
        "run:\n  thresholds: [1]\n",
        "run:\n  thresholds:\n    match: high\n",
        "run:\n  thresholds:\n    match: true\n",
    ],
)
def test_run_settings_reject_bad_values(tmp_path, body):
    with pytest.raises(config.ConfigError):
        config.load_run_settings(_cfg(tmp_path, body))


def test_project_config_has_a_run_section_with_thresholds():
    # `send` is the user's live switch, so it is not pinned here; the default (False) is
    # covered by test_run_settings_defaults_when_section_missing.
    s = config.load_run_settings()
    assert s.every == timedelta(minutes=30)
    assert s.thresholds  # a run without thresholds would send everything


# --- state --------------------------------------------------------------------


def test_state_starts_empty(tmp_path):
    st = runstate.RunState(tmp_path)
    assert st.last_success is None and st.is_sent("1:2") is False


def test_state_remembers_sent_messages_across_instances(tmp_path):
    runstate.RunState(tmp_path).mark_sent("1:2")
    again = runstate.RunState(tmp_path)
    assert again.is_sent("1:2") and not again.is_sent("1:3")


def test_state_remembers_last_success_across_instances(tmp_path):
    runstate.RunState(tmp_path).mark_success(NOW)
    assert runstate.RunState(tmp_path).last_success == NOW


def test_state_creates_its_directory_and_leaves_no_temp_files(tmp_path):
    target = tmp_path / "nested" / ".state"
    st = runstate.RunState(target)
    st.mark_sent("1:2")
    st.mark_success(NOW)
    assert sorted(p.name for p in target.iterdir()) == ["run.json", "sent.json"]


def test_state_refuses_a_corrupt_sent_file_rather_than_resending_everything(tmp_path):
    (tmp_path / "sent.json").write_text("{not json")
    with pytest.raises(config.ConfigError):
        runstate.RunState(tmp_path)


def test_state_refuses_a_corrupt_run_file(tmp_path):
    (tmp_path / "run.json").write_text(json.dumps({"last_success": "yesterday-ish"}))
    with pytest.raises(config.ConfigError):
        runstate.RunState(tmp_path)


# --- thresholds ---------------------------------------------------------------


def _answers(is_vacancy=0.95, match=4.0, match_ok=0.9):
    return {
        "is_vacancy": {"type": "noul", "p": is_vacancy, "pred": is_vacancy > 0.5, "conf": 0.9},
        "category": {"type": "choice", "pred": "vacancy", "conf": 0.88, "probs": {}},
        "match": {"type": "score", "score": match, "pred": round(match), "conf": 0.5, "probs": {}},
        "match_ok": {"type": "noul", "p": match_ok, "pred": match_ok > 0.5, "conf": 0.9},
    }


def _item(msg_id=1, text="Senior TPM, remote", chat_id="c1", chat="Jobs", link="https://t.me/c/1/1"):
    return classify.Item(
        id=f"{chat_id}:{msg_id}", text=text, chat_id=chat_id, chat=chat, msg_id=msg_id,
        date="2026-11-04T11:00:00", link=link,
    )


def _verdict(item=None, error=None, **answers):
    return classify.Verdict(item or _item(), {} if error else _answers(**answers), error)


QSET = classify.QuestionSet(
    skills="s",
    tpl="{message}",
    tpl_skills="{message}{skills}",
    questions={
        "is_vacancy": {"type": "noul"},
        "category": {"type": "choice"},
        "match": {"type": "score"},
        "match_ok": {"type": "noul"},
    },
)


def test_validate_thresholds_accepts_known_noul_and_score_questions():
    runner.validate_thresholds({"is_vacancy": 0.8, "match": 3}, QSET)


def test_validate_thresholds_refuses_an_empty_set():
    with pytest.raises(config.ConfigError, match="thresholds"):
        runner.validate_thresholds({}, QSET)


def test_validate_thresholds_refuses_unknown_question():
    with pytest.raises(config.ConfigError, match="nope"):
        runner.validate_thresholds({"nope": 1}, QSET)


def test_validate_thresholds_refuses_choice_questions():
    with pytest.raises(config.ConfigError, match="category"):
        runner.validate_thresholds({"category": 0.5}, QSET)


T = {"is_vacancy": 0.8, "match": 3, "match_ok": 0.7}


def test_passes_when_every_threshold_is_met():
    assert runner.passes_thresholds(_verdict(), T)


def test_fails_when_any_single_threshold_is_missed():
    assert not runner.passes_thresholds(_verdict(is_vacancy=0.5), T)
    assert not runner.passes_thresholds(_verdict(match=2.9), T)
    assert not runner.passes_thresholds(_verdict(match_ok=0.69), T)


def test_a_value_equal_to_the_threshold_passes():
    assert runner.passes_thresholds(_verdict(is_vacancy=0.8, match=3.0, match_ok=0.7), T)


def test_questions_without_a_threshold_are_not_checked():
    assert runner.passes_thresholds(_verdict(is_vacancy=0.1, match_ok=0.0), {"match": 3})


def test_errored_or_unanswered_verdicts_never_pass():
    assert not runner.passes_thresholds(_verdict(error="boom"), T)
    v = _verdict()
    del v.answers["match"]
    assert not runner.passes_thresholds(v, T)


# --- message text ---------------------------------------------------------------


def test_candidate_text_has_chat_text_answers_and_link():
    text = runner.format_candidate(_verdict())
    assert "Jobs" in text and "Senior TPM, remote" in text
    assert "is_vacancy P=0.95" in text and "match 4.0" in text
    assert text.rstrip().endswith("https://t.me/c/1/1")


def test_candidate_text_is_cut_to_the_snippet_limit():
    long = "w" * 5000
    text = runner.format_candidate(_verdict(_item(text=long)))
    assert "w" * 599 + "…" in text and "w" * 600 not in text


def test_candidate_text_without_link_has_no_empty_line_at_the_end():
    text = runner.format_candidate(_verdict(_item(link=None)))
    assert text == text.rstrip() and "None" not in text


# --- run_pass -------------------------------------------------------------------


class Harness:
    """Fakes for the three boundaries of a pass: Telegram read, model, Telegram send."""

    def __init__(self, tmp_path, messages, answers_by_id=None, fail_send_at=None, chat_error=None):
        self.state = runstate.RunState(tmp_path / "state")
        self.messages = messages  # list of dicts (id, text, date, link)
        self.answers_by_id = answers_by_id or {}
        self.fail_send_at = fail_send_at
        self.chat_error = chat_error
        self.fetch_since = []
        self.classified = []
        self.sent = []
        self.chats = {"c1": config.Chat(id="c1", name="Jobs")}

    async def fetch(self, since):
        self.fetch_since.append(since)
        res = reader.ChatResult("c1", messages=self.messages, error=self.chat_error)
        return [res], self.chats

    def classify_items(self, items):
        self.classified.extend(i.msg_id for i in items)
        out = []
        for it in items:
            spec = self.answers_by_id.get(it.msg_id, {})
            out.append(classify.Verdict(it, {} if spec == "error" else _answers(**spec), "boom" if spec == "error" else None))
        return out

    async def send(self, text):
        if self.fail_send_at is not None and len(self.sent) == self.fail_send_at:
            return notifier.NotifyResult(sent=False, error="no access")
        self.sent.append(text)
        return notifier.NotifyResult(sent=True, message_ids=[100 + len(self.sent)])

    async def run(self, settings=None, now=NOW):
        return await runner.run_pass(
            settings or _run_settings(), self.state,
            fetch=self.fetch, classify_items=self.classify_items, send=self.send,
            now=now, log=lambda m: None,
        )


def _msgs(*ids):
    return [{"id": i, "text": f"vacancy {i}", "date": "2026-11-04T11:00:00", "link": f"https://t.me/c/1/{i}"} for i in ids]


async def test_pass_sends_each_match_as_its_own_message_and_records_them(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3), {2: {"match": 1.0}})  # message 2 is weak
    report = await h.run()
    assert len(h.sent) == 2 and "vacancy 1" in h.sent[0] and "vacancy 3" in h.sent[1]
    assert h.state.is_sent("c1:1") and h.state.is_sent("c1:3") and not h.state.is_sent("c1:2")
    assert (report.read, report.matched, report.sent, report.ok) == (3, 2, 2, True)
    assert h.state.last_success == NOW


async def test_pass_with_send_off_sends_and_records_nothing_but_reports_what_would_go(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2))
    report = await h.run(_run_settings(send=False))
    assert h.sent == [] and not h.state.is_sent("c1:1")
    assert len(report.would_send) == 2 and "vacancy 1" in report.would_send[0]
    assert report.sent == 0 and report.ok is True
    assert h.state.last_success is None  # a preview must not consume the matches it shows


async def test_already_sent_messages_are_neither_classified_nor_sent_again(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2))
    h.state.mark_sent("c1:1")
    await h.run()
    assert h.classified == [2] and len(h.sent) == 1 and "vacancy 2" in h.sent[0]


async def test_first_pass_reads_first_window_and_later_passes_read_from_last_success(tmp_path):
    h = Harness(tmp_path, [])
    await h.run(_run_settings(first_window=timedelta(hours=24)))
    assert h.fetch_since[0] == (NOW - timedelta(hours=24)).isoformat()
    later = NOW + timedelta(minutes=30)
    await h.run(now=later)
    assert h.fetch_since[1] == NOW.isoformat()


async def test_pass_with_nothing_to_read_skips_the_model_and_moves_the_window(tmp_path):
    h = Harness(tmp_path, [])
    report = await h.run()
    assert h.classified == [] and report.ok and h.state.last_success == NOW


async def test_send_failure_stops_the_pass_keeps_what_went_out_and_holds_the_window(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3), fail_send_at=1)
    report = await h.run()
    assert len(h.sent) == 1 and h.state.is_sent("c1:1")
    assert not h.state.is_sent("c1:2") and not h.state.is_sent("c1:3")
    assert report.ok is False and h.state.last_success is None
    assert any("no access" in e for e in report.errors)


async def test_classification_error_holds_the_window_but_other_matches_still_go(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2), {1: "error"})
    report = await h.run()
    assert len(h.sent) == 1 and "vacancy 2" in h.sent[0]
    assert report.ok is False and h.state.last_success is None


async def test_chat_read_error_holds_the_window(tmp_path):
    h = Harness(tmp_path, [], chat_error="FLOOD_WAIT")
    report = await h.run()
    assert report.ok is False and h.state.last_success is None
    assert any("FLOOD_WAIT" in e for e in report.errors)


async def test_more_matches_than_max_per_run_are_spread_over_passes(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3))
    first = await h.run(_run_settings(max_per_run=2))
    assert len(h.sent) == 2 and first.ok is False and h.state.last_success is None
    second = await h.run(_run_settings(max_per_run=2))
    assert len(h.sent) == 3 and "vacancy 3" in h.sent[2]  # nothing sent twice
    assert second.ok is True and h.state.last_success == NOW


async def test_messages_without_an_id_are_never_sent(tmp_path):
    msgs = [{"id": None, "text": "no id", "date": "", "link": None}] + _msgs(2)
    h = Harness(tmp_path, msgs)
    await h.run()
    assert len(h.sent) == 1 and "vacancy 2" in h.sent[0]


# --- the loop -------------------------------------------------------------------


def test_loop_runs_passes_with_the_interval_between_them():
    calls, sleeps = [], []
    runner.run_loop(lambda: calls.append(1), timedelta(minutes=30), sleep=sleeps.append, max_passes=3, log=lambda m: None)
    assert len(calls) == 3 and sleeps == [1800, 1800]  # no sleep after the last pass


def test_loop_survives_a_failing_pass():
    seen, logged = [], []

    def flaky():
        seen.append(1)
        if len(seen) == 1:
            raise RuntimeError("model crashed")

    runner.run_loop(flaky, timedelta(hours=1), sleep=lambda s: None, max_passes=2, log=logged.append)
    assert len(seen) == 2 and any("model crashed" in m for m in logged)


def test_loop_stops_quietly_on_ctrl_c():
    def stop():
        raise KeyboardInterrupt

    runner.run_loop(stop, timedelta(hours=1), sleep=lambda s: None, log=lambda m: None)


# --- CLI ------------------------------------------------------------------------


@pytest.fixture
def cli_env(monkeypatch, tmp_path):
    """Everything `tege run` loads, replaced with in-memory values; returns the recorder."""
    rec = SimpleNamespace(passes=0, loop_every=None, report=runner.PassReport(ok=True))
    monkeypatch.setattr(config, "STATE_DIR", tmp_path / "state")
    monkeypatch.setattr(config, "load_whitelist", lambda *a, **k: [config.Chat("c1", "Jobs")])
    monkeypatch.setattr(config, "load_classify_settings", lambda *a, **k: SimpleNamespace(
        questions_file=None, models_file=None, model="m", classifier_dir=tmp_path))
    monkeypatch.setattr(classify, "load_questions", lambda p: QSET)
    monkeypatch.setattr(classify, "load_models", lambda p: {"m": SimpleNamespace(grouping="joint")})
    monkeypatch.setattr(config, "load_notify_settings", lambda *a, **k: config.NotifySettings(True, CHAT, None))
    monkeypatch.setattr(config, "load_run_settings", lambda *a, **k: _run_settings(send=False))

    async def fake_pass(*a, **k):
        rec.passes += 1
        return rec.report

    def fake_loop(pass_fn, every, **k):
        rec.loop_every = every

    monkeypatch.setattr(runner, "run_pass", fake_pass)
    monkeypatch.setattr(runner, "run_loop", fake_loop)
    return rec


def test_cli_run_once_does_a_single_pass_and_exits_zero(cli_env):
    assert cli.main(["run", "--once"]) == 0
    assert cli_env.passes == 1 and cli_env.loop_every is None


def test_cli_run_once_exits_one_when_the_pass_had_errors(cli_env):
    cli_env.report = runner.PassReport(ok=False, errors=["x"])
    assert cli.main(["run", "--once"]) == 1


def test_cli_run_without_once_hands_over_to_the_loop_with_the_configured_interval(cli_env):
    assert cli.main(["run"]) == 0
    assert cli_env.loop_every == timedelta(minutes=30) and cli_env.passes == 0


def test_cli_every_overrides_the_interval(cli_env):
    cli.main(["run", "--every", "2h"])
    assert cli_env.loop_every == timedelta(hours=2)


def test_cli_send_flag_with_notifications_disabled_is_an_error(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(config, "load_notify_settings", lambda *a, **k: config.NotifySettings(False, CHAT, None))
    assert cli.main(["run", "--once", "--send"]) == 2
    assert "notify.enabled" in capsys.readouterr().err and cli_env.passes == 0


def test_cli_config_send_true_with_notifications_disabled_is_an_error(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(config, "load_run_settings", lambda *a, **k: _run_settings(send=True))
    monkeypatch.setattr(config, "load_notify_settings", lambda *a, **k: config.NotifySettings(False, CHAT, None))
    assert cli.main(["run", "--once"]) == 2


def test_cli_no_send_overrides_a_config_that_sends(cli_env, monkeypatch):
    seen = {}

    async def spy_pass(run, *a, **k):
        seen["send"] = run.send
        return runner.PassReport(ok=True)

    monkeypatch.setattr(config, "load_run_settings", lambda *a, **k: _run_settings(send=True))
    monkeypatch.setattr(runner, "run_pass", spy_pass)
    assert cli.main(["run", "--once", "--no-send"]) == 0
    assert seen["send"] is False


def test_cli_refuses_to_start_without_thresholds(cli_env, monkeypatch, capsys):
    monkeypatch.setattr(config, "load_run_settings", lambda *a, **k: _run_settings(thresholds={}))
    assert cli.main(["run", "--once"]) == 2
    assert "thresholds" in capsys.readouterr().err and cli_env.passes == 0


def test_cli_prints_what_would_be_sent_when_send_is_off(cli_env, capsys):
    cli_env.report = runner.PassReport(ok=True, would_send=["hello vacancy"])
    cli.main(["run", "--once"])
    assert "hello vacancy" in capsys.readouterr().out


def test_cli_has_no_way_to_override_the_destination_for_run():
    args = cli.build_parser().parse_args(["run", "--once"])
    assert not hasattr(args, "chat_id") and "--chat-id" not in cli.build_parser().format_help()


# --- hardening: overflow, poison messages, send failures, state -----------------


async def test_unread_overflow_holds_the_window_instead_of_losing_messages(tmp_path):
    h = Harness(tmp_path, _msgs(1))

    async def overflowing_fetch(since):
        return [reader.ChatResult("c1", messages=_msgs(1), has_more=True)], h.chats

    h.fetch = overflowing_fetch
    report = await h.run()
    assert report.ok is False and h.state.last_success is None
    assert any("more messages than could be read" in e for e in report.errors)


async def test_a_message_that_keeps_failing_is_skipped_after_max_attempts(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2), {1: "error"})
    settings = _run_settings(max_attempts=2)
    for _ in range(2):
        assert (await h.run(settings)).ok is False
    assert h.state.failures("c1:1") == 2
    h.classified.clear()
    final = await h.run(settings)
    assert 1 not in h.classified  # not re-run through the model any more
    assert final.ok is True and h.state.last_success == NOW  # and it no longer blocks the window


async def test_when_every_message_fails_at_once_nobody_is_blamed(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2), {1: "error", 2: "error"})  # looks like a broken model, not bad messages
    await h.run()
    assert h.state.failures("c1:1") == 0 and h.state.failures("c1:2") == 0


async def test_failures_are_not_counted_in_a_preview(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2), {1: "error"})
    await h.run(_run_settings(send=False))
    assert h.state.failures("c1:1") == 0


async def test_a_message_telegram_rejects_does_not_block_the_ones_after_it(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3))
    real_send = h.send

    async def picky(text):
        if "vacancy 2" in text:
            return notifier.NotifyResult(sent=False, error="MESSAGE_INVALID")
        return await real_send(text)

    h.send = picky
    report = await h.run()
    assert len(h.sent) == 2 and h.state.is_sent("c1:3")  # 3 went out although 2 failed
    assert h.state.failures("c1:2") == 1  # others got through, so the blame goes to message 2
    assert report.ok is False


async def test_two_failed_sends_in_a_row_stop_the_pass_without_blaming_the_messages(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3, 4), fail_send_at=0)  # Telegram unreachable
    report = await h.run()
    assert h.sent == [] and len([e for e in report.errors if e.startswith("send")]) == 2
    assert all(h.state.failures(f"c1:{i}") == 0 for i in (1, 2, 3, 4))


def test_validate_thresholds_rejects_values_that_can_never_be_reached():
    qset = classify.QuestionSet("s", "{message}", "{message}", {
        "p": {"type": "noul"},
        "lvl": {"type": "score", "criteria": ["a", "b", "c"]},
    })
    runner.validate_thresholds({"p": 1, "lvl": 2}, qset)  # the maxima themselves are fine
    with pytest.raises(config.ConfigError, match="0..1"):
        runner.validate_thresholds({"p": 5}, qset)
    with pytest.raises(config.ConfigError, match="0..2"):
        runner.validate_thresholds({"lvl": 3}, qset)


def test_state_counts_failures_across_instances_and_refuses_a_corrupt_file(tmp_path):
    st = runstate.RunState(tmp_path)
    assert st.failures("1:2") == 0 and st.record_failure("1:2") == 1 and st.record_failure("1:2") == 2
    assert runstate.RunState(tmp_path).failures("1:2") == 2
    (tmp_path / "failed.json").write_text('{"failures": {"1:2": "x"}}')
    with pytest.raises(config.ConfigError):
        runstate.RunState(tmp_path)


def test_run_lock_is_exclusive_and_released_afterwards(tmp_path):
    with runstate.RunLock(tmp_path):
        with pytest.raises(config.ConfigError, match="already active"):
            with runstate.RunLock(tmp_path):
                pass
    with runstate.RunLock(tmp_path):  # free again
        pass


@pytest.mark.parametrize("body", ["run:\n  max_attempts: 0\n", "run:\n  max_attempts: true\n", "classify:\n  timeout: 0\n"])
def test_new_settings_are_validated(tmp_path, body):
    cfg = _cfg(tmp_path, body)
    with pytest.raises(config.ConfigError):
        config.load_run_settings(cfg) if body.startswith("run") else config.load_classify_settings(cfg)


def test_cli_run_refuses_to_start_while_another_run_holds_the_lock(cli_env, capsys):
    with runstate.RunLock():
        assert cli.main(["run", "--once"]) == 2
    assert "already active" in capsys.readouterr().err and cli_env.passes == 0


def test_cli_run_sends_escaped_html_through_one_sender_and_closes_it(cli_env, monkeypatch):
    log = SimpleNamespace(parse_mode=None, sent=[], closed=0)

    class FakeSender:
        def __init__(self, settings, *, verbose=False, parse_mode=None):
            log.parse_mode = parse_mode

        async def send(self, text, reply_to=None):
            log.sent.append(text)
            return notifier.NotifyResult(sent=True, message_ids=[1])

        async def aclose(self):
            log.closed += 1

    async def pass_that_sends(run, state, *, fetch, classify_items, send, **k):
        await send("<b>x</b> & <a href='http://evil'>click</a>")
        await send("second")
        return runner.PassReport(ok=True)

    monkeypatch.setattr(notifier, "Sender", FakeSender)
    monkeypatch.setattr(runner, "run_pass", pass_that_sends)
    assert cli.main(["run", "--once"]) == 0
    assert log.parse_mode == "html" and log.closed == 1
    assert log.sent[0] == "&lt;b&gt;x&lt;/b&gt; &amp; &lt;a href='http://evil'&gt;click&lt;/a&gt;"


async def test_messages_the_worker_never_reached_are_not_blamed(tmp_path):
    h = Harness(tmp_path, _msgs(1, 2, 3))

    def killed_worker(items):  # message 1 classified, then the worker died: 2 and 3 got no result
        h.classified.extend(i.msg_id for i in items)
        done = classify.Verdict(items[0], _answers())
        lost = [classify.Verdict(i, {}, "no result from the model", transient=True) for i in items[1:]]
        return [done] + lost

    h.classify_items = killed_worker
    for _ in range(3):
        await h.run(_run_settings(max_attempts=1))
    assert h.state.failures("c1:2") == 0 and h.state.failures("c1:3") == 0  # never skipped for the worker's death
