from datetime import timedelta
from pathlib import Path

import pytest

from tegerun import cli, config, preflight, runner

REV = "a" * 40
CONFIG = """\
whitelist:
  - {id: "42", name: Real chat}
notify: {enabled: true, chat_id: "-100777"}
run:
  send: false
  thresholds: {is_vacancy: 0.8, match: 3}
"""
QUESTIONS = """\
skills: |
  Senior backend engineer, remote only.
questions:
  is_vacancy: {type: noul, instructions: "Is it a vacancy?"}
  match: {type: score, needs_skills: true, criteria: [no, maybe, yes, great]}
"""
MODELS = f"""\
models:
  m1: {{runner: d1_worker, venv: .venv-x, repo: org/model-one, revision: {REV}}}
"""


def _write(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def stand(tmp_path):
    """A complete, working setup in a temp dir; each test breaks one thing."""
    root = tmp_path
    _write(root / ".env", "tege_api_id=123\ntege_api_hash=abcdef\n")
    _write(root / ".sessions" / "telegram.session")
    _write(root / ".config.yaml", CONFIG.replace("}\nrun", "}\nclassify: {model: m1}\nrun"))
    _write(root / "classifier" / "questions.yaml", QUESTIONS)
    _write(root / "classifier" / "models.yaml", MODELS)
    _write(root / "classifier" / ".venv-x" / "bin" / "python")
    hub = root / "hub"
    snap = hub / "models--org--model-one" / "snapshots" / REV
    blob = _write(hub / "models--org--model-one" / "blobs" / "w", "weights")
    snap.mkdir(parents=True)
    (snap / "model.safetensors").symlink_to(blob)
    _write(snap / "config.json", "{}")

    def check(**over):
        kw = dict(
            root=root, env_file=root / ".env", session=root / ".sessions" / "telegram.session",
            config_file=root / ".config.yaml", hub_cache=hub, model_name="m1",
        )
        kw.update(over)
        # classifier_dir comes from the config file, which is relative to config.PROJECT_ROOT: point it here
        return preflight.check(**kw)

    check.root, check.hub = root, hub
    return check


@pytest.fixture(autouse=True)
def project_root(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(config, "DEFAULT_CLASSIFIER_DIR", tmp_path / "classifier")


def whats(findings, level="error"):
    return [f.what for f in findings if f.level == level]


def test_a_complete_setup_has_nothing_to_report(stand):
    assert stand() == []


def test_missing_env_says_how_to_create_it(stand):
    (stand.root / ".env").unlink()
    (stand.root / ".env.example").write_text("tege_api_id=YOUR_API_ID\n")
    [f] = [f for f in stand() if f.level == "error"]
    assert "cp ./.env.example ./.env" in f.fix and "my.telegram.org" in f.fix


def test_placeholder_credentials_are_an_error(stand):
    (stand.root / ".env").write_text("tege_api_id=YOUR_API_ID\ntege_api_hash=YOUR_API_HASH\n")
    assert any("placeholder" in w for w in whats(stand()))


def test_missing_login_points_at_setup(stand):
    (stand.root / ".sessions" / "telegram.session").unlink()
    [f] = stand()
    assert f.level == "error" and "uv run tege setup" in f.fix


def test_example_whitelist_is_an_error_but_a_mix_is_a_warning(stand):
    cfg = stand.root / ".config.yaml"
    cfg.write_text(cfg.read_text().replace("{id: \"42\", name: Real chat}", "{id: \"1234567890\", name: Example}"))
    assert any("example chats" in w for w in whats(stand()))
    cfg.write_text(cfg.read_text().replace("  - {id: \"1234567890\", name: Example}",
                                           "  - {id: \"1234567890\", name: Example}\n  - {id: \"42\", name: Real}"))
    findings = stand()
    assert whats(findings) == [] and any("1234567890" in w for w in whats(findings, "warning"))


def test_missing_weights_and_environment_give_one_download_command(stand):
    for p in (stand.root / "classifier" / ".venv-x" / "bin" / "python",):
        p.unlink()
    for p in (stand.hub / "models--org--model-one" / "snapshots" / REV).iterdir():
        p.unlink()
    [f] = stand()
    assert "not downloaded" in f.what and "environment is not set up" in f.what
    assert "setup.sh --download m1" in f.fix


def test_missing_environment_alone_points_at_plain_setup(stand):
    (stand.root / "classifier" / ".venv-x" / "bin" / "python").unlink()
    [f] = stand()
    assert f.fix.startswith("./classifier/setup.sh") and "--download" not in f.fix


def test_code_only_snapshot_does_not_count_as_downloaded(stand):
    (stand.hub / "models--org--model-one" / "snapshots" / REV / "model.safetensors").unlink()
    assert any("not downloaded" in w for w in whats(stand()))


def test_dangling_weight_symlink_does_not_count(stand):
    (stand.hub / "models--org--model-one" / "blobs" / "w").unlink()
    assert any("not downloaded" in w for w in whats(stand()))


def test_online_mode_does_not_demand_the_weights(stand):
    (stand.hub / "models--org--model-one" / "snapshots" / REV / "model.safetensors").unlink()
    assert stand(online=True) == []


def test_unknown_model_and_bad_thresholds_are_reported_not_raised(stand):
    assert any("Unknown model" in w for w in whats(stand(model_name="nope")))
    cfg = stand.root / ".config.yaml"
    cfg.write_text(cfg.read_text().replace("match: 3}", "ghost: 3}"))
    assert any("ghost" in w for w in whats(stand()))


def test_every_problem_is_listed_at_once_in_the_order_to_fix_them(stand):
    (stand.root / ".env").unlink()
    (stand.root / ".sessions" / "telegram.session").unlink()
    (stand.root / "classifier" / ".venv-x" / "bin" / "python").unlink()
    order = [f.what for f in stand()]
    assert len(order) == 3
    assert "credentials" in order[0] and "logged in" in order[1] and "environment" in order[2]


def test_example_skills_are_only_a_recommendation(stand):
    q = stand.root / "classifier" / "questions.yaml"
    q.write_text(q.read_text().replace("Senior backend engineer", "Example: senior backend engineer"))
    findings = stand()
    assert whats(findings) == [] and any("example profile" in w for w in whats(findings, "warning"))


def test_example_skills_are_ignored_when_no_threshold_depends_on_them(stand):
    q = stand.root / "classifier" / "questions.yaml"
    q.write_text(q.read_text().replace("Senior backend engineer", "Example: x"))
    cfg = stand.root / ".config.yaml"
    cfg.write_text(cfg.read_text().replace("thresholds: {is_vacancy: 0.8, match: 3}", "thresholds: {is_vacancy: 0.8}"))
    assert stand() == []


def test_sending_needs_a_real_destination_but_a_preview_does_not(stand):
    cfg = stand.root / ".config.yaml"
    cfg.write_text(cfg.read_text().replace("chat_id: \"-100777\"", "chat_id: \"-1001234567890\""))
    assert stand() == []
    assert any("example value" in w for w in whats(stand(send=True)))
    cfg.write_text(cfg.read_text().replace("enabled: true", "enabled: false"))
    assert any("notify.enabled" in w for w in whats(stand(send=True)))


def test_shipped_example_values_are_the_ones_preflight_recognises():
    """.config.yaml and .env.example ship placeholders; preflight must keep recognising them."""
    shipped = Path(__file__).resolve().parent.parent
    ids = {c.id for c in config.load_whitelist(shipped / ".config.yaml")}
    assert ids <= preflight.EXAMPLE_CHAT_IDS
    assert config.load_notify_settings(shipped / ".config.yaml").chat_id == preflight.EXAMPLE_NOTIFY_CHAT_ID
    env = config.load_credentials(shipped / ".env.example")
    assert env.api_id.startswith("YOUR_") and env.api_hash.startswith("YOUR_")


@pytest.mark.parametrize(
    "env, expected",
    [
        ({"HF_HUB_CACHE": "/c"}, "/c"),
        ({"HF_HOME": "/h"}, "/h/hub"),
        ({"XDG_CACHE_HOME": "/x"}, "/x/huggingface/hub"),
        ({}, "/home/u/.cache/huggingface/hub"),
        ({"HF_HUB_CACHE": "/c", "HF_HOME": "/h"}, "/c"),
    ],
)
def test_hf_cache_is_resolved_like_huggingface_does(env, expected):
    assert preflight.hf_hub_cache(env, home=Path("/home/u")) == Path(expected)


def test_render_numbers_the_to_do_list_and_lists_recommendations_separately():
    text = preflight.render([
        preflight.Finding("error", "Not logged in.", "uv run tege setup"),
        preflight.Finding("warning", "skills is an example", "rewrite it"),
    ])
    assert "1. Not logged in." in text and "uv run tege setup" in text
    assert "Recommendations" in text and "- skills is an example" in text
    assert preflight.render([]) == ""


def test_doctor_reports_without_running_anything(monkeypatch, capsys):
    monkeypatch.setattr(preflight, "check", lambda **k: [preflight.Finding("error", "Not logged in.", "uv run tege setup")])
    assert cli.main(["doctor"]) == 2
    assert "uv run tege setup" in capsys.readouterr().err
    monkeypatch.setattr(preflight, "check", lambda **k: [])
    assert cli.main(["doctor"]) == 0
    assert "in place" in capsys.readouterr().err


# --- what to tell the user after a pass ---------------------------------------------


def _settings(**over):
    return config.RunSettings(
        every=timedelta(minutes=30), first_window=timedelta(hours=24), limit=50, send=False,
        max_per_run=10, thresholds={"is_vacancy": 0.8, "match": 3}, **over,
    )


def test_explain_says_how_far_the_best_message_was_from_the_thresholds():
    rep = runner.PassReport(ok=True, read=5, classified=5, matched=0, best={"is_vacancy": 0.62, "match": 2})
    [hint] = runner.explain(rep, _settings())
    assert "is_vacancy 0.62 (needs 0.80)" in hint and "match 2 (needs 3)" in hint


def test_explain_suggests_a_wider_window_when_nothing_was_read():
    [hint] = runner.explain(runner.PassReport(ok=True), _settings())
    assert "first_window" in hint


def test_explain_points_at_sending_after_a_preview_with_matches():
    rep = runner.PassReport(ok=True, read=3, classified=3, matched=1)
    [hint] = runner.explain(rep, _settings())
    assert "run.send: true" in hint
    assert runner.explain(rep, config.RunSettings(**{**_settings().__dict__, "send": True})) == []


def test_explain_stays_quiet_when_the_pass_had_errors():
    assert runner.explain(runner.PassReport(ok=False, errors=["x"]), _settings()) == []

