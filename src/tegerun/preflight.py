"""First-run diagnosis: what is still missing before `tege run` can work, and how to fix it.

`check()` only looks (files, YAML, the Hugging Face cache on disk); it never runs a setup step, downloads
anything or imports a model library. It collects *every* problem, so one run shows the whole to-do list.
All locations are arguments, so tests can point it at a temp directory.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass
from pathlib import Path

from tegerun import classify, config, runner

# The values shipped in .config.yaml / .env.example. A test keeps them in sync with those files.
EXAMPLE_CHAT_IDS = frozenset({"1234567890", "@example_chat"})
EXAMPLE_NOTIFY_CHAT_ID = "-1001234567890"
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".gguf", ".onnx")  # what `fetch.py --code-only` leaves out


@dataclass(frozen=True)
class Finding:
    level: str  # "error": `run` cannot work yet; "warning": it works, but results will be poor
    what: str  # one line: what is wrong
    fix: str  # what to do, commands included (may span lines)


def hf_hub_cache(env: dict[str, str] | None = None, home: Path | None = None) -> Path:
    """Where Hugging Face keeps downloaded models, resolved the way huggingface_hub does it."""
    env = os.environ if env is None else env
    explicit = env.get("HF_HUB_CACHE") or env.get("HUGGINGFACE_HUB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    hf_home = env.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser() / "hub"
    base = env.get("XDG_CACHE_HOME") or str((home or Path.home()) / ".cache")
    return Path(base).expanduser() / "huggingface" / "hub"


def weights_present(model: classify.ModelSpec, hub_cache: Path) -> bool:
    """True if the pinned revision is in the cache *with weights* (a code-only fetch has none)."""
    snapshot = hub_cache / f"models--{model.repo.replace('/', '--')}" / "snapshots" / str(model.revision)
    if not snapshot.is_dir():
        return False
    # exists() follows the symlink into blobs/, so a half-finished download does not count.
    return any(p.suffix in WEIGHT_SUFFIXES and p.exists() for p in snapshot.rglob("*"))


def _rel(path: Path, root: Path) -> str:
    try:
        return f"./{path.relative_to(root)}"
    except ValueError:
        return str(path)


def check(
    *,
    root: Path = config.PROJECT_ROOT,
    env_file: Path = config.ENV_FILE,
    session: Path | None = None,
    config_file: Path = config.CONFIG_FILE,
    hub_cache: Path | None = None,
    model_name: str | None = None,
    send: bool | None = None,
    online: bool = False,
) -> list[Finding]:
    """Everything that stands between a fresh checkout and a working `tege run`, in the order to fix it."""
    session = session or config.session_file()
    hub_cache = hub_cache or hf_hub_cache()
    findings: list[Finding] = []

    def error(what: str, fix: str) -> None:
        findings.append(Finding("error", what, fix))

    def warn(what: str, fix: str) -> None:
        findings.append(Finding("warning", what, fix))

    # 1. Telegram credentials
    try:
        creds = config.load_credentials(env_file)
    except config.ConfigError as exc:
        if env_file.exists():
            error(str(exc), f"Fill in tege_api_id and tege_api_hash in {_rel(env_file, root)} (get them at https://my.telegram.org).")
        else:
            error(
                "No Telegram API credentials yet (.env is missing).",
                f"cp {shlex.quote(_rel(env_file.with_name('.env.example'), root))} {shlex.quote(_rel(env_file, root))}\n"
                "then put your api_id and api_hash in it (get them at https://my.telegram.org).",
            )
    else:
        if creds.api_id.upper().startswith("YOUR_") or creds.api_hash.upper().startswith("YOUR_"):
            error(
                f"{_rel(env_file, root)} still has the placeholder values.",
                "Put your real api_id and api_hash in it (get them at https://my.telegram.org).",
            )

    # 2. Telegram login
    if not session.exists():
        error("Not logged in to Telegram yet.", "uv run tege setup   # interactive: phone number, login code, 2FA")

    # 3. Chats to read
    try:
        chats = config.load_whitelist(config_file)
    except config.ConfigError as exc:
        error(str(exc), f"Fix `whitelist` in {_rel(config_file, root)}.")
    else:
        examples = [c.id for c in chats if c.id in EXAMPLE_CHAT_IDS]
        if examples and len(examples) == len(chats):
            error(
                "`whitelist` still holds only the example chats.",
                f"In {_rel(config_file, root)} replace them with chats you read (numeric id or @username).",
            )
        elif examples:
            warn(
                f"`whitelist` still has example chat(s): {', '.join(examples)}.",
                f"Remove them from {_rel(config_file, root)}; reading them only produces errors.",
            )

    # 4. Classifier: questions, thresholds, model environment, model weights
    try:
        settings = config.load_classify_settings(config_file)
        models = classify.load_models(settings.models_file)
        name = model_name or settings.model
        if name not in models:
            raise config.ConfigError(f"Unknown model {name!r}; available: {', '.join(models)}.")
        model = models[name]
        qset = classify.load_questions(settings.questions_file)
        run = config.load_run_settings(config_file)
        runner.validate_thresholds(run.thresholds, qset)
    except config.ConfigError as exc:
        error(f"Configuration problem: {exc}", f"Fix it in {_rel(config_file, root)} or the classifier/ files.")
        return findings + _notify_findings(config_file, root, send)

    setup_sh = shlex.quote(_rel(settings.classifier_dir / "setup.sh", root))
    venv_ok = (settings.classifier_dir / model.venv / "bin" / "python").exists()
    weights_ok = online or weights_present(model, hub_cache)  # --online lets the worker download them itself
    if not weights_ok:
        error(
            f"Model {name} is not downloaded" + ("" if venv_ok else " and its environment is not set up") + ".",
            f"{setup_sh} --download {name}   # several GB, one time",
        )
    elif not venv_ok:
        error(f"The environment for model {name} ({model.venv}) is not set up.", f"{setup_sh}   # one time")

    # The `match` questions are scored against `skills:`; the shipped text is only an example.
    needs_skills = [q for q in run.thresholds if qset.questions[q]["needs_skills"]]
    if needs_skills and qset.skills.lower().startswith("example"):
        warn(
            f"`skills:` in {_rel(settings.questions_file, root)} is still the example profile, "
            f"but {', '.join(needs_skills)} are scored against it.",
            "Rewrite it to describe what you are looking for (roles, places, what you do not want).",
        )

    return findings + _notify_findings(config_file, root, run.send if send is None else send)


def _notify_findings(config_file: Path, root: Path, send: bool | None) -> list[Finding]:
    """Sending needs a destination. Only matters when it is going to happen."""
    out: list[Finding] = []
    if not send:
        return out
    try:
        notify = config.load_notify_settings(config_file)
    except config.ConfigError as exc:
        return [Finding("error", str(exc), f"Fix `notify` in {_rel(config_file, root)}.")]
    if not notify.enabled:
        out.append(Finding("error", "Sending is on but notify.enabled is false.",
                           f"Set notify.enabled: true in {_rel(config_file, root)}, or run without --send."))
    elif notify.chat_id == EXAMPLE_NOTIFY_CHAT_ID:
        out.append(Finding("error", "notify.chat_id is still the example value.",
                           f"Set it to the chat that should receive matches in {_rel(config_file, root)}."))
    return out


def errors(findings: list[Finding]) -> list[Finding]:
    return [f for f in findings if f.level == "error"]


def _block(findings: list[Finding], numbered: bool) -> list[str]:
    lines: list[str] = []
    for i, f in enumerate(findings, 1):
        lines.append(f"\n  {f'{i}.' if numbered else '-'} {f.what}")
        lines.extend(f"     {line}" for line in f.fix.splitlines())
    return lines


def render(findings: list[Finding]) -> str:
    """The to-do list (errors, numbered, in the order to do them) and the recommendations."""
    problems = errors(findings)
    advice = [f for f in findings if f.level == "warning"]
    lines: list[str] = []
    if problems:
        lines.append(f"Setup is not finished: {len(problems)} thing(s) to do first.")
        lines += _block(problems, numbered=True)
        lines.append("\nThen run `uv run tege run` again (`uv run tege doctor` re-checks without running anything).")
    if advice:
        lines.append(("\n" if problems else "") + "Recommendations for better results:")
        lines += _block(advice, numbered=False)
    return "\n".join(lines)
