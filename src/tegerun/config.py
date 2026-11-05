"""Project-local configuration: credentials, whitelist, session location.

Everything that decides *where* the Telegram session lives and *what* the MCP
server subprocess sees is built here, so `tege setup` and `tege read` can never
disagree about it.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import yaml
from dotenv import dotenv_values

# Everything (config, credentials, session, state) lives next to the checkout. `TEGE_HOME` overrides
# that, e.g. for a non-editable install where the package no longer sits inside the repository.
if os.environ.get("TEGE_HOME"):
    PROJECT_ROOT = Path(os.environ["TEGE_HOME"]).expanduser().resolve()
else:
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # src/tegerun -> src -> repo root

ENV_FILE = PROJECT_ROOT / ".env"
CONFIG_FILE = PROJECT_ROOT / ".config.yaml"
SESSION_DIR = PROJECT_ROOT / ".sessions"
SESSION_NAME = "telegram"
LOG_DIR = PROJECT_ROOT / "logs"
STATE_DIR = PROJECT_ROOT / ".state"  # what `tegerun` remembers between passes (gitignored)


class ConfigError(Exception):
    """User-facing configuration problem (missing file, key, session...)."""


@dataclass(frozen=True)
class Credentials:
    api_id: str
    api_hash: str

    def __repr__(self) -> str:  # never leak secrets into logs / tracebacks
        return "Credentials(api_id=<hidden>, api_hash=<hidden>)"


def load_credentials(env_file: Path = ENV_FILE) -> Credentials:
    if not env_file.exists():
        raise ConfigError(f"{env_file} not found (expected tege_api_id / tege_api_hash).")
    values = dotenv_values(env_file)
    api_id = (values.get("tege_api_id") or "").strip()
    api_hash = (values.get("tege_api_hash") or "").strip()
    missing = [k for k, v in (("tege_api_id", api_id), ("tege_api_hash", api_hash)) if not v]
    if missing:
        raise ConfigError(f"Missing in {env_file.name}: {', '.join(missing)}")
    return Credentials(api_id=api_id, api_hash=api_hash)


@dataclass(frozen=True)
class Chat:
    id: str
    name: str
    description: str = ""


def _read_config(config_file: Path) -> object:
    if not config_file.exists():
        raise ConfigError(f"{config_file} not found.")
    try:
        return yaml.safe_load(config_file.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{config_file.name} is not valid YAML: {exc}") from exc


def _section(config_file: Path, name: str) -> dict:
    """The optional mapping `name:` of the config file ({} when absent)."""
    data = _read_config(config_file)
    section = data.get(name) if isinstance(data, dict) else None
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ConfigError(f"{config_file.name}: '{name}' must be a mapping.")
    return section


def _positive_int(config_file: Path, key: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ConfigError(f"{config_file.name}: {key} must be a positive integer (got {value!r}).")
    return value


def load_whitelist(config_file: Path = CONFIG_FILE) -> list[Chat]:
    """Chats from the `whitelist` list, de-duplicated by id, order kept.

    Each item is a mapping: `id` and `name` are required, `description` is optional.
    """
    data = _read_config(config_file)
    raw = data.get("whitelist") if isinstance(data, dict) else None
    if not isinstance(raw, list) or not raw:
        raise ConfigError(f"{config_file.name}: 'whitelist' must be a non-empty list.")
    chats: dict[str, Chat] = {}
    for i, item in enumerate(raw, 1):
        if not isinstance(item, dict):
            raise ConfigError(f"{config_file.name}: whitelist item #{i} must be a mapping (id, name, description).")
        fields = {k: ("" if item.get(k) is None else str(item[k]).strip()) for k in ("id", "name", "description")}
        for key in ("id", "name"):
            if not fields[key]:
                raise ConfigError(f"{config_file.name}: whitelist item #{i} is missing '{key}'.")
        chats.setdefault(fields["id"], Chat(**fields))
    return list(chats.values())


DEFAULT_CLASSIFIER_DIR = PROJECT_ROOT / "classifier"
DEFAULT_MODEL = "d1-3B"
DEFAULT_WORKER_TIMEOUT = 1800


@dataclass(frozen=True)
class ClassifySettings:
    """Where the classifier (questions, model registry, workers, venvs) lives and which model to use."""

    classifier_dir: Path
    model: str
    questions_file: Path
    models_file: Path
    timeout: int = DEFAULT_WORKER_TIMEOUT  # seconds one worker run may take (model load + all messages)


def load_classify_settings(config_file: Path = CONFIG_FILE) -> ClassifySettings:
    """Optional `classify:` section of .config.yaml.

    classifier_dir  classifier directory (default: ./classifier in this project)
    model      default model name from classifier/models.yaml (default: d1-3B)
    questions  questions file (default: <classifier_dir>/questions.yaml)
    models     model registry (default: <classifier_dir>/models.yaml)
    timeout    seconds a worker run may take before it is killed (default: 1800)
    Relative paths: `classifier_dir` is relative to this project, the others to `classifier_dir`.
    """
    section = _section(config_file, "classify")
    classifier_dir = Path(str(section.get("classifier_dir") or DEFAULT_CLASSIFIER_DIR)).expanduser()
    if not classifier_dir.is_absolute():
        classifier_dir = PROJECT_ROOT / classifier_dir
    if not classifier_dir.is_dir():
        raise ConfigError(
            f"Classifier directory {classifier_dir} not found (set classify.classifier_dir in {config_file.name})."
        )

    def _path(key: str, default: str) -> Path:
        path = Path(str(section.get(key) or default)).expanduser()
        return path if path.is_absolute() else classifier_dir / path

    return ClassifySettings(
        classifier_dir=classifier_dir,
        model=str(section.get("model") or DEFAULT_MODEL),
        questions_file=_path("questions", "questions.yaml"),
        models_file=_path("models", "models.yaml"),
        timeout=_positive_int(config_file, "classify.timeout", section.get("timeout", DEFAULT_WORKER_TIMEOUT)),
    )


@dataclass(frozen=True)
class NotifySettings:
    """Where `tege notify` / notifier.send_notification() delivers messages."""

    enabled: bool
    chat_id: str
    reply_to: int | None
    topic_id: int | None = None  # forum topic: every part of a message goes there


def load_notify_settings(config_file: Path = CONFIG_FILE) -> NotifySettings:
    """Optional `notify:` section of .config.yaml. Sending is OFF unless `enabled: true`.

    enabled   true/false (must be a real YAML boolean; default false)
    chat_id   destination chat, e.g. "-1003842301085" (required when enabled)
    reply_to  optional message id the notification replies to (first part only)
    topic_id  optional forum topic id (the number in t.me/c/<chat>/<topic>); all parts go there
    """
    section = _section(config_file, "notify")

    enabled = section.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ConfigError(f"{config_file.name}: notify.enabled must be true or false (got {enabled!r}).")
    chat_id = "" if section.get("chat_id") is None else str(section["chat_id"]).strip()
    if enabled and not chat_id:
        raise ConfigError(f"{config_file.name}: notify.chat_id is required when notify.enabled is true.")

    ids: dict[str, int | None] = {}
    for key, what in (("reply_to", "a message id"), ("topic_id", "a topic id")):
        value = section.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ConfigError(f"{config_file.name}: notify.{key} must be {what} (integer).")
        ids[key] = value
    return NotifySettings(enabled=enabled, chat_id=chat_id, reply_to=ids["reply_to"], topic_id=ids["topic_id"])


_DURATION = re.compile(r"^(\d{1,6})\s*([mhdw])$")
_DURATION_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def try_parse_duration(value: object) -> timedelta | None:
    """'30m' / '1h' / '2d' / '1w' -> timedelta; None when it is not a positive duration."""
    m = _DURATION.match(str(value).strip())
    delta = timedelta(**{_DURATION_UNITS[m.group(2)]: int(m.group(1))}) if m else None
    return delta or None  # a zero duration is as useless as a malformed one


def parse_duration(value: object, what: str) -> timedelta:
    """Like try_parse_duration, but a bad value is a ConfigError naming `what`."""
    delta = try_parse_duration(value)
    if delta is None:
        raise ConfigError(f"{what}: bad duration {value!r}; use e.g. 30m, 1h, 3h, 2d.")
    return delta


@dataclass(frozen=True)
class RunSettings:
    """How `tege run` repeats: interval, window, whether to send, and when a message is worth sending."""

    every: timedelta
    first_window: timedelta
    limit: int
    send: bool
    max_per_run: int
    thresholds: dict[str, float]
    max_attempts: int = 3  # a message that failed this many passes is skipped so it cannot block the window


def load_run_settings(config_file: Path = CONFIG_FILE) -> RunSettings:
    """Optional `run:` section of .config.yaml. Sending is OFF unless `send: true`.

    every         pause between passes in `tege run` (30m, 1h, 3h...; default 30m)
    first_window  how far back the very first pass reads (default 24h)
    limit         messages per chat per pass (default 50)
    send          true/false: deliver matches to Telegram (must be a real boolean; default false)
    max_per_run   at most this many messages are sent per pass; the rest go on the next pass (default 10)
    max_attempts  passes a message may fail (model error, send error) before it is skipped (default 3)
    thresholds    question id -> minimum value (P for yes/no questions, score for scale ones);
                  a message is sent only if it reaches all of them
    """
    section = _section(config_file, "run")

    send = section.get("send", False)
    if not isinstance(send, bool):
        raise ConfigError(f"{config_file.name}: run.send must be true or false (got {send!r}).")

    counts = {
        key: _positive_int(config_file, f"run.{key}", section.get(key, default))
        for key, default in (("limit", 50), ("max_per_run", 10), ("max_attempts", 3))
    }

    raw = section.get("thresholds")
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_file.name}: run.thresholds must be a mapping (question id: minimum).")
    thresholds: dict[str, float] = {}
    for qid, value in raw.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{config_file.name}: run.thresholds.{qid} must be a number (got {value!r}).")
        thresholds[str(qid)] = value

    return RunSettings(
        every=parse_duration(section.get("every", "30m"), f"{config_file.name}: run.every"),
        first_window=parse_duration(section.get("first_window", "24h"), f"{config_file.name}: run.first_window"),
        limit=counts["limit"],
        send=send,
        max_per_run=counts["max_per_run"],
        thresholds=thresholds,
        max_attempts=counts["max_attempts"],
    )


def session_file() -> Path:
    return SESSION_DIR / f"{SESSION_NAME}.session"


def require_session() -> Path:
    path = session_file()
    if not path.exists():
        raise ConfigError(
            f"No Telegram session at {path}.\n"
            "Log in once (interactive: phone number, code, 2FA):\n"
            "    uv run tege setup"
        )
    return path


def server_env(creds: Credentials) -> dict[str, str]:
    """Explicit, minimal environment for the fast-mcp-telegram subprocess."""
    env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR") if k in os.environ}
    env.update(
        API_ID=creds.api_id,
        API_HASH=creds.api_hash,
        SESSION_DIR=str(SESSION_DIR),
        SESSION_NAME=SESSION_NAME,
        SERVER_MODE="stdio",
        DO_NOT_TRACK="1",  # opt out of the server's anonymous telemetry
    )
    return env


def worker_env() -> dict[str, str]:
    """Environment for a classifier worker. It runs third-party model code, so it gets no secrets:
    only what Python and the Hugging Face cache need (HF_* may carry a token for --online)."""
    keep = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "XDG_CACHE_HOME", "TRANSFORMERS_CACHE")
    return {
        k: v
        for k, v in os.environ.items()
        if k in keep or k.startswith(("HF_", "PYTORCH_", "TOKENIZERS_"))
    }


def venv_script(name: str) -> str:
    """Path of a console script from this project's own virtualenv."""
    path = Path(sys.executable).parent / name
    if not path.exists():
        raise ConfigError(f"{name} not found next to {sys.executable}; run `uv sync`.")
    return str(path)
