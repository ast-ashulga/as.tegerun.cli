"""What `tege run` remembers between passes: the end of the last clean pass and the messages already sent.

Small JSON files in one directory (default ./.state):
  run.json     {"last_success": "<ISO time>"}        where the next pass starts reading
  sent.json    {"sent": ["<chat id>:<message id>"]}  messages that went out, so none goes out twice
  failed.json  {"failures": {"<key>": <count>}}      passes a message failed in; see max_attempts
  run.lock     held (flock) while a `tege run` is active, so two runs cannot race on these files
A corrupt file is an error, not a reset: starting over would send everything again.
"""

from __future__ import annotations

import fcntl
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tegerun import config
from tegerun.config import ConfigError


def _load(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ConfigError(f"{path} is unreadable ({exc}); fix or delete it.") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must hold a JSON object; fix or delete it.")
    return data


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent=2))
        f.flush()
        os.fsync(f.fileno())  # on disk before the rename: a crash or power loss never leaves half a state
    os.replace(tmp, path)


class RunState:
    def __init__(self, directory: Path | None = None) -> None:
        self._dir = Path(directory) if directory is not None else config.STATE_DIR
        self._run_file = self._dir / "run.json"
        self._sent_file = self._dir / "sent.json"
        self._failed_file = self._dir / "failed.json"

        raw_when = _load(self._run_file).get("last_success")
        self.last_success: datetime | None = None
        if raw_when is not None:
            try:
                when = datetime.fromisoformat(str(raw_when))
            except ValueError as exc:
                raise ConfigError(f"{self._run_file}: bad last_success {raw_when!r}; fix or delete it.") from exc
            self.last_success = when if when.tzinfo else when.replace(tzinfo=timezone.utc)

        sent = _load(self._sent_file).get("sent", [])
        if not isinstance(sent, list):
            raise ConfigError(f"{self._sent_file}: 'sent' must be a list; fix or delete it.")
        self._sent: set[str] = {str(k) for k in sent}

        failures = _load(self._failed_file).get("failures", {})
        if not isinstance(failures, dict) or not all(isinstance(n, int) for n in failures.values()):
            raise ConfigError(f"{self._failed_file}: 'failures' must map keys to counts; fix or delete it.")
        self._failures: dict[str, int] = {str(k): n for k, n in failures.items()}

    def is_sent(self, key: str) -> bool:
        return key in self._sent

    def mark_sent(self, key: str) -> None:
        self._sent.add(key)
        _save(self._sent_file, {"sent": sorted(self._sent)})

    def mark_success(self, when: datetime) -> None:
        self.last_success = when
        _save(self._run_file, {"last_success": when.isoformat()})

    def failures(self, key: str) -> int:
        return self._failures.get(key, 0)

    def record_failure(self, key: str) -> int:
        """Count one more failed pass for `key`; returns the new count."""
        self._failures[key] = self.failures(key) + 1
        _save(self._failed_file, {"failures": self._failures})
        return self._failures[key]


class RunLock:
    """Exclusive, non-blocking lock on <state dir>/run.lock; the OS drops it if the process dies."""

    def __init__(self, directory: Path | None = None) -> None:
        self._path = (Path(directory) if directory is not None else config.STATE_DIR) / "run.lock"
        self._file: Any = None

    def __enter__(self) -> "RunLock":
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self._path, "w")
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._file.close()
            raise ConfigError(f"another `tege run` is already active ({self._path} is locked).") from None
        return self

    def __exit__(self, *exc: object) -> None:
        self._file.close()  # closing releases the lock
