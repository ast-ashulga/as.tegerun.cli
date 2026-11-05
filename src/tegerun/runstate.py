"""What `tege run` remembers between passes: the end of the last clean pass and the messages already sent.

Two small JSON files in one directory (default ./.state):
  run.json   {"last_success": "<ISO time>"}        where the next pass starts reading
  sent.json  {"sent": ["<chat id>:<message id>"]}  messages that went out, so none goes out twice
A corrupt file is an error, not a reset: starting over would send everything again.
"""

from __future__ import annotations

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
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)  # a crash mid-write never leaves a half-written state


class RunState:
    def __init__(self, directory: Path | None = None) -> None:
        self._dir = Path(directory) if directory is not None else config.STATE_DIR
        self._run_file = self._dir / "run.json"
        self._sent_file = self._dir / "sent.json"

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

    def is_sent(self, key: str) -> bool:
        return key in self._sent

    def mark_sent(self, key: str) -> None:
        self._sent.add(key)
        _save(self._sent_file, {"sent": sorted(self._sent)})

    def mark_success(self, when: datetime) -> None:
        self.last_success = when
        _save(self._run_file, {"last_success": when.isoformat()})
