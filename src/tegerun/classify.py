"""Classify chat messages with System-1 models (LiquidAI d1, pplx-decider).

./classifier (questions.yaml + models.yaml + workers/) is the single source of
truth: questions, the candidate profile and the model registry are read from it,
and the models run in its own virtualenvs through its workers, which take
a requests JSON and write an answers JSON. Nothing heavy (torch, transformers)
is imported here.

Output of the models is probabilities, not text: every enabled question of the
stand gets an answer (`noul` -> P(yes), `choice` -> key, `score` -> level).
"""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tempfile
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from tegerun import config, reader
from tegerun.config import ConfigError

QTYPES = ("noul", "choice", "score")
GROUPINGS = ("joint", "single")


def _one_line(value: Any) -> str:
    return " ".join(str(value).split())


def _read_yaml(path: Path, what: str) -> object:
    if not path.exists():
        raise ConfigError(f"{what} not found: {path}")
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path.name} is not valid YAML: {exc}") from exc


# ───────────────────────────────────────────────────────── classifier config: questions


@dataclass(frozen=True)
class QuestionSet:
    skills: str
    tpl: str
    tpl_skills: str
    questions: dict[str, dict[str, Any]]


def load_questions(path: Path) -> QuestionSet:
    """Enabled questions of classifier/questions.yaml."""
    cfg = _read_yaml(path, "Questions file") or {}
    if not isinstance(cfg, dict):
        raise ConfigError(f"{path.name}: expected a mapping.")
    qs: dict[str, dict[str, Any]] = {}
    for qid, q in (cfg.get("questions") or {}).items():
        if not isinstance(q, dict):
            raise ConfigError(f"{path.name}: question {qid!r} must be a mapping")
        if q.get("enabled", True) is False:
            continue
        t = q.get("type")
        if t not in QTYPES:
            raise ConfigError(f"{path.name}: question {qid!r}: type must be one of {list(QTYPES)}, got {t!r}")
        crit = q.get("criteria")
        if t == "choice" and not (isinstance(crit, dict) and len(crit) >= 2):
            raise ConfigError(f"{path.name}: question {qid!r}: choice needs `criteria` mapping with 2+ options")
        if t == "score" and not (isinstance(crit, list) and 2 <= len(crit) <= 10):
            raise ConfigError(f"{path.name}: question {qid!r}: score needs `criteria` list of 2-10 levels")
        qs[str(qid)] = {
            "type": t,
            "instructions": _one_line(q.get("instructions", "")) or None,
            "criteria": (
                {k: _one_line(v) for k, v in crit.items()}
                if t == "choice"
                else [_one_line(v) for v in crit]
                if t == "score"
                else None
            ),
            "needs_skills": bool(q.get("needs_skills", False)),
        }
    if not qs:
        raise ConfigError(f"{path.name}: no enabled questions.")
    skills = str(cfg.get("skills") or "").strip()
    if any(q["needs_skills"] for q in qs.values()) and not skills:
        print("warning: `skills` is empty, skipping questions with needs_skills.", file=sys.stderr)
        qs = {k: v for k, v in qs.items() if not v["needs_skills"]}
        if not qs:
            raise ConfigError(f"{path.name}: no questions left without skills.")
    return QuestionSet(
        skills=skills,
        tpl=str(cfg.get("state_template") or "{message}"),
        tpl_skills=str(cfg.get("state_template_with_skills") or "{message}\n\n{skills}"),
        questions=qs,
    )


# ───────────────────────────────────────────────────────── classifier config: models


@dataclass(frozen=True)
class ModelSpec:
    name: str
    runner: str
    venv: str
    repo: str
    revision: str | None = None
    grouping: str = "joint"
    device: str = "auto"
    dtype: str = "auto"


def load_models(path: Path) -> dict[str, ModelSpec]:
    data = _read_yaml(path, "Model registry")
    raw = data.get("models") if isinstance(data, dict) else None
    if not isinstance(raw, dict) or not raw:
        raise ConfigError(f"{path.name}: 'models' must be a non-empty mapping.")
    models: dict[str, ModelSpec] = {}
    for name, m in raw.items():
        if not isinstance(m, dict):
            raise ConfigError(f"{path.name}: model {name!r} must be a mapping")
        missing = [k for k in ("runner", "venv", "repo") if not m.get(k)]
        if missing:
            raise ConfigError(f"{path.name}: model {name!r} is missing {', '.join(missing)}")
        grouping = str(m.get("grouping", "joint"))
        if grouping not in GROUPINGS:
            raise ConfigError(f"{path.name}: model {name!r}: grouping must be one of {list(GROUPINGS)}")
        models[str(name)] = ModelSpec(
            name=str(name),
            runner=str(m["runner"]),
            venv=str(m["venv"]),
            repo=str(m["repo"]),
            revision=str(m["revision"]) if m.get("revision") else None,
            grouping=grouping,
            device=str(m.get("device", "auto")),
            dtype=str(m.get("dtype", "auto")),
        )
    return models


# ───────────────────────────────────────────────────────── messages -> items


@dataclass(frozen=True)
class Item:
    id: str  # unique across chats: "<peer_id>:<message id>"
    text: str
    chat_id: str
    chat: str
    msg_id: Any = None
    date: str = ""
    link: str | None = None


def _unique(base: str, seen: set[str]) -> str:
    ident, n = base, 1
    while ident in seen:
        n += 1
        ident = f"{base}#{n}"
    seen.add(ident)
    return ident


def messages_to_items(
    results: list[reader.ChatResult], chats: dict[str, config.Chat] | None = None
) -> tuple[list[Item], int]:
    """Classifiable items from fetched chats; messages without text are counted as skipped."""
    chats = chats or {}
    items: list[Item] = []
    seen: set[str] = set()
    skipped = 0
    for r in results:
        name = chats[r.peer_id].name if r.peer_id in chats else r.peer_id
        for msg in r.messages:
            text = str(msg.get("text") or msg.get("message") or "").strip()
            if not text:
                skipped += 1
                continue
            msg_id = msg.get("id")
            ident = _unique(f"{r.peer_id}:{msg_id if msg_id is not None else len(items)}", seen)
            items.append(
                Item(
                    id=ident,
                    text=text,
                    chat_id=r.peer_id,
                    chat=name,
                    msg_id=msg_id,
                    date=str(msg.get("date") or ""),
                    link=msg.get("link"),
                )
            )
    return items, skipped


def load_jsonl(path: Path) -> tuple[list[Item], int]:
    """Items from a JSON-lines file of {src_id, src_name, msg_id, msg} (see messages.jsonl)."""
    if not path.exists():
        raise ConfigError(f"{path} not found.")
    items: list[Item] = []
    seen: set[str] = set()
    skipped = 0
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ConfigError(f"{path.name}:{n}: not valid JSON: {exc}") from exc
        text = str(rec.get("msg") or rec.get("text") or "").strip() if isinstance(rec, dict) else ""
        if not text:
            skipped += 1
            continue
        peer = str(rec.get("src_id", ""))
        items.append(
            Item(
                id=_unique(f"{peer}:{rec.get('msg_id', n)}", seen),
                text=text,
                chat_id=peer,
                chat=str(rec.get("src_name") or peer),
                msg_id=rec.get("msg_id"),
                date=str(rec.get("date") or ""),
            )
        )
    return items, skipped


# ───────────────────────────────────────────────────────── requests for the workers


def build_requests(
    items: list[Item], qset: QuestionSet, grouping: str
) -> tuple[list[dict[str, Any]], dict[str, tuple[str, list[str]]]]:
    """(worker requests, rid -> (item id, question ids))."""
    qs = qset.questions
    plain = [q for q, d in qs.items() if not d["needs_skills"]]
    withs = [q for q, d in qs.items() if d["needs_skills"]]
    if grouping == "joint":
        groups = [("plain", plain, False), ("skills", withs, True)]
    else:
        groups = [(q, [q], qs[q]["needs_skills"]) for q in qs]
    requests: list[dict[str, Any]] = []
    rid_map: dict[str, tuple[str, list[str]]] = {}
    for it in items:
        for gkey, qids, use_skills in groups:
            if not qids:
                continue
            tpl = qset.tpl_skills if use_skills else qset.tpl
            state = tpl.replace("{message}", it.text).replace("{skills}", qset.skills)
            payload: dict[str, Any] = {}
            for qid in qids:
                q = qs[qid]
                d: dict[str, Any] = {"type": q["type"]}
                if q["instructions"]:
                    d["instructions"] = q["instructions"]
                if q["criteria"] is not None:
                    d["criteria"] = q["criteria"]
                payload[qid] = d
            rid = f"{it.id}::{gkey}"
            requests.append({"rid": rid, "state": state, "questions": payload})
            rid_map[rid] = (it.id, qids)
    return requests, rid_map


def _stderr_fd() -> Any:
    """Worker progress goes to stderr so stdout stays clean for --json."""
    try:
        return sys.stderr.fileno()
    except (AttributeError, OSError, ValueError, io.UnsupportedOperation):
        return subprocess.DEVNULL


def run_worker(
    settings: config.ClassifySettings,
    model: ModelSpec,
    requests: list[dict[str, Any]],
    online: bool = False,
) -> dict[str, Any]:
    """Run the classifier's worker for `model` in its own venv; returns the worker's out.json."""
    venv_py = settings.classifier_dir / model.venv / "bin" / "python"
    worker = settings.classifier_dir / "workers" / f"{model.runner}.py"
    if not venv_py.exists():
        raise ConfigError(f"No environment {venv_py}; run ./setup.sh in {settings.classifier_dir}.")
    if not worker.exists():
        raise ConfigError(f"Worker {worker} not found.")
    wcfg = {
        "repo": model.repo,
        "revision": model.revision,
        "device": model.device,
        "dtype": model.dtype,
        "online": online,
    }
    with tempfile.TemporaryDirectory(prefix="tege-classify-") as tmp:
        req_path, out_path = Path(tmp) / "requests.json", Path(tmp) / "out.json"
        req_path.write_text(json.dumps(requests, ensure_ascii=False), encoding="utf-8")
        cmd = [
            str(venv_py), "-I", str(worker),
            "--config", json.dumps(wcfg),
            "--requests", str(req_path),
            "--out", str(out_path),
        ]  # fmt: skip
        rc = subprocess.run(cmd, stdout=_stderr_fd(), check=False).returncode
        if rc != 0 or not out_path.exists():
            raise ConfigError(
                f"Worker for {model.name} failed (exit code {rc}); see its messages above."
            )
        return json.loads(out_path.read_text(encoding="utf-8"))


# ───────────────────────────────────────────────────────── answers -> verdicts


def normalize_answer(q: dict[str, Any], ans: dict[str, Any]) -> dict[str, Any]:
    """Model answer -> {type, pred, conf, ...};"""
    t = q["type"]
    if t == "noul":
        p = float(ans["noul"])
        return {"type": t, "p": p, "pred": p > 0.5, "conf": max(p, 1 - p)}
    if t == "choice":
        return {
            "type": t,
            "pred": ans["choice"],
            "conf": float(ans["confidence"]),
            "probs": {k: float(v) for k, v in ans["probabilities"].items()},
        }
    probs = {int(k): float(v) for k, v in ans["probabilities"].items()}
    return {
        "type": t,
        "pred": max(probs, key=probs.get),  # type: ignore[arg-type]
        "score": float(ans["score"]),
        "conf": float(ans["confidence"]),
        "probs": probs,
    }


def format_answer(a: dict[str, Any]) -> str:
    """Compact cell, e.g. `P=0.93`, `vacancy 0.88`, `3.2 (ур.3)`."""
    if a["type"] == "noul":
        return f"P={a['p']:.2f}"
    if a["type"] == "choice":
        return f"{a['pred']} {a['conf']:.2f}"
    return f"{a['score']:.1f} (ур.{a['pred']})"


@dataclass
class Verdict:
    item: Item
    answers: dict[str, dict[str, Any]] = field(default_factory=dict)
    error: str | None = None


def collect_verdicts(
    items: list[Item],
    qset: QuestionSet,
    rid_map: dict[str, tuple[str, list[str]]],
    out: dict[str, Any],
) -> list[Verdict]:
    """Fold the worker's per-request results back into one verdict per message."""
    verdicts = {it.id: Verdict(it) for it in items}
    answered: set[str] = set()
    for res in out.get("results", []):
        rid = res.get("rid")
        if rid not in rid_map:
            continue
        item_id, qids = rid_map[rid]
        v = verdicts[item_id]
        if "error" in res:
            v.error = res["error"]
            continue
        try:
            for qid in qids:
                v.answers[qid] = normalize_answer(qset.questions[qid], res["answers"][qid])
            answered.add(rid)
        except (KeyError, TypeError, ValueError) as exc:
            v.error = f"malformed answer ({type(exc).__name__}: {exc})"
    for rid, (item_id, _) in rid_map.items():
        v = verdicts[item_id]
        if rid not in answered and v.error is None:
            v.error = "no result from the model"
    # keep the questions.yaml order in answers
    order = list(qset.questions)
    for v in verdicts.values():
        v.answers = {q: v.answers[q] for q in order if q in v.answers}
    return list(verdicts.values())


def classify(
    items: list[Item],
    qset: QuestionSet,
    model: ModelSpec,
    grouping: str,
    settings: config.ClassifySettings,
    online: bool = False,
    worker: Callable[..., dict[str, Any]] | None = None,
) -> tuple[list[Verdict], dict[str, Any]]:
    requests, rid_map = build_requests(items, qset, grouping)
    out = (worker or run_worker)(settings, model, requests, online)
    return collect_verdicts(items, qset, rid_map, out), out.get("meta", {})
