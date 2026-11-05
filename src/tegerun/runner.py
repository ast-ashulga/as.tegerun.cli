"""One pass of `tege run` (read -> classify -> thresholds -> send) and the loop that repeats it.

The three boundaries (Telegram read, the model, Telegram send) are passed in, so the pass itself is
plain logic. What is worth sending is decided here by thresholds from `run.thresholds` only.
"""

from __future__ import annotations

import asyncio
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from tegerun import classify, config, notifier, reader
from tegerun.config import ConfigError
from tegerun.runstate import RunState

SNIPPET_LIMIT = 600
THRESHOLD_TYPES = ("noul", "score")  # yes/no -> P, scale -> score; a `choice` has no single number


def _log(message: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}", file=sys.stderr, flush=True)


# ───────────────────────────────────────────────────────── thresholds


def validate_thresholds(thresholds: dict[str, float], qset: classify.QuestionSet) -> None:
    """Refuse to run with no thresholds (it would send everything) or ones that can never be checked."""
    if not thresholds:
        raise ConfigError("run.thresholds is empty: set at least one, otherwise every message would be sent.")
    for qid in thresholds:
        question = qset.questions.get(qid)
        if question is None:
            raise ConfigError(
                f"run.thresholds.{qid}: no such enabled question (have: {', '.join(qset.questions)})."
            )
        if question["type"] not in THRESHOLD_TYPES:
            raise ConfigError(f"run.thresholds.{qid}: a {question['type']} question has no numeric value to compare.")
        criteria = question.get("criteria")
        top = 1 if question["type"] == "noul" else (len(criteria) - 1 if criteria else None)
        if top is not None and not 0 <= thresholds[qid] <= top:
            raise ConfigError(f"run.thresholds.{qid}: {thresholds[qid]} can never be reached; use a value in 0..{top}.")


def passes_thresholds(verdict: classify.Verdict, thresholds: dict[str, float]) -> bool:
    """True if the message reached every threshold. Errored or unanswered questions never pass."""
    if verdict.error:
        return False
    for qid, minimum in thresholds.items():
        answer = verdict.answers.get(qid)
        if answer is None:
            return False
        value = answer["p"] if answer["type"] == "noul" else answer.get("score")
        if value is None or value < minimum:
            return False
    return True


# ───────────────────────────────────────────────────────── message text


def format_candidate(verdict: classify.Verdict) -> str:
    """The Telegram message for one vacancy: chat, text snippet, the model's answers, link."""
    item = verdict.item
    text = item.text.strip()
    if len(text) > SNIPPET_LIMIT:
        text = text[: SNIPPET_LIMIT - 1] + "…"
    answers = " · ".join(f"{q} {classify.format_answer(a)}" for q, a in verdict.answers.items())
    parts = [f"🎯 {item.chat}", text, answers]
    if item.link:
        parts.append(item.link)
    return "\n\n".join(p for p in parts if p)


# ───────────────────────────────────────────────────────── one pass


@dataclass
class PassReport:
    ok: bool  # False: something failed or was left for later; the window did not move on
    read: int = 0
    classified: int = 0
    matched: int = 0
    sent: int = 0
    would_send: list[str] = field(default_factory=list)  # filled instead of sending when run.send is off
    errors: list[str] = field(default_factory=list)


def _key(item: classify.Item) -> str:
    return f"{item.chat_id}:{item.msg_id}"


async def run_pass(
    run: config.RunSettings,
    state: RunState,
    *,
    fetch: Callable[[str], Awaitable[tuple[list[reader.ChatResult], dict[str, config.Chat]]]],
    classify_items: Callable[[list[classify.Item]], list[classify.Verdict]],
    send: Callable[[str], Awaitable[notifier.NotifyResult]],
    now: datetime | None = None,
    log: Callable[[str], None] = _log,
) -> PassReport:
    now = now or datetime.now(timezone.utc)
    report = PassReport(ok=True)
    since = (state.last_success or now - run.first_window).isoformat()

    results, chats = await fetch(since)
    for r in results:
        name = chats[r.peer_id].name if r.peer_id in chats else r.peer_id
        if r.error:
            report.errors.append(f"read {name}: {r.error}")
        elif r.has_more:
            # Moving on would drop the unread messages for good, so the window is held instead.
            report.errors.append(
                f"read {name}: the window holds more messages than could be read "
                f"(run.limit {run.limit} per page); raise run.limit or run more often."
            )
    items, _ = classify.messages_to_items(results, chats)
    report.read = len(items)
    fresh = [it for it in items if it.msg_id is not None and not state.is_sent(_key(it))]

    # A message that keeps failing must not hold the window (and re-run the model) forever.
    exhausted = {_key(it) for it in fresh if state.failures(_key(it)) >= run.max_attempts}
    if exhausted:
        log(f"skipping {len(exhausted)} message(s) that failed {run.max_attempts} passes: {', '.join(sorted(exhausted))}")
        fresh = [it for it in fresh if _key(it) not in exhausted]

    verdicts = await asyncio.to_thread(classify_items, fresh) if fresh else []
    report.classified = len(verdicts)
    failed = [v for v in verdicts if v.error]
    for v in failed:
        report.errors.append(f"classify {_key(v.item)}: {v.error}")
    # Only errors the worker reported for a message count against it. "No result" (the worker died
    # before reaching it) is transient. Everything failing at once looks like a broken model, not broken
    # messages: blame nobody then. A lone message that fails cannot be told apart, so it is blamed
    # (otherwise it would block forever).
    if run.send and (len(failed) < len(verdicts) or len(verdicts) == 1):
        for v in failed:
            if not v.transient:
                state.record_failure(_key(v.item))

    matches = [v for v in verdicts if passes_thresholds(v, run.thresholds)]
    report.matched = len(matches)
    batch, rest = matches[: run.max_per_run], matches[run.max_per_run :]
    if rest:
        log(f"{len(rest)} more match(es) than run.max_per_run ({run.max_per_run}); they go on the next pass.")

    failed_sends: list[str] = []
    in_a_row = 0
    for v in batch:
        text = format_candidate(v)
        if not run.send:
            report.would_send.append(text)
            continue
        result = await send(text)
        if not result.sent:
            report.errors.append(f"send {_key(v.item)}: {result.error}")
            failed_sends.append(_key(v.item))
            in_a_row += 1
            if in_a_row >= 2:
                break  # most likely systemic (access, network): do not hammer the rest
            continue
        in_a_row = 0
        state.mark_sent(_key(v.item))
        report.sent += 1
    if report.sent:  # others got through, so these failures are about the message itself
        for key in failed_sends:
            state.record_failure(key)

    if report.errors or rest:
        report.ok = False
    elif run.send:
        # The next pass reads from the start of this one: nothing falls in a gap. A preview pass
        # (send off) leaves the window alone, otherwise looking at matches would consume them.
        state.mark_success(now)
    return report


# ───────────────────────────────────────────────────────── the loop


def run_loop(
    pass_fn: Callable[[], object],
    every: timedelta,
    *,
    sleep: Callable[[float], None] = time.sleep,
    max_passes: int | None = None,
    log: Callable[[str], None] = _log,
) -> None:
    """Call `pass_fn` over and over with `every` between passes. A failing pass is logged, not fatal."""
    done = 0
    try:
        while max_passes is None or done < max_passes:
            try:
                pass_fn()
            except Exception as exc:  # one bad pass (model crashed, MCP down) must not stop the schedule
                log(f"pass failed: {type(exc).__name__}: {exc}")
            done += 1
            if max_passes is not None and done >= max_passes:
                break
            log(f"next pass in {every}")
            sleep(every.total_seconds())
    except KeyboardInterrupt:
        log("stopped")
