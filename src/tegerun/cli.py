"""`tegerun` command line: one-time login, read-only Telegram message fetching, classification and notifications."""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import sys
from pathlib import Path
from typing import Any

from tegerun import classify, config, notifier, reader, runner, runstate


def _sender(msg: dict[str, Any]) -> str:
    sender = msg.get("sender")
    if isinstance(sender, dict):
        name = sender.get("name") or " ".join(
            p for p in (sender.get("first_name"), sender.get("last_name")) if p
        )
        return name or sender.get("username") or str(sender.get("id", "?"))
    return str(sender or msg.get("sender_id") or msg.get("from") or "?")


def format_message(msg: dict[str, Any]) -> str:
    when = str(msg.get("date", ""))[:16].replace("T", " ")
    text = msg.get("text") or msg.get("message") or ""
    if not text:
        text = "[no text]" if not msg.get("media") else f"[{msg.get('media')}]"
    text = "\n".join(("    " + ln) if ln.strip() else "" for ln in str(text).split("\n")).lstrip()
    return f"[{when}] {_sender(msg)}: {text}"


def _chat_for(r: reader.ChatResult, chats: dict[str, config.Chat]) -> config.Chat:
    return chats.get(r.peer_id) or config.Chat(id=r.peer_id, name=r.peer_id)


def render_text(results: list[reader.ChatResult], chats: dict[str, config.Chat] | None = None) -> str:
    chats = chats or {}
    out: list[str] = []
    for r in results:
        chat = _chat_for(r, chats)
        out.append(f"=== {chat.name} ({chat.id}) ===")
        if chat.description:
            out.append(f"  {chat.description}")
        if r.error:
            out.append(f"  ERROR: {r.error}")
        elif not r.messages:
            out.append("  (no messages)")
        else:
            out.extend(format_message(m) for m in r.messages)
            if r.has_more:
                out.append("  ... more messages available (raise --limit)")
        out.append("")
    return "\n".join(out)


def render_json(results: list[reader.ChatResult], chats: dict[str, config.Chat] | None = None) -> str:
    chats = chats or {}
    return json.dumps(
        [
            {
                "peer_id": r.peer_id,
                "name": _chat_for(r, chats).name,
                "description": _chat_for(r, chats).description,
                "error": r.error,
                "has_more": r.has_more,
                "messages": r.messages,
            }
            for r in results
        ],
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def cmd_setup(_: argparse.Namespace) -> int:
    creds = config.load_credentials()
    config.SESSION_DIR.mkdir(parents=True, exist_ok=True)
    config.SESSION_DIR.chmod(0o700)
    script = config.venv_script("fast-mcp-telegram-setup")
    env = config.server_env(creds)
    print(f"Logging in; session will be stored in {config.session_file()}", file=sys.stderr)
    # Replace this process so the OTP / 2FA prompts get the real terminal.
    os.chdir(config.SESSION_DIR)
    os.execve(script, [script], env)
    return 0  # unreachable


def _fetch(args: argparse.Namespace) -> tuple[list[reader.ChatResult], dict[str, config.Chat]]:
    """Whitelist check + read of the requested window (shared by `read` and `classify`)."""
    chats = {c.id: c for c in config.load_whitelist()}
    peers = args.chat or list(chats)
    outside = [p for p in peers if p not in chats]
    if outside:
        raise config.ConfigError(
            f"Not in whitelist: {', '.join(outside)} (edit {config.CONFIG_FILE.name} to allow)."
        )
    since = reader.parse_since(args.since) if args.since else None
    until = reader.parse_since(args.until, flag="--until") if args.until else None
    results = asyncio.run(reader.fetch_recent(peers, args.limit, since, verbose=args.verbose, until=until))
    return results, chats


def cmd_read(args: argparse.Namespace) -> int:
    results, chats = _fetch(args)
    print(render_json(results, chats) if args.json else render_text(results, chats))
    return 1 if all(r.error for r in results) else 0


def _snippet(text: str, n: int = 160) -> str:
    s = " ".join(text.split())
    return s if len(s) <= n else s[: n - 1] + "…"


def render_classify_text(verdicts: list[classify.Verdict], skipped: int, model: str) -> str:
    out: list[str] = [f"Model: {model}. Classified: {len(verdicts)}, skipped (no text): {skipped}.", ""]
    current = None
    for v in verdicts:
        it = v.item
        if it.chat_id != current:
            current = it.chat_id
            out.append(f"=== {it.chat} ({it.chat_id}) ===")
        when = it.date[:16].replace("T", " ")
        out.append(f"{f'[{when}] ' if when else ''}#{it.msg_id}: {_snippet(it.text)}")
        if v.error:
            out.append(f"    ERROR: {v.error}")
        else:
            out.append("    " + " | ".join(f"{q}: {classify.format_answer(a)}" for q, a in v.answers.items()))
    return "\n".join(out)


def render_classify_json(
    verdicts: list[classify.Verdict], skipped: int, model: str, grouping: str, meta: dict[str, Any]
) -> str:
    return json.dumps(
        {
            "model": model,
            "grouping": grouping,
            "meta": meta,
            "skipped_no_text": skipped,
            "results": [
                {
                    "id": v.item.id,
                    "chat_id": v.item.chat_id,
                    "chat": v.item.chat,
                    "msg_id": v.item.msg_id,
                    "date": v.item.date,
                    "link": v.item.link,
                    "text": v.item.text,
                    "error": v.error,
                    "answers": v.answers,
                }
                for v in verdicts
            ],
        },
        ensure_ascii=False,
        indent=2,
        default=str,
    )


def cmd_classify(args: argparse.Namespace) -> int:
    settings = config.load_classify_settings()
    qset = classify.load_questions(settings.questions_file)
    models = classify.load_models(settings.models_file)
    name = args.model or settings.model
    if name not in models:
        raise config.ConfigError(f"Unknown model {name!r}; available: {', '.join(models)}.")
    model = models[name]
    grouping = args.grouping or model.grouping

    fetch_failed = False
    if args.input:
        items, skipped = classify.load_jsonl(args.input)
    else:
        results, chats = _fetch(args)
        for r in results:
            if r.error:
                print(f"warning: {_chat_for(r, chats).name} ({r.peer_id}): {r.error}", file=sys.stderr)
            elif r.has_more and args.since:
                print(
                    f"warning: {_chat_for(r, chats).name}: more messages in the window than --limit "
                    f"({args.limit}); the rest is not classified.",
                    file=sys.stderr,
                )
        fetch_failed = all(r.error for r in results)
        items, skipped = classify.messages_to_items(results, chats)
    if not items:
        print("Nothing to classify (no messages with text).", file=sys.stderr)
        return 1 if fetch_failed else 0

    if args.dry_run:
        requests, _ = classify.build_requests(items, qset, grouping)
        print(json.dumps({"model": name, "grouping": grouping, "requests": requests}, ensure_ascii=False, indent=2))
        return 0

    print(f"Classifying {len(items)} messages with {name} ({grouping}) ...", file=sys.stderr)
    verdicts, meta = classify.classify(items, qset, model, grouping, settings, online=args.online)
    print(
        render_classify_json(verdicts, skipped, name, grouping, meta)
        if args.json
        else render_classify_text(verdicts, skipped, name)
    )
    return 1 if all(v.error for v in verdicts) else 0


def cmd_notify(args: argparse.Namespace) -> int:
    settings = config.load_notify_settings()
    text = args.text if args.text is not None else sys.stdin.read()
    reply_to = args.reply_to if args.reply_to is not None else settings.reply_to
    if args.dry_run:
        state = "enabled" if settings.enabled else "DISABLED (notify.enabled is false)"
        print(f"[dry run] notifications {state}")
        print(f"to: {settings.chat_id or '(notify.chat_id not set)'}" + (f", topic #{settings.topic_id}" if settings.topic_id else "")
              + (f", reply to #{reply_to}" if reply_to else ""))
        print(text.strip())
        return 0
    if not settings.enabled:
        print("error: notifications are disabled (set notify.enabled: true in .config.yaml).", file=sys.stderr)
        return 1
    result = asyncio.run(notifier.send_notification(text, settings=settings, reply_to=args.reply_to, verbose=args.verbose))
    if not result.sent:
        print(f"error: not sent: {result.error}", file=sys.stderr)
        return 1
    print(f"Sent to {settings.chat_id}: message id(s) {', '.join(str(i) for i in result.message_ids)}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    run = config.load_run_settings()
    if args.every:
        run = dataclasses.replace(run, every=config.parse_duration(args.every, "--every"))
    if args.send is not None:
        run = dataclasses.replace(run, send=args.send)
    notify = config.load_notify_settings()
    if run.send and not notify.enabled:
        raise config.ConfigError("sending is on but notify.enabled is false in .config.yaml; enable it or use --no-send.")

    settings = config.load_classify_settings()
    qset = classify.load_questions(settings.questions_file)
    models = classify.load_models(settings.models_file)
    name = args.model or settings.model
    if name not in models:
        raise config.ConfigError(f"Unknown model {name!r}; available: {', '.join(models)}.")
    model = models[name]
    runner.validate_thresholds(run.thresholds, qset)

    chats = {c.id: c for c in config.load_whitelist()}
    state = runstate.RunState()

    async def fetch(since: str) -> tuple[list[reader.ChatResult], dict[str, config.Chat]]:
        results = await reader.fetch_recent(list(chats), run.limit, since, verbose=args.verbose)
        return results, chats

    def classify_items(items: list[classify.Item]) -> list[classify.Verdict]:
        print(f"Classifying {len(items)} messages with {name} ({model.grouping}) ...", file=sys.stderr)
        return classify.classify(items, qset, model, model.grouping, settings, online=args.online)[0]

    async def send(text: str) -> notifier.NotifyResult:
        return await notifier.send_notification(text, settings=notify, verbose=args.verbose)

    def one_pass() -> runner.PassReport:
        report = asyncio.run(
            runner.run_pass(run, state, fetch=fetch, classify_items=classify_items, send=send)
        )
        for text in report.would_send:
            print(f"[send is off] would send:\n{text}\n")
        for err in report.errors:
            print(f"error: {err}", file=sys.stderr)
        print(
            f"Pass: read {report.read}, classified {report.classified}, matched {report.matched}, "
            f"sent {report.sent}{'' if report.ok else ' (window not advanced)'}.",
            file=sys.stderr,
        )
        return report

    if args.once:
        return 0 if one_pass().ok else 1
    print(f"Running every {run.every}; send is {'ON' if run.send else 'off'}. Ctrl+C to stop.", file=sys.stderr)
    runner.run_loop(one_pass, run.every)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tegerun", description="Read-only Telegram triage CLI: read, classify, notify via fast-mcp-telegram.")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("setup", help="one-time Telegram login (interactive)")
    s.set_defaults(func=cmd_setup)

    def add_window(sp: argparse.ArgumentParser, default_limit: int) -> None:
        sp.add_argument("--chat", action="append", help="chat id to read (repeatable); default: whole whitelist")
        sp.add_argument("--limit", type=int, default=default_limit, help=f"messages per chat (default {default_limit})")
        sp.add_argument("--since", help="only newer than e.g. 24h, 7d, 2w or an ISO date")
        sp.add_argument("--until", help="only older than e.g. 1h, 1d or an ISO date")
        sp.add_argument("--json", action="store_true", default=False, help="machine-readable output")
        sp.add_argument("--verbose", action="store_true", default=False, help="show MCP server logs on stderr")

    r = sub.add_parser("read", help="read recent messages from whitelisted chats")
    add_window(r, 20)
    r.set_defaults(func=cmd_read)

    c = sub.add_parser(
        "classify",
        help="read messages (as `read`) and classify them with the models and questions from ./classifier",
    )
    add_window(c, 20)
    c.add_argument("--model", help="model from classifier/models.yaml (default: classify.model, d1-3B)")
    c.add_argument("--grouping", choices=list(classify.GROUPINGS), help="override the model's grouping")
    c.add_argument("--input", type=Path, help="classify a JSON-lines file ({src_id,src_name,msg_id,msg}) instead of reading Telegram")
    c.add_argument("--online", action="store_true", default=False, help="allow the model worker to use the network (default: HF cache only)")
    c.add_argument("--dry-run", action="store_true", default=False, help="print the requests for the model and exit; nothing is loaded")
    c.set_defaults(func=cmd_classify)

    n = sub.add_parser(
        "notify",
        help="send a message to the chat configured in .config.yaml (notify:); off unless notify.enabled is true",
    )
    n.add_argument("text", nargs="?", help="message text (default: read from stdin)")
    n.add_argument("--reply-to", type=int, help="reply to this message id (default: notify.reply_to)")
    n.add_argument("--dry-run", action="store_true", default=False, help="show where and what would be sent; send nothing")
    n.add_argument("--verbose", action="store_true", default=False, help="show MCP server logs on stderr")
    n.set_defaults(func=cmd_notify)

    u = sub.add_parser(
        "run",
        help="read -> classify -> keep matches above the thresholds -> send; repeats every run.every (see run: in .config.yaml)",
    )
    u.add_argument("--once", action="store_true", default=False, help="a single pass, then exit (for cron/launchd)")
    u.add_argument("--every", help="pause between passes, e.g. 30m, 1h, 3h (default: run.every)")
    u.add_argument(
        "--send", action=argparse.BooleanOptionalAction, default=None,
        help="deliver matches to Telegram / only print them (default: run.send; needs notify.enabled)",
    )
    u.add_argument("--model", help="model from classifier/models.yaml (default: classify.model)")
    u.add_argument("--online", action="store_true", default=False, help="allow the model worker to use the network")
    u.add_argument("--verbose", action="store_true", default=False, help="show MCP server logs on stderr")
    u.set_defaults(func=cmd_run)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (config.ConfigError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
