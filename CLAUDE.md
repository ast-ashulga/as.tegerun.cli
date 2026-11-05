# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`tegerun` (CLI entry point `tege`) is a read-only Telegram triage tool. It reads whitelisted chats through the `fast-mcp-telegram` MCP server, scores each message with local "System-1" classifier models (probabilities and scores, never generated text), and forwards the messages that pass every threshold to one notify chat. It targets macOS and uses `uv` throughout.

## Commands

```bash
uv sync                                   # install deps (incl. dev group: pytest, pytest-asyncio)
uv run pytest                             # full suite (fast, no network/Telegram/models needed)
uv run ruff check src classifier tests    # pyflakes: undefined names, unused imports (also run by the suite)
uv run pytest tests/test_run.py           # one file
uv run pytest tests/test_run.py::test_state_starts_empty   # one test
uv run pytest -k thresholds               # by keyword

uv run tege setup                         # one-time interactive Telegram login -> .sessions/telegram.session
uv run tege read --limit 20 [--chat ID] [--since 2h] [--json]
uv run tege classify --dry-run            # print worker requests; loads no model
uv run tege classify --input msgs.jsonl   # classify a JSONL file instead of reading Telegram
uv run tege notify "text" --dry-run
uv run tege run --once [--no-send]        # one read->classify->filter->send pass

./classifier/setup.sh                     # create classifier venvs (.venv-liquid, .venv-pplx)
./classifier/setup.sh --download d1-3B    # download model weights to the HF cache
./classifier/setup.sh --code-only <name>  # fetch only model code, to review it before downloading weights
```

pytest is configured with `asyncio_mode = "auto"` (async tests need no decorator). Ruff runs pyflakes rules only (`select = ["F"]`); no formatter is configured. `tests/test_smoke.py` compiles every `.py` (including the workers, which run in their own venvs and are never imported by tests) and imports every module.

## Architecture

Package `src/tegerun/`. `cli.py` builds the argparse subcommands and turns `ConfigError` or `ValueError` into `error: ...` and exit code 2.

- **Paths and config (`config.py`)**: `PROJECT_ROOT` is the checkout the package lives in (two levels above `src/tegerun`), or `$TEGE_HOME` when set. Every path is relative to it: `.env`, `.config.yaml`, `.sessions/`, `.state/` and `logs/`. Each `.config.yaml` section (`whitelist`, `classify`, `notify`, `run`) has a `load_*` function that validates strictly and raises `ConfigError` with a message the user can act on. User-facing problems should use `ConfigError`, not raw exceptions.
- **Telegram via MCP (`mcp_client.py`, `reader.py`, `notifier.py`)**: `mcp_client.open_session()` starts `fast-mcp-telegram` from this project's venv (`config.venv_script`) as a stdio subprocess. Its environment is built explicitly by `config.server_env()`: a minimal allowlist plus the API credentials. Its cwd is `.sessions/` so that it never auto-loads the repo's `.env`. Server logs go to `logs/server.log` unless `--verbose` is set. `open_session(allow=...)` yields a `GuardedSession`: only the allowlisted tool names get through (`READ_TOOLS` for `reader`, `SEND_TOOLS` for `notifier`) and each call has a timeout. The server exposes much more (edit, raw MTProto, send-to-phone), so never widen an allowlist casually. `reader.fetch_recent` also enforces the whitelist (`allowed=`), not just the CLI. The server returns the *newest* `limit` messages of a window; with `paginate=True` older pages are fetched via a moving `max_date` and `has_more` stays true only if pages ran out. `notifier.Sender` delivers a whole pass through one MCP session; text from other people is sent HTML-escaped with `parse_mode="html"` because the server's default `auto` would interpret markup. `notifier` supports forum `topic_id` and `reply_to` and splits long text.
- **Classification (`classify.py` + `classifier/`)**: `classifier/` is the single source of truth. It holds `questions.yaml` (question types `noul`, `choice` and `score`, plus an optional `skills` profile substituted into questions with `needs_skills: true`), `models.yaml` (registry with pinned HF revisions) and `workers/*.py`. `classify.py` never imports torch or transformers. It writes a `requests.json`, runs `classifier/<venv>/bin/python -I workers/<runner>.py --config ... --requests ... --out ...` as a subprocess (scratch cwd, `config.worker_env()` with no secrets, `classify.timeout`), reads `out.json` (workers save it every 10 requests, so a killed worker's partial results are used), and normalizes the answers into `Verdict`s. A crash or timeout with no output raises `WorkerError` (exit code 1; `ConfigError` stays for bad configuration, exit code 2). Message text goes into the state template in one pass (`render_state`), so placeholders inside a message are not expanded. Each model family has its own venv because their `transformers` versions conflict. `grouping: joint` sends all questions for a message in one request; `single` sends one request per question. Workers run offline (HF cache only) unless `--online` is passed. The worker I/O contract is described at the top of `classifier/workers/d1_worker.py`.
- **Run loop (`runner.py`, `runstate.py`)**: `run_pass()` is pure logic. The three boundaries (`fetch`, `classify_items`, `send`) are injected callables, which `cli.cmd_run` wires to the real MCP and model calls; tests pass fakes. One pass works like this:
  1. Read from `last_success`, or from `now - first_window` when there is none.
  2. Skip keys already in `sent.json` (key is `chat_id:msg_id`), and keys that failed `run.max_attempts` passes (`failed.json`) so one poison message cannot hold the window forever. Failures are only blamed on messages when not everything failed at once (that looks like a broken model).
  3. Classify the remaining messages.
  4. Keep messages that pass **all** `run.thresholds`. Only `noul` (its `p`) and `score` questions can have thresholds. An empty thresholds map is rejected.
  5. Send at most `max_per_run`; stop after two send failures in a row.
  6. Advance `last_success` only when the pass sent for real (`run.send`), had no errors (including an unreadable overflow, `has_more`) and no leftover matches. The next pass then re-reads the same window, and `sent.json` prevents duplicates. A preview (send off) changes no state.

  `cmd_run` holds `runstate.RunLock` (flock on `.state/run.lock`) so two runs cannot race. `RunState` writes atomically (tmp file + fsync + `os.replace`). A corrupt state file raises an error and is never reset, because a reset would resend everything. `run_loop()` logs a failing pass and keeps the schedule going.

## Conventions and gotchas

- **Safety defaults are off.** Feature and behaviour flags must default to `False`. Sending needs `notify.enabled: true` (it must be a real YAML boolean) and, for `run`, also `run.send` or `--send`. `cmd_run` refuses `send` when notify is disabled. Keep new flags off by default.
- `.config.yaml` is tracked and holds example values. `tests/test_run.py::test_project_config_has_a_run_section_with_thresholds` reads it, so keep a valid `run.thresholds` section in it. Real credentials go in `.env` (`tege_api_id`, `tege_api_hash`; see `.env.example`). `.sessions/`, `.state/` and `logs/` are gitignored.
- Threshold keys in `run.thresholds` must name enabled questions in `classifier/questions.yaml`; `validate_thresholds` checks this at startup.
- Model repos run remote code (`trust_remote_code`). Revisions in `models.yaml` are pinned on purpose and `load_models` rejects anything but a full 40-character commit hash. Do not unpin them or bump them without reviewing that revision's code.
- Text from chats is untrusted: strip control characters (`textutil.printable`) before printing it to a terminal, and escape it before sending it back to Telegram.
- In the CLI, stdout carries results (including `--json`); progress, worker output and logs go to stderr.
