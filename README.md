# tege run

Read-only Telegram triage CLI that scans whitelisted chats via [fast-mcp-telegram](https://pypi.org/project/fast-mcp-telegram/), classifies messages with local System-1 models, and forwards matching messages to a designated chat.

- **Narrow by construction**: reads only whitelisted chats (`get_messages`) and sends only to the one configured notify chat (`send_message`). Each MCP session is limited to the tools it needs, so the server's other tools (edit, raw MTProto, send-to-phone) cannot be called. Note the session itself is your full Telegram account.
- **Private**: All data stays local; session stored in `.sessions/` (gitignored)
- **macOS-focused** for now (Linux/Windows planned)
- **Easy to configure**: Simple YAML config + `.env` for credentials

## How it works

```
whitelisted chats --read--> local classifier models --scores--> thresholds --pass--> one notify chat
```

You choose the chats to watch and the questions to ask. Job vacancies are only the built-in example ("is this a vacancy, and does it fit my profile?"); you can describe any task in plain language, see [Make it yours](#make-it-yours). Only messages that pass **all** thresholds are forwarded. Nothing is sent until you turn it on.

## Requirements

- macOS, [`uv`](https://docs.astral.sh/uv/), git
- Telegram API credentials (`api_id` / `api_hash`) from [my.telegram.org](https://my.telegram.org)
- Disk space for the classifier model (`d1-3B` is about 6 GB; `d1-2B` about 4 GB)

## Quick Start

All commands run from the repo root. Use `uv run tege ...` (a bare `tege` works only with the venv activated).

### The short way
```bash
git clone https://github.com/ast-ashulga/as.tegerun.cli.git && cd as.tegerun.cli
uv sync
uv run tege run
```
`tege run` is one pass in preview mode: it prints what it would forward, sends nothing and saves nothing. On a fresh checkout it first works out what is missing and prints a numbered to-do list with the exact commands (credentials, Telegram login, your chats, classifier setup and model download), plus recommendations such as rewriting the example `skills:` profile. Do the items, run it again, repeat until it completes a pass. After a pass it explains the result (why nothing matched, how to start sending). `uv run tege doctor` prints the same list without running anything.

The steps below are that list in detail.

### 1) Install
```bash
uv sync
uv run tege --help        # sanity check
```

### 2) Set up the classifier (one time)
```bash
./classifier/setup.sh                    # creates the model venvs
./classifier/setup.sh --download d1-3B   # downloads the model weights
```
Without this step `tege classify` and `tege run` cannot work (`tege run` will tell you). Details: [classifier/README.md](classifier/README.md).

### 3) Add your Telegram credentials
Copy `.env.example` to `.env` and fill it in:
```env
tege_api_id=YOUR_TELEGRAM_API_ID
tege_api_hash=YOUR_TELEGRAM_API_HASH
```

### 4) Log in to Telegram (one time, interactive)
```bash
uv run tege setup
```
Enter your phone number, the login code and the 2FA password if you have one. The session is saved to `.sessions/telegram.session`.

### 5) Pick your chats
Edit `.config.yaml`: replace the example `whitelist` entries with your own chats (numeric id or `@username`). The other sections have working defaults; the main ones are:

```yaml
whitelist:
  - id: "1234567890"
    name: "Jobs Channel"
    description: "IT vacancies"

classify:
  model: d1-3B                # model from classifier/models.yaml

notify:
  enabled: false              # sending is off until you set true
  chat_id: "-1001234567890"   # where matches are forwarded
  topic_id: 123               # optional forum topic

run:
  every: 30m                  # pause between passes with --loop
  first_window: 24h           # how far back the first pass reads
  send: false                 # set true (and notify.enabled) to really send
  max_per_run: 10
  thresholds:                 # a message must pass ALL of these
    is_vacancy: 0.8
    match_ok: 0.55
    match: 3
```

To change *what* counts as a match (the built-in questions are tuned for job vacancies), see [Make it yours](#make-it-yours).

### 6) Check it step by step (optional)
Each step is safe: nothing is sent and no state is saved until you enable sending.
```bash
uv run tege read --limit 5                  # 1. can we read your chats?
uv run tege classify --limit 5              # 2. do the models score them?
uv run tege run                             # 3. full pass, shows what would be forwarded
uv run tege notify "Test" --dry-run         # 4. check the notify target
```
When the output looks right, set `notify.enabled: true` and `run.send: true` in `.config.yaml`, then:
```bash
uv run tege run          # one real pass (use it from cron/launchd)
uv run tege run --loop   # or keep running every run.every
```

## Make it yours

The classifier does not know about vacancies; it answers whatever questions you give it in `classifier/questions.yaml`. To use tege run for another task, change two things together:

1. In `classifier/questions.yaml` replace `skills:` (a free-text description of what *you* want) and the `questions:` block.
2. In `.config.yaml` set `run.thresholds` to your question ids (startup fails if a threshold names a question that does not exist).

Question types:
- `noul`: yes/no, answered as a probability. Can have a threshold (`0.8` means 80% sure).
- `score`: a level from your `criteria` list, counted from 0. Can have a threshold (`3` means the 4th level or higher).
- `choice`: pick one option. It has no number, so it cannot be a threshold, but the answer is shown in the forwarded message.
- `needs_skills: true` also shows the model your `skills:` text for that question. Use it for "does this fit *me*?", and leave it off for plain "what is this?" questions.

Each enabled question costs model time, so keep the list short: usually one "is this the kind of message I mean?" question plus one "does it fit me?" question. Also reword "Candidate profile" in `state_template_with_skills` (same file) to match your task.

### Example 1: a flat to rent
```yaml
# classifier/questions.yaml
skills: |
  Looking for a 1-2 bedroom flat to rent in Lisbon, up to 1200 EUR/month.
  Pets must be allowed, lease of 6 months or more. Not interested in shared rooms.
questions:
  is_rental_offer:
    type: noul
    instructions: Is this message an offer of a flat or house for long-term rent?
  fits_me:
    type: noul
    instructions: Does this listing meet the requirements and budget in the profile?
    needs_skills: true
```
```yaml
# .config.yaml
run:
  thresholds: {is_rental_offer: 0.8, fits_me: 0.6}
```

### Example 2: bug reports in your product's community chat
No `skills:` needed; the questions do not depend on who is reading. `kind` is only shown in the forwarded message.
```yaml
# classifier/questions.yaml
questions:
  is_bug_report:
    type: noul
    instructions: Is the author reporting something that is broken or not working in a product?
  severity:
    type: score
    instructions: How serious is the problem for the author?
    criteria:
      - Not a problem; a question or a comment
      - Minor annoyance or cosmetic issue
      - A feature does not work, a workaround exists
      - A feature does not work, no workaround
      - Data loss, security issue or full outage
  kind:
    type: choice
    instructions: What kind of message is this?
    criteria:
      bug: A defect report
      question: A how-to question
      request: A feature request
      other: Anything else
```
```yaml
# .config.yaml
run:
  thresholds: {is_bug_report: 0.7, severity: 3}   # severity 3 = "no workaround" or worse
```

### Example 3: a technical news digest
Forward only substantial posts on your topics from busy channels.
```yaml
# classifier/questions.yaml
skills: |
  Interested in: LLM inference optimization, GPU kernels, open-weight model releases.
  Not interested in: funding news, launches without technical detail, hot takes.
questions:
  is_technical_news:
    type: noul
    instructions: Does this post report a concrete technical development (release, paper, benchmark, tool) rather than opinion or promotion?
  relevant:
    type: score
    instructions: How relevant is this post to the interests in the profile?
    criteria:
      - Unrelated
      - Loosely related
      - Related
      - Directly about my interests
    needs_skills: true
```
```yaml
# .config.yaml
run:
  thresholds: {is_technical_news: 0.7, relevant: 2}
```

### Tuning the thresholds
Run `uv run tege classify --limit 20` on real messages: it prints every answer, so you can see which scores the messages you want get and which scores the ones you do not want get. Put each threshold between the two. After a `tege run` pass with no matches, the hint lists the best scores it saw against your thresholds.

## Command reference

```bash
uv run tege read --chat 1234567890 --since 2h --json   # filter one chat / window, JSON output
uv run tege classify --model d1-2B --dry-run           # print worker requests, load no model
uv run tege classify --input msgs.jsonl                # classify a JSONL file instead of Telegram
echo "Hello" | uv run tege notify                      # send text from stdin
uv run tege run --every 1h --no-send                   # repeat every hour (--every implies --loop), preview only
uv run tege doctor                                     # what is missing for `tege run`, and how to fix it
```

A preview (`send: false` / `--no-send`) never touches `.state/`, so looking at matches does not consume them.
Only one `tege run` can be active at a time (`.state/run.lock`).

## Configuration Files

- `.env` - Telegram API credentials (never commit)
- `.config.yaml` - Whitelist, classify, notify, run settings
- `classifier/questions.yaml` - Questions and candidate profile/skills for classification
- `classifier/models.yaml` - Model registry (d1-2B, d1-3B, pplx-flash)
- `classifier/workers/*_worker.py` - Model workers running in isolated venvs (no secrets in their environment, killed after `classify.timeout` seconds, partial results are kept)
- `TEGE_HOME` (env) - use another directory for config, `.env`, session and state (needed for a non-editable install)
- `.sessions/` - Telegram session (created by `setup`, gitignored)
- `.state/` - Run state tracking sent messages and last success (gitignored)
- `logs/` - MCP server logs (gitignored)

## Classification

Classifiers return probabilities/scores, not text. Questions have types:
- `noul` - yes/no probability `p`
- `choice` - pick from criteria
- `score` - numeric score from criteria levels

Messages are sent only if they pass **all** thresholds defined in `run.thresholds`.

## Architecture

```
tegerun/
├── cli.py        # CLI commands and argument parsing
├── config.py     # Configuration loading/validation
├── mcp_client.py # MCP stdio connection to fast-mcp-telegram; per-session tool allowlist + timeouts
├── reader.py     # Fetch messages from Telegram via MCP (whitelist enforced here, pagination)
├── textutil.py   # Strip terminal control characters from untrusted text
├── classify.py   # Classify messages using local model workers
├── runner.py     # Orchestrate run passes (read->classify->filter->send)
├── runstate.py   # Persist sent messages and last success
└── notifier.py   # Send notifications via MCP
```

## Notes

- macOS support only currently
- Uses `uv` for environment/dependency management
- Model workers require setup via `classifier/setup.sh` (see classifier docs)
- Session must exist before `read/classify/run/notify` (run `tege setup` first)
