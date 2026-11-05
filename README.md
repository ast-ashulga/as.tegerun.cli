# tegerun

Read-only Telegram triage CLI that scans whitelisted chats via [fast-mcp-telegram](https://pypi.org/project/fast-mcp-telegram/), classifies messages with local System-1 models, and forwards matching messages to a designated chat.

- **Read-only**: Only uses `get_messages` / `send_message` through MCP; no edits or deletions
- **Private**: All data stays local; session stored in `.sessions/` (gitignored)
- **macOS-focused** for now (Linux/Windows planned)
- **Easy to configure**: Simple YAML config + `.env` for credentials

## Quick Start

### 1) Install dependencies
```bash
uv sync
```

### 2) Configure credentials
Create `.env`:
```env
tege_api_id=YOUR_TELEGRAM_API_ID
tege_api_hash=YOUR_TELEGRAM_API_HASH
```
Get credentials from [my.telegram.org](https://my.telegram.org).

### 3) Configure chats and behavior
Copy/edit `.config.yaml` in repo root:

```yaml
whitelist:
  - id: "1234567890"
    name: "Jobs Channel"
    description: "IT vacancies"

classify:
  classifier_dir: classifier  # default
  model: d1-3B                # default model from classifier/models.yaml

notify:
  enabled: true
  chat_id: "-1001234567890"
  topic_id: 123              # optional forum topic
  # reply_to: 456            # optional reply target

run:
  every: 30m
  first_window: 24h
  limit: 100
  send: true
  max_per_run: 10
  thresholds:
    is_vacancy: 0.8
    match_ok: 0.55
    match: 3
```

### 4) One-time Telegram login
```bash
tege setup  # or: uv run tege setup
```
Follow prompts (phone number, code, 2FA). Session saved to `.sessions/telegram.session`.

### 5) Use the CLI

```bash
# Read recent messages from whitelisted chats
tege read --limit 20
tege read --chat 1234567890 --since 2h --json

# Classify messages (uses classifier/ questions and models)
tege classify --limit 20
tege classify --model d1-2B --dry-run

# Send a notification manually
echo "Hello" | tege notify
tege notify "Test message" --dry-run

# Run continuously: read -> classify -> filter by thresholds -> notify
tege run --once        # single pass (for cron/launchd)
tege run               # repeat every run.every
tege run --every 1h --no-send  # override interval, dry send
```

## Configuration Files

- `.env` - Telegram API credentials (never commit)
- `.config.yaml` - Whitelist, classify, notify, run settings
- `classifier/questions.yaml` - Questions and candidate profile/skills for classification
- `classifier/models.yaml` - Model registry (d1-2B, d1-3B, pplx-flash)
- `classifier/*_worker.py` - Model workers running in isolated venvs
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
├── mcp_client.py # Shared MCP stdio connection to fast-mcp-telegram
├── reader.py     # Fetch messages from Telegram via MCP
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
