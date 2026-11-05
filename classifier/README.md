# classifier — models and questions for `tegerun classify`

This directory contains everything needed for local message classification with System-1 models.
All paths are relative to the project; nothing outside this repo is required.

| File/Dir | Description |
|---|---|
| `questions.yaml` | Questions (`is_vacancy`, `category`, `match`, `match_ok`, …), optional candidate profile `skills:`, and state templates. Disable a question with `enabled: false`. |
| `models.yaml` | Model registry: HF repo, pinned revision, grouping (`joint`/`single`), device, dtype. |
| `workers/` | Model workers (one per family). Each runs in its own venv, reads `requests.json`, writes `out.json`. |
| `setup.sh`, `fetch.py` | Create venvs and download model weights. |
| `.venv-liquid/`, `.venv-pplx/` | Virtual environments created by `setup.sh` (gitignored in practice). |

## First-time setup

```bash
./classifier/setup.sh                # create environments
./classifier/setup.sh --download d1-3B  # download weights (~6 GB) to HF cache
uv run tegerun classify --since 24h --limit 50
```

Models run offline by default (HuggingFace cache only). Pass `--online` to allow network access.
Default classifier path and model are controlled by the `classify:` section in `.config.yaml`.
