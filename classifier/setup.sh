#!/usr/bin/env bash
# Prepare stand: two environments (different transformers versions) and, by flag, model weights.
#
#   ./setup.sh                                  # create .venv-liquid and .venv-pplx, install dependencies
#   ./setup.sh --download d1-2B pplx-flash      # + download weights of selected models (names from models.yaml)
#   ./setup.sh --code-only d1-2B pplx-flash     # download only code/configs (a few MB) — to read
#                                               # executable model code before downloading weights
#
# Weight sizes: d1-2B ~4 GB, d1-3B ~6 GB (already in cache), pplx-flash ~9 GB (bf16).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
MODE=""
NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) MODE="full"; shift ;;
    --code-only) MODE="code"; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "Unknown argument: $1" >&2; exit 2 ;;
    *) NAMES+=("$1"); shift ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
command -v uv >/dev/null || { echo "Need uv (https://docs.astral.sh/uv/)" >&2; exit 1; }

# d1-2B/d1-3B: transformers>=5.19 (d1-2B card requirement), any fresh torch.
if [[ ! -x .venv-liquid/bin/python ]]; then
  log "Creating .venv-liquid (Python 3.12)"
  uv venv --python 3.12 .venv-liquid
fi
log "Installing dependencies in .venv-liquid"
uv pip install --python .venv-liquid/bin/python "transformers>=5.19" torch torchvision pillow pyyaml huggingface_hub

# pplx-decider: exact versions from model requirements.txt (Python >= 3.12).
if [[ ! -x .venv-pplx/bin/python ]]; then
  log "Creating .venv-pplx (Python 3.12)"
  uv venv --python 3.12 .venv-pplx
fi
log "Installing dependencies in .venv-pplx (versions pinned per model requirements.txt)"
uv pip install --python .venv-pplx/bin/python \
  torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 safetensors==0.8.0 pillow==12.3.0 huggingface-hub==1.31.0

log "Verification"
.venv-liquid/bin/python -I - <<'PY'
import torch, transformers
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f".venv-liquid: transformers {transformers.__version__}, torch {torch.__version__}, device {dev}")
PY
.venv-pplx/bin/python -I - <<'PY'
import torch, transformers
print(f".venv-pplx:   transformers {transformers.__version__}, torch {torch.__version__}")
PY

if [[ -n "$MODE" ]]; then
  [[ ${#NAMES[@]} -gt 0 ]] || { echo "Specify model names: d1-2B d1-3B pplx-flash" >&2; exit 2; }
  log "Download: ${NAMES[*]} ($MODE)"
  if [[ "$MODE" == "code" ]]; then
    .venv-liquid/bin/python fetch.py --code-only "${NAMES[@]}"
  else
    .venv-liquid/bin/python fetch.py "${NAMES[@]}"
  fi
fi

log "Done. Next: uv run tege doctor   # what is still missing; then: uv run tege run"
echo "Reminder: models run code from their repositories (trust_remote_code / source/)."
echo "Revisions are pinned in models.yaml; to read the code before downloading weights, from the project root:"
echo "  ./classifier/setup.sh --code-only <name>"
