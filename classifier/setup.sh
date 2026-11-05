#!/usr/bin/env bash
# Подготовка стенда: два окружения (разные версии transformers) и, по флагу, веса моделей.
#
#   ./setup.sh                                  # создать .venv-liquid и .venv-pplx, поставить зависимости
#   ./setup.sh --download d1-2B pplx-flash      # + скачать веса выбранных моделей (имена из models.yaml)
#   ./setup.sh --code-only d1-2B pplx-flash     # скачать только код/конфиги (несколько МБ) — чтобы прочитать
#                                               # исполняемый код модели до скачивания весов
#
# Размеры весов: d1-2B ~4 ГБ, d1-3B ~6 ГБ (уже в кэше), pplx-flash ~9 ГБ (bf16).
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
MODE=""
NAMES=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --download) MODE="full"; shift ;;
    --code-only) MODE="code"; shift ;;
    -h|--help) sed -n '2,10p' "${BASH_SOURCE[0]}"; exit 0 ;;
    -*) echo "Неизвестный аргумент: $1" >&2; exit 2 ;;
    *) NAMES+=("$1"); shift ;;
  esac
done

log() { printf '\n==> %s\n' "$*"; }
command -v uv >/dev/null || { echo "Нужен uv (https://docs.astral.sh/uv/)" >&2; exit 1; }

# d1-2B/d1-3B: transformers>=5.19 (требование карточки d1-2B), любой свежий torch.
if [[ ! -x .venv-liquid/bin/python ]]; then
  log "Создаю .venv-liquid (Python 3.12)"
  uv venv --python 3.12 .venv-liquid
fi
log "Ставлю зависимости в .venv-liquid"
uv pip install --python .venv-liquid/bin/python "transformers>=5.19" torch torchvision pillow pyyaml huggingface_hub

# pplx-decider: точные версии из requirements.txt модели (Python >= 3.12).
if [[ ! -x .venv-pplx/bin/python ]]; then
  log "Создаю .venv-pplx (Python 3.12)"
  uv venv --python 3.12 .venv-pplx
fi
log "Ставлю зависимости в .venv-pplx (версии зафиксированы по requirements.txt модели)"
uv pip install --python .venv-pplx/bin/python \
  torch==2.14.0 torchvision==0.29.0 transformers==5.17.0 safetensors==0.8.0 pillow==12.3.0 huggingface-hub==1.31.0

log "Проверка"
.venv-liquid/bin/python -I - <<'PY'
import torch, transformers
dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f".venv-liquid: transformers {transformers.__version__}, torch {torch.__version__}, устройство {dev}")
PY
.venv-pplx/bin/python -I - <<'PY'
import torch, transformers
print(f".venv-pplx:   transformers {transformers.__version__}, torch {torch.__version__}")
PY

if [[ -n "$MODE" ]]; then
  [[ ${#NAMES[@]} -gt 0 ]] || { echo "Укажите имена моделей: d1-2B d1-3B pplx-flash" >&2; exit 2; }
  log "Загрузка: ${NAMES[*]} ($MODE)"
  if [[ "$MODE" == "code" ]]; then
    .venv-liquid/bin/python fetch.py --code-only "${NAMES[@]}"
  else
    .venv-liquid/bin/python fetch.py "${NAMES[@]}"
  fi
fi

log "Готово. Дальше: uv run tege classify --dry-run --input messages.jsonl"
echo "Напоминание: модели запускают код из своих репозиториев (trust_remote_code / source/)."
echo "Ревизии зафиксированы в models.yaml; прочитать код: ./setup.sh --code-only <имя>."
