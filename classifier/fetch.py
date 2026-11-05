#!/usr/bin/env python3
"""Скачивание моделей из models.yaml в кэш Hugging Face (зафиксированная ревизия).

  ./setup.sh --download d1-2B pplx-flash      # веса целиком
  .venv-liquid/bin/python fetch.py --code-only d1-2B pplx-flash   # только код/конфиги (мало МБ) — чтобы просмотреть

--code-only нужен, чтобы прочитать исполняемый код модели (trust_remote_code / source/) до скачивания весов.
Файлы появятся в кэше: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent
CODE_PATTERNS = ["*.py", "**/*.py", "*.json", "*.md", "*.txt", "*.jinja", "LICENSE", "NOTICE", "source/**"]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("names", nargs="+", help="имена из models.yaml (d1-2B, d1-3B, pplx-flash)")
    p.add_argument("--code-only", action="store_true", help="только код и конфиги, без весов")
    args = p.parse_args()

    models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))["models"]
    for name in args.names:
        if name not in models:
            sys.exit(f"Неизвестная модель {name!r}. Есть: {', '.join(models)}")
        m = models[name]
        print(f"==> {name}: {m['repo']}@{m['revision'][:8]}{' (только код)' if args.code_only else ''}", flush=True)
        path = snapshot_download(
            m["repo"], revision=m["revision"],
            allow_patterns=CODE_PATTERNS if args.code_only else None,
            ignore_patterns=["*.safetensors", "*.bin", "*.gguf", "*.onnx"] if args.code_only else None,
        )
        print(f"    {path}")


if __name__ == "__main__":
    main()
