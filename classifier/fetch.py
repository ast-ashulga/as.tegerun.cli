#!/usr/bin/env python3
"""Download models from models.yaml into the Hugging Face cache (pinned revision).

  ./setup.sh --download d1-2B pplx-flash      # full weights
  .venv-liquid/bin/python fetch.py --code-only d1-2B pplx-flash   # code/configs only (a few MB) — to inspect

--code-only is needed to read the model executable code (trust_remote_code / source/) before downloading weights.
Files will appear in cache: ~/.cache/huggingface/hub/models--<org>--<name>/snapshots/<revision>/
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
    p.add_argument("names", nargs="+", help="names from models.yaml (d1-2B, d1-3B, pplx-flash)")
    p.add_argument("--code-only", action="store_true", help="code and configs only, no weights")
    args = p.parse_args()

    models = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))["models"]
    for name in args.names:
        if name not in models:
            sys.exit(f"Unknown model {name!r}. Available: {', '.join(models)}")
        m = models[name]
        print(f"==> {name}: {m['repo']}@{m['revision'][:8]}{' (code only)' if args.code_only else ''}", flush=True)
        path = snapshot_download(
            m["repo"], revision=m["revision"],
            allow_patterns=CODE_PATTERNS if args.code_only else None,
            ignore_patterns=["*.safetensors", "*.bin", "*.gguf", "*.onnx"] if args.code_only else None,
        )
        print(f"    {path}")


if __name__ == "__main__":
    main()
