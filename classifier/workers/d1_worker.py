#!/usr/bin/env python3
"""Liquid d1 worker (d1-2B, d1-3B): reads requests.json, writes out.json.

Run from tegerun/classify.py inside .venv-liquid. Manually:
  .venv-liquid/bin/python workers/d1_worker.py --config '{"repo":"LiquidAI/d1-3B"}' \
      --requests requests.json --out out.json

requests.json: [{"rid": "...", "state": "...", "questions": {id: {type, instructions, criteria}}}]
out.json:      {"meta": {...}, "results": [{"rid", "ms", "input_tokens", "answers"} | {"rid", "error"}]}
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def _quiet_libraries() -> None:
    """Drop torch's note about output redirects (macOS): it reads like a warning but is not one.
    Real warnings from the libraries stay."""
    import logging

    logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON: repo, revision, device, dtype, online")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config)

    if not cfg.get("online"):
        os.environ["HF_HUB_OFFLINE"] = "1"  # before importing transformers

    _quiet_libraries()  # before the imports below
    try:
        import torch
        import transformers
        from transformers import AutoModel
        from transformers.utils import logging as hf_logging

        hf_logging.disable_progress_bar()  # the "Loading weights" bar
    except ImportError as e:
        sys.exit(f"No module {e.name!r}: run ./setup.sh (interpreter: {sys.executable})")

    parts = re.match(r"(\d+)\.(\d+)", transformers.__version__)  # tolerates "5.19.0.dev0", "5.20rc1"
    if not parts or (int(parts[1]), int(parts[2])) < (5, 19):
        sys.exit(f"d1 models need transformers>=5.19, found {transformers.__version__}. ./setup.sh will update.")

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = cfg.get("dtype", "auto")
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtypes.get(dtype_name) or (torch.float32 if device == "cpu" else torch.bfloat16)

    repo, revision = cfg["repo"], cfg.get("revision")
    print(f"[d1] loading {repo}@{(revision or 'main')[:8]} on {device}, "
          f"{str(dtype).split('.')[-1]}{'' if cfg.get('online') else ', offline'} ...", flush=True)
    t0 = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=dtype, revision=revision)
    except OSError as e:
        sys.exit(f"Failed to load {repo}: {e}\nWeights not in cache? Run: ./setup.sh --download <model name>")
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    print(f"[d1] ready in {load_s:.1f} s", flush=True)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    results = []
    meta = {"repo": repo, "revision": revision, "device": device,
            "dtype": str(dtype).split(".")[-1], "load_s": load_s,
            "transformers": transformers.__version__, "torch": torch.__version__}

    def save() -> None:
        """Atomic, so the caller can use the partial file if this process is killed (OOM, timeout)."""
        tmp = args.out.with_name(args.out.name + ".tmp")
        tmp.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, default=float),
                       encoding="utf-8")
        os.replace(tmp, args.out)

    if requests:  # warm-up: first call with new shape is more expensive (kernels/compilation)
        try:
            with torch.no_grad():
                model.system_one(requests[0]["state"], requests[0]["questions"])
            sync()
        except Exception as e:  # noqa: BLE001 - the real call below reports this request's error
            print(f"[d1] warm-up failed: {type(e).__name__}: {e}", flush=True)

    for i, req in enumerate(requests, 1):
        try:
            sync()
            t = time.perf_counter()
            with torch.no_grad():
                res = model.system_one(req["state"], req["questions"])
            sync()
            ms = (time.perf_counter() - t) * 1000
            results.append({"rid": req["rid"], "ms": ms,
                            "input_tokens": res["usage"]["input_tokens"], "answers": res["answers"]})
        except Exception as e:  # noqa: BLE001 — one bad request must not ruin the whole run
            results.append({"rid": req["rid"], "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0 or i == len(requests):
            print(f"[d1] {i}/{len(requests)}", flush=True)
            save()
    save()  # also covers an empty request list


if __name__ == "__main__":
    main()
