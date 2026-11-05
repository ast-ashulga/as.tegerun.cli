#!/usr/bin/env python3
"""Воркер Liquid d1 (d1-2B, d1-3B): читает requests.json, пишет out.json.

Запускается из tegerun/classify.py внутри .venv-liquid. Вручную:
  .venv-liquid/bin/python workers/d1_worker.py --config '{"repo":"LiquidAI/d1-3B"}' \
      --requests requests.json --out out.json

requests.json: [{"rid": "...", "state": "...", "questions": {id: {type, instructions, criteria}}}]
out.json:      {"meta": {...}, "results": [{"rid", "ms", "input_tokens", "answers"} | {"rid", "error"}]}
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="JSON: repo, revision, device, dtype, online")
    p.add_argument("--requests", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config)

    if not cfg.get("online"):
        os.environ["HF_HUB_OFFLINE"] = "1"  # до импорта transformers

    try:
        import torch
        import transformers
        from transformers import AutoModel
    except ImportError as e:
        sys.exit(f"Нет модуля {e.name!r}: запустите ./setup.sh (интерпретатор: {sys.executable})")

    major_minor = tuple(int(x) for x in transformers.__version__.split(".")[:2])
    if major_minor < (5, 19):
        sys.exit(f"Для d1-2B нужен transformers>=5.19, найден {transformers.__version__}. ./setup.sh обновит.")

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = cfg.get("dtype", "auto")
    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtypes.get(dtype_name) or (torch.float32 if device == "cpu" else torch.bfloat16)

    repo, revision = cfg["repo"], cfg.get("revision")
    print(f"[d1] загрузка {repo}@{(revision or 'main')[:8]} на {device}, "
          f"{str(dtype).split('.')[-1]}{'' if cfg.get('online') else ', офлайн'} ...", flush=True)
    t0 = time.perf_counter()
    try:
        model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=dtype, revision=revision)
    except OSError as e:
        sys.exit(f"Не удалось загрузить {repo}: {e}\nВеса нет в кэше? Выполните: ./setup.sh --download <имя модели>")
    model = model.to(device).eval()
    load_s = time.perf_counter() - t0
    print(f"[d1] готово за {load_s:.1f} с", flush=True)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    results = []
    if requests:  # прогрев: первый вызов с новой формой дороже (ядра/компиляция)
        with torch.no_grad():
            model.system_one(requests[0]["state"], requests[0]["questions"])
        sync()

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
        except Exception as e:  # noqa: BLE001 — один плохой запрос не должен ронять весь прогон
            results.append({"rid": req["rid"], "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0 or i == len(requests):
            print(f"[d1] {i}/{len(requests)}", flush=True)

    meta = {"repo": repo, "revision": revision, "device": device,
            "dtype": str(dtype).split(".")[-1], "load_s": load_s,
            "transformers": transformers.__version__, "torch": torch.__version__}
    args.out.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, default=float),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
