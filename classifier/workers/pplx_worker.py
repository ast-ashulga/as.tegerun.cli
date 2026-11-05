#!/usr/bin/env python3
"""Воркер perplexity-ai/pplx-decider-flash-v1.2: читает requests.json, пишет out.json.

Запускается из tegerun/classify.py внутри .venv-pplx (Python 3.12, transformers==5.17.0, как в requirements.txt модели).
Формат requests.json / out.json — как у d1_worker.py.

Заметки:
 • Код модели лежит в репозитории (source/src/pplx_decider) и подключается через sys.path — без pip install.
 • Оригинальный код рассчитан на CUDA. На Mac берётся референсный путь (fast=False) на mps/cpu в fp32
   (~18 ГБ). --dtype bf16 приводит бэкбон и readout к bf16: быстрее и вдвое меньше памяти, но это
   отклонение от оригинала (проверяйте качество).
 • Вероятности калиброваны температурой из decision_config.json (применяется внутри decide()).
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
        os.environ["HF_HUB_OFFLINE"] = "1"

    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
    except ImportError as e:
        sys.exit(f"Нет модуля {e.name!r}: запустите ./setup.sh (интерпретатор: {sys.executable})")

    repo, revision = cfg["repo"], cfg.get("revision")
    try:
        checkpoint = Path(snapshot_download(repo, revision=revision, local_files_only=not cfg.get("online")))
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Не удалось найти {repo} в кэше: {e}\nВыполните: ./setup.sh --download pplx-flash")
    if not (checkpoint / "readout.safetensors").exists() or not list(checkpoint.glob("model-*.safetensors")):
        sys.exit(f"В {checkpoint} нет весов. Выполните: ./setup.sh --download pplx-flash")
    sys.path.insert(0, str(checkpoint / "source" / "src"))
    from pplx_decider import Decider  # код из репозитория модели (ревизия зафиксирована в models.yaml)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dtype_name = cfg.get("dtype", "auto")

    print(f"[pplx] загрузка {repo}@{(revision or 'main')[:8]} на {device}"
          f"{'' if cfg.get('online') else ' (офлайн)'} ...", flush=True)
    t0 = time.perf_counter()
    decider = Decider(checkpoint, device="cpu" if device == "mps" else device, fast=False)
    # Decider грузит в fp32 на не-CUDA; приводим dtype и переносим на устройство.
    if dtype_name == "bf16":
        decider.backbone.to(torch.bfloat16)
        decider.readout.to(torch.bfloat16)
    elif dtype_name == "fp16":
        decider.backbone.to(torch.float16)
        decider.readout.to(torch.float16)
    elif dtype_name == "auto" and device == "cuda":
        pass  # Decider уже выбрал bf16 на CUDA
    if device == "mps":
        decider.backbone.to("mps")
        decider.readout.to("mps")
        decider.device = torch.device("mps")
    load_s = time.perf_counter() - t0
    actual_dtype = str(next(decider.backbone.parameters()).dtype).split(".")[-1]
    print(f"[pplx] готово за {load_s:.1f} с (dtype={actual_dtype})", flush=True)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()
        elif device == "mps":
            torch.mps.synchronize()

    requests = json.loads(args.requests.read_text(encoding="utf-8"))
    results = []
    if requests:
        decider.decide({"state": requests[0]["state"], "questions": requests[0]["questions"]})
        sync()

    for i, req in enumerate(requests, 1):
        try:
            sync()
            t = time.perf_counter()
            res = decider.decide({"state": req["state"], "questions": req["questions"]})
            sync()
            ms = (time.perf_counter() - t) * 1000
            results.append({"rid": req["rid"], "ms": ms,
                            "input_tokens": res["usage"]["input_tokens"], "answers": res["answers"]})
        except Exception as e:  # noqa: BLE001
            results.append({"rid": req["rid"], "error": f"{type(e).__name__}: {e}"})
        if i % 10 == 0 or i == len(requests):
            print(f"[pplx] {i}/{len(requests)}", flush=True)

    meta = {"repo": repo, "revision": revision, "device": device, "dtype": actual_dtype,
            "load_s": load_s, "temperature": decider.temperature,
            "transformers": transformers.__version__, "torch": torch.__version__}
    args.out.write_text(json.dumps({"meta": meta, "results": results}, ensure_ascii=False, default=float),
                        encoding="utf-8")


if __name__ == "__main__":
    main()
