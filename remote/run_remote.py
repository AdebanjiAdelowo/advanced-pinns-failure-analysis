"""Remote DeepXDE Poisson experiment launcher.

Run on Kaggle, Colab, or Lightning AI after installing requirements-pinn.txt.
Writes provenance and held-out metrics to a unique output directory.
"""
import argparse
import json
import os
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.poisson_baseline import exact_solution, relative_l2, train_pinn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 0, 1, 2, 3])
    parser.add_argument("--output", default="results/remote")
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error("--iterations must be positive")
    import torch
    import deepxde as dde

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = Path(args.output) / stamp
    directory.mkdir(parents=True, exist_ok=False)
    x = np.linspace(0, 1, 501)[:, None]
    provenance = {
        "utc": stamp,
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "deepxde": dde.__version__,
        "numpy": np.__version__,
        "device_available": {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()},
        "backend": os.environ.get("DDE_BACKEND"),
        "iterations": args.iterations,
        "seeds": args.seeds,
    }
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    results = []
    for seed in args.seeds:
        start = time.perf_counter()
        model = train_pinn(iterations=args.iterations, seed=seed)
        pred = model.predict(x).reshape(-1)
        results.append({
            "seed": seed,
            "relative_l2": relative_l2(pred, exact_solution(x)),
            "boundary_max_abs": float(np.max(np.abs(model.predict(np.array([[0.0], [1.0]]))))),
            "wall_seconds": time.perf_counter() - start,
        })
        (directory / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(directory)


if __name__ == "__main__":
    main()
