"""Run from repository root: DDE_BACKEND=pytorch python -m experiments.run_poisson"""
import json
from pathlib import Path
import numpy as np
from src.poisson_baseline import exact_solution, relative_l2, train_pinn


def main():
    model = train_pinn()
    x = np.linspace(0, 1, 501)[:, None]
    pred = model.predict(x).reshape(-1)
    metric = {"relative_l2": relative_l2(pred, exact_solution(x).reshape(-1)), "evaluation_points": len(x)}
    out = Path("results/poisson_baseline.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metric, indent=2) + "\n")
    print(json.dumps(metric, indent=2))


if __name__ == "__main__":
    main()
