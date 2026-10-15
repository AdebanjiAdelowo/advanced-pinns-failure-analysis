# Claude Code handoff — Advanced PINNs ablation

Code is prepared. No full GPU training has been executed or benchmark improvement established.

## Lightweight checks

From repository root run `python -m pytest -q tests/test_manufactured_validation.py tests/test_pinn_torch_features.py tests/test_pinn_ablation_runner.py`. Inspect latest GitHub Actions before proceeding.

## Remote GPU smoke tests

Use an authorized Kaggle, Colab or Lightning AI CUDA environment. Install the repository's dependencies, set `DDE_BACKEND=pytorch`, and run:

```bash
python -m remote.smoke_pinn_ablation --problem burgers --iterations 2
python -m remote.smoke_pinn_ablation --problem helmholtz --iterations 2
```

Resolve any DeepXDE feature-transform, adaptive-anchor, CUDA placement or evaluation failure before scheduling full runs. The smoke gate is a runtime check, not convergence evidence.

## Full remote ablations

```bash
python -m remote.run_pinn_ablation --problem burgers --iterations 10000 --seeds 42 0 1 2 3 --output results/burgers_ablation.json
python -m remote.run_pinn_ablation --problem helmholtz --iterations 10000 --seeds 42 0 1 2 3 --output results/helmholtz_ablation.json
```

Preserve raw results, failed seeds, commit hash, dependency versions, runtime and GPU type. Compare baseline, Fourier, adaptive and combined variants by per-seed relative L2, wall time and parameter count. Equal optimizer iterations are not equal compute budgets. The adaptive stage adds candidate-residual evaluation; Fourier features change parameter counts. Do not claim improvements or update scientific-audit classifications without validated measurements. Never run heavy training locally.
