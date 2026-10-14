# PINN failure-mode study protocol

Prepared utilities:
- `fourier_features`: deterministic sine/cosine features at prescribed frequencies.
- `select_largest_residual`: deterministic top-|residual| selection from candidate points.

Planned controlled factors: soft vs hard Dirichlet conditions, standard tanh vs Fourier-feature input, uniform vs residual-adaptive collocation, and PDE frequency/viscosity regimes.

For each comparison, freeze PDE, exact solution, held-out grid, initialization seeds, training budget, network capacity where feasible, and evaluation metrics. Log added points and residual candidate distribution for adaptive sampling. Assess held-out relative L2, maximum boundary violation, PDE residual, runtime, failures, and multi-seed dispersion.

**Status:** Pure NumPy utilities and tests committed; integration into DeepXDE training loops and quantitative results pending. No heavy training has been run.
