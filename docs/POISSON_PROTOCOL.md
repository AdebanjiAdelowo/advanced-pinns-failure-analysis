# Poisson baseline protocol

Equation: -u''(x)=pi^2 sin(pi*x), x in (0,1); u(0)=u(1)=0.
Manufactured exact solution: u(x)=sin(pi*x).

Run `python -m pytest -q` for independent reference tests.
After installing DeepXDE, PyTorch and NumPy, run
`DDE_BACKEND=pytorch python -m experiments.run_poisson`.

The training grid and 501-point evaluation grid are distinct. Report seed, device,
library versions, relative L2 error, wall time and boundary residual in later benchmarks.
The present script does not claim validated training results or a multi-seed comparison.
