# Fourier features and residual-adaptive PINN training

The DeepXDE PyTorch `src.deepxde_models.train` API supports two opt-in extensions:

- `fourier_frequencies=[0.5, 1, 2, 4]`: differentiable spatial Fourier features, appended to original coordinates. Burgers time coordinate is retained.
- `adaptive_points=64, candidate_points=1024`: split the optimizer budget in two, select high-residual candidate points after the first half, add them as PDE anchors, and train the remaining half.

Use an identical seed, optimizer iteration count, domain sampling, network width/depth, and evaluation grid for the baseline and ablations. Fourier feature models have a larger input layer and therefore a different parameter count; do not claim equal-parameter fairness. Residual-adaptive runs include an additional candidate-residual evaluation and must report its overhead.

Example remote-only invocation in a CUDA-enabled Python environment:

```python
import os
os.environ["DDE_BACKEND"]="pytorch"
from src.deepxde_models import train
model,exact=train("helmholtz",seed=42,iterations=10000,
                  fourier_frequencies=[0.5,1,2,4],
                  adaptive_points=64,candidate_points=1024)
```

Do not run full training locally. First perform remote two-stage smoke tests and verify that DeepXDE's `model.predict(..., operator=model.data.pde)` and `model.data.add_anchors` work for both PDE families and backend versions. No empirical improvement claims have been made.
