# Manufactured benchmarks: Burgers and Helmholtz

These analytical references enable correctness checks independently of neural-network training.

## Forced viscous Burgers
u=exp(-t) sin(pi*x), with u_t+u u_x-nu u_xx=f, x in (0,1), t in (0,1).
Dirichlet boundaries are zero and initial data sin(pi*x). The source term is explicitly derived from u; this is a **forced** benchmark, not the classical unforced Burgers benchmark. Report viscosity, training points and held-out residuals.

## Helmholtz
-u_xx-k²u=f, u=sin(pi*x), f=(pi²-k²)sin(pi*x), zero Dirichlet boundaries.
The chosen default k=2 is not a Dirichlet eigenfrequency. Test higher k carefully: Helmholtz near resonance is ill-conditioned and may be nonunique at eigenfrequencies.

## Status
Reference formulas and tests only; DeepXDE training and performance comparisons pending.
