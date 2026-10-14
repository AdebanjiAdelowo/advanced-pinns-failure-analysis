"""Manufactured 1D Poisson problem and DeepXDE PINN baseline.

Solve -u'' = pi^2 sin(pi*x), x in (0,1), u(0)=u(1)=0.
Exact solution u(x)=sin(pi*x).
"""
import numpy as np


def exact_solution(x):
    x = np.asarray(x, dtype=float)
    return np.sin(np.pi * x)


def forcing(x):
    return np.pi**2 * exact_solution(x)


def finite_difference_reference(n=101):
    """Second-order central-difference solution at n grid points."""
    if n < 3:
        raise ValueError("n must be >= 3")
    x = np.linspace(0.0, 1.0, n)
    h = x[1] - x[0]
    m = n - 2
    a = (2 * np.eye(m) - np.eye(m, k=1) - np.eye(m, k=-1)) / h**2
    u = np.zeros(n)
    u[1:-1] = np.linalg.solve(a, forcing(x[1:-1]))
    return x, u


def relative_l2(prediction, reference):
    prediction = np.asarray(prediction, dtype=float).reshape(-1)
    reference = np.asarray(reference, dtype=float).reshape(-1)
    if prediction.shape != reference.shape:
        raise ValueError("prediction and reference shapes differ")
    denominator = np.linalg.norm(reference)
    if denominator == 0:
        raise ValueError("reference norm is zero")
    return float(np.linalg.norm(prediction - reference) / denominator)


def train_pinn(iterations=1000, seed=42, num_domain=128, num_boundary=16):
    """Requires DeepXDE configured with the PyTorch backend before import."""
    import deepxde as dde

    dde.config.set_random_seed(seed)
    geometry = dde.geometry.Interval(0.0, 1.0)

    def pde(x, y):
        u_xx = dde.grad.hessian(y, x)
        return -u_xx - np.pi**2 * dde.backend.sin(np.pi * x)

    bc = dde.icbc.DirichletBC(geometry, lambda x: 0.0, lambda _, on_boundary: on_boundary)
    data = dde.data.PDE(geometry, pde, bc, num_domain=num_domain, num_boundary=num_boundary)
    net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)
    model.train(iterations=iterations, display_every=max(1, iterations // 10))
    return model
