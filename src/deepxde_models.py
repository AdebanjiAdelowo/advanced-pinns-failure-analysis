"""DeepXDE PyTorch manufactured Burgers and Helmholtz PINNs.

Import DeepXDE only after setting DDE_BACKEND=pytorch.
"""
import numpy as np


def build_model(problem, seed=42, nu=0.01, wavenumber=2.0, hard_bc=False,\n                fourier_frequencies=None):
    import deepxde as dde
    import torch

    dde.config.set_random_seed(seed)
    if problem == "burgers":
        geom = dde.geometry.Interval(0, 1)
        time = dde.geometry.TimeDomain(0, 1)
        domain = dde.geometry.GeometryXTime(geom, time)

        def exact(x):
            return np.exp(-x[:, 1:2]) * np.sin(np.pi * x[:, 0:1])

        def pde(x, u):
            ut = dde.grad.jacobian(u, x, j=1)
            ux = dde.grad.jacobian(u, x, j=0)
            uxx = dde.grad.hessian(u, x, i=0, j=0)
            target = torch.exp(-x[:, 1:2]) * torch.sin(np.pi * x[:, 0:1])
            source = (-target + target * np.pi * torch.exp(-x[:, 1:2])
                      * torch.cos(np.pi * x[:, 0:1]) + nu * np.pi**2 * target)
            return ut + u * ux - nu * uxx - source

        bc = dde.icbc.DirichletBC(domain, lambda _: 0.0,
                                 lambda _, on_boundary: on_boundary)
        ic = dde.icbc.IC(domain, lambda x: np.sin(np.pi * x[:, 0:1]),
                         lambda _, on_initial: on_initial)
        data = dde.data.TimePDE(domain, pde, [bc, ic],
                                num_domain=512, num_boundary=64, num_initial=64,
                                num_test=512)
        net = dde.nn.FNN([2, 64, 64, 64, 1], "tanh", "Glorot normal")
        if hard_bc:
            net.apply_output_transform(lambda x, y: x[:, 0:1] * (1-x[:, 0:1]) * y)
    elif problem == "helmholtz":
        if np.isclose(wavenumber / np.pi, round(wavenumber / np.pi), atol=1e-10):
            raise ValueError("Dirichlet resonance: choose non-eigenfrequency")
        domain = dde.geometry.Interval(0, 1)

        def exact(x):
            return np.sin(np.pi * x)

        def pde(x, u):
            source = (np.pi**2 - wavenumber**2) * torch.sin(np.pi * x)
            return -dde.grad.hessian(u, x) - wavenumber**2 * u - source

        bc = dde.icbc.DirichletBC(domain, lambda _: 0.0,
                                 lambda _, on_boundary: on_boundary)
        data = dde.data.PDE(domain, pde, bc, num_domain=256,
                            num_boundary=32, num_test=256)
        net = dde.nn.FNN([1, 64, 64, 64, 1], "tanh", "Glorot normal")
        if hard_bc:
            net.apply_output_transform(lambda x, y: x * (1-x) * y)
    else:
        raise ValueError("problem must be burgers or helmholtz")
    if fourier_frequencies is not None:\n        from src.pinn_torch_features import make_fourier_feature_transform\n        frequencies=np.asarray(fourier_frequencies,dtype=float).reshape(-1)\n        if frequencies.size==0 or not np.isfinite(frequencies).all():\n            raise ValueError("Fourier frequencies must be finite and nonempty")\n        transformed_dim=(2 if problem == "burgers" else 1)+2*len(frequencies)\n        net=dde.nn.FNN([transformed_dim,64,64,64,1], "tanh", "Glorot normal")\n        net.apply_feature_transform(make_fourier_feature_transform(\n            frequencies, input_dim=2 if problem == "burgers" else 1))\n        if hard_bc:\n            if problem=="burgers":\n                net.apply_output_transform(lambda x,y: x[:,0:1]*(1-x[:,0:1])*y)\n            else:\n                net.apply_output_transform(lambda x,y: x*(1-x)*y)\n    model = dde.Model(data, net)
    return model, exact


def train(problem, iterations=10000, seed=42, adaptive_points=0,\n          candidate_points=1024, **kwargs):
    model, exact = build_model(problem, seed=seed, **kwargs)
    model.compile("adam", lr=1e-3)
    model.train(iterations=iterations, display_every=max(1, iterations // 10))
    return model, exact
