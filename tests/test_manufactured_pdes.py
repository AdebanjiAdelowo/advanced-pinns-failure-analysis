import numpy as np
from src.burgers_reference import exact_solution as burgers_exact, residual_from_exact as burgers_residual
from src.helmholtz_reference import exact_solution as helmholtz_exact, residual_from_exact as helmholtz_residual


def test_burgers_initial_boundary_and_residual():
    x = np.linspace(0, 1, 33)
    t = np.linspace(0, 1, 33)
    assert np.allclose(burgers_exact(x, 0), np.sin(np.pi*x))
    assert np.allclose(burgers_exact(0, t), 0)
    assert np.allclose(burgers_exact(1, t), 0, atol=1e-14)
    xx, tt = np.meshgrid(x, t)
    assert np.max(np.abs(burgers_residual(xx, tt))) < 1e-12


def test_helmholtz_boundary_and_residual():
    x = np.linspace(0, 1, 41)
    assert np.allclose(helmholtz_exact([0, 1]), 0, atol=1e-14)
    assert np.max(np.abs(helmholtz_residual(x))) < 1e-12
