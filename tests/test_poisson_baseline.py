import numpy as np
import pytest
from src.poisson_baseline import exact_solution, forcing, finite_difference_reference, relative_l2


def test_exact_boundary_conditions():
    assert np.allclose(exact_solution([0, 1]), [0, 0], atol=1e-14)


def test_manufactured_equation():
    x = np.linspace(0, 1, 51)
    assert np.allclose(forcing(x), np.pi**2 * exact_solution(x))


def test_fd_convergence():
    errors = []
    for n in (21, 41, 81):
        x, u = finite_difference_reference(n)
        errors.append(relative_l2(u, exact_solution(x)))
    assert errors[0] / errors[1] > 3.8
    assert errors[1] / errors[2] > 3.8


def test_invalid_grid():
    with pytest.raises(ValueError):
        finite_difference_reference(2)


def test_shape_mismatch():
    with pytest.raises(ValueError):
        relative_l2([1, 2], [1])
