import numpy as np
import pytest
from src.manufactured_validation import (
    burgers_manufactured_residual,
    helmholtz_manufactured_residual,
    check_manufactured_solutions,
)


def test_manufactured_residuals_zero():
    report=check_manufactured_solutions()
    assert report["burgers_max_abs"]<1e-11
    assert report["helmholtz_max_abs"]<1e-11


@pytest.mark.parametrize("nu",[0.001,0.01,0.1])
def test_burgers_viscosity(nu):
    x=np.linspace(0,1,17)
    assert np.max(np.abs(burgers_manufactured_residual(x,0.4,nu)))<1e-11


@pytest.mark.parametrize("k",[0.5,2.0,5.0])
def test_helmholtz_wavenumber(k):
    x=np.linspace(0,1,17)
    assert np.max(np.abs(helmholtz_manufactured_residual(x,k)))<1e-11


def test_invalid_viscosity_rejected():
    with pytest.raises(ValueError):
        burgers_manufactured_residual([0.5],0.1,nu=0)
