"""Manufactured 1D Helmholtz benchmark: -u_xx - k^2*u = f.

u(x)=sin(pi*x), u(0)=u(1)=0. Choose k away from Dirichlet eigenfrequencies
for a uniquely solvable problem.
"""
import numpy as np


def exact_solution(x):
    return np.sin(np.pi * np.asarray(x))


def forcing(x, wavenumber=2.0):
    return (np.pi**2 - float(wavenumber)**2) * exact_solution(x)


def residual_from_exact(x, wavenumber=2.0):
    u = exact_solution(x)
    return (np.pi**2 - float(wavenumber)**2) * u - forcing(x, wavenumber)
