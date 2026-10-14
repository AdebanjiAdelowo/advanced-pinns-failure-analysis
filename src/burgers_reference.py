"""Viscous Burgers manufactured-solution benchmark.

u(x,t)=exp(-t)*sin(pi*x) on [0,1]x[0,1].
Forced equation: u_t + u*u_x - nu*u_xx = f.
Homogeneous Dirichlet boundaries, sinusoidal initial condition.
"""
import numpy as np


def exact_solution(x, t):
    return np.exp(-np.asarray(t)) * np.sin(np.pi * np.asarray(x))


def forcing(x, t, nu=0.01):
    x = np.asarray(x)
    t = np.asarray(t)
    u = exact_solution(x, t)
    u_t = -u
    u_x = np.pi * np.exp(-t) * np.cos(np.pi * x)
    u_xx = -(np.pi**2) * u
    return u_t + u * u_x - nu * u_xx


def residual_from_exact(x, t, nu=0.01):
    """Analytical residual; zero up to floating-point arithmetic."""
    x, t = np.asarray(x), np.asarray(t)
    u = exact_solution(x, t)
    return -u + u * np.pi * np.exp(-t) * np.cos(np.pi * x) + nu * np.pi**2 * u - forcing(x, t, nu)
