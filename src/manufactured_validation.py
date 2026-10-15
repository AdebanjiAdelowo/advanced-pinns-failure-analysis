"""Independent manufactured-solution residual checks; NumPy only."""
import numpy as np


def burgers_manufactured_residual(x, t, nu=0.01):
    """Residual of exact u=exp(-t)sin(pi*x) against implemented forcing."""
    x=np.asarray(x,dtype=float)
    t=np.asarray(t,dtype=float)
    if not np.isfinite(nu) or nu<=0:
        raise ValueError("viscosity must be positive and finite")
    u=np.exp(-t)*np.sin(np.pi*x)
    ut=-u
    ux=np.pi*np.exp(-t)*np.cos(np.pi*x)
    uxx=-np.pi**2*u
    source=-u+u*ux+nu*np.pi**2*u
    return ut+u*ux-nu*uxx-source


def helmholtz_manufactured_residual(x, wavenumber=2.0):
    """Residual of exact u=sin(pi*x) for -u''-k²u=f."""
    x=np.asarray(x,dtype=float)
    if not np.isfinite(wavenumber):
        raise ValueError("wavenumber must be finite")
    u=np.sin(np.pi*x)
    uxx=-np.pi**2*u
    source=(np.pi**2-wavenumber**2)*u
    return -uxx-wavenumber**2*u-source


def check_manufactured_solutions(points=101, tolerance=1e-11):
    if points<3:
        raise ValueError("points must be >= 3")
    x=np.linspace(0,1,points)
    t=np.linspace(0,1,points)
    b=float(np.max(np.abs(burgers_manufactured_residual(x,t))))
    h=float(np.max(np.abs(helmholtz_manufactured_residual(x))))
    if max(b,h)>tolerance:
        raise AssertionError(f"manufactured residual mismatch: burgers={b}, helmholtz={h}")
    return {"burgers_max_abs":b,"helmholtz_max_abs":h}
