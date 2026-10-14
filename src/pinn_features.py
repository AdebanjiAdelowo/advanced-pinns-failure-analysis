"""Lightweight reusable Fourier feature map and residual-based sampling utilities."""
import numpy as np


def fourier_features(x, frequencies):
    x=np.asarray(x,dtype=float)
    frequencies=np.asarray(frequencies,dtype=float).reshape(-1)
    if x.ndim!=2 or x.shape[1]!=1:
        raise ValueError("x must have shape (n,1)")
    angles=2*np.pi*x*frequencies[None,:]
    return np.concatenate([np.sin(angles),np.cos(angles)],axis=1)


def select_largest_residual(points, residuals, count):
    points=np.asarray(points)
    residuals=np.asarray(residuals).reshape(-1)
    if len(points)!=len(residuals) or count<1 or count>len(points):
        raise ValueError("invalid points, residuals or count")
    if not np.isfinite(residuals).all():
        raise ValueError("residuals must be finite")
    indices=np.argsort(-np.abs(residuals),kind="stable")[:count]
    return points[indices],indices
