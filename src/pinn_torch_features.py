"""Differentiable Fourier features for DeepXDE PyTorch networks."""
import numpy as np


def make_fourier_feature_transform(frequencies, input_dim=1):
    import torch
    frequencies=np.asarray(frequencies,dtype=float).reshape(-1)
    if input_dim not in (1,2):
        raise ValueError("input_dim must be 1 or 2")
    if not len(frequencies) or not np.isfinite(frequencies).all():
        raise ValueError("frequencies must be finite and nonempty")

    def transform(x):
        if x.ndim!=2 or x.shape[1]!=input_dim:
            raise ValueError("unexpected input dimensions")
        freq=torch.as_tensor(frequencies,dtype=x.dtype,device=x.device)
        angles=2*torch.pi*x[:,0:1]*freq.reshape(1,-1)
        return torch.cat([x,torch.sin(angles),torch.cos(angles)],dim=1)
    return transform
