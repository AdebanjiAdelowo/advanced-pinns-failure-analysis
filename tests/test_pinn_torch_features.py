import numpy as np
import pytest
from src.pinn_features import select_largest_residual


def test_residual_selector_ties_are_deterministic():
    points=np.arange(5).reshape(-1,1)
    selected,indices=select_largest_residual(points,[2,-2,1,0,2],2)
    assert indices.tolist()==[0,1]
    assert selected[:,0].tolist()==[0,1]


def test_torch_fourier_features_have_spatial_gradients():
    torch=pytest.importorskip("torch")
    from src.pinn_torch_features import make_fourier_feature_transform
    x=torch.tensor([[0.25,0.5]],dtype=torch.float64,requires_grad=True)
    features=make_fourier_feature_transform([0.5,1.0],input_dim=2)(x)
    assert features.shape==(1,6)
    grad=torch.autograd.grad(features[:,2].sum(),x)[0]
    assert torch.isfinite(grad).all()
    assert abs(grad[0,0])>1e-6
