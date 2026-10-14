import numpy as np
import pytest
from src.pinn_features import fourier_features,select_largest_residual


def test_fourier_feature_shape_and_values():
    out=fourier_features(np.array([[0.0],[0.25]]),[1,2])
    assert out.shape==(2,4)
    assert np.allclose(out[0,:2],0)
    assert np.allclose(out[0,2:],1)


def test_residual_selection_deterministic():
    points=np.arange(5)[:,None]
    selected,indices=select_largest_residual(points,[.1,-3,2,0,-3],2)
    assert indices.tolist()==[1,4]
    assert selected.ravel().tolist()==[1,4]


def test_invalid_residuals_rejected():
    with pytest.raises(ValueError):
        select_largest_residual(np.arange(2),[0,np.nan],1)
