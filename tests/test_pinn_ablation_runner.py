import json
import numpy as np
import pytest
from remote.run_pinn_ablation import variant_kwargs,evaluation_points,evaluate_prediction
from remote.smoke_pinn_ablation import commands,validate


def test_variants():
    assert variant_kwargs("baseline")=={}
    assert "fourier_frequencies" in variant_kwargs("fourier")
    assert "adaptive_points" in variant_kwargs("adaptive")
    assert len(variant_kwargs("fourier-adaptive"))==3


def test_evaluation_grid():
    assert evaluation_points("burgers").shape==(1681,2)
    assert evaluation_points("helmholtz").shape==(201,1)


def test_smoke_commands_distinct():
    c=commands("burgers")
    assert len(c)==4
    assert len(set(cmd[-1] for cmd in c.values()))==4


def test_smoke_validation(tmp_path):
    path=tmp_path/"smoke.json"
    row={"problem":"burgers","variant":"baseline","seed":42,
         "relative_l2":0.2,"wall_seconds":1.0,
         "parameter_count":100,"cuda_device":"test"}
    path.write_text(json.dumps([row]))
    assert validate(path,"burgers","baseline",42)==row
    with pytest.raises(ValueError,match="metadata"):
        validate(path,"burgers","fourier",42)


def test_evaluation_rejects_nan():
    class Model:
        def predict(self,points):
            return np.full((len(points),1),np.nan)
    with pytest.raises(ValueError,match="invalid"):
        evaluate_prediction(Model(),lambda p:np.ones((len(p),1)),np.zeros((3,1)))
