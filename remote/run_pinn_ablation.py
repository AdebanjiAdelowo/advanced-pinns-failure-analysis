"""Remote-only, multi-seed PINN ablation runner; no work on import."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np


VARIANTS=("baseline","fourier","adaptive","fourier-adaptive")


def variant_kwargs(name,adaptive_points=64):
    if name not in VARIANTS:
        raise ValueError(f"unknown variant: {name}")
    result={}
    if "fourier" in name:
        result["fourier_frequencies"]=[0.5,1,2,4]
    if "adaptive" in name:
        result["adaptive_points"]=adaptive_points
        result["candidate_points"]=1024
    return result


def evaluation_points(problem):
    if problem=="burgers":
        space=np.linspace(0,1,41)
        time=np.linspace(0,1,41)
        x,t=np.meshgrid(space,time,indexing="ij")
        return np.column_stack([x.ravel(),t.ravel()])
    if problem=="helmholtz":
        return np.linspace(0,1,201)[:,None]
    raise ValueError("unknown problem")


def evaluate_prediction(model,exact,points):
    prediction=np.asarray(model.predict(points),dtype=float).reshape(-1)
    target=np.asarray(exact(points),dtype=float).reshape(-1)
    if prediction.shape!=target.shape or not np.isfinite(prediction).all():
        raise ValueError("invalid model prediction")
    norm=np.linalg.norm(target)
    if norm==0:
        raise ValueError("zero reference norm")
    return float(np.linalg.norm(prediction-target)/norm)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--problem",choices=["burgers","helmholtz"],required=True)
    p.add_argument("--variants",nargs="+",choices=VARIANTS,default=list(VARIANTS))
    p.add_argument("--seeds",nargs="+",type=int,default=[42,0,1,2,3])
    p.add_argument("--iterations",type=int,default=10000)
    p.add_argument("--adaptive-points",type=int,default=64)
    p.add_argument("--output",default="results/pinn_ablation.json")
    args=p.parse_args()
    if os.environ.get("DDE_BACKEND")!="pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    if args.iterations<2 or args.adaptive_points<1:
        raise ValueError("iterations must be >= 2; adaptive-points positive")
    if len(set(args.seeds))!=len(args.seeds):
        raise ValueError("duplicate seeds")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA GPU required")
    from src.deepxde_models import train
    points=evaluation_points(args.problem)
    records=[]
    output=Path(args.output)
    for variant in args.variants:
        for seed in args.seeds:
            start=time.perf_counter()
            model,exact=train(args.problem,seed=seed,iterations=args.iterations,
                              **variant_kwargs(variant,args.adaptive_points))
            if next(model.net.parameters()).device.type!="cuda":
                raise RuntimeError("DeepXDE network not on CUDA")
            error=evaluate_prediction(model,exact,points)
            records.append({"problem":args.problem,"variant":variant,"seed":seed,
                            "iterations":args.iterations,"relative_l2":error,
                            "wall_seconds":time.perf_counter()-start,
                            "cuda_device":torch.cuda.get_device_name(0),
                            "parameter_count":sum(p.numel() for p in model.net.parameters())})
            output.parent.mkdir(parents=True,exist_ok=True)
            output.write_text(json.dumps(records,indent=2)+"\n")
            print(f"{args.problem} {variant} seed={seed}: L2={error:.6g}",flush=True)
    print(output)


if __name__=="__main__":
    main()
