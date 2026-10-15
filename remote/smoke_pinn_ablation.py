"""Remote-only PINN smoke gate, one seed per variant."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def commands(problem,iterations=2,seed=42,output_dir="results/smoke"):
    if problem not in ("burgers","helmholtz") or iterations<2:
        raise ValueError("invalid smoke settings")
    return {
        name:[sys.executable,"-m","remote.run_pinn_ablation","--problem",problem,
              "--variants",name,"--seeds",str(seed),"--iterations",str(iterations),
              "--adaptive-points","4","--output",
              str(Path(output_dir)/(problem+"_"+name+".json"))]
        for name in ("baseline","fourier","adaptive","fourier-adaptive")
    }


def validate(path,problem,variant,seed):
    import math
    rows=json.loads(Path(path).read_text())
    if len(rows)!=1:
        raise ValueError("expected one smoke record")
    row=rows[0]
    if (row["problem"],row["variant"],row["seed"])!=(problem,variant,seed):
        raise ValueError("wrong smoke metadata")
    if not all(math.isfinite(row[k]) for k in ("relative_l2","wall_seconds")):
        raise ValueError("non-finite smoke metric")
    if row["parameter_count"]<1 or not row.get("cuda_device"):
        raise ValueError("missing hardware or parameter metadata")
    return row


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--problem",choices=["burgers","helmholtz"],required=True)
    p.add_argument("--iterations",type=int,default=2)
    p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output-dir",default="results/smoke")
    p.add_argument("--timeout-seconds",type=int,default=600)
    args=p.parse_args()
    if os.environ.get("DDE_BACKEND")!="pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch")
    if args.timeout_seconds<1:
        raise ValueError("timeout must be positive")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Remote CUDA required")
    for variant,cmd in commands(args.problem,args.iterations,args.seed,args.output_dir).items():
        path=Path(cmd[-1])
        if path.exists():
            path.unlink()
        proc=subprocess.run(cmd,capture_output=True,text=True,
                            timeout=args.timeout_seconds,check=False)
        if proc.returncode:
            print(proc.stdout[-3000:],file=sys.stderr)
            print(proc.stderr[-3000:],file=sys.stderr)
            raise RuntimeError(f"smoke failed for {variant}")
        validate(path,args.problem,variant,args.seed)
        print(f"PASS {args.problem}/{variant}",flush=True)


if __name__=="__main__":
    main()
