"""Remote-only training for manufactured Burgers and Helmholtz."""
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
from src.deepxde_models import train

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--problem", choices=["burgers","helmholtz"], required=True)
    p.add_argument("--iterations", type=int, default=10000)
    p.add_argument("--seeds", type=int, nargs="+", default=[42,0,1,2,3])
    p.add_argument("--output", default="results/manufactured_remote.json")
    args=p.parse_args()
    if os.environ.get("DDE_BACKEND") != "pytorch":
        raise RuntimeError("Set DDE_BACKEND=pytorch before execution")
    records=[]
    for seed in args.seeds:
        start=time.perf_counter()
        model, exact=train(args.problem, iterations=args.iterations, seed=seed)
        if args.problem=="burgers":
            x=np.stack(np.meshgrid(np.linspace(0,1,41),np.linspace(0,1,41)),axis=-1).reshape(-1,2)
        else:
            x=np.linspace(0,1,201)[:,None]
        pred=model.predict(x).reshape(-1)
        ref=exact(x).reshape(-1)
        records.append({"problem":args.problem,"seed":seed,"iterations":args.iterations,
                        "relative_l2":float(np.linalg.norm(pred-ref)/np.linalg.norm(ref)),
                        "wall_seconds":time.perf_counter()-start})
        output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(records,indent=2)+"\n")
    print(output)

if __name__=="__main__":
    main()
