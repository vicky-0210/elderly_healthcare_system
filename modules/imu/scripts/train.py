"""Run the prespecified, resumable subject-grouped comparison from the repository root."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_imu.training.catalogue import catalogue,available
from wearable_imu.training.protocol import RUN


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family",choices=["neural","classical","all"],default="all")
    parser.add_argument("--experiments",nargs="+")
    parser.add_argument("--root",type=Path,default=RUN)
    parser.add_argument("--device",default="auto")
    parser.add_argument("--seed",type=int,default=4907)
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    if args.seed!=4907:
        from wearable_imu.training import protocol
        protocol.SEED=args.seed
        if args.root==RUN: args.root=RUN.with_name(RUN.name+f"_seed{args.seed}")
    specs=available() if args.experiments or args.list else catalogue()
    if args.list:
        for s in specs.values(): print(s.name,s.role,s.model)
        return
    grouped=defaultdict(list)
    for key in args.experiments or specs:
        spec=specs[key]
        family="classical" if spec.model in ("rf","svm","xgb") else "neural"
        if args.family in (family,"all"):
            group=json.dumps([spec.view,spec.seconds,spec.overlap,spec.settings,spec.scaler,spec.bag_size],sort_keys=True)
            grouped[family,group].append(spec)
    if not grouped: parser.error("No matching experiments")
    for (family,_),series in grouped.items():
        if family=="neural":
            import torch
            from wearable_imu.training.neural import run_group
            torch.set_num_threads(2)
            device=("cuda" if torch.cuda.is_available() else "cpu") if args.device=="auto" else args.device
            run_group(series,args.root,device)
        else:
            from wearable_imu.training.classical import run_group
            run_group(series,args.root)


if __name__=="__main__": main()
