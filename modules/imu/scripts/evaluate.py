import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_imu.evaluation.compare import compare
from wearable_imu.datasets.sisfall import RESULTS
from wearable_imu.training.protocol import RUN

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--root",type=Path,default=RUN); p.add_argument("--out",type=Path,default=RESULTS)
    args=p.parse_args(); print({k:v["final"] for k,v in compare(args.root,args.out).items() if isinstance(v,dict) and "final" in v})
