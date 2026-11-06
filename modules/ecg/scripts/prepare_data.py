"""Read source PTB-XL without modifying it; write derived cache separately."""
from pathlib import Path
import sys
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ecg.datasets.prepare import prepare

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source",required=True,type=Path); p.add_argument("--cache",required=True,type=Path)
    p.add_argument("--workers",type=int,default=4)
    args=p.parse_args(); prepare(args.source,args.cache,args.workers)
