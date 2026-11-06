"""Run the development/refit/calibration suite. Does not evaluate test data."""
from pathlib import Path
import sys
import json
import argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ecg.training.runner import train_suite

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--config",required=True,type=Path)
    args=p.parse_args(); train_suite(args.cache,args.output,json.loads(args.config.read_text()))
