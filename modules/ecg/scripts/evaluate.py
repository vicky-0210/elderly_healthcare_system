"""Evaluate only a completed, frozen model comparison and save plots."""
from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ecg.evaluation.report import evaluate_suite

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(); evaluate_suite(a.cache,a.output)
