"""Rebuild WESAD participant metrics, comparisons, selection and plots."""
import argparse
from pathlib import Path
import sys

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE/"src"))
from wearable_eda.evaluation.report import make_report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs",type=Path,default=Path("artifacts/eda/runs/wesad_loso_v1"))
    parser.add_argument("--out",type=Path,default=Path("research/results/wesad_loso_v1"))
    parser.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    parser.add_argument("--contrasts",type=Path,default=MODULE/"configs"/"contrasts.json")
    parser.add_argument("--exploratory",type=Path,default=MODULE/"configs"/"exploratory_contrasts.json")
    parser.add_argument("--partial",action="store_true",help="Summarize completed experiments without a full-study claim or paired contrast report.")
    args = parser.parse_args()
    make_report(args.runs,args.out,args.cache,None if args.partial else args.contrasts,args.partial,
                None if args.partial else args.exploratory)
