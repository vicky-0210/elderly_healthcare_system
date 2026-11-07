import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import torch
from wearable_ppg.models.dual_rate import VARIANTS
from wearable_ppg.training.dual_rate import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    run(args.models, args.device)
