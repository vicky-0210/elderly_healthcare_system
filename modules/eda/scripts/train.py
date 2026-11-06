"""Train or resume a WESAD LOSO experiment family."""
import argparse
from pathlib import Path
import subprocess
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__,epilog="Forward family-specific options after the family name.")
    parser.add_argument("family",choices=["classical","neural"])
    args, extra = parser.parse_known_args()
    script = Path(__file__).with_name("train_" + args.family + ".py")
    raise SystemExit(subprocess.call([sys.executable,str(script),*extra]))
