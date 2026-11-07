import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ppg.datasets.osasud import prepare

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path,required=True)
    prepare(p.parse_args().source)
