from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import torch
from wearable_ppg.training.study import run
from wearable_ppg.models.respiratory import PRIMARY,CONTROLS
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--models",nargs="+",choices=PRIMARY+CONTROLS,default=PRIMARY+CONTROLS)
    parser.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args(); run(args.models,args.device)
