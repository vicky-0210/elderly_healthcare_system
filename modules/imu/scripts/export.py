import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_imu.training.exporting import export_one,EXPORT
from wearable_imu.training.protocol import RUN
from wearable_imu.datasets.sisfall import RESULTS

if __name__=="__main__":
    import torch
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--root",type=Path,default=RUN); p.add_argument("--export",type=Path,default=EXPORT)
    p.add_argument("--experiments",nargs="+",help="Export explicitly identified secondary comparators")
    args=p.parse_args()
    torch.set_num_threads(2)
    choice=json.loads((RESULTS/"selection.json").read_text())
    for k in args.experiments or dict.fromkeys(choice[menu]["final"] for menu in ("all","neural")):
        export_one(k,args.root,args.export,RESULTS,args.device)
