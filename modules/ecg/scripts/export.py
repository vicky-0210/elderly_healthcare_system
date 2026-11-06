from pathlib import Path
import sys,argparse
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ecg.inference.export import export_bundle
if __name__=="__main__":
    p=argparse.ArgumentParser(description="Export a trained model with preprocessing/calibration and ONNX parity checks")
    p.add_argument("--run",required=True,type=Path); p.add_argument("--cache",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    a=p.parse_args(); export_bundle(a.run,a.cache,a.output)
