from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from wearable_ppg.evaluation.compare import compare
if __name__=="__main__": compare()
