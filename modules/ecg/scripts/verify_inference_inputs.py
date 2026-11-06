"""Check raw-window preprocessing against cached inputs on training records."""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd
import joblib
import wfdb
from wearable_ecg.preprocessing.pipeline import resample_mv,process

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache",required=True,type=Path); p.add_argument("--output",required=True,type=Path)
    p.add_argument("--source",required=True,type=Path)
    a=p.parse_args(); frame=pd.read_csv(a.cache/"manifest.csv")
    ids=frame.loc[frame.strat_fold.le(6),"row_index"].sample(128,random_state=910).to_numpy()
    waves=np.load(a.cache/"constrained_waves.npy",mmap_mode="r")
    features=np.load(a.cache/"constrained_features.npy",mmap_mode="r")
    controller=joblib.load(a.cache/"preprocessing/controller.joblib")
    wave_error=0.; feature_error=np.zeros(16); rounded_feature_error=np.zeros(16)
    for i in ids:
        record,info=wfdb.rdsamp(str(a.source/frame.iloc[i].filename_hr))
        raw=record[:,info["sig_name"].index("II")]
        y,f,_,_=process(resample_mv(raw,500,360),controller,"constrained")
        wave_error=max(wave_error,float(np.max(abs(y-waves[i]))))
        feature_error=np.maximum(feature_error,abs(f-features[i]))
        _,rounded,_,_=process(resample_mv(raw.astype(np.float32),500,360),controller,"constrained")
        rounded_feature_error=np.maximum(rounded_feature_error,abs(rounded-features[i]))
    result={"records_checked":len(ids),"folds_eligible":[1,2,3,4,5,6],
            "max_waveform_error_mv":wave_error,"max_feature_errors":feature_error.tolist(),
            "reference":"Original WFDB physical mV, float64 retained until after resampling",
            "float32_rounded_input_feature_errors":rounded_feature_error.tolist(),
            "precision_note":"Pre-rounding raw samples to float32 is a different input representation; the flat-sample count can differ by one sample. Preserve float64 calibration through resampling.",
            "waveform_tolerance_mv":2e-6,"feature_tolerance":2e-5,
            "passed":bool(wave_error<=2e-6 and np.max(feature_error)<=2e-5)}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps(result,indent=2))
    if not result["passed"]: raise RuntimeError("Offline/live preprocessing parity check failed")
