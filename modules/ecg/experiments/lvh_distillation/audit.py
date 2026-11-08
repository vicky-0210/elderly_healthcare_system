"""Audit all source-window integrity flags and patient independence without training."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from study import C, R, A, array
from wearable_ecg.runtime import save_json
from wearable_ecg.preprocessing.pipeline import VOLTAGE_NAMES
from wearable import convert_counts

def main():
    frame=pd.read_csv(C/"manifest.csv")
    eligible=pd.read_csv(C/"lvh_vs_rest_manifest.csv")
    if frame.groupby("patient_id").strat_fold.nunique().max()!=1:
        raise ValueError("Patient assigned to multiple folds")
    x=array("lead_ii_500hz_mv")
    features=array("constrained_features")
    row=[]
    for k in range(1,11):
        ids=frame.loc[frame.strat_fold==k,"row_index"].to_numpy()
        count=np.zeros(6,dtype=int)
        for g in np.array_split(ids,int(np.ceil(len(ids)/128))):
            v=np.array(x[g])
            same=np.diff(v,axis=-1)==0
            # Runs of >=100 ms at identical ADC values, including possible plateau artifacts.
            from scipy.ndimage import minimum_filter1d
            flat=minimum_filter1d(same.astype(np.uint8),49,axis=1,mode="constant").any(1)
            dwell=np.maximum((v==v.max(1,keepdims=True)).mean(1),(v==v.min(1,keepdims=True)).mean(1))
            count+=np.array([np.any(~np.isfinite(v),1).sum(),(np.ptp(v,axis=1)<1e-5).sum(),flat.sum(),
                              (dwell>=.10).sum(),(abs(v).max(1)>5).sum(),(features[g,14]<.5).sum()])
        row.append(dict(fold=k,records=len(ids),eligible=int((eligible.strat_fold==k).sum()),**dict(zip(
            ["nonfinite","whole_flat","plateau_100ms","extrema_dwell_10pct","over_5mv","low_reliability"],count))))
    pd.DataFrame(row).to_csv(R/"signal_integrity.csv",index=False)
    np.testing.assert_allclose(convert_counts([2000,2010],2000,.01,1),[0,.1])
    save_json(R/"split_audit.json",dict(patients=eligible.patient_id.nunique(),recordings=len(eligible),
        patient_fold_overlap=0,features=VOLTAGE_NAMES,source_adc_rails="unavailable; plateau and dwell flags are proxies"))
    print(pd.DataFrame(row).sum().to_dict())

if __name__=="__main__":main()
