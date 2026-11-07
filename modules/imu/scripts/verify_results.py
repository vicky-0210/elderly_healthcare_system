"""Verify released cohort, all predictions, grouped decisions and deployed stream parity."""
import argparse,hashlib,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from wearable_imu.datasets.sisfall import RESULTS,CACHE,COUNTS_TO_PHYSICAL,load_cache
from wearable_imu.training.catalogue import catalogue,available
from wearable_imu.training.protocol import RUN,complete
from wearable_imu.training.exporting import EXPORT
from wearable_imu.inference.predictor import IMUPredictor


def check(source=None):
    m,counts=load_cache(); saved=json.loads((RESULTS/"dataset_audit.json").read_text())
    digest=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
    assert digest(CACHE/"manifest.csv")==saved["manifest_sha256"]
    if source:
        for r in m.itertuples():
            assert digest(source/r.subject/(r.id+".txt"))==r.source_sha256,r.id
    comparisons=pd.read_csv(RESULTS/"comparison.csv").set_index("model")
    checks=0
    for key,spec in catalogue().items():
        parts=[]
        for k in range(5):
            f=RUN/key/f"fold{k}"; assert complete(f,spec)
            meta=json.loads((f/"done.json").read_text())
            assert set(meta["refit_subjects"]).isdisjoint(meta["held_subjects"])
            assert set(meta["train_subjects"]).isdisjoint(meta["val_subjects"])
            r=pd.read_csv(f/"test.csv"); assert set(r.subject)==set(meta["held_subjects"]); parts.append(r); checks+=1
        r=pd.concat(parts); assert len(r)==len(m) and r.id.is_unique and set(r.id)==set(m.id)
        for col,y,d in (("tp",1,1),("fp",0,1),("tn",0,0),("fn",1,0)):
            assert comparisons.loc[key,col]==((r.y==y)&(r.decision==d)).sum(),(key,col)
    export_reports=[]
    from wearable_imu.training import protocol
    original_seed=protocol.SEED
    supplementary=0
    try:
        for seed in (4907,6203,7919):
            protocol.SEED=seed
            root=RUN if seed==4907 else RUN.with_name(RUN.name+f"_seed{seed}")
            for name in (("dm_distilled",) if seed==4907 else ("dm_distilled","inception","dm_tcn","dm_standard")):
                for k in range(5):
                    directory=root/name/f"fold{k}"
                    assert complete(directory,available()[name])
                    meta=json.loads((directory/"done.json").read_text())
                    truth=m[m.fold==k].set_index("id").y.sort_index()
                    pred=pd.read_csv(directory/"test.csv").set_index("id").sort_index()
                    assert truth.equals(pred.y) and set(pred.subject).isdisjoint(meta["refit_subjects"])
                    supplementary+=1
    finally:
        protocol.SEED=original_seed
    for path in sorted(RESULTS.glob("export_*.json")):
        report=json.loads(path.read_text()); key=report["experiment"]
        directory=EXPORT/key; meta=json.loads((directory/"runtime.json").read_text())
        assert digest(directory/meta["file"])==report["model_sha256"]
        predictor=IMUPredictor(directory); worst=0.
        for r in [m[m.y==0].iloc[0],m[m.y==1].iloc[0],m[(m.older)&(m.y==1)].iloc[0]]:
            data=counts[r.offset:r.offset+r.n]*COUNTS_TO_PHYSICAL
            if predictor.fs==100:
                from scipy.signal import butter,sosfilt,sosfilt_zi
                sos=butter(4,40,fs=200,output="sos")
                data=sosfilt(sos,data,axis=0,zi=sosfilt_zi(sos)[:,:,None]*data[0])[0][1::2]
            predictor.reset(); expected=predictor.push(data,0); predictor.reset(); replay=[]
            for n in range(0,len(data),37): replay+=predictor.push(data[n:n+37],n/predictor.fs)
            assert len(replay)==len(expected)>0
            assert [v["alert"] for v in expected]==[v["alert"] for v in replay]
            a=np.array([v["window_logit"] for v in expected]); b=np.array([v["window_logit"] for v in replay])
            worst=max(worst,float(np.abs(a-b).max())); np.testing.assert_allclose(a,b,atol=.0002,rtol=.0002)
            assert not predictor.push(data[:1],9999), "Gap did not discard the old window"
        export_reports.append(dict(model=key,max_chunk_difference=worst,source_trials=3))
    result=dict(folds_verified=checks,additional_folds_verified=supplementary,trial_decisions_verified=len(m)*len(comparisons),source_files_rehashed=len(m) if source else 0,
        manifest_sha256=saved["manifest_sha256"],export_streaming=export_reports)
    return result

if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--source",type=Path); p.add_argument("--save",action="store_true"); args=p.parse_args()
    result=check(args.source)
    if args.save: (RESULTS/"verification.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))
