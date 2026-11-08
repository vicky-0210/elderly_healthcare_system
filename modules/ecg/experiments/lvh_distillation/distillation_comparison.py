"""Paired fold-10 contrast of predefined KD stages, independent of student selection."""
from run import Study, A, R, DEV, results, choose, patient, original
from wearable_ecg.runtime import save_json
from wearable_ecg.evaluation.metrics import calibrated
import numpy as np
import pandas as pd
import json
from sklearn.metrics import average_precision_score

def main():
    sel=json.loads((R/"selection.json").read_text())
    st=Study();st.teacher=sel["teacher"]
    dev=A/"development_amp"
    candidates=[json.loads(p.read_text()) for p in dev.glob("E4*/done.json")]
    logit=choose(candidates)
    names=list(dict.fromkeys(["E3",logit["name"],"E5_repr","E5_gate_only","E5_conf_gate","joint_full","joint_no_teacher",sel["student"],sel["teacher"],"E0"]))
    save_json(R/"kd_contrasts.json",dict(names=names,rule="E3, validation-selected E4, all E5 factorial controls, selected student, teacher, E0"))
    rows=[]; preds={}
    for name in names:
        if name=="E0": ids,z=original(st,True)
        else:
            row=json.loads((dev/name/"done.json").read_text())
            rows.extend(results(st,row,name,"teacher" if name==st.teacher else "student"))
            f=np.load(A/"refit"/name/"scores.npz");ids,z=f["ids"],f["logit"]
        frame=st.m.iloc[ids];score=patient(frame[frame.strat_fold==10],z[frame.strat_fold==10])
        if preds: np.testing.assert_array_equal(ids0,score.index)
        ids0=score.index;y=score.y.to_numpy()
        preds[name]=score.z.to_numpy()
    pd.DataFrame(rows).to_csv(R/"kd_final.csv",index=False)
    rng=np.random.default_rng(42)
    pairs=list(dict.fromkeys([(logit["name"],"E3"),("E5_repr",logit["name"]),("E5_conf_gate","E3"),("joint_full","joint_no_teacher"),("joint_full","E3"),(sel["teacher"],sel["student"]),(sel["student"],"E0")]))
    out=[]
    for a,b in pairs:
        def delta(idx): return average_precision_score(y[idx],preds[a][idx])-average_precision_score(y[idx],preds[b][idx])
        d=delta(np.arange(len(y)))
        draws=[delta(rng.integers(len(y),size=len(y))) for _ in range(1000)]
        lo,hi=np.quantile(draws,[.025,.975]);out.append(dict(model=a,reference=b,ap_difference=d,low=lo,high=hi))
    pd.DataFrame(out).to_csv(R/"paired_kd.csv",index=False)
    print(pd.DataFrame(out).round(4).to_string(index=False))

if __name__=="__main__":main()
