"""Execute E0–E10; selection precedes all new fold-10 prediction."""
from study import *
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from wearable_ecg.models.backbones import ECGClassifier


def original(st,final):
    frame=st.eligible
    sel=frame[frame.strat_fold.isin([9,10] if final else DEV)]
    ids=sel.row_index.to_numpy()
    folder=A/"lvh_vs_rest/cnn__constrained__seed42"
    ck=torch.load(folder/("refit/model.pt" if final else "development/best.pt"),weights_only=False,map_location="cpu")
    net=ECGClassifier(**ck["model_config"]).to(DEVICE)
    net.load_state_dict(ck["state_dict"])
    x=torch.tensor(array("constrained_waves")[ids],device=DEVICE)
    f=torch.tensor(array("constrained_features")[ids],device=DEVICE)
    z=predict_logits(net,x,f)
    return (ids,z) if final else dict(name="E0",config={},params=sum(p.numel() for p in net.parameters()),**devscore(sel,z))


def results(st,record,name,role):
    if name=="E0": ids,z=original(st,True)
    else:
        st.train(name,record["config"],True)
        f=np.load(A/"refit"/name/"scores.npz");ids,z=f["ids"],f["logit"]
    df=st.m.iloc[ids].copy()
    a=patient(df[df.strat_fold==9],z[df.strat_fold==9])
    b=patient(df[df.strat_fold==10],z[df.strat_fold==10])
    ab=fit_calibration(a.z.astype(float),a.y)
    pa,pb=calibrated(a.z,ab),calibrated(b.z,ab)
    rows=[]
    for spec in (.80,.85,.90,.95):
        thr=choose_threshold(a.y,pa,spec)
        rows.append(dict(name=name,role=role,unit="patient",target_spec=spec,**measure(b.y,pb,thr)))
        if spec==.9:
            test=df[df.strat_fold==10]; pr=calibrated(z[df.strat_fold==10],ab)
            rows.append(dict(name=name,role=role,unit="record",target_spec=spec,**measure(test.label,pr,thr)))
            save_json(A/"calibration"/(name+".json"),dict(**ab,threshold=thr,aggregation="max record logit and any-positive label per patient"))
    return rows

def main():
    baseline()
    prepare()
    st=Study(); base=original(st,False); rows=[base]
    def run(name,**kw):
        row=st.train(name,kw);rows.append(row)
        pd.DataFrame([{k:v for k,v in r.items() if k not in ("config","history")} for r in rows]).to_csv(R/"development.csv",index=False)
        return row
    run("E1",deep=True,fusion="features")
    ts=[run("E2_"+t,teacher=t,aux=[0,1,2,3]) for t in ("concat","inception","lead_attention")]
    teacher=choose(ts);st.teacher=teacher["name"]
    plain=run("E3")
    kd=[run(f"E4_T{t}_w{w}",T=t,kd=w) for t in (1,3) for w in (.2,.6)]
    logits=choose(kd)["config"]
    priv={k:logits[k] for k in ("T","kd")}
    run("E5_repr",**priv,rep=.15)
    run("E5_conf_gate",**priv,rep=.15,gate=True)
    run("E5_gate_only",**priv,gate=True)
    for mode in ("soft","weighted"):run("E6_"+mode,label=mode)
    for cut in (50,80):run(f"E6_exclude{cut}",exclude=cut)
    for j,key in enumerate(("stt","conduction","other_hyp","mi")):run("E7_"+key,aux=[j])
    run("E8_masked",ssl=True)
    for f in ("all","beat","features","no_quality"):run("E9_"+f,fusion=f)
    for loss in ("focal","cbfocal","adjust","balanced","mining"):run("imb_"+loss,loss=loss)
    for pp in ("minimal","drift","conventional_fixed","mains","hz100"):run("prep_"+pp,pp=pp,hz=100 if pp=="hz100" else 360)
    run("prep_robust",pp="minimal",robust=True)
    run("aug",augment=True)
    # Sequential forward selection. Only accept a change that improves the development rule.
    current=plain
    for prefix in ("E4","E5","E6","E7","E8","E9","imb","prep","aug"):
        pool=[r for r in rows if r["name"].startswith(prefix) and r["ap"]>plain["ap"]]
        addition=choose(pool,plain) if pool else None
        if not addition:continue
        change={k:v for k,v in addition["config"].items() if BASE.get(k)!=v}
        # Cached morphology templates represent the constrained 360 Hz signal only.
        c=current["config"]|change
        if c.get("fusion") and c["pp"]!="constrained":continue
        candidate=addition if c==addition["config"] else run("E10_"+prefix,**c)
        if candidate["ap"]>current["ap"] and choose([current,candidate],plain)==candidate:current=candidate
    students=[r for r in rows if not r["config"].get("teacher")]
    winner=choose(students,base)
    if not winner: winner=base
    save_json(R/"selection.json",dict(student=winner["name"],teacher=st.teacher,combined=current["name"],rule="mean fold 7/8 patient AP; within .005 maximize sensitivity at 90% specificity, then F1, then fewer parameters and latency; cross-fold Brier <= E0 + .01"))
    # Quality and demographic shortcut probes are development-only diagnostics.
    eligible=st.eligible
    tr=eligible[eligible.strat_fold<=6].row_index.to_numpy()
    va=eligible[eligible.strat_fold.isin(DEV)].row_index.to_numpy()
    nuisance=[]
    for nm,xx in (("quality",array("constrained_features")[:,12:]),("age_sex",st.m[["age","sex"]].to_numpy())):
        probe=make_pipeline(SimpleImputer(),StandardScaler(),LogisticRegression()).fit(xx[tr],st.m.label[tr])
        nuisance.append(dict(probe=nm,**devscore(st.m.iloc[va],probe.decision_function(xx[va]))))
    pd.DataFrame(nuisance).to_csv(R/"shortcut_probes.csv",index=False)
    coverage=[]
    for cut in (0,50,80,100):
        frame=st.m.iloc[va]; use=(frame.label==0)|(frame.lvh_confidence>=cut)
        z=np.load(A/"development_amp"/"E3"/"scores.npz")["logit"]
        coverage.append(dict(cut=cut,records=int(use.sum()),positives=int(frame.label[use].sum()),coverage=use.mean(),**devscore(frame[use],z[use])))
    pd.DataFrame(coverage).to_csv(R/"confidence_coverage.csv",index=False)
    st.m.iloc[eligible.row_index].groupby(["strat_fold","label","lvh_confidence"],dropna=False).size().rename("records").to_csv(R/"confidence_counts.csv")
    final=[]
    for r in {r["name"]:r for r in (teacher,plain,current,winner,base)}.values():
        final+=results(st,r,r["name"],"teacher" if r==teacher else "single_lead")
    pd.DataFrame(final).to_csv(R/"final.csv",index=False)
    print("Final selection:",winner["name"],flush=True)

if __name__=="__main__": main()
