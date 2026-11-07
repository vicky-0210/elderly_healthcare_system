"""Two-pass weak witness learning; ADL windows are allowed as known negatives."""
import time
import joblib
import numpy as np
from sklearn.preprocessing import RobustScaler,StandardScaler
from wearable_imu.models.classical import descriptors,logits
from . import protocol as p


def estimator(kind,seed):
    if kind=="rf":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=64,max_features="sqrt",min_samples_leaf=2,random_state=seed,n_jobs=2)
    if kind=="svm":
        from sklearn.svm import SVC
        return SVC(C=2.,kernel="rbf",gamma="scale",cache_size=160)
    from xgboost import XGBClassifier
    return XGBClassifier(n_estimators=60,max_depth=3,learning_rate=.12,tree_method="hist",
        n_jobs=2,random_state=seed,objective="binary:logistic")


def make_features(d):
    steps=np.arange(d["width"])
    starts=d["starts"]
    f=np.empty((len(starts),7*len(d["names"])),dtype="f")
    for i in range(0,len(starts),384):
        s=starts[i:i+384]
        f[i:i+len(s)]=descriptors(d["x"][s[:,None]+steps])
    return f


def fit(spec,d,features,rows,seed):
    positives=d["bags"][rows[d["y"][rows]==1]]
    negatives=d["bags"][rows[d["y"][rows]==0]]
    # Seed a latent witness using the native-rate peak channel, never a timestamp.
    n=len(d["names"])
    intensity=features[:,3*n+d["names"].index("a_peak")]
    pos=positives[np.arange(len(positives)),np.argmax(intensity[positives],axis=1)]
    neg=np.unique(negatives[:,np.linspace(0,negatives.shape[1]-1,3).round().astype(int)])
    scaler=(RobustScaler() if spec.scaler=="robust" else StandardScaler()).fit(features[np.unique(d["bags"][rows])])
    def train(pos_ids,neg_ids):
        model=estimator(spec.model,seed)
        y=np.r_[np.ones(len(pos_ids)),np.zeros(len(neg_ids))].astype(int)
        x=scaler.transform(features[np.r_[pos_ids,neg_ids]])
        model.fit(x,y,sample_weight=np.where(y==1,len(y)/(2*len(pos_ids)),len(y)/(2*len(neg_ids))))
        return model
    first=train(pos,neg)
    unique,undo=np.unique(d["bags"][rows],return_inverse=True)
    s=logits(first,scaler.transform(features[unique]))[undo].reshape(len(rows),-1)
    pos=positives[np.arange(len(positives)),s[d["y"][rows]==1].argmax(1)]
    hard=negatives[np.arange(len(negatives)),s[d["y"][rows]==0].argmax(1)]
    return train(pos,np.unique(np.r_[neg,hard])),scaler


def predict(bundle,features):
    model,scaler=bundle
    return logits(model,scaler.transform(features))


def run_group(specs,root):
    if all(p.complete(root/s.name/f"fold{k}",s) for s in specs for k in range(5)): return
    d=p.make_data(specs[0]); f=make_features(d)
    for outer in range(5):
        train,val,refit,test=p.partitions(d["folds"],outer)
        for spec in specs:
            folder=root/spec.name/f"fold{outer}"
            if p.complete(folder,spec): continue
            start=time.perf_counter()
            folder.mkdir(parents=True,exist_ok=True)
            inner=fit(spec,d,f,train,p.SEED+outer)
            vl=predict(inner,f[np.isin(d["group"],val)])
            bag=p.trial_max(d,val,vl)
            score,thr=p.operating_point(d["y"][val],bag,d["subjects"][val])
            ab=p.calibration(d["y"][val],bag,d["subjects"][val])
            model=fit(spec,d,f,refit,p.SEED+outer)
            tl=predict(model,f[np.isin(d["group"],test)])
            path=folder/"model.joblib"; joblib.dump(model,path,compress=3)
            p.write_output(folder,spec,d,outer,val,test,vl,tl,score,thr,ab,
                dict(bytes=path.stat().st_size,parameters=None,wall_seconds=time.perf_counter()-start))
            print(f"{spec.name} fold {outer}: validation {score:.4f}, {time.perf_counter()-start:.1f}s",flush=True)
