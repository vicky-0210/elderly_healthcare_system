"""NumPy runtime for the fitted binary tree ensembles and RBF-SVM."""
import numpy as np
from wearable_imu.models.classical import descriptors


class PortableClassifier:
    def __init__(self,path):
        with np.load(path,allow_pickle=False) as data:
            self.data={k:data[k] for k in data.files}
        self.kind=str(self.data["kind"])

    def __call__(self,windows):
        d=self.data
        f=descriptors(np.asarray(windows,dtype="f"))
        f-=d["center"]; f/=d["scale"]
        if self.kind=="svm":
            support=d["support"]; x=f.astype(float)
            dist=np.maximum((x*x).sum(1)[:,None]+d["support_norm"]-2*x@support.T,0)
            return (np.exp(-d["gamma"]*dist)@d["dual"]+d["bias"]).ravel()
        loc=np.broadcast_to(d["root"],(len(f),len(d["root"]))).copy()
        row=np.arange(len(f))[:,None]
        for _ in range(int(d["depth"])):
            leaf=d["left"][loc]<0
            feature=f[row,d["feature"][loc]]
            branch=feature<d["value"][loc] if self.kind=="xgb" else feature<=d["value"][loc]
            loc=np.where(leaf,loc,np.where(branch,d["left"][loc],d["right"][loc]))
        if self.kind=="xgb":
            z=np.sum(d["value"][loc],axis=1,dtype=np.float32)+d["bias"]
            # Match clipping in the research predictor.
            from scipy.special import expit
            q=np.clip(expit(z.astype("f")),1e-6,1-1e-6)
        else:
            q=np.clip(d["value"][loc].mean(1),1e-6,1-1e-6)
        return np.log(q/(1-q)).astype("f")
