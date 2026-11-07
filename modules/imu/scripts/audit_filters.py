"""Causal design sensitivity with zero-phase analysis controls. No artifact removal."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import pandas as pd
from scipy.signal import sosfilt,sosfiltfilt
from wearable_imu.datasets.sisfall import load_cache,COUNTS_TO_PHYSICAL,RESULTS
from wearable_imu.preprocessing.ipmv import butter
from wearable_imu.evaluation.figures import plt,plotfile


def main():
    m,raw=load_cache()
    records=[]
    for r in m.itertuples():
        x=raw[r.offset:r.offset+r.n]*COUNTS_TO_PHYSICAL
        denom=np.r_[r.peak_g,r.peak_dps]
        for order in (2,4):
            for hz in (20,30,40):
                sos,zi=butter(200,hz,order)
                low=sosfilt(sos,x,axis=0,zi=zi[:,:,None]*x[0])[0]
                res=x-low
                size=np.linalg.norm(res.reshape(-1,2,3),axis=2)
                adapt=low+res*np.repeat(size/(size+[.35,50.]),3,axis=1)
                for mode,a in (("causal",low),("adaptive",adapt),("zero_phase",sosfiltfilt(sos,x,axis=0))):
                    ratio=np.linalg.norm(a.reshape(-1,2,3),axis=2).max(0)/np.maximum(denom,1e-9)
                    records.append((r.id,r.y,order,hz,mode,*ratio))
    f=pd.DataFrame(records,columns=["id","y","order","hz","mode","acc_ratio","gyro_ratio"])
    f.to_csv(RESULTS/"filter_trials.csv.gz",compression="gzip",index=False)
    summary=f.groupby(["y","mode","order","hz"])[["acc_ratio","gyro_ratio"]].quantile([.1,.5,.9])
    summary.index.names=["y","mode","order","hz","quantile"]; summary.to_csv(RESULTS/"filter_peaks.csv")
    fig,axs=plt.subplots(1,2,figsize=(10,4),sharey=True)
    for ax,n in zip(axs,["acc_ratio","gyro_ratio"]):
        for (mode,order),g in f[f.y==1].groupby(["mode","order"]):
            med=g.groupby("hz")[n].median(); ax.plot(med.index,med.values,marker="o",label=f"{mode} / order {order}")
        ax.set(xlabel="Cutoff (Hz)",ylabel="Median peak / native peak",title="Acceleration" if n=="acc_ratio" else "Gyroscope",xticks=[20,30,40])
        ax.axhline(1,color="k",lw=.8)
    axs[1].legend(fontsize=7)
    path=RESULTS/"figures"; path.mkdir(exist_ok=True); plotfile(fig,path/"filter_peak_retention.png")
    print(summary.to_string())

if __name__=="__main__": main()
