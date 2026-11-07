import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,ConfusionMatrixDisplay
from sklearn.calibration import calibration_curve
from wearable_imu.datasets.sisfall import load_cache,COUNTS_TO_PHYSICAL


def plotfile(fig,target):
    fig.tight_layout()
    fig.savefig(target,dpi=150,bbox_inches="tight")
    plt.close(fig)


def figures(models,folds,panels,m,out):
    folder=out/"figures"; folder.mkdir(exist_ok=True)
    prim=models[models.role=="primary"].sort_values("subject_ba")
    fig,ax=plt.subplots(figsize=(8.5,7))
    ax.hlines(prim.model,prim.subject_ba,prim.inner_score,color="#94a3b8")
    ax.scatter(prim.subject_ba,prim.model,label="Held-out participants",color="#075985")
    ax.scatter(prim.inner_score,prim.model,label="Inner selection",marker="x",color="#c2410c")
    ax.set(xlabel="Subject-balanced accuracy",title="22 candidate models on shared two-second windows",xlim=(.45,1.01))
    ax.legend(loc="lower right"); plotfile(fig,folder/"model_comparison.png")
    fig,axs=plt.subplots(1,3,figsize=(13,4))
    for k,label in (("fixed_all","Selected model"),("fixed_neural","Selected neural"),("dm_tcn","DM-TCN")):
        r=panels[k]; f,t,_=roc_curve(r.y,r.p_fall); axs[0].plot(f,t,label=label)
        pr,se,_=precision_recall_curve(r.y,r.p_fall); axs[1].plot(se,pr)
        obs,pred=calibration_curve(r.y,r.p_fall,n_bins=10,strategy="quantile"); axs[2].plot(pred,obs,".-")
    axs[0].set(xlabel="False-positive rate",ylabel="Sensitivity",title="ROC"); axs[0].legend(fontsize=8)
    axs[1].set(xlabel="Recall",ylabel="Precision",title="Precision–recall")
    axs[2].plot([0,1],[0,1],"k--",alpha=.5); axs[2].set(xlabel="Predicted trial probability",ylabel="Observed fraction",title="Reliability")
    plotfile(fig,folder/"discrimination_calibration.png")
    fig,axs=plt.subplots(1,2,figsize=(9,4))
    for ax,k,title in zip(axs,["nested_all","nested_neural"],["Nested selection, all candidates","Nested selection, neural menu"]):
        r=panels[k]; ConfusionMatrixDisplay.from_predictions(r.y,r.decision,display_labels=["ADL","Fall"],cmap="Blues",colorbar=False,ax=ax)
        ax.set_title(title)
    plotfile(fig,folder/"confusion.png")
    ab=models[models.role=="ablation"].sort_values("model")
    fig,axs=plt.subplots(1,2,figsize=(12,8),sharex=True)
    for ax,ix in zip(axs,np.array_split(np.arange(len(ab)),2)):
        block=ab.iloc[ix]; ax.scatter(block.subject_ba,block.model,c="#0f766e"); ax.set_xlabel("Held-out subject-balanced accuracy"); ax.grid(axis="x",alpha=.2)
    plotfile(fig,folder/"ablations.png")
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for y,name in ((0,"ADL"),(1,"Fall")):
        r=m[m.y==y]; axs[0].hist(r.seconds,bins=np.arange(9,32,.5),alpha=.65,label=name)
        axs[1].hist(r.peak_g,bins=np.arange(0,22,.4),alpha=.55,label=name)
    axs[0].set(xlabel="Trial duration (s), 9–32 s detail",ylabel="Trials",title="Protocol-dependent recording lengths"); axs[0].legend()
    axs[1].set(xlabel="Native peak acceleration (g)",ylabel="Trials",title="Impact alone does not separate all trials")
    plotfile(fig,folder/"dataset_profile.png")
    _,counts=load_cache(); fig,axs=plt.subplots(2,2,figsize=(12,5),sharex="col")
    for col,activity in enumerate(("D11","F01")):
        row=m[(m.activity==activity)&(m.subject=="SA01")].iloc[0]
        x=counts[row.offset:row.offset+row.n]*COUNTS_TO_PHYSICAL
        for ax,s,val in zip(axs[:,col],["Acceleration (g)","Rotation (deg/s)"],[x[:,:3],x[:,3:]]):
            ax.plot(np.arange(len(x))/200,np.linalg.norm(val,axis=1),lw=.7); ax.set_ylabel(s)
        axs[0,col].set_title(row.id+" · recording label only"); axs[1,col].set_xlabel("Elapsed time (s)")
    plotfile(fig,folder/"native_signals.png")
