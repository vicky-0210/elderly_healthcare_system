"""Visualize per-axis causal gravity estimates and the saved fold training histories."""
import json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
from wearable_imu.datasets.sisfall import RESULTS,COUNTS_TO_PHYSICAL,load_cache
from wearable_imu.preprocessing.ipmv import LowPass
from wearable_imu.evaluation.figures import plt,plotfile
from wearable_imu.training.protocol import RUN


def main():
    manifest,raw=load_cache()
    output=Path("artifacts/imu/reports/sisfall_v1"); output.mkdir(parents=True,exist_ok=True)
    dest=RESULTS/"figures"
    fig,axes=plt.subplots(3,2,figsize=(11,7),sharex="col")
    for col,activity in enumerate(("D14","F10")):
        r=manifest[(manifest.subject=="SA01")&(manifest.activity==activity)].iloc[0]
        x=raw[r.offset:r.offset+r.n]*COUNTS_TO_PHYSICAL
        a=x[:,:3]; low=LowPass(200,30,2).apply(a)
        innovation=a-low; size=np.linalg.norm(innovation,axis=1,keepdims=True)
        smooth=low+innovation*size/(size+.35)
        gravity=LowPass(200,.75,2).apply(smooth); dynamic=smooth-gravity
        magnitude=np.linalg.norm(a,axis=1); signed_jerk=np.diff(magnitude,prepend=magnitude[0])*200
        t=np.arange(len(a))/200
        # Original trial labels supply no point annotation for these traces.
        np.savez_compressed(output/(r.id+"_components.npz"),t=t,calibrated=x,gravity=gravity,dynamic=dynamic,
            gravity_magnitude=np.linalg.norm(gravity,axis=1),dynamic_magnitude=np.linalg.norm(dynamic,axis=1),
            acceleration_magnitude=magnitude,signed_magnitude_jerk=signed_jerk)
        for ax,signal,label in zip(axes[:,col],[a,gravity,dynamic],["Acceleration (g)","Estimated gravity (g)","Dynamic acceleration (g)"]):
            ax.plot(t,signal,lw=.65); ax.set_ylabel(label)
        axes[0,col].set_title(f"{activity} · "+("ADL" if r.y==0 else "Simulated fall"))
        axes[-1,col].set_xlabel("Elapsed time (s)")
    axes[0,0].legend(["X","Y","Z"],ncol=3,fontsize=8)
    plotfile(fig,dest/"gravity_components.png")
    keys=["inception","dm_standard","dm_distilled"]
    fig,axes=plt.subplots(2,len(keys),figsize=(12,6),sharex="col")
    for i,key in enumerate(keys):
        for k in range(5):
            history=json.loads((RUN/key/f"fold{k}"/"history.json").read_text())["inner"]
            epochs=[h["epoch"] for h in history]
            axes[0,i].plot(epochs,[h["loss"] for h in history],".-",label=f"Fold {k+1}")
            axes[1,i].plot(epochs,[h["validation"] for h in history],".-")
        axes[0,i].set_title(key); axes[1,i].set_xlabel("Epoch")
    axes[0,0].set_ylabel("Training objective"); axes[1,0].set_ylabel("Inner subject-balanced accuracy")
    axes[0,-1].legend(fontsize=7); plotfile(fig,dest/"neural_training.png")

if __name__=="__main__": main()
