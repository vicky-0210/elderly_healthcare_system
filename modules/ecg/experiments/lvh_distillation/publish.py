"""Package the selected Lead-II student, verify raw-input parity, and write aggregate reports."""
import shutil
from study import *
from wearable_ecg.models.backbones import ECGClassifier
from wearable import WearableStudent


class Logit(torch.nn.Module):
    def __init__(self,model,original):
        super().__init__();self.model=model;self.original=original

    def forward(self,wave,features,beat):
        if self.original:return self.model(wave[:,0],features)
        return self.model(wave,features,beat)[0]


class BeatPool(torch.nn.Module):
    """Static ONNX equivalent of AdaptiveMaxPool1d(6) for 46 beat positions."""
    def forward(self,x):
        return torch.stack([x[:,:,i*46//6:((i+1)*46+5)//6].amax(-1) for i in range(6)],dim=-1)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import onnx
    sel=json.loads((R/"selection.json").read_text());name=sel["student"]
    folder=Path("artifacts/ecg/exports/lvh_distill_v1")/name
    folder.mkdir(parents=True,exist_ok=True)
    if name=="E0":
        ck=torch.load(A/"lvh_vs_rest/cnn__constrained__seed42/refit/model.pt",weights_only=True,map_location="cpu")
        c=BASE;model=ECGClassifier(**ck["model_config"]);model.load_state_dict(ck["state_dict"])
    else:
        ref=A/"refit"/name
        c=json.loads((ref/"done.json").read_text())["config"]
        model=MorphNet(c);model.load_state_dict(torch.load(ref/"model.pt",map_location="cpu",weights_only=True))
    if c.get("fusion"):model.beat[2]=BeatPool()
    normalization={}
    for key,v in model.named_buffers():
        if key in ("offset","spread","center","scale","feature_center","feature_scale"):
            normalization[key]=v.detach().numpy().tolist()
    model=Logit(model.eval(),name=="E0").eval()
    patient_cal=json.loads((A/"calibration"/(name+".json")).read_text())
    from run import original
    if name=="E0":
        ids,z=original(Study(),True)
    else:
        scores=np.load(A/"refit"/name/"scores.npz");ids,z=scores["ids"],scores["logit"]
    frame=pd.read_csv(C/"manifest.csv").iloc[ids];val=frame.strat_fold.values==9
    cal=fit_calibration(z[val].astype(float),frame.label.values[val])
    cal["threshold"]=choose_threshold(frame.label.values[val],calibrated(z[val],cal),.90)
    cal["aggregation"]="single ten-second record"
    save_json(R/"wearable_window_metrics.json",measure(frame.label.values[~val],calibrated(z[~val],cal),cal["threshold"]))
    hz=c["hz"];pp=c["pp"]
    if pp=="constrained":
        for file in ("controller.joblib","controller.json"):
            shutil.copyfile(C/"preprocessing"/file,folder/file)
    from math import gcd
    import pywt
    up=hz//gcd(hz,500); down=500//gcd(hz,500); cut=max(up,down)
    save_json(folder/"frontend.json",dict(normalization=normalization,source_hz=500,model_hz=hz,
        resample_up=up,resample_down=down,
        resample_fir_before_polyphase_padding=(up*signal.firwin(20*cut+1,1/cut,window=("kaiser",5.))).tolist(),
        delineation_sos=signal.butter(2,[5,25],fs=360,btype="bandpass",output="sos").tolist(),
        fixed_bandpass_sos=signal.butter(3,[.5,100],fs=360,btype="bandpass",output="sos").tolist(),
        wavelet_filters=pywt.Wavelet("sym4").filter_bank, drift_boxcar_length=217,
        note="Only filters in the selected preprocessing path are applied. Constrained correction uses record-adaptive PR spline, conditional least-squares sinusoid subtraction and bounded SWT when selected by controller."))
    inputs=(torch.zeros(1,1,hz*10),torch.zeros(1,16),torch.zeros(1,1,144))
    torch.onnx.export(model,inputs,str(folder/"model.onnx"),input_names=["wave","features","beat"],output_names=["logit"],opset_version=17)
    onnx.checker.check_model(str(folder/"model.onnx"))
    save_json(folder/"contract.json",{**c,**cal,"patient_aggregation_calibration":patient_cal,"name":name,"duration_s":10,"hop_s":2,"units":"mV","lead":"II",
        "input_shape":[1,1,10*hz],"source_rates_hz":[500,hz],"missing_fraction_allowed":0,"polarity":"LL positive relative to RA; third electrode reference",
        "adc":"mV = polarity * (counts - bench_zero_count) * measured_input_mV_per_count",
        "raw_rate_conversion":"scipy.signal.resample_poly Kaiser beta 5; default zero padding; whole ten-second window",
        "hardware_validation":"pending BioAmp and physical Orin Nano"})
    eng=WearableStudent(folder)
    frame=pd.read_csv(C/"lvh_vs_rest_manifest.csv");ids=frame[frame.strat_fold<=6].row_index.to_numpy()[::225]
    import wfdb
    manifest=pd.read_csv(C/"manifest.csv")
    source=Path(json.loads((C/"COMPLETE.json").read_text())["source"])
    waves=DATA[pp]();feat=array("constrained_features");beats=np.load(A/"beats.npy",mmap_mode="r")
    errors=[]
    for i in ids:
        x=np.array(waves[i]).reshape(1,1,-1)
        with torch.no_grad():z=model(torch.tensor(x),torch.tensor(feat[i:i+1]),torch.tensor(beats[i:i+1,None])).item()
        raw,fields=wfdb.rdsamp(str(source/manifest.filename_hr.iloc[i]))
        p=eng.predict(raw[:,fields["sig_name"].index("II")],np.arange(5000)/500,500)
        if p["score"] is not None:errors.append(abs(p["logit"]-z))
    if not errors or max(errors)>.02:raise RuntimeError(f"raw chain parity failure {errors}")
    assert eng.predict(np.zeros(5000),np.arange(5000)/500,500)["score"] is None
    save_json(R/"export.json",dict(model=name,onnx_bytes=(folder/"model.onnx").stat().st_size,checked=len(errors),attempted=len(ids),max_raw_chain_logit_error=max(errors)))
    dev=pd.read_csv(R/"development.csv")
    fig,ax=plt.subplots(figsize=(9,max(8,len(dev)*.23)))
    dev.sort_values("ap").plot.barh(x="name",y="ap",ax=ax,legend=False,color="#327e84")
    ax.set(xlabel="Mean patient average precision on development folds 7 and 8",ylabel="")
    fig.tight_layout();fig.savefig(R/"ablation.png",dpi=140);plt.close(fig)
    final=pd.read_csv(R/"final.csv")
    tab=final[(final.unit=="patient")&(final.target_spec==.9)]
    cols=["name","auprc","auroc","sensitivity","specificity","f1","ppv","npv","accuracy","balanced_accuracy","mcc","brier","ece_10bins"]
    def md(df):
        return "\n".join(["| "+" | ".join(df.columns)+" |","|"+"---|"*len(df.columns)]+["| "+" | ".join(str(i) for i in row)+" |" for row in df.round(4).itertuples(index=False,name=None)])
    report=f"""# Lead-II LVH-associated ECG-pattern study

The frozen development rule selected **{name}** for single-channel inference and **{sel['teacher']}** as the privileged teacher. The forward-selected combination was **{sel['combined']}**. The endpoint is the PTB-XL LVH statement against eligible remaining ECGs. Repeated ECGs for each patient use maximum logit and an any-positive reference. The original seed-42 CNN reproduced development AP and sensitivity exactly.

## Official fold 10

Thresholds target 90% specificity on fold 9. These are achieved patient-level results on fold 10. This previously studied fold provides an internal follow-up comparison.

{md(tab[cols])}

[All patient operating points and record-level metrics](final.csv) · [Development ranking](development.csv) · [Ablation figure](ablation.png)

## Training and selection

One seed per challenger, seven complete training epochs with AdamW (0.001 learning rate and weight decay), batch size up to 128, gradient norm limit 2. E0 retained its original optimizer and epoch policy. Folds 1–6 train each challenger, 7 and 8 supply the mean patient AP. Each development fold's Brier uses calibration fitted on the other development fold. Specificity thresholds for the development tie-break are empirical ROC operating points. For refit, folds 1–8 train weights and fold 9 fits positive-slope logistic calibration and the operating thresholds. No representation learning uses folds 9–10. The equivalence margin is .005 AP, with sensitivity at 90% specificity, F1, size and amortized batch inference time as tie-breakers. The Brier tolerance is E0 + .01 and the quality-input permutation AP loss limit is .015 for fusion challengers.

E1 uses a deeper residual multiscale encoder with physical-voltage features. E2 compares plain channel concatenation, residual Inception-style processing and shared lead patches with identity embeddings and cross-lead attention. All teachers have masked statement losses and random lead dropout while always retaining II. E3 uses the plain single-channel residual family. E4 tests temperature 1 or 3 and KD coefficient .2 or .6. E5 adds cosine representation alignment (.15) and compares confidence weighting with its removal. The teacher gate uses twice the distance of the calibrated teacher probability from .5. Teacher logits for training students are in-sample, temperature-scaled Bernoulli targets. This can transmit teacher overconfidence. Both latent vectors are 48-dimensional. No patient demographics enter the networks.

E6 compares hard labels, positive-example likelihood weighting, likelihood/100 soft targets, and training with positive likelihood cutoffs 50 or 80. Negatives retain unit weight and zero target. The 100-likelihood validation analysis is a sensitivity subset. The original LVH/NORM contradiction is excluded, and there were no present LVH codes with unknown zero likelihood in this release. Filtering by annotation confidence changes case mix and is unavailable on wearable inputs. [Exact fold counts](confidence_counts.csv) · [Coverage versus performance](confidence_coverage.csv)

E7 tests ST–T, conduction, other hypertrophy and infarction groups separately. Both LVH and VCLVH are excluded from the group definitions. Absent group statements are annotation negatives; present statements with likelihood below 50 are masked. Each auxiliary loss has coefficient .2. E8 uses two development-training-only masked patch reconstruction epochs, MSE on the hidden 25% of patches. E9 combines raw waveform, median aligned beat, training-standardized voltage measurements and a quality gate. Beat-only, feature-only and fixed-gate variants retain the raw branch. Landmarks, QRS spans, ST and T measurements are automatic estimates. E10 only admits individually improving components, then tests combined configurations using the frozen development rule. Each checkpoint directory records its effective configuration and training losses.

Imbalance comparisons use positive-class BCE weighting, focal gamma 2, effective-number focal with beta .9999, log-odds adjusted BCE, balanced random sampling and a score-weighted sampler for difficult eligible negatives. All use the same patient partitions. Minimal, drift-corrected, fixed band-pass, conditional mains removal, training-fitted robust scaling and the original constrained preprocessing were compared. The 100 Hz experiment anti-aliases the calibrated 500 Hz source directly to 100 Hz and trains at that resolution. Synthetic gain, noise, offset, clock jitter, shifts, quiet-segment holds, quantization and clipping were tested as one augmentation package. Clean and perturbed development AP appear in the development table. The clipping audit uses extrema dwell as a proxy because original ADC rails are unavailable. [Waveform fidelity](preprocessing.csv) · [Shortcut-only probes](shortcut_probes.csv)

## Wearable contract

A three-electrode Lead-II approximation requires a verified RA negative and LL positive orientation and a reference electrode. The BioAmp EXG Pill analog output goes through an MCU or dedicated ADC. Timestamped counts, measured offset, verified polarity and calibrated input-referred mV/count are required. The Jetson Orin Nano receives digital samples. The standalone predictor uses a ten-second buffered window, accepts zero missing samples, and withholds scores on clock gaps, clipping or flat input. The bundle includes its fold-9 threshold, calibration, a fixed-shape ONNX student and any required fitted controller. The preprocessing implementation and serialized parameters determine the exact filters. The default polyphase FIR uses a Kaiser window with beta 5. Physical hardware sensitivity, power and latency await a paired device study. [Raw-input parity](export.json) · [Local student bundle](../../../artifacts/ecg/exports/lvh_distill_v1/{name}/)

Run the study from the repository root with the existing ECG virtual environment, first running modules/ecg/experiments/lvh_distillation/run.py and then publish.py. Individual phase markers permit resumption of this exact experiment. Raw waveforms and patient-level outputs stay local. [PTB-XL release and SCP likelihood semantics](https://physionet.org/content/ptb-xl/1.0.3/)
"""
    (R/"REPORT.md").write_text(report,encoding="utf-8")
    print(tab[cols].to_string(index=False))

if __name__=="__main__":main()
