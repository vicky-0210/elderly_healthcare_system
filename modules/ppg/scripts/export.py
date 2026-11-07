"""Refit the selected architecture on all subjects, then check a static ONNX export."""
import json,sys,time,platform
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
import numpy as np
import torch
from wearable_ppg.datasets.osasud import RUN,RESULT,LABELS,CACHE
from wearable_ppg.datasets.windows import load_views
from wearable_ppg.training.study import train_net,calibration,SEED
from wearable_ppg.evaluation.metrics import choose_threshold
from wearable_ppg.evaluation.compare import join,read_scores
from wearable_ppg.preprocessing.optical import NAMES
from wearable_ppg.inference.predictor import PPGPredictor
from scipy.special import expit

EXPORT=Path("artifacts/ppg/exports/osasud_v1")


def main():
    torch.set_num_threads(2); torch.backends.mha.set_fastpath_enabled(False)
    name=json.loads((RESULT/"selection.json").read_text())["winner"]
    out=EXPORT/name; out.mkdir(parents=True,exist_ok=True)
    x,y,t,ids,ends=load_views()
    meta=[json.loads((RUN/name/f"fold{k}"/"done.json").read_text()) for k in range(5)]
    d=join([read_scores(RUN/name/f"fold{k}","test") for k in range(5)])
    ab=calibration(d["y"],d["logit"]); prob=expit(ab[0]*d["logit"]+ab[1])
    threshold=choose_threshold(d["y"],prob,d["t"],d["patient"],d["arrival"])
    epochs=int(np.median([v["epoch"] for v in meta])); device="cuda" if torch.cuda.is_available() else "cpu"
    net,_,w,center,scale,hist,_=train_net(name,x,y,t,ids,ends,None,epochs,device,SEED)
    model=net.cpu().eval(); torch.save(model.state_dict(),out/"weights.pt")
    example_ends=np.random.default_rng(501).choice(ends,128,replace=False)
    examples=np.arcsinh((x[example_ends[:,None]+np.arange(1-w,1)]-center)/scale)
    tensor=torch.tensor(examples)
    with torch.inference_mode(): reference=model(tensor).numpy()
    torch.onnx.export(model,tensor[:1],str(out/"model.onnx"),input_names=["views"],output_names=["scores"],opset_version=17)
    import onnx
    onnx.checker.check_model(str(out/"model.onnx"))
    runtime=dict(model=name,context=w,hop=5,features=NAMES,labels=LABELS,center=center.tolist(),scale=scale.tolist(),
        calibration=ab,threshold=threshold,epochs=epochs,training_rate_hz=80,target_rate_hz=100,history=hist)
    (out/"runtime.json").write_text(json.dumps(runtime,indent=2))
    engine=PPGPredictor(out); times=[]; actual=[]
    for a in examples:
        start=time.perf_counter(); actual.append(engine.session.run(None,{"views":a[None]})[0][0]); times.append((time.perf_counter()-start)*1e3)
    actual=np.asarray(actual); np.testing.assert_allclose(actual,reference,atol=3e-4,rtol=2e-4)
    # Compare full archive preprocessing to arbitrarily fragmented replay on actual contiguous data.
    from wearable_ppg.datasets.windows import spans
    verify=[]
    for pid in (1,15,28):
        with np.load(CACHE/f"p{pid:02}.npz") as src, np.load(CACHE/"views"/f"p{pid:02}.npz") as feat:
            run=max(spans(src["t"],src["valid"]),key=len)[:180]; segment=src["raw"][run].ravel(); engine.reset()
            rows=[]
            for i in range(0,len(segment),137): rows+=engine.push(segment[i:i+137],i/80)
            end=np.arange(w-1,len(run),5)
            xx=np.arcsinh((feat["x"][run][end[:,None]+np.arange(1-w,1)]-center)/scale)
            pred=np.concatenate([engine.session.run(None,{"views":v[None]})[0][0,:,0] for v in xx])
            expected=expit(ab[0]*pred+ab[1]); got=np.asarray([r["p_event"] for r in rows])
            np.testing.assert_allclose(expected,got,atol=1e-5)
            verify.append(float(np.max(np.abs(expected-got))))
    report=dict(model=name,epochs=epochs,onnx_bytes=(out/"model.onnx").stat().st_size,probe_blocks=len(tensor),
        max_logit_error=float(np.max(np.abs(reference-actual))),stream_probability_errors=verify,
        host=platform.platform(),host_model_p50_ms=float(np.median(times)),host_model_p95_ms=float(np.percentile(times,95)))
    (RESULT/"export_verification.json").write_text(json.dumps(report,indent=2)); print(report)

if __name__=="__main__": main()
