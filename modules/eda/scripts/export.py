"""Refit the predeclared zero-calibration deployment choice and freeze its contract."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import numpy as np

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODULE/"src"))
from wearable_eda.evaluation.loso import load_runs, require_complete, select


def export(runs, cache, target, choice, calibration):
    loaded = load_runs(runs)
    require_complete(loaded)
    if choice == "neural":
        candidates = {k:r for k,r in loaded.items() if r.family=="neural"}
        rule = "highest_inner"
    else:
        candidates, rule = loaded, "lightweight"
    choices,_ = select(candidates, calibration)
    name = next(c["selected"] for c in choices if c["subject"]=="refit" and c["rule"]==rule)
    record = loaded[name]
    family, experiment = name.split("/")
    target.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(dir=target) as tmp:
        stage = Path(tmp)
        if family=="classical":
            from wearable_eda.training.classical_run import export_experiment
            weights = stage/"model.joblib"
            bundle = export_experiment(cache,runs/name,weights)
            if record.meta["model"]=="logistic":
                from wearable_eda.datasets.wesad import build_windows
                from wearable_eda.inference.linear import LinearPredictor
                from wearable_eda.training.classical import predict_windows
                transformer,model = bundle["fitted"]
                imputer,scaler = transformer.steps[0][1],transformer.steps[1][1]
                frozen = dict(features=bundle["feature_names"],impute=imputer.statistics_.tolist(),
                        center=scaler.center_.tolist(),scale=scaler.scale_.tolist(),weights=model.coef_[0].tolist(),
                        intercept=float(model.intercept_[0]),temperature=bundle["temperature"])
                weights = stage/"linear.json"
                weights.write_text(json.dumps(frozen,indent=2,allow_nan=False))
                data = build_windows(cache,modalities=record.modalities,calibration=calibration)
                args = data["F"],data["features"]
                np.testing.assert_allclose(LinearPredictor(weights).predict_windows(*args),
                                           predict_windows(bundle,*args),rtol=1e-10,atol=1e-12)
        else:
            from wearable_eda.training.neural_suite import export_full
            weights = stage/"model.pt"
            export_full(cache,runs,experiment,weights,device="cpu")
        contract = dict(format="wesad-wrist-v1",family=family,experiment=name,selection=f"{choice}/{rule}",model=record.meta["model"],
                        classes=[1,2],modalities=record.modalities,calibration=calibration,decomposition="mcsaed",
                        filtering=record.meta.get("filtering","conservative"),quality=record.meta.get("quality","auto"),
                        seconds=60,stride=15,training_subjects=list(record.folds.index.astype(int)),
                        score_semantics=["baseline","stress_associated_arousal"],
                        payload=weights.name,
                        weights_sha256=hashlib.sha256(weights.read_bytes()).hexdigest())
        if weights.name=="linear.json":
            contract["reference_sha256"] = hashlib.sha256((stage/"model.joblib").read_bytes()).hexdigest()
        (stage/"contract.json").write_text(json.dumps(contract,indent=2))
        from wearable_eda.inference.predictor import EDAPredictor
        EDAPredictor().load(stage)
        if weights.name=="linear.json":
            (stage/"model.joblib").replace(target/"model.joblib")
        weights.replace(target/weights.name)
        (stage/"contract.json").replace(target/"contract.json")
    print(f"Refit {name} on 15 participants; saved {target.resolve()}")


if __name__=="__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs",type=Path,default=Path("artifacts/eda/runs/wesad_loso_v1"))
    p.add_argument("--cache",type=Path,default=Path("data/eda/processed/wesad_v1"))
    p.add_argument("--choice",choices=["lightweight","neural"],default="lightweight")
    p.add_argument("--calibration",choices=["zero","rest"],default="zero")
    p.add_argument("--out",type=Path,help="Default is a separate export directory for each choice.")
    args = p.parse_args()
    stem = "wesad_loso_v1" if args.choice=="lightweight" else "wesad_loso_v1_neural"
    if args.calibration=="rest":
        stem += "_rest"
    out = args.out or Path("artifacts/eda/exports")/stem
    export(args.runs,args.cache,out,args.choice,args.calibration)
