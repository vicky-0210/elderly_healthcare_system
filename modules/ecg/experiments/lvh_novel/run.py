"""VC-BiMamba Lead II LVH study on the same PTB-XL folds, recipe and cache as the 2026-11-05 comparison.

Phases (run in order):
  train     development (folds 1-6 train, 7-8 epoch selection), refit 1-8, calibrate on 9; then freeze
  dev       summarise development-fold metrics only (no test access)
  evaluate  single frozen fold-10 evaluation, paired comparison with the 2026-11-05 CNN ensemble

The primary model (vc_bimamba) is fixed in experiment.json before any training; the
other variants are ablations. Fold 10 is never used to choose a model or epoch.
"""
from pathlib import Path
import sys, json, argparse
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "modules/ecg/src"))
import numpy as np
import pandas as pd
import torch
from wearable_ecg.training import runner
from wearable_ecg.evaluation import report
from wearable_ecg.models import backbones, vcmamba
from wearable_ecg.evaluation.metrics import metrics, bootstrap, paired_bootstrap, choose_threshold
from wearable_ecg.runtime import save_json, digest

CACHE = ROOT / "data/ecg/processed/ptbxl_360hz_v1"
BASELINE = ROOT / "artifacts/ecg/runs/20261105_lvh_comparison"


def make_model(architecture, features=None, **options):
    if architecture in vcmamba.VARIANTS: return vcmamba.make_novel(architecture, features)
    return backbones.make_model(architecture, features, **options)


def classifier(**config):
    if config["architecture"] in vcmamba.VARIANTS: return vcmamba.build(config)
    return backbones.ECGClassifier(**config)


runner.make_model = make_model
report.ECGClassifier = classifier


def train(root, config):
    root.mkdir(parents=True, exist_ok=True)
    if (root / "TEST_EVALUATION_STARTED.json").exists(): raise RuntimeError("Test already opened")
    save_json(root / "experiment.json", config)
    runs = []
    for task in config["tasks"]:
        for architecture in config["architectures"]:
            for seed in config["seeds"]:
                runs.append(str(runner.train_run(CACHE, root, task, architecture, seed, config["training"])))
    save_json(root / "FROZEN_FOR_TEST.json", {"runs": runs, "configuration_sha256": digest(root / "experiment.json"),
              "primary": config["primary"], "rule": "Frozen before fold-10 scoring."})


def dev_summary(root, config):
    rows = []
    for task in config["tasks"]:
        archs = config["architectures"] + ["cnn", "cnn_transformer", "cnn_ssm"]
        for architecture in archs:
            base = root if architecture in config["architectures"] else BASELINE
            for run in sorted((base / task).glob(f"{architecture}__constrained__seed*")):
                done = json.loads((run / "COMPLETE.json").read_text())
                h = pd.read_csv(run / "development/history.csv")
                best = h.loc[h.epoch == done["development"]["best_epoch"]].iloc[0]
                rows.append({"task": task, "architecture": architecture, "seed": run.name.split("seed")[-1],
                             "dev_auroc": best.dev_auroc, "dev_auprc": best.dev_auprc,
                             "dev_sens90": best.dev_sensitivity_at_90spec, "parameters": done["refit"]["parameters"]})
    df = pd.DataFrame(rows)
    summary = df.groupby(["task", "architecture"])[["dev_auroc", "dev_auprc", "dev_sens90", "parameters"]].agg(["mean", "std"]).round(4)
    summary.to_csv(root / "development_summary.csv")
    print(summary.to_string())


def ensemble(task_dir, architecture):
    runs = sorted(task_dir.glob(f"{architecture}__constrained__seed*"))
    preds = [pd.read_csv(r / "test/predictions.csv").sort_values("ecg_id").reset_index(drop=True) for r in runs]
    cals = [pd.read_csv(r / "calibration_predictions.csv").sort_values("ecg_id").reset_index(drop=True) for r in runs]
    df = preds[0][["ecg_id", "patient_id", "label"]].copy()
    df["probability"] = np.mean([p.probability for p in preds], axis=0)
    threshold = choose_threshold(cals[0].label, np.mean([c.probability for c in cals], axis=0))
    return df, threshold


def evaluate(root):
    frozen = json.loads((root / "FROZEN_FOR_TEST.json").read_text())
    config = json.loads((root / "experiment.json").read_text())
    if digest(root / "experiment.json") != frozen["configuration_sha256"]: raise ValueError("Changed after freeze")
    save_json(root / "TEST_EVALUATION_STARTED.json", {"purpose": "One frozen fold-10 evaluation; no retuning."})
    rows = [report.evaluate_run(CACHE, run, 1000) for run in frozen["runs"]]
    per_seed = pd.DataFrame(rows)
    base = pd.read_csv(BASELINE / "comparison_per_seed.csv")
    base = base[base.architecture.isin(["cnn", "cnn_transformer", "cnn_ssm"])]
    allrows = pd.concat([per_seed, base], ignore_index=True)
    allrows.to_csv(root / "comparison_per_seed.csv", index=False)
    keys = ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "ppv", "npv", "f1", "brier", "parameters"]
    summary = allrows.groupby(["task", "architecture"])[keys].agg(["mean", "std"])
    summary.columns = ["_".join(c) for c in summary.columns]
    summary.reset_index().to_csv(root / "comparison_summary.csv", index=False)
    ens_rows, pairs = [], []
    for task in config["tasks"]:
        models = {}
        for a in config["architectures"]: models[a] = ensemble(root / task, a)
        for a in ("cnn", "cnn_transformer"): models[a] = ensemble(BASELINE / task, a)
        for a, (df, t) in models.items():
            m = metrics(df.label, df.probability, t)
            ens_rows.append({"task": task, "architecture": a, **m,
                             "ci": bootstrap(df.label.to_numpy(), df.probability.to_numpy(), df.patient_id, t, 1000)})
            df.to_csv(root / task / f"{a}_ensemble_predictions.csv", index=False)
        p, _ = models[config["primary"]]
        for other in [a for a in models if a != config["primary"]]:
            q, _ = models[other]
            assert np.array_equal(p.ecg_id, q.ecg_id)
            pairs.append({"task": task, "model_a": config["primary"], "model_b": other,
                          **paired_bootstrap(p.label.to_numpy(), p.probability.to_numpy(), q.probability.to_numpy(), p.patient_id.to_numpy())})
    save_json(root / "ensemble_metrics.json", ens_rows)
    save_json(root / "paired_ensemble_bootstrap.json", pairs)
    show = summary[[f"{k}_mean" for k in ["auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]]].round(4)
    print(show.to_string())
    print(pd.DataFrame([{k: r[k] for k in ["task", "architecture", "auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]} for r in ens_rows]).round(4).to_string())
    print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase", choices=["train", "dev", "evaluate"])
    p.add_argument("--output", type=Path, default=ROOT / "artifacts/ecg/runs/20261110_lvh_vcbimamba")
    p.add_argument("--architectures", nargs="+", default=list(vcmamba.VARIANTS))
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    p.add_argument("--tasks", nargs="+", default=["lvh_vs_normal", "lvh_vs_rest"])
    a = p.parse_args()
    baseline_cfg = json.loads((BASELINE / "experiment.json").read_text())
    config = {"experiment": "ptbxl_1_0_3_lead_ii_lvh_vcbimamba_20261110", "primary": "vc_bimamba",
              "tasks": a.tasks, "architectures": a.architectures, "seeds": a.seeds,
              "preprocessing": "constrained", "training": baseline_cfg["training"],
              "baseline_experiment": str(BASELINE)}
    if a.phase == "train": train(a.output, config)
    elif a.phase == "dev": dev_summary(a.output, json.loads((a.output / "experiment.json").read_text()))
    else: evaluate(a.output)
