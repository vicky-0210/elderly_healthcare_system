"""VC-BiMamba ablations on the frozen PTB-XL Lead II protocol (reporting only; nothing is selected here).

    .venv/Scripts/python modules/ecg/experiments/lvh_novel/run.py train --output artifacts/ecg/runs/20261111_lvh_vcbimamba_ablations \
        --architectures vc_bimamba_nocond vc_bimamba_unidir vc_bimamba_noconv
    .venv/Scripts/python modules/ecg/experiments/lvh_novel/ablations.py
"""
from pathlib import Path
import sys, json
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(Path(__file__).parent))
import run as base  # registers the VC-BiMamba factories with the shared runner/report
import numpy as np
import pandas as pd
from wearable_ecg.evaluation.metrics import metrics, bootstrap, paired_bootstrap
from wearable_ecg.runtime import save_json

MAIN = ROOT / "artifacts/ecg/runs/20261110_lvh_vcbimamba"
ABL = ROOT / "artifacts/ecg/runs/20261111_lvh_vcbimamba_ablations"
OUT = ROOT / "research/results/vcbimamba_ablations"
VARIANTS = {"vc_bimamba": "Full VC-BiMamba", "vc_bimamba_nocond": "No voltage conditioning",
            "vc_bimamba_unidir": "Forward scan only", "vc_bimamba_noconv": "No CNN stem (Mamba only)"}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows, pairs, seeds = [], [], []
    for task in ("lvh_vs_normal", "lvh_vs_rest"):
        models = {}
        for arch in VARIANTS:
            root = MAIN if arch == "vc_bimamba" else ABL
            for run in sorted((root / task).glob(f"{arch}__constrained__seed*")):
                seeds.append(base.report.evaluate_run(base.CACHE, run, 1000))
            models[arch] = base.ensemble(root / task, arch)
        ref, _ = models["vc_bimamba"]
        for arch, (df, t) in models.items():
            m = metrics(df.label, df.probability, t)
            rows.append({"task": task, "variant": arch, "description": VARIANTS[arch], **m,
                         "ci": bootstrap(df.label.to_numpy(), df.probability.to_numpy(), df.patient_id, t, 1000)})
            if arch != "vc_bimamba":
                assert np.array_equal(ref.ecg_id, df.ecg_id)
                pairs.append({"task": task, "model_a": "vc_bimamba", "model_b": arch,
                              **paired_bootstrap(ref.label.to_numpy(), ref.probability.to_numpy(),
                                                 df.probability.to_numpy(), ref.patient_id.to_numpy())})
    save_json(OUT / "ablation_ensembles.json", rows)
    save_json(OUT / "ablation_paired.json", pairs)
    pd.DataFrame(seeds).to_csv(OUT / "ablation_per_seed.csv", index=False)
    keys = ["task", "variant", "auroc", "auprc", "accuracy", "sensitivity", "specificity", "f1"]
    table = pd.DataFrame([{k: r[k] for k in keys} for r in rows]).round(4)
    table.to_csv(OUT / "ablation_table.csv", index=False)
    print(table.to_string()); print(json.dumps(pairs, indent=1))


if __name__ == "__main__":
    main()
