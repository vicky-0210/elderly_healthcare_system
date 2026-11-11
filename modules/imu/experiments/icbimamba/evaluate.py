"""Score IC-BiMamba outer folds (ensemble and single seeds) and compare with the sisfall_v1 models.

    .venv/Scripts/python modules/imu/experiments/icbimamba/evaluate.py --tag main
"""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from wearable_imu.training.catalogue import Experiment
from wearable_imu.training import protocol as p
from wearable_imu.evaluation.compare import metrics, bootstrap
from wearable_imu.inference.alerts import AlertPolicy

RUNS = Path("artifacts/imu/runs")
RESULTS = Path("research/results/icbimamba_v1")
BASELINES = {"xgb_native_100": RUNS / "sisfall_v1/native_100", "dm_tcn_standard": RUNS / "sisfall_v1/dm_standard",
             "inception": RUNS / "sisfall_v1/inception", "dm_distilled": RUNS / "sisfall_v1/dm_distilled"}


def alerts(z, t, thr):
    pol = AlertPolicy(thr)
    return sum(pol.step(s, h) for s, h in zip(z, t))


def score_fold(d, outer, val_logits, test_logits):
    tr, val, refit, test = p.partitions(d["folds"], outer)
    vbag = p.trial_max(d, val, val_logits)
    inner, thr = p.operating_point(d["y"][val], vbag, d["subjects"][val])
    ab = p.calibration(d["y"][val], vbag, d["subjects"][val])
    bag = p.trial_max(d, test, test_logits)
    r = d["m"].iloc[test][["id", "subject", "activity", "trial", "older", "y", "seconds", "fold"]].copy()
    r["logit"], r["p_fall"], r["decision"] = bag, p.calibrated(bag, ab), (bag >= thr).astype(int)
    wi = np.flatnonzero(np.isin(d["group"], test))
    g = d["group"][wi]; end = (d["starts"][wi] - d["offs"][g] + d["width"]) / 20
    cuts = np.r_[0, np.flatnonzero(np.diff(g)) + 1, len(g)]
    r["alerts"] = [alerts(test_logits[a:b], end[a:b], thr) for a, b in zip(cuts[:-1], cuts[1:])]
    r["monitored"] = [end[b - 1] - end[a] for a, b in zip(cuts[:-1], cuts[1:])]
    return r, dict(fold=outer, inner_ba=inner, threshold=thr)


def baseline(folder):
    return pd.concat([pd.read_csv(folder / f"fold{k}/test.csv") for k in range(5)], ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="main"); args = ap.parse_args()
    run = RUNS / "icbimamba_v1" / args.tag
    cfg = json.loads((run / "config.json").read_text())
    d = p.make_data(Experiment("icbimamba", "icbimamba", seconds=4, overlap=.75, scaler="standard"))
    out = RESULTS / args.tag; out.mkdir(parents=True, exist_ok=True)
    ens, per_seed, folds = [], {s: [] for s in cfg["seeds"]}, []
    for outer in range(5):
        packs = {s: np.load(run / f"fold{outer}_seed{s}.npz") for s in cfg["seeds"]}
        r, info = score_fold(d, outer, np.mean([v["val"] for v in packs.values()], 0),
                             np.mean([v["test"] for v in packs.values()], 0))
        ens.append(r); folds.append(dict(info, epochs=[int(v["epoch"]) for v in packs.values()]))
        for s, v in packs.items(): per_seed[s].append(score_fold(d, outer, v["val"], v["test"])[0])
    r = pd.concat(ens, ignore_index=True)
    assert len(r) == 4500 and not r.id.duplicated().any()
    r.to_csv(out / "oof_trials.csv", index=False)
    summary = dict(model="IC-BiMamba ensemble", seeds=cfg["seeds"], config=cfg["cfg"], folds=folds,
                   parameters_per_member=int(np.load(run / f"fold0_seed{cfg['seeds'][0]}.npz")["parameters"]),
                   ensemble=metrics(r), subject_ba_95ci=bootstrap(r))
    rows = [dict(model="IC-BiMamba (5-seed ensemble)", **summary["ensemble"])]
    for s, parts in per_seed.items():
        rows.append(dict(model=f"IC-BiMamba seed {s}", **metrics(pd.concat(parts, ignore_index=True))))
    paired = {}
    for name, folder in BASELINES.items():
        b = baseline(folder).set_index("id").loc[r.id].reset_index()
        rows.append(dict(model=name, **metrics(b)))
        paired[name] = dict(delta_subject_ba_95ci=bootstrap(r, b),
                            delta_accuracy=float((r.decision == r.y).mean() - (b.decision == b.y).mean()),
                            fixed_by_icbm=int(((r.decision == r.y) & (b.decision != b.y)).sum()),
                            broken_by_icbm=int(((r.decision != r.y) & (b.decision == b.y)).sum()))
    summary["paired_vs_baselines"] = paired
    errs = r[r.decision != r.y]
    summary["errors_by_activity"] = errs.groupby(["y", "activity"]).size().rename("n").reset_index().to_dict("records")
    pd.DataFrame(rows).to_csv(out / "comparison.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    e = summary["ensemble"]
    print(f"ensemble: acc {e['accuracy']:.4f} P {e['precision']:.4f} R {e['recall']:.4f} Sp {e['specificity']:.4f} "
          f"F1 {e['f1']:.4f} BA {e['subject_ba']:.4f} AUC {e['auc']:.4f} FN {e['fn']} FP {e['fp']} FA/h {e['fa_per_hour']:.2f}")
    print(pd.DataFrame(rows)[["model", "accuracy", "precision", "recall", "specificity", "f1", "subject_ba", "auc", "fn", "fp"]].to_string())
    print(json.dumps(paired, indent=1))


if __name__ == "__main__":
    main()
