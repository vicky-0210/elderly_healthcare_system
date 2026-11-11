"""Figures for report v4 (latest models only): VC-BiMamba (LVH), IC-BiMamba (fall), PhysioMamba (stress).

    .venv/Scripts/python -W ignore research/thesis/make_v4_figures.py [lvh] [fall] [stress] [summary]
Writes research/thesis/figures/v4/*.png and research/thesis/figures/v4/numbers.json (numbers quoted in the report).
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, precision_recall_curve, roc_auc_score,
                             roc_curve)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "research/thesis/figures/v4"
OUT.mkdir(parents=True, exist_ok=True)
NUM = OUT / "numbers.json"
numbers = json.loads(NUM.read_text()) if NUM.is_file() else {}

OURS, PAPER, ABL, INK, MUTED, GRID = "#2a6fdb", "#c9a227", "#9aa6b2", "#1f2328", "#59636e", "#d8dee4"
ABL_SHADES = ["#2a6fdb", "#7f8c99", "#a9b4bf", "#c9d1d9", "#5b6b7a"]
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": GRID, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10.5, "axes.titlecolor": INK,
                     "savefig.dpi": 200, "savefig.bbox": "tight"})


def save(fig, name):
    fig.savefig(OUT / name); plt.close(fig); print("saved", name)


def ygrid(ax): ax.yaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


def xgrid(ax): ax.xaxis.grid(True, color=GRID, lw=.6); ax.set_axisbelow(True)


def binary(y, score, decision):
    tn, fp, fn, tp = confusion_matrix(y, decision, labels=[0, 1]).ravel()
    p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); s = tn / max(tn + fp, 1)
    return dict(accuracy=(tp + tn) / len(y), precision=p, recall=r, specificity=s, f1=2 * p * r / max(p + r, 1e-12),
                balanced_accuracy=(r + s) / 2, auroc=roc_auc_score(y, score), auprc=average_precision_score(y, score),
                mcc=float((tp * tn - fp * fn) / np.sqrt(max((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn), 1))),
                tn=int(tn), fp=int(fp), fn=int(fn), tp=int(tp))


def metric_bars(ax, m, keys, color=OURS, ci=None, lo=.6):
    xs = np.arange(len(keys)); v = [m[k] for k, _ in keys]
    err = None
    if ci:
        err = [[max(m[k] - ci[k][0], 0) if k in ci else 0 for k, _ in keys],
               [max(ci[k][1] - m[k], 0) if k in ci else 0 for k, _ in keys]]
    ax.bar(xs, v, .62, color=color, yerr=err, capsize=3, error_kw=dict(lw=.8, ecolor=INK))
    tops = [val + (err[1][i] if err else 0) for i, val in enumerate(v)]
    for x, val, top in zip(xs, v, tops):
        ax.text(x, min(top, 1.01) + .012, f"{val:.4f}" if lo > .9 else f"{val:.3f}", ha="center", fontsize=8.3, color=INK)
    ax.set_xticks(xs, [n for _, n in keys], fontsize=8.8); ax.set_ylim(lo, 1.05); ygrid(ax)


def cm_plot(ax, tn, fp, fn, tp, labels, title):
    cm = np.array([[tn, fp], [fn, tp]]); rn = cm / cm.sum(1, keepdims=True)
    ax.imshow(rn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}\n({rn[i, j] * 100:.1f}%)", ha="center", va="center", fontsize=11,
                    color="white" if rn[i, j] > .55 else INK)
    ax.set_xticks([0, 1], [f"Pred. {labels[0]}", f"Pred. {labels[1]}"]); ax.set_yticks([0, 1], labels)
    ax.set_title(title, fontsize=10)
    for s in ax.spines.values(): s.set_visible(False)


def roc_pr(axr, axp, y, score, label, color=OURS, ls="-", digits=3):
    f, t, _ = roc_curve(y, score); axr.plot(f, t, color=color, ls=ls, lw=2, label=f"{label} (AUROC {roc_auc_score(y, score):.{digits}f})")
    pr, rc, _ = precision_recall_curve(y, score)
    axp.plot(rc, pr, color=color, ls=ls, lw=2, label=f"{label} (AP {average_precision_score(y, score):.{digits}f})")


def finish_roc_pr(axr, axp, prevalence=None):
    axr.plot([0, 1], [0, 1], color="#aaa", lw=.8); axr.set_xlabel("False positive rate"); axr.set_ylabel("Sensitivity")
    axp.set_xlabel("Recall"); axp.set_ylabel("Precision")
    if prevalence is not None: axp.axhline(prevalence, color="#aaa", lw=.8, ls="--")
    for a in (axr, axp): a.legend(frameon=False, fontsize=8, loc="lower right" if a is axr else "lower left"); ygrid(a)


def reliability(ax, y, p, label, color=OURS, bins=10):
    q = np.quantile(p, np.linspace(0, 1, bins + 1)); idx = np.clip(np.searchsorted(q, p, side="right") - 1, 0, bins - 1)
    mp = [p[idx == b].mean() for b in range(bins) if (idx == b).any()]
    fy = [y[idx == b].mean() for b in range(bins) if (idx == b).any()]
    ax.plot([0, 1], [0, 1], color="#aaa", lw=.8); ax.plot(mp, fy, "o-", color=color, lw=1.8, ms=5, label=label)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency"); ygrid(ax)


# =============================== LVH ======================================
def lvh():
    sys.path.insert(0, str(ROOT / "modules/ecg/experiments/lvh_novel"))
    import run as base
    main = ROOT / "artifacts/ecg/runs/20261110_lvh_vcbimamba"
    abl = ROOT / "artifacts/ecg/runs/20261111_lvh_vcbimamba_ablations"
    ens = {(r["task"], r["architecture"]): r for r in json.loads((main / "ensemble_metrics.json").read_text())}
    tasks = {"lvh_vs_normal": "LVH vs normal (915 ECGs)", "lvh_vs_rest": "LVH vs all other ECGs (2,198 ECGs)"}
    res, dfs = {}, {}
    for t in tasks:
        df, thr = base.ensemble(main / t, "vc_bimamba"); dfs[t] = df
        m = binary(df.label.to_numpy(), df.probability.to_numpy(), (df.probability >= thr).astype(int).to_numpy())
        e = ens[(t, "vc_bimamba")]
        ci = {k: (e["ci"][k2]["low"], e["ci"][k2]["high"]) for k, k2 in
              [("auroc", "auroc"), ("auprc", "auprc"), ("recall", "sensitivity"), ("specificity", "specificity"),
               ("precision", "ppv"), ("f1", "f1")]}
        res[t] = dict(m, threshold=float(thr), ci=ci, brier=e["brier"], ece=e["ece_10bins"], npv=e["npv"])
    seeds = pd.read_csv(main / "comparison_per_seed.csv") if (main / "comparison_per_seed.csv").is_file() else None
    numbers["lvh"] = res
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Sensitivity"), ("specificity", "Specificity"),
            ("f1", "F1"), ("auroc", "AUROC"), ("auprc", "AUPRC")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, t in zip(axs, tasks):
        metric_bars(ax, res[t], keys, ci=res[t]["ci"], lo=.3); ax.set_title(tasks[t])
    fig.suptitle("VC-BiMamba, 5-seed ensemble on PTB-XL test fold 10 (error bars: 95% patient bootstrap CI)", color=MUTED, fontsize=10)
    fig.tight_layout(); save(fig, "lvh_metrics.png")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, t in zip(axs, tasks):
        r = res[t]; cm_plot(ax, r["tn"], r["fp"], r["fn"], r["tp"], ["Non-LVH", "LVH"] if t == "lvh_vs_rest" else ["Normal", "LVH"], tasks[t])
    fig.tight_layout(); save(fig, "lvh_confusion.png")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.7))
    for i, t in enumerate(tasks):
        df = dfs[t]; roc_pr(axs[2 * i], axs[2 * i + 1], df.label, df.probability, "VC-BiMamba")
        finish_roc_pr(axs[2 * i], axs[2 * i + 1], df.label.mean())
        axs[2 * i].set_title(f"ROC, {tasks[t].split(' (')[0]}"); axs[2 * i + 1].set_title(f"Precision–recall, {tasks[t].split(' (')[0]}")
    fig.tight_layout(); save(fig, "lvh_roc_pr.png")
    fig, axs = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for t, c in zip(tasks, (OURS, "#7a5aa6")):
        reliability(axs[0], dfs[t].label.to_numpy(), dfs[t].probability.to_numpy(), tasks[t].split(" (")[0], c)
    axs[0].legend(frameon=False, fontsize=8); axs[0].set_title("Reliability (10 quantile bins)")
    if seeds is not None:
        s = seeds[seeds.architecture == "vc_bimamba"]
        for i, t in enumerate(tasks):
            v = s[s.task == t].auroc.to_numpy()
            axs[1].scatter(np.full(len(v), i) + np.linspace(-.12, .12, len(v)), v, color=OURS, s=30, zorder=3)
            axs[1].hlines(res[t]["auroc"], i - .25, i + .25, color=INK, lw=2)
            axs[1].text(i + .28, res[t]["auroc"], f"ensemble {res[t]['auroc']:.3f}", va="center", fontsize=8)
        axs[1].set_xticks([0, 1], ["LVH vs normal", "LVH vs rest"]); axs[1].set_xlim(-.5, 1.9); axs[1].set_ylabel("Test AUROC")
        axs[1].set_title("Single seeds (dots) and 5-seed ensemble (bar)"); ygrid(axs[1])
        numbers["lvh_seed_auroc"] = {t: [float(x) for x in s[s.task == t].auroc] for t in tasks}
    fig.tight_layout(); save(fig, "lvh_calibration_seeds.png")
    # ablations
    tab = pd.read_csv(ROOT / "research/results/vcbimamba_ablations/ablation_table.csv")
    paired = json.loads((ROOT / "research/results/vcbimamba_ablations/ablation_paired.json").read_text())
    names = {"vc_bimamba": "Full VC-BiMamba", "vc_bimamba_nocond": "− voltage conditioning",
             "vc_bimamba_unidir": "− backward scan", "vc_bimamba_noconv": "− CNN stem"}
    numbers["lvh_ablation"] = dict(table=tab.to_dict("records"), paired=paired)
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, t in zip(axs, tasks):
        d = tab[tab.task == t].set_index("variant").loc[list(names)]
        mk = ["auroc", "auprc", "f1", "accuracy"]; xs = np.arange(len(mk)); w = .2
        for k, (v, lab) in enumerate(names.items()):
            vals = d.loc[v, mk].to_numpy(float)
            ax.bar(xs + (k - 1.5) * w, vals, w - .02, color=ABL_SHADES[k], label=lab)
        ax.set_xticks(xs, ["AUROC", "AUPRC", "F1", "Accuracy"]); ax.set_ylim(.3 if t == "lvh_vs_rest" else .6, 1); ygrid(ax)
        ax.set_title(tasks[t].split(" (")[0])
    axs[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    fig.suptitle("VC-BiMamba ablations, each variant retrained (5 seeds) on identical folds", color=MUTED, fontsize=10)
    fig.tight_layout(); save(fig, "lvh_ablation.png")
    # literature
    lit = [("Angelaki et al. 2026, single lead + clinical, echo", .82), ("Yang et al. 2026, 12-lead, echo (external)", .82),
           ("Cai et al. 2024, 12-lead 2D-CNN, echo", .916), ("Yu et al. 2026, EFFNet 12-lead, CMR", .933),
           ("Strodthoff et al. 2021, 12-lead, PTB-XL fold 10 (LVH vs all)", .953),
           ("VC-BiMamba (ours), Lead II, LVH vs all other ECGs", res["lvh_vs_rest"]["auroc"]),
           ("VC-BiMamba (ours), Lead II, LVH vs normal", res["lvh_vs_normal"]["auroc"])]
    with plt.rc_context({"font.size": 11.5}):
        fig, ax = plt.subplots(figsize=(10, 4.2)); y = np.arange(len(lit))[::-1]
        for yi, (lab, v) in zip(y, lit):
            ax.barh(yi, v, .6, color=OURS if "ours" in lab else PAPER); ax.text(v + .003, yi, f"{v:.3f}", va="center", fontsize=10.5)
        ax.set_yticks(y, [l for l, _ in lit]); ax.set_xlim(.7, 1.0); ax.set_xlabel("AUROC"); xgrid(ax)
        fig.tight_layout(); save(fig, "lvh_literature.png")


# =============================== FALL =====================================
FALL = ROOT / "research/results/icbimamba_v1"


def fall():
    s = json.loads((FALL / "main/summary.json").read_text()); e = s["ensemble"]
    oof = pd.read_csv(FALL / "main/oof_trials.csv"); cmp_ = pd.read_csv(FALL / "main/comparison.csv")
    y, z, dec = oof.y.to_numpy(), oof.logit.to_numpy(), oof.decision.to_numpy()
    m = binary(y, z, dec); m["subject_ba"] = e["subject_ba"]; m["fa_per_hour"] = e["fa_per_hour"]
    m["subject_ba_ci"] = s["subject_ba_95ci"]
    numbers["fall"] = dict(m, folds=s["folds"], parameters=s["parameters_per_member"],
                           seeds={r.model: dict(accuracy=r.accuracy, precision=r.precision, recall=r.recall,
                                                specificity=r.specificity, f1=r.f1, auc=r.auc)
                                  for r in cmp_.itertuples() if r.model.startswith("IC-BiMamba seed")},
                           errors=s["errors_by_activity"])
    numbers["fall"]["per_fold"] = {int(k): binary(g.y.to_numpy(), g.logit.to_numpy(), g.decision.to_numpy())
                                   for k, g in oof.groupby("fold")}
    numbers["fall"]["per_fold_n"] = {int(k): [int(g.subject.nunique()), int(len(g))] for k, g in oof.groupby("fold")}
    older = oof[oof.older == 1]
    numbers["fall"]["older"] = binary(older.y.to_numpy(), older.logit.to_numpy(), older.decision.to_numpy()) if older.y.nunique() == 2 else None
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("specificity", "Specificity"),
            ("f1", "F1"), ("balanced_accuracy", "Bal. acc."), ("auroc", "AUROC"), ("auprc", "AUPRC")]
    fig, ax = plt.subplots(figsize=(9, 3.6)); metric_bars(ax, m, keys, lo=.95)
    ax.set_title("IC-BiMamba 5-seed ensemble, all 4,500 SisFall trials from held-out participants")
    fig.tight_layout(); save(fig, "fall_metrics.png")
    fig, ax = plt.subplots(figsize=(4.6, 4)); cm_plot(ax, m["tn"], m["fp"], m["fn"], m["tp"], ["ADL", "Fall"], "IC-BiMamba ensemble, 4,500 trials")
    fig.tight_layout(); save(fig, "fall_confusion.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    roc_pr(axs[0], axs[1], y, z, "IC-BiMamba", digits=4)
    finish_roc_pr(axs[0], axs[1], y.mean())
    for a, lim in ((axs[0], (0, .2, .8, 1.0)), (axs[1], (.8, 1.0, .8, 1.01))): a.set_xlim(*lim[:2]); a.set_ylim(*lim[2:])
    axs[0].set_title("ROC (zoomed)"); axs[1].set_title("Precision–recall (zoomed)")
    fig.tight_layout(); save(fig, "fall_roc_pr.png")
    papers = [("Mohammad et al. 2023, Sensors†", 98.0, None, None, None), ("Jishnuraj et al. 2026, Front. AI†", 98.0, None, None, None),
              ("Villa & Casilari 2026, Sensors", 98.9, None, 96.7, None), ("Zhang et al. 2024, JMIR", 99.32, 98.58, 99.15, 98.86),
              ("Ahirwar & Soni 2025, Sci. Rep.", 99.50, 99.44, 99.58, 99.51)]
    rows = papers + [("IC-BiMamba (ours)", 100 * m["accuracy"], 100 * m["precision"], 100 * m["recall"], 100 * m["f1"])]
    with plt.rc_context({"font.size": 12, "axes.titlesize": 12.5}):
        fig, axs = plt.subplots(2, 2, figsize=(11, 5.8), sharey=True)
        for k, (ax, nm) in enumerate(zip(axs.flat, ["Accuracy", "Precision", "Recall", "F1"])):
            for i, r in enumerate(rows):
                v = r[k + 1]
                if v is None: ax.text(85.7, i, "not reported", va="center", fontsize=10.5, color=MUTED); continue
                ax.barh(i, v, .6, color=OURS if "ours" in r[0] else PAPER); ax.text(v + .05, i, f"{v:.2f}", va="center", fontsize=10.5)
            ax.set_xlim(85.5, 100.9); ax.set_title(nm + " (%)"); xgrid(ax)
        axs[0, 0].set_yticks(range(len(rows)), [r[0] for r in rows]); axs[0, 0].set_ylim(len(rows) - .5, -.5)
        fig.tight_layout(); save(fig, "fall_literature.png")
    err = pd.DataFrame(s["errors_by_activity"]).sort_values("n")
    err["label"] = err.activity + err.y.map({1: " (missed fall)", 0: " (false alarm)"})
    fig, ax = plt.subplots(figsize=(6.4, .3 * len(err) + 1))
    ax.barh(err.label, err.n, color=[OURS if v else "#d1495b" for v in err.y], height=.6)
    for i, v in enumerate(err.n): ax.text(v + .1, i, str(v), va="center", fontsize=8)
    ax.set_title(f"Errors by activity ({m['fn']} missed falls, {m['fp']} false alarms)"); xgrid(ax)
    fig.tight_layout(); save(fig, "fall_errors.png")
    # seeds vs ensemble
    sd = cmp_[cmp_.model.str.startswith("IC-BiMamba")].copy()
    sd["label"] = sd.model.str.replace("IC-BiMamba ", "").str.replace("(5-seed ensemble)", "Ensemble")
    fig, ax = plt.subplots(figsize=(9, 3.314)); mk = ["accuracy", "precision", "recall", "f1"]; xs = np.arange(len(sd)); w = .2
    for k, col in enumerate(mk):
        ax.bar(xs + (k - 1.5) * w, sd[col] * 100, w - .02, color=["#2a6fdb", "#7a5aa6", "#3b8a3b", "#c8641e"][k], label=col.title() if col != "f1" else "F1")
    ax.set_xticks(xs, sd.label); ax.set_ylim(93, 100.5); ax.set_ylabel("%"); ygrid(ax)
    ax.set_title("Single seeds versus the logit-averaged ensemble")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=4, fontsize=9, loc="lower center", bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "fall_seeds.png")
    # knock-outs
    ko = {"full": "Full model", "no_raw": "− raw 200 Hz patch", "no_cond": "− impact conditioning",
          "unidir": "− backward scan", "no_local": "− multiscale local conv."}
    have = [k for k in ko if (FALL / f"knockout_{k}/summary.json").is_file()]
    if len(have) == len(ko):
        rows = {}
        for k in ko:
            ss = json.loads((FALL / f"knockout_{k}/summary.json").read_text())
            rows[k] = dict(ss["ensemble"], ci=ss["subject_ba_95ci"])
        numbers["fall_knockout"] = rows
        fig, ax = plt.subplots(figsize=(10, 3.7)); mk = [("accuracy", "Accuracy"), ("precision", "Precision"),
                                                         ("recall", "Recall"), ("f1", "F1"), ("subject_ba", "Subject BA")]
        xs = np.arange(len(mk)); w = .16
        for i, k in enumerate(ko):
            v = [rows[k][c] * 100 for c, _ in mk]
            ax.bar(xs + (i - 2) * w, v, w - .02, color=ABL_SHADES[i], label=ko[k])
        ax.set_xticks(xs, [n for _, n in mk]); lo = min(rows[k][c] for k in ko for c, _ in mk) * 100
        ax.set_ylim(max(lo - 3, 0), 100.5); ax.set_ylabel("%"); ygrid(ax); ax.legend(frameon=False, ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(.5, -.1))
        ax.set_title("IC-BiMamba component knock-outs at inference (25 trained members, no retraining)")
        fig.tight_layout(); save(fig, "fall_knockout.png")


# =============================== STRESS ===================================
def stress():
    sys.path.insert(0, str(ROOT / "modules/eda/src"))
    from wearable_eda.evaluation.loso import load_runs, summarize
    runs = load_runs(ROOT / "artifacts/eda/runs/wesad_physiomamba_v1")
    res, sub = {}, {}
    for n, r in runs.items():
        if "physiomamba" not in n: continue
        row, per = summarize(r); res[n] = row; sub[n] = pd.DataFrame(per).set_index("subject")
    clean = lambda d: {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in d.items()
                       if isinstance(v, (int, float, str, np.floating, np.integer))}
    numbers["stress"] = {n: clean(v) for n, v in res.items()}
    for cal in ("zero", "rest"):
        n = f"neural/physiomamba_{cal}"
        if n not in runs: continue
        d = runs[n].pred; dec = d[["p0", "p1"]].to_numpy().argmax(1)
        numbers["stress"][n]["pooled"] = binary(d.y_true.to_numpy(), d.p1.to_numpy(), dec)
        numbers["stress"][n]["per_subject_f1"] = sub[n].macro_f1.round(4).to_dict()
    z = res["neural/physiomamba_zero"]
    keys = [("accuracy", "Accuracy"), ("precision", "Precision"), ("sensitivity", "Recall"), ("specificity", "Specificity"),
            ("macro_f1", "Macro F1"), ("auroc", "AUROC"), ("auprc", "AUPRC")]
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), sharey=True)
    for ax, cal in zip(axs, ("zero", "rest")):
        r = res.get(f"neural/physiomamba_{cal}")
        if r is None: continue
        ci = {k: (r[k + "_lo"], r[k + "_hi"]) for k, _ in keys if k + "_lo" in r}
        metric_bars(ax, r, keys, ci=ci, lo=.6)
        ax.set_title("No per-person calibration" if cal == "zero" else "2-minute resting calibration")
    fig.suptitle("PhysioMamba, WESAD leave-one-subject-out (participant means, 95% bootstrap CI)", color=MUTED, fontsize=10)
    fig.tight_layout(); save(fig, "stress_metrics.png")
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.9))
    for ax, cal in zip(axs, ("zero", "rest")):
        p = numbers["stress"][f"neural/physiomamba_{cal}"]["pooled"]
        cm_plot(ax, p["tn"], p["fp"], p["fn"], p["tp"], ["Baseline", "Stress"], "No calibration" if cal == "zero" else "Rest calibration")
    fig.suptitle("All 1,636 held-out windows pooled over 15 participants", color=MUTED, fontsize=10)
    fig.tight_layout(); save(fig, "stress_confusion.png")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.9))
    for cal, ls in (("zero", "-"), ("rest", "--")):
        d = runs[f"neural/physiomamba_{cal}"].pred
        roc_pr(axs[0], axs[1], d.y_true, d.p1, "no calibration" if cal == "zero" else "rest calibration", OURS if cal == "zero" else "#7a5aa6", ls)
    finish_roc_pr(axs[0], axs[1], runs["neural/physiomamba_zero"].pred.y_true.mean())
    axs[0].set_title("ROC, pooled held-out windows"); axs[1].set_title("Precision–recall, pooled held-out windows")
    fig.tight_layout(); save(fig, "stress_roc_pr.png")
    fig, ax = plt.subplots(figsize=(10, 3.319))
    pz = sub["neural/physiomamba_zero"].macro_f1; pr_ = sub.get("neural/physiomamba_rest")
    x = np.arange(len(pz)); ax.bar(x - .2, pz, .38, color=OURS, label=f"No calibration (mean {pz.mean():.3f})")
    if pr_ is not None: ax.bar(x + .2, pr_.macro_f1.reindex(pz.index), .38, color="#a9b4bf", label=f"Rest calibration (mean {pr_.macro_f1.mean():.3f})")
    ax.set_xticks(x, [f"S{s}" for s in pz.index]); ax.set_ylim(.5, 1.03); ax.set_ylabel("Macro F1"); ygrid(ax)
    ax.set_title("Per-participant macro F1 (each person unseen during training)")
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2, fontsize=9.5, loc="lower center", bbox_to_anchor=(.5, 0))
    fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "stress_subjects.png")
    pap = [("Prajwal et al. 2026, Discov. Appl. Sci.", 83.0, None), ("Vos et al. 2023, J. Biomed. Inform.", 85.0, 78.0),
           ("Kumar et al. 2021, Pattern Recognit. Lett.", 87.7, None), ("Rashid et al. 2021, IEEE EMBC", 88.56, 86.18),
           ("Kumar et al. 2024, IEEE Access (Resp-BoostNet)", 94.0, None),
           ("PhysioMamba (ours)", z["accuracy"] * 100, z["macro_f1"] * 100)]
    with plt.rc_context({"font.size": 11.5}):
        fig, ax = plt.subplots(figsize=(10, 4.4)); yy = np.arange(len(pap))[::-1]
        for yi, (lab, a, f) in zip(yy, pap):
            c = OURS if "ours" in lab else PAPER
            ax.barh(yi + .18, a, .34, color=c); ax.text(a + .3, yi + .18, f"{a:.1f}% acc.", va="center", fontsize=10)
            if f: ax.barh(yi - .18, f, .34, color=c, alpha=.5); ax.text(f + .3, yi - .18, f"{f:.1f} F1", va="center", fontsize=10)
        ax.set_yticks(yy, [p[0] for p in pap]); ax.set_xlim(70, 102); ax.set_xlabel("Percent (solid = accuracy, light = F1)"); xgrid(ax)
        fig.tight_layout(); save(fig, "stress_literature.png")
    abl = {"physiomamba": "Full PhysioMamba", "physiomamba_nocond": "− context conditioning", "physiomamba_nogate": "− reliability gate (mean fusion)"}
    if all(f"neural/{a}_{c}" in res for a in abl for c in ("zero", "rest")):
        fig, axs = plt.subplots(1, 2, figsize=(12, 3.7), sharey=True)
        mk = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity"), ("auroc", "AUROC")]
        for ax, cal in zip(axs, ("zero", "rest")):
            xs = np.arange(len(mk)); w = .26
            for i, a in enumerate(abl):
                r = res[f"neural/{a}_{cal}"]; ax.bar(xs + (i - 1) * w, [r[k] for k, _ in mk], w - .02, color=ABL_SHADES[i], label=abl[a])
            ax.set_xticks(xs, [n for _, n in mk]); ax.set_ylim(.85, 1.005); ygrid(ax)
            ax.set_title("No calibration" if cal == "zero" else "Rest calibration")
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, fontsize=8.5, ncol=3, loc="lower center", bbox_to_anchor=(.5, 0))
        fig.suptitle("PhysioMamba ablations, each variant retrained with the same LOSO protocol", color=MUTED, fontsize=10)
        fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "stress_ablation.png")
        from scipy.stats import wilcoxon
        pairs = {}
        for cal in ("zero", "rest"):
            ref = sub[f"neural/physiomamba_{cal}"].macro_f1
            for a in list(abl)[1:]:
                d = (ref - sub[f"neural/{a}_{cal}"].macro_f1.reindex(ref.index)).to_numpy()
                rng = np.random.default_rng(0); bs = rng.integers(0, len(d), (2000, len(d)))
                lo, hi = np.quantile(d[bs].mean(1), [.025, .975])
                pairs[f"{a}_{cal}"] = dict(delta=float(d.mean()), lo=float(lo), hi=float(hi), wins=int((d > 0).sum()),
                                           losses=int((d < 0).sum()), p=float(wilcoxon(d).pvalue) if np.any(d != 0) else 1.0)
        numbers["stress_ablation_paired"] = pairs


def stress_ko():
    ko = json.loads((ROOT / "research/results/physiomamba_knockout/zero.json").read_text())
    numbers["stress_knockout"] = {k: v["mean"] for k, v in ko.items()}
    names = {"full": "Full model", "no_cardiac": "− cardiac (BVP)", "no_eda": "− EDA", "no_motion": "− motion (ACC)"}
    mk = [("accuracy", "Accuracy"), ("macro_f1", "Macro F1"), ("sensitivity", "Sensitivity"), ("specificity", "Specificity")]
    fig, ax = plt.subplots(figsize=(9, 3.4)); xs = np.arange(len(mk)); w = .2
    for i, k in enumerate(names):
        v = [ko[k]["mean"][c] for c, _ in mk]
        ax.bar(xs + (i - 1.5) * w, v, w - .02, color=ABL_SHADES[i], label=names[k])
        for x, val in zip(xs + (i - 1.5) * w, v): ax.text(x, val + .004, f"{val:.3f}", ha="center", fontsize=6.8, rotation=90)
    ax.set_xticks(xs, [n for _, n in mk]); ax.set_ylim(.85, 1.02); ygrid(ax); ax.legend(frameon=False, ncol=4, fontsize=8, loc="lower center")
    ax.set_title("PhysioMamba modality knock-outs at inference (zero calibration, 15 held-out participants)")
    fig.tight_layout(); save(fig, "stress_knockout.png")


# =============================== SUMMARY ==================================
def summary():
    lv = numbers["lvh"]["lvh_vs_normal"]; fa = numbers["fall"]; st = numbers["stress"]["neural/physiomamba_zero"]
    rows = [("LVH (VC-BiMamba)", [lv["accuracy"], lv["precision"], lv["recall"], lv["specificity"], lv["f1"], lv["auroc"]]),
            ("Fall (IC-BiMamba)", [fa["accuracy"], fa["precision"], fa["recall"], fa["specificity"], fa["f1"], fa["auroc"]]),
            ("Stress (PhysioMamba)", [st["accuracy"], st["precision"], st["sensitivity"], st["specificity"], st["macro_f1"], st["auroc"]])]
    names = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "AUROC"]
    fig, ax = plt.subplots(figsize=(11, 3.8)); xs = np.arange(len(names)); w = .26
    for i, (lab, v) in enumerate(rows):
        ax.bar(xs + (i - 1) * w, v, w - .02, color=["#2a6fdb", "#3b8a3b", "#c8641e"][i], label=lab)
        for x, val in zip(xs, v): ax.text(x + (i - 1) * w, val + .008, f"{val:.4f}" if val > .995 else f"{val:.3f}", ha="center", fontsize=7, rotation=90)
    ax.set_xticks(xs, names); ax.set_ylim(.6, 1.08); ygrid(ax); ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="upper left")
    ax.set_title("Headline held-out results of the three proposed models (LVH: vs normal; stress: zero calibration, macro F1)")
    fig.tight_layout(); save(fig, "summary_three_models.png")


def apnea_lit():
    rows = [("Lazazzera et al. 2021, IEEE TBME (PPG + SpO2, 1-min)", 75.1, None, None, None),
            ("Choksatchawathi et al. 2024, IEEE IoT J. (PPG, MESA)", 73.93, 77.17, 73.26, 82.45),
            ("Thuptimdang et al. 2026, Sleep Breath. (PPG + SpO2)", None, 77.1, None, None),
            ("Yeo et al. 2026, JMIR meta-analysis (PPG AI, pooled)", None, 79.6, 76.5, None),
            ("Katsaridis et al. 2026, Physiol. Meas. (PPG, per-time)", None, None, None, 92.0),
            ("Ours: dual-rate TCN + GRU (PPG, per-second)", 62.9, 52.1, 62.9, 59.4),
            ("Ours: SpO2 rank ensemble (60 s windows)", 73.9, 73.3, 76.6, 82.5)]
    with plt.rc_context({"font.size": 11.5, "axes.titlesize": 12.5}):
        fig, axs = plt.subplots(2, 2, figsize=(11.5, 6.4), sharey=True)
        for k, (ax, nm) in enumerate(zip(axs.flat, ["Accuracy (%)", "Sensitivity (%)", "Specificity (%)", "AUROC (x100)"])):
            for i, r in enumerate(rows):
                v = r[k + 1]
                if v is None: ax.text(41, i, "not reported", va="center", fontsize=10, color=MUTED); continue
                ax.barh(i, v, .6, color=OURS if r[0].startswith("Ours") else PAPER); ax.text(v + .6, i, f"{v:.1f}", va="center", fontsize=10)
            ax.set_xlim(40, 101); ax.set_title(nm); xgrid(ax)
        axs[0, 0].set_yticks(range(len(rows)), [r[0] for r in rows]); axs[0, 0].set_ylim(len(rows) - .5, -.5)
        fig.tight_layout(); save(fig, "apnea_literature.png")


def ecg_prep():
    """Redraw the Lead-II preprocessing example (same record and noise as the original development figure)."""
    sys.path.insert(0, str(ROOT / "modules/ecg/src"))
    import pandas as pd
    from wearable_ecg.labels.ptbxl import cohort
    from wearable_ecg.preprocessing.pipeline import process, corrupt, landmarks
    cache = ROOT / "data/ecg/processed/ptbxl_360hz_v1"
    frame = pd.read_csv(cache / "manifest.csv")
    eligible = cohort(frame, "lvh_vs_normal").query("strat_fold in [7,8]")
    waves = np.load(cache / "minimal_waves.npy", mmap_mode="r")
    rng = np.random.default_rng(906)
    selected = pd.concat([g.sample(min(64, len(g)), random_state=906) for _, g in eligible.groupby("label")])
    ex = None
    for r in selected.itertuples():
        clean = np.array(waves[r.row_index])
        if len(landmarks(clean)) < 3: continue
        for s in (0., 1., 3., 6.):
            noisy = corrupt(clean, rng, s)
            if ex is None and r.label == 1 and s == 3: ex = (r.ecg_id, clean, noisy)
        if ex: break
    eid, clean, noisy = ex
    assert int(eid) == 14648, eid
    swt = process(noisy, None, "swt")[0]
    t = np.arange(len(clean)) / 360; keep = (t >= 2) & (t <= 5)
    with plt.rc_context({"font.size": 12, "legend.fontsize": 11, "axes.titlesize": 12.5}):
        fig, axs = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
        for ax, (name, tr) in zip(axs, (("Corrupted input (baseline wander, 50 Hz, white noise)", noisy),
                                        ("After robust baseline + SWT denoising", swt))):
            ax.plot(t[keep], clean[keep], color=INK, lw=1.1, label="PTB-XL reference")
            ax.plot(t[keep], tr[keep], color="#d1603d", lw=1.0, alpha=.9, label=name.split(" (")[0])
            ax.set_title(name, loc="left"); ax.set_ylabel("Amplitude (mV)"); ygrid(ax)
        axs[1].set_xlabel("Time (s)")
        h, _ = axs[0].get_legend_handles_labels()
        fig.legend(h, ["PTB-XL reference", "Signal shown in the panel title"], frameon=False, ncol=2,
                   loc="lower center", bbox_to_anchor=(.5, 0))
        fig.tight_layout(rect=(0, .07, 1, 1)); save(fig, "ecg_prep_example.png")


def imu_signals():
    """Two full-width IMU illustrations (replace the old four-panel images in Section 3.6)."""
    sys.path.insert(0, str(ROOT / "modules/imu/src"))
    from wearable_imu.datasets.sisfall import COUNTS_TO_PHYSICAL, load_cache
    from wearable_imu.preprocessing.ipmv import LowPass
    manifest, raw = load_cache()
    trial = lambda act: manifest[(manifest.subject == "SA01") & (manifest.activity == act)].iloc[0]
    AX = {"X": "#2a6fdb", "Y": "#e8833a", "Z": "#2a9d8f"}
    big = {"font.size": 12, "axes.titlesize": 12.5, "legend.fontsize": 11}
    with plt.rc_context(big):
        # Figure A: raw vs dynamic (gravity-removed) acceleration, ADL vs fall
        fig, axs = plt.subplots(2, 2, figsize=(11, 6.2), sharex="col")
        for col, (act, name) in enumerate((("D14", "Daily activity (ADL D14)"), ("F10", "Simulated fall (F10)"))):
            r = trial(act); a = raw[r.offset:r.offset + r.n, :3] * COUNTS_TO_PHYSICAL[:3]
            low = LowPass(200, 30, 2).apply(a); inn = a - low; sz = np.linalg.norm(inn, axis=1, keepdims=True)
            sm = low + inn * sz / (sz + .35); grav = LowPass(200, .75, 2).apply(sm); dyn = sm - grav
            t = np.arange(len(a)) / 200
            for row, (sig, lab) in enumerate(((a, "Acceleration (g)"), (dyn, "Dynamic acceleration (g)"))):
                ax = axs[row, col]
                for i, (k, c) in enumerate(AX.items()): ax.plot(t, sig[:, i], lw=1.1, color=c, label=k)
                ygrid(ax); ax.set_ylabel(lab if col == 0 else "")
            axs[0, col].set_title(name); axs[1, col].set_xlabel("Elapsed time (s)")
        h, l = axs[0, 0].get_legend_handles_labels()
        fig.legend(h, [f"{k} axis" for k in l], frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(.5, 0))
        fig.tight_layout(rect=(0, .06, 1, 1)); save(fig, "imu_gravity_dynamic.png")
        # Figure B: native 200 Hz magnitudes, ADL vs fall aligned at the impact peak
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.0))
        for act, name, c in (("D11", "Daily activity (ADL D11)", "#7f8c99"), ("F01", "Simulated fall (F01)", "#2a6fdb")):
            r = trial(act); x = raw[r.offset:r.offset + r.n] * COUNTS_TO_PHYSICAL
            acc = np.linalg.norm(x[:, :3], axis=1); gyr = np.linalg.norm(x[:, 3:6], axis=1)
            pk = int(np.argmax(acc)); t = (np.arange(len(acc)) - pk) / 200; keep = np.abs(t) <= 3
            axs[0].plot(t[keep], acc[keep], lw=1.2, color=c, label=name)
            axs[1].plot(t[keep], gyr[keep], lw=1.2, color=c, label=name)
        axs[0].set_ylabel("Acceleration magnitude (g)"); axs[1].set_ylabel("Rotation magnitude (deg/s)")
        for ax in axs: ax.set_xlabel("Time relative to acceleration peak (s)"); ygrid(ax)
        h, l = axs[0].get_legend_handles_labels()
        fig.legend(h, l, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(.5, 0))
        fig.tight_layout(rect=(0, .1, 1, 1)); save(fig, "imu_native_peak.png")


if __name__ == "__main__":
    parts =sys.argv[1:] or ["lvh", "fall", "stress", "summary"]
    for p in parts: globals()[p]()
    NUM.write_text(json.dumps(numbers, indent=1, default=float))
