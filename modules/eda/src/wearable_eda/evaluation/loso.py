"""Audit held-out probabilities and summarize WESAD results by participant."""
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import metrics


SUBJECTS = tuple(i for i in range(2, 18) if i != 12)
METRICS = ("accuracy", "precision", "sensitivity", "specificity", "fpr", "f1",
           "macro_f1", "balanced_accuracy", "mcc", "auroc", "auprc", "brier")
BOOT = np.random.default_rng(2606).integers(0, len(SUBJECTS), (5000, len(SUBJECTS)))


@dataclass
class Run:
    name: str
    family: str
    meta: dict
    pred: pd.DataFrame
    folds: pd.DataFrame
    classes: list
    modalities: list

    @property
    def protocol(self):
        return self.meta["calibration"]

    @property
    def eligible(self):
        return (self.classes == [1, 2] and self.meta["role"] in ("primary", "candidate")
                and self.protocol in ("zero", "rest")
                and self.meta.get("decomposition", "mcsaed") == "mcsaed")


def load_runs(root):
    runs = {}
    for path in sorted(Path(root).glob("*/*/predictions.csv")):
        folder = path.parent
        name = folder.parent.name + "/" + folder.name
        meta = json.loads((folder / "meta.json").read_text(encoding="utf-8-sig"))
        pred = pd.read_csv(path).sort_values(["subject", "start"]).reset_index(drop=True)
        folds = pd.read_csv(folder / "folds.csv").set_index("subject").sort_index()
        classes = list(meta["classes"])
        if classes not in ([1,2], [1,2,3], [1,2,3,4]):
            raise ValueError(f"{name}: unexpected WESAD condition order")
        modalities = meta["modalities"]
        if isinstance(modalities, str):
            modalities = modalities.split("+")
        if tuple(folds.index) != SUBJECTS or tuple(pred.subject.unique()) != SUBJECTS:
            raise ValueError(f"{name}: expected 15 distinct held-out participants")
        if "inner_score" not in folds or not folds.inner_score.between(0, 1).all():
            raise ValueError(f"{name}: invalid grouped selection scores")
        for subject, fold in folds.iterrows():
            i = SUBJECTS.index(subject)
            val = {SUBJECTS[(i+j)%15] for j in (1,8)}
            refit = set(SUBJECTS)-{subject}
            train = refit-val
            columns = (("inner_subjects", "inner_validation_subjects", val),
                       ("inner_fit_subjects", "inner_train_subjects", train),
                       ("refit_subjects", "refit_subjects", refit))
            for classical, neural, expected in columns:
                observed = json.loads(fold[classical if classical in fold else neural])
                if set(observed) != expected or len(observed) != len(expected):
                    raise ValueError(f"{name}: invalid partition in S{subject} {classical}")
        prob = pred[[f"p{i}" for i in range(len(classes))]].to_numpy()
        if (not np.isfinite(prob).all() or (prob < 0).any() or (prob > 1).any()
                or not np.allclose(prob.sum(1), 1, atol=2e-5)):
            raise ValueError(f"{name}: invalid class probabilities")
        if not pred.y_true.isin(range(len(classes))).all() or pred.duplicated(["subject", "start"]).any():
            raise ValueError(f"{name}: unexpected labels or duplicate subject/window")
        if not np.isfinite(pred.start).all() or (pred.start < 0).any():
            raise ValueError(f"{name}: invalid window time")
        runs[name] = Run(name, folder.parent.name, meta, pred, folds, classes, list(modalities))
    if not runs:
        raise ValueError(f"No completed LOSO experiments under {root}")
    return runs


def verify_annotations(runs, cache):
    """Reconstruct the entire scoring roster from original-clock annotation counts."""
    specs = {}
    for run in runs.values():
        key = (tuple(run.classes),run.meta.get("seconds",60),
               run.meta.get("stride",15),run.meta.get("purity",.8))
        specs.setdefault(key,[]).append(run)
    expected = {key:[] for key in specs}
    for subject in SUBJECTS:
        with np.load(Path(cache)/f"S{subject}.npz") as data:
            votes = np.asarray(data["counts"],dtype=np.int64)
            if not np.all(votes.sum(1)==175):
                raise ValueError(f"S{subject}: annotation clock mismatch")
            calibration_end = int(data["rest"][1])
            missing = ~np.isfinite(data["eda"])
        edges = np.flatnonzero(np.diff(np.r_[False,missing,False])).reshape(-1,2)
        gaps = [(left,right) for left,right in edges if right-left>=20]
        total_votes = np.vstack((np.zeros((1,8),dtype=np.int64),votes.cumsum(axis=0)))
        missing_ticks = np.r_[0,missing.cumsum()]
        for key in specs:
            classes,seconds,stride,purity = key
            width, step = int(seconds*4), int(stride*4)
            # Use the grid relative to the recording origin, then independently
            # reject calibration overlap, long gaps and insufficient class votes.
            for left in range(0,len(votes)-width+1,step):
                right = left+width
                if left<calibration_end or missing_ticks[right]-missing_ticks[left]>.2*width:
                    continue
                if any(left<b and right>a for a,b in gaps):
                    continue
                support = (total_votes[right]-total_votes[left])[list(classes)]
                label = int(support.argmax())
                if support[label]>=purity*175*width:
                    expected[key].append((subject,left/4,right/4,label))
    for key, group in specs.items():
        records = np.asarray(expected[key])
        for run in group:
            saved = run.pred[["subject","start","end","y_true"]].to_numpy()
            if not np.array_equal(saved,records):
                raise ValueError(f"{run.name}: predictions differ from original-clock label/window roster")


def require_complete(runs):
    from wearable_eda.training.classical_run import experiments
    from wearable_eda.training.neural_experiments import catalogue
    expected = {"classical/"+s["name"] for s in experiments()} | {"neural/"+name for name in catalogue()}
    missing = sorted(expected-set(runs))
    if missing:
        raise ValueError(f"{len(missing)} scheduled experiments incomplete, starting with {missing[:5]}")


def measure(frame, k):
    truth = frame.y_true.to_numpy(dtype=int)
    prob = frame[[f"p{i}" for i in range(k)]].to_numpy()
    decision = prob.argmax(1)
    labels = np.arange(k)
    cm = metrics.confusion_matrix(truth, decision, labels=labels)
    target = [1] if k == 2 else labels
    if np.any(cm.sum(axis=1) == 0):
        raise ValueError("A participant has an absent target class; check label mapping")
    tp = cm.diagonal()
    fp, fn = cm.sum(axis=0) - tp, cm.sum(axis=1) - tp
    tn = cm.sum() - tp - fp - fn
    pr = np.divide(tp, tp+fp, out=np.zeros(k), where=tp+fp>0)
    rc = tp / (tp+fn)
    sp = tn / (tn+fp)
    f1 = 2 * tp / (2*tp + fp + fn)
    auc = np.mean([metrics.roc_auc_score(truth == i, prob[:, i]) for i in target])
    ap = np.mean([metrics.average_precision_score(truth == i, prob[:, i]) for i in target])
    err = (np.eye(k)[truth] - prob)**2
    return dict(zip(METRICS, (tp.sum()/len(truth), pr[target].mean(), rc[target].mean(),
            sp[target].mean(), 1-sp[target].mean(), f1[target].mean(), f1.mean(),
            rc.mean(), metrics.matthews_corrcoef(truth, decision), auc, ap,
            err[:, 1].mean() if k == 2 else err.sum(1).mean())))


def choose(table, lightweight=False):
    """Select exclusively on grouped inner validation scores and resource use."""
    if not lightweight:
        return table.sort_values(["score", "name"], ascending=[False, True]).iloc[0]["name"]
    if not np.isfinite(table["size"]).all():
        raise ValueError("Lightweight selection requires serialized model sizes for every candidate")
    pool = table.loc[table.score >= table.score.max() - 0.01]
    return pool.sort_values(["sensors", "size", "name"]).iloc[0]["name"]


def select(runs, protocol):
    candidates = [r for r in runs.values() if r.eligible and r.protocol == protocol]
    if not candidates:
        return [], []
    first = candidates[0].pred[["subject", "start", "y_true"]]
    for run in candidates[1:]:
        other = run.pred[["subject", "start", "y_true"]]
        if not np.array_equal(first.to_numpy(), other.to_numpy()):
            raise ValueError(f"Nonmatching primary windows: {run.name} and {candidates[0].name}")
    choices, results = [], []
    for mode in ("highest_inner", "lightweight"):
        pred = []
        for subject in (*SUBJECTS, "refit"):
            rows = []
            for run in candidates:
                scores, sizes = run.folds.inner_score, run.folds.artifact_bytes
                rows.append(dict(name=run.name, sensors=len(run.modalities),
                                 score=scores.mean() if subject == "refit" else scores.loc[subject],
                                 size=sizes.median() if subject == "refit" else sizes.loc[subject]))
            table = pd.DataFrame(rows)
            name = choose(table, mode == "lightweight")
            choices.append(dict(protocol=protocol, rule=mode, subject=subject, selected=name,
                                inner_macro_f1=float(table.loc[table.name == name,"score"].iloc[0])))
            if subject != "refit":
                one = runs[name].pred.query("subject == @subject").copy()
                pred.append(one)
        meta = dict(model=mode, role="nested", category="nested", calibration=protocol)
        results.append(Run(f"nested/{protocol}_{mode}", "nested", meta,
                           pd.concat(pred, ignore_index=True), pd.DataFrame(), [1,2], ["varies"]))
    return choices, results


def summarize(run):
    subjects = []
    for sub, frame in run.pred.groupby("subject"):
        subjects.append(dict(experiment=run.name, subject=sub, windows=len(frame),
                             **measure(frame, len(run.classes))))
    df = pd.DataFrame(subjects)
    row = dict(experiment=run.name, model=run.meta["model"], family=run.family,
               category=run.meta.get("category", run.meta.get("group", run.family)),
               role=run.meta["role"], calibration=run.protocol, classes=len(run.classes),
               modalities="+".join(run.modalities), subjects=len(df), windows=int(df.windows.sum()))
    if not run.folds.empty:
        row["inner_macro_f1"] = run.folds.inner_score.mean()
        row["filtering"] = run.meta.get("filtering","conservative")
        row["decomposition"] = run.meta.get("decomposition","mcsaed")
        row["quality"] = run.meta.get("quality","auto")
        row["seconds"] = run.meta.get("seconds",60)
        row["stride"] = run.meta.get("stride",15)
        row["purity"] = run.meta.get("purity",.8)
    for key in METRICS:
        vals = df[key].to_numpy()
        low, high = np.quantile(vals[BOOT].mean(1), [.025, .975])
        row.update({key: vals.mean(), key+"_sd": vals.std(ddof=1), key+"_lo":low, key+"_hi":high})
    for key in ("artifact_bytes", "cpu_latency_p50_ms", "cpu_latency_p95_ms", "trainable_parameters"):
        if key in run.folds:
            row[key] = pd.to_numeric(run.folds[key], errors="coerce").dropna().median()
    return row, subjects


def paired(contrasts, subjects, runs):
    wide = subjects.pivot(index="subject", columns="experiment", values="macro_f1")
    rows = []
    for label, a, b in contrasts:
        if (a not in wide) or (b not in wide):
            continue
        d = (wide[a]-wide[b]).to_numpy()
        low, high = np.quantile(d[BOOT].mean(1), [.025, .975])
        p = wilcoxon(d, zero_method="pratt", method="auto").pvalue if np.any(d) else 1.0
        sd = d.std(ddof=1)
        keys = ["subject", "start", "y_true"]
        identical = np.array_equal(runs[a].pred[keys].to_numpy(), runs[b].pred[keys].to_numpy())
        rows.append(dict(contrast=label, a=a, b=b, subjects=len(d), same_windows=identical,
                         delta_macro_f1=d.mean(), low=low, high=high, improved=(d>0).sum(),
                         tied=(d==0).sum(), paired_dz=d.mean()/sd if sd else np.nan, p=p))
    df = pd.DataFrame(rows)
    if len(df):
        order = df.p.sort_values().index
        adjusted = np.maximum.accumulate(df.loc[order,"p"] * np.arange(len(df), 0, -1))
        df.loc[order,"holm_p"] = np.minimum(1, adjusted)
    return df
