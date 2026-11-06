"""PTB-XL statement prediction, not echocardiographic LVH diagnosis."""
import ast
import numpy as np
import pandas as pd


def annotate(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    codes = out.scp_codes.map(ast.literal_eval)
    out["label"] = codes.map(lambda c: int("LVH" in c))
    out["contradiction"] = codes.map(lambda c: "LVH" in c and "NORM" in c)
    out["strict_normal"] = codes.map(lambda c: "NORM" in c and set(c) <= {"NORM", "SR"})
    out["lvh_confidence"] = codes.map(lambda c: c.get("LVH", np.nan))
    out["isolated_lvh"] = codes.map(lambda c: "LVH" in c and set(c) <= {"LVH", "SR"})
    out["stt_comorbidity"] = codes.map(lambda c: bool(set(c) & {"ISC_", "NST_", "NDT", "DIG", "ISCAL", "ISCIN", "ISCLA", "ISCAS", "ISCI"}))
    out["conduction_comorbidity"] = codes.map(lambda c: bool(set(c) & {"CLBBB", "CRBBB", "ILBBB", "IRBBB", "LAFB", "LPFB", "IVCD", "1AVB", "2AVB", "3AVB", "WPW"}))
    out["age_encoded"] = out.age > 200
    out["age65plus"] = out.age >= 65
    out["age_for_control"] = out.age.where(out.age <= 100, 90)
    out["patient_id"] = out.patient_id.astype(int)
    out["row_index"] = np.arange(len(out))
    out["partition"] = np.select([out.strat_fold <= 6, out.strat_fold <= 8, out.strat_fold == 9],
                                 ["train_inner", "development", "calibration"], default="test")
    return out


def cohort(frame, task):
    if task not in {"lvh_vs_normal", "lvh_vs_rest"}: raise ValueError(task)
    keep = ~frame.contradiction.astype(bool)
    if task == "lvh_vs_normal": keep &= frame.label.eq(1) | frame.strict_normal.astype(bool)
    if "usable" in frame: keep &= frame.usable.astype(bool)
    return frame.loc[keep].copy()


def assert_patient_folds(frame):
    if frame.groupby("patient_id").strat_fold.nunique().max() != 1:
        raise ValueError("A patient crosses official folds")
    if not frame.strat_fold.isin(range(1, 11)).all(): raise ValueError("Invalid official fold")
    if frame.ecg_id.duplicated().any(): raise ValueError("Duplicate ECG ID")
