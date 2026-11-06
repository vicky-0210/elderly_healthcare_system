"""Fixed architecture comparison and paired experiments for WESAD wrist data."""
from __future__ import annotations


def catalogue():
    records = {}

    def add(name, model, calibration="zero", modalities=("eda",), role="primary",
            category="standalone", classes=(1, 2), decomposition="mcsaed", **settings):
        key = f"{name}_{calibration}"
        records[key] = dict(id=key, model=model, calibration=calibration,
                            modalities=list(modalities), role=role, category=category,
                            classes=list(classes), decomposition=decomposition, **settings)

    standalone = ("cnn", "lstm", "bilstm", "gru", "bigru", "tcn", "multitcn", "transformer")
    hybrids = ("cnn_bilstm", "cnn_transformer", "tcn_attention", "tonic_phasic",
               "multitcn_gate", "tonic_attention", "hybrid_features", "multitcn_gru",
               "multitcn_transformer", "cnn_tcn_attention")
    for protocol in ("zero", "rest"):
        for name in standalone + hybrids:
            add(name, name, protocol, category="standalone" if name in standalone else "hybrid")
        for model in ("pg", "reliable", "physics"):
            add(model, model, protocol, ("eda", "acc"), category="hybrid")
        for name, mods in (("eda", ("eda",)), ("temp", ("eda", "acc", "temp")),
                           ("fullwrist", ("eda", "acc", "temp", "bvp"))):
            add("pg_" + name, "pg", protocol, mods, category="hybrid")
        add("masked_multitcn", "multitcn", protocol, category="ssl", ssl="masked")
        add("contrast_tonic", "tonic_attention", protocol, category="ssl", ssl="contrast")
    for fusion in ("concat", "weighted", "gated", "cross"):
        add("pg_" + fusion, "pg", modalities=("eda", "acc"), fusion=fusion,
            role="ablation", category="hybrid")
    for agg in ("gru", "bigru", "transformer"):
        add("pg_" + agg, "pg", modalities=("eda", "acc"), aggregation=agg,
            role="ablation", category="hybrid")
    variants = {
        "pg_gru_cross": {"aggregation": "gru", "fusion": "cross"},
        "pg_no_hand": {"hand": False},
        "pg_no_components_no_hand": {"hand": False, "drop": "components"},
        "pg_single_scale": {"multiscale": False},
        "pg_focal": {"loss": "focal"},
    }
    for name, options in variants.items():
        add(name, "pg", modalities=("eda", "acc"), category="hybrid", role="ablation", **options)
    add("pg_no_relative_no_hand", "pg", "rest", ("eda", "acc"), category="hybrid",
        role="ablation", hand=False, drop="relative")
    add("pg_no_hand", "pg", "rest", ("eda", "acc"), category="hybrid",
        role="ablation", hand=False)
    add("physics_no_aux", "physics", modalities=("eda", "acc"), category="hybrid",
        role="ablation", physics_weight=0.)
    for cls in ((1, 2, 3), (1, 2, 3, 4)):
        add(f"pg_{len(cls)}class", "pg", modalities=("eda", "acc"), classes=cls,
            category="hybrid", role="secondary")
    add("pg_cvxeda", "pg", modalities=("eda", "acc"), decomposition="cvxeda",
        category="hybrid", role="ablation")
    add("pg_cosine", "pg", modalities=("eda", "acc"), schedule="cosine",
        category="hybrid", role="ablation")
    add("pg_intrinsic_quality", "pg", modalities=("eda","acc"), quality="eda",
        category="hybrid", role="ablation")
    add("pg_cv_physics", "physics", modalities=("eda", "acc"), decomposition="cvxeda",
        category="hybrid", role="ablation")
    for name, model, sensors in (("pg_long","pg",("eda","acc")),
                                 ("pg_fullwrist_long","pg",("eda","acc","temp","bvp")),
                                 ("multitcn_long","multitcn",("eda",))):
        add(name,model,modalities=sensors,epoch_limit=18,role="budget_sensitivity",
            category="standalone" if model=="multitcn" else "hybrid")
    return records


def arguments(spec):
    return {k: spec[k] for k in ("calibration", "modalities", "classes", "decomposition", "quality") if k in spec}


def selected_channels(channels, spec):
    drop = spec.get("drop")
    if drop == "relative":
        return [n for n in channels if not n.startswith("relative_")]
    if drop == "components":
        derived = {"tonic", "phasic", "relative_tonic", "relative_phasic", "tonic_slope",
                   "phasic_slope", "phasic_energy"}
        return [n for n in channels if n not in derived]
    return list(channels)

