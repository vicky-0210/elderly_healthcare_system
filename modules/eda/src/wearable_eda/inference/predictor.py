"""Research inference from time-aligned wrist streams, including signal abstention."""
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
from wearable_common.inference import SignalPredictor
from wearable_eda.preprocessing.features import window_features
from wearable_eda.preprocessing.wrist import transform_recording


RATES = {"eda":4, "acc":32, "temp":4, "bvp":64}
UNITS = {"eda":"uS", "acc":"g", "temp":"degC", "bvp":"E4_arbitrary"}


class EDAPredictor(SignalPredictor):
    """Score the last completed window of a synchronized recording prefix.

    Each stream carries samples, per-sample timestamps (seconds, same clock),
    and a unit. Inputs begin at the common session origin on every call so
    the causal decomposition has its true history. Call at the exported hop.
    """
    def __init__(self):
        self.contract = None

    def load(self, artifact_directory):
        root = Path(artifact_directory)
        contract = json.loads((root/"contract.json").read_text(encoding="utf-8"))
        if (contract["format"] != "wesad-wrist-v1" or contract["classes"] != [1,2]
                or contract["calibration"] not in ("zero","rest") or contract["decomposition"] != "mcsaed"):
            raise ValueError("Incompatible EDA export contract")
        family = contract["family"]
        if family not in ("neural","classical"):
            raise ValueError("Unknown predictor family")
        payload = contract.get("payload", "model.pt" if family=="neural" else "model.joblib")
        if payload not in (("model.pt",) if family=="neural" else ("model.joblib","linear.json")):
            raise ValueError("Unexpected model payload")
        weights = root/payload
        if sha256(weights.read_bytes()).hexdigest() != contract["weights_sha256"]:
            raise ValueError("Model checksum mismatch")
        if family == "neural":
            from .neural import NeuralPredictor
            model = NeuralPredictor(weights)
        elif payload=="linear.json":
            from .linear import LinearPredictor
            model = LinearPredictor(weights)
        else:
            from wearable_eda.training.classical import load_bundle
            model = load_bundle(weights)
        self.contract, self.model = contract, model
        return self

    def predict(self, window):
        result = dict(schema_version="1.0",signal="eda",status="model_unavailable",scores=None,
                      stream_id=window.get("stream_id"),clock_domain=window.get("clock_domain"),
                      quality_flags=[])
        if self.contract is None:
            return result
        cfg = self.contract
        result.update(model_version=cfg["experiment"],preprocessing_version=cfg["format"],
                      threshold_version="temperature-scaled-argmax-v1")
        try:
            if not result["clock_domain"] or not result["stream_id"]:
                raise ValueError("Supply a clock domain and stream ID")
            values, origin, n = self._inputs(window["streams"])
            length = int(cfg["seconds"]*4)
            if n < length:
                raise ValueError("Awaiting a complete observation window")
            result.update(window_start=origin+(n-length)/4,window_end=origin+n/4)
            observed = values["eda"][-length:]
            valid = np.isfinite(observed) & (observed >= 0) & (observed <= 100)
            transitions = np.flatnonzero(np.diff(np.r_[False, ~valid, False])).reshape(-1,2)
            if valid.mean()<.8 or np.any(transitions[:,1]-transitions[:,0]>=20):
                raise ValueError("Too much missing or out-of-range EDA")
            if np.ptp(observed[valid])<1e-8:
                raise ValueError("Flat EDA throughout the observation window")
            for key in values.keys()-{"eda"}:
                portion = values[key][-int(cfg["seconds"]*RATES[key]):]
                if not np.isfinite(portion).all():
                    raise ValueError(f"Missing observations in required {key} stream")
            rest = None
            if cfg["calibration"] == "rest":
                interval = np.asarray(window.get("rest_interval_s",[]),dtype=float)
                if interval.shape != (2,) or not np.isfinite(interval).all():
                    raise ValueError("Designate a two-minute resting interval on the supplied session clock")
                ticks = (interval-origin)*4
                rest = np.rint(ticks).astype(int)
                if (not np.allclose(ticks,rest,atol=.008,rtol=0) or rest[0]<0
                        or rest[1]-rest[0]!=480 or rest[1]>n-length):
                    raise ValueError("Reserve 120 seconds of aligned resting data before the scored window")
            matrix, names = transform_recording(values, rest=rest, modalities=cfg["modalities"],
                          calibration=cfg["calibration"],decomposition=cfg["decomposition"],
                          filtering=cfg["filtering"],quality=cfg["quality"])
            X = matrix[-length:].T
            quality = float(X[names.index("quality")].mean())
            result["mean_eda_quality"] = quality
            if quality<.4:
                raise ValueError("Low EDA contact quality")
            f, features = window_features(X,names)
        except (KeyError,ValueError,TypeError) as exc:
            result.update(status="insufficient_signal",quality_flags=[str(exc)])
            return result
        if cfg["family"]=="neural":
            probabilities = self.model.predict_windows(X[None],f[None],names,features)[0]
        elif cfg.get("payload")=="linear.json":
            probabilities = self.model.predict_windows(f[None],features)[0]
        else:
            from wearable_eda.training.classical import predict_windows
            probabilities = predict_windows(self.model,f[None],features)[0]
        if not np.isfinite(probabilities).all():
            raise ArithmeticError("Non-finite model output")
        result.update(status="ok",scores=dict(zip(("baseline","stress_associated_arousal"),
                                                  map(float,probabilities))))
        return result

    def _inputs(self, streams):
        values, origins, ticks = {}, [], []
        for key in self.contract["modalities"]:
            stream = streams[key]
            if stream["unit"] != UNITS[key]:
                raise ValueError(f"{key} requires {UNITS[key]}")
            x = np.asarray(stream["samples"],dtype=float)
            ts = np.asarray(stream["timestamps"],dtype=float)
            if key=="acc" and (x.ndim!=2 or x.shape[1]!=3):
                raise ValueError("ACC requires three time-major axes")
            if key!="acc" and x.ndim!=1:
                raise ValueError(f"{key} requires a scalar time series")
            if (ts.ndim!=1 or not len(ts) or len(x)!=len(ts) or not np.isfinite(ts).all()
                    or not np.allclose(ts-ts[0],np.arange(len(ts))/RATES[key],rtol=0,atol=.002)):
                raise ValueError(f"Missing or irregular timestamps in {key}")
            factor = RATES[key]//4
            if len(x)%factor:
                raise ValueError(f"Incomplete quarter-second bin in {key}")
            values[key] = x*64 if key=="acc" else x
            origins.append(float(ts[0]))
            ticks.append(len(x)//factor)
        if max(origins)-min(origins)>.002 or len(set(ticks))!=1:
            raise ValueError("Required wrist streams must share their origin and duration")
        return values,origins[0],ticks[0]
