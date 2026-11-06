"""Inference with fold-fitted frozen statistics and named channel contracts."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import torch

from wearable_eda.models.wrist import WristNet


def robust_scale(values, axis):
    quartiles = np.percentile(values, [25, 50, 75], axis=axis)
    center = quartiles[1]
    spread = quartiles[2] - quartiles[0]
    fallback = np.std(values, axis=axis, dtype=np.float64)
    spread = np.where(spread > 1e-5, spread, np.where(fallback > 1e-5, fallback, 1.))
    return center.astype("f"), spread.astype("f")


def fit_scaler(x, features, train, channels):
    # Deterministic one-sample-per-second subset for training quantiles.
    xc, xs = robust_scale(x[train, :, ::4], (0, 2))
    fc, fs = robust_scale(features[train], 0)
    for col, name in enumerate(channels):
        if name in ("quality", "artifact", "missing"):
            xc[col], xs[col] = 0., 1.
    return dict(xc=xc, xs=xs, fc=fc, fs=fs)


def scale(x, f, scaler):
    clean = np.clip((x - scaler["xc"][None,:,None]) / scaler["xs"][None,:,None], -30, 30)
    feats = np.clip((f - scaler["fc"]) / scaler["fs"], -30, 30)
    return clean.astype("f", copy=False), feats.astype("f", copy=False)


class NeuralPredictor:
    def __init__(self, path, device="cpu"):
        self.device = torch.device(device)
        state = torch.load(Path(path), map_location=self.device, weights_only=True)
        self.spec = state["spec"]
        self.channels = list(state["channels"])
        self.features = list(state["features"])
        self.classes = list(self.spec["classes"])
        self.temperature = state["temperature"]
        self.window_samples = state["window_samples"]
        self.scaler = {k: v.cpu().numpy() for k, v in state["scaler"].items()}
        self.model = WristNet(self.spec, self.channels, len(self.features), len(self.classes),
                              self.scaler["xc"], self.scaler["xs"]).to(self.device)
        self.model.load_state_dict(state["weights"])
        self.model.eval()

    def predict_windows(self, X, F, channels, features, *, batch_size=128):
        if batch_size < 1:
            raise ValueError("Batch size must be positive")
        if list(features) != self.features or not all(n in channels for n in self.channels):
            raise ValueError("Window feature/channel contract differs from the saved model")
        if X.ndim != 3 or F.ndim != 2 or X.shape[1] != len(channels) or F.shape != (len(X),len(features)):
            raise ValueError("Expected X[N,C,T] and F[N,D]")
        if X.shape[-1] != self.window_samples:
            raise ValueError(f"Expected {self.window_samples} samples per channel")
        if not np.isfinite(X).all() or not np.isfinite(F).all():
            raise ValueError("Preprocessing must supply finite, imputed model inputs")
        x, f = scale(X[:, [list(channels).index(n) for n in self.channels]], F, self.scaler)
        chunks = []
        with torch.inference_mode():
            for i in range(0,len(x),batch_size):
                a = torch.from_numpy(x[i:i+batch_size]).to(self.device)
                b = torch.from_numpy(f[i:i+batch_size]).to(self.device)
                logits = self.model(a,b)[0]
                chunks.append((logits / self.temperature).softmax(1).cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0,len(self.classes)),dtype="f")
