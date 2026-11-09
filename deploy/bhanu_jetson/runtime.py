"""Streaming apnea/hypopnea detector for the Jetson Nano: ensemble of an SpO2 CNN and an SpO2 gradient-boosting model.

Feed ONE value per second:  det.push(spo2, ppg_ok)   spo2 = SpO2 in % (NaN if no reading), ppg_ok = False if that second of PPG
is unusable (NaN / clipped / flat, see ppg_second_ok). Every 10 s it returns a decision for the 60 s window that ended 5 s ago
(5 s extra latency because the SpO2 smoothing used in training looks up to 4 s ahead), otherwise None.
Decision = dict(t_end, p_cnn, p_gbm, score, alarm, ok)   ok=False -> window had >10% unusable seconds (no decision, alarm False).
Only numpy / pandas / scipy / torch are needed (no scikit-learn).
"""
import numpy as np
import torch

from gbm_numpy import GbmNumpy
from models import BaselineCNN
from spo2_features import cnn_channels, gbm_features

FS_MODEL, WIN, HOP, DELAY = 25, 60, 10, 5
KEEP_S = 3600          # seconds of history kept (multiple of 10); >= 2400 s needed for the 30-minute context features


def ppg_second_ok(x, state):
    """x: 80 raw PPG samples of one second. state: dict holding the expanding min/max (causal clipping detector)."""
    if np.isnan(x).any():
        return False
    state["mx"], state["mn"] = max(state.get("mx", -np.inf), x.max()), min(state.get("mn", np.inf), x.min())
    return not (((x == state["mx"]) | (x == state["mn"])).mean() >= 0.2 or x.max() == x.min())


class Detector:
    def __init__(self, folder="."):
        ck = torch.load(f"{folder}/cnn_spo2.pt", map_location="cpu", weights_only=False)
        self.cnn = BaselineCNN(ck["n_ch"], 2)
        self.cnn.load_state_dict(ck["state"]); self.cnn.eval()
        self.norm = (ck["norm"][0].astype(np.float32), ck["norm"][1].astype(np.float32))
        self.gbm = GbmNumpy(f"{folder}/gbm.npz")
        cal = np.load(f"{folder}/calibration.npz")
        self.grid_cnn, self.grid_gbm, self.thr = cal["cnn"], cal["gbm"], float(cal["thr"])
        self.s, self.ok, self.b0, self.t = [], [], 0, 0

    @staticmethod
    def _rank(grid, p):
        return float(np.searchsorted(grid, p, side="right")) / len(grid)

    def push(self, spo2, ppg_ok):
        good = bool(ppg_ok) and np.isfinite(spo2) and 50 < spo2 <= 100
        self.s.append(float(spo2) if good else np.nan)
        self.ok.append(bool(ppg_ok))
        self.t += 1
        if len(self.s) > KEEP_S + 120:                              # trim, keeping the buffer start a multiple of 10 s
            cut = (len(self.s) - KEEP_S) // HOP * HOP
            self.s, self.ok, self.b0 = self.s[cut:], self.ok[cut:], self.b0 + cut
        if self.t < WIN + DELAY or (self.t - WIN - DELAY) % HOP:
            return None
        return self._decide()

    def _decide(self):
        start = self.t - WIN - DELAY                                # absolute start second of the window
        i0 = start - self.b0
        s = np.array(self.s)
        okv = np.array(self.ok[i0:i0 + WIN])
        out = dict(t_end=start + WIN, ok=bool((~okv).mean() <= 0.10))
        if not out["ok"]:
            return dict(out, p_cnn=np.nan, p_gbm=np.nan, score=np.nan, alarm=False)
        f = gbm_features(s, self.ok).iloc[[-1]].values              # last kept window = the window that started at `start`
        p_gbm = float(self.gbm.predict_proba1(f)[0])
        ch = cnn_channels(s)[:, i0:i0 + WIN]
        m = np.repeat(okv.astype(np.float32), FS_MODEL)
        x = np.repeat(ch, FS_MODEL, axis=1).astype(np.float32)
        x = np.clip((x - self.norm[0][:, None]) / self.norm[1][:, None], -10, 10) * m
        x = np.concatenate([x, m[None]])[None]
        with torch.no_grad():
            p_cnn = float(torch.softmax(self.cnn(torch.from_numpy(x)), 1)[0, 1])
        score = (self._rank(self.grid_cnn, p_cnn) + self._rank(self.grid_gbm, p_gbm)) / 2
        return dict(out, p_cnn=p_cnn, p_gbm=p_gbm, score=score, alarm=bool(score >= self.thr))
