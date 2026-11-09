"""Framework-free inference for a scikit-learn HistGradientBoostingClassifier (binary, numeric features only).
Jetson Nano (JetPack 4.x, Python 3.6-3.8) cannot install a modern scikit-learn, so the trained trees are exported to plain arrays
(export_gbm) and evaluated with numpy only (GbmNumpy). Verified against sklearn's predict_proba in build_deploy.py."""
import numpy as np


def export_gbm(model, path):
    trees = [it[0].nodes for it in model._predictors]
    n = max(len(t) for t in trees)
    f = lambda k, dt: np.stack([np.pad(t[k].astype(dt), (0, n - len(t))) for t in trees])
    np.savez_compressed(path, baseline=np.float64(np.ravel(model._baseline_prediction)[0]), value=f("value", np.float64),
                        feature=f("feature_idx", np.int32), thr=f("num_threshold", np.float64), left=f("left", np.int32),
                        right=f("right", np.int32), leaf=f("is_leaf", np.bool_), mgl=f("missing_go_to_left", np.bool_))


class GbmNumpy:
    def __init__(self, path):
        z = np.load(path)
        self.__dict__.update({k: z[k] for k in z.files})

    def predict_proba1(self, X):
        """X: (n, n_features) float array (NaN = missing) -> P(event) (n,)"""
        X = np.asarray(X, dtype=np.float64)
        raw = np.full(len(X), float(self.baseline))
        rows = np.arange(len(X))
        for t in range(len(self.value)):
            node = np.zeros(len(X), dtype=np.int64)
            for _ in range(64):
                leaf = self.leaf[t][node]
                if leaf.all():
                    break
                x = X[rows, self.feature[t][node]]
                go_left = np.where(np.isnan(x), self.mgl[t][node], x <= self.thr[t][node])
                nxt = np.where(go_left, self.left[t][node], self.right[t][node])
                node = np.where(leaf, node, nxt)
            raw += self.value[t][node]
        return 1.0 / (1.0 + np.exp(-raw))
