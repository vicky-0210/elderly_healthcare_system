"""Portable binary logistic inference in the same feature order as the fitted model."""
import json
from pathlib import Path
import numpy as np
from scipy.special import expit,softmax


class LinearPredictor:
    def __init__(self, path):
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        self.features = spec["features"]
        self.intercept = float(spec["intercept"])
        self.temperature = float(spec["temperature"])
        for key in ("impute","center","scale","weights"):
            array = np.asarray(spec[key],dtype=np.float64)
            if array.shape!=(len(self.features),) or not np.isfinite(array).all():
                raise ValueError(f"Invalid portable logistic parameter {key}")
            setattr(self,key,array)
        if np.any(self.scale<=0) or not np.isfinite(self.temperature) or self.temperature<=0:
            raise ValueError("Invalid feature or temperature scale")

    def predict_windows(self, features, names):
        if list(names)!=self.features:
            raise ValueError("Feature order differs from the exported model")
        x = np.asarray(features,dtype=np.float64)
        if x.ndim!=2 or x.shape[1]!=len(self.features) or np.isinf(x).any():
            raise ValueError("Expected a window-by-feature matrix")
        scaled = (np.where(np.isnan(x),self.impute,x)-self.center)/self.scale
        positive = expit(scaled @ self.weights + self.intercept)
        scores = np.column_stack((1-positive,positive))
        return softmax(np.log(np.clip(scores,1e-7,1.0))/self.temperature, axis=1)
