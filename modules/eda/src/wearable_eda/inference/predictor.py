"""EDA/GSR inference boundary; no model is implemented."""

from wearable_common.inference import SignalPredictor


class EDAPredictor(SignalPredictor):
    """Implement EDA/GSR-specific input validation and matching preprocessing here."""

    def load(self, artifact_directory):
        raise NotImplementedError("EDA/GSR artifact loading is not implemented.")

    def predict(self, window):
        raise NotImplementedError("EDA/GSR prediction is not implemented.")
