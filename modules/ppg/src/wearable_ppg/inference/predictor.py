"""PPG inference boundary; no model is implemented."""

from wearable_common.inference import SignalPredictor


class PPGPredictor(SignalPredictor):
    """Implement PPG-specific input validation and matching preprocessing here."""

    def load(self, artifact_directory):
        raise NotImplementedError("PPG artifact loading is not implemented.")

    def predict(self, window):
        raise NotImplementedError("PPG prediction is not implemented.")
