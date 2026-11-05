"""ECG inference boundary; no model is implemented."""

from wearable_common.inference import SignalPredictor


class ECGPredictor(SignalPredictor):
    """Implement ECG-specific input validation and matching preprocessing here."""

    def load(self, artifact_directory):
        raise NotImplementedError("ECG artifact loading is not implemented.")

    def predict(self, window):
        raise NotImplementedError("ECG prediction is not implemented.")
