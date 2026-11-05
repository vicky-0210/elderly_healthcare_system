"""IMU inference boundary; no model is implemented."""

from wearable_common.inference import SignalPredictor


class IMUPredictor(SignalPredictor):
    """Implement IMU-specific input validation and matching preprocessing here."""

    def load(self, artifact_directory):
        raise NotImplementedError("IMU artifact loading is not implemented.")

    def predict(self, window):
        raise NotImplementedError("IMU prediction is not implemented.")
