"""Framework-neutral predictor interface; payloads are documented in contracts/."""

from abc import ABC, abstractmethod


class SignalPredictor(ABC):
    """External inference contract shared by otherwise independent signal pipelines."""

    @abstractmethod
    def load(self, artifact_directory):
        """Load a validated export and its matching preprocessing/input metadata."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, window):
        """Validate a signal window and return a versioned prediction/status payload."""
        raise NotImplementedError
