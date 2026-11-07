"""Binary dense event scores from short optical pulses and a longer modulation history."""
import torch
from torch import nn
from torch.nn import functional as F

from wearable_ppg.models.respiratory import RespiratoryNet, TemporalBlock
from wearable_ppg.preprocessing.optical import MORPH, MOD, QUALITY


VARIANTS = ("binary_tcn_gru", "dual_hard", "dual_fraction", "dual_aux")
SHORT = 12
LONG = 90
FRACTION = 10


class BinaryControl(nn.Module):
    """Same 30-second feature encoder as the original TCN + GRU."""
    context = 30

    def __init__(self):
        super().__init__()
        self.encoder = RespiratoryNet("tcn_gru")
        self.encoder.head[-1] = nn.Linear(self.encoder.head[-1].in_features, 1)

    def forward(self, frames):
        return self.encoder(frames[:, :, :-1])


class DualRate(nn.Module):
    context = LONG

    def __init__(self, auxiliary=False):
        super().__init__()
        width = 24
        self.pulse_filter = nn.Conv1d(1, 12, kernel_size=9)
        self.pulse_time = nn.Sequential(TemporalBlock(width, 1), TemporalBlock(width, 2))
        self.modulation = nn.Linear(len(MOD) + len(QUALITY) + 1, width)
        self.slow_time = nn.Sequential(*(TemporalBlock(width, d) for d in (2, 4, 8)))
        self.gru = nn.GRU(width, width, batch_first=True)
        self.fusion = nn.Sequential(nn.Linear(2*width, width), nn.LayerNorm(width), nn.SiLU(), nn.Dropout(.15))
        self.event = nn.Linear(width, 1)
        self.fraction = nn.Linear(width, 1) if auxiliary else None

    def forward(self, frames):
        # The 16 pulse samples per second are uniformly spaced in time.
        pulse = frames[:, -SHORT:, :MORPH].reshape(len(frames), 1, SHORT*MORPH)
        filtered = F.silu(self.pulse_filter(F.pad(pulse, (8, 0)))).unflatten(-1, (SHORT, MORPH))
        pulse_stats = torch.cat((filtered.mean(-1), filtered.amax(-1)), dim=1)
        pulse_state = self.pulse_time(pulse_stats.transpose(1, 2))[:, -5:]
        slow = self.slow_time(F.silu(self.modulation(frames[:, :, MORPH:])))
        hidden, _ = self.gru(slow)
        combined = self.fusion(torch.cat((pulse_state, hidden[:, -5:]), dim=-1))
        event = self.event(combined)
        if self.fraction is None:
            return event
        return torch.cat((event, self.fraction(combined)), dim=-1)


def make_model(name):
    if name not in VARIANTS:
        raise ValueError(f"Unknown binary experiment {name}")
    return BinaryControl() if name == "binary_tcn_gru" else DualRate(name == "dual_aux")
