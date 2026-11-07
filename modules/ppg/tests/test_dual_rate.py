import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from wearable_ppg.models.dual_rate import VARIANTS, make_model
from wearable_ppg.preprocessing.optical import NAMES
from wearable_ppg.training.dual_rate import run_starts, trailing_targets, window_batch


def test_trailing_fraction_keeps_boundaries():
    y = np.zeros(40)
    y[25:30] = 4
    y[34:] = 1
    hard, fraction = trailing_targets(y, np.array([29, 34]))
    np.testing.assert_equal(hard, [[1]*5, [0, 0, 0, 0, 1]])
    np.testing.assert_allclose(fraction, [[.1, .2, .3, .4, .5], [.5, .5, .5, .5, .6]])


def test_history_resets_at_missing_packet_and_clock_gap():
    t = np.r_[np.arange(8), 20, 21, 22]
    valid = np.ones(len(t), bool)
    valid[3] = False
    starts = torch.tensor(run_starts(t, valid))
    x = torch.tensor(np.arange(len(t))[:, None], dtype=torch.float32)
    batch = window_batch(x, starts, np.array([6, 9]), 5).numpy()
    np.testing.assert_equal(batch[:, :, -1], [[0,0,1,1,1], [0,0,0,1,1]])
    np.testing.assert_equal(batch[:, :, 0], [[0,0,4,5,6], [0,0,0,8,9]])


@pytest.mark.parametrize("name", VARIANTS)
def test_no_future_leakage_per_second(name):
    torch.manual_seed(12)
    model = make_model(name).eval()
    x = torch.randn(2, model.context, len(NAMES)+1)
    with torch.inference_mode():
        original = model(x)
        x[:, -2:] += 10
        shifted = model(x)
    assert original.shape[:2] == (2, 5)
    torch.testing.assert_close(original[:, :3], shifted[:, :3])
