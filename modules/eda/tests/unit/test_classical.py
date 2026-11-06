"""Regression checks for held-out partition isolation and feature-order rejection."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from wearable_eda.training import classical
from wearable_eda.training.classical_run import SUBJECTS, experiments, fit_fold


class ClassicalFoldTests(unittest.TestCase):
    def test_outer_replay_and_train_only_scaler(self):
        rng = np.random.default_rng(121)
        groups = np.repeat(SUBJECTS, 12)
        y = np.tile([0, 1], len(groups)//2)
        f = rng.normal(size=(len(y), 3))
        f[:, 1] += y * 1.3
        held_out = groups == 2
        f[held_out, 0] += 1e6
        start = np.tile(np.arange(12) * 15, 15)
        data = dict(F=f, features=['scl_mean', 'scr_count', 'quality_mean'],
                    y=y, groups=groups, start=start, end=start+60)
        spec = next(s for s in experiments() if s['name'] == 'logistic__eda__zero')

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            row = fit_fold(spec, data, 2, root, 'test-schema')
            self.assertEqual(row['inner_subjects'], [3, 10])
            self.assertEqual(set(row['inner_fit_subjects']), set(SUBJECTS) - {2, 3, 10})
            self.assertEqual(set(row['refit_subjects']), set(SUBJECTS) - {2})
            checkpoint = root/'subjects'/'S2'/'model.joblib'
            model = classical.load_bundle(checkpoint)
            scaler = model['fitted'][0].steps[-1][1]
            np.testing.assert_allclose(scaler.center_, np.median(f[~held_out], axis=0))
            saved = pd.read_csv(checkpoint.with_name('predictions.csv'))
            replay = classical.predict_windows(model, f[held_out], data['features'])
            np.testing.assert_allclose(replay, saved[['p0', 'p1']], atol=1e-12)
            with self.assertRaisesRegex(ValueError, 'Feature names/order'):
                classical.predict_windows(model, f[held_out], list(reversed(data['features'])))
            with self.assertRaisesRegex(ValueError, 'provenance mismatch'):
                fit_fold(spec, data, 2, root, 'different-run')
            meta = json.loads(checkpoint.with_name('fit.json').read_text())
            self.assertEqual(meta['n_test'], 12)


if __name__ == '__main__':
    unittest.main()
