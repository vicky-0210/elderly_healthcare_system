"""Replay saved LOSO classical predictions and audit participant separation."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE/'src'))
from wearable_eda.training import classical
from wearable_eda.training.classical_run import SUBJECTS, feature_data, representation, source_state


def verify(directory, cache, state):
    spec = json.loads((directory/'meta.json').read_text())
    if any(spec.get(k) != value for k, value in state.items()):
        raise ValueError(f'Source fingerprint mismatch: {directory.name}')
    data = feature_data(cache, representation(spec), state)
    folds = pd.read_csv(directory/'folds.csv')
    saved = pd.read_csv(directory/'predictions.csv')
    if len(folds) != 15 or sorted(folds.subject) != list(SUBJECTS) or len(saved) != len(data['y']):
        raise ValueError('Unexpected participant/window counts')
    for row in folds.itertuples():
        subject = int(row.subject)
        mask = data['groups'] == subject
        observed = saved[saved.subject == subject]
        checkpoint = directory/'subjects'/f'S{subject}'/'model.joblib'
        model = classical.load_bundle(checkpoint)
        valid = classical.inner_subjects(list(SUBJECTS), subject)
        refit = sorted(set(SUBJECTS) - {subject})
        assert json.loads(row.inner_subjects) == valid
        assert json.loads(row.inner_fit_subjects) == sorted(set(refit) - set(valid))
        assert json.loads(row.refit_subjects) == model['train_subjects'] == refit
        assert model['parameters'] == json.loads(row.parameters)
        assert np.isclose(model['temperature'], row.temperature)
        assert checkpoint.stat().st_size == row.artifact_bytes
        assert np.array_equal(data['start'][mask], observed.start)
        assert np.array_equal(data['end'][mask], observed.end)
        assert np.array_equal(data['y'][mask], observed.y_true)
        probabilities = classical.predict_windows(model, data['F'][mask], data['features'])
        columns = [f'p{i}' for i in range(len(spec['classes']))]
        np.testing.assert_allclose(probabilities, observed[columns].to_numpy(), rtol=1e-6, atol=1e-7)
        np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    print(f"{directory.name}: replayed all {len(saved)} windows from all 15 checkpoints", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/eda/processed/wesad_v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/eda/runs/wesad_loso_v1/classical'))
    parser.add_argument('--experiment', action='append', help='Repeat to replay selected experiment names.')
    args = parser.parse_args()
    directories = ([args.output/name for name in args.experiment] if args.experiment else
                   sorted(p.parent for p in args.output.glob('*/meta.json')))
    if not directories:
        parser.error('No experiments found.')
    state = source_state(args.cache)
    with threadpool_limits(1):
        for directory in directories:
            verify(directory, args.cache, state)


if __name__ == '__main__':
    main()
