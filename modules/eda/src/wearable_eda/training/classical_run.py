"""Participant-isolated WESAD experiments for classical feature classifiers."""
from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from importlib import import_module
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_config
from threadpoolctl import threadpool_limits

from . import classical


SUBJECTS = tuple(n for n in range(2, 18) if n != 12)
SEED = 6703
BASE = dict(seconds=60, stride=15, purity=.8, classes=(1, 2), calibration='zero',
            decomposition='mcsaed', modalities=('eda',), filtering='conservative', quality='auto')


def experiments():
    def specification(name, model, role, **updates):
        return dict(name=name, model=model, group='classical', role=role, **(BASE | updates))

    specs = []
    for c in ('zero', 'rest'):
        for m in (('eda',), ('eda', 'acc'), ('eda', 'acc', 'temp'), ('eda', 'acc', 'temp', 'bvp')):
            for model in classical.GRIDS:
                specs.append(specification(f"{model}__{'-'.join(m)}__{c}", model, 'candidate',
                                          calibration=c, modalities=m))
    # Predeclared DSP and segmentation controls. Compare against the identical
    # random forest configuration with the unmodified processing setting.
    for variable, choices in (
        ('seconds', (30, 90)), ('stride', (30,)), ('purity', (1.0,)),
        ('filtering', ('raw', '1hz')), ('decomposition', ('none', 'sparse', 'choi')),
        ('calibration', ('rolling',)),
    ):
        for value in choices:
            specs.append(specification(f'ablation__rf__{variable}-{value}', 'random_forest',
                                      'ablation', **{variable: value}))
    for model in classical.GRIDS:
        specs.append(specification(f'ablation__{model}__cvxeda', model, 'ablation', decomposition='cvxeda'))
    for q in ('eda', 'off'):
        specs.append(specification(f'ablation__rf__eda-acc__quality-{q}', 'random_forest', 'ablation',
                                   modalities=('eda', 'acc'), quality=q))
    for m in ('acc', 'temp'):
        specs.append(specification(f'control__rf__{m}', 'random_forest', 'negative_control', modalities=(m,)))
    for classes in ((1, 2, 3), (1, 2, 3, 4)):
        for model in classical.GRIDS:
            specs.append(specification(f'secondary{len(classes)}__{model}', model, 'secondary', classes=classes))
    return specs


def representation(spec):
    return {k: spec[k] for k in BASE}


def json_text(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False)


def write_json(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    tmp.write_text(json_text(value), encoding='utf-8')
    tmp.replace(path)


def table(path, frame):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def source_state(cache):
    from wearable_eda import datasets as dataset_package
    module_root = Path(dataset_package.__file__).resolve().parents[1]
    files = [path for directory in ('datasets', 'preprocessing', 'windowing', 'labels')
             for path in sorted((module_root / directory).rglob('*.py'))]
    return {
        'manifest_sha256': digest((cache / 'manifest.json').read_bytes()),
        'signals_sha256': digest(b''.join(path.read_bytes() for path in files)),
        'training_sha256': digest(Path(__file__).read_bytes() + Path(classical.__file__).read_bytes()),
        'versions': {k: version(k) for k in ('numpy', 'scipy', 'scikit-learn', 'lightgbm', 'xgboost', 'joblib')},
        'seed': SEED,
    }


def feature_data(cache, options, source):
    feature_id = digest(json_text([source['manifest_sha256'], source['signals_sha256'], options]).encode())[:24]
    folder = cache / 'classical_features'
    folder.mkdir(exist_ok=True)
    target = folder / (feature_id + '.npz')
    if not target.exists():
        # Context-only controls use the identical paired-window cohort. They
        # select columns computed exclusively from the named context stream.
        modalities = tuple(options['modalities'])
        context_only = modalities in (('acc',), ('temp',))
        paired_options = options | {'modalities': ('eda', 'acc', 'temp')} if context_only else options
        raw = import_module('wearable_eda.datasets.wesad').build_windows(cache, **paired_options)
        arrays = {k: np.asarray(raw[k]) for k in ('F', 'y', 'groups', 'start', 'end')}
        arrays['features'] = np.asarray(raw['features'], dtype=str)
        if context_only:
            prefixes = ('acc_', 'motion_') if modalities == ('acc',) else ('temperature_',)
            columns = [i for i, name in enumerate(arrays['features']) if name.startswith(prefixes)]
            if not columns:
                raise ValueError(f'The feature schema has no {prefixes} columns for the control')
            arrays['F'] = arrays['F'][:, columns]
            arrays['features'] = arrays['features'][columns]
        tmp = target.with_suffix(f'.{os.getpid()}.tmp.npz')
        np.savez_compressed(tmp, **arrays)
        tmp.replace(target)
    with np.load(target, allow_pickle=False) as cached:
        data = {k: cached[k] for k in cached.files}
    data['features'] = data['features'].tolist()
    x, groups, labels = data['F'], data['groups'], data['y']
    if len(np.unique(np.column_stack([groups, data['start']]), axis=0)) != len(x):
        raise ValueError('Participant-window starts must be unique')
    if x.shape != (len(groups), len(data['features'])) or list(np.unique(groups)) != list(SUBJECTS):
        raise ValueError(f'Feature schema or WESAD subject coverage mismatch in {options}')
    if np.isinf(x).any() or len(data['features']) != len(set(data['features'])):
        raise ValueError('Invalid infinite values or duplicate feature names')
    for subject in SUBJECTS:
        if list(np.unique(labels[groups == subject])) != list(range(len(options['classes']))):
            raise ValueError(f'S{subject} lacks at least one class in {options}')
    return data


def complexity(model):
    if hasattr(model, 'coef_') and model.__class__.__name__ == 'LogisticRegression':
        n = int(model.coef_.size + model.intercept_.size)
        return n, f'{n} linear coefficients including intercepts'
    if model.__class__.__name__ == 'SVC':
        return None, f'{len(model.support_)} support vectors'
    if hasattr(model, 'estimators_'):
        return None, f'{sum(t.tree_.node_count for t in model.estimators_)} tree nodes'
    if hasattr(model, 'get_booster'):
        return None, f'{len(model.get_booster().get_dump())} boosted trees'
    return None, f'{model.booster_.num_trees()} boosted trees'


def bundle(spec, data, fitted, params, temp, train_subjects):
    return {'format': 'wesad-classical-v1', 'model': spec['model'], 'fitted': fitted,
            'parameters': params, 'temperature': temp, 'feature_names': data['features'],
            'representation': representation(spec), 'train_subjects': list(train_subjects),
            'classes': list(spec['classes']), 'experiment': spec['name']}


def dump_bundle(path, value):
    tmp = path.with_suffix(f'.{os.getpid()}.tmp')
    joblib.dump(value, tmp, compress=2)
    tmp.replace(path)


def fit_fold(spec, data, outer, directory, signature):
    folder = directory / 'subjects' / f'S{outer}'
    folder.mkdir(parents=True, exist_ok=True)
    record, checkpoint, predictions = folder/'fit.json', folder/'model.joblib', folder/'predictions.csv'
    if record.exists() and checkpoint.exists() and predictions.exists():
        row = json.loads(record.read_text())
        if row['signature'] != signature:
            raise ValueError(f'Fold provenance mismatch at {folder}')
        return row

    with threadpool_limits(1):
        g = data['groups']
        valid_ids = classical.inner_subjects(list(SUBJECTS), outer)
        test = g == outer
        valid = np.isin(g, valid_ids)
        fit = ~(test | valid)
        train_ids = [n for n in SUBJECTS if n != outer]
        inner_ids = [n for n in train_ids if n not in valid_ids]
        assert len(inner_ids) == 12 and len(train_ids) == 14 and set(valid_ids).isdisjoint(inner_ids)
        seed = SEED + outer
        model = spec['model']
        params, temp, score, grid = classical.select_hyperparameters(model, seed, data, fit, valid)
        begin = perf_counter()
        fitted = classical.fit_estimator(model, params, seed, data['F'][~test], data['y'][~test])
        elapsed = perf_counter() - begin
        result = bundle(spec, data, fitted, params, temp, train_ids)
        dump_bundle(checkpoint, result)
        probability = classical.predict_windows(result, data['F'][test], data['features'])
        pred = pd.DataFrame({key: data[key][test] for key in ('start', 'end')})
        pred.insert(0, 'subject', outer)
        pred['y_true'] = data['y'][test]
        for i in range(probability.shape[1]):
            pred[f'p{i}'] = probability[:, i]
        table(predictions, pred)
        median, p95 = classical.benchmark(result, data['F'][test][0])
        n_params, description = complexity(fitted[1])
        row = dict(subject=outer, signature=signature, inner_score=score, inner_subjects=valid_ids,
                   inner_fit_subjects=inner_ids, refit_subjects=train_ids, parameters=params,
                   inner_grid=[dict(parameters=p, score=s) for p, s in zip(classical.GRIDS[model], grid)],
                   temperature=temp, trainable_parameters=n_params, complexity=description,
                   fit_seconds=elapsed, cpu_latency_p50_ms=median, cpu_latency_p95_ms=p95,
                   artifact_bytes=checkpoint.stat().st_size, n_features=len(data['features']),
                   n_inner_fit=int(fit.sum()), n_inner_valid=int(valid.sum()),
                   n_refit=int((~test).sum()), n_test=int(test.sum()))
        write_json(record, row)  # completion marker written after the model and its predictions
        return row


def run(cache, root, specs, jobs=2):
    cache, root = Path(cache), Path(root)
    if not (cache/'manifest.json').is_file():
        raise FileNotFoundError('Prepare all 15 WESAD participants first (--cache must have manifest.json)')
    state = source_state(cache)
    previous, data = None, None
    for spec in specs:
        directory = root / spec['name']
        directory.mkdir(parents=True, exist_ok=True)
        signature = digest(json_text([spec, state]).encode())
        metadata = directory / 'meta.json'
        if metadata.exists() and json.loads(metadata.read_text())['signature'] != signature:
            raise ValueError(f'Existing run has different code/data. Choose a new --output for {directory.name}.')
        if (directory/'folds.csv').exists() and (directory/'predictions.csv').exists() and metadata.exists():
            continue
        write_json(metadata, {**spec, **state, 'signature': signature, 'category': 'traditional',
                              'selection_metric': 'unweighted mean inner-subject macro F1'})
        opts = representation(spec)
        if opts != previous:
            data = feature_data(cache, opts, state)
            previous = opts
        print(f"{spec['name']}: {len(data['y'])} windows, {len(data['features'])} features", flush=True)
        with parallel_config(backend='loky', inner_max_num_threads=1):
            rows = Parallel(n_jobs=jobs)(delayed(fit_fold)(spec, data, i, directory, signature) for i in SUBJECTS)
        frames = [pd.read_csv(directory/'subjects'/f'S{i}'/'predictions.csv') for i in SUBJECTS]
        table(directory/'predictions.csv', pd.concat(frames, ignore_index=True))
        summarized = [{k: json.dumps(v, sort_keys=True) if isinstance(v, (list, dict)) else v
                       for k, v in row.items()} for row in rows]
        table(directory/'folds.csv', pd.DataFrame(summarized))
        print(f"Completed {spec['name']} ({len(rows)} outer subjects)", flush=True)


def export_experiment(cache, experiment_dir, destination):
    """Refit an already selected specification on all 15 subjects; this is a deployment fit."""
    directory = Path(experiment_dir)
    spec = json.loads((directory/'meta.json').read_text())
    cache = Path(cache)
    source = source_state(cache)
    if any(source[k] != spec.get(k) for k in source):
        raise ValueError('Training code, data or dependency versions differ from the evaluated model')
    folds = pd.read_csv(directory/'folds.csv')
    if sorted(folds.subject.tolist()) != list(SUBJECTS):
        raise ValueError('Expected 15 completed outer folds')
    # Fixed tie-break uses grid order. Each participant votes exactly once.
    counts = Counter(folds.parameters)
    params = max(classical.GRIDS[spec['model']], key=lambda p: counts[json.dumps(p, sort_keys=True)])
    temp = float(np.exp(np.mean(np.log(folds.temperature.to_numpy()))))
    data = feature_data(cache, representation(spec), source)
    with threadpool_limits(1):
        fitted = classical.fit_estimator(spec['model'], params, SEED, data['F'], data['y'])
    result = bundle(spec, data, fitted, params, temp, SUBJECTS)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dump_bundle(destination, result)
    return result
