"""Train or resume the classical WESAD study in the EDA module."""
import argparse
import sys
from pathlib import Path

MODULE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE/'src'))
from wearable_eda.training.classical_run import experiments, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, default=Path('data/eda/processed/wesad_v1'))
    parser.add_argument('--output', type=Path, default=Path('artifacts/eda/runs/wesad_loso_v1/classical'))
    parser.add_argument('--jobs', type=int, default=2, help='Number of independent outer folds to fit at once.')
    parser.add_argument('--experiment', action='append', help='Exact experiment name; repeat to select more.')
    parser.add_argument('--role', action='append', choices=['candidate', 'ablation', 'negative_control', 'secondary'])
    parser.add_argument('--list', action='store_true', help='Show the selected experiments and exit.')
    args = parser.parse_args()
    specs = experiments()
    if args.experiment:
        unknown = set(args.experiment) - {spec['name'] for spec in specs}
        if unknown:
            parser.error(f'Unknown experiments: {sorted(unknown)}')
        specs = [spec for spec in specs if spec['name'] in args.experiment]
    if args.role:
        specs = [spec for spec in specs if spec['role'] in args.role]
    if not specs or args.jobs < 1:
        parser.error('Select at least one experiment and a positive number of workers.')
    if args.list:
        print('\n'.join(spec['name'] for spec in specs))
    else:
        run(args.cache, args.output, specs, args.jobs)


if __name__ == '__main__':
    main()
