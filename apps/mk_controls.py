import logging
import yaml
import types
import typing as ty
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import Callable, Dict

from egr.log import init_logging

LOG = logging.getLogger('mk_controls')


def load_yaml(path: Path) -> ty.Dict:
    return yaml.safe_load(path.open())


def load_config(run_cfg_path: Path, default_cfg_path: Path) -> ty.Dict:
    cfg = load_yaml(default_cfg_path)
    cfg.update(**load_yaml(run_cfg_path))
    return cfg


def run_random(args):
    from apps.create_random_features import main as random_main

    random_main(args)


def run_label(args):
    from apps.create_label_features import main as label_main

    label_main(args)


def run_pattern(args):
    from apps.create_pattern_features import annotate

    annotate(args)


def run_indices(args):
    from apps.create_index import main as indices_main

    indices_main(args)


def run_default(args):
    from apps.create_default_features import main as default_main

    default_main(args)


def read_config(args: Namespace) -> Namespace:
    run_cfg: Dict = load_config(args.config, args.run_defaults)
    LOG.info('%s', run_cfg)
    num_dim: int = run_cfg['num_dim']
    train_fraction: float = run_cfg['train_fraction']
    root_dir = Path(run_cfg['root_dir'])
    input_dir = root_dir / 'input_data'
    run_cfg.update(
        {'root_dir': root_dir, 'input_dir': input_dir, 'variants': []}
    )
    master = load_yaml(args.pattern_master)
    for v in run_cfg['variant_list']:
        m = master[v.lower()]
        variant_dir = input_dir / v
        samples = list(variant_dir.glob('*.json'))
        num_samples: int = len(samples)

        try:
            num_nodes: int = m['details']['total_size']
            labels_file: Path = variant_dir / 'labels.txt'
            run_cfg['variants'].append(
                Namespace(
                    name=v,
                    variant_dir=variant_dir,
                    indices=Namespace(
                        variant=v,
                        count=num_nodes,
                        folds=types.SimpleNamespace(**run_cfg.get('fold')),
                        train_fraction=train_fraction,
                        output_dir=root_dir / 'indices',
                        labels_file=labels_file,
                    ),
                    default=Namespace(
                        rows=num_nodes,
                        cols=num_dim,
                        output_path=input_dir
                        / f'features/default/features-{num_nodes}.npy',
                    ),
                    label=Namespace(
                        labels_file=labels_file,
                        features_file=variant_dir / 'label_features.npy',
                        num_dim=num_dim,
                    ),
                    random=Namespace(
                        num_dim=num_dim,
                        num_nodes=num_nodes,
                        num_samples=num_samples,
                    ),
                )
            )
            pattern_cfg = run_cfg.get('pattern_names')
            if pattern_cfg:
                p = Namespace(
                    root_path=input_dir,
                    data_dim=num_dim,
                    pattern_names=pattern_cfg,
                    patterns_dir=input_dir / 'pattern_graphs',
                    variants=[v],
                    sample_ids=[p.stem for p in samples],
                )
                run_cfg['variants'][-1].__dict__.update({'pattern': p})

        except KeyError as err:
            LOG.error('%s', err)
            break

    return Namespace(**run_cfg)


def main(args):
    cfg = read_config(args)
    LOG.info('cfg=%s', cfg)
    for v in cfg.variants:
        for op_type in cfg.types:
            LOG.info('v=%s', v)
            op_args = getattr(v, op_type)
            LOG.info(
                'Running type=%s on variant %s, params=%s',
                op_type,
                v.name,
                op_args,
            )
            func: Callable = globals()[f'run_{op_type}']
            func(op_args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--log-level',
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    parser.add_argument(
        '--run-defaults',
        type=Path,
        default='run_configs/run_defaults.yml',
    )
    parser.add_argument(
        '--pattern-master',
        type=Path,
        default='run_configs/pattern_master.yml',
    )
    parser.add_argument('--config', type=Path)
    args = parser.parse_args()

    init_logging(level_name=args.log_level)
    main(args)
