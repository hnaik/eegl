import logging
from argparse import ArgumentParser
from collections import Counter

import pandas as pd

import egr
import egr.data.pyg_data_helpers as pdh

LOG = logging.getLogger('apps.validate_data')


def get_counts(data, name):
    indices = getattr(data, f'{name}_index')
    y = data.y[indices].tolist()
    counter = Counter(y)
    return {str(k): counter[k] for k in sorted(counter.keys())}


def main(args):
    cfg = {
        'src': args.data_src,
        'data_class': args.data_class,
        'params': {
            'name': args.name,
            'force_reload': args.force_reload,
            'log': args.log,
        },
    }

    dataset = pdh.load_dataset(cfg)
    LOG.info('loaded data from cfg: %s', cfg)
    table, index = [], []
    for i in range(dataset.train_mask.shape[1]):
        data = pdh.process_data(cfg, fold=i, kw={}, data=dataset[0])
        try:
            table.append(get_counts(data, args.partition))
            index.append(i + 1)
        except AttributeError as err:
            LOG.error('%s %s', err, dir(data))
    df = pd.DataFrame(data=table, index=index)
    df['total'] = df.sum(axis=1)
    LOG.info('df:\n%s', df)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data-class', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    parser.add_argument(
        '--data-src', type=str, default='egr', choices=['egr', 'pyg']
    )
    parser.add_argument('--force-reload', action='store_true', default=False)
    parser.add_argument('--log', action='store_true', default=False)
    parser.add_argument('--partition', type=str, required=True)

    egr.init_logging(level_name='info')

    main(parser.parse_args())
