import logging
import time
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

from apps.gnn_explainer.gaston_sgm.gaston_sgm import run
from egr.log import init_logging

LOG = logging.getLogger(__name__)


def main(args):
    begin = time.time()
    run(args)
    seconds = time.time() - begin
    minutes, seconds = seconds // 60, seconds % 60
    LOG.info('Duration %d:%d', minutes, seconds)


if __name__ == '__main__':
    init_logging(level_name='debug')
    parser = ArgumentParser()
    parser.add_argument('--data-dim', type=int, default=10)
    parser.add_argument('--freq', type=float, default=0.5)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--nproc', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument(
        '--run-id', type=str, default=datetime.now().strftime('%y%m%d%H%M%S%f')
    )
    parser.add_argument('--input-file', type=Path, required=True)
    parser.add_argument('--intermediate', action='store_true', default=False)

    try:
        main(parser.parse_args())
    except KeyboardInterrupt as err:
        LOG.warning('Keyboard interrupt, exiting')
