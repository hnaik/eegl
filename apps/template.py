import logging
from argparse import ArgumentParser

from egr.log import init_logging

LOG = logging.getLogger('template')


def main(args):
    LOG.info('Starting app with log-level %s', args.log_level)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '-l',
        '--log-level',
        type=str,
        default='info',
        choices=['debug', 'info', 'warning', 'error', 'critical'],
    )
    args = parser.parse_args()
    init_logging(args.log_level)
    main(args)
