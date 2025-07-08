__all__ = ['init_logging']

import logging
import typing as ty
from pathlib import Path
from typing import Optional

import coloredlogs


VERBOSE = 5
LOG_FMT = '%(asctime)s %(levelname)s %(name)s:%(lineno)s %(message)s'


def init_logging(
    level_name: str = 'debug',
    fmt: Optional[str] = None,
    handlers: ty.Optional[ty.List[logging.Handler]] = None,
):
    logging.addLevelName(logging.FATAL, 'F')
    logging.addLevelName(logging.ERROR, 'E')
    logging.addLevelName(logging.WARNING, 'W')
    logging.addLevelName(logging.INFO, 'I')
    logging.addLevelName(logging.DEBUG, 'D')
    logging.addLevelName(VERBOSE, 'V')

    coloredlogs.DEFAULT_LEVEL_STYLES = {
        'C': {'bold': True, 'color': 'red'},
        'D': {'faint': True},
        'E': {'color': 'red'},
        'I': {},
        'N': {'color': 'magenta'},
        'P': {'color': 'green', 'faint': True},
        'S': {'bold': True, 'color': 'green'},
        'V': {'color': 'blue'},
        'W': {'color': 'yellow'},
    }

    logging.captureWarnings(capture=True)

    logging.getLogger('fsspec.local').setLevel(logging.INFO)
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(logging.ERROR)
    logging.getLogger('h5py').setLevel(logging.ERROR)
    logging.getLogger('py.warnings').setLevel(logging.ERROR)

    fmt = fmt or LOG_FMT

    formatter = logging.Formatter(fmt=fmt)
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(getattr(logging, level_name.upper()))
    stream_handler.setFormatter(formatter)
    logging.getLogger().addHandler(stream_handler)

    handlers = handlers or []
    for handler, level in handlers:
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logging.getLogger().addHandler(handler)

    coloredlogs.install(level=level_name, fmt=fmt, handler=stream_handler)


def add_log_argument(parser, default_level='info'):
    choices = ['debug', 'info', 'warning', 'error', 'critical']
    args = dict(type=str, default=default_level, choices=choices)
    parser.add_argument('--log-level', **args)

    default_dir = Path().home().expanduser() / 'logs'
    parser.add_argument('--log-dir', type=Path, default=default_dir)
    parser.add_argument('--log-file-level', default='debug', choices=choices)
