import logging
import os
import subprocess as sp
import tempfile
import typing as ty
import uuid
from pathlib import Path

import networkx as nx

import egr.util as eu

LOG = logging.getLogger(__name__)


SOLVER_PATHS = dict(
    default=eu.EXT_BIN_DIR / 'glasgow_subgraph_solver',
    common=eu.EXT_BIN_DIR / 'glasgow_common_subgraph_solver',
)

ENV = os.environ.copy()
ENV.update(LD_LIBRARY_PATH=str(eu.DEPS_LIB_DIR))
# ENV = {'LD_LIBRARY_PATH': str(eu.DEPS_LIB_DIR), **os.environ}

TMPFS: Path = Path('/dev')


def subgraph_is_isomorphic(G: nx.Graph, H: nx.Graph, mode: str = 'default'):
    with tempfile.TemporaryDirectory() as dirpath:
        target_file = TMPFS / Path(dirpath) / 'target.csv'
        pattern_file = TMPFS / Path(dirpath) / 'pattern.csv'

        G = save_csv(G, target_file)
        H = save_csv(H, pattern_file)

        return gss_is_isomorphic(G, H, mode)


def gss_is_isomorphic(G: nx.Graph, H: nx.Graph, mode: str = 'default'):
    bin_path = os.getenv('EEGL_SOLVER_PATH', str(SOLVER_PATHS[mode]))
    cmd = [bin_path]

    if mode == 'default':
        cmd.append('--parallel')
    cmd.extend(
        ['--format', 'csv', str(H.graph['csv_path']), str(G.graph['csv_path'])]
    )

    try:
        ret = sp.run(cmd, capture_output=True, check=True, env=ENV)
        ret.check_returncode()
        out = ret.stdout.decode()
        return 'status = true' in out
    except sp.CalledProcessError as err:
        egr_debug: Path = Path('/tmp') / f'egr_debug/{uuid.uuid4()}'
        egr_debug.mkdir(parents=True, exist_ok=True)
        debug_target_file = egr_debug / 'target.csv'
        debug_pattern_file = egr_debug / 'pattern.csv'
        LOG.error('EX:%s', err)
        LOG.info(
            'Saving debug files: %s, %s',
            debug_target_file,
            debug_pattern_file,
        )
        LOG.info('LD_LIBRARY_PATH:%s', str(eu.DEPS_LIB_DIR))
        save_csv(G, debug_target_file)
        save_csv(H, debug_pattern_file)
        raise

    # lines = ret.stdout.decode().split('\n')
    # for line in lines:
    #     if line.isspace() or line == '':
    #         continue
    #     key, value = get_key_value(line, '=')
    #     if key != 'status':
    #         continue
    #     return value.lower() == 'true'
    # raise RuntimeError(
    #     f'Did not detect result in output {ret.stdout.decode()}'
    # )


def solve(pattern_path: Path, target_path: Path, mode: str = 'default'):
    bin_path = os.getenv('EEGL_SOLVER_PATH', str(SOLVER_PATHS[mode]))
    cmd = [bin_path]

    if mode == 'default':
        cmd.append('--parallel')
    cmd.extend(['--format', 'csv', str(pattern_path), str(target_path)])
    ret = sp.run(cmd, capture_output=True, check=True, env=ENV)
    ret.check_returncode()
    output = ret.stdout.decode()
    return 'status = true' in output
    # lines = ret.stdout.decode().split('\n')
    # for line in lines:
    #     if line.isspace() or line == '':
    #         continue
    #     key, value = get_key_value(line, '=')
    #     if key != 'status':
    #         continue
    #     return value.lower() == 'true'
    # raise RuntimeError(
    #     f'Did not detect result in output {ret.stdout.decode()}'
    # )


def save_csv(G: nx.Graph, path: Path) -> nx.Graph:
    s = ''
    for u, v in G.edges():
        s += f'{u},{v}\n'
    for u in G.nodes():
        label = 1 if '__root__' in G.nodes[u] and G.nodes[u]['__root__'] else 0
        s += f'{u},,{label}\n'
    path = path if isinstance(path, Path) else Path(path)
    path.write_text(s)
    G.graph['csv_path'] = path
    return G


def save_lad(G: nx.Graph, path: Path) -> Path:
    n: int = G.number_of_nodes()
    s: str = f'{n}\n'
    for u in range(n):
        neighbors = [str(v) for v in nx.neighbors(G, u)]
        s += '{num} {neighbors}\n'.format(
            num=len(neighbors), neighbors=' '.join(neighbors)
        )
    path = path if isinstance(path, Path) else Path(path)
    path.write_text(s)
    return path


def get_key_value(line: str, delim: str) -> ty.Tuple[str, str]:
    k, v = line.split(delim)
    return cleanup_str(k), cleanup_str(v)


def cleanup_str(s: str) -> str:
    return s.strip().rstrip().lower()
