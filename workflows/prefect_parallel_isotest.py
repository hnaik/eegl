import logging
import time
from argparse import ArgumentParser
from dataclasses import dataclass
from typing import List, Optional

import networkx as nx
import numpy as np
from prefect import flow, task
from prefect.futures import PrefectFuture
from prefect_dask import DaskTaskRunner

from egr.log import init_logging

LOG = logging.getLogger('isotest')
logging.getLogger('prefect.flow_runs').setLevel(logging.WARNING)
logging.getLogger('prefect.task_runs').setLevel(logging.WARNING)
logging.getLogger('distributed').setLevel(logging.WARNING)


def task_complete(task, task_run, state):
    LOG.debug(
        '[C:%s] result=%s, dur=%s',
        task_run.name,
        state.result(),
        task_run.total_run_time,
    )


def task_failed(task, task_run, state):
    LOG.debug(
        '[F:%s] result=%s, dur=%s',
        task_run.name,
        state.result(),
        task_run.total_run_time,
    )


@dataclass
class Payload:
    G: nx.Graph
    H: nx.Graph
    i: int
    j: int
    X: np.ndarray
    dur: Optional[float] = None

    def __str__(self):
        return f'({self.i:04},{self.j:04}) -> {self.X[self.i, self.j]}'


@task(on_completion=[task_complete], on_failure=[task_failed])
def is_iso(payload: Payload) -> True:
    payload.dur = float(np.random.choice(np.arange(1, 20, step=1)))
    time.sleep(payload.dur)
    x = 1 if np.random.choice([True, False]) else 0
    payload.X[payload.i, payload.j] = x
    return payload


@flow(name='make_iso_matrix')
def make_iso_matrix(G, subgraphs):
    X = np.zeros([G.number_of_nodes(), len(subgraphs)], dtype=int)
    for v in G:
        for i, H in enumerate(subgraphs):
            payload = Payload(G=G, H=H, i=v, j=i, X=X)
            is_iso.submit(payload)
    return X


cycle_sizes = np.arange(3, 15, step=1)


def main(args):
    LOG.info('Starting app with log-level %s', args.log_level)
    G: nx.Graph = nx.erdos_renyi_graph(5, 0.5)
    subgraphs: List[nx.Graph] = [
        nx.cycle_graph(np.random.choice(cycle_sizes)) for _ in range(5)
    ]
    begin = time.time()
    X = make_iso_matrix(G, subgraphs)
    LOG.info('Elapsed:%s\n%s', time.time() - begin, X)


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
