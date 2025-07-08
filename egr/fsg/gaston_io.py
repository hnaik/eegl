import logging
import typing as ty
from pathlib import Path

import networkx as nx

import egr.log as log

LOG = logging.getLogger(__name__)


def read_gaston_output(path: Path) -> ty.List[nx.Graph]:
    LOG.debug('Reading %s', path)

    graphs: ty.List[nx.Graph] = []
    G: ty.Optional[nx.Graph] = None
    for i, line in enumerate(path.open()):
        bits = line.split()
        if bits[0] == '#':
            if G is not None:
                graphs.append(G.copy())
        elif bits[0] == 't':
            G = nx.Graph()
        elif bits[0] == 'v':
            v = int(bits[1])
            label = int(bits[2])
            G.add_node(v, label=label)
            if label == 1:
                G.graph['__root__'] = v
                G.nodes[v]['__root__'] = 1
        elif bits[0] == 'e':
            G.add_edge(int(bits[1]), int(bits[2]), label=int(bits[3]))
    return graphs


if __name__ == '__main__':
    import sys

    log.init_logging(level_name='debug')

    g_list = read_gaston_output(Path(sys.argv[1]))
    LOG.info('Loaded %d graphs', len(g_list))
