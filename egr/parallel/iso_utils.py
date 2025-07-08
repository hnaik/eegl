import logging
import typing as ty

import networkx as nx
from joblib import Parallel, delayed

import egr.graph_utils as gu
import egr.glasgow_subgraph_solver as gss


LOG = logging.getLogger(__name__)


def compute_isomorphism(
    G: nx.Graph,
    H: nx.Graph,
    indices: ty.List[int],
    n_jobs: ty.Optional[int] = None,
) -> ty.List[ty.Tuple[int, bool]]:
    n_jobs = n_jobs or -2
    runner = Parallel(n_jobs=n_jobs)
    params = [(G, H, node_id, index) for index, node_id in enumerate(indices)]
    result = runner(delayed(is_isomorphic)(*args) for args in params)
    # return [iso for _, iso in result]
    return list(result)


def is_isomorphic(G, H, node_id, index):
    hops = nx.eccentricity(H, H.graph['__root__'])
    H.nodes[H.graph['__root__']]['__root__'] = True
    G_ = gu.get_neighborhood_subgraph(G, node_id, hops)
    G_.nodes[node_id]['__root__'] = True
    G_.graph['__root__'] = node_id
    return index, gss.subgraph_is_isomorphic(G_, H)
