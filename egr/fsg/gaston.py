# from logging import root
# from sys import maxsize
# from tkinter import N
import logging
import os
import subprocess
import sys
import time

from datetime import datetime
from typing import Callable, List, Optional
from xmlrpc.client import boolean

import networkx as nx

import egr.util as eu
import egr.fsg.gaston_io as efgi

LOG = logging.getLogger(__name__)

gastonLabelAttr = '__gaston_label__'
rootAttr = '__root__'
GASTON_BIN = os.getenv('GASTON_BIN_PATH', eu.default_ext_bin('gaston'))


def mineFreqRootedSubgraphs(
    graphs: List[nx.Graph],
    roots: List[int],
    label: int,
    # freq: float = 0.8,
    args,
    minNodes: Optional[int] = None,
    maxNodes: Optional[int] = None,
    onlyMaximal: boolean = True,
):
    """Mines largest rooted subgraphs which occur frequently in the given rooted graphs.
        Removes the smaller subgraphs which could be part of other larger ones.

    Args:
        graphs: List of graphs which are to be mined for frequent subgraphs.
        roots:  List of root nodes of Graphs in the same order as graphs.
        freq:   The minimum frequency with which the subgraph should occur
                to be called frequent.
        minNodes:   Minimum number of nodes that must be present in the frequent subgraphs.
        maxNodes:   Maximum number of nodes that can be present in the frequent subgraphs.
        onlyMaximal:If true, subgraphs which in turn are subgraphs of some other larger subgraph
                    are filtered out.

    Returns:
        list of frequent sub-graphs. The root will be part of all of these.
        list of root node ids corresponding to each of the frequent subgraph.

    """

    # print('Annotating root/non-root label on nodes')
    # for i, Gi in enumerate(graphs):
    #     makeRootNode(G=Gi, root=roots[i])

    __label_fetcher = lambda G, n: int(__isRootNode(G=G, n=n))

    LOG.debug('Mining frequent subgraphs for label %s', label)
    freqSubGraphs = mineFreqSubgraphs(
        graphs=graphs, args=args, label=label, labelFetcher=__label_fetcher
    )
    count_orig = len(freqSubGraphs)
    LOG.debug('Found %d frequent subraphs', len(freqSubGraphs))

    LOG.debug('Removing graphs which miss the root node')
    freqSubGraphs = [G for G in freqSubGraphs if __isRootedGraph(G)]
    LOG.debug('%d subgraphs left', len(freqSubGraphs))
    count_root_filter = len(freqSubGraphs)

    if maxNodes is not None:
        LOG.debug('Removing graphs with more than 5 nodes')
        freqSubGraphs = [G for G in freqSubGraphs if len(G.nodes) <= maxNodes]
        LOG.debug('%d subgraphs left', len(freqSubGraphs))
    count_max_node_filter = len(freqSubGraphs)

    if onlyMaximal:
        num_subgraphs: int = len(freqSubGraphs)
        LOG.info(
            'Filtering out the smaller graphs based on Topological Order, total:%d',
            num_subgraphs,
        )
        begin_filtering = time.time()
        freqSubGraphs = filterSmallerGraphs(freqSubGraphs)
        filtering_duration = time.time() - begin_filtering
        LOG.info(
            '%d/%d subgraphs left, filtering duration: %s',
            num_subgraphs,
            len(freqSubGraphs),
            filtering_duration,
        )
    count_final = len(freqSubGraphs)

    LOG.debug(
        'Label:%s, original:%3d, after root filter:%3d, after max-node filter:%3d, final:%2d',
        label,
        count_orig,
        count_root_filter,
        count_max_node_filter,
        count_final,
    )

    roots = [__getRootNode(G) for G in freqSubGraphs]

    return freqSubGraphs, roots


# %%
def filterSmallerGraphs(graphs: List[nx.Graph]):
    LOG.debug('Computing topological order on frequent subgraphs')
    seq, dag = getTopoOrder(graphs=graphs)

    largeGraphs: List[nx.Graph] = []

    for i, g in enumerate(seq):
        if dag.in_degree(g) == 0:
            largeGraphs.append(graphs[g])
    return largeGraphs


# %%
def mineFreqSubgraphs(
    # graphs: List[nx.Graph], freq=0.8, labelFetcher: Callable = None
    graphs: List[nx.Graph],
    args,
    label: str,
    labelFetcher: Optional[Callable],
) -> List[nx.Graph]:
    """Mines sub-graphs which occur frequently in the given graphs.
        Labels are matched, if provided.

    Args:
        graphs: The graphs which are to be mined for frequent sub-graphs.
        freq:   The minimum frequency with which the sub-graph should occur
                to be called frequent.
        labelFetcher: A callable function/object which fetches the labels of nodes.
                It should take a graph G and a node N (int) as args
                and return the label(int) of the node N.
                Required only if the graph is labelled.

    Returns:
        list of frequent sub-graphs.
    """

    suffix = f'{args.run_id}_{args.input_tag}_{label}'
    inFilename = args.tmp_dir / f'gaston_in_{suffix}.txt'
    outFilename = args.tmp_dir / f'gaston_out_{suffix}.txt'
    writeGastonGraphs(
        graphs=graphs, labelFetcher=labelFetcher, filename=inFilename
    )
    origNumGraphs = len(graphs)
    freqGaston = max(int(args.freq * origNumGraphs), 1)
    m: str = str(args.largest_frequent_pattern)
    cmd = [GASTON_BIN, '-m', m, str(freqGaston), inFilename, outFilename]
    cmd = [str(c) for c in cmd]
    LOG.debug('%s', ' '.join(cmd))
    try:
        begin = time.time()
        ret = subprocess.run(cmd, capture_output=True, check=True)
        LOG.info(
            'Gaston: label: %2s, freq:%3d, largest-freq-pattern:%s, dur: %.4fs',
            label,
            freqGaston,
            m,
            time.time() - begin,
        )
        ret.check_returncode()
        freq_sub_graphs = efgi.read_gaston_output(outFilename)
        return freq_sub_graphs
    except subprocess.CalledProcessError as err:
        LOG.error('%s failed to run Gaston', err)
        sys.exit(1)


# %%
# def getRootedSubgraphIsomorphismCount(
#     G: nx.Graph, rG: int, sG: nx.Graph, rS: int, boolean=False, draw=False
# ) -> int:
#     """Finds the number of rooted subgraph isomorphisms of the smaller graph
#         into the larger graph.

#     :Params
#         G:  The larger graph.
#         rG: Root node of the larger graph(G).
#         sG: The smaller graph whose rooted subgraph isomorphisms are to be counted in G.
#         rS: Root node of the smaller graph(sG)
#         boolean: Specifies whether to return boolean 1/0 or actual count
#         draw:   Whether to draw the isomorphic graphs or not

#     Returns:
#         The number of rooted subgraph isomorphisms of sG into G rooted at rS and rG respectively.
#         If boolean is True, then returns whether there exists a rooted subgraph isomorphisms
#         of sG into G rooted at rS and rG respectively
#     """
#     makeRootNode(G=G, root=rG)
#     rootMatcher = lambda n1Attrs, n2Attrs: (rootAttr in n1Attrs) == (
#         rootAttr in n2Attrs
#     )
#     count = 0
#     for isoMap in nx.algorithms.isomorphism.ISMAGS(
#         G, sG, node_match=rootMatcher
#     ).find_isomorphisms():
#         count += 1
#         if draw:
#             drawIsoGraphs(G, rG, sG, rS, isoMap)
#         if boolean:
#             break
#     __removeRootNode(G=G)
#     return count


# %%
# def getRootedSubgraphIsomorphisms(G: nx.Graph, rG: int, sG: nx.Graph, rS: int):
#     makeRootNode(G=G, root=rG)
#     rootMatcher = lambda n1Attrs, n2Attrs: (rootAttr in n1Attrs) == (
#         rootAttr in n2Attrs
#     )
#     for isoMap in nx.algorithms.isomorphism.ISMAGS(
#         G, sG, node_match=rootMatcher
#     ).find_isomorphisms():
#         yield (isoMap)
#     __removeRootNode(G=G)


# %%
def neighborhoodGraph(G, root, dist=1):
    nodes = {
        k
        for k, v in nx.single_source_shortest_path_length(
            G, root, cutoff=dist
        ).items()
    }
    nG = nx.induced_subgraph(G, nodes)
    return nG


# %%
# import matplotlib.pyplot as plt


# def drawIsoGraphs(G, rG, sG, rs, isoMap):
#     nodesG = {nG for nG in isoMap.keys()}
#     partG = nx.induced_subgraph(G, nodesG)
#     LOG.info('######################################################')
#     LOG.info('Orig Graph Part:')
#     nx.draw(
#         partG,
#         node_color=__getColors(partG),
#         pos=nx.spring_layout(partG),
#         with_labels=True,
#     )
#     plt.show()

#     LOG.info('Pattern:')
#     nx.draw(
#         sG,
#         node_color=__getColors(sG),
#         pos=nx.spring_layout(sG),
#         with_labels=True,
#     )
#     plt.show()


# %%
# from typing import Callable, List


# def drawGraphs(graphs: List[nx.Graph]):
#     numGraphs = len(graphs)
#     nCols = min(3, numGraphs)
#     import math

#     nRows = int(math.ceil(numGraphs / nCols))
#     fig, ax = plt.subplots(
#         nRows, nCols, sharex=True, sharey=True, figsize=(5 * nCols, 5 * nRows)
#     )
#     ax = ax.flatten()
#     for i, G in enumerate(graphs):
#         nx.draw(
#             G, node_color=__getColors(G), ax=ax[i], pos=nx.spring_layout(G)
#         )
#         ax[i].set_axis_off()
#         ax[i].set_title(f'{i+1}')
#     plt.show()


# def hasIsomorphicSubgraph(g:nx.Graph, gSub:nx.Graph):
#     rootMatcher = lambda n1Attrs, n2Attrs: (rootAttr in n1Attrs) == (rootAttr in n2Attrs)
#     if nx.algorithms.isomorphism.ISMAGS(g, gSub, node_match=rootMatcher).subgraph_is_isomorphic():
#         return True
#     else:
#         return False

# from networkx.algorithms.isomorphism import GraphMatcher
# def hasMonomorphicSubgraph(g:nx.Graph, gSub:nx.Graph):
#     rootMatcher = lambda n1Attrs, n2Attrs: (rootAttr in n1Attrs) == (rootAttr in n2Attrs)
#     for match in GraphMatcher(g, gSub, node_match=rootMatcher).subgraph_monomorphisms_iter():


def getTopoOrder(graphs):
    dag = nx.DiGraph()
    for i, g in enumerate(graphs):
        dag.add_node(i)

    rootMatcher = lambda n1Attrs, n2Attrs: (rootAttr in n1Attrs) == (
        rootAttr in n2Attrs
    )

    from itertools import combinations
    import egr.glasgow_subgraph_solver as gss

    graphPairs = list(combinations(range(len(graphs)), 2))
    for g1, g2 in graphPairs:
        G1 = graphs[g1]
        G2 = graphs[g2]
        # if nx.algorithms.isomorphism.ISMAGS(G1, G2, node_match=rootMatcher).subgraph_is_isomorphic():
        # if GraphMatcher(
        #     G1, G2, node_match=rootMatcher
        # ).subgraph_is_monomorphic():
        if gss.subgraph_is_isomorphic(G1, G2):
            dag.add_edge(g1, g2)
        # if nx.algorithms.isomorphism.ISMAGS(G2, G1, node_match=rootMatcher).subgraph_is_monomorphic():
        # if GraphMatcher(
        #     G2, G1, node_match=rootMatcher
        # ).subgraph_is_monomorphic():
        if gss.subgraph_is_isomorphic(G2, G2):
            dag.add_edge(g2, g1)

    seq = list(nx.topological_sort(dag))
    return seq, dag


def writeGastonGraphs(
    graphs: List[nx.Graph], filename: str = None, labelFetcher: Callable = None
) -> str:
    """Writes the given graphs into a file which can be read by gaston.

    Args:
        graphs:     List of graphs to be written to file.
        filename:   Output file name.
        labelFetcher: A callable function/object which fetches the labels of nodes.
                    It should take a graph G and a node N (int) as args
                    and return the label(int) of the node N.
                    Required only if the graph is labelled.

    Returns:
        Output file name
    """
    for i, G in enumerate(graphs):
        if i == 0:
            mode = 'w'
            filename = writeGastonGraph(
                G=G,
                filename=filename,
                labelFetcher=labelFetcher,
                id=i,
                mode=mode,
            )
        else:
            mode = 'a'
            writeGastonGraph(
                G,
                filename=filename,
                labelFetcher=labelFetcher,
                id=i,
                mode=mode,
            )


def writeGastonGraph(
    G: nx.Graph,
    filename: str = None,
    labelFetcher: Callable = None,
    id: int = 0,
    mode='a',
) -> str:
    """Writes the given graph to a file which can be read by gaston.

    Args:
        G:          The graph to be written to file.
        filename:   Output file name.
        labelFetcher: A callable function/object which fetches the labels of nodes.
                    It should take a graph G and a node N (int) as args,
                    and return the label(int) of the node N.
                    Required only if the graph is labelled.
        id:         Gaston serial number of the graph.
        mode:       'a' to append to file or 'w' to create/overwrite the file.

    Returns:
        Output file name
    """

    if filename is None:
        time = datetime.now().strftime('%Y%m%d%H%M%S%f')
        filename = f'/tmp/gastonGraphs.{time}.txt'

    with open(filename, mode=mode) as file:
        print(f't # {id}', file=file)
        for n in G.nodes:
            label = labelFetcher(G, n)
            print(f'v {n} {label}', file=file)
        for e in G.edges:
            label = 0  # TODO: add support for edge labels
            print(f'e {e[0]} {e[1]} {label}', file=file)

    return filename


def readGastonGraphs(filename: str) -> List[nx.Graph]:
    """Parses gaston formatted file and creates Graphs.

    Args:
        filename: The name of the file to be read.

    Returns:
        List of read graphs from the file.
    """
    lines = []
    with open(filename) as f:
        lines = f.readlines()

    count = 0
    G: nx.Graph = None
    graphs = []
    for line in lines:
        count += 1
        line = line.split('\n')[0]
        line = line.split(' ')
        if line[0] == '#':
            continue
        if line[0] == 't':
            if G is not None:
                graphs.append(G)
            G = nx.Graph()
        if line[0] == 'v':
            nodeId = int(line[1])
            nodeLabel = int(line[2])
            G.add_node(nodeId)
            if (
                nodeLabel == 1
            ):  # TODO: This doesn't belong in this API. Move to the client.
                assert not __isRootedGraph(
                    G=G
                ), f'Root node already found: {__getRootNode(G=G)}, {nodeId} cant be root node.'
                makeRootNode(G, nodeId)

        if line[0] == 'e':
            node1 = int(line[1])
            node2 = int(line[2])
            G.add_edge(node1, node2)
    graphs.append(G)  # the last graph
    return graphs


def makeRootNode(G: nx.Graph, root: int):
    """Makes and marks the given Graph as 'rooted', and the root node as 'root'

    Args:
        G:      Graph which is to be made 'rooted'
        root:   Node which is to be made the 'root'

    Returns:
        Nothing

    Complexity:
        O(1)
    """
    if __isRootedGraph(G=G):
        __removeRootNode(G=G)

    G.graph[rootAttr] = root
    G.nodes[root][rootAttr] = 1


def __removeRootNode(G: nx.Graph):
    """Makes the graph 'G' un-rooted or ordinary.

    Complexity:
        O(1)
    """
    root = __getRootNode(G=G)
    if root == -1:
        return
    G.graph.pop(rootAttr, None)
    G.nodes[root].pop(rootAttr, None)


def __isRootedGraph(G: nx.Graph) -> bool:
    """Tells if the graph 'G' is rooted or not

    Complexity:
        O(1)
    """
    return rootAttr in G.graph


def __isRootNode(G: nx.Graph, n: int) -> bool:
    """Tells if the node with id 'n' is the root of the graph 'G'

    Complexity:
        O(1)
    """
    return rootAttr in G.nodes[n]


def __getRootNode(G: nx.Graph) -> int:
    """Returns the root node id of the given Graph

    Complexity:
        O(1)
    """
    if rootAttr not in G.graph:
        return -1
    return G.graph[rootAttr]


def __getColors(G: nx.Graph, root: int = None):
    """Returns the colors for the nodes of the graph G.
    Root node is assigned color green, all others red.

    Complexity:
        O(|V|)
    """
    colors = []
    if root is None:
        root = __getRootNode(G=G)

    for n in G.nodes:
        if n == root:
            colors.append('tab:green')
        else:
            colors.append('tab:red')
    return colors
