import queue
import typing as ty

from dataclasses import dataclass, field


@dataclass(order=True)
class GraphItem:
    score: float
    num_edges: int
    G: ty.Any = field(compare=False)


class GraphPriorityQueue:
    def __init__(self, graphs, key: str):
        self.key = key
        self._pq = queue.PriorityQueue()

        max_edges = max([g.number_of_edges() for g in graphs])

        for G in graphs:
            pi = GraphItem(
                1.0 - G.graph[key],  # convert to min queue
                max_edges - G.number_of_edges(),  # convert to min queue
                G,
            )
            self._pq.put(pi)

    def pop(self):
        return self._pq.get()

    @property
    def empty(self) -> int:
        return self._pq.empty()


def make_score_heap(label_graphs: ty.Dict, key: str) -> ty.Dict:
    score_heaps = {}
    for label, patterns in label_graphs.items():
        graphs = [p['graph'] for p in patterns]
        score_heaps.update({label: GraphPriorityQueue(graphs, key)})
    return score_heaps
