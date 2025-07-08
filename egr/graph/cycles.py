import typing as ty

import networkx as nx


class DsBase:
    def __init__(self, initial: ty.Optional[ty.List] = None):
        self._q = initial or []

    @property
    def values(self) -> ty.List:
        """Return copy, not reference"""
        return [u for u in self._q]

    @property
    def size(self) -> int:
        return len(self._q)

    @property
    def empty(self) -> bool:
        return len(self._q) == 0

    def push(self, e):
        self._q.append(e)

    def pop(self) -> int:
        if len(self._q) == 0:
            raise RuntimeError('empty')
        return self._pop()

    def clear(self):
        self._q = []

    def has(self, e: int) -> bool:
        return e in self._q

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}{self._q}>'

    def __str__(self) -> str:
        return f'{self._q}'


class Queue(DsBase):
    def __init__(self, initial: ty.Optional[ty.List] = None):
        super().__init__(initial)

    def _pop(self):
        e = self._q[0]
        del self._q[0]
        return e


class Stack(DsBase):
    def __init__(self, initial: ty.Optional[ty.List] = None):
        super().__init__(initial)

    @property
    def top(self) -> int:
        return self._q[-1]

    def _pop(self):
        e = self._q[-1]
        self._q.pop()
        return e


def find_bounded_cycles(
    G: nx.Graph, r: int, length: int
) -> ty.List[ty.List[int]]:
    H = G.copy()
    stack = Stack([r])
    cache = Stack()
    visited = set([])
    all_cycles = []
    popped = []

    while not stack.empty:
        if not cache.has(stack.top):
            cache.push(stack.top)
        visited.add(stack.top)
        N = sorted(
            [
                n
                for n in nx.neighbors(H, stack.top)
                if n == r or (n not in visited)
            ],
            reverse=True,
        )
        if cache.size == length:
            if r in N:
                all_cycles.append(cache.values)
            cache.pop()
            popped.append(stack.pop())
            continue
        if N == [] or N == [r]:
            stack.pop()
            popped.append(cache.pop())
            continue
        for n in N:
            if not cache.has(n):
                for u in popped[:-1]:
                    visited -= {u}
                popped = []
                stack.push(n)
    return all_cycles
