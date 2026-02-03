from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Deque
from collections import deque
import numpy as np


@dataclass
class MaxFlowResult:
    max_flow: float
    residual: np.ndarray


def _bfs_from_source_using_successors(
    residual: np.ndarray,
    succ: List[List[int]],
    s: int,
    t: int,
) -> Optional[List[int]]:
    """
    BFS no residual usando lista de sucessores.
    Retorna caminho [s,...,t] se existir.
    """
    n = residual.shape[0]
    parent = [-1] * n
    parent[s] = s
    q: Deque[int] = deque([s])

    while q:
        u = q.popleft()
        if u == t:
            break
        for v in succ[u]:
            if parent[v] == -1 and residual[u, v] > 0:
                parent[v] = u
                q.append(v)

    if parent[t] == -1:
        return None

    # reconstrói t -> s
    path_rev = [t]
    cur = t
    while cur != s:
        cur = parent[cur]
        path_rev.append(cur)
    return list(reversed(path_rev))


def max_flow_edmonds_karp_succ_bfs(
    capacity: np.ndarray,
    succ: List[List[int]],
    s: int = 0,
    t: Optional[int] = None,
) -> MaxFlowResult:
    """
    Simulação 2:
    - entrada: matriz de capacidade e lista de sucessores
    - algoritmo de caminhos de fluxo + BFS alvo (Edmonds–Karp)
    """
    n = capacity.shape[0]
    if t is None:
        t = n - 1

    residual = capacity.astype(float).copy()
    flow = 0.0

    while True:
        path = _bfs_from_source_using_successors(residual, succ, s, t)
        if path is None:
            break

        bottleneck = float("inf")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            bottleneck = min(bottleneck, residual[u, v])

        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck

        flow += bottleneck

    return MaxFlowResult(max_flow=flow, residual=residual)
