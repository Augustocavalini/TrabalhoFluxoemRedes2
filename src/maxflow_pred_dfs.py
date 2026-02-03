from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class MaxFlowResult:
    max_flow: float
    residual: np.ndarray  # residual capacities


def _dfs_target_from_sink_using_predecessors(
    residual: np.ndarray,
    pred: List[List[int]],
    s: int,
    t: int,
) -> Optional[List[int]]:
    """
    Busca em profundidade "alvo" usando lista de antecessores.

    Implementação natural com antecessores:
      - começamos do destino t e tentamos chegar na origem s
      - seguindo arestas residuais positivas u->v olhando antecessores u de v

    Retorna o caminho como lista de nós [s, ..., t] se existir.
    """
    n = residual.shape[0]
    visited = [False] * n
    parent = [-1] * n  # parent[v] = próximo nó em direção ao destino (porque buscamos ao contrário)
    stack = [t]
    visited[t] = True

    while stack:
        v = stack.pop()
        if v == s:
            break
        # explora antecessores u tal que residual[u][v] > 0
        for u in pred[v]:
            if not visited[u] and residual[u, v] > 0:
                visited[u] = True
                parent[u] = v
                stack.append(u)

    if not visited[s]:
        return None

    # reconstrói s -> t usando parent
    path = [s]
    cur = s
    while cur != t:
        nxt = parent[cur]
        if nxt == -1:
            return None
        path.append(nxt)
        cur = nxt
    return path


def max_flow_ford_fulkerson_pred_dfs(
    capacity: np.ndarray,
    pred: List[List[int]],
    s: int = 0,
    t: Optional[int] = None,
    max_iter: int = 10_000_000,
) -> MaxFlowResult:
    """
    Simulação 1:
    - entrada: matriz de capacidade e lista de antecessores
    - algoritmo de caminhos de fluxo (augmenting paths) + DFS alvo

    Observação: trabalhamos com o grafo residual.
    """
    n = capacity.shape[0]
    if t is None:
        t = n - 1

    residual = capacity.astype(float).copy()
    flow = 0.0

    it = 0
    while it < max_iter:
        it += 1
        path = _dfs_target_from_sink_using_predecessors(residual, pred, s, t)
        if path is None:
            break

        # gargalo
        bottleneck = float("inf")
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            bottleneck = min(bottleneck, residual[u, v])

        if bottleneck <= 0 or bottleneck == float("inf"):
            break

        # atualiza residual (forward - bottleneck, reverse + bottleneck)
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck

        flow += bottleneck

    return MaxFlowResult(max_flow=flow, residual=residual)
