from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Iterable, Optional
import random
import csv

import numpy as np


def label(i: int) -> str:
    """0-indexed -> 'X1', 'X2', ..."""
    return f"X{i+1}"


@dataclass
class CapacitatedDigraph:
    """
    Grafo direcionado capacitado representado por matriz de capacidades (NxN).

    - capacities[u][v] = capacidade do arco u->v (0 se inexistente)
    - Sem laços (u != v).
    """
    capacities: np.ndarray  # shape (n,n), dtype int/float

    @property
    def n(self) -> int:
        return int(self.capacities.shape[0])

    def arcs(self) -> List[Tuple[int, int, float]]:
        """Retorna lista de arcos (u,v,c) apenas para c>0."""
        n = self.n
        out: List[Tuple[int,int,float]] = []
        cap = self.capacities
        for u in range(n):
            for v in range(n):
                if u != v and cap[u, v] > 0:
                    out.append((u, v, float(cap[u, v])))
        return out

    def predecessor_lists(self, include_residual_reverse: bool = True) -> List[List[int]]:
        """
        Lista de antecessores (pred[v] = [u...] tal que pode existir u->v).

        Para permitir caminhar no **residual**, opcionalmente incluímos também os
        pares reversos (v como antecessor de u) quando existe arco u->v no original.
        """
        n = self.n
        pred_sets = [set() for _ in range(n)]
        for u, v, _c in self.arcs():
            pred_sets[v].add(u)
            if include_residual_reverse:
                pred_sets[u].add(v)  # reverse pode surgir no residual
        return [sorted(list(s)) for s in pred_sets]

    def successor_lists(self, include_residual_reverse: bool = True) -> List[List[int]]:
        """
        Lista de sucessores (succ[u] = [v...] tal que pode existir u->v).

        Para permitir caminhar no **residual**, opcionalmente incluímos também o
        vizinho reverso quando existe arco u->v no original.
        """
        n = self.n
        succ_sets = [set() for _ in range(n)]
        for u, v, _c in self.arcs():
            succ_sets[u].add(v)
            if include_residual_reverse:
                succ_sets[v].add(u)  # reverse pode surgir no residual
        return [sorted(list(s)) for s in succ_sets]


def generate_dense_random_graph(
    n: int,
    density: float = 0.45,
    cap_min: int = 1,
    cap_max: int = 20,
    seed: Optional[int] = None,
) -> CapacitatedDigraph:
    """
    Gera um grafo direcionado denso sem laços.

    density: probabilidade de existir arco u->v (u!=v).
    """
    rng = random.Random(seed)
    mat = np.zeros((n, n), dtype=float)
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            if rng.random() < density:
                mat[u, v] = rng.randint(cap_min, cap_max)
    return CapacitatedDigraph(mat)


def load_capacity_matrix_csv(path: str | Path) -> CapacitatedDigraph:
    """
    Lê CSV NxN (com ou sem cabeçalho) e devolve o grafo capacitado.
    """
    path = Path(path)
    rows: List[List[float]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            # tenta converter; se falhar, pula (ex: cabeçalho)
            try:
                rows.append([float(x.strip()) for x in row])
            except Exception:
                continue
    if not rows:
        raise ValueError(f"Nenhuma linha numérica encontrada em: {path}")
    n = len(rows)
    for r in rows:
        if len(r) != n:
            raise ValueError(f"Matriz deve ser NxN. Achei linha com {len(r)} colunas, esperado {n}.")
    mat = np.array(rows, dtype=float)
    # zera diagonal (sem laços)
    np.fill_diagonal(mat, 0.0)
    return CapacitatedDigraph(mat)


def save_capacity_matrix_csv(g: CapacitatedDigraph, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for row in g.capacities.tolist():
            writer.writerow(row)
