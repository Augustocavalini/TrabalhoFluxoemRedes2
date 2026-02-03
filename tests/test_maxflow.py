from __future__ import annotations

import numpy as np
from src.graph import load_capacity_matrix_csv
from src.maxflow_pred_dfs import max_flow_ford_fulkerson_pred_dfs
from src.maxflow_succ_bfs import max_flow_edmonds_karp_succ_bfs


def test_known_graph_matrix_6():
    g = load_capacity_matrix_csv("examples/matrix_6.csv")
    pred = g.predecessor_lists(include_residual_reverse=True)
    succ = g.successor_lists(include_residual_reverse=True)

    r1 = max_flow_ford_fulkerson_pred_dfs(g.capacities, pred, s=0, t=g.n-1)
    r2 = max_flow_edmonds_karp_succ_bfs(g.capacities, succ, s=0, t=g.n-1)

    # Exemplo clássico (CLRS): max flow = 23
    assert abs(r1.max_flow - 23.0) < 1e-9
    assert abs(r2.max_flow - 23.0) < 1e-9
