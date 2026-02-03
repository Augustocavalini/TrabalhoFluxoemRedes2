from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional
import json
import time

import numpy as np

from .graph import generate_dense_random_graph
from .maxflow_pred_dfs import max_flow_ford_fulkerson_pred_dfs
from .maxflow_succ_bfs import max_flow_edmonds_karp_succ_bfs


@dataclass
class BenchRow:
    n: int
    density: float
    seed: int
    ff_pred_dfs_flow: float
    ff_pred_dfs_seconds: float
    ek_succ_bfs_flow: float
    ek_succ_bfs_seconds: float
    equal_flows: bool


def run_benchmark(
    out_dir: str | Path,
    density_10: float = 0.55,
    density_100: float = 0.35,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Roda as simulações pedidas (n=10 e n=100) e mede o tempo das duas soluções.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, density in [(10, density_10), (100, density_100)]:
        g = generate_dense_random_graph(n=n, density=density, seed=seed)
        cap = g.capacities
        pred = g.predecessor_lists(include_residual_reverse=True)
        succ = g.successor_lists(include_residual_reverse=True)

        t0 = time.perf_counter()
        r1 = max_flow_ford_fulkerson_pred_dfs(cap, pred, s=0, t=n-1)
        t1 = time.perf_counter()

        t2 = time.perf_counter()
        r2 = max_flow_edmonds_karp_succ_bfs(cap, succ, s=0, t=n-1)
        t3 = time.perf_counter()

        rows.append(BenchRow(
            n=n,
            density=density,
            seed=seed,
            ff_pred_dfs_flow=float(r1.max_flow),
            ff_pred_dfs_seconds=float(t1 - t0),
            ek_succ_bfs_flow=float(r2.max_flow),
            ek_succ_bfs_seconds=float(t3 - t2),
            equal_flows=abs(float(r1.max_flow) - float(r2.max_flow)) < 1e-9,
        ))

    # salva
    data = {"rows": [asdict(r) for r in rows]}
    (out_dir / "benchmark_part1.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV simples (sem pandas)
    csv_lines = ["n,density,seed,ff_pred_dfs_flow,ff_pred_dfs_seconds,ek_succ_bfs_flow,ek_succ_bfs_seconds,equal_flows"]
    for r in rows:
        csv_lines.append(
            f"{r.n},{r.density},{r.seed},{r.ff_pred_dfs_flow},{r.ff_pred_dfs_seconds},{r.ek_succ_bfs_flow},{r.ek_succ_bfs_seconds},{int(r.equal_flows)}"
        )
    (out_dir / "benchmark_part1.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    return data
