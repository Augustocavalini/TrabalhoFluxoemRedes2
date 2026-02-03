from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
import json
import time

import numpy as np

from .graph import generate_dense_random_graph, save_capacity_matrix_csv
from .maxflow_pred_dfs import max_flow_ford_fulkerson_pred_dfs
from .maxflow_succ_bfs import max_flow_edmonds_karp_succ_bfs


@dataclass
class BenchRow:
    n: int
    density: float
    seed: int
    ff_pred_dfs_flow: float
    ff_pred_dfs_times: List[float]
    ff_pred_dfs_mean: float
    ff_pred_dfs_min: float
    ff_pred_dfs_max: float
    ff_pred_dfs_median: float
    ff_pred_dfs_std: float
    ff_pred_dfs_cv: float
    ff_pred_dfs_p95: float
    ek_succ_bfs_flow: float
    ek_succ_bfs_times: List[float]
    ek_succ_bfs_mean: float
    ek_succ_bfs_min: float
    ek_succ_bfs_max: float
    ek_succ_bfs_median: float
    ek_succ_bfs_std: float
    ek_succ_bfs_cv: float
    ek_succ_bfs_p95: float
    equal_flows: bool


def _calc_stats(times: List[float]) -> Dict[str, float]:
    if not times:
        return {
            "min": 0.0,
            "max": 0.0,
            "median": 0.0,
            "std": 0.0,
            "cv": 0.0,
            "p95": 0.0,
        }
    arr = np.array(times, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = float(std / mean) if mean > 0 else 0.0
    return {
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "std": std,
        "cv": cv,
        "p95": float(np.percentile(arr, 95)),
    }


def run_benchmark(
    out_dir: str | Path,
    density_10: float = 0.70,
    density_100: float = 0.30,
    seed: int = 42,
    repetitions: int = 10,
) -> Dict[str, Any]:
    """
    Roda as simulações pedidas (n=10 e n=100) e mede o tempo das duas soluções.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n, density in [(10, density_10), (100, density_100)]:
        g = generate_dense_random_graph(n=n, density=density, seed=seed)
        if n == 10:
            save_capacity_matrix_csv(g, out_dir / f"matrix_n{n}_seed{seed}.csv")
        cap = g.capacities
        pred = g.predecessor_lists(include_residual_reverse=True)
        succ = g.successor_lists(include_residual_reverse=True)
        ff_times: List[float] = []
        ek_times: List[float] = []
        r1 = None
        r2 = None
        for _i in range(repetitions):
            t0 = time.perf_counter()
            r1 = max_flow_ford_fulkerson_pred_dfs(cap, pred, s=0, t=n-1)
            t1 = time.perf_counter()
            ff_times.append(float(t1 - t0))

            t2 = time.perf_counter()
            r2 = max_flow_edmonds_karp_succ_bfs(cap, succ, s=0, t=n-1)
            t3 = time.perf_counter()
            ek_times.append(float(t3 - t2))

        ff_mean = float(sum(ff_times) / len(ff_times)) if ff_times else 0.0
        ek_mean = float(sum(ek_times) / len(ek_times)) if ek_times else 0.0
        ff_stats = _calc_stats(ff_times)
        ek_stats = _calc_stats(ek_times)

        rows.append(BenchRow(
            n=n,
            density=density,
            seed=seed,
            ff_pred_dfs_flow=float(r1.max_flow) if r1 is not None else 0.0,
            ff_pred_dfs_times=ff_times,
            ff_pred_dfs_mean=ff_mean,
            ff_pred_dfs_min=ff_stats["min"],
            ff_pred_dfs_max=ff_stats["max"],
            ff_pred_dfs_median=ff_stats["median"],
            ff_pred_dfs_std=ff_stats["std"],
            ff_pred_dfs_cv=ff_stats["cv"],
            ff_pred_dfs_p95=ff_stats["p95"],
            ek_succ_bfs_flow=float(r2.max_flow) if r2 is not None else 0.0,
            ek_succ_bfs_times=ek_times,
            ek_succ_bfs_mean=ek_mean,
            ek_succ_bfs_min=ek_stats["min"],
            ek_succ_bfs_max=ek_stats["max"],
            ek_succ_bfs_median=ek_stats["median"],
            ek_succ_bfs_std=ek_stats["std"],
            ek_succ_bfs_cv=ek_stats["cv"],
            ek_succ_bfs_p95=ek_stats["p95"],
            equal_flows=(abs(float(r1.max_flow) - float(r2.max_flow)) < 1e-9) if (r1 is not None and r2 is not None) else False,
        ))

    # salva
    data = {"rows": [asdict(r) for r in rows]}
    (out_dir / "benchmark_part1.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # CSV simples (sem pandas)
    csv_lines = [
        "n,density,seed,ff_pred_dfs_flow,ff_pred_dfs_times,ff_pred_dfs_mean,"
        "ff_pred_dfs_min,ff_pred_dfs_max,ff_pred_dfs_median,ff_pred_dfs_std,ff_pred_dfs_cv,ff_pred_dfs_p95,"
        "ek_succ_bfs_flow,ek_succ_bfs_times,ek_succ_bfs_mean,"
        "ek_succ_bfs_min,ek_succ_bfs_max,ek_succ_bfs_median,ek_succ_bfs_std,ek_succ_bfs_cv,ek_succ_bfs_p95,"
        "equal_flows"
    ]
    for r in rows:
        ff_times = "|".join(f"{t:.6f}" for t in r.ff_pred_dfs_times)
        ek_times = "|".join(f"{t:.6f}" for t in r.ek_succ_bfs_times)
        csv_lines.append(
            f"{r.n},{r.density},{r.seed},{r.ff_pred_dfs_flow},{ff_times},{r.ff_pred_dfs_mean},"
            f"{r.ff_pred_dfs_min},{r.ff_pred_dfs_max},{r.ff_pred_dfs_median},{r.ff_pred_dfs_std},{r.ff_pred_dfs_cv},{r.ff_pred_dfs_p95},"
            f"{r.ek_succ_bfs_flow},{ek_times},{r.ek_succ_bfs_mean},"
            f"{r.ek_succ_bfs_min},{r.ek_succ_bfs_max},{r.ek_succ_bfs_median},{r.ek_succ_bfs_std},{r.ek_succ_bfs_cv},{r.ek_succ_bfs_p95},"
            f"{int(r.equal_flows)}"
        )
    (out_dir / "benchmark_part1.csv").write_text("\n".join(csv_lines), encoding="utf-8")

    return data
