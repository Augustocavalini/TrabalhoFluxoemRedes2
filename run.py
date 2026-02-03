from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime

from src.graph import load_capacity_matrix_csv, generate_dense_random_graph, save_capacity_matrix_csv
from src.maxflow_pred_dfs import max_flow_ford_fulkerson_pred_dfs
from src.maxflow_succ_bfs import max_flow_edmonds_karp_succ_bfs
from src.benchmark import run_benchmark
from src.part2_prompt import run_manual_llm_flow
from src.report import build_report


def _timestamp_dir(base: Path) -> Path:
    tag = datetime.now().strftime("run_%Y-%m-%d_%H%M%S")
    out = base / tag
    out.mkdir(parents=True, exist_ok=True)
    return out


def cmd_simulate(args: argparse.Namespace) -> None:
    out = _timestamp_dir(Path("outputs"))
    data = run_benchmark(
        out_dir=out,
        density_10=args.density10,
        density_100=args.density100,
        seed=args.seed,
    )
    print("\n=== BENCHMARK (Parte 1) ===")
    for row in data["rows"]:
        print(
            f"n={row['n']} | FF(antecessores+DFS) flow={row['ff_pred_dfs_flow']:.2f} time={row['ff_pred_dfs_seconds']:.6f}s"
            f" | EK(sucessores+BFS) flow={row['ek_succ_bfs_flow']:.2f} time={row['ek_succ_bfs_seconds']:.6f}s"
            f" | equal={row['equal_flows']}"
        )
    print(f"\nArquivos salvos em: {out.resolve()}")


def cmd_from_matrix(args: argparse.Namespace) -> None:
    g = load_capacity_matrix_csv(args.path)
    n = g.n
    pred = g.predecessor_lists(include_residual_reverse=True)
    succ = g.successor_lists(include_residual_reverse=True)

    r1 = max_flow_ford_fulkerson_pred_dfs(g.capacities, pred, s=0, t=n-1)
    r2 = max_flow_edmonds_karp_succ_bfs(g.capacities, succ, s=0, t=n-1)

    print("=== RESULTADOS ===")
    print(f"n={n} | origem=X1 | destino=X{n}")
    print(f"FF (antecessores + DFS alvo): {r1.max_flow}")
    print(f"EK (sucessores + BFS alvo):  {r2.max_flow}")
    print(f"Iguais? {abs(r1.max_flow - r2.max_flow) < 1e-9}")


def cmd_prompt(args: argparse.Namespace) -> None:
    out = _timestamp_dir(Path("outputs"))
    # salva também a matriz para referência
    g = generate_dense_random_graph(n=args.n, density=args.density, seed=args.seed)
    save_capacity_matrix_csv(g, out / f"matrix_n{args.n}_seed{args.seed}.csv")

    res = run_manual_llm_flow(n=args.n, density=args.density, seed=args.seed)
    (out / "benchmark_part2_manual.json").write_text(json.dumps(res.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResultado (Parte 2 - manual) salvo em: {out.resolve()}")


def cmd_report(args):
    pdf_path = build_report(
        outputs_dir=args.outputs_dir,
        run_dir=args.run_dir,
        out_pdf=args.out_pdf,
    )
    print("\n=== RELATÓRIO GERADO ===")
    print(f"PDF: {pdf_path.resolve()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Trabalho Fluxo Máximo — Simulações + Benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_sim = sub.add_parser("simulate", help="Roda simulações (n=10 e n=100) e mede tempos.")
    p_sim.add_argument("--seed", type=int, default=42)
    p_sim.add_argument("--density10", type=float, default=0.55)
    p_sim.add_argument("--density100", type=float, default=0.35)
    p_sim.set_defaults(func=cmd_simulate)

    p_mat = sub.add_parser("from-matrix", help="Lê uma matriz CSV NxN e computa o fluxo máximo.")
    p_mat.add_argument("path", type=str)
    p_mat.set_defaults(func=cmd_from_matrix)

    p_pr = sub.add_parser("prompt", help="Parte 2: gera prompt + mede tempo manual da IA.")
    p_pr.add_argument("--n", type=int, default=10)
    p_pr.add_argument("--density", type=float, default=0.55)
    p_pr.add_argument("--seed", type=int, default=42)
    p_pr.set_defaults(func=cmd_prompt)

    p_rep = sub.add_parser("report", help="Gera um PDF com os resultados a partir de outputs/.")
    p_rep.add_argument("--outputs-dir", type=str, default="outputs")
    p_rep.add_argument("--run-dir", type=str, default=None, help="Pasta específica run_... (opcional).")
    p_rep.add_argument("--out-pdf", type=str, default=None, help="Caminho do PDF (opcional).")
    p_rep.set_defaults(func=cmd_report)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
