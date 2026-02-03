from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime
import json
import math

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt
import numpy as np

from .graph import load_capacity_matrix_csv, label
from .maxflow_succ_bfs import max_flow_edmonds_karp_succ_bfs


@dataclass
class Part1Row:
    n: int
    density: float
    seed: int
    ff_pred_dfs_flow: float
    ff_pred_dfs_times: List[float]
    ff_pred_dfs_mean: float
    ek_succ_bfs_flow: float
    ek_succ_bfs_times: List[float]
    ek_succ_bfs_mean: float
    equal_flows: bool


@dataclass
class Part2Row:
    n: int
    seed: int
    density: float
    llm_seconds: float
    llm_max_flow: float


def _find_latest_run_dir(outputs_dir: Path) -> Path:
    if not outputs_dir.exists():
        raise FileNotFoundError(f"Pasta de outputs não existe: {outputs_dir}")
    candidates = [p for p in outputs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    if not candidates:
        raise FileNotFoundError(f"Não encontrei nenhum run_... em {outputs_dir}. Rode: python run.py simulate")
    return sorted(candidates, key=lambda p: p.name)[-1]


def _load_part1(run_dir: Path) -> List[Part1Row]:
    p = run_dir / "benchmark_part1.json"
    if not p.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {p}. Rode: python run.py simulate")
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = []
    for r in data.get("rows", []):
        ff_times = [float(x) for x in r.get("ff_pred_dfs_times", [])]
        ek_times = [float(x) for x in r.get("ek_succ_bfs_times", [])]
        ff_seconds = float(r.get("ff_pred_dfs_seconds", 0.0))
        ek_seconds = float(r.get("ek_succ_bfs_seconds", 0.0))
        if not ff_times:
            ff_times = [ff_seconds]
        if not ek_times:
            ek_times = [ek_seconds]
        ff_mean = float(r.get("ff_pred_dfs_mean", sum(ff_times) / len(ff_times)))
        ek_mean = float(r.get("ek_succ_bfs_mean", sum(ek_times) / len(ek_times)))
        rows.append(Part1Row(
            n=int(r["n"]),
            density=float(r["density"]),
            seed=int(r["seed"]),
            ff_pred_dfs_flow=float(r["ff_pred_dfs_flow"]),
            ff_pred_dfs_times=ff_times,
            ff_pred_dfs_mean=ff_mean,
            ek_succ_bfs_flow=float(r["ek_succ_bfs_flow"]),
            ek_succ_bfs_times=ek_times,
            ek_succ_bfs_mean=ek_mean,
            equal_flows=bool(r["equal_flows"]),
        ))
    if not rows:
        raise ValueError("benchmark_part1.json não possui linhas em 'rows'.")
    return rows


def _load_part2_if_any(run_dir: Path) -> Optional[Part2Row]:
    p = run_dir / "benchmark_part2_manual.json"
    if not p.exists():
        return None
    data = json.loads(p.read_text(encoding="utf-8"))
    return Part2Row(
        n=int(data["n"]),
        seed=int(data["seed"]),
        density=float(data["density"]),
        llm_seconds=float(data["llm_seconds"]),
        llm_max_flow=float(data["llm_max_flow"]),
    )


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a > 0 else 1.0
    return a / b


def _make_part1_table(rows: List[Part1Row]) -> Table:
    header = [
        "n", "densidade", "seed",
        "FF (pred+DFS)\nfluxo", "FF média (s)",
        "EK (succ+BFS)\nfluxo", "EK média (s)",
        "igual?"
    ]
    body = []
    for r in rows:
        body.append([
            str(r.n),
            f"{r.density:.2f}",
            str(r.seed),
            f"{r.ff_pred_dfs_flow:.2f}",
            f"{r.ff_pred_dfs_mean:.6f}",
            f"{r.ek_succ_bfs_flow:.2f}",
            f"{r.ek_succ_bfs_mean:.6f}",
            "sim" if r.equal_flows else "não",
        ])
    table = Table(
        [header] + body,
        colWidths=[1.1*cm, 1.8*cm, 1.2*cm, 2.4*cm, 2.2*cm, 2.4*cm, 2.2*cm, 1.2*cm],
    )
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING", (0,0), (-1,0), 6),
        ("TOPPADDING", (0,0), (-1,0), 6),
    ]))
    return table


def _make_times_table(rows: List[Part1Row]) -> Table:
    header = [
        "n",
        "FF t1", "FF t2", "FF t3", "FF t4", "FF t5", "FF média",
        "EK t1", "EK t2", "EK t3", "EK t4", "EK t5", "EK média",
    ]
    body = []
    for r in rows:
        ff = (r.ff_pred_dfs_times + [0.0] * 5)[:5]
        ek = (r.ek_succ_bfs_times + [0.0] * 5)[:5]
        body.append([
            str(r.n),
            *[f"{t:.6f}" for t in ff],
            f"{r.ff_pred_dfs_mean:.6f}",
            *[f"{t:.6f}" for t in ek],
            f"{r.ek_succ_bfs_mean:.6f}",
        ])
    table = Table(
        [header] + body,
        colWidths=[1.0*cm] + [1.2*cm]*5 + [1.6*cm] + [1.2*cm]*5 + [1.6*cm],
    )
    table.setStyle(TableStyle([
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
        ("VALIGN", (0,0), (-1,-1), "MIDDLE"),
        ("FONTSIZE", (0,0), (-1,-1), 7),
        ("BOTTOMPADDING", (0,0), (-1,0), 4),
        ("TOPPADDING", (0,0), (-1,0), 4),
    ]))
    return table


def _graph_layout(n: int) -> List[Tuple[float, float]]:
    angles = [2 * math.pi * i / n for i in range(n)]
    return [(math.cos(a), math.sin(a)) for a in angles]


def _plot_directed_graph(
    capacity: np.ndarray,
    out_png: Path,
    title: str,
    edge_values: Optional[np.ndarray] = None,
    edge_color: str = "#1f77b4",
) -> None:
    n = capacity.shape[0]
    pos = _graph_layout(n)
    max_val = 0.0
    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            val = edge_values[u, v] if edge_values is not None else capacity[u, v]
            if val > max_val:
                max_val = float(val)
    max_val = max_val if max_val > 0 else 1.0

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            base_val = capacity[u, v]
            if base_val <= 0:
                continue
            val = edge_values[u, v] if edge_values is not None else base_val
            if edge_values is not None and val <= 0:
                continue
            width = 0.5 + 2.5 * (float(val) / max_val)
            ax.annotate(
                "",
                xy=pos[v],
                xytext=pos[u],
                arrowprops=dict(arrowstyle="->", lw=width, color=edge_color, alpha=0.65),
            )
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            ax.text(mx, my, f"{val:.1f}", fontsize=7, color=edge_color)

    for i, (x, y) in enumerate(pos):
        ax.scatter([x], [y], s=200, color="#f2f2f2", edgecolors="#333333", zorder=3)
        ax.text(x, y, label(i), fontsize=8, ha="center", va="center", zorder=4)

    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_flow_heatmap(flow: np.ndarray, out_png: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(flow, cmap="viridis")
    plt.colorbar(label="fluxo por aresta")
    plt.title("Mapa de calor do fluxo (arestas)")
    plt.xlabel("v")
    plt.ylabel("u")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _flow_stats(flow: np.ndarray) -> Tuple[int, float, float]:
    positive = flow[flow > 0]
    count = int(positive.size)
    mean = float(positive.mean()) if count > 0 else 0.0
    max_val = float(positive.max()) if count > 0 else 0.0
    return count, mean, max_val


def _plot_times(rows: List[Part1Row], out_png: Path) -> None:
    ns = [r.n for r in rows]
    ff = [r.ff_pred_dfs_mean for r in rows]
    ek = [r.ek_succ_bfs_mean for r in rows]

    plt.figure(figsize=(8, 4.5))
    x = range(len(ns))
    plt.bar([i - 0.2 for i in x], ff, width=0.4, label="FF (pred+DFS)")
    plt.bar([i + 0.2 for i in x], ek, width=0.4, label="EK (succ+BFS)")
    plt.xticks(list(x), [f"n={n}" for n in ns])
    plt.ylabel("tempo (s)")
    plt.title("Tempo por simulação (Parte 1)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_speedup(rows: List[Part1Row], out_png: Path) -> None:
    ns = [r.n for r in rows]
    speedup = [_safe_div(r.ff_pred_dfs_mean, r.ek_succ_bfs_mean) for r in rows]

    plt.figure(figsize=(8, 4.5))
    x = range(len(ns))
    plt.bar(list(x), speedup)
    plt.xticks(list(x), [f"n={n}" for n in ns])
    plt.ylabel("FF_time / EK_time")
    plt.title("Razão de tempo (FF/EK)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def generate_report_pdf(
    run_dir: str | Path,
    pdf_path: str | Path,
    project_title: str = "Relatório — Fluxo Máximo (Ford–Fulkerson vs Edmonds–Karp)",
) -> Path:
    run_dir = Path(run_dir)
    pdf_path = Path(pdf_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    h1 = styles["Heading1"]
    h2 = styles["Heading2"]
    body = styles["BodyText"]

    rows1 = _load_part1(run_dir)
    row2 = _load_part2_if_any(run_dir)

    png_times = run_dir / "report_times.png"
    png_speedup = run_dir / "report_speedup.png"
    _plot_times(rows1, png_times)
    _plot_speedup(rows1, png_speedup)

    doc = SimpleDocTemplate(
        str(pdf_path),
        pagesize=A4,
        leftMargin=2.0*cm,
        rightMargin=2.0*cm,
        topMargin=2.0*cm,
        bottomMargin=2.0*cm,
        title=project_title,
    )

    story = []
    story.append(Paragraph(project_title, h1))
    story.append(Spacer(1, 0.4*cm))
    story.append(Paragraph(f"Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}", body))
    story.append(Paragraph(f"Pasta do experimento: <b>{run_dir.name}</b>", body))
    story.append(Spacer(1, 0.6*cm))

    story.append(Paragraph("1. Objetivo", h2))
    story.append(Paragraph(
        "Comparar duas implementações de Fluxo Máximo no grafo residual: "
        "(i) Ford–Fulkerson com busca em profundidade alvo usando lista de antecessores; "
        "(ii) Edmonds–Karp (BFS) usando lista de sucessores. "
        "O benchmark registra o fluxo máximo encontrado e o tempo de execução em n=10 e n=100.",
        body
    ))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2. Resultados — Parte 1 (2 simulações, 5 repetições cada)", h2))
    story.append(Paragraph("Tabela consolidada dos testes automáticos (fluxos e tempos médios):", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(_make_part1_table(rows1))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.1 Tempos individuais (5 execuções)", h2))
    story.append(Paragraph("Tabela separada com os 5 tempos e a média:", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(_make_times_table(rows1))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.2 Tempo de execução", h2))
    story.append(Paragraph("Comparativo de tempo médio por n:", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(png_times), width=16*cm, height=9*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.3 Razão de tempo (FF/EK)", h2))
    story.append(Paragraph(
        "Valores maiores que 1 indicam que Ford–Fulkerson (pred+DFS) foi mais lento que Edmonds–Karp (succ+BFS).",
        body
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(png_speedup), width=16*cm, height=9*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.4 Observações", h2))
    obs_lines = []
    for r in rows1:
        ratio = _safe_div(r.ff_pred_dfs_mean, r.ek_succ_bfs_mean)
        ff_times = ", ".join(f"{t:.6f}s" for t in r.ff_pred_dfs_times)
        ek_times = ", ".join(f"{t:.6f}s" for t in r.ek_succ_bfs_times)
        obs_lines.append(
            f"Para n={r.n}, fluxos coincidem: {'sim' if r.equal_flows else 'não'}; "
            f"FF tempos=[{ff_times}] (média {r.ff_pred_dfs_mean:.6f}s), "
            f"EK tempos=[{ek_times}] (média {r.ek_succ_bfs_mean:.6f}s), FF/EK={ratio:.2f}."
        )
    story.append(Paragraph("<br/>".join(obs_lines), body))

    story.append(PageBreak())
    story.append(Paragraph("3. Visualizações do Grafo 1 (n=10)", h2))
    story.append(Paragraph(
        "Representação visual do grafo e do fluxo (usando Edmonds–Karp), além de métricas do fluxo nas arestas.",
        body,
    ))

    graph_png = run_dir / "graph_n10.png"
    flow_png = run_dir / "flow_n10.png"
    heatmap_png = run_dir / "flow_heatmap_n10.png"

    row_n10 = next((r for r in rows1 if r.n == 10), None)
    matrix_path = (run_dir / f"matrix_n10_seed{row_n10.seed}.csv") if row_n10 else None
    if matrix_path is not None and matrix_path.exists():
        g = load_capacity_matrix_csv(matrix_path)
        cap = g.capacities
        succ = g.successor_lists(include_residual_reverse=True)
        result = max_flow_edmonds_karp_succ_bfs(cap, succ, s=0, t=g.n - 1)
        flow = np.maximum(0.0, cap - result.residual)

        _plot_directed_graph(cap, graph_png, "Grafo capacitado (n=10)")
        _plot_directed_graph(cap, flow_png, "Fluxo nas arestas (n=10)", edge_values=flow, edge_color="#d62728")
        _plot_flow_heatmap(flow, heatmap_png)

        used_edges, mean_flow, max_flow_edge = _flow_stats(flow)
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(graph_png), width=14*cm, height=14*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(flow_png), width=14*cm, height=14*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(heatmap_png), width=14*cm, height=12*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph(
            f"Arestas com fluxo positivo: {used_edges}; "
            f"fluxo médio nas arestas usadas: {mean_flow:.3f}; "
            f"maior fluxo em uma aresta: {max_flow_edge:.3f}.",
            body,
        ))
    else:
        story.append(Paragraph(
            "Matriz do grafo n=10 não encontrada na pasta do experimento. Rode: python run.py simulate.",
            body,
        ))

    story.append(PageBreak())
    story.append(Paragraph("4. Parte 2 — Linguagem Generativa (manual)", h2))
    if row2 is None:
        story.append(Paragraph(
            "Não há resultado da Parte 2 nesta pasta. Para gerar, rode: "
            "<b>python run.py prompt</b> e depois execute <b>python run.py report</b> novamente.",
            body
        ))
    else:
        story.append(Paragraph(
            "Resultado colado pelo usuário (IA generativa) e tempo medido no terminal:",
            body
        ))
        story.append(Spacer(1, 0.2*cm))

        t = Table([
            ["n", "seed", "densidade", "tempo (s)", "fluxo máximo informado"],
            [str(row2.n), str(row2.seed), f"{row2.density:.2f}", f"{row2.llm_seconds:.3f}", f"{row2.llm_max_flow:.2f}"],
        ], colWidths=[2*cm, 2*cm, 3*cm, 3*cm, 5*cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 9),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(
            "Obs.: a Parte 2 é manual; o foco é registrar tempo de resposta e permitir comparação direta "
            "com as implementações locais (Parte 1).",
            body
        ))

    doc.build(story)
    return pdf_path


def build_report(
    outputs_dir: str | Path = "outputs",
    run_dir: str | Path | None = None,
    out_pdf: str | Path | None = None,
) -> Path:
    outputs_dir = Path(outputs_dir)
    run_dir_path = _find_latest_run_dir(outputs_dir) if run_dir is None else Path(run_dir)
    out_pdf_path = (run_dir_path / "relatorio_fluxo_maximo.pdf") if out_pdf is None else Path(out_pdf)
    return generate_report_pdf(run_dir_path, out_pdf_path)
