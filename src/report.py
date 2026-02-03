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


@dataclass
class Part2Row:
    n: int
    seed: int
    density: float
    llm_seconds: float
    llm_max_flow: float
    model: str


# Tabela fixa com resultados da IA generativa (edite aqui quando necessário)
PART2_RESULTS: List[Part2Row] = [
    Part2Row(n=10, seed=42, density=0.55, llm_seconds=19, llm_max_flow=48, model="ChatGPT 5.2 Thinking"),
    Part2Row(n=100, seed=42, density=0.55, llm_seconds=186, llm_max_flow=537, model="ChatGPT 5.2 Thinking"),
]


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
        ff_min, ff_max, ff_median, ff_std, ff_cv, ff_p95 = _stats_from_times(ff_times)
        ek_min, ek_max, ek_median, ek_std, ek_cv, ek_p95 = _stats_from_times(ek_times)
        rows.append(Part1Row(
            n=int(r["n"]),
            density=float(r["density"]),
            seed=int(r["seed"]),
            ff_pred_dfs_flow=float(r["ff_pred_dfs_flow"]),
            ff_pred_dfs_times=ff_times,
            ff_pred_dfs_mean=ff_mean,
            ff_pred_dfs_min=float(r.get("ff_pred_dfs_min", ff_min)),
            ff_pred_dfs_max=float(r.get("ff_pred_dfs_max", ff_max)),
            ff_pred_dfs_median=float(r.get("ff_pred_dfs_median", ff_median)),
            ff_pred_dfs_std=float(r.get("ff_pred_dfs_std", ff_std)),
            ff_pred_dfs_cv=float(r.get("ff_pred_dfs_cv", ff_cv)),
            ff_pred_dfs_p95=float(r.get("ff_pred_dfs_p95", ff_p95)),
            ek_succ_bfs_flow=float(r["ek_succ_bfs_flow"]),
            ek_succ_bfs_times=ek_times,
            ek_succ_bfs_mean=ek_mean,
            ek_succ_bfs_min=float(r.get("ek_succ_bfs_min", ek_min)),
            ek_succ_bfs_max=float(r.get("ek_succ_bfs_max", ek_max)),
            ek_succ_bfs_median=float(r.get("ek_succ_bfs_median", ek_median)),
            ek_succ_bfs_std=float(r.get("ek_succ_bfs_std", ek_std)),
            ek_succ_bfs_cv=float(r.get("ek_succ_bfs_cv", ek_cv)),
            ek_succ_bfs_p95=float(r.get("ek_succ_bfs_p95", ek_p95)),
            equal_flows=bool(r["equal_flows"]),
        ))
    if not rows:
        raise ValueError("benchmark_part1.json não possui linhas em 'rows'.")
    return rows


def _load_part2_if_any(run_dir: Path) -> List[Part2Row]:
    return PART2_RESULTS


def _safe_div(a: float, b: float) -> float:
    if b == 0:
        return float("inf") if a > 0 else 1.0
    return a / b


def _stats_from_times(times: List[float]) -> Tuple[float, float, float, float, float, float]:
    if not times:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(times, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
    cv = float(std / mean) if mean > 0 else 0.0
    return (
        float(arr.min()),
        float(arr.max()),
        float(np.median(arr)),
        std,
        cv,
        float(np.percentile(arr, 95)),
    )


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
    header = ["iteração"]
    for r in rows:
        header.append(f"n={r.n} FF")
        header.append(f"n={r.n} EK")

    max_reps = 0
    for r in rows:
        max_reps = max(max_reps, len(r.ff_pred_dfs_times), len(r.ek_succ_bfs_times))

    body = []
    for i in range(max_reps):
        row = [str(i + 1)]
        for r in rows:
            ff_val = r.ff_pred_dfs_times[i] if i < len(r.ff_pred_dfs_times) else None
            ek_val = r.ek_succ_bfs_times[i] if i < len(r.ek_succ_bfs_times) else None
            row.append(f"{ff_val:.6f}" if ff_val is not None else "")
            row.append(f"{ek_val:.6f}" if ek_val is not None else "")
        body.append(row)

    mean_row = ["média"]
    for r in rows:
        mean_row.append(f"{r.ff_pred_dfs_mean:.6f}")
        mean_row.append(f"{r.ek_succ_bfs_mean:.6f}")
    body.append(mean_row)

    table = Table(
        [header] + body,
        colWidths=[1.6*cm] + [2.0*cm] * (len(rows) * 2),
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


def _make_stats_table(rows: List[Part1Row]) -> Table:
    header = [
        "n",
        "FF min", "FF med", "FF max", "FF std", "FF cv%", "FF p95",
        "EK min", "EK med", "EK max", "EK std", "EK cv%", "EK p95",
    ]
    body = []
    for r in rows:
        body.append([
            str(r.n),
            f"{r.ff_pred_dfs_min:.6f}",
            f"{r.ff_pred_dfs_median:.6f}",
            f"{r.ff_pred_dfs_max:.6f}",
            f"{r.ff_pred_dfs_std:.6f}",
            f"{(r.ff_pred_dfs_cv * 100):.2f}",
            f"{r.ff_pred_dfs_p95:.6f}",
            f"{r.ek_succ_bfs_min:.6f}",
            f"{r.ek_succ_bfs_median:.6f}",
            f"{r.ek_succ_bfs_max:.6f}",
            f"{r.ek_succ_bfs_std:.6f}",
            f"{(r.ek_succ_bfs_cv * 100):.2f}",
            f"{r.ek_succ_bfs_p95:.6f}",
        ])
    table = Table(
        [header] + body,
        colWidths=[1.0*cm] + [1.3*cm]*12,
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


def _plot_residual_heatmap(residual: np.ndarray, capacity: np.ndarray, out_png: Path) -> None:
    residual_pct = np.zeros_like(residual)
    mask = capacity > 0
    residual_pct[mask] = (residual[mask] / capacity[mask]) * 100.0
    
    plt.figure(figsize=(6, 5))
    plt.imshow(residual_pct, cmap="magma", vmin=0, vmax=100)
    plt.colorbar(label="capacidade residual (%)")
    plt.title("Mapa de calor do residual (% não utilizado)")
    plt.xlabel("v")
    plt.ylabel("u")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_directed_graph_saturation(
    capacity: np.ndarray,
    flow: np.ndarray,
    out_png: Path,
    title: str,
    cmap: str = "plasma",
) -> None:
    n = capacity.shape[0]
    pos = _graph_layout(n)
    max_flow_val = float(flow.max()) if flow.size else 1.0
    max_flow_val = max_flow_val if max_flow_val > 0 else 1.0

    plt.figure(figsize=(6, 6))
    ax = plt.gca()
    ax.set_aspect("equal")
    ax.axis("off")

    norm = plt.Normalize(0.0, 1.0)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)

    for u in range(n):
        for v in range(n):
            if u == v:
                continue
            cap = capacity[u, v]
            if cap <= 0:
                continue
            f = flow[u, v]
            sat = float(f / cap) if cap > 0 else 0.0
            if f <= 0:
                continue
            width = 0.5 + 2.5 * (float(f) / max_flow_val)
            color = sm.to_rgba(sat)
            ax.annotate(
                "",
                xy=pos[v],
                xytext=pos[u],
                arrowprops=dict(arrowstyle="->", lw=width, color=color, alpha=0.75),
            )
            mx = (pos[u][0] + pos[v][0]) / 2
            my = (pos[u][1] + pos[v][1]) / 2
            ax.text(mx, my, f"{sat*100:.0f}%", fontsize=7, color=color)

    for i, (x, y) in enumerate(pos):
        ax.scatter([x], [y], s=200, color="#f2f2f2", edgecolors="#333333", zorder=3)
        ax.text(x, y, label(i), fontsize=8, ha="center", va="center", zorder=4)

    plt.title(title)
    plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04, label="saturação")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()




def _plot_flow_node_balance(flow: np.ndarray, out_png: Path) -> None:
    n = flow.shape[0]
    out_flow = flow.sum(axis=1)
    in_flow = flow.sum(axis=0)
    x = np.arange(n)

    plt.figure(figsize=(8, 4.5))
    plt.bar(x - 0.2, out_flow, width=0.4, label="saída")
    plt.bar(x + 0.2, in_flow, width=0.4, label="entrada")
    plt.xticks(list(x), [label(i) for i in range(n)], rotation=0)
    plt.ylabel("fluxo")
    plt.title("Fluxo total por nó (entrada vs saída)")
    plt.legend()
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


def _plot_times_errorbars(rows: List[Part1Row], out_png: Path) -> None:
    ns = [r.n for r in rows]
    ff = [r.ff_pred_dfs_mean for r in rows]
    ek = [r.ek_succ_bfs_mean for r in rows]
    ff_std = [r.ff_pred_dfs_std for r in rows]
    ek_std = [r.ek_succ_bfs_std for r in rows]

    plt.figure(figsize=(8, 4.5))
    x = np.arange(len(ns))
    plt.bar(x - 0.2, ff, width=0.4, yerr=ff_std, capsize=4, label="FF (pred+DFS)")
    plt.bar(x + 0.2, ek, width=0.4, yerr=ek_std, capsize=4, label="EK (succ+BFS)")
    plt.xticks(list(x), [f"n={n}" for n in ns])
    plt.ylabel("tempo (s)")
    plt.title("Tempo médio com desvio padrão")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_boxplots(rows: List[Part1Row], out_png: Path) -> None:
    labels = []
    data = []
    for r in rows:
        labels.append(f"n={r.n} FF")
        data.append(r.ff_pred_dfs_times)
        labels.append(f"n={r.n} EK")
        data.append(r.ek_succ_bfs_times)

    plt.figure(figsize=(10, 4.5))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel("tempo (s)")
    plt.title("Distribuição dos tempos por algoritmo")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()


def _plot_histograms(rows: List[Part1Row], out_png: Path) -> None:
    ncols = len(rows)
    if ncols == 0:
        return
    plt.figure(figsize=(4.5 * ncols, 4.0))
    for i, r in enumerate(rows):
        plt.subplot(1, ncols, i + 1)
        bins = min(10, max(5, len(r.ff_pred_dfs_times) // 2))
        plt.hist(r.ff_pred_dfs_times, bins=bins, alpha=0.6, label="FF")
        plt.hist(r.ek_succ_bfs_times, bins=bins, alpha=0.6, label="EK")
        plt.title(f"n={r.n}")
        plt.xlabel("tempo (s)")
        plt.ylabel("freq.")
        plt.legend()
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

    try:
        rows1 = _load_part1(run_dir)
    except FileNotFoundError:
        rows1 = []
    rows2 = _load_part2_if_any(run_dir)

    png_times = run_dir / "report_times.png"
    png_times_std = run_dir / "report_times_std.png"
    png_boxplot = run_dir / "report_times_boxplot.png"
    png_hist = run_dir / "report_times_hist.png"
    png_speedup = run_dir / "report_speedup.png"
    if rows1:
        _plot_times(rows1, png_times)
        _plot_times_errorbars(rows1, png_times_std)
        _plot_boxplots(rows1, png_boxplot)
        _plot_histograms(rows1, png_hist)
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

    story.append(Paragraph("2. Resultados — Parte 1 (2 simulações, 10 repetições cada)", h2))
    if not rows1:
        story.append(Paragraph(
            "Resultados da Parte 1 não encontrados nesta pasta. Rode: <b>python run.py simulate</b> e gere o relatório novamente.",
            body,
        ))
    else:
        story.append(Paragraph("Tabela consolidada dos testes automáticos (fluxos e tempos médios):", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(_make_part1_table(rows1))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.1 Tempos individuais (10 execuções)", h2))
        story.append(Paragraph("Tabela com linhas por iteração, colunas por matriz/método e última linha com as médias:", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(_make_times_table(rows1))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.2 Estatísticas de dispersão", h2))
        story.append(Paragraph("Mínimo, mediana, máximo, desvio padrão, coeficiente de variação e p95:", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(_make_stats_table(rows1))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.3 Tempo de execução", h2))
        story.append(Paragraph("Comparativo de tempo médio por n:", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(png_times), width=16*cm, height=9*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.4 Tempo médio com desvio padrão", h2))
        story.append(Paragraph("Barras com incerteza (desvio padrão):", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(png_times_std), width=16*cm, height=9*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.5 Boxplot dos tempos", h2))
        story.append(Paragraph("Distribuição dos tempos por algoritmo e tamanho de grafo:", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(png_boxplot), width=16*cm, height=8.5*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.6 Histogramas dos tempos", h2))
        story.append(Paragraph("Frequência dos tempos por algoritmo em cada n:", body))
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(png_hist), width=16*cm, height=8*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.7 Razão de tempo (FF/EK)", h2))
        story.append(Paragraph(
            "Valores maiores que 1 indicam que Ford–Fulkerson (pred+DFS) foi mais lento que Edmonds–Karp (succ+BFS).",
            body
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(png_speedup), width=16*cm, height=9*cm))
        story.append(Spacer(1, 0.4*cm))

        story.append(Paragraph("2.8 Observações", h2))
        obs_lines = []
        for r in rows1:
            ratio = _safe_div(r.ff_pred_dfs_mean, r.ek_succ_bfs_mean)
            ff_times = ", ".join(f"{t:.6f}s" for t in r.ff_pred_dfs_times)
            ek_times = ", ".join(f"{t:.6f}s" for t in r.ek_succ_bfs_times)
            obs_lines.append(
                f"Para n={r.n}, fluxos coincidem: {'sim' if r.equal_flows else 'não'}; "
                f"FF tempos=[{ff_times}] (média {r.ff_pred_dfs_mean:.6f}s), "
                f"EK tempos=[{ek_times}] (média {r.ek_succ_bfs_mean:.6f}s), FF/EK={ratio:.2f}. "
                f"Dispersão: FF std={r.ff_pred_dfs_std:.6f}s (CV {r.ff_pred_dfs_cv*100:.2f}%), "
                f"EK std={r.ek_succ_bfs_std:.6f}s (CV {r.ek_succ_bfs_cv*100:.2f}%), "
                f"p95 FF={r.ff_pred_dfs_p95:.6f}s, p95 EK={r.ek_succ_bfs_p95:.6f}s."
            )
        story.append(Paragraph("<br/>".join(obs_lines), body))

    story.append(PageBreak())
    story.append(Paragraph("3. Visualizações do Grafo 1 (n=10)", h2))
    story.append(Paragraph(
        "Representação visual do grafo e do fluxo (usando Edmonds–Karp), com saturação das arestas, "
        "residual e métricas do fluxo.",
        body,
    ))

    graph_png = run_dir / "graph_n10.png"
    flow_png = run_dir / "flow_n10.png"
    residual_png = run_dir / "residual_n10.png"
    saturation_png = run_dir / "saturation_n10.png"
    flow_nodes_png = run_dir / "flow_nodes_n10.png"

    row_n10 = next((r for r in rows1 if r.n == 10), None) if rows1 else None
    matrix_path = (run_dir / f"matrix_n10_seed{row_n10.seed}.csv") if row_n10 else None
    if matrix_path is not None and matrix_path.exists():
        g = load_capacity_matrix_csv(matrix_path)
        cap = g.capacities
        succ = g.successor_lists(include_residual_reverse=True)
        result = max_flow_edmonds_karp_succ_bfs(cap, succ, s=0, t=g.n - 1)
        flow = np.maximum(0.0, cap - result.residual)
        residual = np.maximum(0.0, result.residual)

        _plot_directed_graph(cap, graph_png, "Grafo capacitado (n=10)")
        _plot_directed_graph(cap, flow_png, "Fluxo nas arestas (n=10)", edge_values=flow, edge_color="#d62728")
        _plot_residual_heatmap(residual, cap, residual_png)
        _plot_directed_graph_saturation(cap, flow, saturation_png, "Saturação das arestas (n=10)")
        _plot_flow_node_balance(flow, flow_nodes_png)

        used_edges, mean_flow, max_flow_edge = _flow_stats(flow)
        story.append(Spacer(1, 0.2*cm))
        story.append(Image(str(graph_png), width=14*cm, height=14*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(flow_png), width=14*cm, height=14*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(residual_png), width=14*cm, height=12*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(saturation_png), width=14*cm, height=14*cm))
        story.append(Spacer(1, 0.3*cm))
        story.append(Image(str(flow_nodes_png), width=16*cm, height=8*cm))
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
    if not rows2:
        story.append(Paragraph(
            "Não há resultado da Parte 2 nesta pasta. Para gerar, rode: "
            "<b>python run.py prompt</b> e depois execute <b>python run.py report</b> novamente.",
            body
        ))
    else:
        story.append(Paragraph(
            "Resultados da IA generativa e comparação com os algoritmos locais:",
            body
        ))
        story.append(Spacer(1, 0.2*cm))

        header = [
            "n", "modelo", "tempo LLM (s)", "fluxo LLM",
            "fluxo FF", "fluxo EK",
            "erro abs (EK)", "erro % (EK)",
            "LLM/FF (tempo)", "LLM/EK (tempo)",
        ]
        body_rows = []
        for r in rows2:
            row1 = next((x for x in rows1 if x.n == r.n), None)
            ff_flow = row1.ff_pred_dfs_flow if row1 else 0.0
            ek_flow = row1.ek_succ_bfs_flow if row1 else 0.0
            abs_err = abs(r.llm_max_flow - ek_flow) if row1 else 0.0
            rel_err = (abs_err / ek_flow * 100.0) if (row1 and ek_flow > 0) else 0.0
            llm_ff_ratio = (r.llm_seconds / row1.ff_pred_dfs_mean) if (row1 and row1.ff_pred_dfs_mean > 0) else 0.0
            llm_ek_ratio = (r.llm_seconds / row1.ek_succ_bfs_mean) if (row1 and row1.ek_succ_bfs_mean > 0) else 0.0
            body_rows.append([
                str(r.n),
                r.model,
                f"{r.llm_seconds:.2f}",
                f"{r.llm_max_flow:.2f}",
                f"{ff_flow:.2f}",
                f"{ek_flow:.2f}",
                f"{abs_err:.2f}",
                f"{rel_err:.2f}%",
                f"{llm_ff_ratio:.2f}x",
                f"{llm_ek_ratio:.2f}x",
            ])

        t = Table([header] + body_rows, colWidths=[1.0*cm, 3.2*cm, 2.2*cm, 2.0*cm, 1.8*cm, 1.8*cm, 2.0*cm, 2.0*cm, 2.0*cm, 2.0*cm])
        t.setStyle(TableStyle([
            ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
            ("BACKGROUND", (0,0), (-1,0), colors.lightgrey),
            ("ALIGN", (0,0), (-1,-1), "CENTER"),
            ("GRID", (0,0), (-1,-1), 0.5, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 8),
        ]))
        story.append(t)
        story.append(Spacer(1, 0.4*cm))
        story.append(Paragraph(
            "Obs.: a Parte 2 é manual; o foco é registrar tempo de resposta e permitir comparação direta "
            "com as implementações locais (Parte 1). Modelo utilizado: <b>ChatGPT 5.2 Thinking</b>.",
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
