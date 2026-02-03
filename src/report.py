from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from datetime import datetime
import json

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet

import matplotlib.pyplot as plt


@dataclass
class Part1Row:
    n: int
    density: float
    seed: int
    ff_pred_dfs_flow: float
    ff_pred_dfs_seconds: float
    ek_succ_bfs_flow: float
    ek_succ_bfs_seconds: float
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
        rows.append(Part1Row(
            n=int(r["n"]),
            density=float(r["density"]),
            seed=int(r["seed"]),
            ff_pred_dfs_flow=float(r["ff_pred_dfs_flow"]),
            ff_pred_dfs_seconds=float(r["ff_pred_dfs_seconds"]),
            ek_succ_bfs_flow=float(r["ek_succ_bfs_flow"]),
            ek_succ_bfs_seconds=float(r["ek_succ_bfs_seconds"]),
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
        "FF (pred+DFS)\nfluxo", "FF (pred+DFS)\ntempo (s)",
        "EK (succ+BFS)\nfluxo", "EK (succ+BFS)\ntempo (s)",
        "igual?"
    ]
    body = []
    for r in rows:
        body.append([
            str(r.n),
            f"{r.density:.2f}",
            str(r.seed),
            f"{r.ff_pred_dfs_flow:.2f}",
            f"{r.ff_pred_dfs_seconds:.6f}",
            f"{r.ek_succ_bfs_flow:.2f}",
            f"{r.ek_succ_bfs_seconds:.6f}",
            "sim" if r.equal_flows else "não",
        ])
    table = Table([header] + body, colWidths=[1.1*cm, 2.0*cm, 1.4*cm, 2.6*cm, 2.8*cm, 2.6*cm, 2.8*cm, 1.5*cm])
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


def _plot_times(rows: List[Part1Row], out_png: Path) -> None:
    ns = [r.n for r in rows]
    ff = [r.ff_pred_dfs_seconds for r in rows]
    ek = [r.ek_succ_bfs_seconds for r in rows]

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
    speedup = [_safe_div(r.ff_pred_dfs_seconds, r.ek_succ_bfs_seconds) for r in rows]

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

    story.append(Paragraph("2. Resultados — Parte 1 (2 simulações)", h2))
    story.append(Paragraph("Tabela consolidada dos testes automáticos:", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(_make_part1_table(rows1))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.1 Tempo de execução", h2))
    story.append(Paragraph("Comparativo de tempo por n:", body))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(png_times), width=16*cm, height=9*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.2 Razão de tempo (FF/EK)", h2))
    story.append(Paragraph(
        "Valores maiores que 1 indicam que Ford–Fulkerson (pred+DFS) foi mais lento que Edmonds–Karp (succ+BFS).",
        body
    ))
    story.append(Spacer(1, 0.2*cm))
    story.append(Image(str(png_speedup), width=16*cm, height=9*cm))
    story.append(Spacer(1, 0.4*cm))

    story.append(Paragraph("2.3 Observações", h2))
    obs_lines = []
    for r in rows1:
        ratio = _safe_div(r.ff_pred_dfs_seconds, r.ek_succ_bfs_seconds)
        obs_lines.append(
            f"Para n={r.n}, fluxos coincidem: {'sim' if r.equal_flows else 'não'}; "
            f"FF={r.ff_pred_dfs_seconds:.6f}s, EK={r.ek_succ_bfs_seconds:.6f}s, FF/EK={ratio:.2f}."
        )
    story.append(Paragraph("<br/>".join(obs_lines), body))

    story.append(PageBreak())
    story.append(Paragraph("3. Parte 2 — Linguagem Generativa (manual)", h2))
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
