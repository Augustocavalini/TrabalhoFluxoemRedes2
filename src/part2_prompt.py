from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import re

from .graph import CapacitatedDigraph, generate_dense_random_graph, label


@dataclass
class LLMManualResult:
    n: int
    seed: int
    density: float
    llm_seconds: float
    llm_max_flow: float


def build_prompt(vertices: List[str], arcs: List[Tuple[str, str, float]]) -> str:
    """
    Prompt alinhado com o PDF:
      - pedir SOMENTE para computar o fluxo máximo X1->Xn
      - fornecer conjunto de vértices e lista de arcos com capacidades
    """
    lines = []
    lines.append("Tarefa: compute SOMENTE o valor do FLUXO MÁXIMO do grafo direcionado capacitado abaixo.")
    lines.append("Origem: X1")
    lines.append(f"Destino: {vertices[-1]}")
    lines.append("")
    lines.append("Vértices:")
    lines.append(", ".join(vertices))
    lines.append("")
    lines.append("Arcos (u -> v : capacidade):")
    for u, v, c in arcs:
        lines.append(f"- {u} -> {v} : {int(c) if c.is_integer() else c}")
    lines.append("")
    lines.append("Responda apenas com um número (o valor do fluxo máximo). Sem explicações.")
    return "\n".join(lines)


def _parse_first_number(s: str) -> Optional[float]:
    m = re.search(r"[-+]?\d+(\.\d+)?", s)
    if not m:
        return None
    return float(m.group(0))


def run_manual_llm_flow(
    n: int = 10,
    density: float = 0.55,
    seed: int = 42,
) -> LLMManualResult:
    """
    Parte 2 (manual): gera prompt, inicia cronômetro e pede para o usuário colar a resposta.
    """
    g: CapacitatedDigraph = generate_dense_random_graph(n=n, density=density, seed=seed)
    vertices = [label(i) for i in range(n)]
    arcs = [(label(u), label(v), c) for (u, v, c) in g.arcs()]

    prompt = build_prompt(vertices, arcs)
    print("=" * 90)
    print("COPIE o prompt abaixo e cole na sua IA generativa (ChatGPT, etc.).")
    print("Depois, volte aqui e cole SOMENTE a resposta numérica (ou a primeira linha com número).")
    print("=" * 90)
    print(prompt)
    print("=" * 90)

    input("Quando estiver pronto(a) para iniciar o tempo (você vai colar o prompt na IA), pressione ENTER... ")
    t0 = time.perf_counter()
    answer = input("Cole aqui a resposta da IA (ex: 42) e pressione ENTER: ").strip()
    t1 = time.perf_counter()

    val = _parse_first_number(answer)
    if val is None:
        raise ValueError("Não consegui extrair um número da resposta colada. Tente novamente.")

    return LLMManualResult(
        n=n,
        seed=seed,
        density=density,
        llm_seconds=float(t1 - t0),
        llm_max_flow=float(val),
    )
