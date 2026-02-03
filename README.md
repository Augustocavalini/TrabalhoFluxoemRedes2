# Segundo Trabalho de Fluxo em Redes — Problema do Fluxo Máximo

Este projeto implementa **duas simulações** para computar **Fluxo Máximo** entre `X1` (origem) e `Xn` (destino),
conforme solicitado no PDF do trabalho:

- **Simulação 1 (lista de antecessores + DFS alvo)**: Ford–Fulkerson usando busca em profundidade _alvo_ no **grafo residual**,
  percorrendo **lista de antecessores**.
- **Simulação 2 (lista de sucessores + BFS alvo)**: Edmonds–Karp (Ford–Fulkerson com BFS) no residual,
  percorrendo **lista de sucessores**.
- **Parte 2 (linguagem generativa)**: o projeto gera um **prompt** (vértices + lista de arcos com capacidades) para você colar
  em uma IA generativa e retorna um script **interativo** para medir o tempo de resposta e comparar com as duas implementações.

> Grafo direcionado com **matriz de capacidades** (arcos inexistentes = 0), vértices `X1..Xn`, grafo denso sem laços,
> simulações com `n=10` e `n=100`, e análise comparativa das 3 soluções.

## Como rodar (rápido)

```bash
# 1) Crie ambiente (opcional)
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate

# 2) Instale dependências
pip install -r requirements.txt

# 3) Rodar as simulações automáticas (n=10 e n=100) + benchmark
python run.py simulate
```

Os resultados serão salvos em `outputs/` (CSV/JSON) e também mostrados no console.

## Rodar com uma matriz de capacidade (CSV)

Crie um CSV `N x N` (sem cabeçalho também funciona). Exemplo `examples/matrix_6.csv`.

```bash
python run.py from-matrix examples/matrix_6.csv
```

## Parte 2 — Prompt para IA generativa (manual)

O PDF pede que o prompt solicite **somente** o valor do fluxo máximo, fornecendo os vértices e os arcos com capacidades.

1. Gere o prompt:

```bash
python run.py prompt --n 10 --seed 42
```

2. Copie o prompt gerado e cole em uma IA generativa.
3. Volte no terminal e cole a resposta quando o script pedir. Ele mede o tempo e compara com as duas soluções.

## Estrutura

- `src/graph.py` — leitura/geração do grafo + listas de antecessores/sucessores
- `src/maxflow_pred_dfs.py` — Simulação 1 (antecessores + DFS alvo)
- `src/maxflow_succ_bfs.py` — Simulação 2 (sucessores + BFS alvo)
- `src/benchmark.py` — roda n=10 e n=100, mede tempos e salva resultados
- `src/part2_prompt.py` — gera prompt e roda modo interativo (tempo + parse do valor)
- `tests/` — teste pequeno de corretude

## Observações

- As implementações trabalham com **grafo residual** e garantem não usar laços (i->i).
- Em grafos densos aleatórios, o fluxo máximo pode variar bastante; usamos `seed` para reprodutibilidade.

Boa implementação e bons testes 🙂

## Gerar relatório PDF

Depois de rodar os testes:

```bash
python run.py simulate
```

Gere o PDF a partir do último run em `outputs/`:

```bash
python run.py report
```

Ou apontando para uma pasta específica:

```bash
python run.py report --run-dir outputs/run_YYYY-MM-DD_HHMMSS
```

O relatório inclui estatísticas detalhadas (min/mediana/máximo, desvio padrão, CV e p95),
além de boxplot e histogramas dos tempos.

#### Resumo dos algoritmos utilizados

**Ford–Fulkerson** :

1. Começa com fluxo 0 e residual = capacidade original.
2. Encontra um caminho aumentante (qualquer caminho de X1 a Xn no residual com capacidade > 0) usando DFS.
3. Calcula o gargalo: a menor capacidade no caminho.
4. Atualiza o residual: subtrai o gargalo na direção direta, soma na reversa.
5. Adiciona o gargalo ao fluxo total.
6. Repete até não existir caminho aumentante.

Desvantagem: pode ser lento em redes grandes porque não escolhe caminhos inteligentemente.

**Edmonds–Karp** :

1. Idêntico ao Ford–Fulkerson, mas **sempre escolhe o caminho aumentante mais curto** usando BFS.
2. Isso garante que cada aresta é usada no máximo O(n) vezes.
3. Tempo total: O(n × m²), onde n = nós, m = arestas.

Vantagem: complexidade garantida polinomial, muito mais eficiente em redes grandes.

#### Resumo das condições de fluxo máximo:

- **Capacidade** : cada aresta tem limite; o fluxo nela não pode passar desse valor.
- **Conservação** : em nós intermediários, tudo que entra deve sair (fluxo líquido zero).
- **Origem/Destino** : X1 só envia; Xn só recebe.
- **Fluxo máximo** : é o maior total possível de X1 para Xn, limitado pelos gargalos da rede, não por “encher” todas as arestas do destino.
  - Gargalos são partes da rede que limitam o total que pode passar. São arestas ou conjuntos de arestas com pouca capacidade que “estrangulam” o fluxo.
    Formalmente, é o **corte mínimo** : a soma das capacidades das arestas que, se removidas, separariam X1 de Xn. Esse valor é o limite do fluxo máximo.
