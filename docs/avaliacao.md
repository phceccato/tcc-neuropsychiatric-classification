# Avaliação — métricas e testes

Esta pasta guarda as funções de métrica e de teste estatístico usadas pelo DTSF, pelos baselines clássicos e pela LR de 76 bandas. Ela não tem execução própria: os resultados aparecem em [07_DTSF.md](07_DTSF.md), [08_classicos.md](08_classicos.md) e [06_LR_psd.md](06_LR_psd.md).

- Código: [pipeline/avaliacao/metricas.py](../pipeline/avaliacao/metricas.py)
- Testes: `tests/avaliacao/test_metricas.py`, que confere, entre outras coisas, a AUC vetorizada contra o scikit-learn, com empates, e `tests/avaliacao/test_premissas_ranks.py`, com as simulações que sustentam o teste pareado com ranks

## O que cada função implementa

| Função | O que faz | Onde é usada |
| --- | --- | --- |
| `escores_por_pessoa` | Média da probabilidade fora do fold de cada pessoa nas repetições | Todas as avaliações |
| `avaliar` | AUC por pessoa, IC, AUC por repetição (média ± desvio) e secundárias | Tabelas de resultado |
| `ic_bootstrap` | IC de 95% da AUC, por percentil, reamostrando pessoas dentro de cada classe (2000 vezes) | Incerteza da AUC |
| `metricas_secundarias` | Acurácia balanceada, sensibilidade, especificidade e Macro F1 no limiar 0,5 | Métricas descritivas |
| `p_valor` | (nº de nulas ≥ observado + 1) / (N + 1) | Testes contra o acaso |
| `auc_em_lote` | AUC de Mann-Whitney vetorizada, com empate valendo ½ | Base dos testes por permutação |
| `teste_pareado` | ΔAUC = AUC(a) − AUC(b) e p unilateral, trocando os dois escores de cada pessoa ao acaso (10.000 vezes) | Base do teste com ranks |
| `ic_bootstrap_delta` | IC de 95% do ΔAUC, reamostrando pessoas e mantendo os pares | IC das comparações e das ablações |
| `parear` | Alinha os escores por pessoa de dois modelos e recusa pessoas ausentes, repetidas, rótulos divergentes ou escores inválidos | Comparações pareadas |
| `p_interseccao_uniao` | p = maior p das comparações componentes (BERGER, 1982) | H2 e H4: decisão num transtorno |
| `holm` | Correção de Holm (step-down) | H2–H4: decisão entre SZ e MDD |
| `delta_por_repeticao`, `delta_por_fold` | ΔAUC de cada repetição e de cada (repetição, fold) | Sensibilidades |
| `t_corrigido` | t corrigido de Nadeau e Bengio (2003) sobre os ΔAUC por fold | Sensibilidade |

## Hipóteses H1–H4

As comparações entre dois modelos usam `comparacao_pareada` ([pipeline/e07_DTSF/avaliar.py](../pipeline/e07_DTSF/avaliar.py)):
- `teste_ranks` transforma os escores de cada modelo em ranks entre as pessoas e aplica a permutação pareada (`teste_pareado`);
- `comparacao_pareada` junta a isso o IC do Δ e as sensibilidades.

H1, H2 e H3 estão em [pipeline/e07_DTSF/avaliar.py](../pipeline/e07_DTSF/avaliar.py):
- H1: p de permutação contra o acaso em cada transtorno, a α = 0,05, sem Holm (`decidir_h1`);
- H2 e H3: `comparacao_pareada(D, ablação)`, com ΔAUC = D − ablação; na H2, interseção-união sobre `D_sem_T` e `D_sem_E`; Holm entre SZ e MDD (`decidir_hipotese`).

A H4 (D contra os nove baselines clássicos: LR, RF e KNN × temporal, espectral e combinado) está em [pipeline/e08_classicos/avaliar.py](../pipeline/e08_classicos/avaliar.py):
- interseção-união das nove comparações em cada transtorno;
- Holm entre SZ e MDD.
