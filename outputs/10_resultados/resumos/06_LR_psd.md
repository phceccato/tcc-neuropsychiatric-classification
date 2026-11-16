# LR_psd — resultados

Regressão logística L2 (C = 1), com peso de classe, sobre a média das features por pessoa; z-score com estatísticas do treino de cada fold; 5 folds × 10 repetições. IC de 95% por bootstrap estratificado (2000 reamostragens). Sem teste contra o acaso.

| Conjunto | Features | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SZ | absolutas | 0.724 [0.520, 0.903] | 0.717 ± 0.046 | 0.714 | 0.786 | 0.643 | 0.713 |
| SZ | relativas | 0.668 [0.459, 0.857] | 0.664 ± 0.056 | 0.571 | 0.643 | 0.500 | 0.569 |
| MDD | absolutas | 0.868 [0.761, 0.959] | 0.859 ± 0.021 | 0.803 | 0.828 | 0.778 | 0.803 |
| MDD | relativas | 0.879 [0.774, 0.962] | 0.875 ± 0.011 | 0.838 | 0.862 | 0.815 | 0.839 |

Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.

| Conjunto | Features | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | Especificidade | F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SZ | absolutas | 0.717 ± 0.046 | 0.654 ± 0.048 | 0.654 ± 0.048 | 0.637 ± 0.040 | 0.714 ± 0.075 | 0.593 ± 0.059 | 0.672 ± 0.051 | 0.652 ± 0.048 |
| SZ | relativas | 0.664 ± 0.056 | 0.607 ± 0.069 | 0.607 ± 0.069 | 0.611 ± 0.079 | 0.614 ± 0.060 | 0.600 ± 0.113 | 0.611 ± 0.060 | 0.606 ± 0.070 |
| MDD | absolutas | 0.859 ± 0.021 | 0.802 ± 0.026 | 0.801 ± 0.026 | 0.800 ± 0.035 | 0.824 ± 0.020 | 0.778 ± 0.046 | 0.812 ± 0.022 | 0.801 ± 0.026 |
| MDD | relativas | 0.875 ± 0.011 | 0.816 ± 0.027 | 0.815 ± 0.027 | 0.811 ± 0.030 | 0.841 ± 0.029 | 0.789 ± 0.039 | 0.826 ± 0.025 | 0.815 ± 0.027 |

A potência relativa é a checagem do ganho do aparelho no MDD (docs/04_features.md). Ela não elimina diferenças de ritmo entre sessões de gravação: ver docs/06_LR_psd.md e checagem_lote.py.
