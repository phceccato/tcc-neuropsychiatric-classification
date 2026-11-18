# Resultados exportados

Previsões fora do fold de todos os modelos do TCC (SZ × HC e MDD × HC), para recalcular as métricas e os testes sem
treinar nada. Gerado por `python -m pipeline.e10_resultados.exportar`; o método está em docs/10_resultados.md.

| Arquivo | Conteúdo |
| --- | --- |
| `previsoes/<dataset>/<modelo>.csv` | uma linha por (repetição, fold, pessoa de teste): `y_true` (1 = paciente), `y_prob` (probabilidade de paciente), `y_pred` (`y_prob` ≥ 0,5) |
| `perdas/<dataset>/DTSF_FA.csv` | perda de treino e de validação interna por época, em cada (repetição, fold), do modelo principal; `melhor` marca a época escolhida |
| `folds.csv` | as partições: 5 folds × 10 repetições, por pessoa, com o papel (treino, validação, teste) |
| `metricas.csv` | AUC por pessoa com IC de 95%; acurácia, acurácia balanceada, precisão, sensibilidade (recall), especificidade, F1 e Macro F1 no limiar 0,5; e a média ± desvio de todas nas 10 repetições, recalculadas destes arquivos |
| `resumos/` | os resumos das etapas 06, 07 e 08, com as comparações entre modelos |
| `manifesto.json` | semente, commit do código, dispositivo e parâmetros de cada modelo, modelos incompletos |

AUC por pessoa: média de `y_prob` de cada pessoa nas 10 repetições, e uma AUC sobre as pessoas:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
p = pd.read_csv("previsoes/SZ/DTSF_FA.csv")
pessoa = p.groupby("subject_uid").agg(y=("y_true", "first"), prob=("y_prob", "mean"))
print(roc_auc_score(pessoa["y"], pessoa["prob"]))
```
