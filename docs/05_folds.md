# Partições da validação cruzada — resultados

As partições de 5 folds × 10 repetições, separadas por pessoa, foram geradas e salvas antes de qualquer treino. O DTSF, os baselines clássicos e a LR de 76 bandas usam as mesmas.

- Código: [pipeline/e05_folds/folds.py](../pipeline/e05_folds/folds.py)
- Entrada: `outputs/02_harmonizacao/manifesto.csv`
- Saída: `outputs/05_folds/folds.csv`, com uma linha por (dataset, repetição, fold, pessoa) e o papel da pessoa: `treino`, `validacao` ou `teste`

```bash
venv/bin/python -m pipeline.e05_folds.folds
```

## Como as partições são geradas

- **Folds:** `StratifiedGroupKFold` com k = 5, agrupado por `subject_uid` e estratificado pela classe. São 10 repetições, cada uma com semente `SEMENTE + r`.
- **Validação interna:** dentro do treino de cada fold, round(0,2 × pessoas de treino), estratificadas, formam o split do early stopping, com semente `SEMENTE + 1000·r + fold`. A LR (06) e os baselines clássicos (08) não usam early stopping e treinam com treino e validação juntos.
- **Datasets separados:** cada dataset tem as suas partições.

## Tamanhos

| Conjunto | Pessoas (pacientes + HC) | Teste por fold | HC no teste | Validação | Treino |
| --- | --- | --- | --- | --- | --- |
| SZ | 28 (14 + 14) | 5–6 | 2–3 | 4–5 | 18 |
| MDD | 56 (29 + 27) | 11–12 | 5–6 | 9 | 35–36 |

## Sementes

| Constante | Onde | Valor |
| --- | --- | --- |
| `SEMENTE` | `e05_folds/folds.py` (partições), `e06_LR_psd/lr.py`, `e07_DTSF/rodar.py` (treino do DTSF e dos clássicos da 08), `e07_DTSF/avaliar.py` (bootstrap e permutações pareadas) | 20261108 |
| `SEMENTE_PERMUTACAO` | `e07_DTSF/rodar.py` (permutações de rótulo da H1) | 1108000 |
| `SEMENTE_SHAP` | `e09_investigacao/explicar.py` | 1108 |
