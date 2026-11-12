# LR_psd — regressão logística sobre a PSD em bandas: resultados

A regressão logística sobre as 76 features de banda fica **acima do acaso nos dois transtornos**. As AUCs por pessoa são **0,78 na SZ** e **0,87 no MDD**. Na SZ, o IC de 95% vai de 0,59 a 0,92.

- Código: [pipeline/e06_LR_psd/lr.py](../pipeline/e06_LR_psd/lr.py), com as métricas de [avaliacao.md](avaliacao.md)
- Duração: ~4 min, quase todo gasto nas 1000 permutações
- Saídas em `outputs/06_LR_psd/`:
  - `predicoes.csv`: probabilidade fora do fold de cada pessoa, em cada repetição;
  - `permutacoes.csv`: as AUCs nulas;
  - `resumo.md`.

```bash
venv/bin/python -m pipeline.e06_LR_psd.lr
```

## Modelo

- **Classificador:** regressão logística L2 com C = 1 e peso de classe (`class_weight="balanced"`).
- **Entrada:** média das features nas janelas aceitas de cada pessoa. As pessoas de treino usam a grade com 50% de overlap, e as de teste a grade sem overlap.
- **Normalização:** z-score por feature, com estatísticas só do treino de cada fold.
- **Validação:** as partições de [05_folds.md](05_folds.md), 5 folds × 10 repetições.
- **Teste contra o acaso:** 1000 permutações dos rótulos das pessoas, com retreino completo, nas partições da repetição 0.

## Resultados

| Conjunto | Features | AUC por pessoa [IC 95%] | AUC por repetição | p | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SZ | absolutas | 0,781 [0,592; 0,923] | 0,761 ± 0,042 | 0,010 | 0,714 | 0,786 | 0,643 | 0,713 |
| SZ | relativas | 0,704 [0,495; 0,888] | 0,696 ± 0,037 | — | 0,679 | 0,714 | 0,643 | 0,678 |
| MDD | absolutas | 0,865 [0,747; 0,958] | 0,858 ± 0,015 | 0,001 | 0,821 | 0,828 | 0,815 | 0,821 |
| MDD | relativas | 0,884 [0,778; 0,964] | 0,884 ± 0,010 | — | 0,857 | 0,862 | 0,852 | 0,857 |

Como ler a tabela:
- **AUC por pessoa:** calculada sobre a média da probabilidade de cada pessoa nas 10 repetições. O IC vem de um bootstrap estratificado com 2000 reamostragens.
- **AUC por repetição:** média ± desvio das 10 AUCs.
- **p:** com 1000 permutações, o menor p possível é 1/1001 ≈ 0,001.
- **Secundárias:** calculadas no limiar 0,5 e só descritivas.

## Leitura

- **SZ:** o resultado fica acima do acaso (p = 0,010), mas o IC é largo, como o protocolo previa para 28 pessoas.
- **MDD, potência relativa:** ela vai tão bem quanto a absoluta (0,884 contra 0,865). Pela checagem da proposta, nada indica que o ganho do amplificador explique o resultado.
- **Uso:** é a referência mais simples do projeto. O DTSF está em [07_DTSF.md](07_DTSF.md), e os baselines clássicos sobre as entradas do DTSF em [08_classicos.md](08_classicos.md).
