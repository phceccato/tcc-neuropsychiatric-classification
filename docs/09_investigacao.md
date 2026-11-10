# Investigação dos erros e do confundidor de data — resultados

Quatro análises pós-hoc sobre o DTSF, pedidas na conversa com o orientador: a data de gravação contra o diagnóstico, as pessoas que o D classifica errado, explicações SHAP e a influência da sessão de gravação. **Nenhuma delas testa hipótese**: são descritivas, e todas foram escolhidas depois de ver os resultados.

Resultado que não depende da semente:
- **A data de gravação, sozinha, separa as classes:** AUC de 0,929 na SZ e 0,928 no MDD. Na SZ, nenhuma sessão de gravação tem as duas classes.

Os resultados das seções 2–4 dependem dos modelos treinados e estão pendentes da rodada com a semente 20261108.

- Código: [pipeline/e09_investigacao/](../pipeline/e09_investigacao/)
  - `datas.py`: experimento 1;
  - `erros.py`: experimento 2;
  - `explicar.py` e `relatorio_explicacao.py`: experimento 3 (retreino com pesos gravados, SHAP e resumo);
  - `lote.py`: experimento 4.
- Testes: `tests/e09_investigacao/`.
- Saídas em `outputs/09_investigacao/`: `datas/`, `erros/`, `explicacao/` e `lote/`, cada uma com `resumo.md`; logs `log_explicacao.txt` e `log_lote.txt`.
- Execução sobre as saídas da rodada completa.

```bash
venv/bin/python -m pipeline.e09_investigacao.datas
venv/bin/python -m pipeline.e09_investigacao.explicar 2>&1 | tee -a outputs/09_investigacao/log_explicacao.txt   # 10 retreinos + SHAP (~15 min com 10 processos); retoma
venv/bin/python -m pipeline.e09_investigacao.relatorio_explicacao
venv/bin/python -m pipeline.e09_investigacao.erros          # usa as janelas do explicar, se existirem
venv/bin/python -m pipeline.e09_investigacao.lote 2>&1 | tee -a outputs/09_investigacao/log_lote.txt               # subconjuntos + 50 retreinos; retoma
```

Pré-requisitos: as etapas 02, 03, 05, 06, 07 (com os caches e as previsões do D) e 08.

## 1. Data de gravação × diagnóstico

O EDF não informa lote nem sessão. A sessão foi reconstruída pela data do cabeçalho: datas separadas por mais de 30 dias abrem uma sessão nova.

| Dataset | Sessões | V de Cramér (grupo × sessão) | AUC da data como escore de paciente | HC em sessões mistas | Pacientes em sessões mistas |
| --- | --- | --- | --- | --- | --- |
| SZ | 9 | 1,00 | 0,929 | 0 de 14 | 0 de 14 |
| MDD | 3 | 0,87 | 0,928 | 4 de 27 | 29 de 29 |

| Dataset | Sessão | Período | HC | Pacientes |
| --- | --- | --- | --- | --- |
| SZ | 1 | 11/03/2003 | 0 | 1 (s10) |
| SZ | 2 | 18/06–25/07/2003 | 14 | 0 |
| SZ | 3 | 20–23/10/2003 | 0 | 2 (s06, s08) |
| SZ | 4–9 | 08/2004 a 07/2008 | 0 | 11 |
| MDD | 1 | 23/07–17/08/2011 | 9 | 0 |
| MDD | 2 | 12–14/01/2013 | 14 | 0 |
| MDD | 3 | 07/04–30/07/2013 | 4 | 29 |

A tabela por ano e mês está em `outputs/09_investigacao/datas/resumo.md`.

- **SZ:** classe e sessão coincidem por completo. Um classificador que só soubesse a data teria AUC 0,929.
- **MDD:** só a sessão de abr–jul/2013 tem as duas classes, com 4 HC (S16, S24, S25, S27).
- **Validação por sessão** (deixar uma sessão de fora) não é possível em nenhum dos dois datasets: quase todas as sessões têm uma classe só, e a sessão deixada de fora não teria as duas classes para calcular a AUC.

## 2. Erros individuais

Para cada pessoa: a previsão do D nas 10 repetições, a qualidade do sinal, a sessão e o espectro (Welch nas janelas aceitas sem overlap, média dos 19 canais).
Pessoas em foco: s06 (SZ), h10 (HC da SZ) e MDD S27, com o espectro comparado ao das duas classes (figuras `erros/espectros.png` e `erros/janelas.png`).

**Resultados:** pendentes da rodada com a semente 20261108.

## 3. Explicações SHAP

**Como.** As previsões da etapa 07 não guardam os pesos, então o D foi retreinado nos 5 folds da repetição 0 de cada dataset, com as mesmas partições e sementes, e os pesos foram gravados. Para cada janela de teste aceita sem overlap, os valores SHAP do logit foram calculados por *expected gradients* (ERION et al., 2021), o algoritmo do `shap.GradientExplainer`, implementado em PyTorch para não trazer a biblioteca `shap` (que depende do `numba` e pode exigir outra versão do numpy). O fundo tem 100 janelas de treino de cada classe, e cada janela usa 50 amostras. φ > 0 empurra para paciente.

**Conferências:** a previsão da pessoa no modelo retreinado contra a da etapa 07 (repetição 0) e a completude (soma dos φ contra f(x) − E[f(fundo)]).

**Grupos.** Cada pessoa de teste entra num grupo pela previsão do próprio modelo retreinado (HC certo, HC errado, paciente certo, paciente errado). Antes da média do grupo, o perfil de cada pessoa é normalizado para somar 1.

Figuras: `explicacao/frequencias.png`, `explicacao/canais.png` (topografia) e `explicacao/tempo.png`.

**Resultados:** pendentes da rodada com a semente 20261108.

## 4. Influência da sessão de gravação

### Subconjuntos, sem retreino

As previsões fora do fold já gravadas (D da etapa 07, os 9 baselines da etapa 08 e a LR de 76 bandas da etapa 06) são reavaliadas só nas pessoas de cada subconjunto em que a data se confunde menos com o diagnóstico. Os modelos continuam treinados com todas as pessoas. AUC por pessoa (média das 10 repetições fora do fold); o IC de 95% (bootstrap estratificado) é só do D.

- **SZ:** os 3 pacientes gravados em 2003 contra os 14 HC (todos de 2003), e os 11 pacientes de 2004–2008 contra os mesmos HC.
- **MDD:** sem o lote de jan/2013; só a sessão de abr–jul/2013, a única com as duas classes (4 HC); e cada sessão de HC separada contra todos os pacientes.

### Retreino do D no MDD sem o lote de jan/2013

Partições novas 5 × 10, com o mesmo procedimento de [05_folds.md](05_folds.md), nas 42 pessoas restantes (13 HC e 29 MDD; 8–9 pessoas de teste por fold), e o D treinado do zero com o código da etapa 07: 50 treinos, ~40 min com 10 processos.

Na SZ não há retreino equivalente: os 14 HC estão numa sessão só.

**Resultados:** pendentes da rodada com a semente 20261108.
