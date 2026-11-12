# DTSF com as dimensões do artigo — teste exploratório

O D da etapa 07 tem 80.449 parâmetros. A proposta reduziu o tamanho do artigo original (WANG et al., 2026) por dois motivos: o custo do treino em CPU e o n de 28 e 56 pessoas. Com GPU, o primeiro motivo deixa de valer, e esta etapa testa o segundo: o **D_artigo** é o D com as dimensões do artigo (702.721 parâmetros) e todo o resto igual.

**É exploratório.** As hipóteses H1–H4 foram registradas com o D, e o D continua sendo o modelo principal. O D_artigo é comparado com o D, mas os p não decidem nada. A configuração foi fixada antes de rodar, sem busca e sem olhar o teste.

Os resultados ficam em [Resultados](#resultados).

- Código: [pipeline/e14_DTSF_artigo/](../pipeline/e14_DTSF_artigo/)
  - `rodar.py`: a configuração `ARTIGO` e o runner, com retomada, permutações e `--dispositivo`;
  - `avaliar.py`: as AUCs e a comparação D_artigo − D.
- A rede e o treino são os da etapa 07: `DTSF` e `ConfigTreino` passaram a aceitar as dimensões (filtros, oculto, cabeças, blocos), com os valores do D como padrão.
- Testes: `tests/e14_DTSF_artigo/` (7 testes).
- Saídas: `outputs/14_DTSF_artigo/resultados/<dataset>/D_artigo/r<rep>_f<fold>.{csv,json}` e `resumo.md`; `outputs/14_DTSF_artigo/permutacoes/`, se as permutações forem rodadas.

Pré-requisitos: as etapas 03 e 05, os caches da 07 e as previsões completas do D em `outputs/07_DTSF/resultados`.

```bash
venv/bin/python -m pipeline.e14_DTSF_artigo.rodar --dispositivo cuda --processos 14 2>&1 | tee -a outputs/14_DTSF_artigo/log_treinos.txt   # 100 treinos; retoma
venv/bin/python -m pipeline.e14_DTSF_artigo.rodar --permutacoes 25 --dispositivo cuda --processos 14                                       # opcional: contra o acaso (250 treinos)
venv/bin/python -m pipeline.e14_DTSF_artigo.avaliar                                                                                         # → outputs/14_DTSF_artigo/resultados/resumo.md
```

Use o mesmo dispositivo do D da etapa 07: o runner se recusa a misturar CPU e GPU numa pasta, e o resumo registra o dispositivo do D e do D_artigo e avisa se forem diferentes.

## Dimensões

| Item | D (etapa 07) | D_artigo | Artigo (Tabela 1) |
| --- | --- | --- | --- |
| Filtros das convoluções | 16 | 64 | "Conv2d(64, …)" na coluna de parâmetros; 256 canais na coluna de formas |
| GRU (oculto) e dimensão da fusão | 32 | 128 | 128 |
| Encoder | 2 blocos, 2 cabeças | 2 blocos, 4 cabeças | 2 blocos, 4 cabeças |
| Passos por ramo | 16 | 16 | 116 (iEEG de 60 s) |
| Parâmetros | 80.449 | 702.721 | ordem de 1 milhão |

A Tabela 1 do artigo é ambígua nos filtros: vale a coluna de parâmetros (64). Com 256 filtros, o modelo passaria de 3 milhões de parâmetros. Os kernels, os 16 passos por ramo e as entradas continuam os do D, que foram adaptados ao EEG de escalpo de 19 canais e 4 s.

O treino também continua o do D (Adam, lr 10⁻³, weight decay 10⁻⁴, lotes de 64, até 100 épocas com early stopping de paciência 10, Mixup), e não o do artigo (SGD, lotes de 128, 50 épocas). Assim a única diferença para o D é a capacidade, e a comparação é pareada: mesmas partições, mesmas sementes por (repetição, fold), mesmo sorteio de janelas e mesmo Mixup.

## Validação mínima do fluxo

- Com as dimensões padrão, `DTSF` dá saídas idênticas às de antes da mudança para os quatro modelos da etapa 07 e as variantes da etapa 12, e um fold do D retreinado na CPU (SZ, repetição 0, fold 0) dá probabilidades idênticas às gravadas.
- Um fold do D_artigo na CPU (SZ, repetição 0, fold 0, 1 thread): época escolhida 15, 257 s. O D no mesmo fold: época 1, 48 s.

## Resultados

Ainda não rodado.
