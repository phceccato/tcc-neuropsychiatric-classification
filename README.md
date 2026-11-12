# eeg-multi-input-sz-mdd

Trabalho de Conclusão de Curso — classificação de sinais de EEG de esquizofrenia (SZ) e
transtorno depressivo maior (MDD).

## Dados

Os dados não são versionados. A etapa 00 baixa as duas bases das fontes oficiais para `dados/` e confere
cada arquivo pelo tamanho e pelo MD5 ([docs/00_download-dados.md](docs/00_download-dados.md)):

```bash
venv/bin/python -m pipeline.e00_download_dados.baixar             # baixa (retoma) e confere → dados/
venv/bin/python -m pipeline.e00_download_dados.baixar --conferir  # só confere; imprime a impressão digital de cada base
venv/bin/python -m pytest tests/e00_download_dados                # inclui a conferência de dados/ desta máquina
```

Estrutura esperada:

```
dados/
├── Schizophrenia_dataset/   # h01.edf … h14.edf, s01.edf … s14.edf
└── Depression_dataset/      # "H S<n> <EC|EO|TASK>.edf", "MDD S<n> <EC|EO|TASK>.edf"
```

### Schizophrenia_dataset

- Olejarczyk, E.; Jernajczyk, W. (2017). *EEG in schizophrenia*. RepOD, V1.
  [doi:10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441)
- 14 pacientes com esquizofrenia paranoide + 14 controles saudáveis, repouso de olhos fechados.
- 19 canais (sistema 10-20), 250 Hz, referência entre Fz e Cz (não consta no cabeçalho EDF).

### Depression_dataset

- Mumtaz, W. (2016). *MDD Patients and Healthy Controls EEG Data (New)*. figshare.
  [doi:10.6084/m9.figshare.4244171.v2](https://doi.org/10.6084/m9.figshare.4244171.v2)
- Pacientes com MDD (`MDD`) e controles saudáveis (`H`), em três condições: olhos fechados (`EC`),
  olhos abertos (`EO`) e tarefa P300 (`TASK`).
- 256 Hz, referência nas orelhas interligadas (linked ears).
- Contém arquivos duplicados byte a byte; deduplique antes de qualquer análise.
- Dos 193 arquivos listados, 12 nunca ficaram disponíveis para download (6 deles EC); o download
  completo tem 181 arquivos. Ver [docs/01_auditoria.md](docs/01_auditoria.md).

## Como rodar

Os treinos do DTSF (etapas 07 e 09) rodam na CPU por padrão, como nas rodadas registradas. Numa máquina com
GPU NVIDIA, acrescente `--dispositivo cuda` aos comandos `e07_DTSF.rodar`, `e09_investigacao.explicar` e
`e09_investigacao.lote` ([docs/07_DTSF.md](docs/07_DTSF.md#cpu-ou-gpu)); os resultados ficam próximos, mas não
idênticos aos da CPU. As etapas 06 e 08 (scikit-learn) usam só a CPU: ajuste `--processos` ao número de núcleos.

```bash
python -m venv venv && venv/bin/pip install -r requirements.txt
venv/bin/python -m pytest tests
venv/bin/python -m pipeline.e01_auditoria.auditar_ec        # auditoria das gravações EC → outputs/01_auditoria/
venv/bin/python -m pipeline.e02_harmonizacao.harmonizar     # harmonização das gravações incluídas → outputs/02_harmonizacao/
venv/bin/python -m pipeline.e03_janelas.janelas             # filtro 2–30 Hz, z-score e índice de janelas → outputs/03_janelas/
venv/bin/python -m pipeline.e04_features.psd                # PSD de Welch e as 76 features de banda → outputs/04_features/
venv/bin/python -m pipeline.e05_folds.folds                 # partições 5 × 10 por pessoa → outputs/05_folds/
venv/bin/python -m pipeline.e06_LR_psd.lr           # regressão logística + permutações → outputs/06_LR_psd/
venv/bin/python -m pipeline.e07_DTSF.dados               # cache do sinal z-score a 125 Hz → outputs/07_DTSF/cache/
venv/bin/python -m pipeline.e07_DTSF.dados_dtsf          # cache multibanda do DTSF → outputs/07_DTSF/cache_dtsf/
venv/bin/python -m pipeline.e07_DTSF.rodar 2>&1 | tee -a outputs/07_DTSF/log_dtsf.txt   # os 6 modelos DTSF (retoma) → outputs/07_DTSF/resultados/
venv/bin/python -m pipeline.e07_DTSF.rodar --permutacoes 25 2>&1 | tee -a outputs/07_DTSF/log_dtsf_permutacoes.txt   # D_FA contra o acaso (H1)
venv/bin/python -m pipeline.e07_DTSF.avaliar 2>&1 | tee -a outputs/07_DTSF/log_dtsf_avaliar.txt   # AUCs, ablações e H1–H3 → outputs/07_DTSF/resultados/resumo_dtsf.md
venv/bin/python -m pipeline.e08_classicos.caracteristicas   # 475 + 1102 características por janela → outputs/08_classicos/caracteristicas/
venv/bin/python -m pipeline.e08_classicos.rodar 2>&1 | tee -a outputs/08_classicos/log_classicos.txt   # 9 baselines clássicos (retoma) → outputs/08_classicos/resultados/
venv/bin/python -m pipeline.e08_classicos.avaliar 2>&1 | tee -a outputs/08_classicos/log_classicos_avaliar.txt   # AUCs e H4 (D_FA × 9 baselines) → outputs/08_classicos/resultados/resumo.md
venv/bin/python -m pipeline.e09_investigacao.datas        # data de gravação × diagnóstico → outputs/09_investigacao/datas/
venv/bin/python -m pipeline.e09_investigacao.explicar 2>&1 | tee -a outputs/09_investigacao/log_explicacao.txt   # D retreinado na repetição 0 + SHAP (retoma) → outputs/09_investigacao/explicacao/
venv/bin/python -m pipeline.e09_investigacao.relatorio_explicacao   # resumo e figuras do SHAP → outputs/09_investigacao/explicacao/resumo.md
venv/bin/python -m pipeline.e09_investigacao.erros         # perfil das pessoas classificadas errado → outputs/09_investigacao/erros/
venv/bin/python -m pipeline.e09_investigacao.lote 2>&1 | tee -a outputs/09_investigacao/log_lote.txt   # AUC por subconjunto de sessão + D no MDD sem jan/2013 (retoma) → outputs/09_investigacao/lote/
venv/bin/python -m pipeline.e10_resultados.exportar   # previsões de todos os modelos (y_true, y_pred, y_prob), métricas e resumos → outputs/10_resultados/ (versionada)
venv/bin/python -m pipeline.e11_topomapa.topomapa --pessoa MDD-MDD-S27   # mapas topográficos por banda de uma pessoa → outputs/11_topomapa/
venv/bin/python -m pipeline.e11_topomapa.todos                           # os mesmos mapas para as 84 pessoas → outputs/11_topomapa/<dataset>/<classe>/<sessão>/
venv/bin/python -m pipeline.e12_figuras.h1_nula   # figura da distribuição nula da H1 (pgfplots) → outputs/12_figuras/fig_h1_nula.tex
venv/bin/python -m pipeline.e12_figuras.roc       # curvas ROC por pessoa (pgfplots) → outputs/12_figuras/fig_roc.tex
venv/bin/python -m pipeline.e12_figuras.sessoes   # linha do tempo das gravações (pgfplots) → outputs/12_figuras/fig_sessoes.tex
```

A especificação do experimento está em [docs/00_proposta-tcc.md](docs/00_proposta-tcc.md). Pastas de `docs/` e `outputs/` levam o número da etapa (`01_`…`11_`), e os pacotes de `pipeline/` e `tests/` o mesmo número com o prefixo `e` (`e01_`…), porque um módulo Python não pode começar com dígito; `pipeline/avaliacao` é a biblioteca de métricas usada pelas etapas 06–08 e não tem número. `outputs/` não é versionado, com uma exceção: `outputs/10_resultados/`, com as previsões de todos os modelos, para reproduzir os resultados sem treinar ([docs/10_resultados.md](docs/10_resultados.md)).

## Resultados por etapa

Cada pasta de `pipeline/` tem um doc em `docs/` com os resultados da última execução:

| Etapa | Doc |
| --- | --- |
| `pipeline/e01_auditoria` | [docs/01_auditoria.md](docs/01_auditoria.md) |
| `pipeline/e02_harmonizacao` | [docs/02_harmonizacao.md](docs/02_harmonizacao.md) |
| `pipeline/e03_janelas` | [docs/03_janelas.md](docs/03_janelas.md) |
| `pipeline/e04_features` | [docs/04_features.md](docs/04_features.md) |
| `pipeline/e05_folds` | [docs/05_folds.md](docs/05_folds.md) |
| `pipeline/e06_LR_psd` | [docs/06_LR_psd.md](docs/06_LR_psd.md) |
| `pipeline/e07_DTSF` | [docs/07_DTSF.md](docs/07_DTSF.md) |
| `pipeline/e08_classicos` | [docs/08_classicos.md](docs/08_classicos.md) |
| `pipeline/e09_investigacao` | [docs/09_investigacao.md](docs/09_investigacao.md) |
| `pipeline/e10_resultados` | [docs/10_resultados.md](docs/10_resultados.md) |
| `pipeline/e11_topomapa` | [docs/11_topomapa.md](docs/11_topomapa.md) |
| `pipeline/e12_figuras` | [docs/12_figuras.md](docs/12_figuras.md) |
| `pipeline/avaliacao` (biblioteca) | [docs/avaliacao.md](docs/avaliacao.md) |
