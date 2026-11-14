# Resultados exportados — reprodução sem treinar

Esta etapa junta as previsões fora do fold de todos os modelos numa pasta versionada, `outputs/10_resultados/`. Com ela, qualquer pessoa recalcula as AUCs, os ICs, a média ± desvio e os testes das hipóteses sem treinar nada e sem baixar o EEG. Ela não treina nem testa: só lê as saídas das etapas 05 a 08, confere e grava.

`outputs/10_resultados/` é a única pasta de `outputs/` no git; o resto continua fora (`.gitignore`).

- Código: [pipeline/e10_resultados/exportar.py](../pipeline/e10_resultados/exportar.py)
- Testes: `tests/e10_resultados/` (5 testes).
- Saídas em `outputs/10_resultados/`:

| Arquivo | Conteúdo |
| --- | --- |
| `previsoes/<dataset>/<modelo>.csv` | uma linha por (repetição, fold, pessoa de teste): `repeticao`, `fold`, `subject_uid`, `y_true` (1 = paciente), `y_pred` (`y_prob` ≥ 0,5) e `y_prob` (probabilidade de paciente, fora do fold) |
| `permutacoes/<dataset>/<modelo>.csv` | o mesmo para as 25 permutações de rótulo com retreino da H1 (coluna `permutacao`; `y_true` é o rótulo permutado) |
| `permutacoes/<dataset>/LR_psd_absolutas_nulas.csv` | as 1000 AUCs nulas da etapa 06, que grava só as AUCs |
| `folds.csv` | as partições da etapa 05: dataset, repetição, fold, pessoa, grupo e papel (treino, validação, teste) |
| `metricas.csv` | por (dataset, modelo): AUC por pessoa com IC de 95%; acurácia, acurácia balanceada, precisão, sensibilidade (recall), especificidade, F1 e Macro F1 por pessoa (limiar 0,5); e a média ± desvio de todas nas 10 repetições, recalculadas dos arquivos exportados |
| `resumos/` | cópia dos resumos das etapas 06, 07 e 08, com os testes das hipóteses |
| `manifesto.json` | semente, commit do código, limiar de `y_pred`, e por modelo: etapa, dispositivo (CPU ou GPU), número de parâmetros, linhas e permutações; os modelos que ficaram de fora por estarem incompletos |
| `LEIAME.md` | o que é cada arquivo e como recalcular a AUC por pessoa |

Modelos: os seis DTSF da etapa 07 (`D_FA`, `D_pequeno`, `D`, `D_sem_mixup`, `D_sem_T`, `D_sem_E`), os nove clássicos da etapa 08 e a LR_psd da etapa 06 (`LR_psd_absolutas`, `LR_psd_relativas`), nos dois datasets. Os nomes dos arquivos são os dos modelos nas etapas.

Pré-requisitos: as previsões completas e os resumos das etapas 06, 07 e 08 (os `avaliar` rodados depois dos treinos).

```bash
venv/bin/python -m pipeline.e10_resultados.exportar     # → outputs/10_resultados/ (~1 min); depois, commit da pasta
```

## Como a exportação é conferida

- Só entram modelos com as 10 repetições completas. Os incompletos não são gravados e ficam listados no manifesto.
- Antes de gravar, a AUC por pessoa de cada modelo é recalculada **a partir do arquivo exportado** e comparada com a AUC do resumo da etapa (|diferença| ≤ 0,0005). Se algum não bater, ou se um modelo não aparecer no resumo, o script para com `Divergencia` e não grava nada.
- A pasta é recriada do zero a cada exportação, então não sobram arquivos de uma rodada anterior.
- Os dispositivos ficam no manifesto: uma rodada em GPU e uma em CPU dão números próximos, mas não idênticos ([07_DTSF.md](07_DTSF.md#cpu-ou-gpu)).

## Reprodução

Sem treinar, a partir da pasta exportada:
- **AUC por pessoa:** média de `y_prob` de cada pessoa nas 10 repetições e uma AUC sobre as pessoas (o código está no `LEIAME.md` da pasta);
- **métricas e testes:** as funções de [pipeline/avaliacao/metricas.py](../pipeline/avaliacao/metricas.py) e `comparacao_pareada` ([pipeline/e07_DTSF/avaliar.py](../pipeline/e07_DTSF/avaliar.py)) recebem as previsões com as colunas `y` e `prob`; basta renomear `y_true` → `y` e `y_prob` → `prob`;
- **gráficos:** a etapa 12 lê esta pasta e gera os gráficos principais e suplementares ([12_graficos.md](12_graficos.md)).

Com treino, do zero: os dados são públicos e a etapa 00 baixa e confere cada arquivo pelo MD5 ([00_download-dados.md](00_download-dados.md)); as etapas 01 a 08 seguem os comandos do README. As sementes estão em [05_folds.md](05_folds.md).

## Execução

Testado com as saídas desta máquina (rodada em CPU, com os nomes de modelo anteriores à etapa 07 de seis modelos): 30 pares (dataset, modelo) exportados, todos conferidos contra os resumos, 852 KB no total. A exportação da rodada oficial é feita na máquina em que ela roda.
