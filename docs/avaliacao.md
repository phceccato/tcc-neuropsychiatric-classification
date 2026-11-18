# Avaliação — métricas e testes

Esta pasta guarda as funções de métrica e de teste estatístico usadas pelo DTSF, pelos baselines clássicos e pela LR de 76 bandas. Ela não tem execução própria: os resultados aparecem em [07_DTSF.md](07_DTSF.md), [08_classicos.md](08_classicos.md) e [06_LR_psd.md](06_LR_psd.md).

- Código: [pipeline/avaliacao/metricas.py](../pipeline/avaliacao/metricas.py)
- Testes: `tests/avaliacao/test_metricas.py`, que confere, entre outras coisas, a AUC vetorizada contra o scikit-learn, com empates

## O que cada função implementa

| Função | O que faz | Onde é usada |
| --- | --- | --- |
| `escores_por_pessoa` | Média da probabilidade fora do fold de cada pessoa nas repetições | Todas as avaliações |
| `avaliar` | AUC por pessoa, IC, AUC por repetição (média ± desvio) e secundárias | Tabelas de resultado |
| `ic_bootstrap` | IC de 95% da AUC, por percentil, reamostrando pessoas dentro de cada classe (2000 vezes) | Incerteza da AUC |
| `metricas_secundarias` | Acurácia, acurácia balanceada, precisão, sensibilidade (recall), especificidade, F1 (paciente como classe positiva) e Macro F1 no limiar 0,5 | Métricas descritivas |
| `metricas_por_repeticao`, `media_desvio`, `tabela_media_desvio` | AUC e secundárias de cada repetição (os 5 folds juntos) e a média ± desvio (ddof = 1) das 10 | Tabela de média ± desvio dos resumos das etapas 06, 07 e 08 |
| `auc_em_lote` | AUC de Mann-Whitney vetorizada, com empate valendo ½ | ΔAUC e bootstrap |
| `ic_bootstrap_delta` | IC de 95% do ΔAUC, reamostrando pessoas e mantendo os pares | IC das comparações e das ablações (descritivo) |
| `parear` | Alinha os escores por pessoa de dois modelos e recusa pessoas ausentes, repetidas, rótulos divergentes ou escores inválidos | Comparações pareadas |
| `holm` | Correção de Holm (step-down) | Os 5 p do DeLong em cada transtorno |
| `teste_delong` | Teste de DeLong (DELONG et al., 1988) para a diferença entre duas AUCs correlacionadas nas mesmas pessoas: ΔAUC, erro-padrão, z, p bicaudal e IC de 95% (Δ ± 1,96 dp). Variância zero (as duas AUCs valem 1 ou escores idênticos) devolve p = 1 e `degenerado` | Análise estatística principal (etapa 11) |
| `delta_por_repeticao` | ΔAUC de cada repetição | Descritivo |

## Média ± desvio nas repetições

Em cada uma das 10 repetições, as previsões fora do fold dos 5 folds formam uma previsão para cada pessoa; sobre elas se calculam a AUC e as secundárias (limiar 0,5). Os resumos trazem a média ± desvio dessas 10. O desvio mostra quanto o resultado muda com a partição. Ele não é a incerteza sobre a população, porque as repetições reusam as mesmas pessoas; para isso vale o IC por bootstrap da AUC por pessoa.

## Teste de DeLong com Holm (análise principal)

O trabalho não formula hipóteses a decidir. A inferência é o teste de DeLong, que compara as AUCs de dois modelos por vez, nas mesmas pessoas, e depois ajusta os p por Holm. Ele é calculado na etapa 11 ([11_graficos.md](11_graficos.md)) sobre os escores por pessoa (média das 10 repetições):
1. **Comparações:** o modelo principal (`DTSF_FA`) contra os outros 5 modelos principais: `DTSF` (dimensões do artigo, sem FA), `DTSF_Lite` e os 3 baselines combinados (LR, RF e KNN). Em cada uma, `teste_delong` dá a AUC dos dois modelos, ΔAUC = AUC(`DTSF_FA`) − AUC(outro), o erro-padrão, o Z, o p bruto (bicaudal) e o IC de 95% do Δ.
2. **Holm:** os 5 p brutos de cada transtorno são ajustados por Holm (`holm`, α = 0,05, separadamente em SZ e em MDD). "Significativo" é a decisão do Holm.

A tabela (comparação, AUC dos dois modelos, ΔAUC, Z, p bruto, p ajustado e decisão) sai em `outputs/11_graficos/delong.tex` (LaTeX, booktabs), `delong.csv` e na figura `p10_delong_tabela`.

Todas as outras comparações (ablações, efeitos da FA e do tamanho, baselines de uma modalidade) são **descritivas**: ΔAUC com IC de 95% por bootstrap estratificado e pareado, sem p-valor, e o Δ de cada repetição. Não há teste contra o acaso: o IC da AUC por pessoa mostra a incerteza da estimativa. Ressalvas:
- o DeLong supõe pessoas independentes e escores fixos, e a validação cruzada viola isso (os treinos se sobrepõem e o escore é uma média de repetições): o p é aproximado e tende a ser otimista, e o escore de cada pessoa é a média de 10 repetições, mais estável que o de uma repetição só (a AUC por pessoa é mais alta que a média das AUCs por repetição);
- a aproximação normal é fraca com 14 + 14 pessoas, e perto de AUC = 1 a variância se aproxima de zero (nesse caso o teste devolve p = 1 e marca `degenerado`);
- o teste é bicaudal;
- o Holm só corrige os 5 p de cada transtorno, não as demais comparações do trabalho.

Validação: `tests/avaliacao/test_metricas.py` confere o teste contra uma implementação independente por laços, a AUC contra o scikit-learn, a simetria, os casos degenerados, a calibração sob a nula (taxa de rejeição entre 2% e 9% a α = 0,05, com pessoas independentes) e o poder.

## Comparações descritivas

`comparacao_pareada` ([pipeline/e07_DTSF/avaliar.py](../pipeline/e07_DTSF/avaliar.py)) compara dois modelos nas mesmas pessoas e devolve o ΔAUC (a − b) sobre os escores por pessoa, o IC de 95% (`ic_bootstrap_delta`) e o Δ de cada repetição (`delta_por_repeticao`). Ela é usada:
- na etapa 07, para `DTSF_Lite` − ablação (`DTSF_Lite_SpectralOnly`, `DTSF_Lite_TemporalOnly`, `DTSF_Lite_NoMixup`) e para `DTSF_FA` − `DTSF`, `DTSF` − `DTSF_Lite` e `DTSF_FA` − `DTSF_Lite`;
- na etapa 08, para `DTSF_FA` − cada um dos 9 baselines clássicos;
- na etapa 11 (`numeros.py`), que recalcula essas comparações das previsões exportadas e as confere contra as tabelas dos resumos.
