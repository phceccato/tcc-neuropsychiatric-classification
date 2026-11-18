# Gráficos de resultados

Gráficos do TCC e do material suplementar, como imagens prontas (matplotlib): PDF vetorial para `\includegraphics` no Overleaf e PNG de conferência. Esta etapa não treina nada e calcula o teste de DeLong com Holm, a análise estatística principal: lê a pasta versionada da etapa 10 (`outputs/10_resultados/`), recalcula os números a partir das previsões, confere cada um contra o resumo da etapa e desenha. Por isso, qualquer pessoa com o repositório gera os mesmos gráficos sem treinar nenhum modelo. A única exceção é a linha do tempo das gravações (`s12_sessoes`), que precisa das datas dos EDF (etapa 02).

- Código: [pipeline/e11_graficos/](../pipeline/e11_graficos/)
  - `numeros.py`: lê as previsões (`previsoes/<dataset>/<modelo>.csv`), o histórico de perdas do DTSF_FA e os resumos de `outputs/10_resultados/`; recalcula, com as mesmas funções e sementes das etapas, a AUC por pessoa com IC, a AUC de cada repetição, a média ± desvio de todas as métricas nas 10 repetições, as secundárias por pessoa, a matriz de confusão, os escores por pessoa (rótulo e escore médio, sem identificador) para as curvas ROC, o teste de DeLong de cada comparação, com Holm sobre os 5 p do transtorno (`delong`, e a tabela `delong.csv`; não há resumo de etapa para conferir, a conferência é o teste unitário), as curvas ROC de reamostras bootstrap do DTSF_FA (`bootstrap`, com o IC conferido contra o resumo), as perdas do DTSF_FA por (repetição, fold), conferidas (50 folds, épocas em sequência e a época escolhida é a de menor perda de validação; `Divergencia` se não for), as comparações descritivas (ΔAUC com IC e Δ por repetição: DTSF_Lite contra as ablações, DTSF_FA contra os 9 baselines e entre os modelos da família DTSF); confere contra os resumos e grava `numeros.json`;
  - `graficos.py`: as figuras em matplotlib;
  - `sessoes.py`: a linha do tempo das gravações (matplotlib). Lê o manifesto da etapa 02 e confere as contagens por sessão contra [09_investigacao.md](09_investigacao.md); roda à parte do `graficos.py`, porque o manifesto não vai para o git;
- Testes: `tests/e11_graficos/`, com uma pasta `10_resultados` sintética que passa por `numeros` e `graficos` de ponta a ponta.
- Saídas em `outputs/11_graficos/` (versionada, como `outputs/10_resultados/`): `numeros.json`; `principais/` e `suplementares/`, com um `.pdf` e um `.png` de cada figurapng`.

Pré-requisitos: `outputs/10_resultados/` completa (etapa 10), com as previsões dos 17 modelos, o histórico de perdas do DTSF_FA (`perdas/`) e os resumos; para `sessoes`, o manifesto da etapa 02.

```bash
venv/bin/python -m pipeline.e11_graficos.numeros    # recalcula e confere → outputs/11_graficos/numeros.json
venv/bin/python -m pipeline.e11_graficos.graficos   # → outputs/11_graficos/principais/ e suplementares/
venv/bin/python -m pipeline.e11_graficos.delong     # tabela do DeLong em LaTeX → outputs/11_graficos/delong.tex
venv/bin/python -m pipeline.e11_graficos.sessoes    # → outputs/11_graficos/suplementares/s12_sessoes.pdf e .png (precisa da etapa 02)
```

O `numeros` para com `Divergencia`, sem gravar, se algum número não bater com o resumo da etapa: |diferença| ≤ 0,0005 em AUC, IC e ΔAUC, e ≤ 0,00005 em p.

## Figuras

### Principais

| Arquivo | Conteúdo |
| --- | --- |
| `p1_roc_principais` | Curvas ROC por pessoa dos 6 modelos principais: DTSF_FA, DTSF, DTSF_Lite e LR, RF e KNN combinados, com a AUC na legenda |
| `p2_auc_modelos` | AUC por pessoa e IC de 95% dos 17 modelos, em grupos (DTSF, ablações, clássicos temporais, espectrais e combinados, LR de 76 bandas) |
| `p3_metricas_principais` | AUC, acurácia, precisão, recall e F1 dos 6 principais: média ± desvio nas 10 repetições |
| `p4_sz_mdd` | AUC por pessoa e IC de 95% de cada modelo em SZ e em MDD, lado a lado |
| `p9_delong_sz`, `p9_delong_mdd` | O DTSF_FA contra cada um dos outros 5 modelos principais (DTSF, DTSF_Lite e os 3 combinados), uma figura por transtorno, com um painel por comparação: as duas curvas ROC por pessoa, a área entre elas, a diferença de AUC e o p do teste de DeLong, bruto e ajustado por Holm sobre os 5 p do transtorno. É a análise estatística principal (ver [avaliacao.md](avaliacao.md#teste-de-delong-com-holm-análise-principal)) |
| `p10_delong_tabela` | A tabela do DeLong dos dois transtornos: comparação, AUC dos dois modelos, ΔAUC, Z, p bruto, p ajustado (Holm) e a decisão. A mesma tabela está em `delong.csv` e, em LaTeX para o Overleaf, em `delong.tex` (`python -m pipeline.e11_graficos.delong`; precisa de `\usepackage{booktabs}`) |
| `p8_roc_bootstrap_dtsf_fa` | Curvas ROC de reamostras bootstrap do DTSF_FA (nuvem cinza) e a curva original (preta), com a AUC por pessoa e o IC de 95%, em SZ e em MDD. O bootstrap é estratificado por pessoa e usa as mesmas 2.000 reamostragens do IC do resto do trabalho (o IC da figura é o das tabelas, e o `numeros` confere); só as 500 primeiras curvas são desenhadas, para o `numeros.json` não crescer |
| `p7_metricas_limiar` | Gráfico de linhas com acurácia, precisão, recall e F1 dos 6 modelos principais no eixo x, um painel por transtorno: (a) SZ e (b) MDD. Valores em %, a média das 10 repetições no limiar 0,5, com o paciente como classe positiva. O eixo vertical não começa em zero (a nota está na figura) e é o mesmo nos dois painéis. O desvio entre repetições fica em `p3_metricas_principais` e em `metricas.csv` |
| `p6_roc_pr_sz`, `p6_roc_pr_mdd` | Curvas ROC (esquerda) e Precision-Recall (direita) por pessoa dos 6 modelos principais, uma figura por transtorno, com a AUC e a AP (average precision) de cada modelo na legenda. A linha cinza é o acaso na ROC e a prevalência de pacientes (0,50 na SZ, 0,52 no MDD) na Precision-Recall. Os escores são a média das probabilidades nas 10 repetições |
| `p5_loss_dtsf_fa` | Perda de treino e de validação interna do DTSF_FA por época, nos 50 folds de cada transtorno (linhas finas) e a mediana dos folds (linhas grossas), com a época escolhida. A perda de treino é a do lote com Mixup e dropout e a de validação é a das pessoas do split interno, sem Mixup: as escalas não são comparáveis (a nota está na figura). Só é gerada se `numeros.json` tiver as perdas, que vêm de `perdas/<dataset>/DTSF_FA.csv` da etapa 10; do contrário o `graficos` avisa e segue |

### Suplementares

| Arquivo | Conteúdo |
| --- | --- |
| `s1_roc_ablacoes` | Curvas ROC do DTSF_Lite e das suas 3 ablações |
| `s11_metricas_limiar_todos` | O mesmo gráfico de linhas de `p7` para os 17 modelos, com SZ em cima e MDD embaixo e os modelos agrupados como em `p2` (DTSF, ablações, clássicos temporais, espectrais e combinados, LR de 76 bandas). Como acurácia, precisão, recall e F1 dependem do limiar de 0,5, não use este gráfico para ordenar modelos: a AUC é a métrica principal |
| `s10_roc_todos_sz`, `s10_roc_todos_mdd` | Curvas ROC por pessoa dos 17 modelos, em seis painéis por transtorno (um por grupo, como em `p2`), com o DTSF_FA como referência em cada painel e a AUC na legenda. Nos clássicos, a cor indica a modalidade (azul temporal, laranja espectral, verde combinado) e o tracejado, o classificador (LR contínuo, RF tracejado, KNN pontilhado) |
| `s2_confusao_sz`, `s2_confusao_mdd` | Matriz de confusão por pessoa (limiar 0,5) dos 17 modelos, com contagem e porcentagem da classe real |
| `s3_temporal_espectral` | AUC só com o temporal, só com o espectral e com os dois, no DTSF (DTSF_Lite_TemporalOnly, DTSF_Lite_SpectralOnly e DTSF_Lite) e em LR, RF e KNN |
| `s4_absolutas_relativas` | LR de 76 bandas com potência absoluta e relativa: AUC por pessoa com IC e AUC de cada repetição. Não é uma checagem de lote: a potência relativa não elimina diferenças de ritmo entre sessões ([06_LR_psd.md](06_LR_psd.md#checagem-da-potência-relativa-por-sessão)) |
| `s5_auc_repeticoes` | AUC de cada uma das 10 repetições e a média, dos 17 modelos |
| `s6_delta_auc` | ΔAUC = referência − comparado, com IC de 95%, sem p-valor: DTSF_Lite contra as ablações, DTSF_FA contra os 3 combinados, entre os modelos da família DTSF e DTSF_FA contra os 6 baselines de uma modalidade |
| `s8_barras_ablacoes` | AUC por pessoa em barras do DTSF_Lite e das ablações |
| `s9_barras_classicos` | AUC por pessoa em barras do DTSF_FA e dos 9 baselines, com a cor pela modalidade |
| `s12_sessoes` | Data de gravação de cada participante por classe, com as sessões reconstruídas (nova sessão quando duas datas distam mais de 30 dias); gerada por `sessoes.py`, não pelo `graficos.py`; nenhum identificador vai para a figura |

## Formato

- Largura de 16 cm (cabe em `\linewidth` com as margens da ABNT); fonte serifada, 6,5 a 8,5 pt; vírgula decimal nos eixos e nos textos.
- Figuras de pontos (`p2`, `s5`, `s6`) em preto e cinza, para a leitura impressa: os modelos da família DTSF em preto (DTSF_FA círculo, DTSF quadrado e DTSF_Lite triângulo, preenchidos; as ablações, círculos vazados), os demais em cinza; o classificador se distingue pelo marcador (LR círculo, RF quadrado, KNN triângulo, LR de 76 bandas losango).
- As curvas ROC diferenciam os modelos pela cor e pelo tracejado, para a leitura em P&B; a legenda fica embaixo, para não cobrir as curvas.
- As figuras de barras usam a paleta categórica de referência (azul, laranja, água, amarelo), validada para daltonismo nos pares vizinhos; a legenda ou o eixo nomeia o modelo, e a cor nunca é a única pista. As barras começam em 0, com a linha do acaso em 0,5.
- Nenhum identificador de pessoa vai para as figuras nem para `numeros.json`.

## Leitura

- A AUC por pessoa usa o escore médio das 10 repetições; a média por repetição (`p3`, `s5`) é menor, porque cada repetição é mais ruidosa.
- O desvio de `p3` mostra quanto o resultado muda com a partição das pessoas, não a incerteza sobre a população (as repetições reusam as mesmas pessoas).
- `p4` compara SZ e MDD só de forma descritiva: são pessoas e aparelhos diferentes, e não há teste entre os transtornos.
- Acurácia, precisão, recall e F1 usam o limiar 0,5 sobre a probabilidade, com o paciente como classe positiva.

## Legendas sugeridas

> **p1.** Curvas ROC por pessoa dos modelos principais, sobre a média da probabilidade fora do fold nas 10 repetições. Entre parênteses, a AUC por pessoa. A diagonal é o acaso.

> **p2.** AUC por pessoa e intervalo de confiança de 95% (bootstrap estratificado) de cada modelo. A linha tracejada marca o acaso (0,5) e a pontilhada, a AUC do DTSF_FA, o modelo principal.

> **p3.** Métricas dos modelos principais no limiar de 0,5 (paciente como classe positiva): média e desvio-padrão nas 10 repetições da validação cruzada (5 folds).

> **s6.** Diferença de AUC por pessoa entre o modelo de referência e cada modelo comparado, com intervalo de confiança de 95% (bootstrap estratificado e pareado). Valores positivos favorecem a referência: o DTSF_Lite nas ablações, de que elas derivam, e o DTSF_FA nas demais comparações. É descritivo: não há p-valor; o teste estatístico é o DeLong com Holm (p9 e p10).


## Resultados

As figuras da rodada oficial (GPU) estão em `outputs/11_graficos/` (`principais/`, `suplementares/`, `delong.csv`, `delong.tex`, `numeros.json`), geradas a partir de `outputs/10_resultados/`. As figuras de rodadas anteriores (`12_figuras_rodada2` e `13_graficos_rodada2`) e o experimento exploratório de FA e SDKA (`12_FA_SDKA`) foram arquivados fora do repositório, em `~/Documentos/TCC/arquivo_outputs_antigos/`, e não valem para esta rodada.

DeLong bilateral do `DTSF_FA` contra os outros cinco modelos principais, com Holm sobre os 5 p de cada dataset (α = 0,05). Nenhuma comparação é significativa depois do Holm:

| Dataset | Comparação | ΔAUC | p | p Holm |
| --- | --- | --- | --- | --- |
| SZ | DTSF_FA − DTSF | +0,015 | 0,26 | 0,61 |
| SZ | DTSF_FA − DTSF_Lite | +0,026 | 0,26 | 0,61 |
| SZ | DTSF_FA − LR combinado | +0,041 | 0,20 | 0,61 |
| SZ | DTSF_FA − RF combinado | +0,158 | 0,057 | 0,29 |
| SZ | DTSF_FA − KNN combinado | +0,143 | 0,10 | 0,40 |
| MDD | DTSF_FA − DTSF | +0,003 | 0,73 | 0,83 |
| MDD | DTSF_FA − DTSF_Lite | +0,006 | 0,41 | 0,83 |
| MDD | DTSF_FA − LR combinado | +0,026 | 0,040 | 0,16 |
| MDD | DTSF_FA − RF combinado | +0,061 | 0,025 | 0,12 |
| MDD | DTSF_FA − KNN combinado | +0,052 | 0,049 | 0,16 |

Em MDD, três p brutos ficam abaixo de 0,05, mas nenhum resiste ao Holm. As ressalvas do teste (dependência entre folds, escores médios otimistas, poucas pessoas) estão em [avaliacao.md](avaliacao.md).

`p5_loss_dtsf_fa`: a perda de validação do `DTSF_FA` cai e se estabiliza perto da época escolhida (mediana ~12 em SZ e ~7 em MDD); os 50 folds de cada transtorno são ruidosos, como esperado com tão poucas pessoas.
