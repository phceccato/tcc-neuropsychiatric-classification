# Gráficos de resultados (matplotlib e pgfplots)

Gráficos do TCC e do material suplementar, em dois formatos: imagens prontas (matplotlib: PDF vetorial para `\includegraphics` e PNG de conferência) e figuras em pgfplots (`.tex` para dar `\input` no Overleaf, com um PNG de prévia). Esta etapa não treina nem testa nada: lê a pasta versionada da etapa 10 (`outputs/10_resultados/`), recalcula os números a partir das previsões, confere cada um contra o resumo da etapa e desenha. Por isso, qualquer pessoa com o repositório gera os mesmos gráficos sem treinar nenhum modelo. A única exceção é a linha do tempo das gravações (`fig_sessoes`), que precisa das datas dos EDF (etapa 02).

- Código: [pipeline/e11_graficos/](../pipeline/e11_graficos/)
  - `numeros.py`: lê as previsões (`previsoes/<dataset>/<modelo>.csv`), as permutações do D_FA e os resumos de `outputs/10_resultados/`; recalcula, com as mesmas funções e sementes das etapas, a AUC por pessoa com IC, a AUC de cada repetição, a média ± desvio de todas as métricas nas 10 repetições, as secundárias por pessoa, a matriz de confusão, os escores por pessoa (rótulo e escore médio, sem identificador) para as curvas ROC, as comparações (H2 e H3 sobre o D_pequeno, H4 e as descritivas com os baselines sobre o D_FA, e as descritivas entre os DTSF), as decisões das hipóteses e a distribuição nula da H1; confere contra os resumos e grava `numeros.json`;
  - `graficos.py`: as figuras em matplotlib;
  - `comum.py`, `h1_nula.py`, `roc.py` e `sessoes.py`: as figuras em pgfplots (ambiente `figure` completo, com `\caption` e `\label` em português e os dados embutidos em `coordinates`). `h1_nula` e `roc` leem o `numeros.json`; `sessoes` lê o manifesto da etapa 02 e confere as contagens por sessão contra [09_investigacao.md](09_investigacao.md);
  - `previa.py`: compila cada `.tex` num documento mínimo, recortado na figura e com 16 cm de largura de texto, e grava um PNG ao lado, só para conferência.
- Testes: `tests/e11_graficos/`, com uma pasta `10_resultados` sintética que passa por `numeros`, `graficos` e pgfplots de ponta a ponta (com a compilação no pdflatex, quando ele está instalado).
- Saídas em `outputs/11_graficos/` (versionada, como `outputs/10_resultados/`): `numeros.json`; `principais/` e `suplementares/`, com um `.pdf` e um `.png` de cada figura; e `pgfplots/`, com os `.tex` e as prévias em `.png`.

Pré-requisitos: `outputs/10_resultados/` completa (etapa 10), com as previsões dos 20 modelos, as permutações do D_FA e os resumos; para `sessoes`, o manifesto da etapa 02; para a prévia, o pdflatex com pgfplots e o pdftocairo (`sudo apt install texlive-latex-extra texlive-pictures poppler-utils`).

```bash
venv/bin/python -m pipeline.e11_graficos.numeros    # recalcula e confere → outputs/11_graficos/numeros.json
venv/bin/python -m pipeline.e11_graficos.graficos   # → outputs/11_graficos/principais/ e suplementares/
venv/bin/python -m pipeline.e11_graficos.h1_nula    # → outputs/11_graficos/pgfplots/fig_h1_nula.tex
venv/bin/python -m pipeline.e11_graficos.roc        # → outputs/11_graficos/pgfplots/fig_roc.tex
venv/bin/python -m pipeline.e11_graficos.sessoes    # → outputs/11_graficos/pgfplots/fig_sessoes.tex (precisa da etapa 02)
venv/bin/python -m pipeline.e11_graficos.previa     # PNG de cada .tex → outputs/11_graficos/pgfplots/*.png
```

O `numeros` para com `Divergencia`, sem gravar, se algum número não bater com o resumo da etapa: |diferença| ≤ 0,0005 em AUC, IC e ΔAUC, e ≤ 0,00005 em p.

## Figuras

### Principais

| Arquivo | Conteúdo |
| --- | --- |
| `p1_roc_principais` | Curvas ROC por pessoa dos 6 modelos principais: D_FA, D, D_pequeno e LR, RF e KNN combinados, com a AUC na legenda |
| `p2_auc_modelos` | AUC por pessoa e IC de 95% dos 20 modelos, em grupos (DTSF, ablações, clássicos temporais, espectrais e combinados, LR de 76 bandas) |
| `p3_metricas_principais` | AUC, acurácia, precisão, recall e F1 dos 6 principais: média ± desvio nas 10 repetições |
| `p4_sz_mdd` | AUC por pessoa e IC de 95% de cada modelo em SZ e em MDD, lado a lado |

### Suplementares

| Arquivo | Conteúdo |
| --- | --- |
| `s1_roc_ablacoes` | Curvas ROC do D_pequeno e das suas 3 ablações |
| `s2_confusao_sz`, `s2_confusao_mdd` | Matriz de confusão por pessoa (limiar 0,5) dos 20 modelos, com contagem e porcentagem da classe real |
| `s3_temporal_espectral` | AUC só com o temporal, só com o espectral e com os dois, no DTSF (D_sem_E, D_sem_T e D_pequeno) e em LR, RF e KNN |
| `s4_absolutas_relativas` | LR de 76 bandas com potência absoluta e relativa: AUC por pessoa com IC e AUC de cada repetição |
| `s5_auc_repeticoes` | AUC de cada uma das 10 repetições e a média, dos 20 modelos |
| `s6_delta_hipoteses` | ΔAUC = referência − comparado, com IC de 95% e p: H2 e H3 (D_pequeno contra as ablações), H4 (D_FA contra os 3 combinados), com a decisão; e as descritivas (entre os DTSF e D_FA contra os 6 baselines de uma modalidade), sem decisão |
| `s7_testes_estatisticos` | Distribuição nula da H1 (AUCs com rótulos permutados e a AUC observada) e tabela com o teste, p, p de Holm e decisão de H1 a H4 em cada transtorno |
| `s8_barras_ablacoes` | AUC por pessoa em barras do D_pequeno e das ablações |
| `s9_barras_classicos` | AUC por pessoa em barras do D_FA e dos 9 baselines, com a cor pela modalidade |

### Em pgfplots (Overleaf)

| Arquivo | Conteúdo |
| --- | --- |
| `fig_h1_nula` (`fig:res-h1-nula`) | Dot plot das AUCs do D_FA retreinado com rótulos permutados (25 permutações, repetição 0), com a AUC observada, o p e o acaso |
| `fig_roc` (`fig:res-roc`) | Curvas ROC por pessoa do D_FA e dos 3 baselines combinados da H4 |
| `fig_sessoes` (`fig:des-sessoes`) | Data de gravação de cada participante por classe, com as sessões reconstruídas (nova sessão quando duas datas distam mais de 30 dias); nenhum identificador vai para o `.tex` |

Preâmbulo do Overleaf:

```latex
\usepackage{pgfplots}
\usepgfplotslibrary{groupplots}
\pgfplotsset{compat=1.18}
```

Copie os `.tex` para a pasta do projeto e use `\input{fig_h1_nula}`, `\input{fig_roc}` e `\input{fig_sessoes}`. Formato: largura de no máximo `\linewidth`, fonte `\footnotesize`, vírgula decimal nos eixos (`/pgf/number format/use comma, 1000 sep={.}`) e no texto, e séries diferenciadas também por marcador ou tracejado, para a leitura impressa em P&B. Na prévia em PNG, a legenda sai como "Figura 1", porque cada figura é compilada sozinha.

## Formato

- Largura de 16 cm (cabe em `\linewidth` com as margens da ABNT); fonte serifada, 6,5 a 8,5 pt; vírgula decimal nos eixos e nos textos.
- Figuras de pontos (`p2`, `s5`, `s6`) em preto e cinza, para a leitura impressa: os DTSF em preto (D_FA círculo, D quadrado e D_pequeno triângulo, preenchidos; as ablações, círculos vazados), os demais em cinza; o classificador se distingue pelo marcador (LR círculo, RF quadrado, KNN triângulo, LR de 76 bandas losango).
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

> **p2.** AUC por pessoa e intervalo de confiança de 95% (bootstrap estratificado) de cada modelo. A linha tracejada marca o acaso (0,5) e a pontilhada, a AUC do D_FA, o modelo principal.

> **p3.** Métricas dos modelos principais no limiar de 0,5 (paciente como classe positiva): média e desvio-padrão nas 10 repetições da validação cruzada (5 folds).

> **s6.** Diferença de AUC por pessoa entre o modelo de referência e cada modelo comparado, com intervalo de confiança de 95% (bootstrap estratificado e pareado) e o p unilateral da permutação pareada sobre os ranks (10.000 permutações). Valores positivos favorecem a referência: o D_pequeno em H2 e H3, de que as ablações derivam, e o D_FA em H4 e nas comparações com os baselines. Em cada hipótese, a decisão usa a interseção-união entre as comparações e a correção de Holm entre os transtornos (α = 0,05); os grupos descritivos não têm decisão.

> **s7.** Acima, as AUCs do D_FA retreinado com os rótulos permutados (25 permutações, repetição 0) e a AUC observada na repetição 0. Abaixo, o teste, o p, o p corrigido por Holm entre os transtornos e a decisão de cada hipótese (α = 0,05; a H1 é decidida sem Holm, porque o menor p possível com 25 permutações é 0,038).

## Resultados

Pendentes da rodada oficial: as figuras são geradas a partir de `outputs/10_resultados/` depois que a etapa 10 exportar os resultados.
