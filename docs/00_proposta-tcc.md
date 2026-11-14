# TCC — DTSF adaptado e baselines clássicos em EEG de repouso (SZ×HC e MDD×HC)

@Ceccato

## Visão geral

O TCC classifica EEG de repouso com olhos fechados em dois problemas binários independentes: **esquizofrenia × controles** e **depressão maior × controles**.

- A arquitetura principal é o **DTSF** (WANG et al., 2026). Ele tem dois ramos CNN-GRU, um sobre o sinal em bandas de frequência e outro sobre a STFT, fundidos por um Transformer, e treina com Mixup.
- A pergunta principal é se o DTSF discrimina pacientes e controles e de onde vem o seu desempenho: das duas modalidades juntas (temporal e espectral) e do Mixup, e se ele supera **classificadores clássicos aplicados à sua própria entrada** (regressão logística, Random Forest e KNN sobre os resumos temporais e espectrais juntos; os de uma modalidade só são descritivos). As hipóteses H1–H4 estão em [Pergunta científica e hipóteses](#pergunta-científica-e-hipóteses).

O que vale para todo o trabalho:

- **Não** há classificação de 3 classes (MDD × SZ × HC).
- **Não** há classificação direta MDD × SZ.
- Cada tarefa tem os seus próprios modelos, treinados do zero, **sem compartilhamento de parâmetros** entre SZ e MDD. O que é igual entre as tarefas é o pipeline.

## Tarefas e datasets

Dois problemas binários, cada um com seu próprio dataset e seu próprio modelo. Os indivíduos são pessoas diferentes, então **não há necessidade de parear pacientes entre datasets**.

|                                | Task 1 — Esquizofrenia                                                                                                  | Task 2 — Depressão                                                                                                                             |
| ------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| Dataset                        | [EEG in schizophrenia](https://repod.icm.edu.pl/dataset.xhtml?persistentId=doi:10.18150/repod.0107441) (RepOD, 2017)      | [MDD Patients and Healthy Controls EEG Data (New)](https://figshare.com/articles/dataset/EEG_Data_New-/4244171) (Figshare, versão 2, 2016)       |
| Classificação                | SZ vs HC                                                                                                                 | MDD vs HC                                                                                                                                        |
| Local                          | Instituto de Psiquiatria e Neurologia, Varsóvia (Polônia)                                                              | Hospital Universiti Sains Malaysia (Malásia)                                                                                                    |
| Pessoas segundo a fonte        | 28: 14 SZ + 14 HC                                                                                                        | 63: 33 MDD + 30 HC                                                                                                                               |
| Gravações EC no repositório | 28 arquivos EDF (14 SZ, 14 HC)                                                                                           | 64 listadas (34 MDD, 30 H), 58 disponíveis para download (30 MDD, 28 H); 2 pares suspeitos de ser a mesma gravação (MDD S33/S34, H S27/H S30) |
| Sexo e idade                   | SZ: 7 homens (27,9 ± 3,3 anos) e 7 mulheres (28,3 ± 4,1); HC: 7 homens (26,8 ± 2,9) e 7 mulheres (28,7 ± 3,4)        | MDD: 40,3 ± 12,9 anos; HC pareados por idade                                                                                                    |
| Diagnóstico                   | CID-10 F20.0 (esquizofrenia paranoide), pacientes internados; washout de pelo menos 7 dias; primeiro episódio excluído | DSM-IV                                                                                                                                           |
| Canais                         | 19 (sistema 10–20)                                                                                                      | 19 (sistema 10–20)                                                                                                                              |
| Referência original           | FCz                                                                                                                      | Orelhas ligadas (linked ears)                                                                                                                    |
| Frequência original           | 250 Hz                                                                                                                   | 256 Hz                                                                                                                                           |
| Condições originais          | Olhos fechados (EC)                                                                                                      | EC, EO e TASK (P300)                                                                                                                             |
| Duração EC por pessoa        | 15 min                                                                                                                   | \~5 min                                                                                                                                          |
| Licença                       | CC0 1.0                                                                                                                  | CC BY 4.0                                                                                                                                        |
| Artigo de referência          | [Olejarczyk e Jernajczyk (2017)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0188629)               | Mumtaz et al. (2017)                                                                                                                             |

**Arquivos indisponíveis.** Seis gravações EC listadas no Figshare nunca ficaram disponíveis para download: H S12, H S18, MDD S4, MDD S8, MDD S12 e MDD S16. A API do Figshare mostra esses arquivos sem MD5 calculado, e o link de cada um retorna 404. Por isso a auditoria parte de 58 gravações EC no MDD, e não de 64. No Figshare, as gravações EC aparecem como 34 rótulos MDD e 30 H; trabalhos que relatam 34 MDD contam rótulos de arquivo, não pessoas.

**Resultado da auditoria.** Os critérios de inclusão foram escritos antes de olhar qualquer resultado (docs/01_auditoria.md). O n final é **28 pessoas na SZ (14 SZ + 14 HC) e 56 no MDD (29 MDD + 27 HC)**.

- Na SZ, todas as gravações entram; em s14, o canal C3 fica plano em 33% da gravação e é interpolado.
- No MDD, os dois pares suspeitos são cópias byte a byte (H S30 de H S27, MDD S34 de MDD S33), então cada par conta como uma pessoa. O cabeçalho de H S14 EC indica EO, e a gravação entra sinalizada.
- Nenhuma gravação caiu por duração, saturação ou leitura. O detalhe por arquivo sai de `pipeline/e01_auditoria/auditar_ec.py`.

Outros trabalhos com o dataset de depressão usam n diferentes, como 28 MDD e 27 HC ([PMC11297863](https://pmc.ncbi.nlm.nih.gov/articles/PMC11297863/)), então seus números não se comparam diretamente com os deste TCC. Esse mesmo trabalho informa o equipamento do dataset de depressão (amplificador BrainMaster Discovery 24E), a idade média dos pacientes e o critério DSM-IV usados na tabela.

### Harmonização

Os EDF das gravações incluídas são harmonizados com o mesmo pipeline nos dois datasets, com scipy e, na interpolação do canal ruim, MNE-Python (GRAMFORT et al., 2013): **19 canais, 250 Hz, referência média, EC** (docs/02_harmonizacao.md). A sequência é:

1. µV;
2. corte dos zeros do fim;
3. 19 canais 10–20;
4. interpolação do canal ruim por spline esférica, antes da CAR, para o canal ruim não entrar na média;
5. referência média (CAR);
6. reamostragem para 250 Hz;
7. notch de 50 Hz (Q = 30);
8. passa-banda Butterworth de 1–40 Hz, N = 4 (ordem 8 por passada), fase zero (`sosfiltfilt`, duas passadas).

O objetivo é que os resultados de SZ e MDD sejam obtidos nas mesmas condições. A análise usa só a faixa de 2–30 Hz (ver abaixo).

## Pré-processamento e janelas

| Item                    | Decisão                                                                                                                                                                                                                                                        | Por quê                                                                                                                                                                                                                           |
| ----------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Pré-filtro             | Passa-banda Butterworth 2–30 Hz, N = 4, fase zero (`sosfiltfilt`), na gravação inteira, antes de cortar as janelas. É aplicado sobre o sinal harmonizado (já em 1–40 Hz), então a resposta efetiva é a cascata dos dois filtros                       | Evita efeito de borda. Fora de 2–30 Hz o sinal reflete o equipamento: 27 de 29 MDD têm passa-alta efetivo mais forte (\~−10 dB em 1 Hz) e queda perto de 38 Hz, contra só 4 HC. Essas faixas separariam MDD e HC pelo aparelho |
| Tamanho                 | 4 s (1000 amostras a 250 Hz)                                                                                                                                                                                                                                    | 8 ciclos até em 2 Hz                                                                                                                                                                                                              |
| Overlap                 | 50% na grade de treino; 0% na validação e no teste                                                                                                                                                                                                            | Mais exemplos no treino; janelas independentes na avaliação                                                                                                                                                                      |
| Janelas por gravação  | SZ: \~449 na grade de treino e \~225 sem overlap; MDD: \~149 e \~75                                                                                                                                                                                             | Gravações de \~15 min na SZ e \~5 min no MDD (OLEJARCZYK; JERNAJCZYK, 2017; MUMTAZ et al., 2017)                                                                                                                                 |
| Rejeição de artefato  | Janela descartada se a amplitude pico a pico, medida no sinal de 2–30 Hz em µV, passar de 150 µV em qualquer canal, ou se ela tocar um trecho plano marcado pela auditoria num canal que não foi interpolado. A rejeição vem antes do cálculo do z-score | Descarte desigual entre classes vira viés, então é reportado por pessoa e por classe (docs/03_janelas.md)                                                                                                                       |
| Normalização do sinal | z-score por canal, com média e desvio da própria pessoa, calculados só nas janelas aceitas, e reamostragem para 125 Hz na gravação inteira                                                                                                                 | Usa só dados da pessoa, sem rótulo; remove diferenças de ganho. Sem as janelas rejeitadas, um artefato isolado não infla o desvio                                                                                              |
| Agregação             | Média das previsões das janelas aceitas sem overlap → previsão da pessoa                                                                                                                                                                                    | É a unidade clínica; contar janelas infla o n artificialmente                                                                                                                                                                    |

## DTSF

Resultados: docs/07_DTSF.md.

| Parte            | Decisão                                                                                                                                                                                                                                                                                                                                                          |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Ramo temporal    | Entrada (19 canais, 5, 500): o sinal de 2–30 Hz com z-score por pessoa a 125 Hz e as bandas δ 2–4, θ 4–8, α 8–13 e β 13–30 Hz (Butterworth de ordem 4, `sosfiltfilt`, na gravação inteira). Duas Conv2d (kernels (5, 13) e (1, 25)) com BN, GELU, pooling e dropout de 0,5; no modelo principal, a Frequency Attention entre as duas; depois GRU → 16 tokens                                            |
| Ramo espectral   | Entrada (19, 29, 16): log₁₀(\|X\|² + 10⁻¹²) da STFT da janela em µV (Hann de 125 amostras, passo de 25), de 2 a 30 Hz, com z-score por (canal, frequência) a partir das janelas de treino do fold. Duas Conv2d (kernels (29, 3) e (1, 3)), depois GRU → 16 tokens. Fica sem o z-score por pessoa, porque a potência absoluta carrega informação |
| Fusão           | 32 tokens → Linear → token CLS no fim → posição aprendível → 2 blocos de encoder pre-LN (MHSA, FFN com gate GELU) → LN → último token → logit |
| Mixup            | λ ~ Beta(1, 1), um por lote; mesmo λ e mesma permutação nos dois ramos e no alvo; só no treino, com gerador próprio                                                                                                                                                                                                                                         |
| Dimensões        | As do artigo (Tabela 1): convoluções de 64 filtros, GRU e fusão de dimensão 128, encoder de 2 blocos com 4 cabeças (702.721 parâmetros; 704.769 com a FA). O `D_pequeno` usa 16 filtros, dimensão 32 e 2 cabeças (80.449 parâmetros) |
| Treino           | A cada época, 60 janelas aceitas sorteadas de cada pessoa de treino; Adam (lr 1e-3, weight decay 1e-4), lotes de 64, BCE com peso de classe; early stopping com paciência 10 na perda por pessoa da validação interna, até 100 épocas                                                                                                                       |
| Modelos         | `D_FA` (principal: dimensões do artigo + FA), `D` (dimensões do artigo, sem FA), `D_pequeno` (dimensões reduzidas, sem FA) e as ablações do `D_pequeno`, sem FA: `D_sem_mixup` (H3), `D_sem_T` e `D_sem_E` (H2) |
| Frequency Attention | Atenção de canal do FcaNet (XU et al., 2026) sobre o mapa do 1º bloco temporal: 4 grupos de 16 filtros, cada grupo resumido por um componente DCT fixo (0, 35, 70 e 105, os 4 de menor frequência da grade 7 do FcaNet; o TFST-SDKA escolhe 16 componentes pelo desempenho, aqui eles são fixados a priori) → FC(64 → 16) → ReLU → FC(16 → 64) → sigmoid, que reescala cada filtro (docs/07_DTSF.md) |
| Hiperparâmetros | Fixados a priori, sem busca                                                                                                                                                                                                                                                                                                                                       |

## Baselines clássicos por modalidade

Resultados: docs/08_classicos.md.

**Características**, extraídas das mesmas janelas e tensores que alimentam o DTSF:

- **temporais (475):** 19 canais × 5 representações (o sinal base e as 4 bandas do ramo temporal) × 5 estatísticas: desvio, RMS, pico a pico, assimetria (g₁) e curtose em excesso (g₂), sem correção de viés (JOANES; GILL, 1998);
- **espectrais (1.102):** 19 canais × 29 frequências da STFT do ramo espectral × (média e desvio nos 16 quadros). Esse resumo perde a ordem dos quadros;
- **combinadas (1.577):** as 475 temporais e as 1.102 espectrais da mesma janela, concatenadas. É o baseline clássico que, como o DTSF, vê as duas modalidades ao mesmo tempo.

**Classificadores**, com grades pequenas para limitar o sobreajuste de seleção (CAWLEY; TALBOT, 2010):

| Classificador                 | Fixo                                          | Grade                                                          |
| ----------------------------- | --------------------------------------------- | -------------------------------------------------------------- |
| Regressão logística L2      | z-score                                       | C ∈ {0,01; 0,1; 1}                                            |
| Random Forest (BREIMAN, 2001) | 300 árvores, mtry = √p                      | Folha mínima ∈ {20; 5; 1} (PROBST; WRIGHT; BOULESTEIX, 2019) |
| KNN (COVER; HART, 1967)       | z-score, distância euclidiana, voto uniforme | k ∈ {101; 31; 11}                                             |

Lotte et al. (2018) apontam classificadores clássicos, inclusive o Random Forest, como especialmente úteis com poucos dados de treino em EEG.

**Ajuste e seleção:**

- **Pessoas e janelas de ajuste:** as pessoas de treino e validação do fold, com até 150 janelas aceitas por pessoa, sorteadas com a semente da tarefa.
- **Pesos:** na LR e no RF, cada janela pesa N / (2 · P_c · n_i), para que cada pessoa e cada classe pesem igual (P_c é o número de pessoas da classe e n_i o número de janelas da pessoa). O KNN não aceita pesos: cada janela pesa igual.
- **Seleção interna:** o hiperparâmetro é escolhido por validação cruzada interna de 3 partes sobre as pessoas, pela AUC por pessoa, sem tocar no teste. É a validação cruzada aninhada que Bradshaw et al. (2023) e Scheinost et al. (2019) recomendam.
- **Normalização:** o z-score é ajustado só nas janelas de quem está sendo ajustado.

**Previsão da pessoa:** média das probabilidades de todas as janelas aceitas sem overlap.

**Exposição a janelas:** os baselines ajustam com 22–25% mais pessoas que o DTSF, que reserva a validação para o early stopping.

**Referência adicional:** regressão logística L2 (C = 1) sobre a média por pessoa de 76 potências de banda (Welch; WELCH, 1967), como em Mumtaz et al. (2017) (docs/06_LR_psd.md). Ela não entra na H4.

## Protocolo estatístico

| Item               | Decisão                                                                                                                                                                                             | Por quê                                                                                                                                                                                             |
| ------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Validação        | StratifiedGroupKFold por pessoa, k = 5, repetido 10 vezes, com folds separados em cada dataset; 20% das pessoas de treino de cada fold formam a validação interna (`outputs/05_folds/folds.csv`) | Vazamento por janela infla resultados (BROOKSHIRE et al., 2024; LEE et al., 2023); amostras pequenas exigem repetições (VAROQUAUX, 2018); implementação no scikit-learn (PEDREGOSA et al., 2011) |
| Pareamento         | Mesmas partições e sementes para todos os modelos                                                                                                                                                  | Toda diferença entre modelos vem do modelo, não da partição                                                                                                                                      |
| Métrica primária | ROC-AUC por pessoa                                                                                                                                                                                   | Não depende de limiar e funciona com poucos casos                                                                                                                                                   |
| Estimativa         | Média da probabilidade fora do fold de cada pessoa nas 10 repetições → AUC sobre esses escores                                                                                                   | Com 5–6 pessoas por fold de teste na SZ, métricas por fold são muito ruidosas                                                                                                                     |
| Incerteza          | IC de 95% por bootstrap estratificado por pessoa (2000 reamostragens, percentil), mais média ± desvio das AUCs por repetição                                                                     | O IC mede a incerteza amostral; o desvio, a estabilidade em relação à partição                                                                                                                  |
| Secundárias       | Acurácia, acurácia balanceada, precisão, sensibilidade (recall), especificidade, F1 (paciente como classe positiva) e Macro F1 no limiar 0,5, só descritivas                                                                                                        | Evita multiplicar testes                                                                                                                                                                             |
| Contra o acaso     | DTSF: 25 permutações de rótulo com retreino, na repetição 0; p = (nº ≥ observado + 1) / (N + 1), menor p possível 1/26 ≈ 0,038                                                              | Só o retreino dá um teste válido contra o acaso                                                                                                                                                   |
| Reprodutibilidade  | Sementes fixas, partições salvas em arquivo, versões das bibliotecas registradas e planos registrados antes de cada rodada                                                                         | Permite refazer qualquer comparação exatamente                                                                                                                                                     |

## Pergunta científica e hipóteses

> Nas previsões fora do fold, o DTSF adaptado discrimina pacientes e controles acima do acaso? As duas modalidades juntas e o Mixup contribuem para isso? E ele discrimina melhor do que classificadores clássicos que, como ele, veem as duas modalidades da sua própria entrada ao mesmo tempo?

### Comparações

Os modelos se dividem em dois grupos. "Dimensão" é o tamanho de cada token (H): os três DTSF têm 33 tokens (16 temporais, 16 espectrais e o CLS), de dimensão 128 com as dimensões do artigo e 32 com as reduzidas.

**Comparações principais**

| Modelo | O que é | Papel |
| --- | --- | --- |
| `D_FA` | **Principal.** DTSF (dois ramos, temporal e espectral) com as dimensões do artigo (dimensão 128) e a Frequency Attention | H1 e H4 |
| `D_pequeno` | DTSF adaptado do artigo com as dimensões reduzidas (dimensão 32), sem FA | Referência de H2 e H3; descritivo |
| `D` | DTSF com as dimensões do artigo (dimensão 128), sem FA | Descritivo |
| `KNN_combinado`, `LR_combinado`, `RF_combinado` | KNN, regressão logística e Random Forest sobre as características temporais e espectrais concatenadas ("dual branch") | H4 |

Comparações descritivas entre os DTSF, sem decisão: `D_FA` − `D` (efeito da FA), `D` − `D_pequeno` (efeito do tamanho) e `D_FA` − `D_pequeno`.

**Ablações e referências secundárias**

As ablações do DTSF são pré-fixadas nas dimensões reduzidas (dimensão 32) e derivam do `D_pequeno`, sem FA:

| Modelo | O que é | Papel |
| --- | --- | --- |
| `D_sem_mixup` | `D_pequeno` sem Mixup | H3 |
| `D_sem_T` | `D_pequeno` só com o ramo espectral | H2 |
| `D_sem_E` | `D_pequeno` só com o ramo temporal | H2 |
| `LR_psd_absolutas`, `LR_psd_relativas` | Regressão logística sobre as 76 features de potência absoluta ou relativa (19 canais × 4 bandas; etapa 06) | Referência, com o próprio teste contra o acaso |
| `KNN_temporal`, `LR_temporal`, `RF_temporal` | Clássicos sobre as 475 características temporais | Descritivo: Δ, IC e p contra o `D_FA`, sem decisão |
| `KNN_espectral`, `LR_espectral`, `RF_espectral` | Clássicos sobre as 1.102 características espectrais | Descritivo: Δ, IC e p contra o `D_FA`, sem decisão |

Todas as hipóteses são avaliadas **em cada transtorno separadamente** (SZ × HC e MDD × HC), nas mesmas pessoas e partições, sobre a AUC por pessoa.

| Hipótese                               | Enunciado                                                                                                              | Comparação                                               | Decisão num transtorno                 |
| --------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- | --------------------------------------- |
| H1 — acima do acaso                    | O DTSF adaptado discrimina pacientes e controles acima do acaso                                                        | AUC do `D_FA` contra 25 AUCs com rótulos permutados e retreino | p de permutação ≤ 0,05               |
| H2 — complementaridade das modalidades | O DTSF com os dois ramos discrimina melhor do que com um ramo só                                                      | `D_pequeno` > `D_sem_T` **e** `D_pequeno` > `D_sem_E` | Interseção-união das 2 comparações |
| H3 — Mixup                             | O Mixup no treino melhora a discriminação do DTSF                                                                    | `D_pequeno` > `D_sem_mixup`                              | Uma comparação                        |
| H4 — DTSF × baselines clássicos      | O DTSF discrimina melhor do que **cada um** dos três baselines clássicos combinados (LR, RF e KNN sobre as características temporais e espectrais concatenadas) | `D_FA` > cada baseline combinado                             | Interseção-união das 3 comparações |

H1 e H4 são sobre o modelo principal, o `D_FA`. As ablações derivam do `D_pequeno`, e cada uma muda uma coisa só em relação a ele; por isso H2 e H3 comparam o `D_pequeno` com elas. Formalmente, em H2–H4, cada comparação testa H₀: ΔAUC ≤ 0 contra H₁: ΔAUC > 0, com ΔAUC = AUC(referência) − AUC(outro modelo), sendo a referência o `D_pequeno` em H2 e H3 e o `D_FA` em H4. Em H1, H₀ é que a AUC do `D_FA` é compatível com a distribuição de AUCs com rótulos permutados.

Comparações descritivas, sem decisão: `D_FA` − `D` (efeito da FA), `D` − `D_pequeno` (efeito do tamanho) e `D_FA` − `D_pequeno`.

**Teste e interpretação:**

- **H1:** 25 permutações dos rótulos com retreino completo, na repetição 0 (ver o protocolo estatístico). O menor p possível é 1/26 ≈ 0,038, então **a H1 é decidida em cada transtorno a α = 0,05, sem Holm entre SZ e MDD**: com Holm, o primeiro passo exigiria p ≤ 0,025, impossível com 25 permutações. A H1 condiciona a leitura das demais: o resumo traz H2 e H3 de qualquer forma, mas num transtorno em que a H1 não for rejeitada elas não são interpretadas, porque um modelo que não supera o acaso não tem componente a analisar.
- **Cada comparação** de H2–H4 é testada pelo ΔAUC por pessoa, com permutação pareada sobre os ranks de cada modelo (10.000 permutações), unilateral. Os ranks tornam o teste insensível às escalas muito diferentes dos escores dos modelos.
- **Num transtorno**, H2 e H4 exigem vencer todas as suas comparações (interseção-união, com p = maior dos p; BERGER, 1982).
- **Entre SZ e MDD**, H2–H4 são decididas por Holm (HOLM, 1979), com α = 0,05, separadamente para cada hipótese.
- **Entre hipóteses**, não há correção: cada uma responde a uma pergunta diferente, e a taxa de erro da família H1–H4 não é controlada. Isso é declarado na leitura dos resultados.
- **Ablações que mudam mais de uma coisa:** `D_sem_T` e `D_sem_E` mudam também o número de tokens e de parâmetros. Por isso a H2 compara a arquitetura com e sem o ramo, e não isola o ramo puro. O `D_sem_mixup` muda só o Mixup.
- **IC do ΔAUC:** bootstrap estratificado e pareado, condicional às previsões obtidas.
- **Sensibilidades**, que nunca decidem: o Δ por repetição e o t corrigido de Nadeau e Bengio (2003), que leva em conta a dependência entre os folds.
- **Descritivo:** o `D_FA` contra os 6 baselines de uma modalidade só (temporais e espectrais), com Δ, IC e p, e a interseção-união por modalidade; nada disso decide a H4.
- **Validade:** a interseção-união e o Holm controlam só a multiplicidade. A validade de cada permutação **não está garantida**, porque a validação cruzada introduz dependência: as pessoas de um fold compartilham o modelo, os treinos se sobrepõem e o escore é a média de repetições com partições sobrepostas. O p vale condicionado às previsões obtidas; para a afirmação sobre o método, o erro tipo I real pode passar de α (\~10–13% numa simulação com dois algoritmos equivalentes).
- **Leitura:** um resultado não significativo é inconclusivo e não demonstra equivalência. O poder é baixo, sobretudo na SZ, com 28 pessoas.

## Resultados

Pendentes da rodada completa com a semente 20261108. Os resultados de cada etapa ficam nos docs da etapa (docs/06_LR_psd.md, docs/07_DTSF.md e docs/08_classicos.md).

## Resumo das pendências

| Área                | Pendência                                                                                                                                                                                                                                                                    |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Artefatos            | Decidir com o orientador se o limiar de 150 µV fica. A conferência no sinal de 2–30 Hz, em µV, está feita (docs/03_janelas.md): ele rejeita 3,6% das janelas                                                                                                             |
| Baselines clássicos | As permutações contra o acaso não foram executadas; decidir se entram, como checagem descritiva                                                                                                                                                                            |
| Discussão           | Levar ao orientador: pelo cabeçalho EDF, a data de gravação se confunde com o diagnóstico nos dois datasets (SZ: HC em 2003, 11 de 14 SZ em 2004–2008; MDD: só 4 HC no período dos pacientes) (docs/02_harmonizacao.md). A etapa 09 mede o tamanho do problema |
| Literatura           | Confirmar a lacuna com busca sistemática (PubMed, IEEE Xplore, Scopus)                                                                                                                                                                                                       |

## Trabalhos relacionados

Levantamento preliminar, organizado pelo que cada grupo de trabalhos contribui para a proposta. Citações no formato autor-data (ABNT); a lista completa está em Referências.

### Trabalhos com os mesmos datasets

**Schizophrenia\_dataset (RepOD).** Publicado por Olejarczyk e Jernajczyk (2017): 14 pacientes com esquizofrenia paranoide internados no Instituto de Psiquiatria e Neurologia de Varsóvia e 14 controles pareados por sexo e idade, com 15 min de EEG em repouso de olhos fechados, 19 canais e 250 Hz. O artigo original não faz classificação: estuda conectividade funcional (PLV, PLI, DTF) com índices de teoria dos grafos.

Dezenas de trabalhos posteriores classificam esse dataset com acurácias entre 95% e 100%, por exemplo com redes recorrentes sobre features não lineares (NIKHIL CHANDRAN; SREEKUMAR; SUBHA, 2021) e com CNN-LSTM sobre o sinal (SHOEIBI et al., 2021, que relatam 99,25%). Muitos avaliam por janela, com janelas da mesma pessoa em treino e teste; esses números **não servem de referência** para um split por pessoa (ver Vazamento e validação).

**Depression\_dataset (Mumtaz, Figshare).** Coletado no Hospital Universiti Sains Malaysia, com 5 min de EC e EO, 19 canais e referência linked-ears (MUMTAZ, 2016). O artigo de referência descreve 33 MDD e 30 HC (MUMTAZ et al., 2017); outros trabalhos citam 34 MDD. Mumtaz et al. (2017) usaram potência por banda e assimetria inter-hemisférica de alfa com classificadores clássicos; Mumtaz e Qayyum (2019) usaram 1D-CNN e CNN-LSTM sobre o EEG bruto.

**M-MDD (WANG et al., 2025)** também funde features temporais, espectrais e espaciais nos datasets Mumtaz2016 e Arizona2020, com validação independente de sujeito. Difere deste TCC por somar uma tarefa auxiliar contrastiva de robustez a ruído e por tratar só de depressão.

### Arquitetura

O DTSF (WANG et al., 2026) foi proposto para iEEG de um canal, com 3 classes, e combina um ramo temporal multibanda e um ramo de STFT, fundidos por Transformer, com Mixup. A adaptação deste TCC mantém a estrutura (dois ramos CNN-GRU, fusão por atenção sobre as sequências e Mixup) e muda a entrada (EEG de escalpo, 19 canais), a saída (binária), os tamanhos e o protocolo de avaliação. As escolhas que o artigo não especifica estão na seção DTSF adaptado.

### Esquizofrenia e depressão no mesmo estudo

Yan et al. (2023) treinaram uma rede de convoluções 1D multiescala seguida de GRU em EEG de repouso de 327 pessoas de um único hospital (esquizofrenia, bipolar, depressão e controles). Relatam 91,3% em SZ × HC, 84,9% em MDD × HC e 68,2% em 4 classes. A combinação de convoluções e GRU é próxima do ramo temporal do DTSF; não há ramo espectral, e o texto não deixa claro se a validação separa por pessoa.

A justificativa biológica para o ramo espectral vem de Newson e Thiagarajan (2019), que revisaram 184 estudos de EEG em repouso em dez transtornos psiquiátricos. O padrão mais frequente foi aumento de potência em frequências baixas (delta, teta) e redução em frequências altas (alfa, beta, gama). Alterações espectrais são, portanto, marcadores esperados nos dois transtornos.

### Site, harmonização e generalização entre datasets

Hollenbenders, Maier e Reichenbach (2023) harmonizaram o dataset do Mumtaz com o MODMA usando um pipeline muito parecido com o deste TCC: canais em comum, referência média, 250 Hz e filtro de 1–40 Hz. Um classificador linear separou os dois datasets em quase todas as condições testadas, e as normalizações que apagavam essa diferença também pioravam o diagnóstico. Como cada modelo deste TCC vê um único dataset, isso não contamina o treino, mas impede comparar diretamente os números de SZ e MDD.

Liu et al. (2023) avaliaram transferência entre dois datasets de esquizofrenia em nível de indivíduo e obtiveram cerca de 62% com adaptação de domínio (TCA) e 69% combinando fine-tuning e TCA. Isso dá uma ideia de quanto o desempenho cai fora do site de origem.

### Vazamento, validação e seleção de modelos

Brookshire et al. (2024) mostraram que dividir por segmento, e não por pessoa, superestima fortemente o desempenho em pessoas novas, e que a maioria dos estudos translacionais de deep learning com EEG revisados tinha esse problema. Lee et al. (2023) encontraram, em diagnóstico de TEPT por EEG com recorte em janelas e EEGNet, acurácias infladas em cerca de 15–20 pontos quando a validação era por janela em vez de por sujeito. Os dois fundamentam a separação por pessoa.

Varoquaux (2018) mostra que, com amostras pequenas, as barras de erro da validação cruzada são largas (da ordem de ±10% com 100 amostras) e que o desvio entre folds as subestima. Com no máximo 28 pessoas na task SZ, isso justifica repetir a validação, reportar a variabilidade e evitar conclusões a partir de diferenças pequenas. Para comparar modelos sob reamostragem, Nadeau e Bengio (2003) propõem um teste t corrigido.

Scheinost et al. (2019) recomendam normalização e seleção de características dentro do laço da validação cruzada, e validação cruzada aninhada ou correção para múltiplas comparações quando vários modelos e parâmetros são testados. Bradshaw et al. (2023) descrevem a validação cruzada aninhada para escolher hiperparâmetros sem usar o teste e lembram que as rodadas da validação cruzada não são independentes. Cawley e Talbot (2010) mostram que o sobreajuste da seleção de modelos pode ter a magnitude das diferenças entre algoritmos. Os três fundamentam a seleção interna e as grades pequenas dos baselines clássicos.

### Posicionamento do TCC

Arquiteturas profundas que fundem informação temporal e espectral já aparecem na literatura (WANG et al., 2025; WANG et al., 2026), então a contribuição não é a arquitetura em si. É testar, nos dois transtornos, com o mesmo pipeline e validação estrita por pessoa, se uma dessas arquiteturas supera classificadores clássicos aplicados às mesmas entradas, com uma modalidade de cada vez ou com as duas juntas. Antes da qualificação, convém confirmar a lacuna com busca sistemática (PubMed, IEEE Xplore, Scopus), com termos como *temporal-spectral fusion*, *transformer*, *STFT*, *EEG*, *schizophrenia*, *depression* e *classical baselines*.

## Referências

BERGER, R. L. Multiparameter hypothesis testing and acceptance sampling. **Technometrics**, v. 24, n. 4, p. 295–300, 1982.

BRADSHAW, T. J. et al. A guide to cross-validation for artificial intelligence in medical imaging. **Radiology: Artificial Intelligence**, v. 5, n. 4, e220232, 2023. DOI: [10.1148/ryai.220232](https://doi.org/10.1148/ryai.220232).

BREIMAN, L. Random forests. **Machine Learning**, v. 45, n. 1, p. 5–32, 2001. DOI: [10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324).

BROOKSHIRE, G. et al. Data leakage in deep learning studies of translational EEG. **Frontiers in Neuroscience**, v. 18, art. 1373515, 2024. DOI: [10.3389/fnins.2024.1373515](https://doi.org/10.3389/fnins.2024.1373515).

CAWLEY, G. C.; TALBOT, N. L. C. On over-fitting in model selection and subsequent selection bias in performance evaluation. **Journal of Machine Learning Research**, v. 11, p. 2079–2107, 2010.

COVER, T.; HART, P. Nearest neighbor pattern classification. **IEEE Transactions on Information Theory**, v. 13, n. 1, p. 21–27, 1967. DOI: [10.1109/TIT.1967.1053964](https://doi.org/10.1109/TIT.1967.1053964).

GRAMFORT, A. et al. MEG and EEG data analysis with MNE-Python. **Frontiers in Neuroscience**, v. 7, art. 267, 2013. DOI: [10.3389/fnins.2013.00267](https://doi.org/10.3389/fnins.2013.00267).

HOLLENBENDERS, Y.; MAIER, C.; REICHENBACH, A. Multiverse analysis for robust depression biomarkers based on electroencephalography. In: JAHRESTAGUNG DER DEUTSCHEN GESELLSCHAFT FÜR MEDIZINISCHE INFORMATIK, BIOMETRIE UND EPIDEMIOLOGIE (GMDS), 68., 2023, Heilbronn. Düsseldorf: German Medical Science GMS Publishing House, 2023. DocAbstr. 234. DOI: [10.3205/23gmds125](https://doi.org/10.3205/23gmds125).

HOLM, S. A simple sequentially rejective multiple test procedure. **Scandinavian Journal of Statistics**, v. 6, n. 2, p. 65–70, 1979.

JOANES, D. N.; GILL, C. A. Comparing measures of sample skewness and kurtosis. **The Statistician**, v. 47, n. 1, p. 183–189, 1998. DOI: [10.2307/2988433](https://doi.org/10.2307/2988433).

LEE, H.-T. et al. Risk of data leakage in estimating the diagnostic performance of a deep-learning-based computer-aided system for psychiatric disorders. **Scientific Reports**, v. 13, art. 16633, 2023. DOI: [10.1038/s41598-023-43542-8](https://doi.org/10.1038/s41598-023-43542-8).

LIU, T. et al. Transfer learning based efficient schizophrenia classification on electroencephalogram (EEG) signals: a cross-dataset study. In: INTERNATIONAL CONFERENCE ON BIOINFORMATICS AND BIOMEDICAL TECHNOLOGY, 15., 2023. **Proceedings** \[...\]. New York: ACM, 2023. DOI: [10.1145/3608164.3608184](https://doi.org/10.1145/3608164.3608184).

LOTTE, F. et al. A review of classification algorithms for EEG-based brain–computer interfaces: a 10 year update. **Journal of Neural Engineering**, v. 15, n. 3, art. 031005, 2018. DOI: [10.1088/1741-2552/aab2f2](https://doi.org/10.1088/1741-2552/aab2f2).

MUMTAZ, W. **MDD patients and healthy controls EEG data (new)**. Figshare, 2016. Conjunto de dados. DOI: [10.6084/m9.figshare.4244171.v2](https://doi.org/10.6084/m9.figshare.4244171.v2).

MUMTAZ, W. et al. Electroencephalogram (EEG)-based computer-aided technique to diagnose major depressive disorder (MDD). **Biomedical Signal Processing and Control**, v. 31, p. 108–115, 2017. DOI: [10.1016/j.bspc.2016.07.006](https://doi.org/10.1016/j.bspc.2016.07.006).

MUMTAZ, W.; QAYYUM, A. A deep learning framework for automatic diagnosis of unipolar depression. **International Journal of Medical Informatics**, v. 132, art. 103983, 2019. DOI: [10.1016/j.ijmedinf.2019.103983](https://doi.org/10.1016/j.ijmedinf.2019.103983).

NADEAU, C.; BENGIO, Y. Inference for the generalization error. **Machine Learning**, v. 52, n. 3, p. 239–281, 2003. DOI: [10.1023/A:1024068626366](https://doi.org/10.1023/A:1024068626366).

NEWSON, J. J.; THIAGARAJAN, T. C. EEG frequency bands in psychiatric disorders: a review of resting state studies. **Frontiers in Human Neuroscience**, v. 12, art. 521, 2019. DOI: [10.3389/fnhum.2018.00521](https://doi.org/10.3389/fnhum.2018.00521).

NIKHIL CHANDRAN, A.; SREEKUMAR, K.; SUBHA, D. P. EEG-based automated detection of schizophrenia using long short-term memory (LSTM) network. In: PATNAIK, S.; YANG, X.-S.; SETHI, I. K. (ed.). **Advances in Machine Learning and Computational Intelligence**. Singapore: Springer, 2021. (Algorithms for Intelligent Systems). DOI: [10.1007/978-981-15-5243-4\_19](https://doi.org/10.1007/978-981-15-5243-4_19).

OLEJARCZYK, E.; JERNAJCZYK, W. **EEG in schizophrenia**. RepOD, 2017. Conjunto de dados. DOI: [10.18150/repod.0107441](https://doi.org/10.18150/repod.0107441).

OLEJARCZYK, E.; JERNAJCZYK, W. Graph-based analysis of brain connectivity in schizophrenia. **PLoS ONE**, v. 12, n. 11, e0188629, 2017. DOI: [10.1371/journal.pone.0188629](https://doi.org/10.1371/journal.pone.0188629).

PEDREGOSA, F. et al. Scikit-learn: machine learning in Python. **Journal of Machine Learning Research**, v. 12, p. 2825–2830, 2011.

PROBST, P.; WRIGHT, M. N.; BOULESTEIX, A.-L. Hyperparameters and tuning strategies for random forest. **WIREs Data Mining and Knowledge Discovery**, 2019. DOI: [10.1002/widm.1301](https://doi.org/10.1002/widm.1301).

SCHEINOST, D. et al. Ten simple rules for predictive modeling of individual differences in neuroimaging. **NeuroImage**, v. 193, p. 35–45, 2019. DOI: [10.1016/j.neuroimage.2019.02.057](https://doi.org/10.1016/j.neuroimage.2019.02.057).

SHOEIBI, A. et al. Automatic diagnosis of schizophrenia in EEG signals using CNN-LSTM models. **Frontiers in Neuroinformatics**, v. 15, art. 777977, 2021. DOI: [10.3389/fninf.2021.777977](https://doi.org/10.3389/fninf.2021.777977).

VAROQUAUX, G. Cross-validation failure: small sample sizes lead to large error bars. **NeuroImage**, v. 180, p. 68–77, 2018. DOI: [10.1016/j.neuroimage.2017.06.061](https://doi.org/10.1016/j.neuroimage.2017.06.061).

WANG, K. et al. DTSF: a dual-branch temporal and spectral fusion model with data augmentation for intracranial EEG classification. **Biomedical Signal Processing and Control**, v. 112, art. 108507, 2026. DOI: [10.1016/j.bspc.2025.108507](https://doi.org/10.1016/j.bspc.2025.108507).

WANG, Y. et al. M-MDD: a multi-task deep learning framework for major depressive disorder diagnosis using EEG. **Neurocomputing**, v. 636, art. 130008, 2025. DOI: [10.1016/j.neucom.2025.130008](https://doi.org/10.1016/j.neucom.2025.130008).

WELCH, P. The use of fast Fourier transform for the estimation of power spectra: a method based on time averaging over short, modified periodograms. **IEEE Transactions on Audio and Electroacoustics**, v. 15, n. 2, p. 70–73, 1967. DOI: [10.1109/TAU.1967.1161901](https://doi.org/10.1109/TAU.1967.1161901).

YAN, W. et al. Multi-scale convolutional recurrent neural network for psychiatric disorder identification in resting-state EEG. **Frontiers in Psychiatry**, v. 14, art. 1202049, 2023. DOI: [10.3389/fpsyt.2023.1202049](https://doi.org/10.3389/fpsyt.2023.1202049).
