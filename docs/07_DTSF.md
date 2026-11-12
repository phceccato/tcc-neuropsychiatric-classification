# DTSF adaptado — resultados

O DTSF adaptado é a arquitetura principal do TCC. Ele tem dois ramos CNN-GRU:
- um ramo temporal, sobre o sinal de 2–30 Hz e as bandas δ, θ, α e β;
- um ramo espectral, sobre a STFT do sinal em µV.

Os ramos são fundidos por um Transformer sobre as sequências, com token de classificação, e o treino usa Mixup. A arquitetura é adaptada de Wang et al. (2026). As decisões estão em [00_proposta-tcc.md](00_proposta-tcc.md).

A etapa roda seis modelos ([Modelos](#modelos)):

| Modelo | O que é | Parâmetros | Uso |
| --- | --- | --- | --- |
| `D_FA` | **Principal.** DTSF com as dimensões do artigo e a Frequency Attention (FA) no ramo temporal | 704.769 | H1, H4 |
| `D_pequeno` | DTSF com as dimensões reduzidas (16 filtros, dimensão 32), sem FA | 80.449 | descritivo (efeito do tamanho) |
| `D` | DTSF com as dimensões do artigo, sem FA | 702.721 | H2, H3; descritivo (efeito da FA) |
| `D_sem_mixup` | O `D` sem Mixup | 702.721 | H3 |
| `D_sem_T` | O `D` só com o ramo espectral | 444.353 | H2 |
| `D_sem_E` | O `D` só com o ramo temporal | 507.713 | H2 |

H2 e H3 usam o `D`, e não o `D_FA`, porque as ablações derivam dele: cada uma muda uma coisa em relação ao modelo com que é comparada.

Resultados: pendentes da rodada com a semente 20261108 ([Resultados](#resultados)). A comparação com os baselines clássicos (H4) está em [08_classicos.md](08_classicos.md).

## Código

- Código: [pipeline/e07_DTSF/](../pipeline/e07_DTSF/)
  - `dados.py`: cache do sinal a 125 Hz e seleção das janelas (também usados pelos baselines clássicos);
  - `dados_dtsf.py`: cache multibanda e STFT;
  - `redes_dtsf.py`: os seis modelos (`MODELOS`, com o principal em `PRINCIPAL`) e a Frequency Attention (`AtencaoFrequencia`, também usada pela etapa 12);
  - `treino.py`: o treino de um fold;
  - `rodar.py`: o runner;
  - `avaliar.py`: o resumo do DTSF, as decisões de H1, H2 e H3 (`decidir_h1`, `decidir_hipotese`), as comparações descritivas (`D_FA` − `D`, `D` − `D_pequeno`, `D_FA` − `D_pequeno`) e a comparação pareada por pessoa (`comparacao_pareada`), usada por H2, H3 e pela H4.
- Saídas em `outputs/07_DTSF/`:
  - `cache/` e `cache_dtsf/`: o sinal a 125 Hz e o cache multibanda com a STFT;
  - `resultados/<conjunto>/<modelo>/r<rep>_f<fold>.csv`: previsões das pessoas de teste;
  - `resultados/<conjunto>/<modelo>/r<rep>_f<fold>.json`: época escolhida, dispositivo, número de parâmetros e histórico de perdas;
  - `permutacoes/<conjunto>/<modelo>/p<k>_f<fold>.{csv,json}`: o mesmo para cada permutação de rótulos (só a repetição 0);
  - `resultados/resumo_dtsf.md`: o resumo gerado por `avaliar`.

```bash
venv/bin/python -m pipeline.e07_DTSF.dados                                                      # cache a 125 Hz, uma vez
venv/bin/python -m pipeline.e07_DTSF.dados_dtsf                                                 # cache multibanda, uma vez
venv/bin/python -m pipeline.e07_DTSF.rodar 2>&1 | tee -a outputs/07_DTSF/log_dtsf.txt                                         # os 6 modelos: 600 treinos; retoma
venv/bin/python -m pipeline.e07_DTSF.rodar --permutacoes 25 2>&1 | tee -a outputs/07_DTSF/log_dtsf_permutacoes.txt              # D_FA contra o acaso: 250 treinos
venv/bin/python -m pipeline.e07_DTSF.avaliar 2>&1 | tee -a outputs/07_DTSF/log_dtsf_avaliar.txt                                                          # resumo_dtsf.md e H1–H3
```

### CPU ou GPU

Por padrão os treinos rodam na CPU, com 1 thread por processo; as rodadas registradas foram feitas assim. Numa máquina com GPU NVIDIA e o PyTorch com CUDA, `--dispositivo cuda` treina na GPU:

```bash
venv/bin/python -m pipeline.e07_DTSF.rodar --dispositivo cuda --processos 14 2>&1 | tee -a outputs/07_DTSF/log_dtsf.txt
venv/bin/python -m pipeline.e07_DTSF.rodar --permutacoes 25 --dispositivo cuda --processos 14 2>&1 | tee -a outputs/07_DTSF/log_dtsf_permutacoes.txt
```

- Os processos dividem a mesma GPU. O sorteio das janelas, o Mixup e a STFT continuam na CPU, com os mesmos geradores, e o lote só depois vai para a GPU; por isso vale usar quase todos os núcleos em `--processos`.
- Com `cuda`, os processos são criados com `spawn` (a GPU não sobrevive a `fork`) e a cuDNN fica em modo determinístico.
- **A GPU não reproduz bit a bit as contas da CPU.** Uma rodada em GPU dá números próximos, mas não idênticos aos de uma rodada em CPU (a época escolhida pelo early stopping pode mudar). Cada `.json` registra o dispositivo, e o runner se recusa a completar uma pasta que já tem tarefas feitas em outro dispositivo: para trocar, use outra `--saida` ou apague as tarefas antigas.
- Na CPU com 1 thread, um fold retreinado do modelo de 80 mil parâmetros (hoje `D_pequeno`; SZ, repetição 0, fold 0) dá probabilidades idênticas às gravadas na 2ª rodada (diferença máxima 0).
- A etapa 09 (`explicar` e `lote`) continua treinando o D na CPU.

## Modelos

Arquitetura comum, com F filtros, dimensão H e C cabeças:

| Parte | Arquitetura |
| --- | --- |
| Ramo temporal | (19, 5, 500) → Conv2d(19→F, (5, 13)) → BN → GELU → MaxPool (1, 2) → Dropout 0,5 → **[FA, só no `D_FA`]** → Conv2d(F→F, (1, 25)) → BN → GELU → AdaptiveMaxPool (1, 16) → Dropout 0,5 → GRU H → 16 tokens |
| Ramo espectral | (19, 29, 16), z-score por (canal, frequência) com estatísticas do treino do fold → Conv2d(19→F, (29, 3)) → BN → GELU → Dropout 0,5 → Conv2d(F→F, (1, 3)) → BN → GELU → Dropout 0,5 → GRU H → 16 tokens |
| Fusão | 32 tokens → Linear H → token CLS no fim → posição aprendível → 2 blocos pre-LN (MHSA de C cabeças; FFN com gate GELU) → LN → último token → Linear → logit |

| Dimensões | F | H | C | Blocos | Modelos |
| --- | --- | --- | --- | --- | --- |
| Do artigo (WANG et al., 2026, Tabela 1) | 64 | 128 | 4 | 2 | `D_FA`, `D`, `D_sem_mixup`, `D_sem_T`, `D_sem_E` |
| Reduzidas | 16 | 32 | 2 | 2 | `D_pequeno` |

A Tabela 1 do artigo é ambígua nos filtros ("Conv2d(64, …)" na coluna de parâmetros e 256 canais na coluna de formas): vale a coluna de parâmetros. Os kernels, os 16 passos por ramo e as entradas são os mesmos nas duas dimensões, adaptados ao EEG de escalpo de 19 canais e 4 s. O treino (abaixo) é igual para os seis modelos. Os tokens são 33 (16 + 16 + CLS) em todos, menos nas ablações de um ramo (17); o que muda entre as dimensões é o tamanho de cada token (H).

**Frequency Attention (`D_FA`).** É a atenção de canal do FcaNet, como no TFST-SDKA (XU et al., 2026), aplicada ao mapa [B, 64, 1, 250] que sai do 1º bloco temporal:
- os 64 filtros são divididos em 4 grupos de 16, e cada grupo é resumido por um componente DCT ao longo do tempo: 0, 35, 70 e 105. São os 4 de menor frequência da grade 7 do FcaNet, u · ⌊250/7⌋, fixados a priori. O componente 0 é o GAP vezes √250;
- os 64 resumos passam por FC(64 → 16) → ReLU → FC(16 → 64) → sigmoid, e cada filtro é multiplicado pelo seu peso (2.048 parâmetros a mais);
- as "frequências" são as do mapa de características ao longo da janela, não bandas do EEG em Hz;
- a FA fica só no ramo temporal: a primeira convolução do ramo espectral já junta as 29 frequências;
- a FA é criada por último, então, na mesma semente, o `D_FA` começa com os mesmos pesos do `D`, e `D_FA` − `D` isola a FA.

| Ablação do `D` | O que muda |
| --- | --- |
| D_sem_T | Sem o ramo temporal: o Transformer recebe só os 16 tokens espectrais e o CLS |
| D_sem_E | Sem o ramo espectral |
| D_sem_mixup | Sem Mixup |

**Entradas.**
- **Ramo temporal:** o sinal de 2–30 Hz com z-score por canal da própria pessoa, reamostrado para 125 Hz, e as 4 bandas (Butterworth de ordem 4, `sosfiltfilt`) filtradas na gravação inteira.
- **Ramo espectral:** log₁₀(|X|² + 10⁻¹²) da STFT da janela em µV (Hann de 125 amostras, passo de 25), em 29 bins de 2 a 30 Hz e 16 quadros. Fica sem o z-score por pessoa, porque a potência absoluta carrega informação.

**Treino.**
- A cada época, o treino sorteia 60 janelas aceitas de cada pessoa.
- Usa Adam (lr 1e-3, weight decay 1e-4), lotes de 64 e BCE com peso de classe.
- Mixup com λ ~ Beta(1, 1), um λ por lote, com o mesmo λ e a mesma permutação nos dois ramos e no alvo, só no treino e com gerador próprio.
- O early stopping tem paciência 10, mede a perda média por pessoa da validação interna e vai até no máximo 100 épocas.
- A previsão da pessoa é a média das probabilidades das suas janelas aceitas sem overlap.
- A semente depende só de (repetição, fold).

## Implementação e validação mínima

- **Equivalência com as redes anteriores,** com a mesma semente e as mesmas entradas, saídas idênticas:
  - o `D_pequeno` é o D de 80 mil parâmetros das rodadas anteriores;
  - o `D` é o D_artigo, e o `D_FA` é o D_artigo_FA da etapa 14;
  - as variantes da etapa 12 não mudaram.

  O treino também é o mesmo (mesma configuração e sementes por repetição e fold). Por isso, previsões já gravadas desses modelos no mesmo dispositivo podem ser copiadas para `outputs/07_DTSF/resultados/<conjunto>/<modelo>/`, e o runner pula as tarefas já gravadas.

- Com dados reais, a linha 0 do cache multibanda é idêntica ao cache a 125 Hz nas 84 pessoas.
- A STFT da fonte bate com a STFT do sinal em µV reamostrado diretamente, com erro máximo de 1e-4 em log10.
- **Bandas:** nas janelas aceitas, a soma da potência das 4 bandas cobre 0,84–0,99 da potência do sinal.
  - A perda é explicada pela resposta dos filtros (observado/previsto = 1,01), e as bandas foram mantidas.
  - Um número anterior, de 0,57, estava errado: tinha sido calculado incluindo os trechos rejeitados.
- Ringing de artefatos vizinhos aparece em 14 de 21.872 janelas aceitas (0,06%).
- **Pilotos de custo, com rótulos embaralhados e saída fora de `outputs/`:**
  - 1 processo: 113 s, com ~10,3 s por época;
  - 10 processos: 6 min 8 s, com ~20 s por época na SZ e ~25 s no MDD.

## Resultados

Pendentes da rodada com a semente 20261108.

## Análise anterior

Antes do DTSF, o projeto avaliou outra arquitetura (LSTM). O código dela foi removido, a pedido do autor, e ela não faz parte da pesquisa.
