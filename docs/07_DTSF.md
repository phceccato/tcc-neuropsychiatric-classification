# DTSF adaptado — resultados

O DTSF adaptado (modelo D) é a arquitetura principal do TCC. Ele tem dois ramos CNN-GRU:
- um ramo temporal, sobre o sinal de 2–30 Hz e as bandas δ, θ, α e β;
- um ramo espectral, sobre a STFT do sinal em µV.

Os ramos são fundidos por um Transformer sobre as sequências, com token de classificação, e o treino usa Mixup. A arquitetura é adaptada de Wang et al. (2026). As decisões estão em [00_proposta-tcc.md](00_proposta-tcc.md).

Resultados: pendentes da rodada com a semente 20261108 ([Resultados](#resultados)). A comparação com os baselines clássicos (H4) está em [08_classicos.md](08_classicos.md).

## Código

- Código: [pipeline/e07_DTSF/](../pipeline/e07_DTSF/)
  - `dados.py`: cache do sinal a 125 Hz e seleção das janelas (também usados pelos baselines clássicos);
  - `dados_dtsf.py`: cache multibanda e STFT;
  - `redes_dtsf.py`: o D e as ablações (as dimensões da rede são parâmetros, com as do D como padrão; a etapa 14 usa as do artigo);
  - `treino.py`: o treino de um fold;
  - `rodar.py`: o runner;
  - `avaliar.py`: o resumo do DTSF, as decisões de H1, H2 e H3 (`decidir_h1`, `decidir_hipotese`) e a comparação pareada por pessoa (`comparacao_pareada`), usada por H2, H3 e pela H4.
- Saídas em `outputs/07_DTSF/`:
  - `cache/` e `cache_dtsf/`: o sinal a 125 Hz e o cache multibanda com a STFT;
  - `resultados/<conjunto>/<modelo>/r<rep>_f<fold>.csv`: previsões das pessoas de teste;
  - `resultados/<conjunto>/<modelo>/r<rep>_f<fold>.json`: época escolhida e histórico de perdas;
  - `permutacoes/<conjunto>/<modelo>/p<k>_f<fold>.{csv,json}`: o mesmo para cada permutação de rótulos (só a repetição 0);
  - `resultados/resumo_dtsf.md`: o resumo gerado por `avaliar`.

```bash
venv/bin/python -m pipeline.e07_DTSF.dados                                                      # cache a 125 Hz, uma vez
venv/bin/python -m pipeline.e07_DTSF.dados_dtsf                                                 # cache multibanda, uma vez
venv/bin/python -m pipeline.e07_DTSF.rodar --modelos D D_sem_T D_sem_E D_sem_mixup 2>&1 | tee -a outputs/07_DTSF/log_dtsf.txt              # 500 treinos; retoma
venv/bin/python -m pipeline.e07_DTSF.rodar --modelos D --permutacoes 25 2>&1 | tee -a outputs/07_DTSF/log_dtsf_permutacoes.txt                          # 250 treinos contra o acaso
venv/bin/python -m pipeline.e07_DTSF.avaliar 2>&1 | tee -a outputs/07_DTSF/log_dtsf_avaliar.txt                                                          # resumo_dtsf.md e H1–H3
```

### CPU ou GPU

Por padrão os treinos rodam na CPU, com 1 thread por processo; as rodadas registradas foram feitas assim. Numa máquina com GPU NVIDIA e o PyTorch com CUDA, `--dispositivo cuda` treina na GPU:

```bash
venv/bin/python -m pipeline.e07_DTSF.rodar --modelos D D_sem_T D_sem_E D_sem_mixup --dispositivo cuda --processos 14 2>&1 | tee -a outputs/07_DTSF/log_dtsf.txt
venv/bin/python -m pipeline.e07_DTSF.rodar --modelos D --permutacoes 25 --dispositivo cuda --processos 14 2>&1 | tee -a outputs/07_DTSF/log_dtsf_permutacoes.txt
```

- Os processos dividem a mesma GPU. O sorteio das janelas, o Mixup e a STFT continuam na CPU, com os mesmos geradores, e o lote só depois vai para a GPU; por isso vale usar quase todos os núcleos em `--processos`.
- Com `cuda`, os processos são criados com `spawn` (a GPU não sobrevive a `fork`) e a cuDNN fica em modo determinístico.
- **A GPU não reproduz bit a bit as contas da CPU.** Uma rodada em GPU dá números próximos, mas não idênticos aos de uma rodada em CPU (a época escolhida pelo early stopping pode mudar). Cada `.json` registra o dispositivo, e o runner se recusa a completar uma pasta que já tem tarefas feitas em outro dispositivo: para trocar, use outra `--saida` ou apague as tarefas antigas.
- O caminho padrão não mudou: um fold retreinado na CPU com 1 thread (SZ, D, repetição 0, fold 0) dá probabilidades idênticas às gravadas na 2ª rodada (diferença máxima 0).
- A etapa 09 (`explicar` e `lote`) continua treinando o D na CPU.

## Modelo

| Parte | Arquitetura |
| --- | --- |
| Ramo temporal | (19, 5, 500) → Conv2d(19→16, (5, 13)) → BN → GELU → MaxPool (1, 2) → Dropout 0,5 → Conv2d(16→16, (1, 25)) → BN → GELU → AdaptiveMaxPool (1, 16) → Dropout 0,5 → GRU 32 → 16 tokens |
| Ramo espectral | (19, 29, 16), z-score por (canal, frequência) com estatísticas do treino do fold → Conv2d(19→16, (29, 3)) → BN → GELU → Dropout 0,5 → Conv2d(16→16, (1, 3)) → BN → GELU → Dropout 0,5 → GRU 32 → 16 tokens |
| Fusão | 32 tokens → Linear 32 → token CLS no fim → posição aprendível → 2 blocos pre-LN (MHSA de 2 cabeças; FFN com gate GELU) → LN → último token → Linear → logit |
| Tamanho | 80.449 parâmetros |

| Ablação | O que muda |
| --- | --- |
| D_sem_T | Sem o ramo temporal: o Transformer recebe só os 16 tokens espectrais e o CLS |
| D_sem_E | Sem o ramo espectral |
| D_sem_mixup | O D sem Mixup |

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
