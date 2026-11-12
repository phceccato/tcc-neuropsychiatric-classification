# FA e SDKA no DTSF adaptado — teste exploratório

Teste das duas peças do TFST-SDKA (XU et al., 2026, *Neural Networks* 193, 107965; PDF em `docs/artigos/1-s2.0-S0893608025008469-main.pdf`) sobre o D da etapa 07. **É exploratório**: não entra em H1–H4, os p não decidem nada e o D continua sendo o modelo principal.

Os resultados ficam em [Resultados](#resultados).

- Código: [pipeline/e12_FA_SDKA/](../pipeline/e12_FA_SDKA/)
  - `redes.py`: a Frequency Attention (`AtencaoFrequencia`) e o `DTSF12`, que é o D com a FA opcional e o CLS exposto (`representacao`);
  - `treino.py`: o laço do D com o termo α · SMMD (`smmd`, `pesos_classe`, `nucleo_gaussiano`);
  - `rodar.py`: o runner, com retomada, processos em paralelo e permutações;
  - `avaliar.py`: as AUCs e as comparações variante − D.
- Testes: `tests/e12_FA_SDKA/` (22 testes).
- Saídas: `outputs/12_FA_SDKA/resultados/<dataset>/<modelo>/r<rep>_f<fold>.{csv,json}` e `resumo.md`; `outputs/12_FA_SDKA/permutacoes/`, se as permutações forem rodadas. Nada é gravado nas pastas das outras etapas.

Pré-requisitos: as etapas 03 e 05, os caches da 07 e as previsões completas do D em `outputs/07_DTSF/resultados`.

```bash
venv/bin/python -m pipeline.e12_FA_SDKA.rodar 2>&1 | tee -a outputs/12_FA_SDKA/log_treinos.txt     # 300 tarefas (3 variantes × 2 datasets × 5 folds × 10 repetições); retoma
venv/bin/python -m pipeline.e12_FA_SDKA.avaliar 2>&1 | tee -a outputs/12_FA_SDKA/log_avaliar.txt   # → outputs/12_FA_SDKA/resultados/resumo.md
venv/bin/python -m pipeline.e12_FA_SDKA.rodar --permutacoes 25                                     # opcional: contra o acaso (750 treinos)
```

Numa máquina com GPU, acrescente `--dispositivo cuda` ao `rodar` (ver [07_DTSF.md](07_DTSF.md#cpu-ou-gpu)). Use o mesmo dispositivo do D da etapa 07, com que as variantes são comparadas: o runner se recusa a completar uma pasta feita em outro dispositivo, e o `resumo.md` registra o dispositivo do D e das variantes e avisa se forem diferentes. Na CPU, o resultado não mudou com essa opção: um fold de D_FA e um de D_SDKA retreinados (SZ, repetição 0, fold 0) dão probabilidades idênticas às gravadas.

## Variantes

| Modelo | O que muda em relação ao D |
| --- | --- |
| `D_FA` | Frequency Attention depois do 1º bloco convolucional do ramo temporal |
| `D_SDKA` | Perda de treino BCE + α · SMMD sobre o CLS, com as pessoas de teste como alvo |
| `D_FA_SDKA` | As duas |

Partições, sementes, sorteio das janelas de treino, Mixup, otimizador, early stopping e previsão são os do D na mesma (repetição, fold), então a comparação com o D é pareada. O `D_SDKA` começa com os mesmos pesos do D.

## Frequency Attention

| Item | Valor |
| --- | --- |
| Posição | Saída do 1º bloco do ramo temporal (conv (5, 13) → BN → GELU → MaxPool (1, 2) → Dropout), mapa [B, 16, 1, 250] |
| Grupos | 4 grupos de 4 filtros; cada grupo é resumido por um componente DCT |
| DCT | DCT-II ortonormal, como no FcaNet. A altura do mapa é 1, então a DCT 2D vira uma DCT 1D no tempo (v = 0) |
| Componentes | Fixos a priori: os 4 de menor frequência da grade 7 do FcaNet (u = 0, 1, 2, 3), escalados para a largura do mapa: 0, 35, 70 e 105. O componente 0 é o GAP vezes √250 |
| Pesos | FC(16 → 4, sem viés) → ReLU → FC(4 → 16, sem viés) → sigmoid; cada filtro é multiplicado pelo seu peso |
| Parâmetros a mais | 128 (o D tem 80.449) |

O artigo escolhe os componentes testando os 49 da grade 7×7 um a um e depois combinando os Top-M pelo resultado. Aqui isso usaria o teste para escolher, então a escolha foi fixada antes da rodada. As "frequências" da FA são as do mapa de características ao longo do tempo da janela, não bandas do EEG em Hz.

## SDKA (SMMD)

| Item | Valor |
| --- | --- |
| Espaço alinhado | CLS depois do LayerNorm final (32 dimensões), a entrada da camada de saída |
| Fonte | As 64 janelas do lote de treino **antes** do Mixup, com o rótulo verdadeiro como peso de classe (Eq. 12) |
| Alvo | 64 janelas aceitas de **uma** pessoa de teste por lote, numa ordem sorteada das pessoas de teste, repetida enquanto houver lotes. Peso de classe = probabilidade prevista pela rede, sem gradiente |
| Perda | BCE (no lote misturado, como no D) + α · SMMD (Eqs. 13–14), α = 0,1 fixo (o valor do artigo no MODMA) |
| Núcleo | Soma de 5 gaussianas, larguras em progressão geométrica (fator 2) em torno da distância quadrática média entre os pares (como no DAN/DSAN; o artigo não detalha o núcleo) |
| Classes | Uma classe sem janelas no lote da fonte fica fora da média |
| BatchNorm | Fonte e alvo passam juntos pela rede em modo de treino, então a BatchNorm também vê as janelas do alvo |
| Validação | Não muda: early stopping na BCE das pessoas de validação |

**O SDKA é transdutivo.** O treino usa o EEG das pessoas de teste, sem os rótulos. Os rótulos de teste nunca entram no treino e a validação continua separada, mas a pergunta muda: "o modelo funciona numa pessoa cujo EEG já está disponível no treino", e não "numa pessoa nunca vista". A comparação com o D e com os clássicos, que não veem o teste, não é simétrica.

Adaptações em relação ao artigo: a saída é um logit (sigmoid), não softmax, então os pesos de classe são p e 1 − p; o artigo usa *leave-one-subject-out* com um sujeito-alvo, e aqui o fold de teste tem 5 a 6 pessoas na SZ e 11 a 12 no MDD, por isso cada lote do alvo vem de uma pessoa só.

## Validação mínima do fluxo

Um fold (repetição 0, fold 0) de cada variante nos dois datasets, antes da rodada completa:

| Dataset | Modelo | Época escolhida | Tempo (6 processos) |
| --- | --- | --- | --- |
| SZ | D_FA | 1 | 200 s |
| SZ | D_SDKA | 1 | 283 s |
| SZ | D_FA_SDKA | 2 | 233 s |
| MDD | D_FA | 13 | 369 s |
| MDD | D_SDKA | 8 | 400 s |
| MDD | D_FA_SDKA | 22 | 558 s |

O D, no mesmo fold, levou 247 s (SZ) e 569 s (MDD) com 10 processos: o custo por treino das variantes fica perto do D.

## Resultados

Rodada completa: 300 treinos em 5 h 49 min (10 processos), sem erros. As permutações contra o acaso não foram rodadas. O D é o da etapa 07 (`outputs/07_DTSF/resultados`).

AUC por pessoa [IC 95%] e AUC por repetição (média ± desvio):

| Modelo | SZ | SZ, por repetição | MDD | MDD, por repetição |
| --- | --- | --- | --- | --- |
| D | 0,827 [0,643; 0,964] | 0,781 ± 0,058 | 0,958 [0,897; 0,995] | 0,937 ± 0,023 |
| D_FA | 0,847 [0,668; 0,974] | 0,783 ± 0,047 | 0,957 [0,898; 0,995] | 0,938 ± 0,021 |
| D_SDKA | 0,704 [0,485; 0,883] | 0,605 ± 0,057 | 0,883 [0,785; 0,957] | 0,748 ± 0,086 |
| D_FA_SDKA | 0,699 [0,485; 0,878] | 0,609 ± 0,101 | 0,900 [0,816; 0,967] | 0,746 ± 0,093 |

Variante − D (permutação pareada sobre os ranks, unilateral variante > D; descritivo):

| Dataset | Comparação | ΔAUC [IC 95%] | p | ΔAUC por repetição (positivos) | t corrigido p |
| --- | --- | --- | --- | --- | --- |
| SZ | D_FA − D | +0,020 [−0,015; +0,082] | 0,150 | −0,020 a +0,031 (5/10) | 0,536 |
| SZ | D_SDKA − D | −0,122 [−0,265; +0,015] | 0,983 | −0,270 a −0,036 (0/10) | 0,895 |
| SZ | D_FA_SDKA − D | −0,128 [−0,286; +0,000] | 0,971 | −0,337 a −0,087 (0/10) | 0,814 |
| MDD | D_FA − D | −0,001 [−0,013; +0,009] | 0,711 | −0,010 a +0,010 (5/10) | 0,469 |
| MDD | D_SDKA − D | −0,075 [−0,147; −0,017] | 1,000 | −0,364 a −0,091 (0/10) | 0,968 |
| MDD | D_FA_SDKA − D | −0,057 [−0,125; −0,006] | 0,995 | −0,315 a −0,049 (0/10) | 0,958 |

Leitura:

- **FA:** sem efeito. O ΔAUC fica dentro do ruído nos dois datasets (metade das repetições para cada lado), e as secundárias no limiar 0,5 são idênticas às do D.
- **SDKA:** piora nos dois datasets e em todas as repetições. A queda não é só de calibração entre folds: a AUC média dentro de cada fold também cai (SZ 0,779 → 0,591; MDD 0,945 → 0,799), com a probabilidade média por fold tão estável quanto no D. O SMMD não diminui ao longo do treino (média 0,17 na 1ª época e 0,23 na escolhida, na SZ).
- **Explicação provável:** cada lote do alvo vem de uma pessoa só, e as janelas de uma pessoa recebem probabilidades parecidas. Então as duas médias ponderadas do alvo (controle e paciente) ficam quase iguais, ambas perto da média daquela pessoa, e o SMMD puxa as médias das duas classes da fonte para o mesmo ponto, o que apaga a separação. É uma consequência da escolha de "uma pessoa por lote", não necessariamente do SDKA em si; o *leave-one-subject-out* do artigo tem o mesmo formato, mas lá o alvo é avaliado em acurácia, com outros dados e outra rede.

Diagnóstico (AUC média dentro de cada fold; desvio da probabilidade média entre folds):

| Dataset | D | D_FA | D_SDKA | D_FA_SDKA |
| --- | --- | --- | --- | --- |
| SZ | 0,779; 0,095 | 0,776; 0,099 | 0,591; 0,078 | 0,650; 0,067 |
| MDD | 0,945; 0,088 | 0,946; 0,079 | 0,799; 0,084 | 0,796; 0,099 |

### Rodada em GPU

Rodada no PC do laboratório (RTX 3090 Ti), com `--dispositivo cuda`, e comparada com o D da etapa 07 treinado na mesma GPU, com as mesmas partições e sementes (3ª rodada). Números transcritos do resumo daquela máquina:

| Dataset | D | D_FA (Δ; p) | D_SDKA (Δ) | D_FA_SDKA (Δ) |
| --- | --- | --- | --- | --- |
| SZ | 0,827 | 0,842 (+0,015; p = 0,056) | 0,658 (−0,168) | 0,679 (−0,148) |
| MDD | 0,955 | 0,951 (−0,004; p = 0,759) | 0,900 (−0,055) | 0,898 (−0,057) |

O padrão da rodada em CPU se repete: a FA fica um pouco acima do D na SZ (+0,015, depois de +0,020 na CPU), sem significância e sem efeito no MDD; o SDKA piora nos dois transtornos.
