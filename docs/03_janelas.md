# Janelas — resultados

As 84 gravações geraram **22.681 janelas de 4 s**. Delas, **809 (3,6%) foram rejeitadas** por amplitude, e toda pessoa manteve janelas suficientes para o treino e para o teste.

- Código: [pipeline/e03_janelas/janelas.py](../pipeline/e03_janelas/janelas.py)
- Saídas em `outputs/03_janelas/`:
  - `<subject_uid>.npz`: o sinal contínuo filtrado em 2–30 Hz, em µV, junto com a média e o desvio de cada canal para o z-score;
  - `janelas.csv`: uma linha por janela, com `subject_uid`, `dataset`, `grupo`, `inicio_amostra`, `inicio_s`, `sem_overlap`, `pico_a_pico_uv`, `rejeitada` e `motivo` (`amplitude`, `trecho_plano` ou vazio);
  - `resumo.md`.

```bash
venv/bin/python -m pipeline.e03_janelas.janelas
```

## O que a etapa faz

1. Aplica um passa-banda Butterworth de 2–30 Hz, N = 4 (ordem 8), com `sosfiltfilt`, na gravação inteira, antes de cortar as janelas.
2. Cria janelas de 4 s (1000 amostras) a cada 2 s, ou seja, com 50% de overlap. As que começam em múltiplos de 4 s formam a grade sem overlap, usada na validação e no teste.
3. Rejeita a janela se o pico a pico passar de 150 µV em qualquer canal (`motivo = amplitude`) ou se ela tocar um trecho plano que sobrou nos canais não interpolados (`segmentos_planos` do manifesto da harmonização; `motivo = trecho_plano`). Na execução, as 809 rejeições foram todas por amplitude.
4. Calcula a média e o desvio de cada canal para o z-score, **só nas amostras das janelas aceitas**.

As janelas não são gravadas uma a uma. O carregador as corta a partir de `janelas.csv`. O z-score, junto com a reamostragem para 125 Hz, é aplicado em `pipeline/e07_DTSF/dados.py` e alimenta o ramo temporal do DTSF e as características temporais dos baselines clássicos. A STFT do ramo espectral e a PSD da LR de 76 bandas usam o sinal em µV.

## Rejeição por amplitude

| Task | Classe | Pessoas | Janelas | Rejeitadas | Mediana por pessoa | Pior pessoa |
| --- | --- | --- | --- | --- | --- | --- |
| SZ | HC | 14 | 6479 | 2,7% | 1,6% | 6,8% |
| SZ | SZ | 14 | 7879 | 5,1% | 1,6% | 43,0% (s12) |
| MDD | HC | 27 | 4098 | 0,7% | 0,0% | 8,7% |
| MDD | MDD | 29 | 4225 | 4,8% | 1,3% | 44,7% (MDD S1) |

A taxa de rejeição é maior nos pacientes, mas a diferença vem de duas pessoas, s12 e MDD S1. A mediana por pessoa é parecida entre as classes.

**Conferência do limiar** (pendência da proposta). A tabela mostra que fração das janelas seria rejeitada com outros limiares. O valor usado continua sendo 150 µV.

| Task | Classe | Pico a pico mediano (µV) | P95 (µV) | > 100 µV | > 150 µV | > 200 µV | > 300 µV |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SZ | HC | 56 | 115 | 8,8% | 2,7% | 1,5% | 0,9% |
| SZ | SZ | 64 | 151 | 12,4% | 5,1% | 2,1% | 0,8% |
| MDD | HC | 60 | 110 | 8,0% | 0,7% | 0,5% | 0,4% |
| MDD | MDD | 46 | 142 | 8,3% | 4,8% | 4,0% | 3,4% |

## Janelas aceitas por pessoa

| Task | Grade de treino (50%): mín / mediana / máx | Grade de teste (0%): mín / mediana / máx |
| --- | --- | --- |
| SZ | 309 / 448 / 1072 | 152 / 224 / 536 |
| MDD | 83 / 148 / 187 | 43 / 74 / 94 |

Todas as pessoas têm pelo menos as 60 janelas aceitas que o treino sorteia a cada época.

## Por que o z-score usa só as janelas aceitas

A especificação original calculava a média e o desvio na gravação inteira. Medindo nos dados, o desvio da gravação inteira ficava mais de 10% acima do desvio das janelas aceitas em 16 das 84 pessoas:

| Pessoa | Janelas rejeitadas | Desvio da gravação inteira ÷ desvio das janelas aceitas |
| --- | --- | --- |
| MDD S1 | 45% | 5,9× |
| MDD HC S6 | 1% | 2,8× |
| s07 (SZ) | 3% | 2,5× |
| MDD S15 | 11% | 2,3× |
| MDD S26 | 1% | 2,0× |

Basta um artefato de milhares de µV para que todas as janelas limpas da pessoa cheguem encolhidas ao modelo. Depois da mudança, o desvio gravado difere menos de 0,05% do desvio das janelas aceitas.
