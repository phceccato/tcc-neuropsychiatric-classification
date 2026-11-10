# Harmonização — resultados

As 84 gravações incluídas pela auditoria foram harmonizadas com o mesmo pipeline nos dois datasets: 19 canais, 250 Hz, referência média e filtro de 1–40 Hz. A faixa de análise (2–30 Hz) só é aplicada na etapa de janelas.

- Código: [pipeline/e02_harmonizacao/harmonizar.py](../pipeline/e02_harmonizacao/harmonizar.py)
- Entrada: `outputs/01_auditoria/auditoria_ec.csv`, todas as linhas com decisão diferente de EXCLUIR
- Saídas em `outputs/02_harmonizacao/`:
  - `<subject_uid>.npz`, com o sinal em µV, `float32`, [19, amostras];
  - `manifesto.csv`, com uma linha por pessoa: origem, decisão da auditoria, data de gravação, frequência original, duração, canais interpolados e `segmentos_planos` (trechos planos que sobraram nos canais não interpolados, usados pela etapa de janelas para rejeitar janelas);
  - `resumo.md`.

```bash
venv/bin/python -m pipeline.e02_harmonizacao.harmonizar
```

## Pipeline

Cada gravação passa pelas mesmas etapas, nesta ordem:

1. conversão para µV e corte dos zeros do fim (C5 da auditoria);
2. seleção dos 19 canais do 10–20;
3. interpolação do canal ruim por spline esférica (montagem `colin27_1020`), **antes** da referência média, para que o canal plano não entre na média;
4. referência média comum (CAR);
5. reamostragem para 250 Hz com `resample_poly` (o MDD vem a 256 Hz);
6. notch de 50 Hz (Q = 30, `filtfilt`);
7. passa-banda Butterworth de 1–40 Hz, N = 4 (passa-banda de ordem 8), fase zero (`sosfiltfilt`, duas passadas).

## Resultados

| Task | Classe | Pessoas | Duração mínima (s) | Duração mediana (s) | Duração máxima (s) |
| --- | --- | --- | --- | --- | --- |
| SZ | HC | 14 | 862,8 | 910,8 | 1114,5 |
| SZ | SZ | 14 | 736,6 | 1110,1 | 2165,3 |
| MDD | HC | 27 | 292,0 | 301,0 | 376,0 |
| MDD | MDD | 29 | 180,0 | 300,0 | 303,0 |

Checagens feitas nos arquivos gerados:
- Não há valores não finitos, e a soma dos 19 canais é zero em todas as amostras, como a referência média exige.
- O C3 de s14 foi interpolado na gravação inteira. Nos trechos normais, ele correlaciona 0,65 com Cz.
- No trecho de 1100–1260 s de s14, o T3 tem desvio de 12,5 µV, contra ~5,7 µV nos vizinhos, provavelmente por artefato. Ali o T3 pesa mais na interpolação do C3.

## Data de gravação

A data de gravação vem do cabeçalho EDF e fica no `manifesto.csv`.

| Dataset | Classe | Período de gravação | Pessoas |
| --- | --- | --- | --- |
| MDD | HC | jul–ago/2011 | 9 |
| MDD | HC | 12–14/01/2013 | 14 |
| MDD | HC | abr–jul/2013 | 4 |
| MDD | MDD | abr–jul/2013 | 29 |
| SZ | HC | jun–jul/2003 | 14 |
| SZ | SZ | 03/2003 a 07/2008 (11 gravações depois de 2003) | 14 |

O período de gravação se confunde com o diagnóstico nos dois datasets, considerando que as datas dos cabeçalhos EDF estão corretas:

- **MDD:** todos os pacientes foram gravados em abr–jul/2013, e só 4 controles no mesmo período.
- **SZ:** os 14 controles foram gravados em jun–jul/2003. Dos 14 pacientes, 11 foram gravados entre 2004 e 2008.

Nos dois datasets, isso fica como limitação a discutir. A análise de sensibilidade do MDD sem o lote de jan/2013 foi removida.
