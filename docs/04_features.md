# Features espectrais (PSD) — resultados

Cada uma das 22.681 janelas tem as 76 potências de banda de Welch (19 canais × 4 bandas, de 2 a 30 Hz) que alimentam a LR da etapa 06 ([06_LR_psd.md](06_LR_psd.md)). Esta etapa só existe para ela.

- Código: [pipeline/e04_features/psd.py](../pipeline/e04_features/psd.py)
- Entrada: o sinal em µV de `outputs/03_janelas/`, já filtrado em 2–30 Hz e sem z-score
- Saída: `outputs/04_features/psd.npz`, com as linhas alinhadas às de `janelas.csv`, inclusive as das janelas rejeitadas

```bash
venv/bin/python -m pipeline.e04_features.psd    # → outputs/04_features/psd.npz
```

## Parâmetros

| Item | Valor |
| --- | --- |
| Welch | segmentos de 2 s, janela de Hann, 50% de overlap (3 segmentos por janela de 4 s) |
| Resolução | 0,5 Hz; 57 bins entre 2 e 30 Hz |
| Bandas | delta [2, 4), teta [4, 8), alfa [8, 13), beta [13, 30] Hz |
| Potência de banda | soma dos bins × 0,5 Hz, em µV² |
| `absolutas` | log10 da potência de cada banda × canal: 76 features (entrada principal) |
| `relativas` | potência da banda ÷ soma das 4 bandas, por canal: 76 features (checagem do MDD) |

Um teste unitário confirma que uma senoide de 10 Hz devolve a variância esperada na banda alfa, com erro abaixo de 5%.

## Médias por classe

As tabelas mostram a média entre pessoas (± desvio) de cada banda, calculada assim: média das janelas aceitas de cada pessoa, depois média entre os 19 canais. É só uma checagem descritiva, não um teste estatístico.

**log10 da potência absoluta (µV²)**

| Task | Classe | Delta | Teta | Alfa | Beta |
| --- | --- | --- | --- | --- | --- |
| SZ | HC | 0,29 ± 0,13 | 0,49 ± 0,13 | 0,79 ± 0,43 | 0,47 ± 0,17 |
| SZ | SZ | 0,33 ± 0,14 | 0,58 ± 0,14 | 0,95 ± 0,27 | 0,59 ± 0,24 |
| MDD | HC | 0,28 ± 0,18 | **1,11 ± 0,55** | 1,04 ± 0,40 | 0,61 ± 0,20 |
| MDD | MDD | 0,18 ± 0,17 | 0,52 ± 0,32 | 0,80 ± 0,37 | 0,72 ± 0,21 |

**Potência relativa**

| Task | Classe | Delta | Teta | Alfa | Beta |
| --- | --- | --- | --- | --- | --- |
| SZ | HC | 0,16 | 0,21 | 0,43 | 0,21 |
| SZ | SZ | 0,14 | 0,20 | 0,46 | 0,21 |
| MDD | HC | 0,08 | **0,43** | 0,36 | 0,13 |
| MDD | MDD | 0,10 | 0,20 | 0,37 | **0,32** |

O que chama atenção:
- **Teta nos controles do MDD.** A potência teta é ~4× a dos pacientes e tem desvio grande. Isso combina com o lote de controles gravados em 12–14/01/2013, que têm ritmo dominante de ~5 Hz.
- **Beta relativa no MDD.** Ela é maior nos pacientes (0,32 contra 0,13).
- **SZ.** As diferenças entre as classes são pequenas perto do desvio entre pessoas.
