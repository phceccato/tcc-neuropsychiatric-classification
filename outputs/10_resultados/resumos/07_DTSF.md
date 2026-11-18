# DTSF — resultados da etapa 07

Modelo principal: DTSF_FA. AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; IC de 95% por bootstrap estratificado. Secundárias no limiar 0,5, só descritivas. Época: mediana (mín–máx) da época escolhida pelo early stopping.

## SZ

| Modelo | Parâmetros | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 | Época |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 704.769 | 0.852 [0.684, 0.969] | 0.816 ± 0.039 | 0.714 | 0.714 | 0.714 | 0.714 | 12 (1–49) |
| DTSF_Lite | 80.449 | 0.827 [0.643, 0.964] | 0.776 ± 0.056 | 0.750 | 0.786 | 0.714 | 0.750 | 6 (1–42) |
| DTSF | 702.721 | 0.837 [0.658, 0.969] | 0.820 ± 0.044 | 0.714 | 0.714 | 0.714 | 0.714 | 6 (1–33) |
| DTSF_Lite_NoMixup | 80.449 | 0.791 [0.592, 0.949] | 0.731 ± 0.051 | 0.750 | 0.786 | 0.714 | 0.750 | 2 (1–31) |
| DTSF_Lite_SpectralOnly | 48.881 | 0.847 [0.684, 0.969] | 0.787 ± 0.066 | 0.679 | 0.714 | 0.643 | 0.678 | 4 (1–46) |
| DTSF_Lite_TemporalOnly | 47.825 | 0.776 [0.566, 0.939] | 0.691 ± 0.085 | 0.750 | 0.786 | 0.714 | 0.750 | 7 (1–38) |

Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.

| Modelo | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | Especificidade | F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.816 ± 0.039 | 0.746 ± 0.049 | 0.746 ± 0.049 | 0.748 ± 0.060 | 0.750 ± 0.084 | 0.743 ± 0.084 | 0.746 ± 0.052 | 0.745 ± 0.049 |
| DTSF_Lite | 0.776 ± 0.056 | 0.718 ± 0.057 | 0.718 ± 0.057 | 0.710 ± 0.046 | 0.736 ± 0.117 | 0.700 ± 0.066 | 0.719 ± 0.070 | 0.716 ± 0.057 |
| DTSF | 0.820 ± 0.044 | 0.743 ± 0.073 | 0.743 ± 0.073 | 0.742 ± 0.067 | 0.743 ± 0.108 | 0.743 ± 0.069 | 0.741 ± 0.080 | 0.742 ± 0.073 |
| DTSF_Lite_NoMixup | 0.731 ± 0.051 | 0.664 ± 0.068 | 0.664 ± 0.068 | 0.650 ± 0.058 | 0.714 ± 0.135 | 0.614 ± 0.096 | 0.676 ± 0.082 | 0.661 ± 0.068 |
| DTSF_Lite_SpectralOnly | 0.787 ± 0.066 | 0.700 ± 0.059 | 0.700 ± 0.059 | 0.688 ± 0.056 | 0.736 ± 0.083 | 0.664 ± 0.083 | 0.709 ± 0.061 | 0.699 ± 0.060 |
| DTSF_Lite_TemporalOnly | 0.691 ± 0.085 | 0.589 ± 0.111 | 0.589 ± 0.111 | 0.598 ± 0.127 | 0.521 ± 0.158 | 0.657 ± 0.120 | 0.552 ± 0.142 | 0.584 ± 0.115 |

## MDD

| Modelo | Parâmetros | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 | Época |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 704.769 | 0.964 [0.911, 0.996] | 0.937 ± 0.018 | 0.893 | 0.897 | 0.889 | 0.893 | 7 (1–32) |
| DTSF_Lite | 80.449 | 0.958 [0.902, 0.996] | 0.933 ± 0.030 | 0.875 | 0.862 | 0.889 | 0.875 | 10 (1–33) |
| DTSF | 702.721 | 0.962 [0.909, 0.996] | 0.933 ± 0.022 | 0.875 | 0.862 | 0.889 | 0.875 | 9 (1–33) |
| DTSF_Lite_NoMixup | 80.449 | 0.950 [0.885, 0.995] | 0.920 ± 0.029 | 0.893 | 0.897 | 0.889 | 0.893 | 1 (1–28) |
| DTSF_Lite_SpectralOnly | 48.881 | 0.958 [0.908, 0.992] | 0.936 ± 0.023 | 0.857 | 0.862 | 0.852 | 0.857 | 8 (1–33) |
| DTSF_Lite_TemporalOnly | 47.825 | 0.941 [0.858, 0.994] | 0.928 ± 0.017 | 0.857 | 0.862 | 0.852 | 0.857 | 16 (1–66) |

Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.

| Modelo | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | Especificidade | F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.937 ± 0.018 | 0.870 ± 0.021 | 0.870 ± 0.021 | 0.877 ± 0.031 | 0.872 ± 0.028 | 0.867 ± 0.040 | 0.874 ± 0.019 | 0.869 ± 0.021 |
| DTSF_Lite | 0.933 ± 0.030 | 0.871 ± 0.044 | 0.872 ± 0.044 | 0.883 ± 0.052 | 0.869 ± 0.051 | 0.874 ± 0.066 | 0.875 ± 0.040 | 0.871 ± 0.044 |
| DTSF | 0.933 ± 0.022 | 0.868 ± 0.032 | 0.868 ± 0.032 | 0.879 ± 0.034 | 0.866 ± 0.050 | 0.870 ± 0.044 | 0.871 ± 0.032 | 0.868 ± 0.032 |
| DTSF_Lite_NoMixup | 0.920 ± 0.029 | 0.870 ± 0.042 | 0.870 ± 0.043 | 0.883 ± 0.052 | 0.866 ± 0.041 | 0.874 ± 0.061 | 0.873 ± 0.039 | 0.869 ± 0.042 |
| DTSF_Lite_SpectralOnly | 0.936 ± 0.023 | 0.850 ± 0.033 | 0.850 ± 0.033 | 0.864 ± 0.043 | 0.845 ± 0.037 | 0.856 ± 0.051 | 0.854 ± 0.032 | 0.850 ± 0.033 |
| DTSF_Lite_TemporalOnly | 0.928 ± 0.017 | 0.870 ± 0.033 | 0.869 ± 0.033 | 0.861 ± 0.036 | 0.893 ± 0.030 | 0.844 ± 0.046 | 0.877 ± 0.030 | 0.869 ± 0.033 |

## Ablações do DTSF_Lite

ΔAUC = DTSF_Lite − ablação, nas mesmas pessoas. O IC é o bootstrap estratificado e pareado, condicional às previsões obtidas. Descritivo: sem p-valor.

| Dataset | Comparação | ΔAUC [IC 95%] | ΔAUC por repetição: mín–máx (positivos) |
| --- | --- | --- | --- |
| SZ | DTSF_Lite − DTSF_Lite_SpectralOnly | -0.020 [-0.082, +0.041] | -0.138 a +0.153 (5/10) |
| SZ | DTSF_Lite − DTSF_Lite_TemporalOnly | +0.051 [-0.036, +0.168] | -0.020 a +0.204 (8/10) |
| SZ | DTSF_Lite − DTSF_Lite_NoMixup | +0.036 [-0.015, +0.112] | -0.015 a +0.128 (9/10) |
| MDD | DTSF_Lite − DTSF_Lite_SpectralOnly | +0.000 [-0.027, +0.027] | -0.038 a +0.023 (5/10) |
| MDD | DTSF_Lite − DTSF_Lite_TemporalOnly | +0.017 [-0.014, +0.058] | -0.032 a +0.036 (5/10) |
| MDD | DTSF_Lite − DTSF_Lite_NoMixup | +0.008 [-0.006, +0.028] | -0.015 a +0.047 (6/10) |

## Comparações entre os modelos da família DTSF

DTSF_FA − DTSF isola a FA; DTSF − DTSF_Lite isola o tamanho (dimensões do artigo contra as reduzidas). Descritivo: sem p-valor; o teste estatístico do DTSF_FA é o DeLong com Holm (etapa 11).

| Dataset | Comparação | ΔAUC [IC 95%] | ΔAUC por repetição: mín–máx (positivos) |
| --- | --- | --- | --- |
| SZ | DTSF_FA − DTSF | +0.015 [+0.000, +0.051] | -0.036 a +0.020 (3/10) |
| SZ | DTSF − DTSF_Lite | +0.010 [-0.020, +0.051] | -0.015 a +0.097 (8/10) |
| SZ | DTSF_FA − DTSF_Lite | +0.026 [-0.015, +0.082] | -0.026 a +0.107 (8/10) |
| MDD | DTSF_FA − DTSF | +0.003 [-0.013, +0.019] | -0.011 a +0.019 (7/10) |
| MDD | DTSF − DTSF_Lite | +0.004 [-0.004, +0.015] | -0.041 a +0.037 (6/10) |
| MDD | DTSF_FA − DTSF_Lite | +0.006 [-0.008, +0.024] | -0.037 a +0.036 (6/10) |
