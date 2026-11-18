# Baselines clássicos — resultados

AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; IC de 95% por bootstrap estratificado. Secundárias no limiar 0,5, só descritivas. Hiperparâmetros: escolhas da seleção interna nos 50 folds.

## SZ

| Modelo | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 | Hiperparâmetros escolhidos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.852 [0.684, 0.969] | 0.816 ± 0.039 | 0.714 | 0.714 | 0.714 | 0.714 | — |
| LR_Temporal | 0.602 [0.367, 0.806] | 0.617 ± 0.070 | 0.536 | 0.500 | 0.571 | 0.535 | {"C": 1.0} ×22; {"C": 0.1} ×19; {"C": 0.01} ×9 |
| RF_Temporal | 0.587 [0.357, 0.796] | 0.596 ± 0.056 | 0.571 | 0.643 | 0.500 | 0.569 | {"min_samples_leaf": 5} ×19; {"min_samples_leaf": 1} ×17; {"min_samples_leaf": 20} ×14 |
| KNN_Temporal | 0.561 [0.337, 0.791] | 0.538 ± 0.062 | 0.464 | 0.357 | 0.571 | 0.458 | {"n_neighbors": 11} ×39; {"n_neighbors": 31} ×10; {"n_neighbors": 101} ×1 |
| LR_Spectral | 0.852 [0.699, 0.964] | 0.836 ± 0.038 | 0.714 | 0.786 | 0.643 | 0.713 | {"C": 1.0} ×23; {"C": 0.1} ×14; {"C": 0.01} ×13 |
| RF_Spectral | 0.643 [0.429, 0.832] | 0.659 ± 0.066 | 0.571 | 0.643 | 0.500 | 0.569 | {"min_samples_leaf": 1} ×21; {"min_samples_leaf": 5} ×18; {"min_samples_leaf": 20} ×11 |
| KNN_Spectral | 0.709 [0.490, 0.883] | 0.708 ± 0.052 | 0.643 | 0.571 | 0.714 | 0.641 | {"n_neighbors": 11} ×38; {"n_neighbors": 101} ×7; {"n_neighbors": 31} ×5 |
| LR_Combined | 0.811 [0.633, 0.954] | 0.806 ± 0.037 | 0.679 | 0.643 | 0.714 | 0.678 | {"C": 0.1} ×18; {"C": 1.0} ×18; {"C": 0.01} ×14 |
| RF_Combined | 0.694 [0.480, 0.872] | 0.713 ± 0.061 | 0.714 | 0.857 | 0.571 | 0.708 | {"min_samples_leaf": 1} ×23; {"min_samples_leaf": 5} ×15; {"min_samples_leaf": 20} ×12 |
| KNN_Combined | 0.709 [0.490, 0.888] | 0.711 ± 0.058 | 0.607 | 0.500 | 0.714 | 0.603 | {"n_neighbors": 11} ×27; {"n_neighbors": 101} ×12; {"n_neighbors": 31} ×11 |

Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.

| Modelo | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | Especificidade | F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.816 ± 0.039 | 0.746 ± 0.049 | 0.746 ± 0.049 | 0.748 ± 0.060 | 0.750 ± 0.084 | 0.743 ± 0.084 | 0.746 ± 0.052 | 0.745 ± 0.049 |
| LR_Temporal | 0.617 ± 0.070 | 0.561 ± 0.041 | 0.561 ± 0.041 | 0.559 ± 0.039 | 0.571 ± 0.082 | 0.550 ± 0.059 | 0.563 ± 0.055 | 0.559 ± 0.041 |
| RF_Temporal | 0.596 ± 0.056 | 0.596 ± 0.048 | 0.596 ± 0.048 | 0.587 ± 0.044 | 0.650 ± 0.092 | 0.543 ± 0.077 | 0.615 ± 0.057 | 0.593 ± 0.048 |
| KNN_Temporal | 0.538 ± 0.062 | 0.486 ± 0.042 | 0.486 ± 0.042 | 0.477 ± 0.053 | 0.350 ± 0.092 | 0.621 ± 0.059 | 0.401 ± 0.073 | 0.474 ± 0.045 |
| LR_Spectral | 0.836 ± 0.038 | 0.732 ± 0.042 | 0.732 ± 0.042 | 0.702 ± 0.033 | 0.807 ± 0.068 | 0.657 ± 0.045 | 0.750 ± 0.043 | 0.730 ± 0.042 |
| RF_Spectral | 0.659 ± 0.066 | 0.596 ± 0.058 | 0.596 ± 0.058 | 0.583 ± 0.049 | 0.650 ± 0.128 | 0.543 ± 0.050 | 0.612 ± 0.084 | 0.593 ± 0.057 |
| KNN_Spectral | 0.708 ± 0.052 | 0.621 ± 0.059 | 0.621 ± 0.059 | 0.655 ± 0.071 | 0.514 ± 0.100 | 0.729 ± 0.081 | 0.572 ± 0.086 | 0.615 ± 0.064 |
| LR_Combined | 0.806 ± 0.037 | 0.714 ± 0.041 | 0.714 ± 0.041 | 0.711 ± 0.035 | 0.721 ± 0.071 | 0.707 ± 0.041 | 0.715 ± 0.047 | 0.714 ± 0.041 |
| RF_Combined | 0.713 ± 0.061 | 0.675 ± 0.039 | 0.675 ± 0.039 | 0.655 ± 0.030 | 0.736 ± 0.076 | 0.614 ± 0.037 | 0.692 ± 0.048 | 0.673 ± 0.039 |
| KNN_Combined | 0.711 ± 0.058 | 0.629 ± 0.042 | 0.629 ± 0.042 | 0.650 ± 0.045 | 0.557 ± 0.081 | 0.700 ± 0.056 | 0.598 ± 0.058 | 0.625 ± 0.043 |

## MDD

| Modelo | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | Especificidade | Macro F1 | Hiperparâmetros escolhidos |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.964 [0.911, 0.996] | 0.937 ± 0.018 | 0.893 | 0.897 | 0.889 | 0.893 | — |
| LR_Temporal | 0.884 [0.785, 0.962] | 0.885 ± 0.017 | 0.820 | 0.862 | 0.778 | 0.821 | {"C": 0.01} ×42; {"C": 0.1} ×6; {"C": 1.0} ×2 |
| RF_Temporal | 0.885 [0.785, 0.968] | 0.886 ± 0.008 | 0.821 | 0.828 | 0.815 | 0.821 | {"min_samples_leaf": 20} ×31; {"min_samples_leaf": 5} ×10; {"min_samples_leaf": 1} ×9 |
| KNN_Temporal | 0.894 [0.803, 0.966] | 0.890 ± 0.009 | 0.803 | 0.828 | 0.778 | 0.803 | {"n_neighbors": 101} ×32; {"n_neighbors": 11} ×12; {"n_neighbors": 31} ×6 |
| LR_Spectral | 0.934 [0.863, 0.985] | 0.927 ± 0.013 | 0.838 | 0.862 | 0.815 | 0.839 | {"C": 0.01} ×20; {"C": 0.1} ×20; {"C": 1.0} ×10 |
| RF_Spectral | 0.931 [0.853, 0.986] | 0.926 ± 0.015 | 0.857 | 0.862 | 0.852 | 0.857 | {"min_samples_leaf": 5} ×23; {"min_samples_leaf": 1} ×15; {"min_samples_leaf": 20} ×12 |
| KNN_Spectral | 0.917 [0.839, 0.976] | 0.912 ± 0.016 | 0.858 | 0.828 | 0.889 | 0.857 | {"n_neighbors": 11} ×21; {"n_neighbors": 101} ×20; {"n_neighbors": 31} ×9 |
| LR_Combined | 0.939 [0.872, 0.986] | 0.932 ± 0.017 | 0.838 | 0.862 | 0.815 | 0.839 | {"C": 0.01} ×17; {"C": 1.0} ×17; {"C": 0.1} ×16 |
| RF_Combined | 0.903 [0.814, 0.974] | 0.902 ± 0.012 | 0.821 | 0.828 | 0.815 | 0.821 | {"min_samples_leaf": 20} ×22; {"min_samples_leaf": 5} ×17; {"min_samples_leaf": 1} ×11 |
| KNN_Combined | 0.912 [0.830, 0.976] | 0.902 ± 0.016 | 0.877 | 0.828 | 0.926 | 0.875 | {"n_neighbors": 101} ×24; {"n_neighbors": 11} ×15; {"n_neighbors": 31} ×11 |

Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.

| Modelo | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | Especificidade | F1 | Macro F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DTSF_FA | 0.937 ± 0.018 | 0.870 ± 0.021 | 0.870 ± 0.021 | 0.877 ± 0.031 | 0.872 ± 0.028 | 0.867 ± 0.040 | 0.874 ± 0.019 | 0.869 ± 0.021 |
| LR_Temporal | 0.885 ± 0.017 | 0.807 ± 0.024 | 0.806 ± 0.024 | 0.796 ± 0.030 | 0.845 ± 0.018 | 0.767 ± 0.039 | 0.820 ± 0.021 | 0.806 ± 0.024 |
| RF_Temporal | 0.886 ± 0.008 | 0.821 ± 0.015 | 0.821 ± 0.015 | 0.826 ± 0.024 | 0.831 ± 0.020 | 0.811 ± 0.032 | 0.828 ± 0.013 | 0.821 ± 0.015 |
| KNN_Temporal | 0.890 ± 0.009 | 0.816 ± 0.017 | 0.815 ± 0.018 | 0.812 ± 0.027 | 0.841 ± 0.018 | 0.789 ± 0.039 | 0.826 ± 0.014 | 0.815 ± 0.017 |
| LR_Spectral | 0.927 ± 0.013 | 0.843 ± 0.011 | 0.842 ± 0.011 | 0.833 ± 0.014 | 0.872 ± 0.023 | 0.811 ± 0.021 | 0.852 ± 0.012 | 0.842 ± 0.011 |
| RF_Spectral | 0.926 ± 0.015 | 0.861 ± 0.036 | 0.859 ± 0.036 | 0.845 ± 0.035 | 0.897 ± 0.036 | 0.822 ± 0.042 | 0.870 ± 0.033 | 0.860 ± 0.036 |
| KNN_Spectral | 0.912 ± 0.016 | 0.846 ± 0.036 | 0.848 ± 0.036 | 0.884 ± 0.040 | 0.810 ± 0.041 | 0.885 ± 0.041 | 0.845 ± 0.037 | 0.846 ± 0.036 |
| LR_Combined | 0.932 ± 0.017 | 0.850 ± 0.012 | 0.849 ± 0.013 | 0.839 ± 0.015 | 0.879 ± 0.018 | 0.819 ± 0.021 | 0.859 ± 0.012 | 0.849 ± 0.013 |
| RF_Combined | 0.902 ± 0.012 | 0.827 ± 0.028 | 0.826 ± 0.028 | 0.819 ± 0.032 | 0.855 ± 0.022 | 0.796 ± 0.040 | 0.837 ± 0.025 | 0.826 ± 0.028 |
| KNN_Combined | 0.902 ± 0.016 | 0.829 ± 0.046 | 0.830 ± 0.046 | 0.864 ± 0.041 | 0.793 ± 0.063 | 0.867 ± 0.040 | 0.827 ± 0.050 | 0.828 ± 0.046 |

## DTSF_FA contra os baselines clássicos

ΔAUC = DTSF_FA − baseline, nas mesmas pessoas, para os 9 baselines. O IC é o bootstrap estratificado e pareado, condicional às previsões obtidas. Descritivo: sem p-valor; o teste estatístico é o DeLong com Holm (etapa 11).

| Dataset | Comparação | ΔAUC [IC 95%] | ΔAUC por repetição: mín–máx (positivos) |
| --- | --- | --- | --- |
| SZ | DTSF_FA − LR_Temporal | +0.250 [+0.051, +0.480] | +0.117 a +0.291 (10/10) |
| SZ | DTSF_FA − RF_Temporal | +0.265 [+0.041, +0.500] | +0.138 a +0.301 (10/10) |
| SZ | DTSF_FA − KNN_Temporal | +0.291 [+0.071, +0.520] | +0.245 a +0.367 (10/10) |
| SZ | DTSF_FA − LR_Spectral | +0.000 [-0.061, +0.061] | -0.071 a +0.026 (4/10) |
| SZ | DTSF_FA − RF_Spectral | +0.209 [+0.082, +0.378] | +0.082 a +0.240 (10/10) |
| SZ | DTSF_FA − KNN_Spectral | +0.143 [-0.015, +0.342] | +0.020 a +0.168 (10/10) |
| SZ | DTSF_FA − LR_Combined | +0.041 [-0.021, +0.117] | -0.036 a +0.046 (6/10) |
| SZ | DTSF_FA − RF_Combined | +0.158 [+0.005, +0.347] | +0.056 a +0.189 (10/10) |
| SZ | DTSF_FA − KNN_Combined | +0.143 [-0.015, +0.337] | +0.036 a +0.179 (10/10) |
| MDD | DTSF_FA − LR_Temporal | +0.080 [+0.027, +0.149] | +0.020 a +0.097 (10/10) |
| MDD | DTSF_FA − RF_Temporal | +0.079 [+0.027, +0.143] | +0.037 a +0.087 (10/10) |
| MDD | DTSF_FA − KNN_Temporal | +0.070 [+0.017, +0.137] | +0.015 a +0.089 (10/10) |
| MDD | DTSF_FA − LR_Spectral | +0.031 [+0.006, +0.068] | -0.005 a +0.042 (8/10) |
| MDD | DTSF_FA − RF_Spectral | +0.033 [+0.003, +0.077] | -0.003 a +0.054 (8/10) |
| MDD | DTSF_FA − KNN_Spectral | +0.047 [+0.003, +0.105] | +0.011 a +0.056 (10/10) |
| MDD | DTSF_FA − LR_Combined | +0.026 [+0.005, +0.055] | -0.015 a +0.038 (7/10) |
| MDD | DTSF_FA − RF_Combined | +0.061 [+0.015, +0.121] | +0.022 a +0.080 (10/10) |
| MDD | DTSF_FA − KNN_Combined | +0.052 [+0.006, +0.111] | +0.018 a +0.077 (10/10) |
