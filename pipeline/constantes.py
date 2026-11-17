"""Constantes compartilhadas entre as etapas, definidas em um lugar só.

- SEMENTE: usada nas partições (etapa 05), na LR (06), no treino do DTSF e dos clássicos (07 e 08) e no bootstrap das
  avaliações. Trocá-la muda todos os resultados e exige refazer a rodada inteira. A semente das explicações SHAP
  (SEMENTE_SHAP, etapa 09) é outra, e continua na própria etapa.
- JANELAS_POR_PESSOA_DTSF: janelas aceitas sorteadas por pessoa de treino a cada época do DTSF (etapa 07). A etapa 03 a usa
  só para conferir quantas pessoas têm janelas suficientes.
- JANELAS_POR_PESSOA_CLASSICOS: janelas de ajuste por pessoa nos baselines clássicos (etapa 08). É diferente de
  propósito (docs/08_classicos.md): o ajuste dos clássicos usa mais janelas por pessoa que o DTSF.
"""

SEMENTE = 20261108
JANELAS_POR_PESSOA_DTSF = 60
JANELAS_POR_PESSOA_CLASSICOS = 150
