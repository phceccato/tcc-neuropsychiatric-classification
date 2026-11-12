"""Variantes exploratórias do DTSF adaptado com as duas peças do TFST-SDKA (XU et al., 2026).

- Frequency Attention (FA), a atenção de canal do FcaNet aplicada ao EEG: os 16 filtros do mapa
  que sai do 1º bloco convolucional temporal ([B, 16, 1, 250]) são divididos em 4 grupos de 4, e cada
  grupo é resumido por um componente DCT diferente em vez do GAP; os 16 resumos passam por
  FC(16 → 4) → ReLU → FC(4 → 16) → sigmoid e reescalam os filtros. Como a altura do mapa é 1, a DCT
  2D vira uma DCT 1D no tempo (v = 0). Componentes fixos a priori, sem seleção pelo resultado: os 4
  de menor frequência da grade 7 do FcaNet (u = 0, 1, 2, 3), escalados para a largura do mapa
  (u · ⌊250/7⌋ = 0, 35, 70, 105). O componente 0 é o GAP, a menos da escala.
- SDKA: a perda de treino ganha α · SMMD (pipeline/e12_FA_SDKA/treino.py) sobre a representação do
  token CLS (32 dimensões, depois do LayerNorm final, entrada da camada de saída).

A FA (`AtencaoFrequencia`) fica em pipeline/e07_DTSF/redes_dtsf.py, onde o modelo principal (D_FA, com as
dimensões do artigo) também a usa; aqui ela é aplicada ao DTSF com as dimensões reduzidas (16 filtros, GRU 32)
por padrão. As dimensões da rede podem mudar como no D (a etapa 14 usa a FA sobre o D com as dimensões do artigo, com 64
filtros: 4 grupos de 16).

Tipos: D_FA (D + FA), D_SDKA (D + SMMD no treino) e D_FA_SDKA (os dois). Fora a FA, a rede é o D da
etapa 07 (pipeline/e07_DTSF/redes_dtsf.py), com Mixup.
"""
from __future__ import annotations

import torch

from pipeline.e07_DTSF.redes_dtsf import (CABECAS, CAMADAS, COMPONENTES_FA, DROPOUT, DROPOUT_ENCODER, FILTROS,
                                          GRADE_FCANET, LARGURA_MAPA, OCULTO, REDUCAO_FA, AtencaoFrequencia, DTSF,
                                          base_dct)

TIPOS_12 = ("D_FA", "D_SDKA", "D_FA_SDKA")
__all__ = ["AtencaoFrequencia", "base_dct", "DTSF12", "TIPOS_12", "COMPONENTES_FA", "GRADE_FCANET", "LARGURA_MAPA",
           "REDUCAO_FA"]


class DTSF12(DTSF):
    """O D da etapa 07, com a FA depois do 1º bloco temporal (D_FA, D_FA_SDKA) e o CLS exposto."""

    def __init__(self, tipo: str, dropout: float = DROPOUT, dropout_encoder: float = DROPOUT_ENCODER,
                 filtros: int = FILTROS, oculto: int = OCULTO, cabecas: int = CABECAS, camadas: int = CAMADAS):
        if tipo not in TIPOS_12:
            raise ValueError(f"tipo desconhecido: {tipo!r}")
        super().__init__("D", dropout=dropout, dropout_encoder=dropout_encoder, filtros=filtros, oculto=oculto,
                         cabecas=cabecas, camadas=camadas)
        self.tipo = tipo
        self.usa_fa, self.usa_sdka = "FA" in tipo, "SDKA" in tipo
        if self.usa_fa:
            self.fa = AtencaoFrequencia(canais=filtros)

    def ramo_temporal(self, x_t: torch.Tensor) -> torch.Tensor:
        mapa = self.conv_t[0](x_t)                                      # [B, 16, 1, 250]
        if self.usa_fa:
            mapa = self.fa(mapa)
        mapa = self.conv_t[1](mapa)                                     # [B, 16, 1, 16]
        return self.gru_t(mapa.squeeze(2).transpose(1, 2))[0]

    def representacao(self, x_t: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        """CLS depois do LayerNorm final [B, 32]: a entrada da camada de saída e o espaço do SMMD."""
        z = self.tokens(x_t, x_e)
        for bloco in self.encoder:
            z = bloco(z)
        return self.ln_final(z[:, -1])

    def forward(self, x_t: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        return self.saida(self.representacao(x_t, x_e)).squeeze(-1)
