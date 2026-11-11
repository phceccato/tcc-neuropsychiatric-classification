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

Tipos: D_FA (D + FA), D_SDKA (D + SMMD no treino) e D_FA_SDKA (os dois). Fora a FA, a rede é o D da
etapa 07 (pipeline/e07_DTSF/redes_dtsf.py), com Mixup.
"""
from __future__ import annotations

import math

import torch
from torch import nn

from pipeline.e07_DTSF.redes_dtsf import DROPOUT, DROPOUT_ENCODER, DTSF, FILTROS

TIPOS_12 = ("D_FA", "D_SDKA", "D_FA_SDKA")
GRADE_FCANET = 7
COMPONENTES_FA = (0, 1, 2, 3)       # índices na grade 7, um por grupo
REDUCAO_FA = 4
LARGURA_MAPA = 250                  # 500 amostras → MaxPool (1, 2)


def base_dct(n: int, u: int) -> torch.Tensor:
    """Base da DCT-II ortonormal de tamanho n, componente u (normalização do FcaNet)."""
    w = torch.arange(n, dtype=torch.float32)
    base = torch.cos(math.pi * u * (w + 0.5) / n) / math.sqrt(n)
    return base * math.sqrt(2) if u else base


class AtencaoFrequencia(nn.Module):
    """FA sobre um mapa [B, C, 1, W]: um componente DCT por grupo de C/n filtros → pesos por filtro."""

    def __init__(self, canais: int = FILTROS, largura: int = LARGURA_MAPA,
                 componentes: tuple[int, ...] = COMPONENTES_FA, reducao: int = REDUCAO_FA):
        super().__init__()
        if canais % len(componentes):
            raise ValueError(f"{canais} canais não se dividem em {len(componentes)} grupos")
        passo = largura // GRADE_FCANET
        por_grupo = canais // len(componentes)
        filtro = torch.stack([base_dct(largura, u * passo) for u in componentes for _ in range(por_grupo)])
        self.register_buffer("filtro_dct", filtro)                     # [C, W]
        self.componentes = tuple(u * passo for u in componentes)
        self.fc = nn.Sequential(nn.Linear(canais, canais // reducao, bias=False), nn.ReLU(),
                                nn.Linear(canais // reducao, canais, bias=False), nn.Sigmoid())

    def pesos(self, mapa: torch.Tensor) -> torch.Tensor:
        if mapa.shape[2] != 1 or mapa.shape[3] != self.filtro_dct.shape[1]:
            raise ValueError(f"mapa {tuple(mapa.shape)} incompatível com a FA de largura {self.filtro_dct.shape[1]}")
        freq = (mapa[:, :, 0, :] * self.filtro_dct).sum(-1)             # [B, C]
        return self.fc(freq)

    def forward(self, mapa: torch.Tensor) -> torch.Tensor:
        return mapa * self.pesos(mapa)[:, :, None, None]


class DTSF12(DTSF):
    """O D da etapa 07, com a FA depois do 1º bloco temporal (D_FA, D_FA_SDKA) e o CLS exposto."""

    def __init__(self, tipo: str, dropout: float = DROPOUT, dropout_encoder: float = DROPOUT_ENCODER):
        if tipo not in TIPOS_12:
            raise ValueError(f"tipo desconhecido: {tipo!r}")
        super().__init__("D", dropout=dropout, dropout_encoder=dropout_encoder)
        self.tipo = tipo
        self.usa_fa, self.usa_sdka = "FA" in tipo, "SDKA" in tipo
        if self.usa_fa:
            self.fa = AtencaoFrequencia()

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
