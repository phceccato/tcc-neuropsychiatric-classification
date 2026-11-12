"""DTSF adaptado (WANG et al., 2026): o modelo principal, as variantes e as ablações, como em docs/07_DTSF.md.

Arquitetura (F filtros, oculto H, C cabeças, 2 blocos):
- Ramo temporal: (19, 5, 500) → Conv2d(19→F, (5, 13)) → BN → GELU → MaxPool (1, 2) → Dropout 0,5
  → [FA, só nos modelos com FA] → Conv2d(F→F, (1, 25)) → BN → GELU → AdaptiveMaxPool (1, 16) → Dropout 0,5
  → GRU H → (16, H).
- Ramo espectral: (19, 29, 16), z-score por (canal, frequência) com estatísticas do treino do fold
  (buffers) → Conv2d(19→F, (29, 3)) → BN → GELU → Dropout 0,5 → Conv2d(F→F, (1, 3)) → BN → GELU
  → Dropout 0,5 → GRU H → (16, H).
- Fusão: tokens dos dois ramos concatenados no tempo → Linear H → token CLS no fim → posição
  aprendível → 2 blocos pre-LN (MHSA de C cabeças; GMLP: FFN com gate GELU) → LN → último token
  → Linear → logit.

Modelos (`MODELOS`):
- D_FA, o principal: dimensões do artigo (F = 64, H = 128, C = 4) e a Frequency Attention (FA);
- D: o DTSF com as dimensões do artigo, sem FA;
- D_pequeno: o D com as dimensões reduzidas (F = 16, H = 32, C = 2), sem FA;
- D_sem_mixup, D_sem_T e D_sem_E: ablações do D (dimensões do artigo, sem FA): sem Mixup, sem o ramo
  temporal e sem o ramo espectral (o Transformer recebe só os 16 tokens de um ramo + CLS).

Frequency Attention (`AtencaoFrequencia`), a atenção de canal do FcaNet aplicada ao mapa [B, F, 1, 250] que
sai do 1º bloco temporal: os F filtros são divididos em 16 grupos, cada grupo é resumido por um componente
DCT ao longo do tempo (os 16 componentes u · ⌊250/16⌋ = 0, 15, 30, …, 225, fixados a priori, sem seleção pelo
resultado; o TFST-SDKA usa 16 componentes escolhidos pelo desempenho), e os F
resumos passam por FC(F → F/4) → ReLU → FC(F/4 → F) → sigmoid, que reescala cada filtro. A FA é criada por
último: na mesma semente, o D_FA começa com os mesmos pesos do D, e D_FA − D isola o efeito da FA.

O Mixup fica no laço de treino (pipeline/e07_DTSF/treino.py): `sortear_mixup` sorteia λ ~ Beta(1, 1)
e a permutação do lote; `misturar` aplica os dois aos dois ramos e ao alvo.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_CANAIS, N_BANDAS, N_FREQUENCIAS, N_QUADROS = 19, 5, 29, 16
FILTROS, OCULTO, CABECAS, CAMADAS = 16, 32, 2, 2              # dimensões reduzidas (D_pequeno)
FILTROS_ARTIGO, OCULTO_ARTIGO, CABECAS_ARTIGO = 64, 128, 4     # dimensões do artigo (Tabela 1)
DROPOUT, DROPOUT_ENCODER = 0.5, 0.1
ALFA_MIXUP = 1.0
GRADE_FA = 16
COMPONENTES_FA = tuple(range(16))   # índices na grade de 16, um por grupo: 0, 15, …, 225
REDUCAO_FA = 4
LARGURA_MAPA = 250                  # 500 amostras → MaxPool (1, 2)


@dataclass(frozen=True)
class Variante:
    """O que cada modelo usa: ramos, Mixup, FA e dimensões (filtros, oculto, cabeças, blocos)."""
    temporal: bool = True
    espectral: bool = True
    mixup: bool = True
    fa: bool = False
    filtros: int = FILTROS_ARTIGO
    oculto: int = OCULTO_ARTIGO
    cabecas: int = CABECAS_ARTIGO
    camadas: int = CAMADAS


MODELOS = {
    "D_FA": Variante(fa=True),
    "D_pequeno": Variante(filtros=FILTROS, oculto=OCULTO, cabecas=CABECAS),
    "D": Variante(),
    "D_sem_mixup": Variante(mixup=False),
    "D_sem_T": Variante(temporal=False),
    "D_sem_E": Variante(espectral=False),
}
TIPOS_DTSF = tuple(MODELOS)
PRINCIPAL = "D_FA"


def sortear_mixup(rng: np.random.Generator, n: int, alfa: float = ALFA_MIXUP) -> tuple[float, np.ndarray]:
    """λ ~ Beta(α, α), um por lote, e a permutação do lote."""
    return float(rng.beta(alfa, alfa)), rng.permutation(n)


def misturar(x_t: torch.Tensor, x_e: torch.Tensor, alvo: torch.Tensor, lam: float,
             perm: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Mesmo λ e mesma permutação nos dois ramos e no alvo."""
    perm = torch.as_tensor(perm)
    return (lam * x_t + (1 - lam) * x_t[perm], lam * x_e + (1 - lam) * x_e[perm],
            lam * alvo + (1 - lam) * alvo[perm])


class _BlocoConv(nn.Sequential):
    def __init__(self, entrada: int, kernel: tuple[int, int], pool: nn.Module | None, dropout: float = DROPOUT,
                 filtros: int = FILTROS):
        camadas = [nn.Conv2d(entrada, filtros, kernel, padding=(0, kernel[1] // 2)), nn.BatchNorm2d(filtros), nn.GELU()]
        super().__init__(*camadas, *([pool] if pool is not None else []), nn.Dropout(dropout))


class _GMLP(nn.Module):
    """FFN com gate: Linear(d → 2d) dividida em u e v, GELU(u)·v, Linear(d → d)."""

    def __init__(self, d: int, dropout: float = DROPOUT_ENCODER):
        super().__init__()
        self.entrada, self.saida, self.dropout = nn.Linear(d, 2 * d), nn.Linear(d, d), nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u, v = self.entrada(x).chunk(2, dim=-1)
        return self.dropout(self.saida(F.gelu(u) * v))


class _Encoder(nn.Module):
    """Bloco pre-LN: LN → MHSA → residual; LN → GMLP → residual."""

    def __init__(self, d: int, dropout: float = DROPOUT_ENCODER, cabecas: int = CABECAS):
        super().__init__()
        self.ln_atencao, self.ln_gmlp = nn.LayerNorm(d), nn.LayerNorm(d)
        self.atencao = nn.MultiheadAttention(d, cabecas, dropout=dropout, batch_first=True)
        self.gmlp = _GMLP(d, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_atencao(x)
        x = x + self.atencao(h, h, h, need_weights=False)[0]
        return x + self.gmlp(self.ln_gmlp(x))


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
        passo = largura // GRADE_FA
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


class DTSF(nn.Module):
    """O modelo `tipo` de `MODELOS`; dimensões passadas aqui (não None) substituem as do modelo."""

    def __init__(self, tipo: str, dropout: float = DROPOUT, dropout_encoder: float = DROPOUT_ENCODER,
                 filtros: int | None = None, oculto: int | None = None, cabecas: int | None = None,
                 camadas: int | None = None):
        super().__init__()
        if tipo not in MODELOS:
            raise ValueError(f"tipo desconhecido: {tipo!r}")
        v = MODELOS[tipo]
        filtros = v.filtros if filtros is None else filtros
        oculto = v.oculto if oculto is None else oculto
        cabecas = v.cabecas if cabecas is None else cabecas
        camadas = v.camadas if camadas is None else camadas
        self.tipo = tipo
        self.mixup = v.mixup
        self.usa_t, self.usa_e, self.usa_fa = v.temporal, v.espectral, v.fa
        if self.usa_t:
            self.conv_t = nn.Sequential(
                _BlocoConv(N_CANAIS, (N_BANDAS, 13), nn.MaxPool2d((1, 2), (1, 2)), dropout, filtros),
                _BlocoConv(filtros, (1, 25), nn.AdaptiveMaxPool2d((1, N_QUADROS)), dropout, filtros))
            self.gru_t = nn.GRU(filtros, oculto, batch_first=True)
        if self.usa_e:
            self.conv_e = nn.Sequential(_BlocoConv(N_CANAIS, (N_FREQUENCIAS, 3), None, dropout, filtros),
                                        _BlocoConv(filtros, (1, 3), None, dropout, filtros))
            self.gru_e = nn.GRU(filtros, oculto, batch_first=True)
        self.register_buffer("stft_media", torch.zeros(N_CANAIS, N_FREQUENCIAS))
        self.register_buffer("stft_desvio", torch.ones(N_CANAIS, N_FREQUENCIAS))
        n_tokens = N_QUADROS * (self.usa_t + self.usa_e) + 1
        self.embedding = nn.Linear(oculto, oculto)
        self.cls = nn.Parameter(torch.randn(1, 1, oculto))
        self.posicao = nn.Parameter(torch.randn(1, n_tokens, oculto))
        self.encoder = nn.ModuleList([_Encoder(oculto, dropout_encoder, cabecas) for _ in range(camadas)])
        self.ln_final = nn.LayerNorm(oculto)
        self.saida = nn.Linear(oculto, 1)
        if self.usa_fa:                                  # por último: o resto começa igual ao do D
            self.fa = AtencaoFrequencia(canais=filtros)

    def definir_normalizacao(self, media: torch.Tensor, desvio: torch.Tensor) -> None:
        self.stft_media.copy_(media)
        self.stft_desvio.copy_(desvio.clamp_min(1e-6))

    @staticmethod
    def _sequencia(conv: nn.Module, gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
        mapa = conv(x)                                   # [B, filtros, 1, 16]
        return gru(mapa.squeeze(2).transpose(1, 2))[0]   # [B, 16 passos, oculto]

    def ramo_temporal(self, x_t: torch.Tensor) -> torch.Tensor:
        mapa = self.conv_t[0](x_t)                       # [B, filtros, 1, 250]
        if self.usa_fa:
            mapa = self.fa(mapa)
        mapa = self.conv_t[1](mapa)                      # [B, filtros, 1, 16]
        return self.gru_t(mapa.squeeze(2).transpose(1, 2))[0]

    def ramo_espectral(self, x_e: torch.Tensor) -> torch.Tensor:
        x_e = (x_e - self.stft_media[:, :, None]) / self.stft_desvio[:, :, None]
        return self._sequencia(self.conv_e, self.gru_e, x_e)

    def _ramos(self, x_t: torch.Tensor, x_e: torch.Tensor) -> list[torch.Tensor]:
        return ([self.ramo_temporal(x_t)] if self.usa_t else []) + ([self.ramo_espectral(x_e)] if self.usa_e else [])

    def tokens(self, x_t: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        z = self.embedding(torch.cat(self._ramos(x_t, x_e), dim=1))
        return torch.cat([z, self.cls.expand(len(z), -1, -1)], dim=1) + self.posicao

    def forward(self, x_t: torch.Tensor, x_e: torch.Tensor) -> torch.Tensor:
        z = self.tokens(x_t, x_e)
        for bloco in self.encoder:
            z = bloco(z)
        return self.saida(self.ln_final(z[:, -1])).squeeze(-1)

    def perda_treino(self, x_t: torch.Tensor, x_e: torch.Tensor, alvo: torch.Tensor,
                     peso_positivo: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(self(x_t, x_e), alvo, pos_weight=peso_positivo)
