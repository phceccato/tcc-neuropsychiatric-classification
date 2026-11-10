"""DTSF adaptado (WANG et al., 2026) e as ablações, como em docs/00_proposta-tcc.md.

- Ramo temporal: (19, 5, 500) → Conv2d(19→16, (5, 13)) → BN → GELU → MaxPool (1, 2) → Dropout 0,5
  → Conv2d(16→16, (1, 25)) → BN → GELU → AdaptiveMaxPool (1, 16) → Dropout 0,5 → GRU 32 → (16, 32).
- Ramo espectral: (19, 29, 16), z-score por (canal, frequência) com estatísticas do treino do fold
  (buffers) → Conv2d(19→16, (29, 3)) → BN → GELU → Dropout 0,5 → Conv2d(16→16, (1, 3)) → BN → GELU
  → Dropout 0,5 → GRU 32 → (16, 32).
- Fusão: tokens dos dois ramos concatenados no tempo → Linear 32 → token CLS no fim → posição
  aprendível → 2 blocos pre-LN (MHSA de 2 cabeças; GMLP: FFN com gate GELU) → LN → último token
  → Linear → logit.

Ablações: D_sem_T e D_sem_E (o Transformer recebe só os 16 tokens de um ramo + CLS); D_sem_mixup (o D sem
Mixup).

O Mixup fica no laço de treino (pipeline/e07_DTSF/treino.py): `sortear_mixup` sorteia λ ~ Beta(1, 1)
e a permutação do lote; `misturar` aplica os dois aos dois ramos e ao alvo.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

N_CANAIS, N_BANDAS, N_FREQUENCIAS, N_QUADROS = 19, 5, 29, 16
FILTROS, OCULTO, CABECAS, CAMADAS = 16, 32, 2, 2
DROPOUT, DROPOUT_ENCODER = 0.5, 0.1
ALFA_MIXUP = 1.0
TIPOS_DTSF = ("D", "D_sem_T", "D_sem_E", "D_sem_mixup")


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
    def __init__(self, entrada: int, kernel: tuple[int, int], pool: nn.Module | None, dropout: float = DROPOUT):
        camadas = [nn.Conv2d(entrada, FILTROS, kernel, padding=(0, kernel[1] // 2)), nn.BatchNorm2d(FILTROS), nn.GELU()]
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

    def __init__(self, d: int, dropout: float = DROPOUT_ENCODER):
        super().__init__()
        self.ln_atencao, self.ln_gmlp = nn.LayerNorm(d), nn.LayerNorm(d)
        self.atencao = nn.MultiheadAttention(d, CABECAS, dropout=dropout, batch_first=True)
        self.gmlp = _GMLP(d, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln_atencao(x)
        x = x + self.atencao(h, h, h, need_weights=False)[0]
        return x + self.gmlp(self.ln_gmlp(x))


class DTSF(nn.Module):
    def __init__(self, tipo: str, dropout: float = DROPOUT, dropout_encoder: float = DROPOUT_ENCODER):
        super().__init__()
        if tipo not in TIPOS_DTSF:
            raise ValueError(f"tipo desconhecido: {tipo!r}")
        self.tipo = tipo
        self.mixup = tipo != "D_sem_mixup"
        self.usa_t, self.usa_e = tipo != "D_sem_T", tipo != "D_sem_E"
        if self.usa_t:
            self.conv_t = nn.Sequential(_BlocoConv(N_CANAIS, (N_BANDAS, 13), nn.MaxPool2d((1, 2), (1, 2)), dropout),
                                        _BlocoConv(FILTROS, (1, 25), nn.AdaptiveMaxPool2d((1, N_QUADROS)), dropout))
            self.gru_t = nn.GRU(FILTROS, OCULTO, batch_first=True)
        if self.usa_e:
            self.conv_e = nn.Sequential(_BlocoConv(N_CANAIS, (N_FREQUENCIAS, 3), None, dropout),
                                        _BlocoConv(FILTROS, (1, 3), None, dropout))
            self.gru_e = nn.GRU(FILTROS, OCULTO, batch_first=True)
        self.register_buffer("stft_media", torch.zeros(N_CANAIS, N_FREQUENCIAS))
        self.register_buffer("stft_desvio", torch.ones(N_CANAIS, N_FREQUENCIAS))
        n_tokens = N_QUADROS * (self.usa_t + self.usa_e) + 1
        self.embedding = nn.Linear(OCULTO, OCULTO)
        self.cls = nn.Parameter(torch.randn(1, 1, OCULTO))
        self.posicao = nn.Parameter(torch.randn(1, n_tokens, OCULTO))
        self.encoder = nn.ModuleList([_Encoder(OCULTO, dropout_encoder) for _ in range(CAMADAS)])
        self.ln_final = nn.LayerNorm(OCULTO)
        self.saida = nn.Linear(OCULTO, 1)

    def definir_normalizacao(self, media: torch.Tensor, desvio: torch.Tensor) -> None:
        self.stft_media.copy_(media)
        self.stft_desvio.copy_(desvio.clamp_min(1e-6))

    @staticmethod
    def _sequencia(conv: nn.Module, gru: nn.GRU, x: torch.Tensor) -> torch.Tensor:
        mapa = conv(x)                                   # [B, 16, 1, 16]
        return gru(mapa.squeeze(2).transpose(1, 2))[0]   # [B, 16 passos, 32]

    def ramo_temporal(self, x_t: torch.Tensor) -> torch.Tensor:
        return self._sequencia(self.conv_t, self.gru_t, x_t)

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
