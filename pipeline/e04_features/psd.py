"""Features espectrais da LR de 76 bandas (pipeline/e06_LR_psd) em cada janela de 4 s.

Segue docs/00_proposta-tcc.md (referência adicional em "Baselines clássicos por modalidade") e docs/04_features.md:

- Welch na janela de 4 s: segmentos de 2 s, Hann, 50% de overlap (3 segmentos), 0,5 Hz;
- bandas delta [2, 4), teta [4, 8), alfa [8, 13), beta [13, 30] Hz;
- log10 da potência absoluta por canal × banda = 76 features;
- checagem no MDD: potência relativa (banda ÷ potência total de 2–30 Hz).

Usa o sinal em µV de outputs/03_janelas (2–30 Hz, sem z-score). Calcula todas as janelas de
janelas.csv, inclusive as rejeitadas, para manter o alinhamento linha a linha com o índice;
quem consome filtra por `rejeitada`. Grava outputs/04_features/psd.npz com as features absolutas e
relativas.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch

from pipeline.e01_auditoria.auditar_ec import CANAIS

SEGMENTO_S = 2.0
FAIXA_HZ = (2.0, 30.0)
BANDAS = ("delta", "teta", "alfa", "beta")
LIMITES_HZ = {"delta": (2.0, 4.0), "teta": (4.0, 8.0), "alfa": (8.0, 13.0), "beta": (13.0, 30.0)}
TAMANHO_JANELA = 1000  # 4 s a 250 Hz


def cortar(x: np.ndarray, inicios: np.ndarray, tamanho: int) -> np.ndarray:
    """Janelas [n, canais, tamanho] a partir do sinal contínuo [canais, amostras]."""
    return np.stack([x[:, i:i + tamanho] for i in inicios])


def welch_janelas(janelas: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """PSD (µV²/Hz) de cada janela e canal, só nos bins de 2 a 30 Hz."""
    nperseg = int(SEGMENTO_S * fs)
    freqs, espectro = welch(janelas, fs=fs, window="hann", nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    dentro = (freqs >= FAIXA_HZ[0]) & (freqs <= FAIXA_HZ[1])
    return freqs[dentro], espectro[..., dentro]


def potencia_bandas(espectro: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Potência absoluta (µV²) por banda: soma dos bins × resolução. [n, canais, 4]"""
    df = freqs[1] - freqs[0]
    colunas = []
    for banda in BANDAS:
        baixo, alto = LIMITES_HZ[banda]
        mascara = (freqs >= baixo) & ((freqs < alto) if banda != "beta" else (freqs <= alto))
        colunas.append(espectro[..., mascara].sum(axis=-1) * df)
    return np.stack(colunas, axis=-1)


def features_absolutas(potencia: np.ndarray) -> np.ndarray:
    return np.log10(potencia).reshape(len(potencia), -1)


def features_relativas(potencia: np.ndarray) -> np.ndarray:
    return (potencia / potencia.sum(axis=-1, keepdims=True)).reshape(len(potencia), -1)


def nomes_features() -> list[str]:
    return [f"{canal}_{banda}" for canal in CANAIS for banda in BANDAS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/04_features"))
    args = parser.parse_args()

    indice = pd.read_csv(args.janelas / "janelas.csv")
    espectros, freqs = [], None
    for uid, g in indice.groupby("subject_uid", sort=False):
        sinal = np.load(args.janelas / f"{uid}.npz")
        freqs, espectro = welch_janelas(cortar(sinal["x"], g["inicio_amostra"].to_numpy(), TAMANHO_JANELA),
                                        float(sinal["fs"]))
        espectros.append(espectro.astype(np.float32))
        print(f"{uid}: {len(g)} janelas")
    espectro = np.concatenate(espectros)
    potencia = potencia_bandas(espectro.astype(np.float64), freqs)

    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / "psd.npz"
    np.savez(destino, freqs=freqs,
             absolutas=features_absolutas(potencia).astype(np.float32),
             relativas=features_relativas(potencia).astype(np.float32),
             nomes=np.array(nomes_features()), subject_uid=indice["subject_uid"].to_numpy().astype(str),
             inicio_amostra=indice["inicio_amostra"].to_numpy())
    print(f"\n{len(indice)} janelas → {destino}")


if __name__ == "__main__":
    main()
