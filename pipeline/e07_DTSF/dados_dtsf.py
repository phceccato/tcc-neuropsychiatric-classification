"""Entradas do DTSF adaptado: cache multibanda a 125 Hz e STFT do sinal em µV.

- Multibanda: o sinal do cache atual (2–30 Hz, z-score por canal da própria pessoa, 125 Hz) e as
  bandas δ 2–4, θ 4–8, α 8–13 e β 13–30 Hz (Butterworth ordem 4, sosfiltfilt) aplicadas na gravação
  inteira, antes de cortar as janelas: outputs/07_DTSF/cache_dtsf/<subject_uid>.npy, [5, 19, N].
- STFT: a janela em µV (z · desvio + média, com as estatísticas de outputs/03_janelas/<uid>.npz),
  Hann de 125 amostras (1 s), passo de 25, sem padding → 16 quadros; log10(|X|² + 1e-12) nos bins
  de 2 a 30 Hz → [19, 29, 16]. A escala da potência não importa: o modelo aplica o z-score por
  (canal, frequência) com as estatísticas do treino do fold (`estatisticas_stft`).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.signal import butter, sosfiltfilt

from pipeline.e07_DTSF.dados import AMOSTRAS_125

FS = 125
BANDAS = ((2, 4), (4, 8), (8, 13), (13, 30))
ORDEM_BUTTER = 4
N_FFT, PASSO_STFT = 125, 25
FREQUENCIAS = np.arange(2, 31)  # bins de k · fs / N_FFT = k Hz
EPS = 1e-12


def multibanda(base: np.ndarray) -> np.ndarray:
    """[5, canais, amostras]: o sinal base e as 4 bandas, filtradas na gravação inteira."""
    bandas = [sosfiltfilt(butter(ORDEM_BUTTER, banda, btype="bandpass", fs=FS, output="sos"),
                          np.asarray(base, dtype=np.float64), axis=-1) for banda in BANDAS]
    return np.stack([np.asarray(base, dtype=np.float32), *(b.astype(np.float32) for b in bandas)])


def log_stft(x_uv: torch.Tensor) -> torch.Tensor:
    """[janelas, canais, 500] em µV → log10 da potência [janelas, canais, 29, 16]."""
    n, canais, amostras = x_uv.shape
    janela = torch.hann_window(N_FFT, dtype=x_uv.dtype)
    x = torch.stft(x_uv.reshape(n * canais, amostras), n_fft=N_FFT, hop_length=PASSO_STFT, win_length=N_FFT,
                   window=janela, center=False, return_complex=True)
    potencia = torch.log10(x.abs() ** 2 + EPS)[:, FREQUENCIAS]
    return potencia.reshape(n, canais, len(FREQUENCIAS), -1).float()


def estatisticas_stft(fonte, linhas: np.ndarray, tamanho_lote: int = 512) -> tuple[torch.Tensor, torch.Tensor]:
    """Média e desvio por (canal, frequência), sobre janelas e quadros das `linhas` (só treino)."""
    soma = soma_quadrados = None
    contagem = 0
    for i in range(0, len(linhas), tamanho_lote):
        x_e = fonte.lote(linhas[i:i + tamanho_lote])[1].double()
        s, q = x_e.sum(dim=(0, 3)), (x_e ** 2).sum(dim=(0, 3))
        soma, soma_quadrados = (s, q) if soma is None else (soma + s, soma_quadrados + q)
        contagem += x_e.shape[0] * x_e.shape[3]
    media = soma / contagem
    desvio = (soma_quadrados / contagem - media ** 2).clamp_min(0).sqrt()
    return media.float(), desvio.float()


class FonteDTSF:
    """Janelas de um conjunto de pessoas: `indice` (uma linha por janela) e `lote(linhas)` → (multibanda, STFT)."""

    def __init__(self, uids, janelas_csv: Path, cache: Path, janelas: Path):
        indice = pd.read_csv(janelas_csv)
        indice["linha"] = np.arange(len(indice))
        self.indice = indice[indice["subject_uid"].isin(set(uids))].reset_index(drop=True)
        self.sinais = {uid: np.load(cache / f"{uid}.npy", mmap_mode="r") for uid in uids}
        self.uv = {}
        for uid in uids:
            z = np.load(janelas / f"{uid}.npz")
            self.uv[uid] = (z["media"].astype(np.float32), z["desvio"].astype(np.float32))
        self._pessoa = self.indice.set_index("linha")["subject_uid"]
        self._inicio = self.indice.set_index("linha")["inicio_amostra"]

    def _janela(self, linha: int) -> np.ndarray:
        inicio = int(self._inicio[linha]) // 2
        return self.sinais[self._pessoa[linha]][:, :, inicio:inicio + AMOSTRAS_125].transpose(1, 0, 2)

    def lote(self, linhas: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        x_t = np.stack([self._janela(l) for l in linhas])                       # [n, 19, 5, 500]
        media = np.stack([self.uv[self._pessoa[l]][0] for l in linhas])[:, :, None]
        desvio = np.stack([self.uv[self._pessoa[l]][1] for l in linhas])[:, :, None]
        x_uv = torch.from_numpy(x_t[:, :, 0] * desvio + media).double()
        return torch.from_numpy(np.ascontiguousarray(x_t)), log_stft(x_uv)


def construir_cache(cache: Path, destino: Path) -> None:
    destino.mkdir(parents=True, exist_ok=True)
    for caminho in sorted(cache.glob("*.npy")):
        np.save(destino / caminho.name, multibanda(np.load(caminho)))
        print(caminho.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache multibanda a 125 Hz para o DTSF adaptado")
    parser.add_argument("--cache", type=Path, default=Path("outputs/07_DTSF/cache"))
    parser.add_argument("--destino", type=Path, default=Path("outputs/07_DTSF/cache_dtsf"))
    args = parser.parse_args()
    construir_cache(args.cache, args.destino)


if __name__ == "__main__":
    main()
