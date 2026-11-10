"""Cache do sinal a 125 Hz e seleção das janelas, comuns ao DTSF (pipeline/e07_DTSF) e aos baselines clássicos.

O sinal de outputs/03_janelas (2–30 Hz, µV) recebe o z-score por canal da própria pessoa
(média e desvio das janelas aceitas) e é reamostrado para 125 Hz uma vez, em
outputs/07_DTSF/cache/<subject_uid>.npy; o cache multibanda do DTSF (dados_dtsf.py) parte dele.

Cada janela é identificada pela `linha` em janelas.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

AMOSTRAS_125 = 500  # 4 s a 125 Hz


def preparar_sinal(x: np.ndarray, media: np.ndarray, desvio: np.ndarray) -> np.ndarray:
    """z-score por canal e reamostragem de 250 para 125 Hz."""
    z = (np.asarray(x, dtype=np.float64) - media[:, None]) / desvio[:, None]
    return resample_poly(z, 1, 2, axis=1).astype(np.float32)


def sortear(indice_pessoa: pd.DataFrame, por_pessoa: int, rng: np.random.Generator) -> np.ndarray:
    """`por_pessoa` janelas aceitas da grade de treino (50% de overlap), sem repetição."""
    aceitas = indice_pessoa.loc[~indice_pessoa["rejeitada"], "linha"].to_numpy()
    return rng.choice(aceitas, size=min(por_pessoa, len(aceitas)), replace=False)


def janelas_avaliacao(indice_pessoa: pd.DataFrame) -> np.ndarray:
    """Janelas aceitas da grade sem overlap (validação e teste)."""
    usar = ~indice_pessoa["rejeitada"] & indice_pessoa["sem_overlap"]
    return indice_pessoa.loc[usar, "linha"].to_numpy()


def construir_cache(janelas: Path, cache: Path) -> None:
    cache.mkdir(parents=True, exist_ok=True)
    for caminho in sorted(janelas.glob("*.npz")):
        z = np.load(caminho)
        np.save(cache / f"{caminho.stem}.npy", preparar_sinal(z["x"], z["media"], z["desvio"]))
        print(caminho.stem)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache do sinal z-score a 125 Hz para os modelos")
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas"))
    parser.add_argument("--cache", type=Path, default=Path("outputs/07_DTSF/cache"))
    args = parser.parse_args()
    construir_cache(args.janelas, args.cache)


if __name__ == "__main__":
    main()
