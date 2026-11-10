"""Características dos baselines clássicos, extraídas das mesmas entradas do DTSF adaptado.

As entradas vêm de `FonteDTSF.lote` (pipeline/e07_DTSF/dados_dtsf.py), sem nenhuma outra transformação:

- Temporais (475 = 19 canais × 5 representações × 5 estatísticas), sobre os 500 pontos da janela de
  cada (canal, representação) do ramo temporal [19, 5, 500]. As representações são o sinal base do DTSF
  (2–30 Hz, z-score por canal da própria pessoa, 125 Hz) e as bandas δ, θ, α e β. Com x a janela e
  N = 500, em float64:
    desvio      = sqrt(m2), com m_k = (1/N) Σ (x − x̄)^k (ddof = 0)
    rms         = sqrt((1/N) Σ x²)
    pico_a_pico = max(x) − min(x)
    assimetria  = m3 / m2^(3/2)        (g1 de Fisher-Pearson, sem correção de viés)
    curtose     = m4 / m2² − 3         (curtose em excesso: 0 na normal; sem correção de viés)
  São as convenções de `np.std` e de `scipy.stats.skew` e `scipy.stats.kurtosis` com os padrões
  (bias=True, fisher=True).
- Espectrais (1102 = 19 canais × 29 frequências × 2 estatísticas): média e desvio (ddof = 0) ao longo
  dos 16 quadros (último eixo) da STFT do ramo espectral, log10(|X|² + 1e-12) em 2–30 Hz
  (`dados_dtsf.log_stft`).

Ordem dos vetores: canal → representação (ou frequência) → estatística, a ordem C de [19, 5, 5] e
[19, 29, 2]; os nomes estão em `nomes_temporais` e `nomes_espectrais`.

Casos degenerados:
- valor não finito na entrada → ValueError (não há imputação);
- (canal, representação) constante na janela, desvio ≤ DESVIO_MINIMO → assimetria = curtose = 0, por
  convenção (são indefinidas); desvio, rms e pico a pico seguem as fórmulas. A extração conta esses casos.

Cada janela é resumida sozinha: nenhuma estatística usa outra janela ou outra pessoa. A normalização das
características é ajustada depois, só com as pessoas de treino de cada fold (pipeline/e08_classicos/treino.py).

Saída de `python -m pipeline.e08_classicos.caracteristicas`: outputs/08_classicos/caracteristicas/temporais.npy [linhas, 475] e
espectrais.npy [linhas, 1102], float32, alinhados com as linhas de outputs/03_janelas/janelas.csv (NaN nas
janelas rejeitadas, que nunca são calculadas), e extracao.json (contagens, hash do janelas.csv e nomes).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.e01_auditoria.auditar_ec import CANAIS
from pipeline.e07_DTSF.dados_dtsf import FREQUENCIAS, FonteDTSF

REPRESENTACOES = ("base", "delta", "teta", "alfa", "beta")
ESTATISTICAS_TEMPORAIS = ("desvio", "rms", "pico_a_pico", "assimetria", "curtose")
ESTATISTICAS_ESPECTRAIS = ("media", "desvio")
N_TEMPORAIS = len(CANAIS) * len(REPRESENTACOES) * len(ESTATISTICAS_TEMPORAIS)    # 475
N_ESPECTRAIS = len(CANAIS) * len(FREQUENCIAS) * len(ESTATISTICAS_ESPECTRAIS)     # 1102
DESVIO_MINIMO = 1e-10  # em unidades do z-score do ramo temporal
TAMANHO_LOTE = 256


def nomes_temporais() -> list[str]:
    return [f"{c}|{r}|{e}" for c in CANAIS for r in REPRESENTACOES for e in ESTATISTICAS_TEMPORAIS]


def nomes_espectrais() -> list[str]:
    return [f"{c}|{f}Hz|{e}" for c in CANAIS for f in FREQUENCIAS for e in ESTATISTICAS_ESPECTRAIS]


def _finito(x: np.ndarray, nome: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if not np.isfinite(x).all():
        raise ValueError(f"entrada {nome} com valor não finito")
    return x


def constantes(x_t: np.ndarray) -> np.ndarray:
    """[janelas, 19, 5] → True onde o (canal, representação) é constante na janela."""
    return _finito(x_t, "temporal").std(axis=-1) <= DESVIO_MINIMO


def caracteristicas_temporais(x_t: np.ndarray) -> np.ndarray:
    """[janelas, 19, 5, amostras] → [janelas, 475]."""
    x = _finito(x_t, "temporal")
    centrado = x - x.mean(axis=-1, keepdims=True)
    m2, m3, m4 = ((centrado ** k).mean(axis=-1) for k in (2, 3, 4))
    desvio = np.sqrt(m2)
    constante = desvio <= DESVIO_MINIMO
    m2_seguro = np.where(constante, 1.0, m2)
    assimetria = np.where(constante, 0.0, m3 / m2_seguro ** 1.5)
    curtose = np.where(constante, 0.0, m4 / m2_seguro ** 2 - 3.0)
    rms = np.sqrt((x ** 2).mean(axis=-1))
    pico_a_pico = x.max(axis=-1) - x.min(axis=-1)
    return np.stack([desvio, rms, pico_a_pico, assimetria, curtose], axis=-1).reshape(len(x), -1)


def caracteristicas_espectrais(x_e: np.ndarray) -> np.ndarray:
    """[janelas, 19, 29, 16] (log10 da potência) → [janelas, 1102]: média e desvio nos quadros."""
    x = _finito(x_e, "espectral")
    return np.stack([x.mean(axis=-1), x.std(axis=-1)], axis=-1).reshape(len(x), -1)


def hash_arquivo(caminho: Path) -> str:
    return hashlib.sha256(Path(caminho).read_bytes()).hexdigest()


def extrair(janelas_csv: Path, cache_dtsf: Path, janelas: Path, destino: Path) -> dict:
    """Calcula as características das janelas aceitas de todas as pessoas e grava em `destino`."""
    indice = pd.read_csv(janelas_csv)
    for uid in indice["subject_uid"].unique():
        canais = tuple(str(c) for c in np.load(janelas / f"{uid}.npz")["canais"])
        if canais != CANAIS:
            raise ValueError(f"{uid}: canais {canais} diferentes de {CANAIS}")
    fonte = FonteDTSF(indice["subject_uid"].unique(), janelas_csv, cache_dtsf, janelas)
    aceitas = fonte.indice.loc[~fonte.indice["rejeitada"], "linha"].to_numpy()

    temporais = np.full((len(indice), N_TEMPORAIS), np.nan, dtype=np.float32)
    espectrais = np.full((len(indice), N_ESPECTRAIS), np.nan, dtype=np.float32)
    n_constantes, minimo_log = 0, np.inf
    for i in range(0, len(aceitas), TAMANHO_LOTE):
        linhas = aceitas[i:i + TAMANHO_LOTE]
        x_t, x_e = (x.numpy() for x in fonte.lote(linhas))
        temporais[linhas] = caracteristicas_temporais(x_t)
        espectrais[linhas] = caracteristicas_espectrais(x_e)
        n_constantes += int(constantes(x_t).sum())
        minimo_log = min(minimo_log, float(x_e.min()))

    destino.mkdir(parents=True, exist_ok=True)
    np.save(destino / "temporais.npy", temporais)
    np.save(destino / "espectrais.npy", espectrais)
    resumo = dict(linhas=len(indice), aceitas=len(aceitas), janelas_csv_sha256=hash_arquivo(janelas_csv),
                  constantes=n_constantes, minimo_log10_potencia=minimo_log,
                  nomes_temporais=nomes_temporais(), nomes_espectrais=nomes_espectrais())
    (destino / "extracao.json").write_text(json.dumps(resumo, indent=1, ensure_ascii=False), encoding="utf-8")
    return resumo


def carregar(destino: Path, janelas_csv: Path) -> tuple[np.ndarray, np.ndarray]:
    """(temporais, espectrais) por mmap, depois de conferir que vêm do mesmo janelas.csv."""
    resumo = json.loads((destino / "extracao.json").read_text(encoding="utf-8"))
    if resumo["janelas_csv_sha256"] != hash_arquivo(janelas_csv):
        raise ValueError("as características foram extraídas de outro janelas.csv: rode a extração de novo")
    return np.load(destino / "temporais.npy", mmap_mode="r"), np.load(destino / "espectrais.npy", mmap_mode="r")


def main() -> None:
    parser = argparse.ArgumentParser(description="Características temporais e espectrais dos baselines clássicos")
    parser.add_argument("--janelas-csv", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--cache-dtsf", type=Path, default=Path("outputs/07_DTSF/cache_dtsf"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas"))
    parser.add_argument("--destino", type=Path, default=Path("outputs/08_classicos/caracteristicas"))
    args = parser.parse_args()
    resumo = extrair(args.janelas_csv, args.cache_dtsf, args.janelas, args.destino)
    print({k: v for k, v in resumo.items() if not k.startswith("nomes")})


if __name__ == "__main__":
    main()
