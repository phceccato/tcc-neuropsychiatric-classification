"""Mapas topográficos da potência por banda de uma única pessoa (delta, teta, alfa e beta).

Reaproveita, sem recalcular nem alterar nada:
- a potência de banda por janela de outputs/04_features/psd.npz (pipeline/e04_features/psd.py: Welch em
  janelas de 4 s, segmentos de 2 s, Hann, 50% de overlap; bandas delta [2, 4), teta [4, 8), alfa [8, 13) e
  beta [13, 30] Hz; `absolutas` = log10 da potência absoluta em µV²), alinhada linha a linha com
  outputs/03_janelas/janelas.csv;
- os 19 canais (`CANAIS`, pipeline/e01_auditoria) e a montagem 10–20 usada na interpolação de canais
  (`MONTAGEM` = colin27_1020, pipeline/e02_harmonizacao), que já traz os nomes antigos T3–T6. As posições
  são giradas até nivelar Fpz e Oz e transladadas para o centro da cabeça (`posicoes_niveladas`); o
  contorno envolve todos os eletrodos (`raio_contorno`).

Agregação: só as janelas aceitas (`rejeitada` = False) da pessoa escolhida. Em cada canal e banda, a
potência absoluta (10^absolutas, µV²) é a média aritmética entre as janelas; a escala do mapa sai dessa
média (`--escala`):
- `log` (padrão): log10 da potência absoluta média, em log10(µV²);
- `absoluta`: potência absoluta média, em µV²;
- `relativa`: potência média da banda ÷ soma das 4 bandas no mesmo canal, em %.

Escala de cor: por padrão, **comum a todas as pessoas** em cada banda (`limites_comuns`: mínimo e máximo
dos 19 canais de todas as 84 pessoas, na escala escolhida), para as cores serem comparáveis entre figuras.
Os outros participantes só definem os limites da barra; os valores do mapa são só da pessoa. Com
`--escala-individual`, cada banda usa o mínimo–máximo dos 19 canais da própria pessoa. A interpolação entre
os eletrodos é a do `mne.viz.plot_topomap`.

Grava outputs/10_topomapa/topomapa_<pessoa>_<escala>.png a 300 DPI.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from pipeline.e01_auditoria.auditar_ec import CANAIS
from pipeline.e02_harmonizacao.harmonizar import MONTAGEM
from pipeline.e04_features.psd import BANDAS, LIMITES_HZ, nomes_features

NOMES_BANDAS = {"delta": "Delta", "teta": "Teta", "alfa": "Alfa", "beta": "Beta"}
ESCALAS = {"log": "log10 da potência absoluta (µV²)", "absoluta": "potência absoluta (µV²)",
           "relativa": "potência relativa (% de 2–30 Hz)"}
DPI = 300


def janelas_da_pessoa(pessoa: str, psd: Path, janelas_csv: Path) -> np.ndarray:
    """log10 da potência absoluta [janelas aceitas, 19, 4] da pessoa, na ordem (canal, banda) de psd.py."""
    z = np.load(psd)
    indice = pd.read_csv(janelas_csv)
    if len(indice) != len(z["subject_uid"]) or not (indice["subject_uid"].to_numpy() == z["subject_uid"]).all():
        raise ValueError("psd.npz e janelas.csv não estão alinhados linha a linha")
    if list(z["nomes"]) != nomes_features():
        raise ValueError("a ordem das features de psd.npz não é a de psd.py")
    if pessoa not in set(indice["subject_uid"]):
        raise ValueError(f"pessoa desconhecida: {pessoa!r}")
    linhas = ((indice["subject_uid"] == pessoa) & ~indice["rejeitada"]).to_numpy()
    if not linhas.any():
        raise ValueError(f"{pessoa}: nenhuma janela aceita")
    x = z["absolutas"][linhas].astype(np.float64).reshape(-1, len(CANAIS), len(BANDAS))
    if not np.isfinite(x).all():
        raise ValueError(f"{pessoa}: potência ausente ou não finita")
    return x


def agregar(log_potencia: np.ndarray, escala: str) -> np.ndarray:
    """Valores [19, 4] do mapa: média da potência absoluta entre as janelas, na escala pedida."""
    media = (10.0 ** log_potencia).mean(axis=0)  # µV²
    if escala == "log":
        return np.log10(media)
    if escala == "absoluta":
        return media
    if escala == "relativa":
        return 100 * media / media.sum(axis=1, keepdims=True)
    raise ValueError(f"escala desconhecida: {escala!r}")


PERIFERICOS = ("Fp1", "Fp2", "F7", "F8", "T3", "T4", "T5", "T6", "O1", "O2")  # na circunferência Fpz–T3–Oz–T4


def posicoes_niveladas() -> dict[str, np.ndarray]:
    """Posições 3D (m) da montagem `MONTAGEM`, niveladas e centradas na cabeça.

    1. Rotação em torno do eixo x até Fpz e Oz ficarem na mesma altura: na colin27_1020 o plano Fpz–Oz é
       inclinado (Fpz mais baixo que Oz), e a projeção jogaria os eletrodos frontais para fora do contorno e
       os occipitais para dentro.
    2. Translação para o centro da cabeça ficar na origem: centro em x entre T7 e T8, em y entre Fpz e Oz, e
       em z na altura média dos quatro, que ficam na circunferência do 10–20 (o critério do
       `sphere="eeglab"` do MNE). O MNE mede o alcance do mapa a partir da origem (mne/viz/topomap.py,
       `mask_scale`), então um centro fora da origem faria o mapa passar do contorno.

    As duas operações são rígidas: não mudam a distância entre eletrodos nem criam posição nova.
    """
    pos = {c: np.asarray(p, dtype=float) for c, p in mne.channels.make_standard_montage(MONTAGEM).get_positions()["ch_pos"].items()}
    fpz, oz = pos["Fpz"], pos["Oz"]
    angulo = np.arctan2(oz[2] - fpz[2], fpz[1] - oz[1])
    c, s = np.cos(angulo), np.sin(angulo)
    rotacao = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    pos = {nome: rotacao @ p for nome, p in pos.items()}
    fpz, oz, t7, t8 = (pos[n] for n in ("Fpz", "Oz", "T7", "T8"))
    centro = np.array([(t7[0] + t8[0]) / 2, (fpz[1] + oz[1]) / 2, np.mean([fpz[2], oz[2], t7[2], t8[2]])])
    return {nome: p - centro for nome, p in pos.items()}


FOLGA_CONTORNO = 1.03


def raio_contorno(info: mne.Info, pos: dict[str, np.ndarray]) -> float:
    """Raio (m) do contorno da cabeça: o do círculo que passa por Fpz, Oz, T7 e T8 ou, se maior, o que envolve
    o eletrodo mais externo dos 19 com `FOLGA_CONTORNO`.

    Como a cabeça do colin27 não é esférica, os eletrodos periféricos caem a até ~9% fora do círculo de Fpz,
    Oz, T7 e T8; ampliar o contorno faz o mapa terminar nele, como na figura de referência. O raio só muda o
    desenho, não as posições projetadas (a projeção depende só do centro).
    """
    from mne.viz.topomap import _get_pos_outlines  # a projeção do próprio MNE, a mesma do desenho
    fpz, oz, t7, t8 = (pos[n] for n in ("Fpz", "Oz", "T7", "T8"))
    raio = float(np.mean([abs(fpz[1] - oz[1]), abs(t8[0] - t7[0])]) / 2)
    projetadas, _ = _get_pos_outlines(info, None, sphere=(0.0, 0.0, 0.0, raio))
    return max(raio, float(np.linalg.norm(projetadas, axis=1).max()) * FOLGA_CONTORNO)


def valores_de_todos(escala: str, psd: Path, janelas_csv: Path) -> dict[str, np.ndarray]:
    """Valores [19, 4] do mapa (`agregar`) de cada pessoa com janela aceita, lendo psd.npz uma vez só."""
    z = np.load(psd)
    indice = pd.read_csv(janelas_csv)
    if len(indice) != len(z["subject_uid"]) or not (indice["subject_uid"].to_numpy() == z["subject_uid"]).all():
        raise ValueError("psd.npz e janelas.csv não estão alinhados linha a linha")
    if list(z["nomes"]) != nomes_features():
        raise ValueError("a ordem das features de psd.npz não é a de psd.py")
    absolutas = z["absolutas"]
    saida = {}
    for pessoa, g in indice[~indice["rejeitada"]].groupby("subject_uid", sort=False):
        x = absolutas[g.index.to_numpy()].astype(np.float64).reshape(-1, len(CANAIS), len(BANDAS))
        if not np.isfinite(x).all():
            raise ValueError(f"{pessoa}: potência ausente ou não finita")
        saida[pessoa] = agregar(x, escala)
    return saida


def limites_comuns(valores: dict[str, np.ndarray]) -> np.ndarray:
    """[4, 2]: mínimo e máximo de cada banda sobre todos os canais de todas as pessoas."""
    todos = np.stack(list(valores.values()))  # [pessoas, 19, 4]
    return np.stack([todos.min(axis=(0, 1)), todos.max(axis=(0, 1))], axis=1)


def info_10_20() -> tuple[mne.Info, tuple[float, float, float, float]]:
    """Info dos 19 canais com as posições niveladas e centradas, e a esfera (0, 0, 0, raio) do contorno."""
    pos = posicoes_niveladas()
    faltam = [c for c in CANAIS if c not in pos]
    if faltam:
        raise ValueError(f"canais sem coordenada na montagem {MONTAGEM}: {faltam}")
    info = mne.create_info(list(CANAIS), 250.0, "eeg")
    info.set_montage(mne.channels.make_dig_montage(ch_pos={c: pos[c] for c in CANAIS}, coord_frame="head"),
                     verbose="ERROR")
    return info, (0.0, 0.0, 0.0, raio_contorno(info, pos))


def figura(valores: np.ndarray, escala: str, titulo: str, destino: Path, limites: np.ndarray | None = None,
           n_pessoas: int | None = None) -> Path:
    """Os 4 mapas; `limites` [4, 2] fixa a barra de cada banda (escala comum), senão é o mín–máx da pessoa."""
    info, esfera = info_10_20()
    fig, eixos = plt.subplots(1, len(BANDAS), figsize=(3.0 * len(BANDAS), 3.6))
    for i, (eixo, banda) in enumerate(zip(eixos, BANDAS)):
        v = valores[:, i]
        vmin, vmax = (v.min(), v.max()) if limites is None else (float(limites[i, 0]), float(limites[i, 1]))
        if v.min() < vmin - 1e-12 or v.max() > vmax + 1e-12:
            raise ValueError(f"{banda}: valores fora dos limites da escala comum")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            imagem, _ = mne.viz.plot_topomap(v, info, axes=eixo, show=False, cmap="jet", vlim=(vmin, vmax),
                                             contours=np.linspace(vmin, vmax, 8)[1:-1], sensors=True, extrapolate="head", res=128,
                                             sphere=esfera)
        baixo, alto = LIMITES_HZ[banda]
        eixo.set_title(f"{NOMES_BANDAS[banda]} ({baixo:g}–{alto:g} Hz)".replace(".", ","), fontsize=10)
        barra = fig.colorbar(imagem, ax=eixo, orientation="horizontal", fraction=0.06, pad=0.04, shrink=0.85)
        barra.ax.tick_params(labelsize=7)
        barra.ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
            lambda x, _: f"{x:.3g}".replace(".", ",")))
        barra.set_ticks(np.linspace(vmin, vmax, 3))
    comum = (f"barra comum às {n_pessoas} pessoas em cada banda" if n_pessoas else "barra comum a todas as pessoas em cada banda")
    fig.suptitle(f"{titulo}\nEscala: {ESCALAS[escala]}; média das janelas aceitas; "
                 + (comum if limites is not None else "barra individual (mín–máx da pessoa) em cada banda"), fontsize=10)
    fig.tight_layout()
    destino.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destino, dpi=DPI, facecolor="white")
    plt.close(fig)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pessoa", required=True, help="subject_uid, por exemplo MDD-MDD-S27")
    parser.add_argument("--escala", choices=tuple(ESCALAS), default="log")
    parser.add_argument("--psd", type=Path, default=Path("outputs/04_features/psd.npz"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/10_topomapa"))
    parser.add_argument("--escala-individual", action="store_true", help="barra de cada banda pelo mín–máx da pessoa")
    args = parser.parse_args()

    x = janelas_da_pessoa(args.pessoa, args.psd, args.janelas)
    valores = agregar(x, args.escala)
    limites, n = None, None
    if not args.escala_individual:
        todos = valores_de_todos(args.escala, args.psd, args.janelas)
        limites, n = limites_comuns(todos), len(todos)
    sufixo = "_individual" if args.escala_individual else ""
    destino = figura(valores, args.escala, f"{args.pessoa} ({len(x)} janelas de 4 s)",
                     args.saida / f"topomapa_{args.pessoa}_{args.escala}{sufixo}.png", limites, n)
    print(f"{args.pessoa}: {len(x)} janelas aceitas → {destino}")
    for i, banda in enumerate(BANDAS):
        print(f"  {banda}: {valores[:, i].min():.4g} a {valores[:, i].max():.4g} ({ESCALAS[args.escala]})")


if __name__ == "__main__":
    main()
