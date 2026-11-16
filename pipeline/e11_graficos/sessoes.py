"""Linha do tempo das gravações por classe, com as sessões reconstruídas (s12_sessoes, em PDF e PNG).

Fonte: data de gravação do cabeçalho EDF em outputs/02_harmonizacao/manifesto.csv. Sessões e classes
vêm de `pessoas_por_dataset` (pipeline/e09_investigacao/datas.py: sessão nova quando duas datas
consecutivas distam mais de 30 dias). Dois painéis empilhados (SZ e MDD), eixo x em anos decimais com
rótulos de data, uma linha por classe e um ponto por pessoa. Pontos com datas próximas na mesma linha
(a até `FRACAO_EMPILHAR` da largura do eixo, o que inclui a mesma data) se empilham com deslocamento
vertical determinístico (0, +d, −d, +2d, ...), sem sorteio.

Antes de gravar, confere as contagens por sessão contra docs/09_investigacao.md, seção 1
(`CONTAGENS_DOCS`). Nenhum identificador vai para a figura: só a posição de cada ponto.

Precisa da etapa 02 (o manifesto), que não vai para o git; por isso não faz parte do `graficos.py`, que lê só
outputs/10_resultados. Grava outputs/11_graficos/suplementares/s12_sessoes.pdf e .png.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.e09_investigacao.datas import INTERVALO_SESSAO_DIAS, pessoas_por_dataset
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from pipeline.e11_graficos.graficos import CINZA, PRETO, SAIDA, SUPLEMENTARES, _gravar, configurar  # noqa: E402

# (HC, pacientes) por sessão; na SZ, as sessões 4–9 aparecem somadas no doc
CONTAGENS_DOCS = {"SZ": {1: (0, 1), 2: (14, 0), 3: (0, 2), (4, 9): (0, 11)},
                  "MDD": {1: (9, 0), 2: (14, 0), 3: (4, 29)}}
SESSOES_DOCS = {"SZ": 9, "MDD": 3}
FRACAO_EMPILHAR = 1 / 80
PASSO = 0.07
MARCAS = {0: dict(marker="o", mfc="white", mec=CINZA, ms=3.4), 1: dict(marker="s", mfc=PRETO, mec=PRETO, ms=3.2)}
RESPIRO = 0.02  # margem do eixo, em fração da largura
LARGURA_MINIMA = 0.015  # largura mínima de uma faixa de sessão, em fração da largura do eixo


def faixas(inicios: np.ndarray, fins: np.ndarray, largura: float) -> list[tuple[float, float]]:
    """Faixas das sessões (em ordem): alargadas até `LARGURA_MINIMA` em torno do centro, sem invadir o meio
    da lacuna até a sessão vizinha."""
    minimo = LARGURA_MINIMA * largura
    saida = []
    for i, (a, b) in enumerate(zip(inicios, fins)):
        folga = max(0.0, (minimo - (b - a)) / 2)
        a2, b2 = a - folga, b + folga
        if i > 0:
            a2 = max(a2, (fins[i - 1] + a) / 2)
        if i + 1 < len(inicios):
            b2 = min(b2, (b + inicios[i + 1]) / 2)
        saida.append((a2, b2))
    return saida

def ano_decimal(datas: pd.Series) -> np.ndarray:
    d = pd.to_datetime(datas)
    return (d.dt.year + (d.dt.dayofyear - 1) / np.where(d.dt.is_leap_year, 366, 365)).to_numpy(float)


def deslocamentos(x: np.ndarray, tolerancia: float, passo: float = PASSO) -> np.ndarray:
    """Deslocamento vertical de cada ponto: datas a até `tolerancia` da primeira data da pilha entram nela."""
    ordem = np.argsort(x, kind="stable")
    saida = np.zeros(len(x))
    inicio = 0
    for i in range(1, len(ordem) + 1):
        if i == len(ordem) or x[ordem[i]] - x[ordem[inicio]] > tolerancia:
            pilha = ordem[inicio:i]
            d = min(passo, 0.42 / max(1, (len(pilha) - 1) // 2 + (len(pilha) - 1) % 2))
            k = np.arange(len(pilha))
            saida[pilha] = d * ((k + 1) // 2) * np.where(k % 2 == 1, 1, -1)
            inicio = i
    return saida


def conferir_contagens(dataset: str, pessoas: pd.DataFrame) -> None:
    if pessoas["sessao"].nunique() != SESSOES_DOCS[dataset]:
        raise ValueError(f"{dataset}: {pessoas['sessao'].nunique()} sessões, docs {SESSOES_DOCS[dataset]}")
    for chave, esperado in CONTAGENS_DOCS[dataset].items():
        a, b = chave if isinstance(chave, tuple) else (chave, chave)
        g = pessoas[pessoas["sessao"].between(a, b)]
        obtido = (int((g["y"] == 0).sum()), int(g["y"].sum()))
        if obtido != esperado:
            raise ValueError(f"{dataset} sessão {chave}: (HC, pacientes) = {obtido}, docs {esperado}")


def rotulos_x(xmin: float, xmax: float) -> tuple[list[float], list[str]]:
    """Ticks anuais se o eixo passa de 4 anos; senão semestrais (jan e jul)."""
    if xmax - xmin > 4:
        anos = range(int(np.ceil(xmin)), int(np.floor(xmax)) + 1)
        return [float(a) for a in anos], [str(a) for a in anos]
    ticks, rotulos = [], []
    for a in range(int(np.floor(xmin)), int(np.ceil(xmax)) + 1):
        for meio, nome in ((0.0, "jan"), (181 / 365, "jul")):
            if xmin <= a + meio <= xmax:
                ticks.append(a + meio)
                rotulos.append(f"{nome}/{a}")
    return ticks, rotulos


def painel(ax, dataset: str, pessoas: pd.DataFrame, primeiro: bool) -> None:
    x = ano_decimal(pessoas["data_gravacao"])
    largura = x.max() - x.min()
    xmin, xmax = x.min() - RESPIRO * largura * 2, x.max() + RESPIRO * largura * 2
    y = pessoas["y"].to_numpy()
    jitter = np.zeros(len(x))
    for classe in (0, 1):
        m = y == classe
        jitter[m] = deslocamentos(x[m], FRACAO_EMPILHAR * largura)
    limites = pessoas.assign(x=x).groupby("sessao")["x"].agg(["min", "max"])
    for s, (a, b) in zip(limites.index, faixas(limites["min"].to_numpy(), limites["max"].to_numpy(), largura)):
        ax.axvspan(a, b, color="#e4e4e4", lw=0, zorder=0)
        ax.text((a + b) / 2, 1.62, str(s), ha="center", va="bottom", fontsize=6.5)
    for classe in (0, 1):
        m = y == classe
        ax.plot(x[m], classe + jitter[m], ls="none", zorder=3, **MARCAS[classe])
    ticks, rotulos = rotulos_x(xmin, xmax)
    ax.set(xlim=(xmin, xmax), ylim=(-0.6, 2.0), yticks=[0, 1], yticklabels=["HC", dataset])
    ax.set_xticks(ticks, rotulos, fontsize=6.5)
    ax.set_title(dataset, loc="left")
    if not primeiro:
        ax.set_xlabel("Data de gravação")


def figura(tabelas: dict[str, pd.DataFrame]):
    configurar()
    fig, eixos = plt.subplots(2, 1, figsize=(16 / 2.54, 6.2), gridspec_kw=dict(hspace=0.55))
    for i, (ax, d) in enumerate(zip(eixos, ("SZ", "MDD"))):
        painel(ax, d, tabelas[d], i == 0)
    fig.legend(handles=[Line2D([], [], ls="none", label="controles (HC)", **MARCAS[0]),
                        Line2D([], [], ls="none", label="pacientes", **MARCAS[1])],
               loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2)
    return fig


def dados(manifesto: Path) -> dict[str, pd.DataFrame]:
    tabelas = pessoas_por_dataset(pd.read_csv(manifesto))
    for d, pessoas in tabelas.items():
        conferir_contagens(d, pessoas)
    return tabelas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    for caminho in _gravar(figura(dados(args.manifesto)), args.saida / SUPLEMENTARES, "s12_sessoes"):
        print(caminho)


if __name__ == "__main__":
    main()
