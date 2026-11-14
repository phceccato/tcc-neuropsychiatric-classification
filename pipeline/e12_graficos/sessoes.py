"""Figura 3: linha do tempo das gravações por classe, com as sessões reconstruídas (fig:des-sessoes).

Fonte: data de gravação do cabeçalho EDF em outputs/02_harmonizacao/manifesto.csv. Sessões e classes
vêm de `pessoas_por_dataset` (pipeline/e09_investigacao/datas.py: sessão nova quando duas datas
consecutivas distam mais de 30 dias). Dois painéis empilhados (SZ e MDD), eixo x em anos decimais com
rótulos de data, uma linha por classe e um ponto por pessoa. Pontos com datas próximas na mesma linha
(a até `FRACAO_EMPILHAR` da largura do eixo, o que inclui a mesma data) se empilham com deslocamento
vertical determinístico (0, +d, −d, +2d, ...), sem sorteio.

Antes de gravar, confere as contagens por sessão contra docs/09_investigacao.md, seção 1
(`CONTAGENS_DOCS`). Nenhum identificador vai para o .tex: só a posição de cada ponto.

Grava outputs/12_graficos/pgfplots/fig_sessoes.tex.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.e09_investigacao.datas import INTERVALO_SESSAO_DIAS, pessoas_por_dataset
from pipeline.e12_graficos.comum import ESTILO_EIXOS, SAIDA, coordenadas, figura, gravar

# (HC, pacientes) por sessão; na SZ, as sessões 4–9 aparecem somadas no doc
CONTAGENS_DOCS = {"SZ": {1: (0, 1), 2: (14, 0), 3: (0, 2), (4, 9): (0, 11)},
                  "MDD": {1: (9, 0), 2: (14, 0), 3: (4, 29)}}
SESSOES_DOCS = {"SZ": 9, "MDD": 3}
FRACAO_EMPILHAR = 1 / 80
PASSO = 0.07
MARCAS = {0: "mark=o, draw=gray!60!black, mark size=1.7pt", 1: "mark=square*, draw=black, fill=black, mark size=1.5pt"}
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
LEGENDA = ("Data de gravação de cada participante, segundo o cabeçalho EDF, por classe. As faixas marcam as sessões "
           f"reconstruídas (nova sessão quando duas datas distam mais de {INTERVALO_SESSAO_DIAS} dias).")


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


def painel(dataset: str, pessoas: pd.DataFrame, primeiro: bool) -> str:
    x = ano_decimal(pessoas["data_gravacao"])
    largura = x.max() - x.min()
    xmin, xmax = x.min() - RESPIRO * largura * 2, x.max() + RESPIRO * largura * 2
    y = pessoas["y"].to_numpy()
    jitter = np.zeros(len(x))
    for classe in (0, 1):
        m = y == classe
        jitter[m] = deslocamentos(x[m], FRACAO_EMPILHAR * largura)
    ticks, rotulos = rotulos_x(xmin, xmax)
    linhas = [f"\\nextgroupplot[title={{{dataset}}}, xmin={xmin:.4f}, xmax={xmax:.4f}, "
              f"xtick={{{','.join(f'{t:.4f}' for t in ticks)}}}, xticklabels={{{','.join(rotulos)}}}, "
              f"yticklabels={{HC,{dataset}}}{', xlabel={Data de gravação}' if not primeiro else ''}]"]
    limites = pessoas.assign(x=x).groupby("sessao")["x"].agg(["min", "max"])
    for s, (a, b) in zip(limites.index, faixas(limites["min"].to_numpy(), limites["max"].to_numpy(), largura)):
        linhas.append(f"\\addplot[fill=gray!18, draw=none, forget plot] "
                      f"{coordenadas([(a, -0.6), (b, -0.6), (b, 1.6), (a, 1.6)])} -- cycle;")
        linhas.append(f"\\node[font=\\scriptsize, anchor=south] at (axis cs:{(a + b) / 2:.4f},1.6) {{{s}}};")
    for classe in (0, 1):
        m = y == classe
        linhas.append(f"\\addplot[only marks, {MARCAS[classe]}] {coordenadas(zip(x[m], classe + jitter[m]))};")
    return "\n".join(linhas)


def tex(tabelas: dict[str, pd.DataFrame]) -> str:
    paineis = [painel(d, tabelas[d], i == 0) for i, d in enumerate(("SZ", "MDD"))]
    corpo = ("\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=1 by 2, vertical sep=1.3cm}, "
             f"{ESTILO_EIXOS}, scale only axis, width=0.82\\linewidth, height=0.17\\linewidth, axis on top, "
             "ymin=-0.6, ymax=2.0, ytick={0,1}, tick align=outside, xtick pos=bottom, ytick pos=left, "
             "xticklabel style={font=\\scriptsize}]\n" + "\n\n".join(paineis) + "\n\\end{groupplot}\n\\end{tikzpicture}")
    legenda = (LEGENDA + " Os números identificam as sessões. Círculos: controles (HC); quadrados: pacientes. "
               "Pontos de datas muito próximas se empilham na vertical para não se sobreporem.")
    return figura(corpo, legenda, "fig:des-sessoes")


def dados(manifesto: Path) -> dict[str, pd.DataFrame]:
    tabelas = pessoas_por_dataset(pd.read_csv(manifesto))
    for d, pessoas in tabelas.items():
        conferir_contagens(d, pessoas)
    return tabelas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--saida", type=Path, default=SAIDA / "fig_sessoes.tex")
    args = parser.parse_args()
    gravar(tex(dados(args.manifesto)), args.saida)


if __name__ == "__main__":
    main()
