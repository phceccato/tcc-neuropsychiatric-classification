"""Figura em pgfplots: distribuição nula da H1 (fig:res-h1-nula).

Dois painéis (SZ e MDD): dot plot das AUCs do modelo principal (D_FA) retreinado com rótulos permutados, a AUC
observada que a H1 compara com elas (AUC da repetição 0, a mesma de `p_acaso`) e o acaso (0,5).

Lê outputs/11_graficos/numeros.json (`numeros.py`, já conferido contra os resumos das etapas): as AUCs nulas, a
AUC observada e o p de cada dataset. Grava outputs/11_graficos/pgfplots/fig_h1_nula.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pipeline.e11_graficos.comum import ESTILO_EIXOS, SAIDA, coordenadas, figura, gravar, virgula
from pipeline.e11_graficos.numeros import SAIDA as NUMEROS

DATASETS = ("SZ", "MDD")
LARGURA_CLASSE = 0.02


def empilhar(valores: np.ndarray, largura: float = LARGURA_CLASSE) -> list[tuple[float, int]]:
    """Dot plot: cada valor vai para o centro da sua classe de `largura` e sobe uma posição por repetido."""
    centros = (np.floor(np.asarray(valores) / largura) + 0.5) * largura
    contagem: dict[float, int] = {}
    pontos = []
    for c in np.round(np.sort(centros), 6):
        contagem[c] = contagem.get(c, 0) + 1
        pontos.append((float(c), contagem[c]))
    return pontos


def dados(numeros: dict) -> dict[str, dict]:
    """AUCs nulas, AUC observada (repetição 0) e p de cada dataset."""
    return {d: dict(nulas=np.asarray(numeros["h1_distribuicao"][d]["nulas"]),
                    observada=numeros["h1_distribuicao"][d]["observada"], p=numeros["h1"][d]["p"])
            for d in DATASETS}


def tex(dados_h1: dict[str, dict]) -> str:
    pilha = max(max(n for _, n in empilhar(r["nulas"])) for r in dados_h1.values())
    altura = pilha + 1.5
    xmin = np.floor(min(0.5, *(r["nulas"].min() for r in dados_h1.values())) * 10) / 10 - 0.05
    n = len(next(iter(dados_h1.values()))["nulas"])
    paineis = []
    for i, (d, r) in enumerate(dados_h1.items()):
        legenda = ("\\legend{AUC com rótulos permutados, AUC observada (repetição 0), Acaso (0{,}5)}"
                   if i == 0 else "")
        ancora = "south east" if r["observada"] > 0.75 else "south west"
        paineis.append(
            f"\\nextgroupplot[title={{{d}}}{', ylabel={Permutações}' if i == 0 else ''}]\n"
            f"\\addplot[only marks, mark=*, mark size=1.6pt, draw=black, fill=gray!55] {coordenadas(empilhar(r['nulas']))};\n"
            f"\\addplot[black, very thick] {coordenadas([(r['observada'], 0), (r['observada'], altura)])};\n"
            f"\\addplot[gray, dashed, thick] {coordenadas([(0.5, 0), (0.5, altura)])};\n"
            f"\\node[anchor={ancora}, font=\\footnotesize] at (axis cs:{r['observada']:.6g},{altura - 0.9:.6g}) "
            f"{{{virgula(r['observada'])}; p = {virgula(r['p'])}}};\n" + legenda)
    corpo = ("\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.4cm}, "
             f"{ESTILO_EIXOS}, scale only axis, width=0.38\\linewidth, height=0.24\\linewidth, "
             f"xmin={xmin:.2f}, xmax=1.0, ymin=0, ymax={altura}, xlabel={{AUC}}, ytick={{{','.join(map(str, range(pilha + 1)))}}}, "
             "xtick distance=0.1, xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, "
             "legend style={at={(1.09,-0.32)}, anchor=north, legend columns=3, /tikz/every even column/.append "
             "style={column sep=0.4cm}}]\n"
             + "\n".join(paineis) + "\n\\end{groupplot}\n\\end{tikzpicture}")
    ps = " e ".join(f"{virgula(r['p'])} {'na' if d == 'SZ' else 'no'} {d}" for d, r in dados_h1.items())
    observadas = " e ".join(f"{virgula(r['observada'])} {'na' if d == 'SZ' else 'no'} {d}" for d, r in dados_h1.items())
    legenda = (f"Distribuição nula da H1: AUC do D\\_FA retreinado com rótulos permutados ({n} permutações, "
               f"repetição 0) e AUC observada. p = {ps} (o menor possível com {n} permutações é "
               f"{virgula(1 / (n + 1))}). A linha contínua é a AUC da repetição 0, a que o teste compara com as nulas "
               f"({observadas}); a tracejada marca o acaso. Cada ponto é uma permutação, agrupada em classes de 0,02.")
    return figura(corpo, legenda, "fig:res-h1-nula")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numeros", type=Path, default=NUMEROS / "numeros.json")
    parser.add_argument("--saida", type=Path, default=SAIDA / "fig_h1_nula.tex")
    args = parser.parse_args()
    gravar(tex(dados(json.loads(args.numeros.read_text(encoding="utf-8")))), args.saida)


if __name__ == "__main__":
    main()
