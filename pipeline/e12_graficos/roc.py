"""Figura em pgfplots: curvas ROC por pessoa do modelo principal e dos baselines da H4 (fig:res-roc).

Escore de cada pessoa: média da probabilidade fora do fold nas 10 repetições, o mesmo da métrica primária.
Modelos (`MODELOS`): o D_FA e os 3 baselines combinados (LR, RF e KNN sobre as características temporais e
espectrais), os que a H4 compara. ROC empírica em degraus (`roc_curve` sem descartar pontos intermediários),
com a área conferida contra `roc_auc_score` e contra a AUC de numeros.json.

Lê outputs/12_graficos/numeros.json (`numeros.py`: rótulo e escore médio de cada pessoa, sem identificador).
Grava outputs/12_graficos/pgfplots/fig_roc.tex.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from pipeline.e12_graficos.comum import ESTILO_EIXOS, SAIDA, conferir, coordenadas, figura, gravar, virgula
from pipeline.e12_graficos.numeros import SAIDA as NUMEROS

DATASETS = ("SZ", "MDD")
# (rótulo na legenda, modelo, estilo pgfplots)
MODELOS = [("D\\_FA", "D_FA", "black, very thick"),
           ("LR combinado", "LR_combinado", "gray!70!black, thick, dashed"),
           ("RF combinado", "RF_combinado", "blue!70!black, thick, densely dotted"),
           ("KNN combinado", "KNN_combinado", "orange!80!black, thick, dash dot")]


def curva(y: np.ndarray, escore: np.ndarray) -> tuple[list[tuple[float, float]], float]:
    """Pontos (FPR, TPR) da ROC empírica em degraus e a AUC, conferida pela área sob os pontos."""
    fpr, tpr, _ = roc_curve(y, escore, drop_intermediate=False)
    auc = float(roc_auc_score(y, escore))
    if abs(float(np.trapezoid(tpr, fpr)) - auc) > 1e-9:
        raise ValueError("a área sob a curva não bate com roc_auc_score")
    return list(zip(fpr.tolist(), tpr.tolist())), auc


def dados(numeros: dict) -> dict[str, list[dict]]:
    saida = {}
    for d in DATASETS:
        saida[d] = []
        for rotulo, modelo, estilo in MODELOS:
            r = numeros["modelos"][d][modelo]
            pontos, auc = curva(np.asarray(r["escores"]["y"]), np.asarray(r["escores"]["prob"]))
            conferir(f"AUC por pessoa de {modelo} ({d})", auc, r["auc"])
            saida[d].append(dict(rotulo=rotulo, modelo=modelo, estilo=estilo, pontos=pontos, auc=auc))
    return saida


def tex(curvas: dict[str, list[dict]]) -> str:
    paineis = []
    for i, (d, modelos) in enumerate(curvas.items()):
        linhas = [f"\\nextgroupplot[title={{{d}}}{', ylabel={Sensibilidade}' if i == 0 else ''}]"]
        for m in modelos:
            linhas.append(f"\\addplot[{m['estilo']}] {coordenadas(m['pontos'])};")
            linhas.append(f"\\addlegendentry{{{m['rotulo']} ({virgula(m['auc'])})}}")
        linhas.append("\\addplot[gray!70, thin, forget plot] coordinates {(0,0) (1,1)};")
        paineis.append("\n".join(linhas))
    corpo = ("\\begin{tikzpicture}\n\\begin{groupplot}[group style={group size=2 by 1, horizontal sep=1.6cm}, "
             f"{ESTILO_EIXOS}, scale only axis, width=0.36\\linewidth, height=0.36\\linewidth, "
             "xmin=0, xmax=1, ymin=0, ymax=1, xlabel={1 $-$ especificidade}, xtick distance=0.2, ytick distance=0.2, "
             "xticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, "
             "yticklabel style={/pgf/number format/fixed, /pgf/number format/precision=1}, "
             "legend style={at={(0.98,0.02)}, anchor=south east, cells={anchor=west}}, legend cell align=left]\n"
             + "\n".join(paineis) + "\n\\end{groupplot}\n\\end{tikzpicture}")
    legenda = ("Curvas ROC por pessoa do D\\_FA e dos baselines clássicos combinados que a H4 compara com ele, sobre a "
               "média das previsões fora do fold nas 10 repetições. Entre parênteses, a AUC por pessoa. A diagonal é o "
               "acaso.")
    return figura(corpo, legenda, "fig:res-roc")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numeros", type=Path, default=NUMEROS / "numeros.json")
    parser.add_argument("--saida", type=Path, default=SAIDA / "fig_roc.tex")
    args = parser.parse_args()
    gravar(tex(dados(json.loads(args.numeros.read_text(encoding="utf-8")))), args.saida)


if __name__ == "__main__":
    main()
