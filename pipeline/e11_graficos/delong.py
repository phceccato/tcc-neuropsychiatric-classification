"""Tabela do teste de DeLong (análise principal): o D_FA contra os outros 5 modelos principais, com Holm.

Cada comparação testa a diferença entre as AUCs de dois modelos nas mesmas pessoas (`teste_delong`, bicaudal, sobre o
escore médio das 10 repetições). Os 5 p brutos de cada transtorno são ajustados por Holm (α = 0,05), e a coluna
"Significativo" é a decisão do Holm. As comparações (`COMPARADOS`) são o D (dimensões do artigo, sem FA), o
D_pequeno e os 3 baselines combinados (LR, RF e KNN).

Lê outputs/11_graficos/numeros.json (`numeros.py`, chave `delong`) e grava outputs/11_graficos/delong.tex, uma tabela
em LaTeX (booktabs) para o Overleaf. O preâmbulo precisa de `\\usepackage{booktabs}`. A mesma tabela, em texto, sai
na figura `p10_delong_tabela` (graficos.py) e em delong.csv (numeros.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PRINCIPAL = "D_FA"
COMPARADOS = ("D", "D_pequeno", "LR_combinado", "RF_combinado", "KNN_combinado")
ROTULOS = {"D": "D (sem FA)", "D_pequeno": "D_pequeno", "LR_combinado": "LR combinado",
           "RF_combinado": "RF combinado", "KNN_combinado": "KNN combinado"}
CABECALHO = ["Comparação", "AUC D_FA", "AUC do comparado", "ΔAUC (D_FA − outro)", "Z de DeLong", "p bruto",
             "p ajustado (Holm)", "Significativo"]
TITULOS = {"SZ": "Esquizofrenia × controles", "MDD": "Depressão maior × controles"}
SAIDA = Path("outputs/11_graficos")
ALFA = 0.05


def virgula(x: float, casas: int = 4, sinal: bool = False) -> str:
    texto = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return texto.replace(".", ",").replace("-", "−")


def formatar_p(p: float) -> str:
    return "< 0,001" if p < 0.001 else virgula(p)


def linhas(numeros: dict, d: str) -> list[list[str]]:
    """Uma linha de texto por comparação do transtorno `d`, na ordem de `COMPARADOS`."""
    saida = []
    for m in COMPARADOS:
        r = numeros["delong"][d][f"{PRINCIPAL} − {m}"]
        saida.append([f"{PRINCIPAL} vs. {ROTULOS[m]}", virgula(r["auc_a"]), virgula(r["auc_b"]),
                      virgula(r["delta"], sinal=True), "—" if r["z"] is None else virgula(r["z"], 4, True),
                      formatar_p(r["p"]), formatar_p(r["p_holm"]), "Sim" if r["rejeita"] else "Não"])
    return saida


def _tex(celula: str) -> str:
    """Texto da célula para LaTeX: sublinhado escapado, menos tipográfico e símbolos em modo matemático."""
    return (celula.replace("_", "\\_").replace("Δ", "$\\Delta$").replace("−", "$-$").replace("×", "$\\times$")
            .replace("< 0,001", "$<$ 0,001"))


def latex(numeros: dict) -> str:
    colunas = "l" + "r" * 6 + "c"
    cabecalho = " & ".join(_tex(c) for c in CABECALHO)
    corpo = []
    for d in TITULOS:
        corpo.append(f"\\midrule\n\\multicolumn{{{len(CABECALHO)}}}{{l}}{{\\textit{{{_tex(TITULOS[d])}}}}} \\\\")
        corpo += [" & ".join(_tex(c) for c in linha) + " \\\\" for linha in linhas(numeros, d)]
    return ("\\begin{table}[htbp]\n\\centering\n\\footnotesize\n"
            f"\\begin{{tabular}}{{{colunas}}}\n\\toprule\n{cabecalho} \\\\\n" + "\n".join(corpo) + "\n\\bottomrule\n"
            "\\end{tabular}\n\\caption{Teste de DeLong para a diferença entre as AUCs por pessoa do D\\_FA e dos outros "
            "modelos principais (bicaudal). O p bruto de cada transtorno é ajustado por Holm sobre as cinco "
            "comparações; significativo se o p ajustado é no máximo 0,05. A validação cruzada viola a "
            "independência das pessoas que o DeLong supõe, e as AUCs são as dos escores médios das repetições, então os "
            "p são aproximados.}\n\\label{tab:res-delong}\n\\end{table}\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numeros", type=Path, default=SAIDA / "numeros.json")
    parser.add_argument("--saida", type=Path, default=SAIDA / "delong.tex")
    args = parser.parse_args()
    numeros = json.loads(args.numeros.read_text(encoding="utf-8"))
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(latex(numeros), encoding="utf-8")
    print(f"{args.saida} ({args.saida.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
