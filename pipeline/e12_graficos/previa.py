"""Prévia em PNG das figuras em pgfplots, para conferir sem abrir o Overleaf.

Para cada .tex de outputs/12_graficos/pgfplots/ (ou os indicados em --figuras), monta um documento mínimo com o
`PREAMBULO` das figuras, largura de texto de 16 cm (a das margens ABNT, como no texto do TCC) e o
pacote `preview`, que recorta a página na figura; compila com pdflatex numa pasta temporária e converte
o PDF em PNG com pdftocairo. Grava outputs/12_graficos/pgfplots/<nome>.png ao lado do .tex.

O PNG é só para conferência: no texto do TCC, a figura continua entrando pelo `\\input` do .tex.
Precisa do pdflatex com pgfplots (pacotes texlive-latex-extra e texlive-pictures) e do pdftocairo (poppler-utils).
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from pipeline.e12_graficos.comum import PREAMBULO, SAIDA

RESOLUCAO = 200  # dpi
LARGURA_TEXTO = "16cm"


def documento(tex: str, largura: str = LARGURA_TEXTO) -> str:
    """Documento de uma figura só, recortado nela.

    O `preview` não recorta um float, então `figure` vira uma minipage da largura do texto que continua
    numerando e legendando como figura (`\\@captype`).
    """
    return ("\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n"
            f"\\usepackage[textwidth={largura}]{{geometry}}\n"
            "\\usepackage[active,tightpage]{preview}\n\\setlength\\PreviewBorder{6pt}\n"
            "\\makeatletter\n\\renewenvironment{figure}[1][]{\\noindent\\begin{minipage}{\\linewidth}\\def\\@captype{figure}}"
            "{\\end{minipage}}\n\\makeatother\n\\PreviewEnvironment{figure}\n"
            "\\renewcommand{\\figurename}{Figura}\n" + PREAMBULO + "\n\\pagestyle{empty}\n"
            "\\begin{document}\n" + tex.rstrip() + "\n\\end{document}\n")


def faltando() -> list[str]:
    return [p for p in ("pdflatex", "pdftocairo") if shutil.which(p) is None]


def gerar_png(fonte: Path, destino: Path, resolucao: int = RESOLUCAO) -> Path:
    """Compila a figura `fonte` (.tex) e grava o PNG em `destino`; levanta RuntimeError se o pdflatex falhar."""
    with tempfile.TemporaryDirectory() as pasta:
        pasta = Path(pasta)
        (pasta / "fig.tex").write_text(documento(fonte.read_text(encoding="utf-8")), encoding="utf-8")
        r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "fig.tex"], cwd=pasta,
                           capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f"{fonte}: pdflatex falhou\n{r.stdout[-3000:]}")
        destino.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["pdftocairo", "-png", "-r", str(resolucao), "-singlefile", str(pasta / "fig.pdf"),
                        str(destino.with_suffix(""))], check=True, capture_output=True)
    return destino


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--figuras", type=Path, nargs="+", default=None,
                        help=f"arquivos .tex; padrão: todos os de {SAIDA}")
    parser.add_argument("--resolucao", type=int, default=RESOLUCAO)
    args = parser.parse_args()
    if ausentes := faltando():
        raise SystemExit(f"faltam {', '.join(ausentes)}: instale texlive-latex-extra, texlive-pictures e poppler-utils")
    figuras = args.figuras or sorted(SAIDA.glob("*.tex"))
    if not figuras:
        raise SystemExit(f"nenhum .tex em {SAIDA}: rode antes h1_nula, roc e sessoes")
    for fonte in figuras:
        print(gerar_png(fonte, fonte.with_suffix(".png"), args.resolucao))


if __name__ == "__main__":
    main()
