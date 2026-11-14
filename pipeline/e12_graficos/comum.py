"""Peças comuns das figuras em pgfplots: ambiente `figure`, conferência dos números e checagem do .tex.

Cada figura é um .tex com só o ambiente `figure` (dados embutidos, sem imagem), para dar `\\input` no
Overleaf; `previa.py` gera um PNG de conferência de cada uma. O preâmbulo do documento precisa de `PREAMBULO`. Os números do eixo saem com vírgula decimal
(`/pgf/number format/use comma`); os números escritos no texto (legendas, caption) usam `virgula`.
"""
from __future__ import annotations

import re
from pathlib import Path

SAIDA = Path("outputs/12_graficos/pgfplots")
PREAMBULO = "\\usepackage{pgfplots}\n\\usepgfplotslibrary{groupplots}\n\\pgfplotsset{compat=1.18}"
ESTILO_EIXOS = ("/pgf/number format/use comma, /pgf/number format/1000 sep={.}, "
                "tick label style={font=\\footnotesize}, label style={font=\\footnotesize}, "
                "legend style={font=\\footnotesize}, title style={font=\\footnotesize}")
TOLERANCIA = 5e-4  # "bate na terceira casa": |calculado − doc| ≤ meia unidade da 3ª casa


class Divergencia(ValueError):
    """Um número calculado não bate com o de referência."""


def conferir(nome: str, calculado: float, esperado: float) -> None:
    if abs(calculado - esperado) > TOLERANCIA:
        raise Divergencia(f"{nome}: calculado {calculado:.4f}, referência {esperado:.3f} (diferença {calculado - esperado:+.4f})")


def virgula(x: float, casas: int = 3) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def coordenadas(pontos) -> str:
    """`coordinates {...}` com ponto decimal (o pgfplots lê ponto; a vírgula é só na exibição)."""
    return "coordinates {" + " ".join(f"({x:.10g},{y:.6g})" for x, y in pontos) + "}"


def figura(corpo: str, legenda: str, rotulo: str) -> str:
    return ("\\begin{figure}[htbp]\n\\centering\n\\footnotesize\n" + corpo.rstrip() + "\n"
            f"\\caption{{{legenda}}}\n\\label{{{rotulo}}}\n\\end{{figure}}\n")


def checar_tex(texto: str) -> None:
    """Sintaxe básica: chaves e ambientes balanceados, na ordem; levanta ValueError se não."""
    sem_escapes = re.sub(r"\\[{}%]", "", texto)
    nivel = 0
    for c in sem_escapes:
        nivel += {"{": 1, "}": -1}.get(c, 0)
        if nivel < 0:
            raise ValueError("chave fechada sem abrir")
    if nivel:
        raise ValueError(f"{nivel} chave(s) sem fechar")
    pilha = []
    for tipo, nome in re.findall(r"\\(begin|end)\{([^}]*)\}", sem_escapes):
        if tipo == "begin":
            pilha.append(nome)
        elif not pilha or pilha.pop() != nome:
            raise ValueError(f"\\end{{{nome}}} sem o \\begin correspondente")
    if pilha:
        raise ValueError(f"ambientes sem fechar: {pilha}")


def gravar(texto: str, destino: Path) -> Path:
    checar_tex(texto)
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(texto, encoding="utf-8")
    print(f"{destino} ({destino.stat().st_size} bytes)")
    return destino


def documento_minimo(*tex: str) -> str:
    """Documento mínimo para compilar as figuras (teste com pdflatex)."""
    return ("\\documentclass{article}\n\\usepackage[T1]{fontenc}\n\\usepackage[utf8]{inputenc}\n"
            + PREAMBULO + "\n\\begin{document}\n" + "\n".join(tex) + "\\end{document}\n")
