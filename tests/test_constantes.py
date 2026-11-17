"""A semente e as janelas por pessoa vêm de pipeline/constantes.py, sem cópias espalhadas pelas etapas (ME-03)."""
import re
from pathlib import Path

from pipeline import constantes
from pipeline.e03_janelas import janelas as e03
from pipeline.e05_folds import folds as e05
from pipeline.e06_LR_psd import lr as e06
from pipeline.e07_DTSF import avaliar as e07a, rodar as e07r, treino as e07t
from pipeline.e08_classicos import treino as e08t

RAIZ = Path(__file__).resolve().parents[1]


def test_os_valores_do_protocolo():
    assert (constantes.SEMENTE, constantes.JANELAS_POR_PESSOA_DTSF, constantes.JANELAS_POR_PESSOA_CLASSICOS) == (20261108, 60, 150)


def test_cada_etapa_usa_a_constante_compartilhada():
    for modulo in (e05, e06, e07a, e07r):
        assert modulo.SEMENTE is constantes.SEMENTE
    assert e03.JANELAS_POR_PESSOA == e07t.JANELAS_POR_PESSOA == constantes.JANELAS_POR_PESSOA_DTSF
    assert e08t.JANELAS_POR_PESSOA == constantes.JANELAS_POR_PESSOA_CLASSICOS


def test_nenhum_modulo_redefine_a_semente_com_um_literal():
    """Trocar a semente é editar um arquivo só; um `SEMENTE = <número>` fora de constantes.py seria uma cópia esquecida."""
    copias = []
    for caminho in (RAIZ / "pipeline").rglob("*.py"):
        if caminho.name == "constantes.py":
            continue
        for n, linha in enumerate(caminho.read_text(encoding="utf-8").splitlines(), 1):
            if re.match(r"\s*(SEMENTE|JANELAS_POR_PESSOA\w*)\s*=\s*\d", linha) or "20261108" in linha:
                copias.append(f"{caminho.relative_to(RAIZ)}:{n}: {linha.strip()}")
    assert copias == []
