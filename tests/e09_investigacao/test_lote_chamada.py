"""A etapa 09 chama `rodar_tarefa` da etapa 07 com a assinatura atual (AL-01).

A chamada já passou os argumentos na ordem antiga e só quebrou no retreino, sem teste que a cobrisse
(`torch.device(None)` levantava TypeError). Aqui o executor é falso e confere cada chamada contra a assinatura real.
"""
import inspect
from concurrent.futures import Future
from pathlib import Path

import pytest

from pipeline.e07_DTSF import rodar
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL
from pipeline.e09_investigacao import lote


class ExecutorFalso:
    def __init__(self):
        self.chamadas = []

    def submit(self, fn, *args, **kwargs):
        self.chamadas.append((fn, args, kwargs))
        f = Future()
        f.set_result("ok")
        return f


def test_retreino_chama_rodar_tarefa_com_a_assinatura_atual():
    ex = ExecutorFalso()
    futuros = lote.submeter_retreinos(ex, [(0, 0), (3, 4)], Path("folds.csv"), Path("saida"), "cuda")
    assert len(futuros) == len(ex.chamadas) == 2
    for fn, args, kwargs in ex.chamadas:
        assert fn is rodar.rodar_tarefa
        ligados = inspect.signature(rodar.rodar_tarefa).bind(*args, **kwargs)     # TypeError se a ordem ou o nome erra
        ligados.apply_defaults()
        a = ligados.arguments
        assert a["dataset"] == "MDD" and a["modelo"] == PRINCIPAL
        assert a["folds_csv"] == "folds.csv" and a["saida"] == "saida"
        assert a["dispositivo"] == "cuda"
        assert isinstance(a["repeticao"], int) and isinstance(a["fold"], int)
    assert [c[1][2:4] for c in ex.chamadas] == [(0, 0), (3, 4)]


def test_o_dispositivo_vai_por_nome_e_nunca_como_posicional():
    ex = ExecutorFalso()
    lote.submeter_retreinos(ex, [(1, 2)], Path("f.csv"), Path("s"), "cpu")
    _, args, kwargs = ex.chamadas[0]
    assert kwargs == dict(dispositivo="cpu") and len(args) == 6


def test_assinatura_da_etapa_07_que_o_lote_espera():
    nomes = list(inspect.signature(rodar.rodar_tarefa).parameters)
    assert nomes[:6] == ["dataset", "modelo", "repeticao", "fold", "folds_csv", "saida"]
    assert "dispositivo" in nomes
