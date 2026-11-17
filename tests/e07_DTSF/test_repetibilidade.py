import pandas as pd
import pytest

from pipeline.e07_DTSF import repetibilidade as r


def _pred(probs):
    return pd.DataFrame(dict(repeticao=0, fold=0, subject_uid=["A", "B", "C"], y=[0, 1, 1], prob=probs))


def test_comparar_execucoes_identicas():
    c = r.comparar(_pred([0.1, 0.6, 0.9]), _pred([0.1, 0.6, 0.9]))
    assert c == dict(pessoas=3, max_abs=0.0, media_abs=0.0, identicas=True)


def test_comparar_mede_a_maior_diferenca_e_ignora_a_ordem_das_linhas():
    b = _pred([0.1, 0.6, 0.9]).iloc[::-1]
    c = r.comparar(_pred([0.1, 0.6, 0.9 + 3e-4]), b)
    assert c["max_abs"] == pytest.approx(3e-4) and not c["identicas"]
    assert c["media_abs"] == pytest.approx(1e-4)


def test_comparar_recusa_pessoas_diferentes():
    b = _pred([0.1, 0.6, 0.9]).assign(subject_uid=["A", "B", "Z"])
    with pytest.raises(ValueError):
        r.comparar(_pred([0.1, 0.6, 0.9]), b)
