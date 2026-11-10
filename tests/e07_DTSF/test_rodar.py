import pandas as pd

from pipeline.e07_DTSF import rodar as r


def _rotulos():
    return pd.Series([1] * 14 + [0] * 14, index=[f"S{i}" for i in range(28)])


def test_rotulos_permutados_mantem_as_contagens_e_sao_reprodutiveis():
    y = _rotulos()
    a, b = r.rotulos_permutados(y, 3), r.rotulos_permutados(y, 3)
    pd.testing.assert_series_equal(a, b)
    assert a.sum() == 14 and list(a.index) == list(y.index)
    assert not a.equals(y)


def test_permutacoes_diferentes_embaralham_diferente():
    y = _rotulos()
    assert not r.rotulos_permutados(y, 0).equals(r.rotulos_permutados(y, 1))


def test_runner_so_aceita_o_dtsf_e_as_ablacoes(monkeypatch):
    import pytest
    monkeypatch.setattr("sys.argv", ["rodar", "--modelos", "F"])
    with pytest.raises(SystemExit):
        r.main()
