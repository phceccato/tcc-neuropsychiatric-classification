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


# ---------- dispositivo (CPU ou GPU) ----------

def test_dispositivos_gravados_le_os_json_e_trata_os_antigos_como_cpu(tmp_path):
    import json
    pasta = tmp_path / "SZ" / "D"
    pasta.mkdir(parents=True)
    json.dump(dict(melhor_epoca=1), open(pasta / "r0_f0.json", "w"))
    json.dump(dict(melhor_epoca=1, dispositivo="cuda"), open(pasta / "r0_f1.json", "w"))
    assert r.dispositivos_gravados([pasta]) == {"cpu", "cuda"}
    assert r.dispositivos_gravados([tmp_path / "vazia"]) == set()


def test_runner_nao_mistura_cpu_e_gpu_na_mesma_pasta(tmp_path, monkeypatch):
    import json

    import pytest
    pasta = tmp_path / "SZ" / "D"
    pasta.mkdir(parents=True)
    json.dump(dict(melhor_epoca=1), open(pasta / "r0_f0.json", "w"))
    monkeypatch.setattr("torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("sys.argv", ["rodar", "--saida", str(tmp_path), "--datasets", "SZ", "--dispositivo", "cuda"])
    with pytest.raises(SystemExit):
        r.main()


def test_runner_recusa_cuda_sem_gpu(monkeypatch):
    import pytest
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["rodar", "--dispositivo", "cuda"])
    with pytest.raises(SystemExit):
        r.main()


def test_por_padrao_roda_os_seis_modelos_e_as_permutacoes_so_do_principal(tmp_path, monkeypatch):
    from concurrent.futures import Future
    pedidos = []

    class _Executor:                       # não treina: só registra os pedidos
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def submit(self, f, d, m, *resto):
            pedidos.append(m)
            futuro = Future()
            futuro.set_result("ok")
            return futuro

    monkeypatch.setattr(r, "ProcessPoolExecutor", _Executor)
    monkeypatch.setattr("sys.argv", ["rodar", "--saida", str(tmp_path), "--datasets", "SZ", "--repeticoes", "0"])
    r.main()
    assert pedidos == [m for m in r.TIPOS_DTSF for _ in range(5)]
    pedidos.clear()
    monkeypatch.setattr("sys.argv", ["rodar", "--saida", str(tmp_path / "p"), "--datasets", "SZ", "--permutacoes", "1"])
    r.main()
    assert pedidos == ["D_FA"] * 5
