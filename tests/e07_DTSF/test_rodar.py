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


# ---------- configuração gravada no .json ----------

def test_configuracao_tarefa_registra_dispositivo_variante_e_hiperparametros():
    c = r.configuracao_tarefa("D_sem_mixup", 20261108, "cuda")
    assert c["modelo"] == "D_sem_mixup" and c["dispositivo"] == "cuda" and c["semente"] == 20261108
    assert c["variante"]["mixup"] is False and c["variante"]["filtros"] == 16 and c["variante"]["oculto"] == 32
    assert c["hiperparametros"]["lr"] == 1e-3 and c["hiperparametros"]["paciencia"] == 10
    assert c["janelas_por_pessoa"] == 60 and "torch" in c and "cuda" in c
    import json
    json.dumps(c)  # serializável
    assert r.configuracao_tarefa("D_FA", 1, "cpu")["variante"]["fa"] is True


# ---------- checkpoints do modelo principal ----------

def test_checkpoint_so_do_principal_por_padrao():
    assert r.modelos_com_checkpoint("principal", list(r.TIPOS_DTSF)) == {"D_FA"}
    assert r.modelos_com_checkpoint("principal", ["D_pequeno"]) == set()
    assert r.modelos_com_checkpoint("todos", ["D_FA", "D"]) == {"D_FA", "D"}
    assert r.modelos_com_checkpoint("nenhum", list(r.TIPOS_DTSF)) == set()


def _tarefa_sintetica(monkeypatch, tmp_path, modelo, permutacao=None):
    import json
    from pipeline.e07_DTSF import treino as t
    from tests.e07_DTSF.test_dtsf import _FonteSintetica, _papeis
    fonte = _FonteSintetica()
    treino, validacao, teste = _papeis(fonte)
    linhas = [(u, "treino") for u in treino] + [(u, "validacao") for u in validacao] + [(u, "teste") for u in teste]
    pd.DataFrame([dict(dataset="SZ", repeticao=0, fold=0, subject_uid=u, grupo="SZ" if u[0] == "P" else "HC", papel=p)
                  for u, p in linhas]).to_csv(tmp_path / "folds.csv", index=False)
    monkeypatch.setattr(r, "_fonte", lambda dataset, folds: fonte)
    monkeypatch.setattr(r, "treinar", lambda *a, **k: t.treinar(*a, max_epocas=2, **k))
    r.rodar_tarefa("SZ", modelo, 0, 0, str(tmp_path / "folds.csv"), str(tmp_path / "res"), permutacao,
                   pasta_checkpoints=str(tmp_path / "ck"))
    base = "r0_f0" if permutacao is None else f"p{permutacao}_f0"
    return json.load(open(tmp_path / "res" / "SZ" / modelo / f"{base}.json")), base


def test_tarefa_grava_checkpoint_e_historico_completo(monkeypatch, tmp_path):
    from pipeline.e07_DTSF import treino as t
    j, base = _tarefa_sintetica(monkeypatch, tmp_path, "D_FA")
    ck = tmp_path / "ck" / "SZ" / "D_FA" / "r0_f0.pt"
    assert ck.exists() and not list(tmp_path.rglob("*.tmp"))
    assert j["epocas_executadas"] == len(j["historico"]) and j["dispositivo"] == "cpu" and j["variante"]["fa"]
    assert j["perda_validacao_melhor"] == j["historico"][j["melhor_epoca"] - 1]["perda_validacao"]
    modelo, meta = t.carregar_checkpoint(ck)
    assert meta["melhor_epoca"] == j["melhor_epoca"] and meta["configuracao"]["semente"] == j["semente"]
    assert sum(p.numel() for p in modelo.parameters()) == j["parametros"] == 704_769


def test_permutacao_nao_grava_checkpoint(monkeypatch, tmp_path):
    j, _ = _tarefa_sintetica(monkeypatch, tmp_path, "D_FA", permutacao=0)
    assert len(j["historico"]) >= 1 and not (tmp_path / "ck").exists()
