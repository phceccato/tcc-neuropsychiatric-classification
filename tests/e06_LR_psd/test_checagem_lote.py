import numpy as np
import pandas as pd

from pipeline.e06_LR_psd import checagem_lote as c

NOMES = np.array([f"{canal}_{b}" for canal in ("Fp1", "Fp2") for b in c.BANDAS])


def _relativa(potencia):
    """Potência relativa por canal: banda ÷ soma das 4 bandas (como na etapa 04)."""
    p = np.asarray(potencia, dtype=float).reshape(-1, 2, 4)
    return (p / p.sum(axis=2, keepdims=True)).reshape(len(p), -1)


def _janelas(uids):
    return pd.DataFrame({"subject_uid": uids, "sem_overlap": [True] * len(uids), "rejeitada": [False] * len(uids)})


def test_bandas_por_pessoa_agrupa_canais_da_mesma_banda():
    x = np.array([[1, 2, 3, 4, 5, 6, 7, 8], [3, 4, 5, 6, 7, 8, 9, 10]], dtype=float)
    r = c.bandas_por_pessoa(x, NOMES, _janelas(["A", "A"]))
    # média das duas janelas: [2,3,4,5,6,7,8,9]; delta = média de Fp1_delta e Fp2_delta = (2 + 6) / 2
    assert r.loc["A", "delta"] == 4 and r.loc["A", "beta"] == 7


def test_normalizacao_remove_o_ganho_mas_nao_o_ritmo_de_5_hz():
    """O mesmo espectro com ganho 3× tem a mesma potência relativa; um ritmo dominante em teta não."""
    base = np.array([[1.0, 1.0, 3.0, 1.0]] * 2).ravel()
    ganho = 3 * base
    ritmo_teta = np.array([[0.3, 8.0, 1.0, 0.3]] * 2).ravel()
    rel = _relativa([base, ganho, ritmo_teta])
    np.testing.assert_allclose(rel[0], rel[1])
    teta = [i for i, n in enumerate(NOMES) if n.endswith("_teta")]
    assert rel[2, teta].mean() > 3 * rel[0, teta].mean()


def _pessoas():
    return pd.DataFrame({
        "grupo": ["HC", "HC", "HC", "MDD", "MDD"],
        "y": [0, 0, 0, 1, 1],
        "sessao": [1, 2, 2, 3, 3],
    }, index=["a", "b", "c", "d", "e"])


def test_resumo_por_sessao_separa_sessao_e_classe():
    bandas = pd.DataFrame({b: [1.0, 5.0, 7.0, 2.0, 4.0] for b in c.BANDAS}, index=_pessoas().index)
    r = c.resumo_por_sessao(bandas, _pessoas())
    assert r.loc[(2, "HC"), ("teta", "mean")] == 6.0
    assert r.loc[(2, "HC"), ("teta", "count")] == 2
    assert r.loc[(3, "MDD"), ("teta", "mean")] == 3.0


def test_auc_univariada_filtra_e_devolve_nan_sem_as_duas_classes():
    pessoas = _pessoas()
    escore = pd.Series([0.1, 0.9, 0.8, 0.5, 0.6], index=pessoas.index)
    assert abs(c.auc_univariada(pessoas["y"], escore) - 2 / 6) < 1e-12
    assert np.isnan(c.auc_univariada(pessoas["y"], escore, pessoas["sessao"] == 2))  # só HC
    assert c.auc_univariada(pessoas["y"], escore, pessoas["sessao"] != 2) == 1.0


def test_tabela_auc_univariada_da_sessao_de_jan2013_mostra_o_confundimento():
    """Só os HC da sessão 2 têm teta alto: o AUC cai de ~0 para 0,5 quando ela sai."""
    rng = np.random.default_rng(0)
    pessoas = pd.DataFrame({"y": [0] * 6 + [1] * 6, "grupo": ["HC"] * 6 + ["MDD"] * 6,
                            "sessao": [1] * 3 + [2] * 3 + [3] * 6}, index=[f"p{i}" for i in range(12)])
    teta = np.r_[rng.normal(.2, .01, 3), rng.normal(.6, .01, 3), rng.normal(.2, .01, 6)]
    bandas = pd.DataFrame({b: teta for b in c.BANDAS}, index=pessoas.index)
    t = c.tabela_auc_univariada(bandas, pessoas).set_index("banda")
    assert t.loc["teta", "todos"] < 0.2
    assert 0.3 < t.loc["teta", "sem_jan2013"] < 0.7


def test_lr_sem_sessao_nao_preve_as_pessoas_excluidas(monkeypatch):
    chamadas = {}

    def falso_prever(x_treino, x_teste, y, folds):
        chamadas["uids"] = set(folds["subject_uid"]) | set(y.index) | set(x_teste.index)
        return pd.DataFrame(dict(repeticao=0, fold=0, subject_uid=y.index, y=y.to_numpy(),
                                 prob=np.linspace(0, 1, len(y))))

    monkeypatch.setattr(c, "prever_cv", falso_prever)
    monkeypatch.setattr(c, "media_por_pessoa", lambda x, j, so_sem_overlap: pd.DataFrame(
        np.zeros((4, 2)), index=["a", "b", "c", "d"]))
    y = pd.Series([0, 0, 1, 1], index=["a", "b", "c", "d"])
    folds = pd.DataFrame({"subject_uid": ["a", "b", "c", "d"], "repeticao": 0, "fold": 0, "papel": "teste"})
    c.lr_sem_sessao(np.zeros((4, 2)), _janelas(["a"]), y, folds, excluir={"b"})
    assert "b" not in chamadas["uids"] and {"a", "c", "d"} <= chamadas["uids"]


def test_resumo_versionado_da_etapa_06_traz_a_ressalva_sobre_ritmo_entre_sessoes():
    """O resumo copiado para outputs/10_resultados (única pasta no git) tem de dizer o mesmo que o lr.py grava."""
    from pathlib import Path
    import pytest
    raiz = Path(__file__).resolve().parents[2]
    versionado = raiz / "outputs/10_resultados/resumos/06_LR_psd.md"
    if not versionado.exists():
        pytest.skip("outputs/10_resultados não gerado")
    texto = versionado.read_text(encoding="utf-8")
    assert "diferenças de ritmo entre sessões" in texto
    assert "se a absoluta vencer muito só no MDD" not in texto
