import numpy as np
import pandas as pd
import pytest

from pipeline.avaliacao import metricas as m


def _predicoes():
    # 2 repetições, 4 pessoas; P1 e P2 pacientes (y=1)
    return pd.DataFrame({
        "repeticao": [0, 0, 0, 0, 1, 1, 1, 1],
        "subject_uid": ["P1", "P2", "H1", "H2"] * 2,
        "y": [1, 1, 0, 0] * 2,
        "prob": [0.9, 0.4, 0.2, 0.6, 0.7, 0.6, 0.4, 0.2],
    })


def test_escores_por_pessoa_fazem_a_media_das_repeticoes():
    e = m.escores_por_pessoa(_predicoes()).set_index("subject_uid")
    assert e.loc["P1", "prob"] == pytest.approx(0.8)
    assert e.loc["H2", "prob"] == pytest.approx(0.4)
    assert e.loc["P2", "y"] == 1


def test_auc_por_repeticao():
    aucs = m.auc_por_repeticao(_predicoes())
    # rep 0: P1 0,9 e P2 0,4 contra H1 0,2 e H2 0,6 → 3 de 4 pares; rep 1: 4 de 4
    np.testing.assert_allclose(aucs, [0.75, 1.0])


def test_ic_bootstrap_estratificado_contem_a_auc_e_e_reprodutivel():
    rng = np.random.default_rng(0)
    y = np.array([1] * 14 + [0] * 14)
    escore = y * 0.3 + rng.uniform(0, 1, 28)
    baixo, alto = m.ic_bootstrap(y, escore, reamostragens=500, semente=1)
    auc = m.roc_auc_score(y, escore)
    assert 0 <= baixo <= auc <= alto <= 1
    assert (baixo, alto) == m.ic_bootstrap(y, escore, reamostragens=500, semente=1)


def test_metricas_secundarias_no_limiar_de_meio():
    y = np.array([1, 1, 1, 0, 0])
    prob = np.array([0.9, 0.6, 0.3, 0.2, 0.7])
    r = m.metricas_secundarias(y, prob)
    assert r["sensibilidade"] == pytest.approx(2 / 3)
    assert r["especificidade"] == pytest.approx(1 / 2)
    assert r["acuracia_balanceada"] == pytest.approx((2 / 3 + 1 / 2) / 2)
    assert 0 < r["macro_f1"] < 1


def test_p_valor_de_permutacao():
    nulas = np.array([0.4, 0.5, 0.6, 0.7, 0.8])
    assert m.p_valor(0.7, nulas) == pytest.approx((2 + 1) / (5 + 1))
    assert m.p_valor(0.9, nulas) == pytest.approx(1 / 6)


def test_auc_vetorizada_bate_com_o_sklearn():
    rng = np.random.default_rng(0)
    y = np.array([1] * 10 + [0] * 12)
    escores = rng.uniform(0, 1, (5, 22))
    escores[2, :3] = escores[2, 15]  # empates
    esperado = [m.roc_auc_score(y, e) for e in escores]
    np.testing.assert_allclose(m.auc_em_lote(y, escores), esperado)


def test_teste_pareado_detecta_modelo_melhor():
    rng = np.random.default_rng(1)
    y = np.array([1] * 20 + [0] * 20)
    bom = y + rng.normal(0, 0.5, 40)
    ruim = rng.normal(0, 1, 40)
    delta, p = m.teste_pareado(y, bom, ruim, n=2000, semente=0)
    assert delta > 0.3 and p < 0.01


def test_teste_pareado_com_modelos_iguais_nao_rejeita():
    rng = np.random.default_rng(2)
    y = np.array([1] * 20 + [0] * 20)
    a = y + rng.normal(0, 1, 40)
    delta, p = m.teste_pareado(y, a, a.copy(), n=2000, semente=0)
    assert delta == 0 and p == 1.0


def test_ic_bootstrap_do_delta_contem_o_delta_observado():
    rng = np.random.default_rng(3)
    y = np.array([1] * 15 + [0] * 15)
    a = y + rng.normal(0, 0.7, 30)
    b = y + rng.normal(0, 1.5, 30)
    delta = m.roc_auc_score(y, a) - m.roc_auc_score(y, b)
    baixo, alto = m.ic_bootstrap_delta(y, a, b, reamostragens=500, semente=0)
    assert baixo <= delta <= alto
    assert m.ic_bootstrap_delta(y, a, a, reamostragens=100, semente=0) == (0.0, 0.0)


# ---------- preservação: valores gravados antes da extensão do DTSF ----------

def _fixos():
    rng = np.random.default_rng(42)
    y = np.array([1] * 12 + [0] * 12)
    return y, y + rng.normal(0, 1, 24), y * 0.5 + rng.normal(0, 1, 24)


def test_teste_pareado_mantem_o_comportamento_anterior():
    y, a, b = _fixos()
    assert m.teste_pareado(y, a, b, n=500, semente=7) == pytest.approx((0.02777777777777779, 0.4431137724550898))


def test_ic_bootstrap_delta_mantem_o_comportamento_anterior():
    y, a, b = _fixos()
    assert m.ic_bootstrap_delta(y, a, b, reamostragens=300, semente=7) == pytest.approx(
        (-0.30225694444444445, 0.3161458333333332))


# ---------- pareamento por pessoa ----------

def _por_pessoa(uids, y, prob):
    return pd.DataFrame({"subject_uid": uids, "y": y, "prob": prob})


def test_parear_alinha_os_dois_modelos_pela_pessoa():
    a = _por_pessoa(["P1", "P2", "H1"], [1, 1, 0], [0.9, 0.8, 0.1])
    b = _por_pessoa(["H1", "P1", "P2"], [0, 1, 1], [0.3, 0.7, 0.6])
    y, ea, eb = m.parear(a, b)
    np.testing.assert_array_equal(y, [0, 1, 1])  # ordem de subject_uid: H1, P1, P2
    np.testing.assert_allclose(ea, [0.1, 0.9, 0.8])
    np.testing.assert_allclose(eb, [0.3, 0.7, 0.6])


@pytest.mark.parametrize("b", [
    _por_pessoa(["P1", "H1"], [1, 0], [0.7, 0.3]),                    # pessoa ausente
    _por_pessoa(["P1", "P2", "H9"], [1, 1, 0], [0.7, 0.6, 0.3]),      # pessoa diferente
    _por_pessoa(["P1", "P2", "H1"], [1, 0, 0], [0.7, 0.6, 0.3]),      # rótulo diferente
    _por_pessoa(["P1", "P2", "H1"], [1, 1, 0], [0.7, np.nan, 0.3]),   # escore inválido
    _por_pessoa(["P1", "P1", "H1"], [1, 1, 0], [0.7, 0.6, 0.3]),      # pessoa repetida
])
def test_parear_rejeita_entradas_invalidas(b):
    a = _por_pessoa(["P1", "P2", "H1"], [1, 1, 0], [0.9, 0.8, 0.1])
    with pytest.raises(ValueError):
        m.parear(a, b)


def test_parear_exige_as_duas_classes():
    a = _por_pessoa(["P1", "P2"], [1, 1], [0.9, 0.8])
    with pytest.raises(ValueError):
        m.parear(a, a.copy())


# ---------- Holm ----------

def test_holm_rejeita_as_duas_quando_os_dois_passos_passam():
    r = m.holm({"SZ": 0.01, "MDD": 0.04})
    assert r["SZ"] == {"p": 0.01, "p_ajustado": pytest.approx(0.02), "rejeita": True}
    assert r["MDD"]["p_ajustado"] == pytest.approx(0.04) and r["MDD"]["rejeita"]


def test_holm_para_no_primeiro_passo_se_o_menor_p_passa_de_alfa_sobre_2():
    r = m.holm({"SZ": 0.03, "MDD": 0.04})
    assert not r["SZ"]["rejeita"] and not r["MDD"]["rejeita"]
    assert r["SZ"]["p_ajustado"] == pytest.approx(0.06) and r["MDD"]["p_ajustado"] == pytest.approx(0.06)


def test_holm_rejeita_so_a_primeira_quando_a_segunda_passa_de_alfa():
    r = m.holm({"SZ": 0.2, "MDD": 0.001})
    assert r["MDD"]["rejeita"] and not r["SZ"]["rejeita"]
    assert r["SZ"]["p_ajustado"] == pytest.approx(0.2)


def test_holm_limita_o_p_ajustado_em_1():
    r = m.holm({"SZ": 0.7, "MDD": 0.9})
    assert r["SZ"]["p_ajustado"] == 1.0 and r["MDD"]["p_ajustado"] == 1.0


@pytest.mark.parametrize("ps", [{"SZ": 1.2, "MDD": 0.1}, {"SZ": np.nan, "MDD": 0.1}, {}])
def test_holm_rejeita_p_invalido(ps):
    with pytest.raises(ValueError):
        m.holm(ps)


# ---------- ΔAUC por repetição e por fold; t corrigido ----------

def _predicoes_modelo(probs):
    """2 repetições × 2 folds × 2 pessoas (um paciente e um controle por fold)."""
    linhas = []
    for r in range(2):
        for f in range(2):
            for uid, y in ((f"P{f}", 1), (f"H{f}", 0)):
                linhas.append(dict(repeticao=r, fold=f, subject_uid=uid, y=y))
    tabela = pd.DataFrame(linhas)
    tabela["prob"] = probs
    return tabela


def test_delta_por_repeticao_pareia_pessoa_e_repeticao():
    a = _predicoes_modelo([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])  # AUC 1 nas duas
    b = _predicoes_modelo([0.4, 0.6, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])  # rep 0: AUC 3/4; rep 1: 1
    np.testing.assert_allclose(m.delta_por_repeticao(a, b), [0.25, 0.0])
    np.testing.assert_allclose(m.delta_por_repeticao(a, b.sample(frac=1, random_state=0)), [0.25, 0.0])


def test_delta_por_repeticao_rejeita_pessoas_diferentes():
    a = _predicoes_modelo([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
    with pytest.raises(ValueError):
        m.delta_por_repeticao(a, a.iloc[1:])


def test_delta_por_fold_traz_o_tamanho_do_teste():
    a = _predicoes_modelo([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
    b = _predicoes_modelo([0.4, 0.6, 0.8, 0.2, 0.9, 0.1, 0.1, 0.8])
    folds = m.delta_por_fold(a, b)
    assert list(folds["delta"]) == [1.0, 0.0, 0.0, 1.0] and list(folds["n_teste"]) == [2] * 4


def test_t_corrigido_de_nadeau_e_bengio():
    from scipy.stats import t as dist_t
    deltas = np.array([0.1, 0.2, 0.3, 0.0, -0.1])
    media, t, p = m.t_corrigido(deltas, razao_teste_treino=6 / 22)
    esperado = 0.1 / np.sqrt((1 / 5 + 6 / 22) * 0.025)
    assert media == pytest.approx(0.1) and t == pytest.approx(esperado)
    assert p == pytest.approx(1 - dist_t.cdf(esperado, 4))


def test_t_corrigido_com_deltas_constantes():
    assert m.t_corrigido(np.zeros(5), 0.25)[2] == 1.0
    with pytest.raises(ValueError):
        m.t_corrigido(np.array([0.1]), 0.25)


def test_delta_por_fold_recusa_fold_com_uma_classe_so():
    a = _predicoes_modelo([0.9, 0.1, 0.8, 0.2, 0.9, 0.1, 0.8, 0.2])
    a.loc[a["subject_uid"] == "H0", "fold"] = 1
    a.loc[a["subject_uid"] == "P1", "fold"] = 0  # fold 0 só com pacientes, fold 1 só com controles
    with pytest.raises(ValueError):
        m.delta_por_fold(a, a.copy())
