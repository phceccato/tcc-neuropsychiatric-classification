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
    assert r["acuracia"] == pytest.approx(3 / 5)                    # acertos: 0,9; 0,6; 0,2
    assert r["precisao"] == pytest.approx(2 / 3)                    # previstos pacientes: 0,9; 0,6; 0,7
    assert r["f1"] == pytest.approx(2 * (2 / 3) * (2 / 3) / (2 / 3 + 2 / 3))


def test_precisao_sem_nenhum_previsto_paciente_e_zero():
    r = m.metricas_secundarias(np.array([1, 0]), np.array([0.1, 0.2]))
    assert r["precisao"] == 0 and r["f1"] == 0 and r["acuracia"] == 0.5


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


def _predicoes_repeticoes():
    linhas = []
    for r, probs in enumerate([[0.9, 0.6, 0.3, 0.2, 0.7], [0.8, 0.7, 0.6, 0.1, 0.2]]):
        for i, (yy, pp) in enumerate(zip([1, 1, 1, 0, 0], probs)):
            linhas.append(dict(repeticao=r, fold=i % 2, subject_uid=f"S{i}", y=yy, prob=pp))
    return pd.DataFrame(linhas)


def test_metricas_por_repeticao_usam_as_previsoes_de_cada_repeticao():
    pred = _predicoes_repeticoes()
    t = m.metricas_por_repeticao(pred)
    assert list(t.index) == [0, 1] and list(t.columns) == list(m.METRICAS_REPETICAO)
    assert t.loc[0, "sensibilidade"] == pytest.approx(2 / 3) and t.loc[1, "sensibilidade"] == pytest.approx(1)
    assert t.loc[1, "auc"] == pytest.approx(1) and t.loc[0, "auc"] == m.roc_auc_score([1, 1, 1, 0, 0], [0.9, 0.6, 0.3, 0.2, 0.7])
    md = m.media_desvio(pred)
    assert md["sensibilidade"][0] == pytest.approx((2 / 3 + 1) / 2)
    assert md["sensibilidade"][1] == pytest.approx(np.std([2 / 3, 1], ddof=1))
    assert md["auc"][0] == pytest.approx(m.auc_por_repeticao(pred).mean())


def test_tabela_media_desvio_tem_uma_coluna_por_metrica_e_marca_incompletos():
    t = m.tabela_media_desvio([(["X"], _predicoes_repeticoes()), (["Y"], None)], ["Modelo"])
    assert t[0] == ("| Modelo | AUC | Acurácia | Acurácia balanceada | Precisão | Sensibilidade (recall) | "
                    "Especificidade | F1 | Macro F1 |")
    assert t[2].startswith("| X | ") and t[2].count("±") == 8
    assert t[3].startswith("| Y | incompleto |")


# ---------- bootstrap das curvas ROC ----------

def test_bootstrap_roc_repete_as_reamostragens_do_ic_bootstrap():
    import numpy as np
    from pipeline.avaliacao import metricas as m
    rng = np.random.default_rng(3)
    y = np.r_[np.ones(9), np.zeros(11)].astype(int)
    escore = rng.normal(size=20) + y
    b = m.bootstrap_roc(y, escore, reamostragens=300, semente=7)
    assert tuple(np.percentile(b["aucs"], [2.5, 97.5])) == pytest.approx(m.ic_bootstrap(y, escore, 300, semente=7), abs=1e-12)
    assert b["tpr"].shape == (300, 101) and (b["tpr"][:, -1] == 1).all()


def test_tpr_na_grade_e_a_curva_em_degraus():
    import numpy as np
    from pipeline.avaliacao import metricas as m
    y, escore = np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])
    t = m.tpr_na_grade(y, escore, np.array([0.0, 0.25, 0.5, 1.0]))
    assert t.tolist() == [0.5, 0.5, 1.0, 1.0]


# ---------- teste de DeLong ----------

def _delong_ingenuo(y, a, b):
    """Implementação independente: componentes estruturais por laços, sem vetorização."""
    import numpy as np
    pos, neg = [i for i in range(len(y)) if y[i] == 1], [i for i in range(len(y)) if y[i] == 0]
    def psi(x, i, j):
        return 1.0 if x[i] > x[j] else 0.5 if x[i] == x[j] else 0.0
    v10 = np.array([[np.mean([psi(x, i, j) for j in neg]) for i in pos] for x in (a, b)])
    v01 = np.array([[np.mean([psi(x, i, j) for i in pos]) for j in neg] for x in (a, b)])
    s = np.cov(v10) / len(pos) + np.cov(v01) / len(neg)
    delta = v10[0].mean() - v10[1].mean()
    return delta, s[0, 0] + s[1, 1] - 2 * s[0, 1]


def test_delong_bate_com_a_implementacao_ingenua_e_com_a_auc_do_sklearn():
    import numpy as np
    from scipy.stats import norm
    rng = np.random.default_rng(1)
    y = np.r_[np.ones(12), np.zeros(15)].astype(int)
    a = np.round(rng.normal(size=27) + 1.0 * y, 1)          # arredondado: há empates
    b = np.round(0.5 * a + rng.normal(size=27) + 0.3 * y, 1)
    r = m.teste_delong(y, a, b)
    delta, variancia = _delong_ingenuo(y, a, b)
    assert r["delta"] == pytest.approx(delta) and r["erro_padrao"] == pytest.approx(np.sqrt(variancia))
    assert r["auc_a"] == pytest.approx(m.roc_auc_score(y, a)) and r["auc_b"] == pytest.approx(m.roc_auc_score(y, b))
    assert r["p"] == pytest.approx(2 * norm.sf(abs(delta / np.sqrt(variancia))))
    assert r["ic"][0] < r["delta"] < r["ic"][1] and not r["degenerado"]


def test_delong_e_simetrico_e_o_p_nao_depende_da_ordem_dos_modelos():
    import numpy as np
    rng = np.random.default_rng(2)
    y = np.r_[np.ones(10), np.zeros(10)].astype(int)
    a, b = rng.normal(size=20) + y, rng.normal(size=20) + 0.4 * y
    ab, ba = m.teste_delong(y, a, b), m.teste_delong(y, b, a)
    assert ab["p"] == pytest.approx(ba["p"]) and ab["delta"] == pytest.approx(-ba["delta"])


def test_delong_com_escores_identicos_ou_auc_1_e_degenerado_e_nao_rejeita():
    import numpy as np
    y = np.r_[np.ones(5), np.zeros(5)].astype(int)
    perfeito = y + 0.1 * np.arange(10)
    for a, b in ((perfeito, perfeito), (perfeito, perfeito * 2 + 1)):
        r = m.teste_delong(y, a, b)
        assert r["degenerado"] and r["p"] == 1.0 and r["z"] is None and r["delta"] == 0.0


def test_delong_exige_dois_pacientes_e_dois_controles():
    import numpy as np
    with pytest.raises(ValueError):
        m.teste_delong(np.array([1, 0, 0]), np.array([0.9, 0.1, 0.2]), np.array([0.8, 0.3, 0.1]))


def test_delong_calibrado_sob_a_nula_com_pessoas_independentes():
    """Dois modelos igualmente bons (mesmo sinal, ruído independente): rejeita perto de 5% a α = 0,05."""
    import numpy as np
    rng = np.random.default_rng(5)
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    rejeicoes = np.mean([m.teste_delong(y, rng.normal(size=80) + 0.8 * y, rng.normal(size=80) + 0.8 * y)["p"] <= 0.05
                         for _ in range(600)])
    assert 0.02 <= rejeicoes <= 0.09


def test_delong_detecta_uma_diferenca_grande():
    import numpy as np
    rng = np.random.default_rng(6)
    y = np.r_[np.ones(40), np.zeros(40)].astype(int)
    r = m.teste_delong(y, rng.normal(size=80) + 2.0 * y, rng.normal(size=80) + 0.2 * y)
    assert r["delta"] > 0.2 and r["p"] < 0.001
