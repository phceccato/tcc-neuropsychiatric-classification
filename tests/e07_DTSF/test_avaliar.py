import inspect

import numpy as np
import pandas as pd
import pytest
from scipy.stats import rankdata

from pipeline.avaliacao import metricas as m
from pipeline.e07_DTSF import avaliar as a
from pipeline.e07_DTSF.redes_dtsf import TIPOS_DTSF


def test_p_acaso_usa_so_permutacoes_completas(tmp_path):
    y = [1, 1, 0, 0]
    real = pd.DataFrame(dict(repeticao=0, subject_uid=list("ABCD"), y=y, prob=[0.9, 0.8, 0.2, 0.1]))
    for k, probs in enumerate([[0.1, 0.2, 0.8, 0.9], [0.9, 0.8, 0.2, 0.1]]):
        for f in range(2):
            pd.DataFrame(dict(subject_uid=list("ABCD")[2 * f:2 * f + 2], y=y[2 * f:2 * f + 2],
                              prob=probs[2 * f:2 * f + 2])).to_csv(tmp_path / f"p{k}_f{f}.csv", index=False)
    pd.DataFrame(dict(subject_uid=["A"], y=[1], prob=[0.5])).to_csv(tmp_path / "p2_f0.csv", index=False)  # incompleta
    p, n = a.p_acaso(real, tmp_path, folds=2)
    assert n == 2 and p == (1 + 1) / (2 + 1)


def test_ablacoes_sao_as_variantes_do_dtsf():
    assert a.ABLACOES == ("D_sem_T", "D_sem_E", "D_sem_mixup")


# ---------- teste pareado com ranks (a contra b) ----------

def _tabela(uids, y, prob):
    return pd.DataFrame({"subject_uid": uids, "y": y, "prob": prob})


def _dados(semente=0, ruido_a=0.5, ruido_b=1.5, n=20):
    rng = np.random.default_rng(semente)
    y = np.array([1] * n + [0] * n)
    uids = [f"S{i:02d}" for i in range(2 * n)]
    return _tabela(uids, y, y + rng.normal(0, ruido_a, 2 * n)), _tabela(uids, y, y + rng.normal(0, ruido_b, 2 * n))


def test_teste_ranks_e_a_permutacao_pareada_sobre_os_ranks_de_cada_modelo():
    x, z = _dados()
    r = a.teste_ranks(x, z, n=2000, semente=3)
    esperado = m.teste_pareado(x["y"].to_numpy(), rankdata(x["prob"], method="average"),
                               rankdata(z["prob"], method="average"), n=2000, semente=3)
    assert (r["delta"], r["p"]) == esperado


def test_teste_ranks_usa_ranks_medios_nos_empates():
    y = np.array([1, 1, 1, 0, 0, 0])
    x = _tabela(list("ABCDEF"), y, [0.9, 0.5, 0.5, 0.5, 0.1, 0.1])
    z = _tabela(list("ABCDEF"), y, [0.2, 0.7, 0.7, 0.3, 0.3, 0.9])
    r = a.teste_ranks(x, z, n=500, semente=0)
    rx = rankdata(x["prob"], method="average")
    np.testing.assert_allclose(rx, [6, 4, 4, 4, 1.5, 1.5])
    assert r["p"] == m.teste_pareado(y, rx, rankdata(z["prob"], method="average"), n=500, semente=0)[1]


def test_delta_com_ranks_igual_ao_delta_com_escores():
    x, z = _dados(semente=4)
    r = a.teste_ranks(x, z, n=100, semente=0)
    assert r["delta"] == pytest.approx(m.roc_auc_score(x["y"], x["prob"]) - m.roc_auc_score(z["y"], z["prob"]))


def test_teste_ranks_e_unilateral_a_maior_que_b():
    x, z = _dados(semente=1)
    assert a.teste_ranks(x, z, n=2000, semente=0)["p"] < 0.01
    assert a.teste_ranks(z, x, n=2000, semente=0)["p"] > 0.99  # mesma diferença na direção contrária


def test_teste_ranks_e_reprodutivel_com_a_mesma_semente():
    x, z = _dados(semente=2, ruido_a=1.0, ruido_b=1.2)
    assert a.teste_ranks(x, z, n=1000, semente=5) == a.teste_ranks(x, z, n=1000, semente=5)


def test_teste_ranks_usa_10_mil_permutacoes_por_padrao():
    assert inspect.signature(a.teste_ranks).parameters["n"].default == 10_000


def test_teste_ranks_pareia_pela_pessoa_e_nao_pela_ordem_das_linhas():
    x, z = _dados(semente=3)
    embaralhado = z.sample(frac=1, random_state=1).reset_index(drop=True)
    assert a.teste_ranks(x, z, n=500, semente=0) == a.teste_ranks(x, embaralhado, n=500, semente=0)


def test_teste_ranks_rejeita_pessoa_ausente():
    x, z = _dados()
    with pytest.raises(ValueError):
        a.teste_ranks(x, z.iloc[1:], n=100, semente=0)


def test_teste_ranks_traz_o_ic_bootstrap_pareado_condicional():
    x, z = _dados(semente=6)
    r = a.teste_ranks(x, z, n=100, semente=0)
    assert r["ic"][0] <= r["delta"] <= r["ic"][1]


# ---------- comparação pareada: teste primário e sensibilidades ----------

def _predicoes_cv(semente, ruido, n=16, repeticoes=3, folds=4):
    """Previsões fora do fold sintéticas: cada pessoa em um fold de teste por repetição."""
    rng = np.random.default_rng(semente)
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    linhas = []
    for r in range(repeticoes):
        g = np.random.default_rng(100 + r)  # estratificado: cada fold com pacientes e controles
        fold_de = np.r_[g.permutation(np.arange(n // 2) % folds), g.permutation(np.arange(n // 2) % folds)]
        for i in range(n):
            linhas.append(dict(repeticao=r, fold=int(fold_de[i]), subject_uid=f"S{i:02d}", y=int(y[i]),
                               prob=float(1 / (1 + np.exp(-(2 * y[i] - 1 + rng.normal(0, ruido)))))))
    return pd.DataFrame(linhas)


def test_comparacao_pareada_traz_o_primario_e_as_sensibilidades():
    x, z = _predicoes_cv(0, 0.5), _predicoes_cv(1, 2.0)
    r = a.comparacao_pareada(x, z, n=500, semente=0)
    primario = a.teste_ranks(m.escores_por_pessoa(x), m.escores_por_pessoa(z), n=500, semente=0)
    assert (r["delta"], r["p"], r["ic"]) == (primario["delta"], primario["p"], primario["ic"])
    assert len(r["delta_repeticoes"]) == 3 and len(r["delta_folds"]) == 12
    assert "delong_p" not in r
    assert r["razao_teste_treino"] == pytest.approx(4 / 12)
    assert 0 < r["t_corrigido_p"] <= 1


def test_hipoteses_h2_e_h3_comparam_o_d_com_as_ablacoes_da_proposta():
    assert a.HIPOTESES == {"H2": ("D_sem_T", "D_sem_E"), "H3": ("D_sem_mixup",)}
    assert a.REFERENCIA_ABLACOES == "D_pequeno" and set(a.ABLACOES) == {"D_sem_T", "D_sem_E", "D_sem_mixup"}
    assert a.DESCRITIVAS == (("D_FA", "D"), ("D", "D_pequeno"), ("D_FA", "D_pequeno"))


def test_comparacoes_da_hipotese_sao_d_menos_ablacao():
    d = _predicoes_cv(0, 0.5)
    ablacao = d.assign(prob=1 - d["prob"])                        # ablação anti-preditiva: AUC(D) > AUC(ablação)
    r = a.comparacoes_hipotese({"D_pequeno": d, "D_sem_T": ablacao}, ("D_sem_T",), n=500)
    esperado = a.comparacao_pareada(d, ablacao, n=500)
    assert r["D_sem_T"]["delta"] == esperado["delta"] > 0
    assert r["D_sem_T"]["p"] == esperado["p"]


def _p(**ps):
    return {abl: dict(p=p) for abl, p in ps.items()}


def test_decidir_hipotese_usa_interseccao_uniao_e_holm_entre_os_transtornos():
    d = a.decidir_hipotese({"SZ": _p(D_sem_T=0.01, D_sem_E=0.2), "MDD": _p(D_sem_T=0.01, D_sem_E=0.01)},
                           ("D_sem_T", "D_sem_E"))
    assert d["SZ"]["p"] == 0.2 and d["MDD"]["p"] == 0.01         # interseção-união: maior p
    assert d["MDD"]["p_ajustado"] == pytest.approx(0.02) and d["MDD"]["rejeita"]
    assert d["SZ"]["p_ajustado"] == pytest.approx(0.2) and not d["SZ"]["rejeita"]
    d1 = a.decidir_hipotese({"SZ": _p(D_sem_T=0.03), "MDD": _p(D_sem_T=0.04)}, ("D_sem_T",))
    assert not d1["SZ"]["rejeita"] and not d1["MDD"]["rejeita"]   # 2·0,03 > 0,05: Holm para no 1º passo


def test_decidir_hipotese_exige_todas_as_ablacoes_nos_dois_transtornos():
    with pytest.raises(ValueError):
        a.decidir_hipotese({"SZ": _p(D_sem_T=0.01), "MDD": _p(D_sem_T=0.01, D_sem_E=0.01)}, ("D_sem_T", "D_sem_E"))
    with pytest.raises(ValueError):
        a.decidir_hipotese({"SZ": _p(D_sem_T=0.01)}, ("D_sem_T",))


def test_h1_e_decidida_em_cada_transtorno_sem_holm():
    d = a.decidir_h1({"SZ": (1 / 26, 25), "MDD": (1 / 26, 25)})
    assert d["SZ"]["rejeita"] and d["MDD"]["rejeita"]               # com Holm, 2/26 > 0,05 não rejeitaria
    assert not a.decidir_h1({"SZ": (3 / 26, 25)})["SZ"]["rejeita"]


def test_h1_exige_as_25_permutacoes():
    with pytest.raises(ValueError):
        a.decidir_h1({"SZ": (1 / 11, 10)})


def test_main_escreve_o_resumo_do_dtsf_sem_tocar_no_resumo_historico(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    uids, y = [f"S{i}" for i in range(8)], [1, 1, 1, 1, 0, 0, 0, 0]
    for dataset in ("SZ", "MDD"):
        for modelo in TIPOS_DTSF:
            (tmp_path / "modelos" / dataset / modelo).mkdir(parents=True)
        for r in range(10):
            for fold in (0, 1):
                idx = [fold, fold + 2, fold + 4, fold + 6]  # 2 pacientes e 2 controles por fold
                for modelo in TIPOS_DTSF:
                    pd.DataFrame(dict(repeticao=r, fold=fold, subject_uid=[uids[i] for i in idx], y=[y[i] for i in idx],
                                      prob=rng.uniform(0, 1, 4))).to_csv(
                        tmp_path / "modelos" / dataset / modelo / f"r{r}_f{fold}.csv", index=False)
    (tmp_path / "modelos" / "resumo.md").write_text("histórico")
    monkeypatch.setattr("sys.argv", ["avaliar", "--modelos", str(tmp_path / "modelos"), "--permutacoes", str(tmp_path / "perm")])
    a.main()
    resumo = (tmp_path / "modelos" / "resumo_dtsf.md").read_text()
    assert "| D | incompleto" not in resumo and "D_pequeno − D_sem_mixup" in resumo
    assert "## H2" in resumo and "## H3: Mixup" in resumo and "D_pequeno − D_sem_T" in resumo and "D_sem_fusao" not in resumo
    assert "| D_FA − D |" in resumo and "| D − D_pequeno |" in resumo and "| D_FA − D_pequeno |" in resumo
    assert "| D_FA | 704.769 |" in resumo and "| D_pequeno | 80.449 |" in resumo
    assert "## H1: D_FA acima do acaso" in resumo and "incompleta" in resumo
    assert "Média ± desvio nas 10 repetições" in resumo and resumo.count("| D_FA | 0.") >= 2  # duas tabelas por dataset            # sem permutações gravadas, a H1 não é decidida
    assert (tmp_path / "modelos" / "resumo.md").read_text() == "histórico"
