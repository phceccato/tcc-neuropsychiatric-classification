import numpy as np
import pandas as pd
import pytest

from pipeline.e09_investigacao import datas as dt
from pipeline.e09_investigacao import erros as er
from pipeline.e09_investigacao import lote as lt


def test_sessoes_abrem_nova_sessao_depois_da_lacuna():
    datas = pd.Series(["2013-05-01", "2011-07-01", "2011-07-20", "2013-01-12", "2013-01-14"], index=list("abcde"))
    assert dt.sessoes(datas).to_dict() == dict(a=3, b=1, c=1, d=2, e=2)
    assert dt.sessoes(datas, intervalo_dias=1000).nunique() == 1


def test_auc_da_data_e_v_de_cramer():
    y = pd.Series([0, 0, 1, 1])
    assert dt.auc_da_data(y, pd.Series(["2003-01-01", "2003-02-01", "2004-01-01", "2005-01-01"])) == 1.0
    assert dt.auc_da_data(y, pd.Series(["2005-01-01", "2004-01-01", "2003-01-01", "2003-02-01"])) == 0.0
    assert dt.v_cramer(pd.DataFrame([[5, 0], [0, 5]])) == pytest.approx(1.0)
    assert dt.v_cramer(pd.DataFrame([[5, 5], [5, 5]])) == pytest.approx(0.0)


def _manifesto():
    linhas = [("SZ", "SZ-HC-h1", "HC", "2003-06-20"), ("SZ", "SZ-HC-h2", "HC", "2003-07-01"),
              ("SZ", "SZ-SZ-s1", "SZ", "2003-10-20"), ("SZ", "SZ-SZ-s2", "SZ", "2006-08-01"),
              ("MDD", "MDD-HC-S1", "HC", "2011-08-01"), ("MDD", "MDD-HC-S2", "HC", "2013-01-12"),
              ("MDD", "MDD-HC-S3", "HC", "2013-05-01"), ("MDD", "MDD-MDD-S1", "MDD", "2013-04-10"),
              ("MDD", "MDD-MDD-S2", "MDD", "2013-05-20")]
    return pd.DataFrame(linhas, columns=["dataset", "subject_uid", "grupo", "data_gravacao"])


def test_sessao_mista_e_associacao():
    pessoas = dt.pessoas_por_dataset(_manifesto())
    mdd = pessoas["MDD"].set_index("subject_uid")
    assert mdd["sessao_mista"].to_dict() == {"MDD-HC-S1": False, "MDD-HC-S2": False, "MDD-MDD-S1": True,
                                             "MDD-HC-S3": True, "MDD-MDD-S2": True}
    a = dt.associacao(pessoas["MDD"])
    assert (a["hc_em_sessao_mista"], a["pacientes_em_sessao_mista"]) == (1, 2)
    assert dt.associacao(pessoas["SZ"])["hc_em_sessao_mista"] == 0


def test_subconjuntos_separam_por_ano_e_por_sessao():
    subs = lt.subconjuntos(dt.pessoas_por_dataset(_manifesto()))
    assert subs[("SZ", "pacientes de 2003 × HC")] == {"SZ-HC-h1", "SZ-HC-h2", "SZ-SZ-s1"}
    assert subs[("SZ", "pacientes de 2004–2008 × HC")] == {"SZ-HC-h1", "SZ-HC-h2", "SZ-SZ-s2"}
    assert subs[("MDD", "sem o lote de jan/2013")] == {"MDD-HC-S1", "MDD-HC-S3", "MDD-MDD-S1", "MDD-MDD-S2"}
    assert subs[("MDD", "só a sessão de abr–jul/2013")] == {"MDD-HC-S3", "MDD-MDD-S1", "MDD-MDD-S2"}
    assert subs[("MDD", "HC de 01/2013 × pacientes")] == {"MDD-HC-S2", "MDD-MDD-S1", "MDD-MDD-S2"}


def test_folds_sem_lote_excluem_o_lote_e_separam_pessoas():
    manifesto = pd.DataFrame(dict(dataset="MDD", subject_uid=[f"P{i}" for i in range(30)],
                                  grupo=["HC"] * 15 + ["MDD"] * 15))
    folds = lt.folds_sem_lote(manifesto, {"P0", "P1"})
    assert not folds["subject_uid"].isin({"P0", "P1"}).any() and (folds["dataset"] == "MDD").all()
    teste = folds[folds["papel"] == "teste"]
    assert teste.groupby("repeticao")["subject_uid"].apply(lambda s: s.is_unique and len(s) == 28).all()


def test_resumo_espectro_em_espectro_conhecido():
    freqs = np.arange(2, 30.5, 0.5)
    espectro = np.where((freqs >= 8) & (freqs < 13), 3.0, 1.0)
    espectro[freqs == 10.5] = 9.0
    m = er.resumo_espectro(freqs, espectro)
    assert m["pico_hz"] == 10.5
    assert sum(m[f"rel_{b}"] for b in ("delta", "teta", "alfa", "beta")) == pytest.approx(1.0)
    assert m["rel_alfa"] > m["rel_teta"]
