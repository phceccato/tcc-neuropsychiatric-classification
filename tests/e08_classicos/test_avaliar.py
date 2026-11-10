import numpy as np
import pandas as pd
import pytest

from pipeline.e08_classicos import avaliar as ca
from pipeline.e08_classicos.treino import MODELOS
from pipeline.e07_DTSF import avaliar as ma


def _predicoes(semente, ruido, n=16, repeticoes=10, folds=4):
    rng = np.random.default_rng(semente)
    y = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    pessoa = rng.normal(size=n)  # ruído da pessoa: sobrevive à média das repetições
    linhas = []
    for r in range(repeticoes):
        ordem = np.random.default_rng(100 + r).permutation(n // 2)  # fold estratificado: posição dentro da classe
        for i in range(n):
            linhas.append(dict(repeticao=r, fold=int(ordem[i % (n // 2)] % folds), subject_uid=f"S{i:02d}",
                               y=y[i], prob=1 / (1 + np.exp(-(2 * y[i] - 1 + ruido * pessoa[i] + 0.3 * rng.normal())))))
    return pd.DataFrame(linhas)


def test_p_da_intersecao_uniao_e_o_maior_p():
    assert ca.p_interseccao_uniao([0.01, 0.2, 0.04]) == 0.2
    with pytest.raises(ValueError):
        ca.p_interseccao_uniao([])


def test_comparacoes_usam_a_comparacao_pareada_por_ranks():
    d = _predicoes(0, 0.5)
    b = {"LR_temporal": _predicoes(1, 2.0)}
    r = ca.comparacoes(d, b)["LR_temporal"]
    esperado = ma.comparacao_pareada(d, b["LR_temporal"])
    assert r["delta"] == esperado["delta"] and r["p"] == esperado["p"] and r["ic"] == esperado["ic"]
    assert r["delta"] > 0


def test_h4_exige_os_seis_baselines():
    with pytest.raises(ValueError):
        ca.decidir_h4({"SZ": {"LR_temporal": dict(p=0.01)}})


def test_h4_rejeita_so_se_d_vencer_todos_e_passar_no_holm():
    todos = lambda p: {m: dict(p=p) for m in MODELOS}
    um_ruim = {**todos(0.001), "RF_espectral": dict(p=0.3)}
    d = ca.decidir_h4({"SZ": um_ruim, "MDD": todos(0.01)})
    assert d["SZ"]["p"] == 0.3 and not d["SZ"]["rejeita"]
    assert d["MDD"]["p"] == 0.01 and d["MDD"]["p_ajustado"] == pytest.approx(0.02) and d["MDD"]["rejeita"]
    d2 = ca.decidir_h4({"SZ": todos(0.03), "MDD": todos(0.04)})   # 2·0,03 > 0,05: Holm para no 1º passo
    assert not d2["SZ"]["rejeita"] and not d2["MDD"]["rejeita"]


def test_completo_exige_10_repeticoes_de_todas_as_pessoas():
    p = _predicoes(0, 1.0)
    assert ca.completo(p) and not ca.completo(p[p["repeticao"] < 9]) and not ca.completo(p.iloc[1:])
    assert not ca.completo(pd.DataFrame())


def _gravar(pasta, predicoes):
    pasta.mkdir(parents=True, exist_ok=True)
    for (r, f), g in predicoes.groupby(["repeticao", "fold"]):
        g.to_csv(pasta / f"r{r}_f{f}.csv", index=False)
        (pasta / f"r{r}_f{f}.json").write_text('{"parametros": {"C": 0.1}}')


def test_main_grava_so_o_resumo_dos_classicos(tmp_path, monkeypatch):
    classicos, modelos = tmp_path / "classicos", tmp_path / "modelos"
    for k, dataset in enumerate(("SZ", "MDD")):
        _gravar(modelos / dataset / "D", _predicoes(10 + k, 0.3))
        for i, m in enumerate(MODELOS):
            _gravar(classicos / dataset / m, _predicoes(20 + 7 * k + i, 1.5))
    antes = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    monkeypatch.setattr("sys.argv", ["avaliar", "--classicos", str(classicos), "--modelos", str(modelos)])
    ca.main()
    depois = sorted(p.relative_to(tmp_path) for p in tmp_path.rglob("*"))
    assert set(depois) - set(antes) == {classicos.relative_to(tmp_path) / "resumo.md"}
    texto = (classicos / "resumo.md").read_text()
    assert "## H4" in texto and "interseção-união" in texto and texto.count("| SZ | ") >= 6
    assert "**Validade.**" in texto and "não garante" in texto
    assert '{"C": 0.1} ×40' in texto


def test_main_sem_resultados_nao_calcula_a_h4(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.argv", ["avaliar", "--classicos", str(tmp_path / "c"), "--modelos", str(tmp_path / "m")])
    ca.main()
    texto = (tmp_path / "c" / "resumo.md").read_text()
    assert "incompleto" in texto and "## H4" not in texto
