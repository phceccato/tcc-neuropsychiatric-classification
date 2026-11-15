import numpy as np
import pandas as pd
import pytest

from pipeline.avaliacao import metricas as m
from pipeline.e07_DTSF import avaliar as a
from pipeline.e07_DTSF.redes_dtsf import TIPOS_DTSF


def test_ablacoes_sao_as_variantes_do_dtsf():
    assert a.ABLACOES == ("D_sem_T", "D_sem_E", "D_sem_mixup")
    assert a.REFERENCIA_ABLACOES == "D_pequeno"
    assert a.DESCRITIVAS == (("D_FA", "D"), ("D", "D_pequeno"), ("D_FA", "D_pequeno"))


def test_nao_ha_hipoteses_nem_teste_de_permutacao_na_avaliacao():
    for nome in ("HIPOTESES", "decidir_h1", "decidir_hipotese", "p_acaso", "teste_ranks", "PERMUTACOES_H1"):
        assert not hasattr(a, nome)


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


def test_comparacao_pareada_traz_delta_ic_e_delta_por_repeticao():
    x, z = _predicoes_cv(0, 0.5), _predicoes_cv(1, 2.0)
    r = a.comparacao_pareada(x, z)
    assert set(r) == {"delta", "ic", "delta_repeticoes"}
    esperado = m.roc_auc_score(*_por_pessoa(x)) - m.roc_auc_score(*_por_pessoa(z))
    assert r["delta"] == pytest.approx(esperado)
    assert r["ic"][0] <= r["delta"] <= r["ic"][1]
    assert len(r["delta_repeticoes"]) == 3


def _por_pessoa(p):
    e = m.escores_por_pessoa(p)
    return e["y"], e["prob"]


def test_comparacao_pareada_e_antissimetrica_no_delta():
    x, z = _predicoes_cv(0, 0.5), _predicoes_cv(1, 2.0)
    assert a.comparacao_pareada(x, z)["delta"] == pytest.approx(-a.comparacao_pareada(z, x)["delta"])


def test_comparacao_pareada_recusa_pessoa_ausente():
    x, z = _predicoes_cv(0, 0.5), _predicoes_cv(1, 2.0)
    with pytest.raises(ValueError):
        a.comparacao_pareada(x, z[z["subject_uid"] != "S00"])


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
    monkeypatch.setattr("sys.argv", ["avaliar", "--modelos", str(tmp_path / "modelos")])
    a.main()
    resumo = (tmp_path / "modelos" / "resumo_dtsf.md").read_text()
    assert "| D | incompleto" not in resumo
    assert "## Ablações do D_pequeno" in resumo and "D_pequeno − D_sem_mixup" in resumo
    assert "D_pequeno − D_sem_T" in resumo and "D_pequeno − D_sem_E" in resumo
    assert "| SZ | D_FA − D |" in resumo and "| SZ | D − D_pequeno |" in resumo and "| MDD | D_FA − D_pequeno |" in resumo
    assert "| D_FA | 704.769 |" in resumo and "| D_pequeno | 80.449 |" in resumo
    assert "Média ± desvio nas 10 repetições" in resumo and resumo.count("| D_FA | 0.") >= 2
    for proibido in ("H1", "H2", "H3", "p (ranks)", "contra o acaso", "Holm"[:0] or "interseção-união"):
        assert proibido not in resumo
    assert (tmp_path / "modelos" / "resumo.md").read_text() == "histórico"
