import functools
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.e11_graficos import graficos as g
from pipeline.e11_graficos import numeros as n
from tests.e11_graficos.conftest import PESSOAS, _resultados_sinteticos

MD_MODELOS = """# x

## SZ

| Modelo | AUC por pessoa [IC 95%] | outra |
| --- | --- | --- |
| D_FA | 0.827 [0.643, 0.964] | 0.781 ± 0.058 |
| LR_temporal | 0.602 [0.367, 0.806] | 0.617 ± 0.070 |

## MDD

| D_FA | 704.769 | 0.958 [0.897, 0.995] | 0.937 ± 0.023 |

## H1: D_FA acima do acaso

| SZ | 0.192 (25) | não rejeita H0 |
"""


def test_tabela_modelos_le_por_dataset_com_e_sem_parametros_e_ignora_outras_secoes():
    t = n.tabela_modelos(MD_MODELOS)
    assert t == {("SZ", "D_FA"): (0.827, 0.643, 0.964), ("SZ", "LR_temporal"): (0.602, 0.367, 0.806),
                 ("MDD", "D_FA"): (0.958, 0.897, 0.995)}


def test_tabelas_de_deltas_das_etapas_07_e_08():
    md = ("| SZ | D_pequeno − D_sem_T | +0.000 [-0.087, +0.087] | 0.5202 | x |\n"
          "| MDD | LR_temporal | +0.074 [+0.022, +0.142] | 0.0008 | x |\n"
          "| SZ | D_FA − D | +0.020 [-0.015, +0.082] | 0.1503 | x |\n")
    assert n.tabela_deltas(md, "D_FA") == {("SZ", "D_pequeno − D_sem_T"): (0.0, -0.087, 0.087, 0.5202),
                                           ("MDD", "D_FA − LR_temporal"): (0.074, 0.022, 0.142, 0.0008),
                                           ("SZ", "D_FA − D"): (0.020, -0.015, 0.082, 0.1503)}


def test_tabela_lr_psd():
    md = "| SZ | absolutas | 0.724 [0.520, 0.903] | 0.717 ± 0.046 |\n| MDD | relativas | 0.879 [0.774, 0.962] | x |\n"
    assert n.tabela_lr_psd(md) == {("SZ", "LR_psd_absolutas"): (0.724, 0.520, 0.903),
                                   ("MDD", "LR_psd_relativas"): (0.879, 0.774, 0.962)}


def test_conferir_para_quando_nao_bate():
    n.conferir("x", 0.8274, 0.827)
    with pytest.raises(n.Divergencia):
        n.conferir("x", 0.8276, 0.827)
    with pytest.raises(n.Divergencia):
        n.conferir("p", 0.5203, 0.5202, n.TOL_P)


def test_confusao_usa_o_escore_medio_da_pessoa():
    pred = pd.DataFrame(dict(subject_uid=["a", "a", "b", "b", "c", "c"], y=[1, 1, 0, 0, 1, 1],
                             prob=[0.9, 0.3, 0.6, 0.6, 0.2, 0.4], repeticao=[0, 1] * 3))
    assert n.confusao(pred) == dict(vp=1, fn=1, fp=1, vn=0)


def test_comparacoes_tem_a_referencia_de_cada_hipotese_e_as_descritivas():
    pares = n.comparacoes()
    assert ("D_pequeno", "D_sem_T") in pares and ("D_pequeno", "D_sem_E") in pares and ("D_pequeno", "D_sem_mixup") in pares
    assert {b for a, b in pares if a == "D_FA"} >= set(n.CLASSICOS) and ("D_FA", "D") in pares
    assert ("D", "D_pequeno") in pares and ("D_FA", "D_pequeno") in pares and len(pares) == len(set(pares))


def test_figura_das_hipoteses_usa_as_comparacoes_calculadas():
    chaves = {chave for _, _, itens in g.HIPOTESES_FIG for chave, _, _ in itens}
    assert chaves == {f"{a} − {b}" for a, b in n.comparacoes()}
    assert [h for _, h, _ in g.HIPOTESES_FIG] == ["H2", "H3", "H4", None, None]
    h4 = next(itens for _, h, itens in g.HIPOTESES_FIG if h == "H4")
    assert [c for c, _, _ in h4] == [f"D_FA − {m}" for m in n.BASELINES_H4]


def test_grupos_dos_graficos_tem_todos_os_modelos_exportados():
    assert sorted(m for _, ms in g.GRUPOS for m in ms) == sorted(n.MODELOS)
    assert set(g.MODELOS_PRINCIPAIS) <= set(n.MODELOS) and set(g.ESTILO_ROC) <= set(n.MODELOS)


def test_virgula_e_sinal():
    assert g.virgula(0.8274) == "0,827"
    assert g.virgula(-0.12, sinal=True) == "−0,120"
    assert g.virgula(0.02, sinal=True) == "+0,020"


def test_numeros_de_ponta_a_ponta(numeros_sinteticos):
    nums = json.loads(json.dumps(numeros_sinteticos))           # o que vai para numeros.json
    assert set(nums["modelos"]["SZ"]) == set(n.MODELOS)
    r = nums["modelos"]["MDD"]["D_FA"]
    assert set(r["media_desvio"]) >= {"auc", "acuracia", "precisao", "sensibilidade", "f1"}
    assert len(r["escores"]["y"]) == PESSOAS and "subject_uid" not in json.dumps(nums)
    assert sum(r["confusao"].values()) == PESSOAS
    assert len(nums["h1_distribuicao"]["SZ"]["nulas"]) == 25 and set(nums["h1"]) == set(n.DATASETS)
    assert set(nums["hipoteses"]) == {"H2", "H3", "H4"} and "D_FA − LR_combinado" in nums["comparacoes"]["SZ"]


def test_numeros_param_se_o_resumo_nao_bate(tmp_path):
    comparar = functools.partial(n.comparacao_pareada, n=200)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(n, "comparacao_pareada", comparar)
        _resultados_sinteticos(tmp_path, comparar)
        md = tmp_path / "resumos" / "08_classicos.md"
        md.write_text(md.read_text().replace("| RF_espectral | 0.", "| RF_espectral | 9.", 1))
        with pytest.raises((n.Divergencia, KeyError)):
            n.calcular(tmp_path)


def test_linhas_da_tabela_de_testes(numeros_sinteticos):
    linhas = g.linhas_testes(numeros_sinteticos)
    assert len(linhas) == 8 and [l[0] for l in linhas[::2]] == ["H1", "H2", "H3", "H4"]
    assert all(l[-1] in ("rejeita H₀", "não rejeita H₀") for l in linhas)


@pytest.mark.parametrize("figura", list(g.FIGURAS))
def test_cada_figura_grava_pdf_e_png_na_sua_pasta(figura, numeros_sinteticos, tmp_path):
    g.configurar()
    caminhos = g.FIGURAS[figura](json.loads(json.dumps(numeros_sinteticos)), tmp_path)
    assert [c.suffix for c in caminhos] == [".pdf", ".png"] * (len(caminhos) // 2)
    assert all(c.stat().st_size > 1000 for c in caminhos)
    pasta = {c.parent.name for c in caminhos}
    assert pasta == ({g.PRINCIPAIS} if caminhos[0].name.startswith("p") else {g.SUPLEMENTARES})


# ---------- perdas do D_FA e ROC de todos os modelos ----------

def test_numeros_trazem_as_perdas_conferidas_do_d_fa(numeros_json):
    folds = numeros_json["perdas"]["SZ"]
    assert len(folds) == 50 and {f["repeticao"] for f in folds} == set(range(10))
    f = folds[0]
    assert len(f["treino"]) == len(f["validacao"]) and f["melhor"] == 1 + int(np.argmin(f["validacao"]))


def test_perdas_param_se_a_epoca_escolhida_nao_e_a_de_menor_validacao(tmp_path):
    from tests.e11_graficos.conftest import _perdas
    t = _perdas(0)
    t.loc[(t["repeticao"] == 0) & (t["fold"] == 0), "melhor"] = 0
    t.loc[(t["repeticao"] == 0) & (t["fold"] == 0) & (t["epoca"] == 1), "melhor"] = 1
    t.loc[(t["repeticao"] == 0) & (t["fold"] == 0) & (t["epoca"] == 1), "perda_validacao"] = 9.0
    t.to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(n.Divergencia):
        n.perdas_folds(tmp_path / "d.csv")


def test_perdas_param_com_folds_faltando(tmp_path):
    from tests.e11_graficos.conftest import _perdas
    t = _perdas(0)
    t[~((t["repeticao"] == 9) & (t["fold"] == 4))].to_csv(tmp_path / "d.csv", index=False)
    with pytest.raises(n.Divergencia, match="49 folds"):
        n.perdas_folds(tmp_path / "d.csv")


def test_figura_de_perdas_sem_perdas_nao_grava_e_avisa(numeros_json, tmp_path, capsys):
    g.configurar()
    sem = {k: v for k, v in numeros_json.items() if k != "perdas"}
    assert g.fig_loss_d_fa(sem, tmp_path) == [] and "perdas" in capsys.readouterr().out
    assert not list(tmp_path.rglob("*.pdf"))


def test_curva_pr_bate_com_o_sklearn_e_a_ap_e_a_area_por_passos():
    from sklearn.metrics import average_precision_score
    e = dict(y=[0, 0, 1, 1, 0, 1, 1, 0], prob=[0.1, 0.4, 0.35, 0.8, 0.5, 0.65, 0.9, 0.2])
    revocacao, precisao, ap = g.curva_pr(e)
    assert ap == pytest.approx(average_precision_score(e["y"], e["prob"])) and revocacao[0] == 1.0
    assert revocacao[-1] == 0.0 and precisao[-1] == 1.0 and len(revocacao) == len(precisao)
    assert -np.sum(np.diff(revocacao) * precisao[:-1]) == pytest.approx(ap)


def test_figura_roc_pr_tem_uma_figura_por_transtorno_com_os_6_principais(numeros_json, tmp_path):
    g.configurar()
    caminhos = g.fig_roc_pr(numeros_json, tmp_path)
    assert sorted(c.name for c in caminhos) == ["p6_roc_pr_mdd.pdf", "p6_roc_pr_mdd.png",
                                                 "p6_roc_pr_sz.pdf", "p6_roc_pr_sz.png"]
    assert {c.parent.name for c in caminhos} == {g.PRINCIPAIS} and len(g.MODELOS_PRINCIPAIS) == 6


def test_valores_do_grafico_de_linhas_sao_as_medias_nas_repeticoes_em_porcentagem(numeros_json):
    md = numeros_json["modelos"]["SZ"]["D_FA"]["media_desvio"]
    v = g.valores_limiar(numeros_json, "SZ", "D_FA")
    assert list(v) == ["acuracia", "precisao", "sensibilidade", "f1"]
    assert v["acuracia"] == pytest.approx(100 * md["acuracia"][0]) and v["f1"] == pytest.approx(100 * md["f1"][0])


def test_limites_do_eixo_incluem_todos_os_valores():
    assert g.limites_y_limiar([53.2, 90.1]) == (40, 100)
    assert g.limites_y_limiar([71.0, 80.0]) == (60, 100)
    assert g.limites_y_limiar([2.0, 50.0]) == (0, 100)


def test_graficos_de_linhas_gravam_nas_pastas_certas(numeros_json, tmp_path):
    g.configurar()
    a = g.fig_metricas_limiar(numeros_json, tmp_path)
    b = g.fig_metricas_limiar_todos(numeros_json, tmp_path)
    assert [c.name for c in a] == ["p7_metricas_limiar.pdf", "p7_metricas_limiar.png"] and a[0].parent.name == g.PRINCIPAIS
    assert [c.name for c in b] == ["s11_metricas_limiar_todos.pdf", "s11_metricas_limiar_todos.png"]
    assert b[0].parent.name == g.SUPLEMENTARES


def test_bootstrap_do_principal_tem_o_mesmo_ic_do_resumo_e_curvas_na_grade(numeros_json):
    b = numeros_json["bootstrap"]["MDD"]
    r = numeros_json["modelos"]["MDD"]["D_FA"]
    assert b["ic"] == pytest.approx(r["ic"], abs=1e-9) and b["reamostragens"] == 2000
    assert len(b["curvas"]) == n.N_CURVAS_BOOTSTRAP and len(b["curvas"][0]) == len(b["grade"]) == 101
    curvas = np.array(b["curvas"])
    assert (np.diff(curvas, axis=1) >= -1e-9).all() and (curvas[:, -1] == 1).all() and (curvas >= 0).all()


def test_figura_do_bootstrap_grava_em_principais(numeros_json, tmp_path):
    g.configurar()
    caminhos = g.fig_roc_bootstrap(numeros_json, tmp_path)
    assert [c.name for c in caminhos] == ["p8_roc_bootstrap_d_fa.pdf", "p8_roc_bootstrap_d_fa.png"]
    assert caminhos[0].parent.name == g.PRINCIPAIS


def test_mediana_por_epoca_so_onde_restam_folds_suficientes():
    x, y = g._mediana_por_epoca([[1.0, 2.0, 3.0], [3.0, 4.0], [5.0]], minimo=2)
    assert x.tolist() == [1, 2] and y.tolist() == [3.0, 3.0]


def test_roc_de_todos_os_modelos_tem_um_arquivo_por_transtorno_e_todos_os_17_modelos(numeros_json, tmp_path):
    g.configurar()
    caminhos = g.fig_roc_todos(numeros_json, tmp_path)
    assert sorted(c.name for c in caminhos) == ["s10_roc_todos_mdd.pdf", "s10_roc_todos_mdd.png",
                                                 "s10_roc_todos_sz.pdf", "s10_roc_todos_sz.png"]
    desenhados = {m for _, ms in g.GRUPOS for m in ms}
    assert len(desenhados) == 17 and desenhados == set(numeros_json["modelos"]["SZ"])
    assert all(g.estilo_roc_todos(m) for m in desenhados)
