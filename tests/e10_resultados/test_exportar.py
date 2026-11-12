import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.avaliacao.metricas import avaliar
from pipeline.e07_DTSF.avaliar import SEMENTE
from pipeline.e10_resultados import exportar as ex


def _predicoes(semente, n=8, repeticoes=10, folds=2):
    rng = np.random.default_rng(semente)
    y = [1] * (n // 2) + [0] * (n // 2)
    return pd.DataFrame([dict(repeticao=r, fold=i % folds, subject_uid=f"S{i}", y=y[i],
                              prob=float(np.clip(0.5 + (y[i] - 0.5) * 0.4 + rng.normal(0, 0.2), 0, 1)))
                         for r in range(repeticoes) for i in range(n)])


def _gravar_modelo(pasta, pred, dispositivo="cuda", prefixo="r", chave="repeticao"):
    pasta.mkdir(parents=True, exist_ok=True)
    for (k, f), g in pred.groupby([chave, "fold"]):
        g.drop(columns=[c for c in ("permutacao",) if c in g]).to_csv(pasta / f"{prefixo}{k}_f{f}.csv", index=False)
        (pasta / f"{prefixo}{k}_f{f}.json").write_text(json.dumps(dict(dispositivo=dispositivo, parametros=704_769)))


def _linha_auc(modelo, pred):
    r = avaliar(pred, semente=SEMENTE)
    return f"| {modelo} | 704.769 | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | x |"


def _outputs(tmp_path, quebrar=False):
    raiz = tmp_path / "outputs"
    md07, md08, md06 = ["# 07"], ["# 08"], ["# 06", "| Conjunto | Features |"]
    lr = []
    for k, d in enumerate(ex.DATASETS):
        md07 += [f"## {d}", ""]
        md08 += [f"## {d}", ""]
        pred = _predicoes(k)
        _gravar_modelo(raiz / "07_DTSF/resultados" / d / "D_FA", pred)
        md07.append(_linha_auc("D_FA", pred if not quebrar else pred.assign(prob=1 - pred["prob"])))
        perm = pred[pred["repeticao"] == 0].drop(columns="repeticao").assign(repeticao=0)
        for j in range(3):
            _gravar_modelo(raiz / "07_DTSF/permutacoes" / d / "D_FA", perm.assign(permutacao=j), prefixo="p",
                           chave="permutacao")
        pc = _predicoes(10 + k)
        _gravar_modelo(raiz / "08_classicos/resultados" / d / "LR_temporal", pc, dispositivo="cpu")
        md08.append(_linha_auc("LR_temporal", pc).replace("| 704.769 ", ""))
        _gravar_modelo(raiz / "07_DTSF/resultados" / d / "D", pred[pred["repeticao"] < 5])   # incompleto
        pl = _predicoes(20 + k)
        lr.append(pl.assign(dataset=d, features="absolutas"))
        r = avaliar(pl, semente=SEMENTE)
        md06.append(f"| {d} | absolutas | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | x |")
    (raiz / "06_LR_psd").mkdir(parents=True)
    pd.concat(lr).to_csv(raiz / "06_LR_psd/predicoes.csv", index=False)
    pd.DataFrame(dict(dataset=["SZ", "SZ", "MDD"], auc_nula=[0.5, 0.6, 0.4])).to_csv(
        raiz / "06_LR_psd/permutacoes.csv", index=False)
    (raiz / "06_LR_psd/resumo.md").write_text("\n".join(md06))
    (raiz / "07_DTSF/resultados/resumo_dtsf.md").write_text("\n".join(md07))
    (raiz / "08_classicos/resultados/resumo.md").write_text("\n".join(md08))
    (raiz / "05_folds").mkdir(parents=True)
    pd.DataFrame(dict(dataset=["SZ"], repeticao=[0], fold=[0], subject_uid=["S0"], grupo=["SZ"], papel=["teste"])).to_csv(
        raiz / "05_folds/folds.csv", index=False)
    return raiz


def test_padronizar_tem_y_true_y_pred_y_prob():
    p = pd.DataFrame(dict(repeticao=[0, 0], fold=[1, 0], subject_uid=["B", "A"], y=[1, 0], prob=[0.5, 0.49]))
    t = ex.padronizar(p)
    assert list(t.columns) == ex.COLUNAS
    assert t.to_dict("records") == [dict(repeticao=0, fold=0, subject_uid="A", y_true=0, y_pred=0, y_prob=0.49),
                                    dict(repeticao=0, fold=1, subject_uid="B", y_true=1, y_pred=1, y_prob=0.5)]


def test_exporta_previsoes_permutacoes_metricas_e_manifesto(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    manifesto = ex.exportar(raiz, saida)
    assert sorted(manifesto["incompletos"]) == sorted(
        [f"{d}/{m}" for d in ex.DATASETS for m in ex.TIPOS_DTSF if m != "D_FA"] +
        [f"{d}/{m}" for d in ex.DATASETS for m in ex.CLASSICOS if m != "LR_temporal"] +
        [f"{d}/LR_psd_relativas" for d in ex.DATASETS])
    p = pd.read_csv(saida / "previsoes/SZ/D_FA.csv")
    assert list(p.columns) == ex.COLUNAS and len(p) == 80
    assert ((p["y_prob"] >= 0.5).astype(int) == p["y_pred"]).all()
    perm = pd.read_csv(saida / "permutacoes/SZ/D_FA.csv")
    assert list(perm.columns)[0] == "permutacao" and perm["permutacao"].nunique() == 3
    assert len(pd.read_csv(saida / "permutacoes/SZ/LR_psd_absolutas_nulas.csv")) == 2
    m = pd.read_csv(saida / "metricas.csv")
    assert set(m["modelo"]) == {"D_FA", "LR_temporal", "LR_psd_absolutas"} and "auc_rep_desvio" in m
    assert manifesto["modelos"]["SZ/D_FA"] == dict(etapa="07", dispositivo=["cuda"], parametros=[704_769], linhas=80,
                                                   permutacoes=3)
    assert sorted(a.name for a in (saida / "resumos").iterdir()) == ["06_LR_psd.md", "07_DTSF.md", "08_classicos.md"]
    assert (saida / "folds.csv").exists() and "y_true" in (saida / "LEIAME.md").read_text()


def test_auc_recalculada_do_exportado_e_a_do_modelo(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    ex.exportar(raiz, saida)
    p = pd.read_csv(saida / "previsoes/MDD/D_FA.csv")
    pessoa = p.groupby("subject_uid").agg(y=("y_true", "first"), prob=("y_prob", "mean"))   # o código do LEIAME
    m = pd.read_csv(saida / "metricas.csv").set_index(["dataset", "modelo"])
    assert roc_auc_score(pessoa["y"], pessoa["prob"]) == pytest.approx(m.loc[("MDD", "D_FA"), "auc_pessoa"])


def test_para_sem_gravar_se_a_auc_nao_bate_com_o_resumo(tmp_path):
    raiz, saida = _outputs(tmp_path, quebrar=True), tmp_path / "10_resultados"
    with pytest.raises(ex.Divergencia):
        ex.exportar(raiz, saida)
    assert not saida.exists()


def test_aucs_do_resumo_le_as_tabelas_por_dataset_com_e_sem_parametros():
    md = ("## SZ\n\n| D_FA | 704.769 | 0.827 [0.643, 0.964] | x |\n| LR_temporal | 0.602 [0.367, 0.806] | x |\n"
          "| D_FA | 0.781 ± 0.058 | x |\n## H2\n| SZ | D_FA − D_sem_T | +0.000 [-0.087, +0.087] |\n"
          "## MDD\n\n| D_FA | 704.769 | 0.958 [0.897, 0.995] | x |\n")
    assert ex.aucs_do_resumo(md) == {("SZ", "D_FA"): 0.827, ("SZ", "LR_temporal"): 0.602, ("MDD", "D_FA"): 0.958}
