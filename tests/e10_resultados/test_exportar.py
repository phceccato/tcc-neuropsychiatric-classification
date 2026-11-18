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
        _gravar_modelo(raiz / "07_DTSF/resultados" / d / "DTSF_FA", pred)
        md07.append(_linha_auc("DTSF_FA", pred if not quebrar else pred.assign(prob=1 - pred["prob"])))
        pc = _predicoes(10 + k)
        _gravar_modelo(raiz / "08_classicos/resultados" / d / "LR_Temporal", pc, dispositivo="cpu")
        md08.append(_linha_auc("LR_Temporal", pc).replace("| 704.769 ", ""))
        _gravar_modelo(raiz / "07_DTSF/resultados" / d / "DTSF", pred[pred["repeticao"] < 5])   # incompleto
        pl = _predicoes(20 + k)
        lr.append(pl.assign(dataset=d, features="absolutas"))
        r = avaliar(pl, semente=SEMENTE)
        md06.append(f"| {d} | absolutas | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | x |")
    (raiz / "06_LR_psd").mkdir(parents=True)
    pd.concat(lr).to_csv(raiz / "06_LR_psd/predicoes.csv", index=False)
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


def test_exporta_previsoes_metricas_e_manifesto(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    manifesto = ex.exportar(raiz, saida)
    assert sorted(manifesto["incompletos"]) == sorted(
        [f"{d}/{m}" for d in ex.DATASETS for m in ex.TIPOS_DTSF if m != "DTSF_FA"] +
        [f"{d}/{m}" for d in ex.DATASETS for m in ex.CLASSICOS if m != "LR_Temporal"] +
        [f"{d}/LR_PSD_Relative" for d in ex.DATASETS])
    p = pd.read_csv(saida / "previsoes/SZ/DTSF_FA.csv")
    assert list(p.columns) == ex.COLUNAS and len(p) == 80
    assert ((p["y_prob"] >= 0.5).astype(int) == p["y_pred"]).all()
    assert not (saida / "permutacoes").exists()
    m = pd.read_csv(saida / "metricas.csv")
    assert set(m["modelo"]) == {"DTSF_FA", "LR_Temporal", "LR_PSD_Absolute"} and "auc_rep_desvio" in m
    assert manifesto["modelos"]["SZ/DTSF_FA"] == dict(etapa="07", dispositivo=["cuda"], parametros=[704_769], linhas=80)
    assert sorted(a.name for a in (saida / "resumos").iterdir()) == ["06_LR_psd.md", "07_DTSF.md", "08_classicos.md"]
    assert (saida / "folds.csv").exists() and "y_true" in (saida / "LEIAME.md").read_text()


def test_auc_recalculada_do_exportado_e_a_do_modelo(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    ex.exportar(raiz, saida)
    p = pd.read_csv(saida / "previsoes/MDD/DTSF_FA.csv")
    pessoa = p.groupby("subject_uid").agg(y=("y_true", "first"), prob=("y_prob", "mean"))   # o código do LEIAME
    m = pd.read_csv(saida / "metricas.csv").set_index(["dataset", "modelo"])
    assert roc_auc_score(pessoa["y"], pessoa["prob"]) == pytest.approx(m.loc[("MDD", "DTSF_FA"), "auc_pessoa"])


def test_para_sem_gravar_se_a_auc_nao_bate_com_o_resumo(tmp_path):
    raiz, saida = _outputs(tmp_path, quebrar=True), tmp_path / "10_resultados"
    with pytest.raises(ex.Divergencia):
        ex.exportar(raiz, saida)
    assert not saida.exists()


def test_aucs_do_resumo_le_as_tabelas_por_dataset_com_e_sem_parametros():
    md = ("## SZ\n\n| DTSF_FA | 704.769 | 0.827 [0.643, 0.964] | x |\n| LR_Temporal | 0.602 [0.367, 0.806] | x |\n"
          "| DTSF_FA | 0.781 ± 0.058 | x |\n## Ablações\n| SZ | DTSF_FA − DTSF_Lite_SpectralOnly | +0.000 [-0.087, +0.087] |\n"
          "## MDD\n\n| DTSF_FA | 704.769 | 0.958 [0.897, 0.995] | x |\n")
    assert ex.aucs_do_resumo(md) == {("SZ", "DTSF_FA"): 0.827, ("SZ", "LR_Temporal"): 0.602, ("MDD", "DTSF_FA"): 0.958}


def test_exporta_o_historico_de_perdas_so_do_modelo_principal(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    hist = [dict(epoca=1, perda_treino=0.7, perda_validacao=0.6), dict(epoca=2, perda_treino=0.5, perda_validacao=0.8)]
    for a in (raiz / "07_DTSF/resultados/SZ/DTSF_FA").glob("r*_f*.json"):
        a.write_text(json.dumps(dict(dispositivo="cuda", parametros=704_769, melhor_epoca=1, historico=hist)))
    ex.exportar(raiz, saida)
    p = pd.read_csv(saida / "perdas/SZ/DTSF_FA.csv")
    assert list(p.columns) == ["repeticao", "fold", "epoca", "perda_treino", "perda_validacao", "melhor"]
    assert len(p) == 2 * len(list((raiz / "07_DTSF/resultados/SZ/DTSF_FA").glob("r*_f*.json")))
    assert p.groupby(["repeticao", "fold"])["melhor"].sum().eq(1).all() and set(p.loc[p["melhor"] == 1, "epoca"]) == {1}
    assert not (saida / "perdas/MDD").exists()          # sem histórico nos .json: nada a exportar
    assert not list((saida / "perdas").rglob("LR*"))


def test_recusa_exportar_sem_o_modelo_principal_e_nao_apaga_a_saida(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    ex.exportar(raiz, saida)                                    # exportação completa, como a que está no git
    antes = (saida / "metricas.csv").read_text()
    import shutil
    shutil.rmtree(raiz / "07_DTSF/resultados/SZ/DTSF_FA")           # máquina sem o DTSF_FA (rodada parcial)
    with pytest.raises(ex.RodadaIncompleta, match="DTSF_FA ausente"):
        ex.exportar(raiz, saida)
    assert (saida / "metricas.csv").read_text() == antes        # a pasta exportada continua intacta


def test_recusa_misturar_dispositivos_nos_dtsf(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    for d in ex.DATASETS:   # um segundo DTSF completo, mas em CPU
        _gravar_modelo(raiz / "07_DTSF/resultados" / d / "DTSF", _predicoes(30), dispositivo="cpu")
        md = raiz / "07_DTSF/resultados/resumo_dtsf.md"
        md.write_text(md.read_text() + f"\n## {d}\n\n" + _linha_auc("DTSF", _predicoes(30)))
    with pytest.raises(ex.RodadaIncompleta, match="dispositivos diferentes"):
        ex.exportar(raiz, saida)
    assert not saida.exists()


def test_permitir_parcial_exporta_mesmo_sem_o_principal(tmp_path):
    raiz, saida = _outputs(tmp_path), tmp_path / "10_resultados"
    import shutil
    shutil.rmtree(raiz / "07_DTSF/resultados/SZ/DTSF_FA")
    manifesto = ex.exportar(raiz, saida, permitir_parcial=True)
    assert "SZ/DTSF_FA" in manifesto["incompletos"] and "SZ/DTSF_FA" not in manifesto["modelos"]
