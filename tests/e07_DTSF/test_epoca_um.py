import json

import numpy as np
import pandas as pd

from pipeline.e07_DTSF import epoca_um as e


def _cenario():
    """2 repetições × 1 fold, 4 pessoas cada; o fold da repetição 1 escolheu a época 1 e previu constante."""
    folds = pd.DataFrame([dict(dataset="SZ", modelo="M", repeticao=r, fold=0, melhor_epoca=ep, dispositivo="cpu")
                          for r, ep in ((0, 7), (1, 1))])
    linhas = []
    for r in (0, 1):
        for u, y in zip("abcd", (0, 0, 1, 1)):
            prob = 0.5 if r == 1 else (0.1 if y == 0 else 0.9)
            linhas.append(dict(dataset="SZ", modelo="M", repeticao=r, fold=0, subject_uid=u, y=y, prob=prob))
    return folds, pd.DataFrame(linhas)


def test_resumo_conta_folds_com_epoca1_e_separa_as_aucs():
    folds, previsoes = _cenario()
    r = e.resumo_epoca_um(folds, previsoes).iloc[0]
    assert (r.folds, r.folds_epoca1, r.fracao_epoca1) == (2, 1, 0.5)
    assert (r.epoca_mediana, r.epoca_min, r.epoca_max) == (4.0, 1, 7)
    assert r.auc_repeticao_todos == 0.75          # (1,0 + 0,5) / 2
    assert r.auc_repeticao_sem_epoca1 == 1.0      # só a repetição 0
    assert r.auc_pessoa_sem_epoca1 == 1.0 and r.pessoas_sem_epoca1 == 4
    assert r.auc_pessoa_todos == 1.0              # a média das duas repetições preserva a ordem


def test_sem_nenhum_fold_com_epoca1_as_duas_aucs_coincidem():
    folds, previsoes = _cenario()
    folds["melhor_epoca"] = [7, 9]
    r = e.resumo_epoca_um(folds, previsoes).iloc[0]
    assert r.folds_epoca1 == 0
    assert r.auc_pessoa_todos == r.auc_pessoa_sem_epoca1
    assert r.auc_repeticao_todos == r.auc_repeticao_sem_epoca1


def test_todos_os_folds_com_epoca1_dao_nan_sem_quebrar():
    folds, previsoes = _cenario()
    folds["melhor_epoca"] = 1
    r = e.resumo_epoca_um(folds, previsoes).iloc[0]
    assert r.fracao_epoca1 == 1.0 and np.isnan(r.auc_pessoa_sem_epoca1) and np.isnan(r.auc_repeticao_sem_epoca1)
    assert r.pessoas_sem_epoca1 == 0


def test_fracao_por_validacao_separa_pelo_tamanho_da_validacao():
    folds = pd.DataFrame([dict(dataset="SZ", repeticao=0, fold=0, melhor_epoca=1),
                          dict(dataset="SZ", repeticao=0, fold=1, melhor_epoca=5)])
    particoes = pd.DataFrame([dict(dataset="SZ", repeticao=0, fold=0, papel="validacao", subject_uid=u) for u in "ab"]
                             + [dict(dataset="SZ", repeticao=0, fold=1, papel="validacao", subject_uid=u) for u in "abcd"]
                             + [dict(dataset="SZ", repeticao=0, fold=0, papel="treino", subject_uid="x")])
    r = e.fracao_por_validacao(folds, particoes).set_index("pessoas_validacao")
    assert r.loc[2, "fracao_epoca1"] == 1.0 and r.loc[4, "fracao_epoca1"] == 0.0


def test_carregar_resultados_le_epoca_e_dispositivo_do_json(tmp_path):
    pasta = tmp_path / "SZ" / "M"
    pasta.mkdir(parents=True)
    pd.DataFrame(dict(repeticao=[3, 3], fold=[2, 2], subject_uid=["a", "b"], y=[0, 1], prob=[.2, .8])).to_csv(
        pasta / "r3_f2.csv", index=False)
    (pasta / "r3_f2.json").write_text(json.dumps(dict(melhor_epoca=1)))           # json antigo: sem dispositivo
    folds, previsoes = e.carregar_resultados(tmp_path)
    assert folds.iloc[0].to_dict() == dict(dataset="SZ", modelo="M", repeticao=3, fold=2, melhor_epoca=1, dispositivo="cpu")
    assert len(previsoes) == 2


def test_carregar_exportados_usa_a_epoca_marcada_como_melhor(tmp_path):
    (tmp_path / "perdas" / "SZ").mkdir(parents=True)
    (tmp_path / "previsoes" / "SZ").mkdir(parents=True)
    pd.DataFrame(dict(repeticao=0, fold=0, epoca=[1, 2, 3], perda_treino=0.5, perda_validacao=0.5, melhor=[0, 1, 0])).to_csv(
        tmp_path / "perdas" / "SZ" / "DTSF_FA.csv", index=False)
    pd.DataFrame(dict(repeticao=0, fold=0, subject_uid=["a", "b"], y_true=[0, 1], y_pred=[0, 1], y_prob=[.1, .9])).to_csv(
        tmp_path / "previsoes" / "SZ" / "DTSF_FA.csv", index=False)
    folds, previsoes = e.carregar_exportados(tmp_path, "DTSF_FA", "cuda")
    assert folds.iloc[0]["melhor_epoca"] == 2 and folds.iloc[0]["dispositivo"] == "cuda"
    assert {"y", "prob"} <= set(previsoes.columns)
