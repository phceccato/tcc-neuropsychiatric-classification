import numpy as np
import pandas as pd

from pipeline.e06_LR_psd import lr as b


def _janelas():
    # 2 pessoas, 4 janelas cada; a janela 2 de A é rejeitada; janelas 0 e 2 são da grade sem overlap
    return pd.DataFrame({
        "subject_uid": ["A"] * 4 + ["B"] * 4,
        "sem_overlap": [True, False, True, False] * 2,
        "rejeitada": [False, False, True, False] + [False] * 4,
    })


def test_media_por_pessoa_usa_so_janelas_aceitas():
    x = np.arange(8, dtype=float)[:, None] * np.ones((1, 3))
    medias = b.media_por_pessoa(x, _janelas(), so_sem_overlap=False)
    np.testing.assert_allclose(medias.loc["A"], [(0 + 1 + 3) / 3] * 3)
    np.testing.assert_allclose(medias.loc["B"], [(4 + 5 + 6 + 7) / 4] * 3)


def test_media_por_pessoa_na_grade_sem_overlap():
    x = np.arange(8, dtype=float)[:, None]
    medias = b.media_por_pessoa(x, _janelas(), so_sem_overlap=True)
    np.testing.assert_allclose(medias.loc["A"], [0.0])          # janela 2 rejeitada
    np.testing.assert_allclose(medias.loc["B"], [(4 + 6) / 2])


def _cenario(n=20, separacao=3.0, semente=0):
    rng = np.random.default_rng(semente)
    uids = [f"S{i}" for i in range(n)]
    y = np.array([1] * (n // 2) + [0] * (n // 2))
    x = rng.normal(0, 1, (n, 5))
    x[:, 0] += separacao * y
    folds = []
    for r in range(2):
        ordem = rng.permutation(n)
        for fold in range(2):
            teste = set(ordem[fold::2])
            for i in range(n):
                folds.append(dict(repeticao=r, fold=fold, subject_uid=uids[i],
                                  papel="teste" if i in teste else "treino"))
    treino = pd.DataFrame(x, index=uids)
    return treino, treino.copy(), pd.Series(y, index=uids), pd.DataFrame(folds)


def test_prever_cv_separa_dados_separaveis_e_preve_cada_pessoa_uma_vez_por_repeticao():
    x_treino, x_teste, y, folds = _cenario()
    pred = b.prever_cv(x_treino, x_teste, y, folds)
    assert pred.groupby(["repeticao", "subject_uid"]).size().eq(1).all()
    assert len(pred) == 2 * 20
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score(pred["y"], pred["prob"]) > 0.95


def test_rotulo_do_teste_nao_influencia_a_previsao():
    x_treino, x_teste, y, folds = _cenario()
    pred = b.prever_cv(x_treino, x_teste, y, folds)
    alvo = pred[(pred["repeticao"] == 0)].iloc[0]
    y_trocado = y.copy()
    y_trocado[alvo["subject_uid"]] = 1 - y_trocado[alvo["subject_uid"]]
    pred2 = b.prever_cv(x_treino, x_teste, y_trocado, folds)
    linha = (pred2["repeticao"] == 0) & (pred2["subject_uid"] == alvo["subject_uid"])
    assert pred2.loc[linha, "prob"].item() == alvo["prob"]


def test_lr_nao_tem_teste_de_permutacao():
    """O trabalho não tem teste contra o acaso: a incerteza da LR é o IC de 95% da AUC (como no resto)."""
    assert not hasattr(b, "permutacao") and not hasattr(b, "PERMUTACOES")


def test_regressao_usa_peso_de_classe():
    x_treino, x_teste, y, folds = _cenario()
    y_desbalanceado = y.copy()
    y_desbalanceado.iloc[:6] = 0  # 4 pacientes, 16 controles
    pred = b.prever_cv(x_treino, x_teste, y_desbalanceado, folds)
    # sem peso de classe, a probabilidade média ficaria perto da prevalência (0,2)
    assert pred["prob"].mean() > 0.3


def test_doc_06_bate_com_as_aucs_exportadas():
    """A tabela de docs/06_LR_psd.md traz as AUCs de outputs/10_resultados/metricas.csv (única saída versionada)."""
    from pathlib import Path
    raiz = Path(__file__).resolve().parents[2]
    metricas, doc = raiz / "outputs/10_resultados/metricas.csv", raiz / "docs/06_LR_psd.md"
    if not metricas.exists():
        import pytest
        pytest.skip("outputs/10_resultados não gerado")
    m = pd.read_csv(metricas).set_index(["dataset", "modelo"])
    linhas = [l for l in doc.read_text(encoding="utf-8").splitlines() if l.startswith(("| SZ |", "| MDD |"))]
    assert len(linhas) == 4
    for linha in linhas:
        c = [x.strip() for x in linha.strip("|").split("|")]
        auc = float(c[2].split("[")[0].replace(",", "."))
        nome = "LR_psd_" + c[1]
        assert abs(m.loc[(c[0], nome), "auc_pessoa"] - auc) <= 0.0005, (c[0], nome)
