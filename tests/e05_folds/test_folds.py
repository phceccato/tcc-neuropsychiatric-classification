import pandas as pd
import pytest

from pipeline.e05_folds import folds as f


def _pessoas(n_pac=14, n_hc=14):
    return pd.DataFrame({
        "subject_uid": [f"P{i}" for i in range(n_pac)] + [f"H{i}" for i in range(n_hc)],
        "grupo": ["SZ"] * n_pac + ["HC"] * n_hc,
    })


@pytest.fixture
def particoes():
    return f.gerar(_pessoas(), repeticoes=3, k=5, semente=0)


def test_cada_pessoa_fica_no_teste_uma_vez_por_repeticao(particoes):
    teste = particoes[particoes["papel"] == "teste"]
    contagem = teste.groupby(["repeticao", "subject_uid"]).size()
    assert (contagem == 1).all()
    assert teste.groupby("repeticao")["subject_uid"].nunique().eq(28).all()


def test_cada_fold_lista_todas_as_pessoas_uma_vez(particoes):
    por_fold = particoes.groupby(["repeticao", "fold"])["subject_uid"]
    assert por_fold.nunique().eq(28).all()
    assert por_fold.size().eq(28).all()


def test_teste_de_cada_fold_tem_as_duas_classes(particoes):
    teste = particoes[particoes["papel"] == "teste"]
    assert teste.groupby(["repeticao", "fold"])["grupo"].nunique().eq(2).all()


def test_validacao_interna_tem_20_por_cento_do_treino_e_as_duas_classes(particoes):
    for _, g in particoes.groupby(["repeticao", "fold"]):
        treino_total = g[g["papel"].isin(["treino", "validacao"])]
        validacao = g[g["papel"] == "validacao"]
        assert len(validacao) == round(0.2 * len(treino_total))
        assert validacao["grupo"].nunique() == 2


def test_repeticoes_diferem_e_a_semente_reproduz(particoes):
    chave = ["repeticao", "fold", "subject_uid"]
    teste = particoes[particoes["papel"] == "teste"]
    r0 = set(teste.loc[teste["repeticao"] == 0, ["fold", "subject_uid"]].itertuples(index=False))
    r1 = set(teste.loc[teste["repeticao"] == 1, ["fold", "subject_uid"]].itertuples(index=False))
    assert r0 != r1
    de_novo = f.gerar(_pessoas(), repeticoes=3, k=5, semente=0)
    pd.testing.assert_frame_equal(particoes.sort_values(chave).reset_index(drop=True),
                                  de_novo.sort_values(chave).reset_index(drop=True))


# ---------- integridade das partições gravadas (usadas pelo DTSF e pelos baselines) ----------

FOLDS_CSV = __import__("pathlib").Path("outputs/05_folds/folds.csv")


@pytest.mark.skipif(not FOLDS_CSV.exists(), reason="outputs/05_folds/folds.csv não gerado")
def test_particoes_gravadas_sao_integras_por_pessoa():
    folds = pd.read_csv(FOLDS_CSV)
    por_fold = folds.groupby(["dataset", "repeticao", "fold"])
    assert por_fold["subject_uid"].apply(lambda s: s.is_unique).all()  # um papel por pessoa em cada fold
    pessoas = folds.groupby("dataset")["subject_uid"].nunique()
    assert (por_fold["subject_uid"].nunique() == por_fold["dataset"].first().map(pessoas)).all()
    teste = folds[folds["papel"] == "teste"].groupby(["dataset", "repeticao", "subject_uid"]).size()
    assert (teste == 1).all()  # cada pessoa no teste exatamente uma vez por repetição
    assert (folds.groupby(["dataset", "repeticao"])["fold"].nunique() == 5).all()
    assert folds.groupby("dataset")["repeticao"].nunique().eq(10).all()
    assert set(folds["papel"]) == {"treino", "validacao", "teste"}
