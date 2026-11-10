import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.e08_classicos import rodar as cr
from pipeline.e08_classicos import treino as ct
from pipeline.e07_DTSF.dados import janelas_avaliacao


def _cenario(n=24, janelas=80, p=12, separacao=1.5, semente=0):
    """n pessoas (metade pacientes), `janelas` por pessoa; a característica 0 desloca nos pacientes."""
    rng = np.random.default_rng(semente)
    uids = [f"P{i:02d}" for i in range(n // 2)] + [f"H{i:02d}" for i in range(n // 2)]
    y = pd.Series([1] * (n // 2) + [0] * (n // 2), index=uids)
    indice = pd.DataFrame([dict(subject_uid=u, dataset="SZ", inicio_amostra=k * 500, sem_overlap=k % 2 == 0,
                                rejeitada=k % 9 == 4) for u in uids for k in range(janelas)])
    indice["linha"] = np.arange(len(indice))
    x = rng.normal(size=(len(indice), p))
    x[:, 0] += separacao * indice["subject_uid"].map(y).to_numpy()
    x[indice["rejeitada"].to_numpy()] = np.nan                 # como no cache real
    return x, indice, y, uids


def _papeis(uids):
    teste = uids[:3] + uids[12:15]
    return [u for u in uids if u not in teste], teste


@pytest.mark.parametrize("modelo", ct.MODELOS)
def test_cada_modelo_separa_dados_separaveis_e_preve_todas_as_pessoas_de_teste(modelo):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    r = ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=1)
    assert sorted(r.prob) == sorted(teste)
    assert roc_auc_score([y[u] for u in r.prob], list(r.prob.values())) > 0.9
    assert r.parametros in ct.GRADES[ct.separar(modelo)[0]] and len(r.auc_interna) == 3
    aceitas = (~indice["rejeitada"]).groupby(indice["subject_uid"]).sum()
    assert r.janelas_ajuste == sum(min(ct.JANELAS_POR_PESSOA, aceitas[u]) for u in ajuste)


def test_150_janelas_por_pessoa_no_ajuste():
    assert ct.JANELAS_POR_PESSOA == 150


def test_pesos_iguais_por_pessoa_e_por_classe_e_media_1():
    contagens, rotulos = [150, 83, 120, 150, 149], [1, 1, 1, 0, 0]
    w = ct.pesos_janelas(contagens, rotulos)
    assert len(w) == sum(contagens) and w.mean() == pytest.approx(1.0)
    inicio = np.cumsum([0, *contagens])
    por_pessoa = [w[a:b].sum() for a, b in zip(inicio[:-1], inicio[1:])]
    assert all(np.allclose(w[a:b], w[a]) for a, b in zip(inicio[:-1], inicio[1:]))   # constante dentro da pessoa
    assert por_pessoa[0] == pytest.approx(por_pessoa[1]) == pytest.approx(por_pessoa[2])
    assert por_pessoa[3] == pytest.approx(por_pessoa[4])
    assert sum(por_pessoa[:3]) == pytest.approx(sum(por_pessoa[3:]))                    # classes pesam igual


def test_pesos_com_contagens_iguais_equivalem_ao_class_weight_balanced():
    from sklearn.utils.class_weight import compute_sample_weight
    rotulos = [1, 1, 1, 0, 0]
    alvo = np.repeat(rotulos, 150)
    np.testing.assert_allclose(ct.pesos_janelas([150] * 5, rotulos), compute_sample_weight("balanced", alvo))


def test_pesos_recusam_uma_classe_so():
    with pytest.raises(ValueError):
        ct.pesos_janelas([150, 150], [1, 1])


@pytest.mark.parametrize("modelo", ["LR_espectral", "RF_temporal", "RF_espectral"])
def test_os_pesos_chegam_ao_classificador(monkeypatch, modelo):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    recebidos = []
    original = ct.construir

    def espiao(*args):
        m = original(*args)
        alvo = m[-1] if hasattr(m, "steps") else m
        fit = alvo.fit
        alvo.fit = lambda X, yy, sample_weight=None: (recebidos.append(sample_weight), fit(X, yy, sample_weight=sample_weight))[1]
        return m

    monkeypatch.setattr(ct, "construir", espiao)
    ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=0)
    assert len(recebidos) == 10 and all(w is not None and w.mean() == pytest.approx(1.0) for w in recebidos)
    classificador = ct.separar(modelo)[0]
    for g in ct.GRADES[classificador]:   # o equilíbrio vem só dos pesos: nenhum class_weight somado a eles
        parametros = original(classificador, g, 0).get_params()
        assert all(v is None for k, v in parametros.items() if k.endswith("class_weight"))


def test_nove_modelos_com_nomes_proprios():
    assert ct.MODELOS == ("LR_temporal", "RF_temporal", "KNN_temporal", "LR_espectral", "RF_espectral", "KNN_espectral",
                          "LR_combinado", "RF_combinado", "KNN_combinado")
    from pipeline.e07_DTSF.redes_dtsf import TIPOS_DTSF
    historicos = {"T", "E", "F", "E_bins", "F_aux", "E_rel", "E_1_40", "baseline"}  # pastas com resultados gravados
    assert not set(ct.MODELOS) & ({*TIPOS_DTSF, *historicos})
    with pytest.raises(ValueError):
        ct.separar("XYZ_temporal")


def test_grades_pequenas_e_do_mais_regularizado_para_o_menos():
    assert all(len(g) == 3 for g in ct.GRADES.values())
    assert [g["C"] for g in ct.GRADES["LR"]] == sorted(g["C"] for g in ct.GRADES["LR"])
    folhas = [g["min_samples_leaf"] for g in ct.GRADES["RF"]]
    assert folhas == sorted(folhas, reverse=True)
    vizinhos = [g["n_neighbors"] for g in ct.GRADES["KNN"]]
    assert vizinhos == sorted(vizinhos, reverse=True)


def test_empate_na_selecao_fica_com_o_mais_regularizado(monkeypatch):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    monkeypatch.setattr(ct, "roc_auc_score", lambda *a, **k: 0.5)
    assert ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=0).parametros == dict(C=0.01)


def test_normalizacao_ajustada_so_nas_janelas_de_quem_e_ajustado(monkeypatch):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    x[indice["subject_uid"].isin(teste).to_numpy()] += 1000.0
    ajustados = []
    original = ct.construir

    def espiao(*args):
        m = original(*args)
        ajustados.append(m)
        return m

    monkeypatch.setattr(ct, "construir", espiao)
    ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=0)
    final = ajustados[-1]
    rng = np.random.default_rng(0)
    from pipeline.e07_DTSF.dados import sortear
    por_pessoa = {u: g for u, g in indice.groupby("subject_uid")}
    linhas = np.concatenate([sortear(por_pessoa[u], ct.JANELAS_POR_PESSOA, rng) for u in ajuste])
    np.testing.assert_allclose(final.named_steps["zscore"].mean_, x[linhas].mean(axis=0))
    assert all(np.abs(m.named_steps["zscore"].mean_).max() < 100 for m in ajustados)


def test_partes_internas_separam_pessoas_e_tem_as_duas_classes():
    x, indice, y, uids = _cenario(n=22)
    ajuste, _ = _papeis(uids)
    partes = ct.partes_internas(ajuste, y, semente=5)
    assert len(partes) == 3
    validacoes = [set(v) for _, v in partes]
    assert set().union(*validacoes) == set(ajuste) and sum(map(len, validacoes)) == len(ajuste)
    for a, v in partes:
        assert not set(a) & set(v) and set(a) | set(v) == set(ajuste)
        assert {y[u] for u in v} == {0, 1}


def test_pessoa_no_ajuste_e_no_teste_falha():
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    with pytest.raises(ValueError):
        ct.treinar_e_prever("LR_temporal", x, indice, ajuste + teste[:1], teste, y, semente=0)


def test_selecao_e_ajuste_nao_veem_o_teste():
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    a = ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=2)
    x2, y2 = x.copy(), y.copy()
    no_teste = indice["subject_uid"].isin(teste).to_numpy() & ~indice["rejeitada"].to_numpy()
    x2[no_teste] = np.random.default_rng(9).normal(size=(no_teste.sum(), x.shape[1]))
    y2[teste] = 1 - y2[teste]
    b = ct.treinar_e_prever("LR_temporal", x2, indice, ajuste, teste, y2, semente=2)
    assert a.parametros == b.parametros and a.auc_interna == b.auc_interna and a.partes_internas == b.partes_internas


def _final(monkeypatch, modelo, x, indice, y, ajuste, teste):
    ajustados = []
    original = ct.construir
    monkeypatch.setattr(ct, "construir", lambda *a: ajustados.append(original(*a)) or ajustados[-1])
    return ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=0), ajustados[-1]


@pytest.mark.parametrize("modelo", ["LR_temporal", "RF_espectral"])
def test_lr_e_rf_pessoa_e_a_media_das_probabilidades_das_janelas(monkeypatch, modelo):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    r, final = _final(monkeypatch, modelo, x, indice, y, ajuste, teste)
    linhas = janelas_avaliacao(indice[indice["subject_uid"] == teste[0]])
    assert r.prob[teste[0]] == pytest.approx(final.predict_proba(x[linhas])[:, 1].mean())


def test_150_janelas_so_no_ajuste_e_todas_as_aceitas_sem_overlap_na_previsao(monkeypatch):
    x, indice, y, uids = _cenario(janelas=400)                   # ~355 aceitas e ~178 aceitas sem overlap por pessoa
    ajuste, teste = _papeis(uids)
    ajustes, previsoes = [], []
    original_ajustar, original_escore = ct._ajustar, ct.escore_pessoa

    def espiao_ajustar(classificador, parametros, semente, x_, sorteadas, uids_, y_):
        ajustes.append(len(np.concatenate([sorteadas[u] for u in uids_])) / len(uids_))
        return original_ajustar(classificador, parametros, semente, x_, sorteadas, uids_, y_)

    def espiao_escore(modelo, x_):
        previsoes.append(x_)
        return original_escore(modelo, x_)

    monkeypatch.setattr(ct, "_ajustar", espiao_ajustar)
    monkeypatch.setattr(ct, "escore_pessoa", espiao_escore)
    ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=0)
    assert ajustes == [ct.JANELAS_POR_PESSOA] * (3 * 3 + 1)      # 3 candidatos × 3 partes + reajuste final
    # 3 candidatos × todas as pessoas de ajuste (validação interna) + as pessoas de teste, nessa ordem
    assert len(previsoes) == 3 * len(ajuste) + len(teste)
    esperadas = {u: x[janelas_avaliacao(indice[indice["subject_uid"] == u])] for u in uids}
    assert min(len(v) for v in esperadas.values()) > ct.JANELAS_POR_PESSOA
    for xp, u in zip(previsoes[-len(teste):], teste):
        np.testing.assert_array_equal(xp, esperadas[u])
    validadas = previsoes[:3 * len(ajuste)]
    assert all(any(np.array_equal(xp, esperadas[u]) for u in ajuste) for xp in validadas)


def test_janela_rejeitada_nunca_entra():
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    rejeitadas = indice["rejeitada"].to_numpy()
    assert np.isnan(x[rejeitadas]).all()
    ct.treinar_e_prever("RF_temporal", x, indice, ajuste, teste, y, semente=0)  # falharia com NaN


@pytest.mark.parametrize("modelo", ["RF_espectral", "LR_temporal", "LR_espectral"])
def test_reprodutivel_com_a_mesma_semente(modelo):
    x, indice, y, uids = _cenario()
    ajuste, teste = _papeis(uids)
    a = ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=7)
    b = ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=7)
    assert a.prob == b.prob and a.parametros == b.parametros and a.auc_interna == b.auc_interna


# ---------- runner ----------

def _arquivos_runner(tmp_path):
    """Características, janelas.csv e folds.csv sintéticos com SZ e MDD."""
    from pipeline.e08_classicos import caracteristicas as cc
    tabelas, folds = [], []
    for dataset, n in (("SZ", 24), ("MDD", 30)):
        _, indice, y, uids = _cenario(n=n, semente=len(dataset))
        indice = indice.assign(dataset=dataset, subject_uid=dataset + "-" + indice["subject_uid"])
        y.index = [f"{dataset}-{u}" for u in y.index]
        tabelas.append(indice.drop(columns="linha"))
        rng = np.random.default_rng(n)
        for r in range(2):
            ordem = rng.permutation(len(y))
            for f in range(5):
                teste = set(ordem[f::5])
                validacao = set(ordem[(f + 1) % 5::5])
                for i, u in enumerate(y.index):
                    papel = "teste" if i in teste else "validacao" if i in validacao else "treino"
                    folds.append(dict(dataset=dataset, repeticao=r, fold=f, subject_uid=u,
                                      grupo="HC" if y[u] == 0 else dataset, papel=papel))
    janelas = pd.concat(tabelas, ignore_index=True)
    janelas.to_csv(tmp_path / "janelas.csv", index=False)
    pd.DataFrame(folds).to_csv(tmp_path / "folds.csv", index=False)
    rng = np.random.default_rng(0)
    paciente = janelas["subject_uid"].str.contains("-P").to_numpy()
    destino = tmp_path / "carac"
    destino.mkdir()
    for nome, p in (("temporais", cc.N_TEMPORAIS), ("espectrais", cc.N_ESPECTRAIS)):
        x = rng.normal(size=(len(janelas), p)).astype(np.float32)
        x[:, 0] += 1.5 * paciente
        x[janelas["rejeitada"].to_numpy()] = np.nan
        np.save(destino / f"{nome}.npy", x)
    (destino / "extracao.json").write_text(json.dumps(dict(janelas_csv_sha256=cc.hash_arquivo(tmp_path / "janelas.csv"))))
    return destino


@pytest.mark.parametrize("dataset", ["SZ", "MDD"])
def test_rodar_tarefa_nas_duas_tarefas_grava_csv_e_json(tmp_path, dataset):
    carac = _arquivos_runner(tmp_path)
    cr._DADOS.clear()
    saida = tmp_path / "res"
    cr.rodar_tarefa(dataset, "LR_espectral", 1, 3, str(tmp_path / "folds.csv"), str(saida),
                    caracteristicas=str(carac), janelas_csv=str(tmp_path / "janelas.csv"))
    pred = pd.read_csv(saida / dataset / "LR_espectral" / "r1_f3.csv")
    folds = pd.read_csv(tmp_path / "folds.csv")
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == 1) & (folds["fold"] == 3)]
    assert sorted(pred["subject_uid"]) == sorted(g.loc[g["papel"] == "teste", "subject_uid"])
    assert list(pred.columns) == ["repeticao", "fold", "subject_uid", "y", "prob"]
    assert (pred["y"] == pred["subject_uid"].map(g.set_index("subject_uid")["grupo"] != "HC").astype(int)).all()
    meta = json.loads((saida / dataset / "LR_espectral" / "r1_f3.json").read_text())
    assert meta["parametros"] in ct.GRADES["LR"] and len(meta["auc_interna"]) == 3
    internas = {u for parte in meta["partes_internas"] for u in parte}
    assert internas == set(g.loc[g["papel"] != "teste", "subject_uid"])   # treino + validação, nunca teste


def test_combinado_concatena_temporais_e_espectrais(tmp_path):
    carac = _arquivos_runner(tmp_path)
    cr._DADOS.clear()
    x, _ = cr._dados(str(carac), str(tmp_path / "janelas.csv"))
    assert x["combinado"].shape[1] == x["temporal"].shape[1] + x["espectral"].shape[1]
    np.testing.assert_array_equal(x["combinado"], np.concatenate([x["temporal"], x["espectral"]], axis=1))
    assert ct.separar("RF_combinado") == ("RF", "combinado")


def test_rodar_com_permutacao_grava_arquivo_p(tmp_path):
    carac = _arquivos_runner(tmp_path)
    cr._DADOS.clear()
    saida = tmp_path / "perm"
    cr.rodar_tarefa("SZ", "RF_temporal", 0, 0, str(tmp_path / "folds.csv"), str(saida), permutacao=0,
                    caracteristicas=str(carac), janelas_csv=str(tmp_path / "janelas.csv"))
    assert (saida / "SZ" / "RF_temporal" / "p0_f0.csv").exists()
    assert not (saida / "SZ" / "RF_temporal" / "r0_f0.csv").exists()


def test_saidas_padrao_separadas_dos_resultados_historicos():
    from pathlib import Path
    assert cr.SAIDA == Path("outputs/08_classicos/resultados")
    assert cr.SAIDA_PERMUTACOES == Path("outputs/08_classicos/permutacoes")
    assert cr.CARACTERISTICAS == Path("outputs/08_classicos/caracteristicas")


# ---------- verificação dos pesos e da comparabilidade ----------

def _cenario_desigual():
    """Pessoas com quantidades diferentes de janelas aceitas (algumas abaixo de 150), como no MDD."""
    x, indice, y, uids = _cenario(janelas=360)
    rng = np.random.default_rng(11)
    for u in uids[::3]:                                        # 1/3 das pessoas: só ~100 janelas aceitas
        linhas = indice.index[indice["subject_uid"] == u]
        cortar = rng.choice(linhas[~indice.loc[linhas, "rejeitada"]], 220, replace=False)
        indice.loc[cortar, "rejeitada"] = True
        x[cortar] = np.nan
    return x, indice, y, uids


@pytest.mark.parametrize("classificador", [c for c in ct.CLASSIFICADORES if c not in ct.SEM_PESOS])
def test_peso_muda_o_ajuste_e_e_o_mesmo_de_um_ajuste_feito_a_mao(classificador):
    from sklearn.pipeline import Pipeline
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    alvo = (X[:, 0] + rng.normal(size=200) > 0).astype(int)
    w = rng.uniform(0.2, 3, 200)
    g = ct.GRADES[classificador][1]
    m = ct.construir(classificador, g, 0)
    m.fit(X, alvo, **{"clf__sample_weight" if isinstance(m, Pipeline) else "sample_weight": w})
    ref = ct.construir(classificador, g, 0)
    if isinstance(ref, Pipeline):
        ref[-1].fit(ref[0].fit_transform(X), alvo, sample_weight=w)
    else:
        ref.fit(X, alvo, sample_weight=w)
    sem = ct.construir(classificador, g, 0).fit(X, alvo)
    saida = lambda mm: mm.predict_proba(X)[:, 1]
    np.testing.assert_allclose(saida(m), saida(ref))
    assert not np.allclose(saida(m), saida(sem))


@pytest.mark.parametrize("modelo", ["LR_temporal", "LR_espectral", "RF_temporal"])
def test_cada_ajuste_da_selecao_e_o_final_recebem_os_pesos_das_suas_proprias_pessoas(monkeypatch, modelo):
    x, indice, y, uids = _cenario_desigual()
    ajuste, teste = _papeis(uids)
    chamadas, original = [], ct._ajustar

    def espiao(classificador, parametros, semente, x_, sorteadas, uids_, y_):
        esperado = ct.pesos_janelas([len(sorteadas[u]) for u in uids_], [int(y_[u]) for u in uids_])
        m = original(classificador, parametros, semente, x_, sorteadas, uids_, y_)
        chamadas.append((list(uids_), esperado, {u: len(sorteadas[u]) for u in uids_}))
        return m

    recebidos, construir = [], ct.construir

    def espiao_construir(*args):
        m = construir(*args)
        alvo = m[-1] if hasattr(m, "steps") else m
        fit = alvo.fit
        alvo.fit = lambda X, yy, sample_weight=None: (recebidos.append(sample_weight), fit(X, yy, sample_weight=sample_weight))[1]
        return m

    monkeypatch.setattr(ct, "_ajustar", espiao)
    monkeypatch.setattr(ct, "construir", espiao_construir)
    r = ct.treinar_e_prever(modelo, x, indice, ajuste, teste, y, semente=3)
    assert len(chamadas) == len(recebidos) == 10                 # 9 da seleção interna + o final
    partes = ct.partes_internas(ajuste, y, 3)
    for (uids_, esperado, contagens), recebido in zip(chamadas, recebidos):
        np.testing.assert_allclose(recebido, esperado)
        assert not set(uids_) & set(teste)
        assert len(set(contagens.values())) > 1                  # contagens desiguais: os pesos importam
        inicio = np.cumsum([0, *contagens.values()])
        soma = {u: recebido[a:b].sum() for u, a, b in zip(uids_, inicio[:-1], inicio[1:])}
        por_classe = {c: [soma[u] for u in uids_ if y[u] == c] for c in (0, 1)}
        assert all(np.allclose(v, v[0]) for v in por_classe.values())      # pessoas da mesma classe pesam igual
        assert sum(por_classe[0]) == pytest.approx(sum(por_classe[1]))     # as duas classes pesam igual
        assert recebido.mean() == pytest.approx(1.0)
    assert [c[0] for c in chamadas[:3]] == [a for a, _ in partes]   # 1º candidato: as 3 partes internas
    assert chamadas[-1][0] == ajuste


def test_janelas_ajustadas_sao_so_das_pessoas_do_ajuste_e_aceitas(monkeypatch):
    x, indice, y, uids = _cenario_desigual()
    ajuste, teste = _papeis(uids)
    vistas, original = [], ct._ajustar

    def espiao(classificador, parametros, semente, x_, sorteadas, uids_, y_):
        vistas.append((set(uids_), np.concatenate([sorteadas[u] for u in uids_])))
        return original(classificador, parametros, semente, x_, sorteadas, uids_, y_)

    monkeypatch.setattr(ct, "_ajustar", espiao)
    ct.treinar_e_prever("LR_espectral", x, indice, ajuste, teste, y, semente=0)
    pessoa = indice.set_index("linha")["subject_uid"]
    for uids_, linhas in vistas:
        assert set(pessoa[linhas]) == uids_ and not indice.loc[linhas, "rejeitada"].any()
        assert len(linhas) == len(set(linhas))                   # sem repetição
    for (uids_, linhas), (_, validacao) in zip(vistas[:3], ct.partes_internas(ajuste, y, 0)):
        assert not set(pessoa[linhas]) & set(validacao)


def test_janelas_de_previsao_nao_dependem_do_numero_sorteado_para_o_ajuste(monkeypatch):
    x, indice, y, uids = _cenario_desigual()
    ajuste, teste = _papeis(uids)
    entradas = {}
    for k in (60, 150):
        monkeypatch.setattr(ct, "JANELAS_POR_PESSOA", k)
        vistas, original = [], ct.escore_pessoa
        monkeypatch.setattr(ct, "escore_pessoa", lambda m, x_, vistas=vistas, original=original: vistas.append(x_) or original(m, x_))
        ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=0)
        monkeypatch.setattr(ct, "escore_pessoa", original)
        entradas[k] = vistas
    assert len(entradas[60]) == len(entradas[150])
    for a, b in zip(entradas[60], entradas[150]):
        np.testing.assert_array_equal(a, b)


def test_previsao_de_uma_pessoa_de_teste_nao_depende_das_outras_nem_dos_rotulos_do_teste():
    x, indice, y, uids = _cenario_desigual()
    ajuste, teste = _papeis(uids)
    a = ct.treinar_e_prever("LR_temporal", x, indice, ajuste, teste, y, semente=1)
    x2, y2 = x.copy(), y.copy()
    outras = indice["subject_uid"].isin(teste[1:]).to_numpy() & ~indice["rejeitada"].to_numpy()
    x2[outras] += 5.0
    y2[teste] = 1 - y2[teste]
    b = ct.treinar_e_prever("LR_temporal", x2, indice, ajuste, teste, y2, semente=1)
    assert a.prob[teste[0]] == b.prob[teste[0]] and a.parametros == b.parametros


def test_knn_e_ajustado_sem_pesos(monkeypatch):
    """O KNN não aceita sample_weight: todo ajuste dele (seleção e final) é feito sem pesos."""
    x, indice, y, uids = _cenario_desigual()
    ajuste, teste = _papeis(uids)
    recebidos = []
    original = ct.construir

    def espiao(*args):
        m = original(*args)
        fit = m.fit
        m.fit = lambda X, yy, **kw: (recebidos.append(kw), fit(X, yy, **kw))[1]
        return m

    monkeypatch.setattr(ct, "construir", espiao)
    r = ct.treinar_e_prever("KNN_espectral", x, indice, ajuste, teste, y, semente=0)
    assert len(recebidos) == 10 and all(kw == {} for kw in recebidos)
    assert set(r.prob) == set(teste) and r.parametros in ct.GRADES["KNN"]
