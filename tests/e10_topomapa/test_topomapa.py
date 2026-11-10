from itertools import combinations
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
from mne.viz.topomap import _get_pos_outlines

from pipeline.e01_auditoria.auditar_ec import CANAIS
from pipeline.e02_harmonizacao.harmonizar import MONTAGEM
from pipeline.e04_features.psd import BANDAS, nomes_features
from pipeline.e10_topomapa import topomapa as t


def _psd_sintetico(tmp_path: Path) -> tuple[Path, Path]:
    """Duas pessoas com potência constante e diferente (10 µV² e 1000 µV²); uma janela rejeitada."""
    uids = ["SZ-HC-a"] * 3 + ["SZ-SZ-b"] * 2
    rejeitada = [False, False, True, False, False]
    absolutas = np.array([[1.0] * 76, [1.0] * 76, [9.0] * 76, [3.0] * 76, [3.0] * 76], dtype=np.float32)
    psd = tmp_path / "psd.npz"
    np.savez(psd, absolutas=absolutas, nomes=np.array(nomes_features()), subject_uid=np.array(uids))
    janelas = tmp_path / "janelas.csv"
    pd.DataFrame(dict(subject_uid=uids, rejeitada=rejeitada)).to_csv(janelas, index=False)
    return psd, janelas


def test_so_janelas_aceitas_da_pessoa_escolhida(tmp_path):
    psd, janelas = _psd_sintetico(tmp_path)
    x = t.janelas_da_pessoa("SZ-HC-a", psd, janelas)
    assert x.shape == (2, 19, 4) and np.all(x == 1.0)  # sem a janela rejeitada (9) e sem a outra pessoa (3)
    with pytest.raises(ValueError):
        t.janelas_da_pessoa("SZ-HC-zz", psd, janelas)


def test_agregar_faz_a_media_na_potencia_linear():
    x = np.zeros((2, 19, 4))
    x[1] = 2.0  # 1 µV² e 100 µV²
    np.testing.assert_allclose(t.agregar(x, "absoluta"), 50.5)
    np.testing.assert_allclose(t.agregar(x, "log"), np.log10(50.5))
    rel = t.agregar(np.log10(np.tile([1.0, 1.0, 2.0, 6.0], (3, 19, 1))), "relativa")
    np.testing.assert_allclose(rel.sum(axis=1), 100.0)
    np.testing.assert_allclose(rel[0], [10, 10, 20, 60])
    with pytest.raises(ValueError):
        t.agregar(x, "dB")


def test_posicoes_sao_a_montagem_sem_deformacao():
    original = mne.channels.make_standard_montage(MONTAGEM).get_positions()["ch_pos"]
    niveladas = t.posicoes_niveladas()
    for a, b in combinations(CANAIS, 2):  # rotação + translação: distâncias iguais
        assert np.linalg.norm(niveladas[a] - niveladas[b]) == pytest.approx(np.linalg.norm(original[a] - original[b]))
    assert niveladas["Fpz"][2] == pytest.approx(niveladas["Oz"][2])
    assert niveladas["F8"][0] > 0 and niveladas["F7"][0] < 0 and niveladas["Fp1"][1] > 0 > niveladas["O1"][1]


def test_contorno_envolve_todos_os_eletrodos_e_o_mapa_termina_nele():
    info, esfera = t.info_10_20()
    assert esfera[:3] == (0.0, 0.0, 0.0)
    pos, contornos = _get_pos_outlines(info, None, sphere=esfera)
    assert np.linalg.norm(pos, axis=1).max() < esfera[3]
    assert contornos["clip_radius"][0] == pytest.approx(esfera[3])
    assert np.linalg.norm(pos[CANAIS.index("Cz")]) < 0.05 * esfera[3]


@pytest.mark.skipif(not Path("outputs/04_features/psd.npz").exists(), reason="precisa de outputs/04_features")
def test_figura_real_png_300_dpi(tmp_path):
    from PIL import Image
    x = t.janelas_da_pessoa("MDD-MDD-S27", Path("outputs/04_features/psd.npz"), Path("outputs/03_janelas/janelas.csv"))
    destino = t.figura(t.agregar(x, "log"), "log", "teste", tmp_path / "fig.png")
    dpi = Image.open(destino).info["dpi"]
    assert round(dpi[0]) == 300 and len(BANDAS) == 4


def test_nome_sessao_e_plano_de_pastas():
    from pipeline.e10_topomapa.todos import nome_sessao
    assert nome_sessao(1, pd.Series(["2003-03-11"])) == "sessao_1_mar2003"
    assert nome_sessao(2, pd.Series(["2003-06-18", "2003-07-25"])) == "sessao_2_jun-jul2003"
    assert nome_sessao(4, pd.Series(["2005-12-20", "2006-01-05"])) == "sessao_4_dez2005-jan2006"


@pytest.mark.skipif(not Path("outputs/02_harmonizacao/manifesto.csv").exists(), reason="precisa do manifesto")
def test_plano_tem_uma_pasta_por_pessoa_e_contagens_da_etapa_09(tmp_path):
    from pipeline.e10_topomapa.todos import plano
    p = plano(Path("outputs/02_harmonizacao/manifesto.csv"), tmp_path, "log")
    assert len(p) == 84 and p["subject_uid"].is_unique and p["arquivo"].is_unique
    contagem = p.groupby(["dataset", "classe", "sessao"]).size()
    assert contagem[("MDD", "HC", "sessao_2_jan2013")] == 14 and contagem[("MDD", "MDD", "sessao_3_abr-jul2013")] == 29
    assert contagem[("SZ", "HC", "sessao_2_jun-jul2003")] == 14
    assert all(str(a).startswith(str(tmp_path / ("Schizophrenia_dataset" if d == "SZ" else "Depression_dataset")))
               for a, d in zip(p["arquivo"], p["dataset"]))


def test_limites_comuns_cobrem_todas_as_pessoas():
    valores = {"a": np.full((19, 4), 1.0), "b": np.full((19, 4), 3.0)}
    valores["b"][0, 2] = 5.0
    lim = t.limites_comuns(valores)
    np.testing.assert_allclose(lim[:, 0], 1.0)
    np.testing.assert_allclose(lim[:, 1], [3.0, 3.0, 5.0, 3.0])


def test_media_por_pasta_nao_mistura_classes_nem_sessoes():
    from pipeline.e10_topomapa.todos import medias_por_pasta
    tabela = pd.DataFrame(dict(subject_uid=["a", "b", "c", "d"], dataset="MDD", classe=["HC", "HC", "MDD", "HC"],
                               sessao=["s3", "s3", "s3", "s1"],
                               arquivo=[Path(f"x/{c}/{s}/f.png") for c, s in (("HC", "s3"), ("HC", "s3"), ("MDD", "s3"), ("HC", "s1"))]))
    valores = {u: np.full((19, 4), float(i)) for i, u in enumerate("abcd")}
    m = {(r["classe"], r["sessao"]): r for r in medias_por_pasta(tabela, valores)}
    assert set(m) == {("HC", "s3"), ("MDD", "s3"), ("HC", "s1")}
    np.testing.assert_allclose(m[("HC", "s3")]["valores"], 0.5)
    assert m[("HC", "s3")]["n"] == 2 and m[("HC", "s3")]["pasta"] == Path("x/HC/s3")


@pytest.mark.skipif(not Path("outputs/02_harmonizacao/manifesto.csv").exists(), reason="precisa do manifesto")
def test_subpasta_por_escala(tmp_path):
    from pipeline.e10_topomapa.todos import plano
    for escala, sub in (("log", "absoluta"), ("absoluta", "absoluta"), ("relativa", "relativa")):
        p = plano(Path("outputs/02_harmonizacao/manifesto.csv"), tmp_path, escala)
        assert all(Path(a).parent.name == sub and Path(a).name.endswith(f"_{escala}.png") for a in p["arquivo"])
