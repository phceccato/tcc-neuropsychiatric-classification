"""Nomes dos 17 modelos e a leitura de rodadas anteriores à renomeação (pipeline/nomes.py)."""
import json

import pandas as pd
import pytest

from pipeline import nomes
from pipeline.e07_DTSF import avaliar as av, epoca_um as eu, rodar
from pipeline.e07_DTSF.redes_dtsf import MODELOS, PRINCIPAL, TIPOS_DTSF
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS
from pipeline.e10_resultados import exportar as ex


def test_os_17_modelos_com_os_nomes_novos():
    assert TIPOS_DTSF == ("DTSF_FA", "DTSF_Lite", "DTSF", "DTSF_Lite_NoMixup", "DTSF_Lite_SpectralOnly", "DTSF_Lite_TemporalOnly")
    assert PRINCIPAL == "DTSF_FA"
    assert len(CLASSICOS) == 9 and set(CLASSICOS) == {f"{c}_{m}" for c in ("LR", "RF", "KNN")
                                                      for m in ("Temporal", "Spectral", "Combined")}
    assert list(nomes.LR_PSD.values()) == ["LR_PSD_Absolute", "LR_PSD_Relative"]
    assert len(set(TIPOS_DTSF) | set(CLASSICOS) | set(nomes.LR_PSD.values())) == 17


def test_ablacoes_batem_com_o_que_cada_nome_diz():
    """SpectralOnly = sem o ramo temporal (D_sem_T); TemporalOnly = sem o ramo espectral (D_sem_E)."""
    assert not MODELOS["DTSF_Lite_SpectralOnly"].temporal and MODELOS["DTSF_Lite_SpectralOnly"].espectral
    assert not MODELOS["DTSF_Lite_TemporalOnly"].espectral and MODELOS["DTSF_Lite_TemporalOnly"].temporal
    assert not MODELOS["DTSF_Lite_NoMixup"].mixup and MODELOS["DTSF_FA"].fa and not MODELOS["DTSF"].fa


def test_mapa_de_nomes_antigos_para_os_novos():
    assert nomes.nome_novo("D_FA") == "DTSF_FA" and nomes.nome_novo("D") == "DTSF"
    assert nomes.nome_novo("D_sem_T") == "DTSF_Lite_SpectralOnly" and nomes.nome_novo("D_sem_E") == "DTSF_Lite_TemporalOnly"
    assert nomes.nome_novo("LR_psd_absolutas") == "LR_PSD_Absolute"
    assert nomes.nome_novo("LR_espectral") == "LR_Spectral" and nomes.nome_novo("RF_combinado") == "RF_Combined"
    assert nomes.nome_novo("KNN_temporal") == "KNN_Temporal"
    for nome in (*TIPOS_DTSF, *CLASSICOS, *nomes.LR_PSD.values()):
        assert nomes.nome_novo(nome) == nome                  # idempotente: os 17 nomes novos ficam como estão
    assert len(nomes.NOVOS) == 8 + 9                          # antigos conhecidos: 8 (07 e 06) + 9 clássicos
    assert nomes.nome_classico("LR", "espectral") == "LR_Spectral"


def test_resolver_pasta_prefere_a_nova_e_cai_na_antiga(tmp_path):
    antiga = tmp_path / "SZ" / "D_FA"
    antiga.mkdir(parents=True)
    assert nomes.resolver_pasta(tmp_path / "SZ" / "DTSF_FA") == antiga
    (tmp_path / "SZ" / "DTSF_FA").mkdir()
    assert nomes.resolver_pasta(tmp_path / "SZ" / "DTSF_FA") == tmp_path / "SZ" / "DTSF_FA"
    assert nomes.resolver_pasta(tmp_path / "SZ" / "LR_temporal") == tmp_path / "SZ" / "LR_temporal"   # clássico: nada muda


def _gravar(pasta, rep=0, fold=0, epoca=3):
    pasta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(dict(repeticao=rep, fold=fold, subject_uid=["a", "b"], y=[0, 1], prob=[0.2, 0.8])).to_csv(
        pasta / f"r{rep}_f{fold}.csv", index=False)
    (pasta / f"r{rep}_f{fold}.json").write_text(json.dumps(dict(melhor_epoca=epoca, dispositivo="cuda")))


def test_ler_modelo_le_a_pasta_com_nome_antigo(tmp_path):
    _gravar(tmp_path / "SZ" / "D_pequeno")
    p = av.ler_modelo(tmp_path / "SZ" / "DTSF_Lite")
    assert len(p) == 2 and av._epocas(tmp_path / "SZ" / "DTSF_Lite") == "3 (3–3)"


def test_runner_nao_retreina_o_que_ja_existe_na_pasta_antiga(tmp_path):
    _gravar(tmp_path / "SZ" / "D_FA")
    pasta = rodar.resolver_pasta(tmp_path / "SZ" / "DTSF_FA")
    assert (pasta / "r0_f0.csv").exists() and pasta.name == "D_FA"
    assert rodar.caminho_checkpoint(tmp_path / "pesos", "SZ", "DTSF_FA", 0, 0).parent.name == "DTSF_FA"   # sem pasta antiga de pesos
    (tmp_path / "pesos" / "SZ" / "D_FA").mkdir(parents=True)                                              # com pesos de uma rodada antiga
    assert rodar.caminho_checkpoint(tmp_path / "pesos", "SZ", "DTSF_FA", 0, 0).parent.name == "D_FA"


def test_resumo_com_nomes_antigos_e_lido_com_nomes_novos():
    md = "## SZ\n\n| D_FA | 704.769 | 0.852 [0.684, 0.969] | x |\n| D | 702.721 | 0.837 [0.658, 0.969] | x |\n"
    assert ex.aucs_do_resumo(md) == {("SZ", "DTSF_FA"): 0.852, ("SZ", "DTSF"): 0.837}


def test_epoca_um_aceita_exportacao_com_nome_antigo(tmp_path):
    (tmp_path / "perdas" / "SZ").mkdir(parents=True)
    (tmp_path / "previsoes" / "SZ").mkdir(parents=True)
    pd.DataFrame(dict(repeticao=0, fold=0, epoca=[1, 2], perda_treino=.5, perda_validacao=.5, melhor=[1, 0])).to_csv(
        tmp_path / "perdas" / "SZ" / "D_FA.csv", index=False)
    pd.DataFrame(dict(repeticao=0, fold=0, subject_uid=["a", "b"], y_true=[0, 1], y_pred=[0, 1], y_prob=[.1, .9])).to_csv(
        tmp_path / "previsoes" / "SZ" / "D_FA.csv", index=False)
    folds, previsoes = eu.carregar_exportados(tmp_path, "DTSF_FA", "cuda")
    assert folds.iloc[0]["modelo"] == "DTSF_FA" and folds.iloc[0]["melhor_epoca"] == 1 and len(previsoes) == 2


def test_classicos_separar_aceita_nome_novo_e_antigo():
    from pipeline.e08_classicos.treino import separar
    assert separar("RF_Spectral") == ("RF", "espectral") and separar("RF_espectral") == ("RF", "espectral")
    assert separar("KNN_Combined") == ("KNN", "combinado")


def test_runner_dos_classicos_retoma_pasta_com_nome_antigo(tmp_path):
    from pipeline.e08_classicos import rodar as cr
    _gravar(tmp_path / "MDD" / "LR_temporal")
    assert cr.resolver_pasta(tmp_path / "MDD" / "LR_Temporal").name == "LR_temporal"
    assert av.ler_modelo(tmp_path / "MDD" / "LR_Temporal")["subject_uid"].tolist() == ["a", "b"]


def test_resumo_dos_classicos_com_nomes_antigos_e_lido_com_nomes_novos():
    md = "## MDD\n\n| LR_temporal | 0.884 [0.785, 0.962] | x |\n| RF_combinado | 0.903 [0.814, 0.974] | x |\n"
    assert ex.aucs_do_resumo(md) == {("MDD", "LR_Temporal"): 0.884, ("MDD", "RF_Combined"): 0.903}
