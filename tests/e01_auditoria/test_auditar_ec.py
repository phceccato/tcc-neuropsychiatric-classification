from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.e01_auditoria import auditar_ec as a

DADOS = Path(__file__).resolve().parents[2] / "dados"


def test_normalizar_canal_aceita_os_dois_formatos():
    assert a.normalizar_canal("EEG Fp1-LE") == "Fp1"
    assert a.normalizar_canal("FP2") == "Fp2"
    assert a.normalizar_canal("Cz") == "Cz"
    assert a.normalizar_canal("EEG A2-A1") is None
    assert a.normalizar_canal("EEG 23A-23R") is None


def test_identificar_arquivo():
    assert a.identificar("SZ", "h03.edf") == ("HC", "h03")
    assert a.identificar("SZ", "s14.edf") == ("SZ", "s14")
    assert a.identificar("MDD", "MDD S11  EC.edf") == ("MDD", "S11")
    assert a.identificar("MDD", "H S2 EC.edf") == ("HC", "S2")


def test_zeros_no_fim_conta_so_amostras_zeradas_em_todos_os_canais():
    x = np.ones((3, 100), dtype=np.int16)
    x[:, 90:] = 0
    x[0, 85:90] = 0  # zero em um canal só não conta
    assert a.zeros_no_fim(x) == 10
    assert a.zeros_no_fim(np.ones((3, 5), dtype=np.int16)) == 0


def test_segmentos_planos_detecta_trecho_de_pelo_menos_10_s():
    fs = 100
    rng = np.random.default_rng(0)
    x = rng.normal(0, 20, 60 * fs)
    x[20 * fs:35 * fs] = 3.0  # 15 s constante
    x[45 * fs:50 * fs] = 3.0  # 5 s: curto demais
    assert a.segmentos_planos(x, fs) == [(20.0, 35.0)]


def test_fracao_saturada():
    d = np.array([0, 32767, 5, -32768, 7, 32767], dtype=np.int16)
    assert a.fracao_saturada(d, -32768, 32767) == pytest.approx(0.5)


def _registro(arquivo, grupo, sujeito, **extra):
    base = dict(dataset="MDD", arquivo=arquivo, grupo=grupo, sujeito=sujeito,
                erro="", duracao_util_s=300.0, canais_ruins="",
                sha256=arquivo, sinal_sha256=arquivo, paciente_hash=arquivo,
                condicao_cabecalho="EC")
    base.update(extra)
    return base


def test_decidir_regras_de_duracao_e_canais():
    df = pd.DataFrame([
        _registro("ok", "HC", "S1"),
        _registro("curta", "HC", "S2", duracao_util_s=100.0),
        _registro("um_canal", "MDD", "S3", canais_ruins="C3"),
        _registro("dois_canais", "MDD", "S4", canais_ruins="C3;O2"),
        _registro("ilegivel", "MDD", "S5", erro="sem canal Pz"),
    ])
    d = a.decidir(df).set_index("arquivo")["decisao"]
    assert d["ok"] == "INCLUIR"
    assert d["curta"] == "EXCLUIR"
    assert d["um_canal"] == "SINALIZAR"
    assert d["dois_canais"] == "EXCLUIR"
    assert d["ilegivel"] == "EXCLUIR"


def test_decidir_duplicata_com_mesmo_rotulo_vira_uma_pessoa():
    df = pd.DataFrame([
        _registro("H S27 EC.edf", "HC", "S27", sinal_sha256="x"),
        _registro("H S30 EC.edf", "HC", "S30", sinal_sha256="x"),
    ])
    r = a.decidir(df).set_index("arquivo")
    assert r.loc["H S27 EC.edf", "decisao"] == "SINALIZAR"
    assert r.loc["H S30 EC.edf", "decisao"] == "EXCLUIR"
    assert r.loc["H S30 EC.edf", "subject_uid"] == r.loc["H S27 EC.edf", "subject_uid"]


def test_decidir_duplicata_com_rotulos_diferentes_exclui_as_duas():
    df = pd.DataFrame([
        _registro("H S1 EC.edf", "HC", "S1", sha256="y"),
        _registro("MDD S1 EC.edf", "MDD", "S1", sha256="y"),
    ])
    assert set(a.decidir(df)["decisao"]) == {"EXCLUIR"}


def test_decidir_cabecalho_repetido_ou_de_outra_condicao_so_sinaliza():
    df = pd.DataFrame([
        _registro("H S1 EC.edf", "HC", "S1", paciente_hash="p"),
        _registro("H S2 EC.edf", "HC", "S2", paciente_hash="p"),
        _registro("H S3 EC.edf", "HC", "S3", condicao_cabecalho="EO"),
    ])
    assert set(a.decidir(df)["decisao"]) == {"SINALIZAR"}


@pytest.mark.skipif(not (DADOS / "Schizophrenia_dataset/h01.edf").exists(), reason="sem dados")
def test_ler_edf_bate_com_o_mne():
    import mne

    caminho = DADOS / "Schizophrenia_dataset/h01.edf"
    edf = a.ler_edf(caminho)
    raw = mne.io.read_raw_edf(caminho, preload=True, verbose="ERROR")
    i = raw.ch_names.index("Cz")
    j = edf.rotulos.index("Cz")
    np.testing.assert_allclose(edf.microvolts(j), raw.get_data()[i] * 1e6, atol=1e-3)
    assert edf.fs[j] == raw.info["sfreq"]
