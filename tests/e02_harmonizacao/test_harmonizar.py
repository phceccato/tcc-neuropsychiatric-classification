from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.signal import welch

from pipeline.e01_auditoria.auditar_ec import CANAIS
from pipeline.e02_harmonizacao import harmonizar as h

DADOS = Path(__file__).resolve().parents[2] / "dados"


def _senoide(freq, fs, segundos, amplitude=10.0):
    t = np.arange(int(segundos * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def _potencia(x, fs, freq):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs))
    return p[np.argmin(np.abs(f - freq))]


def test_selecionar_descarta_so_as_exclusoes():
    df = pd.DataFrame({"arquivo": ["a", "b", "c"],
                       "decisao": ["INCLUIR", "SINALIZAR", "EXCLUIR"]})
    assert list(h.selecionar(df)["arquivo"]) == ["a", "b"]


def test_harmonizar_reamostra_256_para_250_hz():
    x = np.random.default_rng(0).normal(0, 10, (19, 256 * 60))
    y = h.harmonizar(x, 256.0)
    assert y.shape == (19, 250 * 60)
    assert y.dtype == np.float32


def test_harmonizar_mantem_250_hz():
    x = np.random.default_rng(0).normal(0, 10, (19, 250 * 60))
    assert h.harmonizar(x, 250.0).shape == (19, 250 * 60)


def test_harmonizar_aplica_referencia_media():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 10, (19, 250 * 30)) + _senoide(10, 250, 30, 50)  # sinal comum a todos
    y = h.harmonizar(x, 250.0)
    np.testing.assert_allclose(y.sum(axis=0), 0, atol=1e-3)


def test_harmonizar_passa_10_hz_e_corta_fora_da_banda():
    fs = 256.0
    rng = np.random.default_rng(2)
    canal = _senoide(10, fs, 60) + _senoide(0.3, fs, 60, 50) + _senoide(50, fs, 60, 20)
    x = rng.normal(0, 1, (19, len(canal)))
    x[0] += canal  # só um canal, para a CAR não anular o sinal
    y = h.harmonizar(x, fs)[0].astype(float)
    entrada = x[0]
    for freq, minimo, maximo in [(10, 0.8, 1.0), (0.3, 0, 0.05), (50, 0, 0.01)]:
        razao = _potencia(y, 250.0, freq) / _potencia(entrada, fs, freq)
        assert minimo <= razao <= maximo, (freq, razao)


def test_interpolar_reconstroi_canal_plano():
    fs = 250.0
    rng = np.random.default_rng(3)
    comum = _senoide(10, fs, 20, 30)
    x = comum + rng.normal(0, 1, (19, len(comum)))
    c3 = CANAIS.index("C3")
    x[c3] = 0.0
    y = h.interpolar(x, fs, ["C3"])
    assert np.corrcoef(y[c3], comum)[0, 1] > 0.95
    np.testing.assert_array_equal(np.delete(y, c3, 0), np.delete(x, c3, 0))


def test_interpolar_sem_canais_ruins_nao_muda_nada():
    x = np.random.default_rng(4).normal(0, 1, (19, 500))
    np.testing.assert_array_equal(h.interpolar(x, 250.0, []), x)


def test_processar_interpola_antes_da_referencia_media():
    # C3 plano entraria na CAR e puxaria todos os canais; interpolado antes, some
    fs = 250.0
    rng = np.random.default_rng(5)
    comum = _senoide(10, fs, 30, 30)
    x = comum + rng.normal(0, 1, (19, len(comum)))
    x[CANAIS.index("C3")] = 0.0
    y = h.processar(x, fs, ["C3"])
    assert np.std(y[CANAIS.index("C3")]) < 2 * np.std(y[CANAIS.index("Cz")])
    assert np.corrcoef(y[CANAIS.index("C3")], y[CANAIS.index("Cz")])[0, 1] > 0


@pytest.mark.skipif(not (DADOS / "Schizophrenia_dataset/h01.edf").exists(), reason="sem dados")
def test_carregar_microvolts_bate_com_o_mne_e_corta_o_fim():
    import mne

    caminho = DADOS / "Schizophrenia_dataset/h01.edf"
    x = h.carregar_microvolts(caminho, n_amostras=1000)
    assert x.shape == (19, 1000)
    raw = mne.io.read_raw_edf(caminho, preload=True, verbose="ERROR")
    np.testing.assert_allclose(x[CANAIS.index("O2")],
                               raw.get_data(picks=["O2"])[0, :1000] * 1e6, atol=1e-3)


def test_data_gravacao_le_o_cabecalho_edf(tmp_path):
    cabecalho = bytearray(b" " * 256)
    cabecalho[168:176] = b"13.01.13"
    caminho = tmp_path / "x.edf"
    caminho.write_bytes(bytes(cabecalho))
    assert h.data_gravacao(caminho) == "2013-01-13"
