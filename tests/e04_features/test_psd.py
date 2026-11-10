import numpy as np
import pytest

from pipeline.e04_features import psd as p

FS = 250.0


def _senoide(freq, amplitude, n=1000):
    return amplitude * np.sin(2 * np.pi * freq * np.arange(n) / FS)


def test_welch_devolve_bins_de_meio_hz_entre_2_e_30():
    janelas = np.random.default_rng(0).normal(0, 1, (5, 19, 1000))
    freqs, espectro = p.welch_janelas(janelas, FS)
    np.testing.assert_allclose(freqs, np.arange(2.0, 30.01, 0.5))
    assert espectro.shape == (5, 19, 57)


def test_potencia_de_banda_recupera_a_variancia_de_uma_senoide():
    janelas = np.zeros((1, 19, 1000))
    janelas[0, 0] = _senoide(10.0, 20.0)  # variância = 20² / 2 = 200 µV²
    freqs, espectro = p.welch_janelas(janelas, FS)
    potencia = p.potencia_bandas(espectro, freqs)
    assert potencia.shape == (1, 19, 4)
    assert potencia[0, 0, p.BANDAS.index("alfa")] == pytest.approx(200.0, rel=0.05)
    assert potencia[0, 0, p.BANDAS.index("teta")] < 2.0


def test_bandas_sao_semiabertas_e_beta_inclui_30_hz():
    freqs = np.arange(2.0, 30.01, 0.5)
    espectro = np.ones((1, 1, len(freqs)))
    potencia = p.potencia_bandas(espectro, freqs)[0, 0]
    # delta 2–3,5 (4 bins), teta 4–7,5 (8), alfa 8–12,5 (10), beta 13–30 (35); bin de 0,5 Hz
    np.testing.assert_allclose(potencia, [2.0, 4.0, 5.0, 17.5])


def test_features_absolutas_em_log10_na_ordem_canal_banda():
    potencia = np.arange(1, 1 + 2 * 19 * 4, dtype=float).reshape(2, 19, 4)
    x = p.features_absolutas(potencia)
    assert x.shape == (2, 76)
    np.testing.assert_allclose(x[1], np.log10(potencia[1].reshape(-1)))
    nomes = p.nomes_features()
    assert len(nomes) == 76 and nomes[:5] == ["Fp1_delta", "Fp1_teta", "Fp1_alfa", "Fp1_beta", "Fp2_delta"]


def test_features_relativas_somam_um_por_canal():
    potencia = np.random.default_rng(1).uniform(1, 10, (3, 19, 4))
    x = p.features_relativas(potencia)
    assert x.shape == (3, 76)
    np.testing.assert_allclose(x.reshape(3, 19, 4).sum(axis=2), 1.0)


def test_cortar_janelas_pelo_indice():
    x = np.arange(2 * 3000, dtype=np.float32).reshape(2, 3000)
    janelas = p.cortar(x, np.array([0, 500]), 1000)
    assert janelas.shape == (2, 2, 1000)
    np.testing.assert_array_equal(janelas[1, 0], x[0, 500:1500])

