import numpy as np
import pytest
from scipy.signal import welch

from pipeline.e03_janelas import janelas as j


def _senoide(freq, fs, segundos, amplitude=10.0):
    t = np.arange(int(segundos * fs)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t)


def _potencia(x, fs, freq):
    f, p = welch(x, fs=fs, nperseg=int(4 * fs))
    return p[np.argmin(np.abs(f - freq))]


def test_filtrar_passa_10_hz_e_corta_1_e_38_hz():
    fs = 250.0
    x = _senoide(10, fs, 60) + _senoide(1, fs, 60) + _senoide(38, fs, 60)
    y = j.filtrar(x[None, :], fs)[0].astype(float)
    for freq, minimo, maximo in [(10, 0.8, 1.05), (1, 0, 0.05), (38, 0, 0.05)]:
        razao = _potencia(y, fs, freq) / _potencia(x, fs, freq)
        assert minimo <= razao <= maximo, (freq, razao)


def test_filtrar_devolve_float32_com_o_mesmo_formato():
    x = np.random.default_rng(0).normal(0, 1, (19, 2500))
    y = j.filtrar(x, 250.0)
    assert y.shape == x.shape and y.dtype == np.float32


def test_estatisticas_por_canal_so_nas_janelas_aceitas():
    x = np.zeros((2, 3000))
    x[0, :2000] = np.tile([1.0, 3.0], 1000)
    x[1, :2000] = 10.0
    x[:, 2500] = 5000.0  # artefato fora das janelas aceitas
    media, desvio = j.estatisticas(x, inicios_aceitos=np.array([0, 1000]), tamanho=1000)
    np.testing.assert_allclose(media, [2.0, 10.0])
    np.testing.assert_allclose(desvio, [1.0, 0.0])


def test_estatisticas_contam_amostras_de_janelas_sobrepostas_uma_vez():
    x = np.zeros((1, 1500))
    x[0, 500:1000] = 4.0
    media, _ = j.estatisticas(x, inicios_aceitos=np.array([0, 500]), tamanho=1000)
    np.testing.assert_allclose(media, [4.0 * 500 / 1500])


def test_inicios_com_50_por_cento_e_subconjunto_sem_overlap():
    # 11 s a 250 Hz: janelas de 4 s a cada 2 s começam em 0, 2, 4, 6 s
    inicios, sem_overlap = j.inicios_janelas(11 * 250, 250.0)
    np.testing.assert_array_equal(inicios, [0, 500, 1000, 1500])
    np.testing.assert_array_equal(sem_overlap, [True, False, True, False])


def test_inicios_nao_passam_do_fim():
    inicios, _ = j.inicios_janelas(4 * 250 - 1, 250.0)
    assert len(inicios) == 0


def test_pico_a_pico_pega_o_maior_entre_os_canais():
    x = np.zeros((3, 2000))
    x[1, 100] = 80.0
    x[1, 200] = -90.0       # 170 µV no canal 1, primeira janela
    x[2, 1500] = 20.0       # 20 µV no canal 2, segunda janela
    pp = j.pico_a_pico(x, np.array([0, 1000]), 1000)
    np.testing.assert_allclose(pp, [170.0, 20.0])


def test_janelas_que_tocam_trecho_plano():
    fs = 250.0
    inicios = np.array([0, 500, 1000, 1500, 2000])          # 0, 2, 4, 6, 8 s
    toca = j.toca_trechos(inicios, 1000, fs, ["O1:5-7"])     # trecho de 5 a 7 s
    np.testing.assert_array_equal(toca, [False, True, True, True, False])


def test_janelas_sem_trechos():
    assert not j.toca_trechos(np.array([0, 500]), 1000, 250.0, []).any()


def test_indexar_marca_rejeicao_por_amplitude():
    fs = 250.0
    rng = np.random.default_rng(1)
    x = rng.normal(0, 5, (19, 20 * 250)).astype(np.float32)
    x[3, 2 * 250] = 400.0  # artefato entre 2 e 3 s: toca as janelas que começam em 0 e 2 s
    tabela = j.indexar(x, fs, trechos=[])
    assert list(tabela["inicio_s"][:3]) == [0.0, 2.0, 4.0]
    assert list(tabela["rejeitada"][:3]) == [True, True, False]
    assert (tabela["motivo"][:2] == "amplitude").all()
    assert tabela["pico_a_pico_uv"].iloc[0] == pytest.approx(x[3, :1000].max() - x[3, :1000].min())
