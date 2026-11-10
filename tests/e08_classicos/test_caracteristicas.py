import json

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.stats import kurtosis, skew

from pipeline.e08_classicos import caracteristicas as c
from pipeline.e07_DTSF import dados_dtsf as dd


# ---------- temporais ----------

def test_475_caracteristicas_temporais_por_janela():
    x = np.random.default_rng(0).normal(size=(3, 19, 5, 500))
    assert c.caracteristicas_temporais(x).shape == (3, 475) == (3, c.N_TEMPORAIS)
    assert len(c.nomes_temporais()) == 475 and len(set(c.nomes_temporais())) == 475


def test_ordem_canal_representacao_estatistica():
    x = np.zeros((1, 19, 5, 500))
    x[0, 7, 3] = np.random.default_rng(1).normal(size=500)  # só T3, alfa
    f = c.caracteristicas_temporais(x)[0]
    i = 7 * 25 + 3 * 5                                      # canal · 25 + representação · 5 + estatística
    assert f[i] > 0 and np.count_nonzero(f) == np.count_nonzero(f[i:i + 5])
    assert c.nomes_temporais()[i:i + 5] == [f"T3|alfa|{e}" for e in c.ESTATISTICAS_TEMPORAIS]
    assert c.nomes_temporais()[0] == "Fp1|base|desvio" and c.nomes_temporais()[-1] == "O2|beta|curtose"


def test_valores_iguais_ao_numpy_e_ao_scipy():
    x = np.random.default_rng(2).gamma(2.0, size=(4, 19, 5, 500)) - 1.0
    f = c.caracteristicas_temporais(x).reshape(4, 19, 5, 5)
    np.testing.assert_allclose(f[..., 0], x.std(axis=-1), rtol=1e-12)
    np.testing.assert_allclose(f[..., 1], np.sqrt((x ** 2).mean(axis=-1)), rtol=1e-12)
    np.testing.assert_allclose(f[..., 2], np.ptp(x, axis=-1), rtol=1e-12)
    np.testing.assert_allclose(f[..., 3], skew(x, axis=-1), rtol=1e-9)
    np.testing.assert_allclose(f[..., 4], kurtosis(x, axis=-1), rtol=1e-9)  # excesso, sem correção de viés


def test_senoide_de_ciclos_inteiros():
    t = np.arange(500) / 125
    seno = 3.0 * np.sin(2 * np.pi * 10 * t)                  # 40 ciclos na janela
    x = np.broadcast_to(seno, (1, 19, 5, 500))
    desvio, rms, ptp, assim, curt = c.caracteristicas_temporais(x).reshape(19, 5, 5)[0, 0]
    assert desvio == pytest.approx(3 / np.sqrt(2), rel=1e-9)
    assert rms == pytest.approx(3 / np.sqrt(2), rel=1e-9)
    # 12,5 amostras por ciclo: a amostra mais próxima do pico fica a 0,02π de fase, então ±3·cos(π/50)
    assert ptp == pytest.approx(6.0 * np.cos(np.pi / 50), rel=1e-9)
    assert assim == pytest.approx(0.0, abs=1e-9)
    assert curt == pytest.approx(-1.5, rel=1e-9)               # curtose de Pearson 1,5 − 3


def test_curtose_em_excesso_de_uma_normal_fica_perto_de_zero():
    x = np.random.default_rng(3).normal(size=(1, 19, 5, 200_000))
    assert np.abs(c.caracteristicas_temporais(x).reshape(19, 5, 5)[..., 4]).max() < 0.1


def test_sinal_constante_tem_assimetria_e_curtose_zero_por_convencao():
    x = np.random.default_rng(4).normal(size=(2, 19, 5, 500))
    x[1, 2, 4] = 0.7
    f = c.caracteristicas_temporais(x)
    assert np.isfinite(f).all()
    desvio, rms, ptp, assim, curt = f.reshape(2, 19, 5, 5)[1, 2, 4]
    assert (desvio, ptp, assim, curt) == (0.0, 0.0, 0.0, 0.0) and rms == pytest.approx(0.7)
    assert c.constantes(x).sum() == 1 and c.constantes(x)[1, 2, 4]


@pytest.mark.parametrize("invalido", [np.nan, np.inf, -np.inf])
def test_valor_invalido_na_entrada_falha(invalido):
    x = np.zeros((1, 19, 5, 500))
    x[0, 0, 0, 10] = invalido
    with pytest.raises(ValueError):
        c.caracteristicas_temporais(x)


def test_sem_nan_nem_infinito_em_entradas_validas():
    x = np.random.default_rng(5).standard_t(3, size=(8, 19, 5, 500)).astype(np.float32)
    assert np.isfinite(c.caracteristicas_temporais(x)).all()


# ---------- espectrais ----------

def test_1102_caracteristicas_espectrais_por_janela():
    x = np.random.default_rng(6).normal(size=(3, 19, 29, 16))
    assert c.caracteristicas_espectrais(x).shape == (3, 1102) == (3, c.N_ESPECTRAIS)
    nomes = c.nomes_espectrais()
    assert len(nomes) == 1102 and nomes[:2] == ["Fp1|2Hz|media", "Fp1|2Hz|desvio"] and nomes[-1] == "O2|30Hz|desvio"


def test_media_e_desvio_no_eixo_dos_quadros():
    x = np.random.default_rng(7).normal(size=(2, 19, 29, 16))
    f = c.caracteristicas_espectrais(x).reshape(2, 19, 29, 2)
    np.testing.assert_allclose(f[..., 0], x.mean(axis=3))
    np.testing.assert_allclose(f[..., 1], x.std(axis=3))      # ddof = 0
    x2 = np.zeros((1, 19, 29, 16))
    x2[0, 4, 8] = np.arange(16)                               # Fz, 10 Hz, rampa no tempo
    f2 = c.caracteristicas_espectrais(x2)
    i = (4 * 29 + 8) * 2
    assert f2[0, i] == pytest.approx(7.5) and f2[0, i + 1] == pytest.approx(np.arange(16).std())
    assert c.nomes_espectrais()[i] == "Fz|10Hz|media"


def test_espectrais_vem_da_stft_do_dtsf():
    uv = torch.from_numpy(np.random.default_rng(8).normal(scale=10, size=(2, 19, 500)))
    x_e = dd.log_stft(uv).numpy()
    f = c.caracteristicas_espectrais(x_e).reshape(2, 19, 29, 2)
    np.testing.assert_allclose(f[..., 0], x_e.mean(axis=-1), rtol=1e-6)
    assert np.isfinite(f).all()


@pytest.mark.parametrize("invalido", [np.nan, np.inf])
def test_espectral_invalido_falha(invalido):
    x = np.zeros((1, 19, 29, 16))
    x[0, 0, 0, 0] = invalido
    with pytest.raises(ValueError):
        c.caracteristicas_espectrais(x)


# ---------- extração ----------

def _arquivos(tmp_path):
    rng = np.random.default_rng(9)
    linhas = []
    for uid in ("A", "B"):
        base = rng.normal(size=(19, 125 * 20)).astype(np.float32)
        np.save(tmp_path / f"{uid}.npy", dd.multibanda(base))
        np.savez(tmp_path / f"{uid}.npz", media=rng.normal(size=19).astype(np.float32),
                 desvio=rng.uniform(5, 20, 19).astype(np.float32), canais=np.array(c.CANAIS))
        linhas += [dict(subject_uid=uid, dataset="SZ", inicio_amostra=k * 500, sem_overlap=k % 2 == 0,
                        rejeitada=(uid == "A" and k == 1)) for k in range(5)]
    pd.DataFrame(linhas).to_csv(tmp_path / "janelas.csv", index=False)


def test_extracao_alinha_com_janelas_csv_e_usa_a_fonte_do_dtsf(tmp_path):
    _arquivos(tmp_path)
    destino = tmp_path / "saida"
    resumo = c.extrair(tmp_path / "janelas.csv", tmp_path, tmp_path, destino)
    temporais, espectrais = c.carregar(destino, tmp_path / "janelas.csv")
    assert temporais.shape == (10, 475) and espectrais.shape == (10, 1102)
    assert np.isnan(temporais[1]).all() and np.isnan(espectrais[1]).all()      # rejeitada: nunca calculada
    aceitas = np.r_[0, 2:10]
    assert np.isfinite(temporais[aceitas]).all() and np.isfinite(espectrais[aceitas]).all()
    fonte = dd.FonteDTSF(["A", "B"], tmp_path / "janelas.csv", tmp_path, tmp_path)
    x_t, x_e = fonte.lote(np.array([7]))
    np.testing.assert_allclose(temporais[7], c.caracteristicas_temporais(x_t.numpy())[0], rtol=1e-5)
    np.testing.assert_allclose(espectrais[7], c.caracteristicas_espectrais(x_e.numpy())[0], rtol=1e-5)
    assert resumo["aceitas"] == 9 and resumo["constantes"] == 0
    assert json.loads((destino / "extracao.json").read_text())["nomes_temporais"] == c.nomes_temporais()


def test_carregar_recusa_outro_janelas_csv(tmp_path):
    _arquivos(tmp_path)
    c.extrair(tmp_path / "janelas.csv", tmp_path, tmp_path, tmp_path / "saida")
    with open(tmp_path / "janelas.csv", "a") as f:
        f.write("B,SZ,2500,False,True,\n")
    with pytest.raises(ValueError):
        c.carregar(tmp_path / "saida", tmp_path / "janelas.csv")


def test_extracao_recusa_canais_em_outra_ordem(tmp_path):
    _arquivos(tmp_path)
    z = dict(np.load(tmp_path / "B.npz"))
    z["canais"] = z["canais"][::-1]
    np.savez(tmp_path / "B.npz", **z)
    with pytest.raises(ValueError):
        c.extrair(tmp_path / "janelas.csv", tmp_path, tmp_path, tmp_path / "saida")
