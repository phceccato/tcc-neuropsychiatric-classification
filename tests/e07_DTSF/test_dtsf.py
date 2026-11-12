import numpy as np
import pandas as pd
import pytest
import torch
from scipy.signal import stft as stft_scipy

from pipeline.e07_DTSF import dados_dtsf as dd
from pipeline.e07_DTSF import redes_dtsf as rd
from pipeline.e07_DTSF import treino as t

TIPOS = list(rd.TIPOS_DTSF)


def _entradas(n=4, amostras=500):
    return torch.randn(n, 19, 5, amostras), torch.randn(n, 19, 29, 16)


def _n(m):
    return sum(p.numel() for p in m.parameters())


# ---------- redes ----------

def test_os_seis_modelos_da_etapa_e_o_principal():
    assert rd.TIPOS_DTSF == ("D_FA", "D_pequeno", "D", "D_sem_mixup", "D_sem_T", "D_sem_E")
    assert rd.PRINCIPAL == "D_FA"


@pytest.mark.parametrize("tipo", TIPOS)
def test_cada_variante_devolve_um_logit_por_janela(tipo):
    assert rd.DTSF(tipo)(*_entradas(8)).shape == (8,)


def test_numero_de_parametros():
    assert _n(rd.DTSF("D_pequeno")) == 80_449        # o D das rodadas anteriores
    assert _n(rd.DTSF("D")) == 702_721               # dimensões do artigo
    assert _n(rd.DTSF("D_FA")) == 704_769            # + FA: FC(64 → 16) e FC(16 → 64), sem viés
    assert _n(rd.DTSF("D_sem_mixup")) == _n(rd.DTSF("D_FA"))


def test_dimensoes_do_artigo_e_reduzidas():
    for tipo in TIPOS:
        v = rd.MODELOS[tipo]
        esperado = (16, 32, 2, 2) if tipo == "D_pequeno" else (64, 128, 4, 2)
        assert (v.filtros, v.oculto, v.cabecas, v.camadas) == esperado
    m = rd.DTSF("D_FA")
    assert m.encoder[0].atencao.num_heads == 4 and len(m.encoder) == 2


@pytest.mark.parametrize("tipo, oculto", [("D_pequeno", 32), ("D", 128), ("D_FA", 128)])
def test_formas_dos_ramos_e_dos_tokens(tipo, oculto):
    modelo = rd.DTSF(tipo).eval()
    x_t, x_e = _entradas(3)
    assert modelo.ramo_temporal(x_t).shape == (3, 16, oculto)
    assert modelo.ramo_espectral(x_e).shape == (3, 16, oculto)
    assert modelo.tokens(x_t, x_e).shape == (3, 33, oculto)


def test_fa_no_principal_e_nas_ablacoes_com_ramo_temporal_e_64_filtros_em_16_grupos():
    assert {tipo: rd.DTSF(tipo).usa_fa for tipo in TIPOS} == {
        "D_FA": True, "D_pequeno": False, "D": False, "D_sem_mixup": True, "D_sem_T": False, "D_sem_E": True}
    fa = rd.DTSF("D_FA").fa
    assert fa.filtro_dct.shape == (64, 250) and fa.componentes == tuple(range(0, 226, 15))
    assert all(torch.equal(fa.filtro_dct[4 * g], fa.filtro_dct[4 * g + 3]) for g in range(16))  # 4 filtros por grupo


def test_d_fa_comeca_com_os_pesos_do_d():
    torch.manual_seed(3)
    a = rd.DTSF("D_FA")
    torch.manual_seed(3)
    b = rd.DTSF("D")
    comuns = {k: v for k, v in a.state_dict().items() if not k.startswith("fa.")}
    assert comuns.keys() == b.state_dict().keys()
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in comuns.items())


def test_fa_reescala_cada_filtro_do_mapa_temporal():
    torch.manual_seed(0)
    m = rd.DTSF("D_FA").eval()
    x_t, _ = _entradas(2)
    mapa = m.conv_t[0](x_t)
    pesos = m.fa.pesos(mapa)
    assert pesos.shape == (2, 64) and bool(((pesos > 0) & (pesos < 1)).all())
    assert torch.allclose(m.fa(mapa), mapa * pesos[:, :, None, None])


def test_fa_componente_zero_e_o_gap_escalado():
    fa = rd.AtencaoFrequencia(canais=64)
    mapa = torch.randn(2, 64, 1, 250)
    freq = (mapa[:, :, 0, :] * fa.filtro_dct).sum(-1)
    assert torch.allclose(freq[:, :4], mapa.mean(-1)[:, :4, 0] * 250 ** 0.5, atol=1e-4)


def test_fa_exige_filtros_divisiveis_pelos_16_componentes():
    with pytest.raises(ValueError):
        rd.AtencaoFrequencia(canais=24)


def test_d_sem_t_ignora_o_sinal_e_d_sem_e_ignora_a_stft():
    torch.manual_seed(0)
    x_t, x_e = _entradas()
    sem_t, sem_e = rd.DTSF("D_sem_T").eval(), rd.DTSF("D_sem_E").eval()
    assert torch.equal(sem_t(x_t, x_e), sem_t(torch.randn_like(x_t), x_e))
    assert torch.equal(sem_e(x_t, x_e), sem_e(x_t, torch.randn_like(x_e)))
    assert sem_t.tokens(x_t, x_e).shape[1] == 17 and sem_e.tokens(x_t, x_e).shape[1] == 17
    assert not sem_t.usa_fa and _n(sem_t) < _n(rd.DTSF("D"))


def test_so_d_sem_mixup_desliga_o_mixup():
    assert {tipo: rd.DTSF(tipo).mixup for tipo in TIPOS} == {t: t != "D_sem_mixup" for t in TIPOS}


def test_ablacoes_do_d_fa_mudam_uma_coisa_so():
    v = rd.MODELOS
    assert v["D_sem_mixup"] == rd.Variante(mixup=False, fa=True) and v["D_sem_E"] == rd.Variante(espectral=False, fa=True)
    assert v["D"] == rd.Variante() and v["D_FA"] == rd.Variante(fa=True)
    torch.manual_seed(2)
    a = rd.DTSF("D_FA")
    torch.manual_seed(2)
    b = rd.DTSF("D_sem_mixup")
    assert all(torch.equal(x, b.state_dict()[k]) for k, x in a.state_dict().items())   # mesma rede, só o treino muda


def test_dimensoes_explicitas_substituem_as_do_modelo():
    torch.manual_seed(5)
    a = rd.DTSF("D", filtros=16, oculto=32, cabecas=2, camadas=2)
    torch.manual_seed(5)
    b = rd.DTSF("D_pequeno")
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in a.state_dict().items())


def test_d_sem_fusao_foi_removido():
    assert "D_sem_fusao" not in rd.TIPOS_DTSF


def test_tipo_desconhecido_falha():
    with pytest.raises(ValueError):
        rd.DTSF("X")


def test_a_stft_e_normalizada_pelos_buffers():
    torch.manual_seed(0)
    modelo = rd.DTSF("D").eval()
    x_t, x_e = _entradas()
    media, desvio = torch.full((19, 29), 3.0), torch.full((19, 29), 2.0)
    esperado = modelo(x_t, x_e)
    modelo.definir_normalizacao(media, desvio)
    assert torch.allclose(modelo(x_t, x_e * 2.0 + 3.0), esperado, atol=1e-5)


# ---------- Mixup ----------

def test_misturar_usa_o_mesmo_lambda_e_permutacao_nos_dois_ramos_e_no_alvo():
    x_t, x_e = _entradas(4)
    alvo = torch.tensor([1.0, 0.0, 1.0, 0.0])
    perm = np.array([2, 0, 3, 1])
    m_t, m_e, m_alvo = rd.misturar(x_t, x_e, alvo, 0.3, perm)
    assert torch.allclose(m_t, 0.3 * x_t + 0.7 * x_t[perm])
    assert torch.allclose(m_e, 0.3 * x_e + 0.7 * x_e[perm])
    assert torch.allclose(m_alvo, 0.3 * alvo + 0.7 * alvo[perm])


def test_sortear_mixup_e_reprodutivel_e_valido():
    lam, perm = rd.sortear_mixup(np.random.default_rng(1), 10)
    assert 0 <= lam <= 1 and sorted(perm) == list(range(10))
    lam2, perm2 = rd.sortear_mixup(np.random.default_rng(1), 10)
    assert lam == lam2 and list(perm) == list(perm2)


def test_bce_com_pos_weight_e_linear_no_alvo_misturado():
    f = torch.nn.functional.binary_cross_entropy_with_logits
    logit, y, w = torch.randn(8), torch.tensor([1.0, 0, 1, 0, 1, 1, 0, 0]), torch.tensor(1.3)
    perm = torch.randperm(8)
    misturado = f(logit, 0.3 * y + 0.7 * y[perm], pos_weight=w)
    assert torch.allclose(misturado, 0.3 * f(logit, y, pos_weight=w) + 0.7 * f(logit, y[perm], pos_weight=w))


def test_misturar_antes_ou_depois_da_normalizacao_da_o_mesmo():
    x_t, x_e = _entradas(4)
    perm, media, desvio = np.array([1, 2, 3, 0]), 3.0, 2.0
    _, antes, _ = rd.misturar(x_t, (x_e - media) / desvio, torch.zeros(4), 0.4, perm)
    _, depois, _ = rd.misturar(x_t, x_e, torch.zeros(4), 0.4, perm)
    assert torch.allclose(antes, (depois - media) / desvio, atol=1e-5)


# ---------- STFT e multibanda ----------

def test_frequencias_da_stft_sao_2_a_30_hz():
    np.testing.assert_array_equal(dd.FREQUENCIAS, np.arange(2, 31))


def test_log_stft_tem_19_canais_29_frequencias_16_quadros():
    assert dd.log_stft(torch.randn(5, 19, 500)).shape == (5, 19, 29, 16)


def test_senoide_de_10_hz_cai_no_bin_de_10_hz():
    tempo = torch.arange(500, dtype=torch.float64) / 125
    x = torch.sin(2 * np.pi * 10 * tempo).repeat(1, 19, 1)
    potencia = dd.log_stft(x)[0, 0].mean(dim=1)
    assert dd.FREQUENCIAS[int(potencia.argmax())] == 10


def test_log_stft_bate_com_o_scipy_a_menos_de_uma_constante():
    x = np.random.default_rng(0).normal(size=500)
    _, _, z = stft_scipy(x, fs=125, window="hann", nperseg=125, noverlap=100, boundary=None, padded=False)
    esperado = np.log10(np.abs(z[2:31]) ** 2)
    obtido = dd.log_stft(torch.from_numpy(x).repeat(1, 19, 1))[0, 0].numpy()
    diferenca = obtido - esperado
    assert diferenca.shape == (29, 16) and np.ptp(diferenca) < 1e-6


def test_multibanda_tem_o_sinal_base_e_4_bandas():
    base = np.random.default_rng(0).normal(size=(19, 125 * 30)).astype(np.float32)
    mb = dd.multibanda(base)
    assert mb.shape == (5, 19, 125 * 30) and mb.dtype == np.float32
    np.testing.assert_array_equal(mb[0], base)


@pytest.mark.parametrize("hz, banda", [(3, 1), (6, 2), (10, 3), (20, 4)])
def test_cada_senoide_fica_na_sua_banda(hz, banda):
    tempo = np.arange(125 * 30) / 125
    base = np.tile(np.sin(2 * np.pi * hz * tempo), (19, 1)).astype(np.float32)
    energia = (dd.multibanda(base)[1:, 0, 500:-500] ** 2).mean(axis=1)
    assert int(energia.argmax()) + 1 == banda


def test_estatisticas_stft_usam_so_as_linhas_pedidas():
    class Fonte:
        def lote(self, linhas):
            x_e = torch.from_numpy(np.asarray(linhas, dtype=np.float32))[:, None, None, None].expand(-1, 19, 29, 16)
            return torch.zeros(len(linhas), 19, 5, 10), x_e.clone()

    media, desvio = dd.estatisticas_stft(Fonte(), np.array([1, 3]), tamanho_lote=1)
    assert media.shape == (19, 29) and torch.allclose(media, torch.full((19, 29), 2.0))
    assert torch.allclose(desvio, torch.full((19, 29), 1.0))


def _arquivos_fonte(tmp_path, n_amostras=125 * 20):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(19, n_amostras)).astype(np.float32)
    np.save(tmp_path / "A.npy", dd.multibanda(base))
    media, desvio = rng.normal(size=19).astype(np.float32), rng.uniform(5, 20, 19).astype(np.float32)
    np.savez(tmp_path / "A.npz", media=media, desvio=desvio)
    janelas = pd.DataFrame(dict(subject_uid="A", inicio_amostra=[0, 500, 1000], sem_overlap=[True, False, True],
                                rejeitada=False))
    janelas.to_csv(tmp_path / "janelas.csv", index=False)
    return base, media, desvio


def test_fonte_dtsf_entrega_multibanda_e_stft_do_sinal_em_uv(tmp_path):
    base, media, desvio = _arquivos_fonte(tmp_path)
    fonte = dd.FonteDTSF(["A"], tmp_path / "janelas.csv", tmp_path, tmp_path)
    x_t, x_e = fonte.lote(np.array([1]))  # janela que começa na amostra 500 a 250 Hz → 250 a 125 Hz
    assert x_t.shape == (1, 19, 5, 500) and x_e.shape == (1, 19, 29, 16)
    np.testing.assert_array_equal(x_t[0, :, 0].numpy(), base[:, 250:750])
    uv = base[:, 250:750] * desvio[:, None] + media[:, None]
    assert torch.allclose(x_e, dd.log_stft(torch.from_numpy(uv)[None]), atol=1e-4)


# ---------- treino ----------

class _FonteSintetica:
    """Pacientes com potência maior em um bin da STFT; as pessoas fora do treino têm STFT enorme."""

    def __init__(self, n_pessoas=8, janelas=20, semente=0):
        rng = np.random.default_rng(semente)
        self.uids = [f"P{i}" for i in range(n_pessoas // 2)] + [f"H{i}" for i in range(n_pessoas // 2)]
        self.y = pd.Series([1] * (n_pessoas // 2) + [0] * (n_pessoas // 2), index=self.uids)
        self.indice = pd.DataFrame([dict(subject_uid=u, inicio_amostra=k * 500, sem_overlap=k % 2 == 0,
                                         rejeitada=False) for u in self.uids for k in range(janelas)])
        self.indice["linha"] = np.arange(len(self.indice))
        self.x_e = rng.normal(size=(len(self.indice), 19, 29, 16)).astype(np.float32)
        self.x_e[:, :, 5] += 2.0 * self.indice["subject_uid"].map(self.y).to_numpy()[:, None, None]
        self.pedidas = []

    def lote(self, linhas):
        self.pedidas.append(np.array(linhas))
        return torch.zeros(len(linhas), 19, 5, 500), torch.from_numpy(self.x_e[linhas])


def _papeis(fonte):
    u = fonte.uids
    return u[1:3] + u[5:7], [u[3], u[7]], [u[0], u[4]]


def test_normalizacao_da_stft_usa_so_pessoas_de_treino():
    fonte = _FonteSintetica()
    treino, validacao, teste = _papeis(fonte)
    fora = fonte.indice["subject_uid"].isin(validacao + teste).to_numpy()
    fonte.x_e[fora] = 1000.0
    modelo = t.treinar("D", fonte, treino, validacao, fonte.y, semente=0, max_epocas=1).modelo
    linhas = fonte.indice.loc[fonte.indice["subject_uid"].isin(treino), "linha"].to_numpy()
    esperado = torch.from_numpy(fonte.x_e[linhas]).mean(dim=(0, 3))
    assert torch.allclose(modelo.stft_media, esperado, atol=1e-4)
    assert float(modelo.stft_media.max()) < 100


@pytest.mark.parametrize("tipo, com_mixup", [("D", True), ("D_sem_mixup", False)])
def test_mixup_so_no_treino(monkeypatch, tipo, com_mixup):
    chamadas = []
    original = t.misturar

    def espiao(*args):
        chamadas.append(len(args[2]))
        return original(*args)

    monkeypatch.setattr(t, "misturar", espiao)
    fonte = _FonteSintetica()
    treino, validacao, teste = _papeis(fonte)
    modelo = t.treinar(tipo, fonte, treino, validacao, fonte.y, semente=0, max_epocas=1).modelo
    lotes_de_treino = -(-len(treino) * min(t.JANELAS_POR_PESSOA, 20) // t.TAMANHO_LOTE)
    assert len(chamadas) == (lotes_de_treino if com_mixup else 0)
    antes = len(chamadas)
    t.prever(modelo, fonte, teste)
    assert len(chamadas) == antes


def test_mixup_nao_muda_o_sorteio_de_janelas():
    sorteios = {}
    for tipo in ("D", "D_sem_mixup"):
        fonte = _FonteSintetica()
        treino, validacao, _ = _papeis(fonte)
        t.treinar(tipo, fonte, treino, validacao, fonte.y, semente=4, max_epocas=2)
        sorteios[tipo] = np.concatenate(fonte.pedidas[1:])  # [0] é a normalização
    np.testing.assert_array_equal(sorteios["D"], sorteios["D_sem_mixup"])


def test_treino_do_d_e_reprodutivel():
    fonte = _FonteSintetica()
    treino, validacao, teste = _papeis(fonte)
    a = t.prever(t.treinar("D", fonte, treino, validacao, fonte.y, semente=3, max_epocas=2).modelo, fonte, teste)
    b = t.prever(t.treinar("D", fonte, treino, validacao, fonte.y, semente=3, max_epocas=2).modelo, fonte, teste)
    assert a == b


def test_previsao_da_pessoa_e_a_media_das_janelas_aceitas_sem_overlap():
    fonte = _FonteSintetica()
    treino, validacao, teste = _papeis(fonte)
    modelo = t.treinar("D", fonte, treino, validacao, fonte.y, semente=0, max_epocas=1).modelo
    prob = t.prever(modelo, fonte, teste)
    uid = teste[0]
    linhas = fonte.indice.loc[(fonte.indice["subject_uid"] == uid) & fonte.indice["sem_overlap"], "linha"].to_numpy()
    with torch.no_grad():
        esperado = float(torch.sigmoid(modelo(*fonte.lote(linhas))).mean())
    assert prob[uid] == pytest.approx(esperado, rel=1e-5)


def test_gerador_do_mixup_e_derivado_da_semente_mas_independente_do_sorteio():
    a, b = t.gerador_mixup(7), t.gerador_mixup(7)
    assert a.random() == b.random()  # reprodutível
    assert t.gerador_mixup(7).random() != np.random.default_rng(7).random()  # não repete o fluxo do sorteio
