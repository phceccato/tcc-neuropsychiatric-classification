import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.e07_DTSF import redes_dtsf as rd
from pipeline.e12_FA_SDKA import redes as r12
from pipeline.e12_FA_SDKA import treino as t12


def _entradas(n=4):
    return torch.randn(n, 19, 5, 500), torch.randn(n, 19, 29, 16)


# ---------- FA ----------

def test_base_dct_e_ortonormal():
    bases = torch.stack([r12.base_dct(250, u) for u in range(6)])
    assert torch.allclose(bases @ bases.T, torch.eye(6), atol=1e-5)


def test_componente_zero_e_o_gap_escalado():
    mapa = torch.randn(3, 16, 1, 250)
    fa = r12.AtencaoFrequencia(componentes=(0,))
    freq = (mapa[:, :, 0, :] * fa.filtro_dct).sum(-1)
    assert torch.allclose(freq, mapa.mean(dim=(2, 3)) * np.sqrt(250), atol=1e-4)


def test_fa_usa_um_componente_por_grupo_de_quatro_filtros():
    fa = r12.AtencaoFrequencia()
    assert fa.componentes == (0, 35, 70, 105)
    assert torch.equal(fa.filtro_dct[4], r12.base_dct(250, 35)) and torch.equal(fa.filtro_dct[7], fa.filtro_dct[4])


def test_fa_reescala_cada_filtro_por_um_peso_entre_0_e_1():
    fa = r12.AtencaoFrequencia()
    mapa = torch.randn(2, 16, 1, 250)
    pesos = fa.pesos(mapa)
    assert pesos.shape == (2, 16) and bool(((pesos > 0) & (pesos < 1)).all())
    assert torch.allclose(fa(mapa), mapa * pesos[:, :, None, None])


def test_fa_rejeita_canais_que_nao_se_dividem_nos_grupos():
    with pytest.raises(ValueError):
        r12.AtencaoFrequencia(canais=16, componentes=(0, 1, 2))


# ---------- redes ----------

@pytest.mark.parametrize("tipo", r12.TIPOS_12)
def test_cada_variante_devolve_um_logit_por_janela(tipo):
    modelo = r12.DTSF12(tipo).eval()
    x_t, x_e = _entradas(5)
    assert modelo(x_t, x_e).shape == (5,)
    assert modelo.representacao(x_t, x_e).shape == (5, 32)


def test_so_as_variantes_com_fa_tem_a_fa():
    assert {tipo: hasattr(r12.DTSF12(tipo), "fa") for tipo in r12.TIPOS_12} == {
        "D_FA": True, "D_SDKA": False, "D_FA_SDKA": True}
    assert {tipo: r12.DTSF12(tipo).usa_sdka for tipo in r12.TIPOS_12} == {
        "D_FA": False, "D_SDKA": True, "D_FA_SDKA": True}


def test_d_sdka_e_o_d_com_a_mesma_inicializacao():
    torch.manual_seed(0)
    d = rd.DTSF("D").eval()
    torch.manual_seed(0)
    v = r12.DTSF12("D_SDKA").eval()
    x_t, x_e = _entradas()
    assert torch.equal(d(x_t, x_e), v(x_t, x_e))


def test_fa_acrescenta_poucos_parametros():
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(r12.DTSF12("D_FA")) - n(rd.DTSF("D")) == 16 * 4 * 2


def test_tipo_desconhecido_falha():
    with pytest.raises(ValueError):
        r12.DTSF12("D")


# ---------- SMMD ----------

def test_pesos_de_classe_somam_um_por_classe():
    w, presente = t12.pesos_classe(torch.tensor([1.0, 1.0, 0.0, 0.0, 1.0]))
    assert torch.allclose(w.sum(0), torch.ones(2)) and bool(presente.all())
    assert torch.allclose(w[:, 1], torch.tensor([1 / 3, 1 / 3, 0, 0, 1 / 3]))


def test_classe_ausente_na_fonte_fica_fora():
    _, presente = t12.pesos_classe(torch.ones(4))
    assert presente.tolist() == [False, True]


def test_smmd_e_zero_com_distribuicoes_iguais_e_cresce_com_o_deslocamento():
    torch.manual_seed(0)
    z = torch.randn(40, 8)
    y = torch.tensor([0.0, 1.0] * 20)
    assert t12.smmd(z, y, z, y).item() == pytest.approx(0, abs=1e-5)
    assert t12.smmd(z, y, z + 3, y).item() > t12.smmd(z, y, z + 0.5, y).item() > 0


def test_smmd_alinha_por_classe_e_nao_pela_distribuicao_global():
    torch.manual_seed(0)
    a, b = torch.randn(30, 4), torch.randn(30, 4) + 4
    z_fonte, y_fonte = torch.cat([a, b]), torch.tensor([0.0] * 30 + [1.0] * 30)
    z_alvo = torch.cat([a, b])
    # mesmas features, classes trocadas no alvo: o MMD global seria 0, o SMMD não
    assert t12.smmd(z_fonte, y_fonte, z_alvo, 1 - y_fonte).item() > 0.5
    assert t12.smmd(z_fonte, y_fonte, z_alvo, y_fonte).item() == pytest.approx(0, abs=1e-5)


def test_smmd_tem_gradiente_nas_features():
    z_fonte = torch.randn(10, 4, requires_grad=True)
    z_alvo = torch.randn(10, 4, requires_grad=True)
    perda = t12.smmd(z_fonte, torch.tensor([0.0, 1.0] * 5), z_alvo, torch.rand(10))
    perda.backward()
    assert z_fonte.grad is not None and z_alvo.grad.abs().sum() > 0


# ---------- treino ----------

class _FonteFalsa:
    def __init__(self, uids, por_pessoa=30):
        self.indice = pd.DataFrame([dict(subject_uid=u, linha=i * por_pessoa + j, sem_overlap=j % 2 == 0,
                                         rejeitada=False)
                                    for i, u in enumerate(uids) for j in range(por_pessoa)])
        self.y = {u: i % 2 for i, u in enumerate(uids)}
        self._pessoa = self.indice.set_index("linha")["subject_uid"]

    def lote(self, linhas):
        g = torch.Generator().manual_seed(int(np.sum(linhas)))
        sinal = torch.tensor([float(self.y[self._pessoa[l]]) for l in linhas])
        x_t = torch.randn(len(linhas), 19, 5, 500, generator=g) + sinal[:, None, None, None]
        x_e = torch.randn(len(linhas), 19, 29, 16, generator=g) + sinal[:, None, None, None]
        return x_t, x_e


def _cenario():
    uids = [f"p{i}" for i in range(10)]
    fonte = _FonteFalsa(uids)
    y = pd.Series(fonte.y)
    return fonte, uids[:6], uids[6:8], uids[8:], y


@pytest.mark.parametrize("tipo", r12.TIPOS_12)
def test_treino_curto_roda_e_preve_cada_pessoa_de_teste(tipo):
    torch.set_num_threads(1)
    fonte, treino, validacao, teste, y = _cenario()
    resultado = t12.treinar(tipo, fonte, treino, validacao, teste, y, semente=1, max_epocas=2, paciencia=1)
    prob = t12.prever(resultado.modelo, fonte, teste)
    assert set(prob) == set(teste) and all(0 < p < 1 for p in prob.values())
    smmd = [h["smmd"] for h in resultado.historico]
    assert all(s is not None for s in smmd) if "SDKA" in tipo else all(s is None for s in smmd)


def test_sem_sdka_o_alvo_nao_e_lido():
    torch.set_num_threads(1)
    fonte, treino, validacao, teste, y = _cenario()
    lidas = []
    lote = fonte.lote
    fonte.lote = lambda linhas: (lidas.extend(linhas.tolist()), lote(linhas))[1]
    t12.treinar("D_FA", fonte, treino, validacao, teste, y, semente=1, max_epocas=1, paciencia=1)
    linhas_teste = set(fonte.indice.loc[fonte.indice["subject_uid"].isin(teste), "linha"])
    assert not linhas_teste & set(lidas)


def test_lote_do_alvo_e_de_uma_pessoa_de_teste_so():
    fonte, _, _, teste, _ = _cenario()
    por_pessoa = {u: g for u, g in fonte.indice.groupby("subject_uid")}
    rng = np.random.default_rng(0)
    linhas = t12._lote_alvo(por_pessoa, teste, 1, np.array([1, 0]), rng)
    assert set(fonte.indice.set_index("linha").loc[linhas, "subject_uid"]) == {teste[0]}
