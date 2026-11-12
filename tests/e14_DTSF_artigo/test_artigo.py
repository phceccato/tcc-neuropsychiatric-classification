import numpy as np
import pandas as pd
import pytest
import torch

from pipeline.e07_DTSF import redes_dtsf as rd
from pipeline.e07_DTSF import treino as t
from pipeline.e14_DTSF_artigo import rodar as r14


def _entradas(n=3):
    return torch.randn(n, 19, 5, 500), torch.randn(n, 19, 29, 16)


def _n(m):
    return sum(p.numel() for p in m.parameters())


def test_configuracao_do_artigo():
    assert (r14.ARTIGO.filtros, r14.ARTIGO.oculto, r14.ARTIGO.cabecas, r14.ARTIGO.camadas) == (64, 128, 4, 2)
    # o resto do treino é o do D
    assert (r14.ARTIGO.lr, r14.ARTIGO.tamanho_lote, r14.ARTIGO.dropout, r14.ARTIGO.alfa_mixup) == \
        (t.PADRAO.lr, t.PADRAO.tamanho_lote, t.PADRAO.dropout, t.PADRAO.alfa_mixup)


def test_d_artigo_tem_702_mil_parametros_e_o_d_pequeno_80_mil():
    a = r14.ARTIGO
    assert _n(rd.DTSF("D", filtros=a.filtros, oculto=a.oculto, cabecas=a.cabecas, camadas=a.camadas)) == 702_721
    assert _n(rd.DTSF("D_pequeno")) == 80_449


def test_d_artigo_tem_as_formas_do_artigo():
    a = r14.ARTIGO
    m = rd.DTSF("D", filtros=a.filtros, oculto=a.oculto, cabecas=a.cabecas, camadas=a.camadas).eval()
    x_t, x_e = _entradas()
    assert m.ramo_temporal(x_t).shape == (3, 16, 128) and m.ramo_espectral(x_e).shape == (3, 16, 128)
    assert m.tokens(x_t, x_e).shape == (3, 33, 128) and m(x_t, x_e).shape == (3,)
    assert m.encoder[0].atencao.num_heads == 4 and len(m.encoder) == 2


def test_dimensoes_padrao_mantem_o_d_identico():
    x_t, x_e = _entradas()
    torch.manual_seed(5)
    a = rd.DTSF("D_pequeno").eval()
    torch.manual_seed(5)
    b = rd.DTSF("D", filtros=rd.FILTROS, oculto=rd.OCULTO, cabecas=rd.CABECAS, camadas=rd.CAMADAS).eval()  # = D_pequeno
    assert torch.equal(a(x_t, x_e), b(x_t, x_e))


class _FonteFalsa:
    def __init__(self, uids, por_pessoa=20):
        self.indice = pd.DataFrame([dict(subject_uid=u, linha=i * por_pessoa + j, sem_overlap=j % 2 == 0,
                                         rejeitada=False)
                                    for i, u in enumerate(uids) for j in range(por_pessoa)])
        self.y = {u: i % 2 for i, u in enumerate(uids)}
        self._pessoa = self.indice.set_index("linha")["subject_uid"]

    def lote(self, linhas):
        g = torch.Generator().manual_seed(int(np.sum(linhas)))
        sinal = torch.tensor([float(self.y[self._pessoa[l]]) for l in linhas])
        return (torch.randn(len(linhas), 19, 5, 500, generator=g) + sinal[:, None, None, None],
                torch.randn(len(linhas), 19, 29, 16, generator=g) + sinal[:, None, None, None])


def test_treino_curto_com_a_configuracao_do_artigo():
    torch.set_num_threads(1)
    uids = [f"p{i}" for i in range(8)]
    fonte = _FonteFalsa(uids)
    res = t.treinar("D", fonte, uids[:4], uids[4:6], pd.Series(fonte.y), semente=1, max_epocas=1, paciencia=1,
                    config=r14.ARTIGO)
    assert _n(res.modelo) == 702_721
    prob = t.prever(res.modelo, fonte, uids[6:])
    assert set(prob) == set(uids[6:])


def test_runner_recusa_cuda_sem_gpu(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("sys.argv", ["rodar", "--dispositivo", "cuda"])
    with pytest.raises(SystemExit):
        r14.main()


# ---------- D_artigo_FA ----------

def test_d_artigo_fa_tem_a_fa_com_64_filtros_e_comeca_igual_ao_d_artigo():
    torch.manual_seed(3)
    a = r14.construtor("D_artigo_FA")()
    torch.manual_seed(3)
    cfg = r14.ARTIGO
    b = rd.DTSF("D", filtros=cfg.filtros, oculto=cfg.oculto, cabecas=cfg.cabecas, camadas=cfg.camadas)
    assert _n(a) == 704_769 and _n(a) - _n(b) == 64 * 16 * 2
    assert a.fa.filtro_dct.shape == (64, 250) and a.fa.componentes == (0, 35, 70, 105)
    comuns = {k: v for k, v in a.state_dict().items() if not k.startswith("fa.")}
    assert all(torch.equal(v, b.state_dict()[k]) for k, v in comuns.items())


def test_construtor_do_d_artigo_e_o_padrao_e_modelo_desconhecido_falha():
    assert r14.construtor("D_artigo") is None
    with pytest.raises(ValueError):
        r14.construtor("D_artigo_X")


def test_treino_curto_do_d_artigo_fa():
    torch.set_num_threads(1)
    uids = [f"p{i}" for i in range(8)]
    fonte = _FonteFalsa(uids)
    res = t.treinar("D", fonte, uids[:4], uids[4:6], pd.Series(fonte.y), semente=1, max_epocas=1, paciencia=1,
                    config=r14.ARTIGO, construir=r14.construtor("D_artigo_FA"))
    assert _n(res.modelo) == 704_769 and hasattr(res.modelo, "fa")


def test_runner_so_aceita_os_modelos_da_etapa(monkeypatch):
    monkeypatch.setattr("sys.argv", ["rodar", "--modelos", "D_FA"])
    with pytest.raises(SystemExit):
        r14.main()
