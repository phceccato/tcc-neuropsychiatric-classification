import numpy as np
import pandas as pd
import torch

from pipeline.e09_investigacao import explicar as ex


class _Linear(torch.nn.Module):
    def __init__(self, w_t, w_e):
        super().__init__()
        self.w_t, self.w_e = w_t, w_e

    def forward(self, x_t, x_e):
        return (x_t * self.w_t).sum((1, 2, 3)) + (x_e * self.w_e).sum((1, 2, 3))


class _Quadratico(_Linear):
    def forward(self, x_t, x_e):
        return super().forward(x_t, x_e) ** 2


def _entradas(n, gerador):
    return torch.randn(n, 2, 3, 4, generator=gerador), torch.randn(n, 2, 2, 3, generator=gerador)


def test_modelo_linear_com_um_fundo_da_atribuicao_exata():
    g = torch.Generator().manual_seed(0)
    x_t, x_e = _entradas(5, g)
    f_t, f_e = _entradas(1, g)
    modelo = _Linear(torch.randn(2, 3, 4, generator=g), torch.randn(2, 2, 3, generator=g))
    phi_t, phi_e = ex.gradientes_esperados(modelo, x_t, x_e, f_t, f_e, amostras=7, rng=np.random.default_rng(0),
                                           janelas_por_passo=2)
    torch.testing.assert_close(phi_t, modelo.w_t * (x_t - f_t))
    torch.testing.assert_close(phi_e, modelo.w_e * (x_e - f_e))


def test_completude_aproximada_em_modelo_nao_linear():
    g = torch.Generator().manual_seed(1)
    x_t, x_e = _entradas(3, g)
    f_t, f_e = _entradas(1, g)
    modelo = _Quadratico(0.3 * torch.randn(2, 3, 4, generator=g), 0.3 * torch.randn(2, 2, 3, generator=g))
    phi_t, phi_e = ex.gradientes_esperados(modelo, x_t, x_e, f_t, f_e, amostras=4000, rng=np.random.default_rng(1))
    soma = phi_t.sum((1, 2, 3)) + phi_e.sum((1, 2, 3))
    torch.testing.assert_close(soma, modelo(x_t, x_e) - modelo(f_t, f_e), rtol=0.05, atol=0.05)


def test_agregar_soma_nos_eixos_certos():
    phi_t = torch.ones(2, 19, 5, 500)
    phi_e = -torch.ones(2, 19, 29, 16)
    a = ex.agregar(phi_t, phi_e)
    assert a["t_abs_canal_entrada"].shape == (19, 5) and float(a["t_abs_canal_entrada"][0, 0]) == 500
    assert a["t_abs_tempo"].shape == (500,) and float(a["t_abs_tempo"][0]) == 19 * 5
    assert a["e_abs_canal_freq"].shape == (19, 29) and float(a["e_sinal_canal_freq"][0, 0]) == -16
    assert a["e_abs_quadro"].shape == (16,) and float(a["e_abs_quadro"][0]) == 19 * 29


def test_fundo_equilibrado_e_so_de_treino_e_aceitas():
    indice = pd.DataFrame(dict(subject_uid=list("AABBCC"), linha=range(6), rejeitada=[False, True, False, False, False, False]))
    fonte = type("F", (), dict(indice=indice))()
    y = pd.Series(dict(A=0, B=1, C=1))
    linhas = ex.sortear_fundo(fonte, ["A", "B"], y, por_classe=5, rng=np.random.default_rng(0))
    assert sorted(linhas) == [0, 2, 3]  # C fora do treino; a janela 1 é rejeitada
