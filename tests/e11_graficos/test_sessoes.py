from pathlib import Path

import numpy as np
import pytest

from pipeline.e11_graficos import sessoes

MANIFESTO = Path("outputs/02_harmonizacao/manifesto.csv")


def test_deslocamentos_sao_deterministicos():
    x = np.array([2003.5, 2003.5, 2003.5, 2006.0])
    d = sessoes.deslocamentos(x, tolerancia=0.01)
    np.testing.assert_allclose(d, [0.0, sessoes.PASSO, -sessoes.PASSO, 0.0])
    np.testing.assert_array_equal(d, sessoes.deslocamentos(x, tolerancia=0.01))


def test_faixas_das_sessoes_nao_se_sobrepoem():
    f = sessoes.faixas(np.array([0.0, 0.10, 0.12]), np.array([0.0, 0.10, 0.5]), largura=1.0)
    assert all(b <= a2 for (_, b), (a2, _) in zip(f, f[1:]))
    assert f[0][1] - f[0][0] == pytest.approx(sessoes.LARGURA_MINIMA)


@pytest.mark.skipif(not MANIFESTO.exists(), reason="precisa do manifesto da etapa 02 (dados)")
def test_sessoes_reais_conferem_com_os_docs_e_gravam_pdf_e_png(tmp_path):
    caminhos = sessoes._gravar(sessoes.figura(sessoes.dados(MANIFESTO)), tmp_path, "s12_sessoes")
    assert [c.name for c in caminhos] == ["s12_sessoes.pdf", "s12_sessoes.png"]
    assert all(c.stat().st_size > 0 for c in caminhos)
