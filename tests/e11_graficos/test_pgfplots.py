import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from pipeline.e11_graficos import comum, h1_nula, previa, roc, sessoes

MANIFESTO = Path("outputs/02_harmonizacao/manifesto.csv")


def _sintetico(n=12, semente=0):
    rng = np.random.default_rng(semente)
    y = np.array([0, 1] * (n // 2))
    return y, y * 0.4 + rng.random(n)


def test_checar_tex_aceita_balanceado_e_recusa_o_resto():
    comum.checar_tex("\\begin{figure}{a}\\begin{tikzpicture}\\end{tikzpicture}\\end{figure}")
    for ruim in ("\\begin{figure}{\\end{figure}", "\\begin{figure}\\end{tikzpicture}", "\\begin{a}", "}{"):
        with pytest.raises(ValueError):
            comum.checar_tex(ruim)


def test_conferir_tolera_so_a_terceira_casa():
    comum.conferir("x", 0.8265, 0.827)
    with pytest.raises(comum.Divergencia):
        comum.conferir("x", 0.8250, 0.827)


def test_estilo_dos_eixos_so_tem_chaves_do_pgfplots():
    assert not re.search(r"(^|, )font=", comum.ESTILO_EIXOS)   # `font` não é chave de eixo: o pdflatex para


def test_curva_em_degraus_tem_a_auc_do_sklearn():
    y, escore = _sintetico()
    pontos, auc = roc.curva(y, escore)
    assert auc == pytest.approx(roc_auc_score(y, escore))
    assert pontos[0] == (0.0, 0.0) and pontos[-1] == (1.0, 1.0)
    for (x0, y0), (x1, y1) in zip(pontos, pontos[1:]):  # sem empates: cada passo anda num eixo só
        assert x0 == x1 or y0 == y1


def test_empilhar_e_deslocamentos_sao_deterministicos():
    v = np.array([0.501, 0.505, 0.73, 0.509])
    assert h1_nula.empilhar(v) == [(0.51, 1), (0.51, 2), (0.51, 3), (0.73, 1)]
    x = np.array([2003.5, 2003.5, 2003.5, 2006.0])
    d = sessoes.deslocamentos(x, tolerancia=0.01)
    np.testing.assert_allclose(d, [0.0, sessoes.PASSO, -sessoes.PASSO, 0.0])
    np.testing.assert_array_equal(d, sessoes.deslocamentos(x, tolerancia=0.01))


def test_faixas_das_sessoes_nao_se_sobrepoem():
    f = sessoes.faixas(np.array([0.0, 0.10, 0.12]), np.array([0.0, 0.10, 0.5]), largura=1.0)
    assert all(b <= a2 for (_, b), (a2, _) in zip(f, f[1:]))
    assert f[0][1] - f[0][0] == pytest.approx(sessoes.LARGURA_MINIMA)


def test_roc_usa_o_d_fa_e_os_combinados_da_h4(numeros_json):
    curvas = roc.dados(numeros_json)
    assert [m["modelo"] for m in curvas["SZ"]] == ["D_FA", "LR_combinado", "RF_combinado", "KNN_combinado"]
    texto = roc.tex(curvas)
    comum.checar_tex(texto)
    for d, modelos in curvas.items():
        trecho = texto.split(f"title={{{d}}}")[1].split("nextgroupplot")[0]
        for m in modelos:
            assert m["auc"] == pytest.approx(numeros_json["modelos"][d][m["modelo"]]["auc"])
            assert f"{m['rotulo']} ({comum.virgula(m['auc'])})" in trecho
    assert "\\label{fig:res-roc}" in texto and texto.startswith("\\begin{figure}[htbp]")


def test_h1_traz_a_auc_observada_e_o_p_de_cada_dataset(numeros_json):
    dados = h1_nula.dados(numeros_json)
    texto = h1_nula.tex(dados)
    comum.checar_tex(texto)
    for d, r in dados.items():
        assert len(r["nulas"]) == 25 and r["p"] == numeros_json["h1"][d]["p"]
        assert f"{{{comum.virgula(r['observada'])}; p = {comum.virgula(r['p'])}}}" in texto
    assert "D\\_FA retreinado" in texto and "\\label{fig:res-h1-nula}" in texto


def _sem_identificador(texto: str) -> None:
    assert not re.search(r"(SZ|MDD)-(HC|SZ|MDD)-", texto)


@pytest.mark.skipif(not MANIFESTO.exists(), reason="precisa do manifesto da etapa 02 (dados)")
def test_sessoes_reais_conferem_com_os_docs_e_sem_identificadores():
    texto = sessoes.tex(sessoes.dados(MANIFESTO))
    comum.checar_tex(texto)
    _sem_identificador(texto)


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="precisa do pdflatex")
def test_figuras_compilam_com_pdflatex(numeros_json, tmp_path):
    textos = [h1_nula.tex(h1_nula.dados(numeros_json)), roc.tex(roc.dados(numeros_json))]
    if MANIFESTO.exists():
        textos.append(sessoes.tex(sessoes.dados(MANIFESTO)))
    for t in textos:
        _sem_identificador(t)
    (tmp_path / "doc.tex").write_text(comum.documento_minimo(*textos), encoding="utf-8")
    r = subprocess.run(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "doc.tex"], cwd=tmp_path,
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, r.stdout[-3000:]


def test_previa_monta_documento_recortado_na_figura():
    corpo = comum.figura("\\begin{tikzpicture}\\end{tikzpicture}", "Legenda", "fig:x")
    doc = previa.documento(corpo)
    comum.checar_tex(doc)
    assert comum.PREAMBULO in doc and "\\PreviewEnvironment{figure}" in doc and "textwidth=16cm" in doc
    assert doc.index("\\begin{document}") < doc.index(corpo.rstrip()) < doc.index("\\end{document}")


@pytest.mark.skipif(bool(previa.faltando()), reason="precisa do pdflatex e do pdftocairo")
def test_previa_gera_png(numeros_json, tmp_path):
    fonte = tmp_path / "fig_roc.tex"
    fonte.write_text(roc.tex(roc.dados(numeros_json)), encoding="utf-8")
    png = previa.gerar_png(fonte, tmp_path / "fig_roc.png", resolucao=72)
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
