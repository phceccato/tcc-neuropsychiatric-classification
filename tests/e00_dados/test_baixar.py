import hashlib
from pathlib import Path

import pytest

from pipeline.e00_dados import baixar as b

# Valores da lista versionada (pipeline/e00_dados/arquivos.csv): o que toda máquina precisa ter em dados/
ESPERADO = {"SZ": dict(arquivos=28, bytes=274_341_860, impressao="91e07382b04646f85630a69e4638b0e4"),
            "MDD": dict(arquivos=181, bytes=851_613_696, impressao="e98187819f54803c9aeade1885997e52")}


# ---------- a lista versionada ----------

def test_lista_tem_os_209_arquivos_com_os_totais_esperados():
    assert b.resumo(b.ler_lista()) == ESPERADO


def test_lista_sem_nomes_repetidos_e_com_md5_valido():
    arquivos = b.ler_lista()
    assert len({(a.dataset, a.arquivo) for a in arquivos}) == len(arquivos)
    assert all(len(a.md5) == 32 and int(a.md5, 16) >= 0 for a in arquivos)
    assert {"6921143_H S15 EO.edf", "6921959_H S15 EO.edf"} <= {a.arquivo for a in arquivos}


def test_lista_tem_58_registros_ec_no_mdd():
    assert sum(a.dataset == "MDD" and a.arquivo.endswith(" EC.edf") for a in b.ler_lista()) == 58


# ---------- os dados desta máquina ----------

@pytest.mark.skipif(not (b.DADOS / "Schizophrenia_dataset").exists() or not (b.DADOS / "Depression_dataset").exists(),
                    reason="dados/ ainda não foi baixado (python -m pipeline.e00_dados.baixar)")
def test_dados_desta_maquina_conferem_com_a_lista():
    """Cada arquivo existe com o tamanho e o MD5 da fonte, e a contagem, os bytes e a impressão digital de cada
    dataset são os mesmos em qualquer máquina."""
    assert b.conferir_pasta(b.ler_lista()) == ESPERADO


# ---------- funções, com uma "fonte" local (file://) ----------

def _fonte(tmp_path: Path, conteudos: dict[str, bytes]) -> list[b.Arquivo]:
    origem = tmp_path / "fonte"
    origem.mkdir()
    arquivos = []
    for nome, dados in conteudos.items():
        (origem / nome).write_bytes(dados)
        arquivos.append(b.Arquivo("SZ", nome, len(dados), hashlib.md5(dados).hexdigest(), (origem / nome).as_uri()))
    return arquivos


def test_baixa_confere_e_pula_o_que_ja_existe(tmp_path):
    arquivos = _fonte(tmp_path, {"h01.edf": b"abc" * 1000, "s01.edf": b"xyz" * 500})
    dados = tmp_path / "dados"
    assert [b.baixar_arquivo(a, dados) for a in arquivos] == [True, True]
    assert [b.baixar_arquivo(a, dados) for a in arquivos] == [False, False]
    assert b.conferir_pasta(arquivos, dados)["SZ"]["arquivos"] == 2
    assert not list((dados / "Schizophrenia_dataset").glob("*.part"))


def test_arquivo_local_corrompido_e_baixado_de_novo(tmp_path):
    (a,) = _fonte(tmp_path, {"h01.edf": b"abc" * 1000})
    dados = tmp_path / "dados"
    b.baixar_arquivo(a, dados)
    a.caminho(dados).write_bytes(b"lixo")
    assert b.baixar_arquivo(a, dados) is True
    b.conferir_arquivo(a, a.caminho(dados))


def test_md5_errado_na_fonte_falha_e_nao_deixa_arquivo(tmp_path):
    (a,) = _fonte(tmp_path, {"h01.edf": b"abc" * 1000})
    errado = b.Arquivo(a.dataset, a.arquivo, a.bytes, "0" * 32, a.url)
    with pytest.raises(b.ErroConferencia):
        b.baixar_arquivo(errado, tmp_path / "dados", tentativas=1)
    assert not list((tmp_path / "dados" / "Schizophrenia_dataset").iterdir())


def test_conferir_pasta_aponta_falta_tamanho_e_arquivo_a_mais(tmp_path):
    arquivos = _fonte(tmp_path, {"h01.edf": b"abc" * 1000, "s01.edf": b"xyz" * 500})
    dados = tmp_path / "dados"
    for a in arquivos:
        b.baixar_arquivo(a, dados)
    arquivos[1].caminho(dados).unlink()
    arquivos[0].caminho(dados).write_bytes(b"abc" * 999)
    (dados / "Schizophrenia_dataset" / "extra.edf").write_bytes(b"1")
    with pytest.raises(b.ErroConferencia) as erro:
        b.conferir_pasta(arquivos, dados)
    texto = str(erro.value)
    assert "3 problema(s)" in texto and "s01.edf: não existe" in texto and "esperado 3000" in texto \
        and "extra.edf" in texto


def test_impressao_digital_nao_depende_da_ordem_e_muda_com_um_md5():
    a = [b.Arquivo("SZ", "x.edf", 1, "a" * 32, ""), b.Arquivo("SZ", "y.edf", 1, "b" * 32, "")]
    assert b.impressao_digital(a) == b.impressao_digital(a[::-1])
    assert b.impressao_digital(a) != b.impressao_digital([a[0], b.Arquivo("SZ", "y.edf", 1, "c" * 32, "")])


def test_lista_figshare_pula_sem_md5_e_prefixa_nomes_repetidos():
    artigo = dict(files=[dict(id=1, name="H S1 EC.edf", size=10, computed_md5="a" * 32, download_url="u1"),
                         dict(id=2, name="H S15 EO.edf", size=11, computed_md5="b" * 32, download_url="u2"),
                         dict(id=3, name="H S15 EO.edf", size=12, computed_md5="c" * 32, download_url="u3"),
                         dict(id=4, name="H S12 EC.edf", size=0, computed_md5="", download_url="u4")])
    assert [a.arquivo for a in b.lista_figshare(artigo)] == ["H S1 EC.edf", "2_H S15 EO.edf", "3_H S15 EO.edf"]


def test_lista_repod():
    versao = dict(files=[dict(label="h01.edf", dataFile=dict(id=267, filesize=5, md5="d" * 32))])
    (a,) = b.lista_repod(versao)
    assert (a.dataset, a.arquivo, a.bytes, a.url) == ("SZ", "h01.edf", 5, b.REPOD_ARQUIVO.format(id=267))
