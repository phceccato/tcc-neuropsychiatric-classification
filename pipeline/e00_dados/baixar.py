"""Baixa as duas bases de EEG das fontes oficiais e confere cada arquivo pelo MD5 e pelo tamanho.

- SZ: Olejarczyk e Jernajczyk (2017), "EEG in schizophrenia", RepOD, versão 1.0,
  doi:10.18150/repod.0107441 → dados/Schizophrenia_dataset/ (28 EDF).
- MDD: Mumtaz (2016), "MDD Patients and Healthy Controls EEG Data (New)", figshare, versão 2,
  doi:10.6084/m9.figshare.4244171.v2, CC BY 4.0 → dados/Depression_dataset/ (181 EDF).

A lista do que baixar fica versionada em pipeline/e00_dados/arquivos.csv (dataset, arquivo, bytes, md5, url),
gerada uma vez das APIs das duas fontes (`--gerar-lista`) e fixa a partir daí: o download não depende de a
listagem da fonte mudar. Ficam fora da lista os 12 arquivos do figshare que a API lista sem MD5 e cujo link
de download nunca funcionou (docs/01_auditoria.md). O nome "H S15 EO.edf" aparece duas vezes no figshare;
quando um nome se repete, o arquivo local leva o id do figshare na frente ("6921143_H S15 EO.edf").

O download é retomável: um arquivo que já existe com o MD5 certo é pulado. Cada arquivo é gravado em
<nome>.part e só é renomeado depois de conferido; um MD5 ou tamanho errado apaga o .part e para com erro.

`--conferir` não baixa nada: confere a pasta dados/ contra a lista (presença, tamanho e MD5 de cada arquivo,
número de arquivos e total de bytes por dataset) e imprime, por dataset, uma impressão digital (MD5 da lista
ordenada "arquivo:md5"). Duas máquinas com a mesma impressão digital têm exatamente os mesmos dados.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import os
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

LISTA = Path(__file__).with_name("arquivos.csv")
DADOS = Path("dados")
PASTAS = {"SZ": "Schizophrenia_dataset", "MDD": "Depression_dataset"}
FIGSHARE_API = "https://api.figshare.com/v2/articles/4244171/versions/2"
REPOD_API = ("https://repod.icm.edu.pl/api/datasets/:persistentId/versions/1.0"
             "?persistentId=doi:10.18150/repod.0107441")
REPOD_ARQUIVO = "https://repod.icm.edu.pl/api/access/datafile/{id}"
AGENTE = "tcc-neuropsychiatric-classification/1.0 (download dos dados)"
TENTATIVAS = 3
BLOCO = 1 << 20


@dataclass(frozen=True)
class Arquivo:
    dataset: str
    arquivo: str
    bytes: int
    md5: str
    url: str

    def caminho(self, dados: Path = DADOS) -> Path:
        return dados / PASTAS[self.dataset] / self.arquivo


class ErroConferencia(RuntimeError):
    """Um arquivo baixado ou local não confere com a lista."""


def md5_arquivo(caminho: Path) -> str:
    h = hashlib.md5()
    with open(caminho, "rb") as f:
        for bloco in iter(lambda: f.read(BLOCO), b""):
            h.update(bloco)
    return h.hexdigest()


def _json(url: str) -> dict:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": AGENTE}), timeout=60) as r:
        return json.load(r)


def lista_figshare(artigo: dict) -> list[Arquivo]:
    """Arquivos do figshare com MD5 (os sem MD5 nunca ficaram disponíveis); nome repetido leva o id na frente."""
    contagem = collections.Counter(f["name"] for f in artigo["files"])
    return [Arquivo("MDD", f["name"] if contagem[f["name"]] == 1 else f"{f['id']}_{f['name']}", int(f["size"]),
                    f["computed_md5"], f["download_url"])
            for f in artigo["files"] if f.get("computed_md5")]


def lista_repod(versao: dict) -> list[Arquivo]:
    return [Arquivo("SZ", f["label"], int(f["dataFile"]["filesize"]), f["dataFile"]["md5"],
                    REPOD_ARQUIVO.format(id=f["dataFile"]["id"])) for f in versao["files"]]


def gravar_lista(arquivos: list[Arquivo], destino: Path = LISTA) -> None:
    with open(destino, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "arquivo", "bytes", "md5", "url"])
        for a in sorted(arquivos, key=lambda a: (a.dataset != "SZ", a.arquivo)):
            w.writerow([a.dataset, a.arquivo, a.bytes, a.md5, a.url])


def ler_lista(origem: Path = LISTA) -> list[Arquivo]:
    with open(origem, encoding="utf-8") as f:
        return [Arquivo(r["dataset"], r["arquivo"], int(r["bytes"]), r["md5"], r["url"]) for r in csv.DictReader(f)]


def conferir_arquivo(a: Arquivo, caminho: Path) -> None:
    if not caminho.exists():
        raise ErroConferencia(f"{a.dataset} {a.arquivo}: não existe em {caminho.parent}")
    tamanho = caminho.stat().st_size
    if tamanho != a.bytes:
        raise ErroConferencia(f"{a.dataset} {a.arquivo}: {tamanho} bytes, esperado {a.bytes}")
    md5 = md5_arquivo(caminho)
    if md5 != a.md5:
        raise ErroConferencia(f"{a.dataset} {a.arquivo}: MD5 {md5}, esperado {a.md5}")


def baixar_arquivo(a: Arquivo, dados: Path = DADOS, tentativas: int = TENTATIVAS) -> bool:
    """Baixa e confere um arquivo; devolve False se ele já existia conferido."""
    destino = a.caminho(dados)
    if destino.exists():
        try:
            conferir_arquivo(a, destino)
            return False
        except ErroConferencia:
            pass
    destino.parent.mkdir(parents=True, exist_ok=True)
    parcial = destino.with_name(destino.name + ".part")
    for tentativa in range(1, tentativas + 1):
        try:
            pedido = urllib.request.Request(a.url, headers={"User-Agent": AGENTE})
            with urllib.request.urlopen(pedido, timeout=120) as r, open(parcial, "wb") as f:
                for bloco in iter(lambda: r.read(BLOCO), b""):
                    f.write(bloco)
            conferir_arquivo(a, parcial)
            os.replace(parcial, destino)
            return True
        except (OSError, ErroConferencia) as erro:
            parcial.unlink(missing_ok=True)
            if tentativa == tentativas:
                raise ErroConferencia(f"{a.dataset} {a.arquivo}: falhou em {tentativas} tentativas ({erro})") from erro
            time.sleep(2 * tentativa)
    return True


def impressao_digital(arquivos: list[Arquivo]) -> str:
    """MD5 da lista ordenada "arquivo:md5": igual em duas máquinas só se os dados forem os mesmos."""
    linhas = sorted(f"{a.arquivo}:{a.md5}" for a in arquivos)
    return hashlib.md5("\n".join(linhas).encode("utf-8")).hexdigest()


def resumo(arquivos: list[Arquivo]) -> dict[str, dict]:
    por_dataset: dict[str, list[Arquivo]] = collections.defaultdict(list)
    for a in arquivos:
        por_dataset[a.dataset].append(a)
    return {d: dict(arquivos=len(v), bytes=sum(a.bytes for a in v), impressao=impressao_digital(v))
            for d, v in por_dataset.items()}


def conferir_pasta(arquivos: list[Arquivo], dados: Path = DADOS) -> dict[str, dict]:
    """Confere cada arquivo da lista na pasta e a soma por dataset; devolve o resumo calculado dos arquivos locais.

    Arquivos .edf na pasta que não estão na lista também contam como erro (dados a mais mudam o pipeline).
    """
    erros = []
    locais = []
    for a in arquivos:
        try:
            conferir_arquivo(a, a.caminho(dados))
            locais.append(a)
        except ErroConferencia as erro:
            erros.append(str(erro))
    esperados = {a.caminho(dados) for a in arquivos}
    for dataset, pasta in PASTAS.items():
        if any(a.dataset == dataset for a in arquivos):
            sobra = sorted(p.name for p in (dados / pasta).glob("*.edf") if p not in esperados)
            erros += [f"{dataset} {nome}: está na pasta mas não na lista" for nome in sobra]
    if erros:
        raise ErroConferencia(f"{len(erros)} problema(s):\n" + "\n".join(erros))
    calculado, esperado = resumo(locais), resumo(arquivos)
    if calculado != esperado:
        raise ErroConferencia(f"resumo local {calculado} difere do esperado {esperado}")
    return calculado


def _imprimir_resumo(r: dict[str, dict]) -> None:
    for d, v in r.items():
        bytes_ = f"{v['bytes']:,}".replace(",", ".")
        mib = f"{v['bytes'] / 2**20:.1f}".replace(".", ",")
        print(f"{d}: {v['arquivos']} arquivos, {bytes_} bytes ({mib} MiB), impressão digital {v['impressao']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dados", type=Path, default=DADOS)
    parser.add_argument("--datasets", nargs="+", default=list(PASTAS), choices=list(PASTAS))
    parser.add_argument("--conferir", action="store_true", help="só confere a pasta, sem baixar")
    parser.add_argument("--gerar-lista", action="store_true",
                        help="regrava arquivos.csv a partir das APIs das fontes (não baixa)")
    args = parser.parse_args()

    if args.gerar_lista:
        arquivos = lista_repod(_json(REPOD_API)["data"]) + lista_figshare(_json(FIGSHARE_API))
        gravar_lista(arquivos)
        print(f"{LISTA}: {len(arquivos)} arquivos")
        _imprimir_resumo(resumo(arquivos))
        return

    arquivos = [a for a in ler_lista() if a.dataset in args.datasets]
    if not args.conferir:
        total = len(arquivos)
        for i, a in enumerate(arquivos, 1):
            novo = baixar_arquivo(a, args.dados)
            print(f"[{i}/{total}] {a.dataset} {a.arquivo}: {'baixado' if novo else 'já existia'}", flush=True)
    try:
        r = conferir_pasta(arquivos, args.dados)
    except ErroConferencia as erro:
        print(f"ERRO: {erro}", file=sys.stderr)
        sys.exit(1)
    print("Todos os arquivos conferem (presença, tamanho e MD5).")
    _imprimir_resumo(r)


if __name__ == "__main__":
    main()
