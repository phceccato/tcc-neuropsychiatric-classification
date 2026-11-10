"""Auditoria das gravações de olhos fechados (EC) dos dois datasets.

Aplica as checagens C1–C9 de docs/01_auditoria.md a cada arquivo EC e grava
outputs/01_auditoria/auditoria_ec.csv (uma linha por arquivo) e resumo.md (n final).

O campo de paciente do EDF traz nome e data de nascimento; só um hash dele é
guardado, o suficiente para comparar cabeçalhos entre arquivos (C4).
"""
from __future__ import annotations

import argparse
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import mne
import numpy as np
import pandas as pd

CANAIS = ("Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
          "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2")
FS_ESPERADA = {"SZ": 250.0, "MDD": 256.0}
PADRAO_ARQUIVOS = {"SZ": ("Schizophrenia_dataset", "*.edf"),
                   "MDD": ("Depression_dataset", "* EC.edf")}

DURACAO_MIN_S = 122.0  # 60 janelas de 4 s com 50% de overlap: (60 - 1) * 2 + 4
PLANO_DESVIO_UV = 0.5
PLANO_MIN_S = 10
PLANO_FRACAO_RUIM = 0.20
SATURACAO_FRACAO_RUIM = 0.01

_GRAVIDADE = {"INCLUIR": 0, "SINALIZAR": 1, "EXCLUIR": 2}
_UNIDADES_UV = {"uV": 1.0, "µV": 1.0, "mV": 1e3, "V": 1e6}


@dataclass
class EDF:
    paciente: str
    gravacao: str
    rotulos: list[str]
    unidades: list[str]
    fs: list[float]
    pmin: np.ndarray
    pmax: np.ndarray
    dmin: np.ndarray
    dmax: np.ndarray
    digitais: list[np.ndarray]

    def microvolts(self, i: int) -> np.ndarray:
        ganho = (self.pmax[i] - self.pmin[i]) / (self.dmax[i] - self.dmin[i])
        fisico = (self.digitais[i].astype(np.float64) - self.dmin[i]) * ganho + self.pmin[i]
        return fisico * _UNIDADES_UV[self.unidades[i]]


def ler_edf(caminho: Path) -> EDF:
    """Lê um EDF mantendo os valores digitais (necessários para C5 e C8)."""
    bruto = Path(caminho).read_bytes()
    bytes_cabecalho = int(bruto[184:192])
    n_registros = int(bruto[236:244])
    duracao_registro = float(bruto[244:252])
    ns = int(bruto[252:256])

    def campo(deslocamento: int, largura: int) -> list[str]:
        inicio = 256 + ns * deslocamento
        return [bruto[inicio + i * largura:inicio + (i + 1) * largura].decode("latin-1").strip()
                for i in range(ns)]

    rotulos = campo(0, 16)
    unidades = campo(96, 8)
    pmin, pmax, dmin, dmax = (np.array(campo(d, 8), dtype=float) for d in (104, 112, 120, 128))
    amostras = np.array(campo(216, 8), dtype=int)

    por_registro = int(amostras.sum())
    dados = np.frombuffer(bruto, dtype="<i2", offset=bytes_cabecalho)
    n_completos = len(dados) // por_registro
    if n_registros < 0 or n_registros > n_completos:
        n_registros = n_completos
    dados = dados[:n_registros * por_registro].reshape(n_registros, por_registro)
    limites = np.concatenate([[0], np.cumsum(amostras)])
    digitais = [dados[:, limites[i]:limites[i + 1]].reshape(-1) for i in range(ns)]

    return EDF(
        paciente=bruto[8:88].decode("latin-1").strip(),
        gravacao=bruto[88:168].decode("latin-1").strip(),
        rotulos=rotulos, unidades=unidades,
        fs=[float(n / duracao_registro) for n in amostras],
        pmin=pmin, pmax=pmax, dmin=dmin, dmax=dmax, digitais=digitais,
    )


def normalizar_canal(rotulo: str) -> str | None:
    """'EEG Fp1-LE' ou 'FP1' → 'Fp1'; canais fora do 10–20 → None."""
    nome = rotulo.removeprefix("EEG ").split("-")[0].strip().lower()
    return next((c for c in CANAIS if c.lower() == nome), None)


def identificar(dataset: str, nome_arquivo: str) -> tuple[str, str]:
    """Grupo (HC, SZ ou MDD) e sujeito a partir do nome do arquivo."""
    if dataset == "SZ":
        sujeito = Path(nome_arquivo).stem
        return ("SZ" if sujeito.startswith("s") else "HC"), sujeito
    m = re.match(r"(H|MDD) (S\d+)\s+EC\.edf$", nome_arquivo)
    if not m:
        raise ValueError(f"nome inesperado: {nome_arquivo!r}")
    return ("HC" if m[1] == "H" else "MDD"), m[2]


def zeros_no_fim(digitais: np.ndarray) -> int:
    """Amostras finais em que todos os canais valem exatamente 0 (digital)."""
    nao_zero = np.flatnonzero(np.any(digitais != 0, axis=0))
    return digitais.shape[1] if len(nao_zero) == 0 else digitais.shape[1] - 1 - int(nao_zero[-1])


def segmentos_planos(x_uv: np.ndarray, fs: float) -> list[tuple[float, float]]:
    """Trechos de >= PLANO_MIN_S s em que o desvio de cada janela de 1 s fica abaixo do limiar."""
    passo = int(round(fs))
    n = len(x_uv) // passo
    if n == 0:
        return []
    plano = x_uv[:n * passo].reshape(n, passo).std(axis=1) < PLANO_DESVIO_UV
    trechos, inicio = [], None
    for k, p in enumerate(np.append(plano, False)):
        if p and inicio is None:
            inicio = k
        elif not p and inicio is not None:
            if k - inicio >= PLANO_MIN_S:
                trechos.append((float(inicio), float(k)))
            inicio = None
    return trechos


def fracao_saturada(digitais: np.ndarray, dmin: float, dmax: float) -> float:
    return float(np.mean((digitais <= dmin) | (digitais >= dmax))) if len(digitais) else 0.0


def _condicao(gravacao: str) -> str:
    m = re.search(r"_(EC|EO|TASK|P300)\b", gravacao, re.IGNORECASE)
    if not m:
        return ""
    return "TASK" if m[1].upper() == "P300" else m[1].upper()


def auditar_arquivo(dataset: str, caminho: Path) -> dict:
    grupo, sujeito = identificar(dataset, caminho.name)
    reg = dict(dataset=dataset, arquivo=caminho.name, grupo=grupo, sujeito=sujeito,
               sha256=hashlib.sha256(caminho.read_bytes()).hexdigest(), erro="",
               fs=np.nan, duracao_bruta_s=np.nan, zeros_cortados_s=np.nan, duracao_util_s=np.nan,
               sinal_sha256="", paciente_hash="", condicao_cabecalho="",
               plano_frac_max=np.nan, saturacao_frac_max=np.nan,
               canais_ruins="", segmentos_planos="")
    try:
        mne.io.read_raw_edf(caminho, preload=False, verbose="ERROR")
        edf = ler_edf(caminho)
    except Exception as e:  # C1: arquivo ilegível
        reg["erro"] = f"leitura falhou: {e}"
        return reg

    reg["paciente_hash"] = hashlib.sha256(edf.paciente.encode()).hexdigest()[:12] if edf.paciente else ""
    reg["condicao_cabecalho"] = _condicao(edf.gravacao)

    indice = {}
    for i, rotulo in enumerate(edf.rotulos):
        canal = normalizar_canal(rotulo)
        if canal and canal not in indice:
            indice[canal] = i
    faltando = [c for c in CANAIS if c not in indice]
    if faltando:
        reg["erro"] = "sem canais " + ",".join(faltando)
        return reg
    taxas = {edf.fs[indice[c]] for c in CANAIS}
    if taxas != {FS_ESPERADA[dataset]}:
        reg["erro"] = f"frequência {sorted(taxas)} Hz, esperado {FS_ESPERADA[dataset]}"
        return reg

    fs = FS_ESPERADA[dataset]
    digitais = np.stack([edf.digitais[indice[c]] for c in CANAIS])
    n_zeros = zeros_no_fim(digitais)
    n_util = digitais.shape[1] - n_zeros
    util = digitais[:, :n_util]
    reg.update(fs=fs, duracao_bruta_s=digitais.shape[1] / fs, zeros_cortados_s=n_zeros / fs,
               duracao_util_s=n_util / fs,
               sinal_sha256=hashlib.sha256(np.ascontiguousarray(util).tobytes()).hexdigest())

    ruins, trechos, fracoes_planas, fracoes_saturadas = [], [], [], []
    for k, canal in enumerate(CANAIS):
        i = indice[canal]
        segmentos = segmentos_planos(edf.microvolts(i)[:n_util], fs)
        fracao_plana = sum(f - a for a, f in segmentos) / reg["duracao_util_s"] if n_util else 0.0
        saturada = fracao_saturada(util[k], edf.dmin[i], edf.dmax[i])
        fracoes_planas.append(fracao_plana)
        fracoes_saturadas.append(saturada)
        trechos += [f"{canal}:{a:.0f}-{f:.0f}" for a, f in segmentos]
        if fracao_plana > PLANO_FRACAO_RUIM or saturada > SATURACAO_FRACAO_RUIM:
            ruins.append(canal)
    reg.update(plano_frac_max=max(fracoes_planas), saturacao_frac_max=max(fracoes_saturadas),
               canais_ruins=";".join(ruins), segmentos_planos=";".join(trechos))
    return reg


def _numero(sujeito: str) -> int:
    return int(re.sub(r"\D", "", sujeito))


def decidir(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica C1–C9 e devolve o DataFrame com decisao, motivos e subject_uid."""
    df = df.copy().reset_index(drop=True)
    df["subject_uid"] = df["dataset"] + "-" + df["grupo"] + "-" + df["sujeito"]
    motivos = {i: [] for i in df.index}
    decisao = {i: "INCLUIR" for i in df.index}

    def marcar(i: int, nivel: str, motivo: str) -> None:
        motivos[i].append(motivo)
        if _GRAVIDADE[nivel] > _GRAVIDADE[decisao[i]]:
            decisao[i] = nivel

    legiveis = df.index[df["erro"].fillna("") == ""]
    for i in df.index.difference(legiveis):
        marcar(i, "EXCLUIR", f"C1: {df.at[i, 'erro']}")

    for i in legiveis:
        if df.at[i, "duracao_util_s"] < DURACAO_MIN_S:
            marcar(i, "EXCLUIR", f"C6: duração útil {df.at[i, 'duracao_util_s']:.0f} s < {DURACAO_MIN_S:.0f} s")
        ruins = [c for c in str(df.at[i, "canais_ruins"] or "").split(";") if c]
        if len(ruins) == 1:
            marcar(i, "SINALIZAR", f"C9: canal {ruins[0]} ruim, interpolar")
        elif len(ruins) >= 2:
            marcar(i, "EXCLUIR", f"C9: {len(ruins)} canais ruins ({', '.join(ruins)})")

    # C2/C3: grupos de arquivos com o mesmo SHA-256 ou o mesmo sinal (união dos dois)
    pai = {i: i for i in legiveis}

    def raiz(i: int) -> int:
        while pai[i] != i:
            i = pai[i]
        return i

    for coluna in ("sha256", "sinal_sha256"):
        for _, idx in df.loc[legiveis].groupby(["dataset", coluna]).groups.items():
            idx = list(idx)
            for j in idx[1:]:
                pai[raiz(j)] = raiz(idx[0])
    grupos: dict[int, list[int]] = {}
    for i in legiveis:
        grupos.setdefault(raiz(i), []).append(i)

    em_duplicata = set()
    for membros in grupos.values():
        if len(membros) < 2:
            continue
        em_duplicata.update(membros)
        nomes = ", ".join(df.loc[membros, "arquivo"])
        if df.loc[membros, "grupo"].nunique() > 1:
            for i in membros:
                marcar(i, "EXCLUIR", f"C2/C3: mesma gravação com rótulos diferentes ({nomes})")
            continue
        mantido = min(membros, key=lambda i: _numero(df.at[i, "sujeito"]))
        for i in membros:
            if i == mantido:
                marcar(i, "SINALIZAR", f"C2/C3: mesma gravação que {nomes}; conta como uma pessoa")
            else:
                regra = "C2" if df.at[i, "sha256"] == df.at[mantido, "sha256"] else "C3"
                marcar(i, "EXCLUIR", f"{regra}: duplicata de {df.at[mantido, 'arquivo']} (mesma pessoa)")
                df.at[i, "subject_uid"] = df.at[mantido, "subject_uid"]

    # C4: cabeçalho igual entre sujeitos diferentes, ou indicando outra condição
    sem_dup = [i for i in legiveis if i not in em_duplicata and df.at[i, "paciente_hash"]]
    for _, idx in df.loc[sem_dup].groupby(["dataset", "paciente_hash"]).groups.items():
        idx = list(idx)
        if len(idx) > 1:
            for i in idx:
                outros = ", ".join(df.at[j, "arquivo"] for j in idx if j != i)
                marcar(i, "SINALIZAR", f"C4: cabeçalho com o mesmo paciente de {outros}")
    for i in legiveis:
        condicao = df.at[i, "condicao_cabecalho"]
        if condicao and condicao != "EC":
            marcar(i, "SINALIZAR", f"C4: cabeçalho indica condição {condicao}")

    df["decisao"] = [decisao[i] for i in df.index]
    df["motivos"] = [" | ".join(motivos[i]) for i in df.index]
    return df


def resumo(df: pd.DataFrame) -> str:
    linhas = ["# Auditoria EC — resumo", "",
              "Critérios em `docs/01_auditoria.md`. Uma linha por arquivo em `auditoria_ec.csv`.", "",
              "## n final", "",
              "| Task | Classe | Arquivos | INCLUIR | SINALIZAR | EXCLUIR | Pessoas no n final |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    for (dataset, grupo), g in df.groupby(["dataset", "grupo"], sort=False):
        n = g.loc[g["decisao"] != "EXCLUIR", "subject_uid"].nunique()
        contagem = g["decisao"].value_counts()
        linhas.append(f"| {dataset} | {grupo} | {len(g)} | {contagem.get('INCLUIR', 0)} | "
                      f"{contagem.get('SINALIZAR', 0)} | {contagem.get('EXCLUIR', 0)} | {n} |")
    linhas += ["", "## Exclusões e sinalizações", "",
               "| Arquivo | Task | Decisão | Motivos |", "| --- | --- | --- | --- |"]
    casos = df[df["decisao"] != "INCLUIR"].sort_values(["dataset", "decisao", "arquivo"])
    linhas += [f"| {r.arquivo} | {r.dataset} | {r.decisao} | {r.motivos} |" for r in casos.itertuples()]
    curtos = df[(df["segmentos_planos"] != "") & (df["decisao"] != "EXCLUIR")]
    if len(curtos):
        linhas += ["", "## Trechos planos que ficam para a etapa de janelas", "",
                   "| Arquivo | Trechos (canal:início-fim, em s) |", "| --- | --- |"]
        linhas += [f"| {r.arquivo} | {r.segmentos_planos} |" for r in curtos.itertuples()]
    return "\n".join(linhas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/01_auditoria"))
    args = parser.parse_args()

    registros = []
    for dataset, (pasta, padrao) in PADRAO_ARQUIVOS.items():
        for caminho in sorted((args.dados / pasta).glob(padrao)):
            registros.append(auditar_arquivo(dataset, caminho))
            print(f"{dataset} {caminho.name}")
    df = decidir(pd.DataFrame(registros))

    args.saida.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.saida / "auditoria_ec.csv", index=False)
    (args.saida / "resumo.md").write_text(resumo(df), encoding="utf-8")
    print(f"\n{len(df)} arquivos auditados → {args.saida}")


if __name__ == "__main__":
    main()
