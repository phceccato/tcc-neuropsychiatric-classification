"""Harmonização das gravações EC incluídas pela auditoria.

Aplica o mesmo pipeline aos dois datasets, gravação a gravação, como em
docs/00_proposta-tcc.md ("Harmonização"):

    µV → corta zeros do fim (C5) → 19 canais 10–20 → interpola canal ruim (C9)
    → referência média (CAR) → 250 Hz → notch 50 Hz → passa-banda 1–40 Hz → float32

A interpolação vem antes da CAR para que o canal ruim não contamine a média.
A faixa de análise (2–30 Hz) é aplicada depois, na etapa de janelas.

Lê outputs/01_auditoria/auditoria_ec.csv e grava em outputs/02_harmonizacao/ um .npz
por pessoa (subject_uid), manifesto.csv e resumo.md.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from pipeline.e01_auditoria.auditar_ec import CANAIS, PADRAO_ARQUIVOS, ler_edf, normalizar_canal

FS_ALVO = 250.0
PASSA_ALTA_HZ, PASSA_BAIXA_HZ = 1.0, 40.0
ORDEM_BUTTER = 4                 # N do butter: passa-banda de ordem 8; sosfiltfilt aplica 2 vezes, com fase zero
NOTCH_HZ, NOTCH_Q = 50.0, 30.0   # rede elétrica de 50 Hz na Polônia e na Malásia
MONTAGEM = "colin27_1020"        # tem os nomes antigos T3–T6

_SOS_PASSA_BANDA = butter(ORDEM_BUTTER, [PASSA_ALTA_HZ, PASSA_BAIXA_HZ], btype="bandpass",
                          fs=FS_ALVO, output="sos")
_B_NOTCH, _A_NOTCH = iirnotch(NOTCH_HZ, NOTCH_Q, fs=FS_ALVO)


def selecionar(df: pd.DataFrame) -> pd.DataFrame:
    """Gravações que entram na análise: tudo que a auditoria não excluiu."""
    return df[df["decisao"] != "EXCLUIR"].reset_index(drop=True)


def carregar_microvolts(caminho: Path, n_amostras: int) -> np.ndarray:
    """Os 19 canais na ordem de CANAIS, em µV, cortados nas n_amostras úteis."""
    edf = ler_edf(caminho)
    indice = {}
    for i, rotulo in enumerate(edf.rotulos):
        canal = normalizar_canal(rotulo)
        if canal and canal not in indice:
            indice[canal] = i
    return np.stack([edf.microvolts(indice[c])[:n_amostras] for c in CANAIS])


def data_gravacao(caminho: Path) -> str:
    """Data de início da gravação no cabeçalho EDF (dd.mm.aa), em ISO."""
    dia, mes, ano = Path(caminho).read_bytes()[168:176].decode("latin-1").split(".")
    seculo = 2000 if int(ano) < 85 else 1900  # convenção do EDF
    return f"{seculo + int(ano):04d}-{int(mes):02d}-{int(dia):02d}"


def interpolar(x: np.ndarray, fs: float, ruins: list[str]) -> np.ndarray:
    """Substitui os canais ruins por interpolação de spline esférica dos outros."""
    if not ruins:
        return x
    info = mne.create_info(list(CANAIS), fs, ch_types="eeg")
    raw = mne.io.RawArray(x * 1e-6, info, verbose="ERROR")
    raw.set_montage(mne.channels.make_standard_montage(MONTAGEM), verbose="ERROR")
    raw.info["bads"] = list(ruins)
    raw.interpolate_bads(reset_bads=True, method={"eeg": "spline"}, verbose="ERROR")
    y = x.copy()
    idx = [CANAIS.index(c) for c in ruins]
    y[idx] = raw.get_data(picks=idx) * 1e6
    return y


def harmonizar(x: np.ndarray, fs: float) -> np.ndarray:
    """CAR → reamostragem para 250 Hz → notch 50 Hz → passa-banda 1–40 Hz."""
    x = x - x.mean(axis=0, keepdims=True)
    if fs != FS_ALVO:
        razao = Fraction(FS_ALVO / fs).limit_denominator(1000)
        x = resample_poly(x, razao.numerator, razao.denominator, axis=1)
    x = filtfilt(_B_NOTCH, _A_NOTCH, x, axis=1)
    return sosfiltfilt(_SOS_PASSA_BANDA, x, axis=1).astype(np.float32)


def processar(x: np.ndarray, fs: float, ruins: list[str]) -> np.ndarray:
    return harmonizar(interpolar(x, fs, ruins), fs)


def _lista(valor) -> list[str]:
    return [v for v in str(valor if pd.notna(valor) else "").split(";") if v]


def _planos_restantes(segmentos: list[str], interpolados: list[str]) -> str:
    """Trechos planos em canais não interpolados (a etapa de janelas descarta as janelas que os tocam)."""
    return ";".join(s for s in segmentos if s.split(":")[0] not in interpolados)


def resumo(manifesto: pd.DataFrame) -> str:
    linhas = ["# Harmonização — resumo", "",
              "Pipeline: µV → corta zeros do fim → 19 canais → interpola canal ruim (spline esférica) → "
              f"CAR → {FS_ALVO:g} Hz (resample_poly) → notch {NOTCH_HZ:g} Hz (Q={NOTCH_Q:g}, filtfilt) → "
              f"Butterworth {PASSA_ALTA_HZ:g}–{PASSA_BAIXA_HZ:g} Hz ordem {ORDEM_BUTTER} (sosfiltfilt) → float32.",
              "", "## Gravações por task e classe", "",
              "| Task | Classe | Pessoas | Duração mínima (s) | Duração mediana (s) | Duração máxima (s) |",
              "| --- | --- | --- | --- | --- | --- |"]
    for (dataset, grupo), g in manifesto.groupby(["dataset", "grupo"], sort=False):
        d = g["duracao_s"]
        linhas.append(f"| {dataset} | {grupo} | {g['subject_uid'].nunique()} | {d.min():.1f} | "
                      f"{d.median():.1f} | {d.max():.1f} |")
    especiais = manifesto[(manifesto["canais_interpolados"] != "") | (manifesto["segmentos_planos"] != "")]
    if len(especiais):
        linhas += ["", "## Canais interpolados e trechos planos restantes", "",
                   "| Arquivo | Canais interpolados | Trechos planos restantes |", "| --- | --- | --- |"]
        linhas += [f"| {r.arquivo} | {r.canais_interpolados or '—'} | {r.segmentos_planos or '—'} |"
                   for r in especiais.itertuples()]
    return "\n".join(linhas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dados", type=Path, default=Path("dados"))
    parser.add_argument("--auditoria", type=Path, default=Path("outputs/01_auditoria/auditoria_ec.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/02_harmonizacao"))
    args = parser.parse_args()

    gravacoes = selecionar(pd.read_csv(args.auditoria))
    if gravacoes["subject_uid"].duplicated().any():
        raise ValueError("mais de uma gravação por subject_uid depois da auditoria")
    args.saida.mkdir(parents=True, exist_ok=True)

    registros = []
    for r in gravacoes.itertuples():
        pasta = PADRAO_ARQUIVOS[r.dataset][0]
        n_amostras = int(round(r.duracao_util_s * r.fs))
        ruins = _lista(r.canais_ruins)
        caminho = args.dados / pasta / r.arquivo
        x = processar(carregar_microvolts(caminho, n_amostras), r.fs, ruins)
        destino = args.saida / f"{r.subject_uid}.npz"
        np.savez(destino, x=x, fs=FS_ALVO, canais=np.array(CANAIS), subject_uid=r.subject_uid,
                 dataset=r.dataset, grupo=r.grupo, arquivo=r.arquivo)
        registros.append(dict(subject_uid=r.subject_uid, dataset=r.dataset, grupo=r.grupo,
                              arquivo=r.arquivo, data_gravacao=data_gravacao(caminho),
                              decisao=r.decisao, fs_original=r.fs,
                              n_amostras=x.shape[1], duracao_s=x.shape[1] / FS_ALVO,
                              canais_interpolados=";".join(ruins),
                              segmentos_planos=_planos_restantes(_lista(r.segmentos_planos), ruins),
                              npz=destino.name))
        print(f"{r.dataset} {r.arquivo} → {destino.name}")

    manifesto = pd.DataFrame(registros)
    manifesto.to_csv(args.saida / "manifesto.csv", index=False)
    (args.saida / "resumo.md").write_text(resumo(manifesto), encoding="utf-8")
    print(f"\n{len(manifesto)} gravações harmonizadas → {args.saida}")


if __name__ == "__main__":
    main()
