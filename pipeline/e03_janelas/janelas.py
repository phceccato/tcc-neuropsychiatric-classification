"""Etapa de janelas: filtro de 2–30 Hz, estatísticas do z-score, janelas de 4 s e rejeição de artefato.

Segue docs/00_proposta-tcc.md ("Pré-processamento e janelas"):

- passa-banda de 2–30 Hz na gravação inteira, antes de cortar as janelas;
- z-score por canal com média e desvio da própria pessoa, nas amostras das janelas aceitas;
- janelas de 4 s com 50% de overlap; as que começam em múltiplos de 4 s formam o
  subconjunto sem overlap usado na validação e no teste;
- janela rejeitada se o pico a pico, em qualquer canal, passar de 150 µV.

O sinal é gravado contínuo e em µV; as janelas são cortadas pelo carregador a partir de
janelas.csv. O z-score fica para o carregamento (pipeline/e07_DTSF/dados.py), que alimenta o ramo
temporal do DTSF e as características temporais dos baselines clássicos; o ramo espectral usa a
potência absoluta e o limiar de artefato está em µV.

Lê outputs/02_harmonizacao/ e grava em outputs/03_janelas/ um .npz por pessoa (x, media, desvio),
janelas.csv (uma linha por janela) e resumo.md (descartes por pessoa e por classe).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

PASSA_ALTA_HZ, PASSA_BAIXA_HZ = 2.0, 30.0
ORDEM_BUTTER = 4                 # por passada; sosfiltfilt dá fase zero
JANELA_S, PASSO_S = 4.0, 2.0     # 50% de overlap
LIMIAR_PICO_A_PICO_UV = 150.0
JANELAS_POR_PESSOA = 60          # sorteadas a cada época no treino
LIMIARES_COMPARADOS_UV = (100.0, 150.0, 200.0, 300.0)


def filtrar(x: np.ndarray, fs: float) -> np.ndarray:
    sos = butter(ORDEM_BUTTER, [PASSA_ALTA_HZ, PASSA_BAIXA_HZ], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, np.asarray(x, dtype=np.float64), axis=-1).astype(np.float32)


def estatisticas(x: np.ndarray, inicios_aceitos: np.ndarray, tamanho: int) -> tuple[np.ndarray, np.ndarray]:
    """Média e desvio de cada canal nas amostras das janelas aceitas (para o z-score do ramo temporal do DTSF e dos clássicos).

    As amostras das janelas rejeitadas ficam de fora: um único artefato de milhares de µV inflaria
    o desvio e encolheria todas as janelas limpas da pessoa.
    """
    mascara = np.zeros(x.shape[1], dtype=bool)
    for i in inicios_aceitos:
        mascara[i:i + tamanho] = True
    aceitas = np.asarray(x[:, mascara], dtype=np.float64)
    return aceitas.mean(axis=1), aceitas.std(axis=1)


def inicios_janelas(n_amostras: int, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Inícios das janelas com 50% de overlap e a marca das que formam a grade sem overlap."""
    tamanho, passo = int(JANELA_S * fs), int(PASSO_S * fs)
    inicios = np.arange(0, n_amostras - tamanho + 1, passo)
    return inicios, (inicios // passo) % 2 == 0


def pico_a_pico(x: np.ndarray, inicios: np.ndarray, tamanho: int) -> np.ndarray:
    """Maior amplitude pico a pico entre os canais, em cada janela."""
    return np.array([np.ptp(x[:, i:i + tamanho], axis=1).max() for i in inicios], dtype=float)


def toca_trechos(inicios: np.ndarray, tamanho: int, fs: float, trechos: list[str]) -> np.ndarray:
    """Janelas que se sobrepõem a algum trecho 'canal:início-fim' (em s)."""
    toca = np.zeros(len(inicios), dtype=bool)
    for trecho in trechos:
        a, b = (float(v) * fs for v in trecho.split(":")[1].split("-"))
        toca |= (inicios < b) & (inicios + tamanho > a)
    return toca


def indexar(x: np.ndarray, fs: float, trechos: list[str]) -> pd.DataFrame:
    """Uma linha por janela: início, grade sem overlap, pico a pico e rejeição."""
    tamanho = int(JANELA_S * fs)
    inicios, sem_overlap = inicios_janelas(x.shape[1], fs)
    pp = pico_a_pico(x, inicios, tamanho)
    plano = toca_trechos(inicios, tamanho, fs, trechos)
    amplitude = pp > LIMIAR_PICO_A_PICO_UV
    motivo = np.where(plano, "trecho_plano", np.where(amplitude, "amplitude", ""))
    return pd.DataFrame(dict(inicio_amostra=inicios, inicio_s=inicios / fs, sem_overlap=sem_overlap,
                             pico_a_pico_uv=pp, rejeitada=plano | amplitude, motivo=motivo))


def _lista(valor) -> list[str]:
    return [v for v in str(valor if pd.notna(valor) else "").split(";") if v]


def _porcentagem(serie: pd.Series) -> str:
    return f"{100 * serie.mean():.1f}%"


def resumo(janelas: pd.DataFrame) -> str:
    linhas = ["# Janelas — resumo", "",
              f"Passa-banda {PASSA_ALTA_HZ:g}–{PASSA_BAIXA_HZ:g} Hz (Butterworth ordem {ORDEM_BUTTER}, sosfiltfilt) "
              f"na gravação inteira; janelas de {JANELA_S:g} s a cada {PASSO_S:g} s; rejeição com pico a pico "
              f"> {LIMIAR_PICO_A_PICO_UV:g} µV em qualquer canal.", "",
              "## Descarte por classe", "",
              "Taxa de janelas rejeitadas (grade com 50% de overlap). Descarte desigual entre classes vira viés.", "",
              "| Task | Classe | Pessoas | Janelas | Rejeitadas | Mediana por pessoa | Máximo por pessoa |",
              "| --- | --- | --- | --- | --- | --- | --- |"]
    por_pessoa = janelas.groupby(["dataset", "grupo", "subject_uid"], sort=False)["rejeitada"].mean()
    for (dataset, grupo), g in janelas.groupby(["dataset", "grupo"], sort=False):
        p = por_pessoa.loc[(dataset, grupo)]
        linhas.append(f"| {dataset} | {grupo} | {g['subject_uid'].nunique()} | {len(g)} | "
                      f"{_porcentagem(g['rejeitada'])} | {100 * p.median():.1f}% | {100 * p.max():.1f}% |")

    linhas += ["", "## Conferência do limiar", "",
               "Taxa de rejeição por amplitude se o limiar fosse outro. O limiar usado é o da proposta "
               f"({LIMIAR_PICO_A_PICO_UV:g} µV); esta tabela só mostra quanto a escolha pesa.", "",
               "| Task | Classe | Pico a pico mediano (µV) | P95 (µV) | "
               + " | ".join(f"> {l:g} µV" for l in LIMIARES_COMPARADOS_UV) + " |",
               "| --- | --- | --- | --- | " + " | ".join("---" for _ in LIMIARES_COMPARADOS_UV) + " |"]
    for (dataset, grupo), g in janelas.groupby(["dataset", "grupo"], sort=False):
        pp = g["pico_a_pico_uv"]
        linhas.append(f"| {dataset} | {grupo} | {pp.median():.0f} | {pp.quantile(0.95):.0f} | "
                      + " | ".join(_porcentagem(pp > l) for l in LIMIARES_COMPARADOS_UV) + " |")

    aceitas = janelas[~janelas["rejeitada"]]
    contagem = pd.DataFrame({
        "treino": aceitas.groupby("subject_uid").size(),
        "teste": aceitas[aceitas["sem_overlap"]].groupby("subject_uid").size(),
    }).reindex(janelas["subject_uid"].unique()).fillna(0).astype(int)
    poucas = contagem[contagem["treino"] < JANELAS_POR_PESSOA]
    linhas += ["", "## Janelas aceitas por pessoa", "",
               f"Mínimo: {contagem['treino'].min()} na grade de treino (50%) e {contagem['teste'].min()} na de "
               f"teste (0%). Pessoas com menos de {JANELAS_POR_PESSOA} janelas aceitas no treino: {len(poucas)}."]
    if len(poucas):
        linhas += ["", "| Pessoa | Aceitas no treino | Aceitas no teste |", "| --- | --- | --- |"]
        linhas += [f"| {uid} | {r.treino} | {r.teste} |" for uid, r in poucas.iterrows()]
    return "\n".join(linhas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--entrada", type=Path, default=Path("outputs/02_harmonizacao"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/03_janelas"))
    args = parser.parse_args()

    manifesto = pd.read_csv(args.entrada / "manifesto.csv")
    args.saida.mkdir(parents=True, exist_ok=True)
    tabelas = []
    for r in manifesto.itertuples():
        harmonizado = np.load(args.entrada / r.npz)
        fs = float(harmonizado["fs"])
        x = filtrar(harmonizado["x"], fs)
        tabela = indexar(x, fs, _lista(r.segmentos_planos))
        aceitas = tabela.loc[~tabela["rejeitada"], "inicio_amostra"].to_numpy()
        media, desvio = estatisticas(x, aceitas, int(JANELA_S * fs))
        np.savez(args.saida / r.npz, x=x, fs=fs, media=media.astype(np.float32),
                 desvio=desvio.astype(np.float32), canais=harmonizado["canais"], subject_uid=r.subject_uid)
        tabela.insert(0, "subject_uid", r.subject_uid)
        tabela.insert(1, "dataset", r.dataset)
        tabela.insert(2, "grupo", r.grupo)
        tabelas.append(tabela)
        print(f"{r.subject_uid}: {len(tabela)} janelas, {int(tabela['rejeitada'].sum())} rejeitadas")

    janelas = pd.concat(tabelas, ignore_index=True)
    janelas.to_csv(args.saida / "janelas.csv", index=False)
    (args.saida / "resumo.md").write_text(resumo(janelas), encoding="utf-8")
    print(f"\n{len(janelas)} janelas de {janelas['subject_uid'].nunique()} pessoas → {args.saida}")


if __name__ == "__main__":
    main()
