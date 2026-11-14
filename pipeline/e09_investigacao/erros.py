"""Experimento 2: perfil das pessoas que o DTSF classifica errado, com foco em s06, h10 e MDD S27.

Para cada pessoa, junta:
- do modelo principal, o D_FA (etapa 07): a probabilidade média nas 10 repetições, o desvio entre repetições e em quantas
  repetições a previsão (limiar 0,5) erra;
- da qualidade do sinal: fração de janelas rejeitadas (etapa 03), duração e canal interpolado (02);
- da data: data e sessão de gravação (outputs/09_investigacao/datas/pessoas.csv);
- do espectro: Welch de cada janela aceita sem overlap, em µV, 2–30 Hz (pipeline/e04_features/psd.py),
  com a média nas janelas e nos 19 canais. Dele saem a potência relativa de δ, θ, α e β, o log10 da
  potência total e a frequência do pico entre 6 e 14 Hz;
- das janelas (experimento 3, se já rodado): a fração das janelas de teste da repetição 0 que o
  modelo retreinado classifica como paciente.

Para as pessoas em foco, cada medida espectral é comparada com as duas classes do dataset pelo
percentil entre os HC e entre os pacientes. Figuras: o espectro médio de cada pessoa em foco sobre
a mediana e o intervalo interquartil de cada classe, e a probabilidade de cada janela ao longo da
gravação.

Grava em outputs/09_investigacao/erros/: pessoas.csv, espectros.npz, espectros.png, janelas.png e resumo.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import percentileofscore

from pipeline.e04_features.psd import BANDAS, LIMITES_HZ, TAMANHO_JANELA, cortar, welch_janelas
from pipeline.e07_DTSF.avaliar import ler_modelo
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL

FOCO = ("SZ-SZ-s06", "SZ-HC-h10", "MDD-MDD-S27")
PICO_HZ = (6.0, 14.0)
LIMIAR = 0.5
MEDIDAS = tuple(f"rel_{b}" for b in BANDAS) + ("log_potencia_total", "pico_hz")


def resumo_espectro(freqs: np.ndarray, espectro: np.ndarray) -> dict[str, float]:
    """Medidas de um espectro médio [freqs] em µV²/Hz."""
    df = freqs[1] - freqs[0]
    total = espectro.sum() * df
    medidas = {}
    for banda in BANDAS:
        baixo, alto = LIMITES_HZ[banda]
        mascara = (freqs >= baixo) & ((freqs < alto) if banda != "beta" else (freqs <= alto))
        medidas[f"rel_{banda}"] = float(espectro[mascara].sum() * df / total)
    faixa = (freqs >= PICO_HZ[0]) & (freqs <= PICO_HZ[1])
    medidas["log_potencia_total"] = float(np.log10(total))
    medidas["pico_hz"] = float(freqs[faixa][np.argmax(espectro[faixa])])
    return medidas


def espectro_pessoa(caminho_npz: Path, janelas: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Espectro médio nas janelas aceitas sem overlap e nos canais."""
    sinal = np.load(caminho_npz)
    usar = janelas[~janelas["rejeitada"] & janelas["sem_overlap"]]
    freqs, espectro = welch_janelas(cortar(sinal["x"], usar["inicio_amostra"].to_numpy(), TAMANHO_JANELA), float(sinal["fs"]))
    return freqs, espectro.mean(axis=(0, 1))


def previsoes_d(dataset: str) -> pd.DataFrame:
    p = ler_modelo(Path("outputs/07_DTSF/resultados") / dataset / PRINCIPAL)
    p["erro"] = (p["prob"] >= LIMIAR).astype(int) != p["y"]
    return p.groupby("subject_uid").agg(y=("y", "first"), prob_media=("prob", "mean"), prob_dp=("prob", "std"),
                                         repeticoes_erradas=("erro", "sum"))


def percentis(tabela: pd.DataFrame, uid: str) -> dict[str, tuple[float, float]]:
    """Percentil de cada medida da pessoa entre os HC e entre os pacientes do dataset (sem ela)."""
    linha, outros = tabela.loc[uid], tabela.drop(index=uid)
    return {m: (percentileofscore(outros.loc[outros["y"] == 0, m], linha[m]),
                percentileofscore(outros.loc[outros["y"] == 1, m], linha[m])) for m in MEDIDAS}


def figura_espectros(freqs: np.ndarray, espectros: dict[str, np.ndarray], tabela: pd.DataFrame, destino: Path) -> None:
    fig, eixos = plt.subplots(1, len(FOCO), figsize=(5 * len(FOCO), 3.8), sharey=True)
    for eixo, uid in zip(eixos, FOCO):
        dataset = tabela.at[uid, "dataset"]
        for classe, cor, nome in ((0, "tab:blue", "HC"), (1, "tab:red", "pacientes")):
            uids = tabela[(tabela["dataset"] == dataset) & (tabela["y"] == classe)].index.drop(uid, errors="ignore")
            log = np.log10(np.stack([espectros[u] for u in uids]))
            q1, med, q3 = np.percentile(log, [25, 50, 75], axis=0)
            eixo.fill_between(freqs, q1, q3, color=cor, alpha=0.2)
            eixo.plot(freqs, med, color=cor, label=f"{dataset} {nome}: mediana e IIQ")
        eixo.plot(freqs, np.log10(espectros[uid]), color="black", lw=2, label=uid)
        eixo.set(title=f"{uid} (real: {'paciente' if tabela.at[uid, 'y'] else 'HC'})", xlabel="Hz")
        eixo.legend(fontsize=7)
    eixos[0].set_ylabel("log10 PSD (µV²/Hz), média dos canais")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def figura_janelas(janelas: pd.DataFrame, destino: Path) -> None:
    fig, eixos = plt.subplots(len(FOCO), 1, figsize=(10, 2.4 * len(FOCO)), sharey=True)
    for eixo, uid in zip(eixos, FOCO):
        j = janelas[janelas["subject_uid"] == uid].sort_values("inicio_s")
        eixo.plot(j["inicio_s"], j["prob"], ".-", ms=3, lw=0.6)
        eixo.axhline(LIMIAR, color="gray", ls="--", lw=0.8)
        eixo.set(title=f"{uid}: probabilidade de paciente por janela ({PRINCIPAL} retreinado, repetição 0)", ylabel="prob", ylim=(0, 1))
    eixos[-1].set_xlabel("início da janela (s)")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--saida", type=Path, default=Path("outputs/09_investigacao/erros"))
    parser.add_argument("--explicacao", type=Path, default=Path("outputs/09_investigacao/explicacao") / PRINCIPAL)
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    indice = pd.read_csv("outputs/03_janelas/janelas.csv")
    manifesto = pd.read_csv("outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
    datas = pd.read_csv("outputs/09_investigacao/datas/pessoas.csv").set_index("subject_uid")
    linhas, espectros, freqs = [], {}, None
    for dataset in ("SZ", "MDD"):
        prev = previsoes_d(dataset)
        for uid, r in prev.iterrows():
            jan = indice[indice["subject_uid"] == uid]
            freqs, espectros[uid] = espectro_pessoa(Path("outputs/03_janelas") / f"{uid}.npz", jan)
            linhas.append(dict(subject_uid=uid, dataset=dataset, **r.to_dict(),
                               rejeitadas=float(jan["rejeitada"].mean()), duracao_s=float(manifesto.at[uid, "duracao_s"]),
                               canais_interpolados=manifesto.at[uid, "canais_interpolados"],
                               data_gravacao=datas.at[uid, "data_gravacao"], sessao=int(datas.at[uid, "sessao"]),
                               **resumo_espectro(freqs, espectros[uid])))
    tabela = pd.DataFrame(linhas).set_index("subject_uid")
    tabela[["y", "repeticoes_erradas"]] = tabela[["y", "repeticoes_erradas"]].astype(int)
    tabela["previsto"] = (tabela["prob_media"] >= LIMIAR).astype(int)

    arquivos = sorted(args.explicacao.glob("janelas/*/f*.csv"))
    janelas = pd.concat([pd.read_csv(a) for a in arquivos]) if arquivos else pd.DataFrame()
    if not janelas.empty:
        tabela["janelas_como_paciente_r0"] = janelas.groupby("subject_uid")["prob"].apply(lambda p: float((p >= LIMIAR).mean()))
        figura_janelas(janelas, args.saida / "janelas.png")
    tabela.to_csv(args.saida / "pessoas.csv")
    np.savez_compressed(args.saida / "espectros.npz", freqs=freqs, uids=np.array(list(espectros)),
                        espectros=np.stack(list(espectros.values())))
    figura_espectros(freqs, espectros, tabela, args.saida / "espectros.png")

    texto = "\n".join(linhas_resumo(tabela)) + "\n"
    (args.saida / "resumo.md").write_text(texto)
    print(texto)


def linhas_resumo(tabela: pd.DataFrame) -> list[str]:
    colunas = ["prob_media", "repeticoes_erradas", "rejeitadas", "data_gravacao", "sessao", "rel_teta", "rel_alfa", "pico_hz"]
    if "janelas_como_paciente_r0" in tabela:
        colunas.append("janelas_como_paciente_r0")
    linhas = ["# Erros individuais do DTSF", "", "## Pessoas classificadas errado (probabilidade média, limiar 0,5)", "",
              "| Pessoa | Classe real | " + " | ".join(colunas) + " |", "| --- | --- | " + " | ".join("---" for _ in colunas) + " |"]
    erradas = tabela[tabela["previsto"] != tabela["y"]].sort_values(["dataset", "y", "prob_media"])
    for uid, r in erradas.iterrows():
        valores = [f"{r[c]:.3f}" if isinstance(r[c], float) else str(r[c]) for c in colunas]
        linhas.append(f"| {uid} | {'paciente' if r['y'] else 'HC'} | " + " | ".join(valores) + " |")
    linhas += ["", "## Pessoas em foco: percentil de cada medida espectral entre os HC e entre os pacientes", "",
               "| Pessoa | " + " | ".join(MEDIDAS) + " |", "| --- | " + " | ".join("---" for _ in MEDIDAS) + " |"]
    for uid in FOCO:
        p = percentis(tabela[tabela["dataset"] == tabela.at[uid, "dataset"]], uid)
        linhas.append(f"| {uid} | " + " | ".join(f"{tabela.at[uid, m]:.3g} (HC {p[m][0]:.0f}; pac. {p[m][1]:.0f})" for m in MEDIDAS) + " |")
    linhas += ["", "Medianas por classe:", "", "| Dataset | Classe | " + " | ".join(MEDIDAS) + " |",
               "| --- | --- | " + " | ".join("---" for _ in MEDIDAS) + " |"]
    for (dataset, y), g in tabela.groupby(["dataset", "y"], sort=False):
        linhas.append(f"| {dataset} | {'paciente' if y else 'HC'} | " + " | ".join(f"{g[m].median():.3g}" for m in MEDIDAS) + " |")
    return linhas


if __name__ == "__main__":
    main()
