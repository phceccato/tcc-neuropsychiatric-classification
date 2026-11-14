"""Experimento 3, resumo: o que os valores SHAP do modelo principal, o D_FA (pipeline/e09_investigacao/explicar.py), mostram.

Lê outputs/09_investigacao/explicacao/D_FA/{janelas,atribuicoes}/ e grava, na mesma pasta, resumo.md e as
figuras frequencias.png, canais.png e tempo.png.

- **Conferências:** a previsão da pessoa no modelo retreinado contra a da etapa 07 na repetição 0, e
  a completude (soma dos φ contra o logit menos o logit médio do fundo).
- **Grupos:** cada pessoa de teste entra em HC certo, HC errado, paciente certo ou paciente errado,
  pela previsão do próprio modelo retreinado (média das janelas, limiar 0,5), que é o modelo explicado.
- **Perfis por pessoa,** cada um normalizado para somar 1 antes da média do grupo, para que
  pessoas com logits maiores não dominem:
  - ramos: fração do |φ| total no ramo temporal e no espectral;
  - frequências: |φ| do ramo espectral por frequência (2–30 Hz) e |φ| do ramo temporal por
    entrada (sinal base, δ, θ, α, β);
  - eletrodos: |φ| por canal, somando os dois ramos;
  - tempo: |φ| do ramo temporal em 10 trechos de 0,4 s da janela e do ramo espectral nos 16 quadros.
- **Sinal:** φ médio do ramo espectral por frequência (positivo empurra para paciente).
- **Pessoas em foco:** correlação de Pearson do perfil de φ por frequência (com sinal) de cada pessoa
  em foco com o perfil médio dos HC certos e dos pacientes certos. O perfil de |φ| tem quase o mesmo
  formato em todas as pessoas (cosseno > 0,95), então só o sinal distingue para que lado cada
  frequência empurra.
"""
from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd

from pipeline.e01_auditoria.auditar_ec import CANAIS
from pipeline.e07_DTSF.avaliar import ler_modelo
from pipeline.e07_DTSF.dados_dtsf import FREQUENCIAS
from pipeline.e09_investigacao.explicar import MODELO, SAIDA as EXPLICACAO
from pipeline.e09_investigacao.erros import FOCO, LIMIAR

TRECHOS = 10
ENTRADAS = ("sinal base", "δ", "θ", "α", "β")
GRUPOS = ("HC certo", "HC errado", "paciente certo", "paciente errado")
CORES = dict(zip(GRUPOS, ("tab:blue", "tab:cyan", "tab:red", "tab:orange")))


def normalizar(x: np.ndarray) -> np.ndarray:
    """Cada linha somando 1."""
    x = x.reshape(len(x), -1) if x.ndim > 2 else x
    return x / x.sum(axis=1, keepdims=True)


def correlacao(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(a, b)[0, 1])


def carregar(pasta: Path, dataset: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    janelas = pd.concat([pd.read_csv(a) for a in sorted((pasta / "janelas" / dataset).glob("f*.csv"))], ignore_index=True)
    partes = [np.load(a) for a in sorted((pasta / "atribuicoes" / dataset).glob("f*.npz"))]
    uids = np.concatenate([p["uids"] for p in partes])
    atrib = {k: np.concatenate([p[k] for p in partes]) for k in partes[0].files if k != "uids"}
    pessoas = janelas.groupby("subject_uid").agg(y=("y", "first"), prob=("prob", "mean")).loc[uids]
    pessoas["grupo"] = [f"{'paciente' if y else 'HC'} {'certo' if (p >= LIMIAR) == y else 'errado'}"
                        for y, p in zip(pessoas["y"], pessoas["prob"])]
    return janelas, pessoas, atrib


def perfis(atrib: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    t_canal = atrib["t_abs_canal_entrada"].sum(2)
    e_canal = atrib["e_abs_canal_freq"].sum(2)
    total_t, total_e = t_canal.sum(1), e_canal.sum(1)
    tempo = atrib["t_abs_tempo"].reshape(len(total_t), TRECHOS, -1).sum(2)
    return dict(fracao_temporal=total_t / (total_t + total_e), frequencia=normalizar(atrib["e_abs_canal_freq"].sum(1)),
                entrada=normalizar(atrib["t_abs_canal_entrada"].sum(1)), canal=normalizar(t_canal + e_canal),
                tempo=normalizar(tempo), quadro=normalizar(atrib["e_abs_quadro"]),
                sinal_frequencia=atrib["e_sinal_canal_freq"].sum(1))


def conferencias(dataset: str, janelas: pd.DataFrame, pessoas: pd.DataFrame) -> dict[str, float]:
    original = ler_modelo(Path("outputs/07_DTSF/resultados") / dataset / MODELO)
    original = original[original["repeticao"] == 0].set_index("subject_uid")["prob"].loc[pessoas.index]
    soma = janelas["phi_temporal"] + janelas["phi_espectral"]
    alvo = janelas["logit"] - janelas["logit_fundo"]
    return dict(diferenca_max=float((pessoas["prob"] - original).abs().max()),
                mesma_classe=float(((pessoas["prob"] >= LIMIAR) == (original >= LIMIAR)).mean()),
                completude_r=float(np.corrcoef(soma, alvo)[0, 1]),
                completude_erro_mediano=float((soma - alvo).abs().median() / alvo.abs().median()))


def figura_frequencias(dados: dict, destino: Path) -> None:
    fig, eixos = plt.subplots(2, 3, figsize=(15, 7.5))
    for linha, (dataset, (pessoas, perf)) in enumerate(dados.items()):
        e_abs, e_sinal, e_ent = eixos[linha]
        for grupo in GRUPOS:
            m = (pessoas["grupo"] == grupo).to_numpy()
            if not m.any():
                continue
            rotulo = f"{grupo} (n = {m.sum()})"
            e_abs.plot(FREQUENCIAS, perf["frequencia"][m].mean(0), color=CORES[grupo], label=rotulo)
            e_sinal.plot(FREQUENCIAS, perf["sinal_frequencia"][m].mean(0), color=CORES[grupo], label=rotulo)
            e_ent.plot(ENTRADAS, perf["entrada"][m].mean(0), "o-", color=CORES[grupo], label=rotulo)
        e_sinal.axhline(0, color="gray", lw=0.8)
        e_abs.set(title=f"{dataset}: |φ| espectral por frequência (fração)", xlabel="Hz")
        e_sinal.set(title=f"{dataset}: φ espectral por frequência (+ → paciente)", xlabel="Hz")
        e_ent.set(title=f"{dataset}: |φ| temporal por entrada (fração)")
        e_abs.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def figura_canais(dados: dict, destino: Path) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        info = mne.create_info(list(CANAIS), 125.0, "eeg")
        info.set_montage(mne.channels.make_standard_montage("standard_1020"))
    fig, eixos = plt.subplots(2, len(GRUPOS), figsize=(3.2 * len(GRUPOS), 6.4))
    for linha, (dataset, (pessoas, perf)) in enumerate(dados.items()):
        medias = {g: perf["canal"][(pessoas["grupo"] == g).to_numpy()].mean(0) for g in GRUPOS if (pessoas["grupo"] == g).any()}
        limite = max(v.max() for v in medias.values())
        for eixo, grupo in zip(eixos[linha], GRUPOS):
            if grupo in medias:
                mne.viz.plot_topomap(medias[grupo], info, axes=eixo, show=False, cmap="viridis", vlim=(1 / 19 * 0.5, limite))
                eixo.set_title(f"{dataset}: {grupo}\n(n = {(pessoas['grupo'] == grupo).sum()})", fontsize=9)
            else:
                eixo.axis("off")
    fig.suptitle("|φ| por eletrodo, dois ramos somados (fração; 1/19 = 0,053 seria uniforme)")
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def figura_tempo(dados: dict, destino: Path) -> None:
    fig, eixos = plt.subplots(2, 2, figsize=(11, 6.5))
    for linha, (dataset, (pessoas, perf)) in enumerate(dados.items()):
        for grupo in GRUPOS:
            m = (pessoas["grupo"] == grupo).to_numpy()
            if m.any():
                eixos[linha, 0].plot((np.arange(TRECHOS) + 0.5) * 4 / TRECHOS, perf["tempo"][m].mean(0), "o-", color=CORES[grupo], label=grupo)
                eixos[linha, 1].plot(np.arange(16), perf["quadro"][m].mean(0), "o-", color=CORES[grupo], label=grupo)
        eixos[linha, 0].set(title=f"{dataset}: |φ| temporal por trecho de 0,4 s", xlabel="s dentro da janela")
        eixos[linha, 1].set(title=f"{dataset}: |φ| espectral por quadro da STFT", xlabel="quadro (passo 0,2 s)")
        eixos[linha, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pasta", type=Path, default=EXPLICACAO)
    args = parser.parse_args()

    dados, conf, linhas = {}, {}, [f"# Explicações SHAP do {MODELO} (repetição 0, retreinada)", ""]
    for dataset in ("SZ", "MDD"):
        janelas, pessoas, atrib = carregar(args.pasta, dataset)
        dados[dataset] = (pessoas, perfis(atrib))
        conf[dataset] = conferencias(dataset, janelas, pessoas)
    figura_frequencias(dados, args.pasta / "frequencias.png")
    figura_canais(dados, args.pasta / "canais.png")
    figura_tempo(dados, args.pasta / "tempo.png")

    linhas += ["## Conferências", "", "| Dataset | Maior diferença de probabilidade contra a etapa 07 (r0) | Mesma classe | "
               "Completude: r | Completude: erro mediano relativo |", "| --- | --- | --- | --- | --- |"]
    for dataset, c in conf.items():
        linhas.append(f"| {dataset} | {c['diferenca_max']:.3f} | {c['mesma_classe']:.0%} | {c['completude_r']:.3f} "
                      f"| {c['completude_erro_mediano']:.2f} |")
    linhas += ["", "## Perfis médios por grupo", ""]
    for dataset, (pessoas, perf) in dados.items():
        linhas += [f"### {dataset}", "", "| Grupo | n | Fração temporal | Frequência com mais |φ| espectral | "
                   "Entrada temporal com mais |φ| | 3 eletrodos com mais |φ| |", "| --- | --- | --- | --- | --- | --- |"]
        for grupo in GRUPOS:
            m = (pessoas["grupo"] == grupo).to_numpy()
            if not m.any():
                continue
            canais = np.argsort(perf["canal"][m].mean(0))[::-1][:3]
            linhas.append(f"| {grupo} | {m.sum()} | {perf['fracao_temporal'][m].mean():.2f} "
                          f"| {FREQUENCIAS[np.argmax(perf['frequencia'][m].mean(0))]} Hz "
                          f"| {ENTRADAS[np.argmax(perf['entrada'][m].mean(0))]} | {', '.join(CANAIS[c] for c in canais)} |")
        linhas.append("")
    linhas += ["## Pessoas em foco", "", "Correlação do perfil de φ por frequência (com sinal) da pessoa com o perfil médio "
               "de cada grupo.", "", "| Pessoa | Grupo (modelo r0) | prob r0 | Fração temporal | Correlação com HC certo | "
               "Correlação com paciente certo | Frequência com mais |φ| | φ espectral total (+ → paciente) |",
               "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for uid in FOCO:
        dataset = uid.split("-")[0]
        pessoas, perf = dados[dataset]
        i = pessoas.index.get_loc(uid)
        ref = {g: perf["sinal_frequencia"][((pessoas["grupo"] == g) & (pessoas.index != uid)).to_numpy()].mean(0)
               for g in ("HC certo", "paciente certo")}
        linhas.append(f"| {uid} | {pessoas.iloc[i]['grupo']} | {pessoas.iloc[i]['prob']:.3f} | {perf['fracao_temporal'][i]:.2f} "
                      f"| {correlacao(perf['sinal_frequencia'][i], ref['HC certo']):+.2f} "
                      f"| {correlacao(perf['sinal_frequencia'][i], ref['paciente certo']):+.2f} "
                      f"| {FREQUENCIAS[np.argmax(perf['frequencia'][i])]} Hz | {perf['sinal_frequencia'][i].sum():+.2f} |")
    texto = "\n".join(linhas) + "\n"
    (args.pasta / "resumo.md").write_text(texto)
    print(texto)


if __name__ == "__main__":
    main()
