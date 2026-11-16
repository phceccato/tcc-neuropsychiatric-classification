"""Frequência da época 1 no early stopping do DTSF, e a AUC com e sem os folds que a escolheram (AL-05).

Só descreve; não treina nem muda o protocolo. O early stopping mede a perda média por pessoa da validação interna
(4–5 pessoas na SZ, 9 no MDD). Com tão poucas pessoas a perda é ruidosa, e em parte dos folds a época 1 (rede
quase sem treino) não é superada em 10 épocas. Para cada (dataset, modelo) esta análise traz:
- quantos folds escolhem a época 1 e a época escolhida (mediana, mínimo e máximo);
- a AUC por pessoa e a AUC por repetição com todos os folds e sem os que escolheram a época 1.

A segunda AUC é condicional ao desfecho da seleção (retira folds depois de olhar a época): serve para dimensionar o
efeito, não como estimativa do desempenho.

Entradas (as que existirem):
- outputs/07_DTSF/resultados/<dataset>/<modelo>/r<rep>_f<fold>.{json,csv}: todos os modelos da etapa 07;
- outputs/10_resultados (versionada): só o D_FA, quando ele não está em resultados/ (perdas/ e previsoes/).

Grava outputs/07_DTSF/epoca_um.md e epoca_um.csv.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.avaliacao.metricas import escores_por_pessoa

CHAVE = ["dataset", "modelo", "repeticao", "fold"]


def carregar_resultados(raiz: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(folds, previsoes) de outputs/07_DTSF/resultados. Folds: CHAVE, melhor_epoca, dispositivo."""
    folds, previsoes = [], []
    for csv in sorted(raiz.glob("*/*/r*_f*.csv")):
        dataset, modelo = csv.parent.parent.name, csv.parent.name
        info = json.loads(csv.with_suffix(".json").read_text())
        p = pd.read_csv(csv)
        rep, fold = int(p["repeticao"].iat[0]), int(p["fold"].iat[0])
        folds.append(dict(dataset=dataset, modelo=modelo, repeticao=rep, fold=fold,
                          melhor_epoca=int(info["melhor_epoca"]), dispositivo=info.get("dispositivo", "cpu")))
        previsoes.append(p.assign(dataset=dataset, modelo=modelo))
    return pd.DataFrame(folds), (pd.concat(previsoes, ignore_index=True) if previsoes else pd.DataFrame())


def carregar_exportados(raiz: Path, modelo: str, dispositivo: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(folds, previsoes) de um modelo em outputs/10_resultados (época escolhida em perdas/, previsões em previsoes/)."""
    folds, previsoes = [], []
    for perdas in sorted(raiz.glob(f"perdas/*/{modelo}.csv")):
        dataset = perdas.parent.name
        p = pd.read_csv(perdas)
        escolhida = p[p["melhor"] == 1].groupby(["repeticao", "fold"])["epoca"].first().rename("melhor_epoca").reset_index()
        folds.append(escolhida.assign(dataset=dataset, modelo=modelo, dispositivo=dispositivo))
        pr = pd.read_csv(raiz / "previsoes" / dataset / f"{modelo}.csv").rename(columns={"y_true": "y", "y_prob": "prob"})
        previsoes.append(pr.assign(dataset=dataset, modelo=modelo))
    if not folds:
        return pd.DataFrame(), pd.DataFrame()
    return pd.concat(folds, ignore_index=True), pd.concat(previsoes, ignore_index=True)


def _auc_pessoa(pred: pd.DataFrame) -> tuple[float, int]:
    if pred.empty:
        return float("nan"), 0
    e = escores_por_pessoa(pred)
    if e["y"].nunique() < 2:
        return float("nan"), len(e)
    return float(roc_auc_score(e["y"], e["prob"])), len(e)


def _auc_repeticao(pred: pd.DataFrame) -> float:
    """Média, entre as repetições, da AUC das previsões da repetição (as que têm as duas classes)."""
    aucs = [roc_auc_score(g["y"], g["prob"]) for _, g in pred.groupby("repeticao") if g["y"].nunique() > 1]
    return float(np.mean(aucs)) if aucs else float("nan")


def resumo_epoca_um(folds: pd.DataFrame, previsoes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por (dataset, modelo): folds com a época 1 e AUCs com todos os folds e sem os da época 1."""
    previsoes = previsoes.merge(folds[CHAVE + ["melhor_epoca"]], on=CHAVE)
    linhas = []
    for (dataset, modelo), g in folds.groupby(["dataset", "modelo"], sort=False):
        p = previsoes[(previsoes["dataset"] == dataset) & (previsoes["modelo"] == modelo)]
        sem = p[p["melhor_epoca"] != 1]
        auc_todos, n_todos = _auc_pessoa(p)
        auc_sem, n_sem = _auc_pessoa(sem)
        e1 = g["melhor_epoca"] == 1
        linhas.append(dict(dataset=dataset, modelo=modelo, dispositivo="/".join(sorted(set(g["dispositivo"]))),
                           folds=len(g), folds_epoca1=int(e1.sum()), fracao_epoca1=float(e1.mean()),
                           epoca_mediana=float(g["melhor_epoca"].median()), epoca_min=int(g["melhor_epoca"].min()),
                           epoca_max=int(g["melhor_epoca"].max()),
                           auc_pessoa_todos=auc_todos, auc_pessoa_sem_epoca1=auc_sem, pessoas_sem_epoca1=n_sem,
                           auc_repeticao_todos=_auc_repeticao(p), auc_repeticao_sem_epoca1=_auc_repeticao(sem)))
    return pd.DataFrame(linhas)


def fracao_por_validacao(folds: pd.DataFrame, particoes: pd.DataFrame) -> pd.DataFrame:
    """Fração de folds com época 1 por dataset e número de pessoas na validação interna."""
    nval = (particoes[particoes["papel"] == "validacao"].groupby(["dataset", "repeticao", "fold"]).size()
            .rename("pessoas_validacao").reset_index())
    j = folds.merge(nval, on=["dataset", "repeticao", "fold"])
    j["epoca1"] = j["melhor_epoca"] == 1
    return j.groupby(["dataset", "pessoas_validacao"]).agg(folds=("epoca1", "size"), fracao_epoca1=("epoca1", "mean")).reset_index()


def _md(r: pd.DataFrame, por_validacao: pd.DataFrame) -> str:
    f = lambda v: "—" if pd.isna(v) else f"{v:.3f}"
    linhas = ["# Época 1 no early stopping do DTSF", "",
              "Descritivo, sem alterar o protocolo (AL-05). Em parte dos folds a perda da validação interna não melhora depois da "
              "época 1 em 10 épocas, e o modelo fica com os pesos da época 1. A AUC \"sem época 1\" retira esses folds depois de "
              "olhar a época escolhida: dimensiona o efeito e não é estimativa de desempenho.", "",
              "| Conjunto | Modelo | Dispositivo | Folds | Folds com época 1 | Época escolhida (mediana, mín–máx) | "
              "AUC por pessoa: todos / sem época 1 | AUC por repetição: todos / sem época 1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in r.itertuples():
        linhas.append(f"| {x.dataset} | {x.modelo} | {x.dispositivo} | {x.folds} | {x.folds_epoca1} ({x.fracao_epoca1:.0%}) | "
                      f"{x.epoca_mediana:g} ({x.epoca_min}–{x.epoca_max}) | {f(x.auc_pessoa_todos)} / {f(x.auc_pessoa_sem_epoca1)} | "
                      f"{f(x.auc_repeticao_todos)} / {f(x.auc_repeticao_sem_epoca1)} |")
    linhas += ["", "**Fração de folds com época 1 por tamanho da validação interna (todos os modelos juntos)**", "",
               "| Conjunto | Pessoas na validação | Folds | Folds com época 1 |", "| --- | --- | --- | --- |"]
    linhas += [f"| {x.dataset} | {x.pessoas_validacao} | {x.folds} | {x.fracao_epoca1:.0%} |" for x in por_validacao.itertuples()]
    return "\n".join(linhas) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resultados", type=Path, default=Path("outputs/07_DTSF/resultados"))
    parser.add_argument("--exportados", type=Path, default=Path("outputs/10_resultados"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/05_folds/folds.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/07_DTSF"))
    parser.add_argument("--principal", default="D_FA")
    args = parser.parse_args()

    folds, previsoes = carregar_resultados(args.resultados)
    if folds.empty or args.principal not in set(folds["modelo"]):
        f2, p2 = carregar_exportados(args.exportados, args.principal, "cuda")
        folds, previsoes = pd.concat([folds, f2], ignore_index=True), pd.concat([previsoes, p2], ignore_index=True)
    if folds.empty:
        raise SystemExit("sem resultados da etapa 07 nem exportados")
    resumo = resumo_epoca_um(folds, previsoes)
    por_validacao = fracao_por_validacao(folds, pd.read_csv(args.folds))
    args.saida.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(args.saida / "epoca_um.csv", index=False)
    (args.saida / "epoca_um.md").write_text(_md(resumo, por_validacao), encoding="utf-8")
    print(_md(resumo, por_validacao))


if __name__ == "__main__":
    main()
