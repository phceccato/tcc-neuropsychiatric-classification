"""Números dos gráficos de resultados, recalculados das previsões exportadas e conferidos contra os resumos.

Lê só outputs/10_resultados/ (a pasta versionada da etapa 10), para que qualquer pessoa gere os gráficos sem
treinar nada: as previsões fora do fold de cada modelo (previsoes/<dataset>/<modelo>.csv, com y_true e y_prob), as
permutações do modelo principal (permutacoes/<dataset>/D_FA.csv) e os resumos das etapas 06, 07 e 08 (resumos/).
Recalcula, com as mesmas funções e sementes dessas etapas:

- por modelo e dataset: AUC por pessoa com IC de 95% (`avaliar`), AUC de cada repetição, a média ± desvio de todas
  as métricas nas 10 repetições (`media_desvio`), as secundárias por pessoa no limiar 0,5, a matriz de confusão por
  pessoa e os escores por pessoa (rótulo e escore médio, sem identificador), de que saem as curvas ROC;
- as comparações, com `comparacao_pareada`, guardadas como "A − B": D_pequeno − ablação (H2, H3), D_FA −
  baseline (os 3 combinados na H4 e os 6 de uma modalidade, descritivos) e as descritivas entre os DTSF
  (D_FA − D, D − D_pequeno, D_FA − D_pequeno); e a decisão de cada hipótese (interseção-união e Holm);
- H1: as AUCs nulas das permutações do D_FA, a AUC observada (repetição 0) e o p contra o acaso.

Cada AUC, IC, ΔAUC e p é conferido contra a tabela correspondente do resumo da etapa (|diferença| ≤ 0,0005 para AUC
e Δ, ≤ 0,00005 para p); se algum não bater, o script para com `Divergencia` e não grava.
Grava outputs/12_graficos/numeros.json, que `pipeline/e12_graficos/graficos.py` lê.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.avaliacao.metricas import (auc_por_repeticao, avaliar, escores_por_pessoa, media_desvio,
                                         metricas_secundarias, p_valor, roc_auc_score)
from pipeline.e07_DTSF.avaliar import (DESCRITIVAS, HIPOTESES, REFERENCIA_ABLACOES, SEMENTE, comparacao_pareada,
                                       completo, decidir_h1, decidir_hipotese)
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL, TIPOS_DTSF
from pipeline.e08_classicos.avaliar import BASELINES_H4, REFERENCIA as REFERENCIA_H4, decidir_h4
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS

DATASETS = ("SZ", "MDD")
LR_PSD = ("LR_psd_absolutas", "LR_psd_relativas")
MODELOS = (*TIPOS_DTSF, *CLASSICOS, *LR_PSD)
TOL_AUC, TOL_P = 5e-4, 5e-5
FOLGA = 1e-9  # um valor como 0,4375 sai "0.438" no resumo: a diferença é 0,0005 exato, mais o erro do ponto flutuante
RESULTADOS = Path("outputs/10_resultados")
SAIDA = Path("outputs/12_graficos")


class Divergencia(ValueError):
    """Um número recalculado não bate com o resumo da etapa."""


def conferir(nome: str, calculado: float, esperado: float, tol: float = TOL_AUC) -> None:
    if abs(calculado - esperado) > tol + FOLGA:
        raise Divergencia(f"{nome}: calculado {calculado:.5f}, resumo {esperado:.5f}")


_NUM = r"([+-]?\d+\.\d+)"


def tabela_modelos(md: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """{(dataset, modelo): (AUC, IC baixo, IC alto)} das tabelas por dataset de um resumo.md."""
    saida, dataset = {}, None
    for linha in md.splitlines():
        if linha.startswith("## "):
            dataset = linha[3:].strip() if linha[3:].strip() in DATASETS else None
            continue
        m = re.match(rf"\| (\S+) \|(?: [\d.]+ \|)? {_NUM} \[{_NUM}, {_NUM}\] \|", linha)
        if dataset and m:
            saida[(dataset, m.group(1))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return saida


def tabela_deltas(md: str, referencia: str) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    """{(dataset, "A − B"): (Δ, IC baixo, IC alto, p)} das linhas "| SZ | [A − ]B | Δ [IC] | p |".

    Sem "A − " (as linhas da etapa 08), A é a `referencia`.
    """
    saida = {}
    for m in re.finditer(rf"^\| (SZ|MDD) \| (?:(\S+) − )?(\S+) \| {_NUM} \[{_NUM}, {_NUM}\] \| (\d\.\d+) \|", md, re.M):
        saida[(m.group(1), f"{m.group(2) or referencia} − {m.group(3)}")] = tuple(float(m.group(i)) for i in (4, 5, 6, 7))
    return saida


def tabela_lr_psd(md: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    saida = {}
    for m in re.finditer(rf"^\| (SZ|MDD) \| (absolutas|relativas) \| {_NUM} \[{_NUM}, {_NUM}\] \|", md, re.M):
        saida[(m.group(1), f"LR_psd_{m.group(2)}")] = tuple(float(m.group(i)) for i in (3, 4, 5))
    return saida


def ler_previsoes(caminho: Path) -> pd.DataFrame:
    """Previsões exportadas (y_true, y_prob) no formato de pipeline/avaliacao (y, prob)."""
    return pd.read_csv(caminho).rename(columns={"y_true": "y", "y_prob": "prob"})


def confusao(predicoes: pd.DataFrame, limiar: float = 0.5) -> dict[str, int]:
    """Matriz de confusão por pessoa (escore médio nas repetições) no limiar, com paciente positivo."""
    e = escores_por_pessoa(predicoes)
    previsto = e["prob"] >= limiar
    y = e["y"].astype(bool)
    return dict(vp=int((previsto & y).sum()), fn=int((~previsto & y).sum()),
                fp=int((previsto & ~y).sum()), vn=int((~previsto & ~y).sum()))


def resumo_modelo(predicoes: pd.DataFrame) -> dict:
    r = avaliar(predicoes, semente=SEMENTE)
    e = escores_por_pessoa(predicoes).sort_values("prob", kind="stable")   # sem identificador: só rótulo e escore
    return dict(auc=r["auc"], ic=[r["ic_baixo"], r["ic_alto"]], auc_repeticoes=auc_por_repeticao(predicoes).tolist(),
                pessoa=metricas_secundarias(e["y"], e["prob"]),
                media_desvio={k: list(v) for k, v in media_desvio(predicoes).items()},
                confusao=confusao(predicoes), escores=dict(y=e["y"].astype(int).tolist(), prob=e["prob"].tolist()))


def _comparacao(c: dict) -> dict:
    return dict(delta=c["delta"], ic=list(c["ic"]), p=c["p"], t_corrigido_p=c["t_corrigido_p"],
                delta_repeticoes=np.asarray(c["delta_repeticoes"]).tolist())


def comparacoes() -> list[tuple[str, str]]:
    """Pares (A, B) de ΔAUC = A − B: H2 e H3, os 9 baselines contra o principal e as descritivas entre os DTSF."""
    pares = [(REFERENCIA_ABLACOES, x) for ablacoes in HIPOTESES.values() for x in ablacoes]
    pares += [(REFERENCIA_H4, m) for m in CLASSICOS]
    return pares + list(DESCRITIVAS)


def h1(real: pd.DataFrame, permutacoes: pd.DataFrame) -> dict:
    """AUCs nulas (uma por permutação, sobre os 5 folds), AUC observada da repetição 0 e p, como em `p_acaso`."""
    nulas = np.array([roc_auc_score(g["y"], g["prob"]) for _, g in permutacoes.groupby("permutacao")])
    r0 = real[real["repeticao"] == 0]
    observada = float(roc_auc_score(r0["y"], r0["prob"]))
    return dict(nulas=nulas.tolist(), observada=observada, p=p_valor(observada, nulas))


def calcular(raiz: Path = RESULTADOS) -> dict:
    md = {e: (raiz / "resumos" / f"{e}.md").read_text(encoding="utf-8") for e in ("06_LR_psd", "07_DTSF", "08_classicos")}
    esperado_aucs = tabela_modelos(md["07_DTSF"]) | tabela_modelos(md["08_classicos"]) | tabela_lr_psd(md["06_LR_psd"])
    esperado_deltas = tabela_deltas(md["07_DTSF"], REFERENCIA_ABLACOES) | tabela_deltas(md["08_classicos"], REFERENCIA_H4)

    numeros = dict(modelos={}, comparacoes={}, hipoteses={}, h1={}, h1_distribuicao={})
    acaso, por_hipotese, h4 = {}, {h: {} for h in HIPOTESES}, {}
    for d in DATASETS:
        pred = {m: ler_previsoes(raiz / "previsoes" / d / f"{m}.csv") for m in MODELOS}
        incompletos = [m for m, p in pred.items() if not completo(p)]
        if incompletos:
            raise ValueError(f"{d}: previsões incompletas de {incompletos}")

        numeros["modelos"][d] = {}
        for m, p in pred.items():
            r = resumo_modelo(p)
            auc, baixo, alto = esperado_aucs[(d, m)]
            conferir(f"{d} {m} AUC", r["auc"], auc)
            conferir(f"{d} {m} IC baixo", r["ic"][0], baixo)
            conferir(f"{d} {m} IC alto", r["ic"][1], alto)
            numeros["modelos"][d][m] = r

        numeros["comparacoes"][d] = {}
        for a, b in comparacoes():
            c = comparacao_pareada(pred[a], pred[b])
            chave = f"{a} − {b}"
            delta, baixo, alto, p = esperado_deltas[(d, chave)]
            conferir(f"{d} {chave} Δ", c["delta"], delta)
            conferir(f"{d} {chave} IC baixo", c["ic"][0], baixo)
            conferir(f"{d} {chave} IC alto", c["ic"][1], alto)
            conferir(f"{d} {chave} p", c["p"], p, TOL_P)
            numeros["comparacoes"][d][chave] = _comparacao(c)
        for h, ablacoes in HIPOTESES.items():
            por_hipotese[h][d] = {x: numeros["comparacoes"][d][f"{REFERENCIA_ABLACOES} − {x}"] for x in ablacoes}
        h4[d] = {m: numeros["comparacoes"][d][f"{REFERENCIA_H4} − {m}"] for m in BASELINES_H4}

        caminho = raiz / "permutacoes" / d / f"{PRINCIPAL}.csv"
        if not caminho.exists():
            raise ValueError(f"{d}: sem as permutações do {PRINCIPAL} (H1) em {caminho}")
        dist = h1(pred[PRINCIPAL], ler_previsoes(caminho))
        numeros["h1_distribuicao"][d] = dist
        acaso[d] = (dist["p"], len(dist["nulas"]))

    for h, ablacoes in HIPOTESES.items():
        numeros["hipoteses"][h] = decidir_hipotese(por_hipotese[h], ablacoes)
    numeros["hipoteses"]["H4"] = decidir_h4(h4)
    numeros["h1"] = {d: {k: (bool(v) if k == "rejeita" else v) for k, v in r.items()} for d, r in decidir_h1(acaso).items()}
    for h, r in numeros["hipoteses"].items():
        numeros["hipoteses"][h] = {d: {k: (bool(v) if k == "rejeita" else float(v)) for k, v in x.items()}
                                   for d, x in r.items()}
    return numeros


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--resultados", type=Path, default=RESULTADOS)
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    numeros = calcular(args.resultados)
    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / "numeros.json"
    destino.write_text(json.dumps(numeros, indent=1, ensure_ascii=False), encoding="utf-8")
    n = sum(len(v) for v in numeros["modelos"].values())
    print(f"{destino}: {n} (modelo, dataset) e {sum(len(v) for v in numeros['comparacoes'].values())} comparações, "
          "todos conferidos contra os resumos das etapas")
    for h, r in {"H1": numeros["h1"], **numeros["hipoteses"]}.items():
        print(h, {d: ("rejeita" if x["rejeita"] else "não rejeita") for d, x in r.items()})


if __name__ == "__main__":
    main()
