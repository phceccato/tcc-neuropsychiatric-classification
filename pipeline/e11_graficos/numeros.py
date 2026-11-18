"""Números dos gráficos de resultados, recalculados das previsões exportadas e conferidos contra os resumos.

Lê só outputs/10_resultados/ (a pasta versionada da etapa 10), para que qualquer pessoa gere os gráficos sem
treinar nada: as previsões fora do fold de cada modelo (previsoes/<dataset>/<modelo>.csv, com y_true e y_prob),
o histórico de perdas do DTSF_FA (perdas/) e os resumos das etapas 06, 07 e 08 (resumos/). Recalcula, com as mesmas
funções e sementes dessas etapas:

- por modelo e dataset: AUC por pessoa com IC de 95% (`avaliar`), AUC de cada repetição, a média ± desvio de todas
  as métricas nas 10 repetições (`media_desvio`), as secundárias por pessoa no limiar 0,5, a matriz de confusão por
  pessoa e os escores por pessoa (rótulo e escore médio, sem identificador), de que saem as curvas ROC;
- as comparações descritivas, com `comparacao_pareada`, guardadas como "A − B" (ΔAUC com IC de 95% e o Δ por
  repetição, sem p-valor): DTSF_Lite − ablação, DTSF_FA − cada um dos 9 baselines e as comparações entre os modelos da família DTSF
  (DTSF_FA − DTSF, DTSF − DTSF_Lite, DTSF_FA − DTSF_Lite);
- teste de DeLong (`teste_delong`), bicaudal, a análise estatística principal: o modelo principal contra os outros
  5 modelos principais (`DELONG_COMPARADOS`: DTSF, DTSF_Lite e os 3 combinados), sobre os escores por pessoa (média
  das 10 repetições); os 5 p brutos de cada transtorno são ajustados por Holm, com α = 0,05. Não há resumo de etapa
  para conferir: a conferência é o teste unitário contra uma implementação independente;
- bootstrap do modelo principal: as curvas ROC de reamostras bootstrap estratificadas por pessoa (as mesmas
  reamostragens do IC de 95% da AUC, que é conferido), numa grade de 1 − especificidade; só as primeiras
  N_CURVAS_BOOTSTRAP curvas vão para o json;
- perdas: o histórico de perda de treino e de validação interna do DTSF_FA em cada (repetição, fold)
  (perdas/<dataset>/DTSF_FA.csv, se existir), conferido (50 folds, épocas 1, 2, …, e a época escolhida é a de menor
  perda de validação).

Cada AUC, IC e ΔAUC é conferido contra a tabela correspondente do resumo da etapa (|diferença| ≤ 0,0005); se algum
não bater, o script para com `Divergencia` e não grava.
Grava outputs/11_graficos/numeros.json, que `pipeline/e11_graficos/graficos.py` lê.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.avaliacao.metricas import (GRADE_FPR, auc_por_repeticao, avaliar, bootstrap_roc, escores_por_pessoa,
                                         holm, media_desvio, metricas_secundarias, parear, teste_delong)
from pipeline.e07_DTSF.avaliar import (ABLACOES, ALFA, DESCRITIVAS, REFERENCIA_ABLACOES, SEMENTE, comparacao_pareada,
                                       completo)
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL, TIPOS_DTSF
from pipeline.e08_classicos.avaliar import REFERENCIA as REFERENCIA_BASELINES
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS
from pipeline.e11_graficos.delong import COMPARADOS as DELONG_COMPARADOS
from pipeline.nomes import LR_PSD as LR_PSD_NOMES, nome_novo, resolver_arquivo

DATASETS = ("SZ", "MDD")
LR_PSD = tuple(LR_PSD_NOMES.values())   # "absolutas" -> LR_PSD_Absolute, "relativas" -> LR_PSD_Relative
MODELOS = (*TIPOS_DTSF, *CLASSICOS, *LR_PSD)
TOL_AUC = 5e-4
N_CURVAS_BOOTSTRAP = 500
FOLGA = 1e-9  # um valor como 0,4375 sai "0.438" no resumo: a diferença é 0,0005 exato, mais o erro do ponto flutuante
RESULTADOS = Path("outputs/10_resultados")
SAIDA = Path("outputs/11_graficos")


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
            saida[(dataset, nome_novo(m.group(1)))] = tuple(float(m.group(i)) for i in (2, 3, 4))
    return saida


def tabela_deltas(md: str, referencia: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    """{(dataset, "A − B"): (Δ, IC baixo, IC alto)} das linhas "| SZ | A − B | Δ [IC] | ... |"."""
    saida = {}
    for m in re.finditer(rf"^\| (SZ|MDD) \| (\S+) − (\S+) \| {_NUM} \[{_NUM}, {_NUM}\] \|", md, re.M):
        saida[(m.group(1), f"{nome_novo(m.group(2))} − {nome_novo(m.group(3))}")] = tuple(float(m.group(i)) for i in (4, 5, 6))
    return saida


def tabela_lr_psd(md: str) -> dict[tuple[str, str], tuple[float, float, float]]:
    saida = {}
    for m in re.finditer(rf"^\| (SZ|MDD) \| (absolutas|relativas) \| {_NUM} \[{_NUM}, {_NUM}\] \|", md, re.M):
        saida[(m.group(1), LR_PSD_NOMES[m.group(2)])] = tuple(float(m.group(i)) for i in (3, 4, 5))
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
    return dict(delta=c["delta"], ic=list(c["ic"]), delta_repeticoes=np.asarray(c["delta_repeticoes"]).tolist())


def comparacoes() -> list[tuple[str, str]]:
    """Pares (A, B) de ΔAUC = A − B: as ablações do DTSF_Lite, o principal contra os 9 baselines e as descritivas."""
    pares = [(REFERENCIA_ABLACOES, x) for x in ABLACOES]
    pares += [(REFERENCIA_BASELINES, m) for m in CLASSICOS]
    return pares + list(DESCRITIVAS)


def delong_comparacoes(pred: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """DeLong do principal contra cada um dos outros 5 modelos principais e o p ajustado por Holm sobre os 5."""
    brutos = {}
    for b in DELONG_COMPARADOS:
        y, ea, eb = parear(escores_por_pessoa(pred[PRINCIPAL]), escores_por_pessoa(pred[b]))
        brutos[f"{PRINCIPAL} − {b}"] = teste_delong(y, ea, eb)
    ajustados = holm({k: v["p"] for k, v in brutos.items()}, alfa=ALFA)
    return {k: v | dict(p_holm=float(ajustados[k]["p_ajustado"]), rejeita=bool(ajustados[k]["rejeita"]))
            for k, v in brutos.items()}


def tabela_delong(delong: dict) -> pd.DataFrame:
    """Uma linha por (transtorno, comparação), para o delong.csv."""
    return pd.DataFrame([dict(dataset=d, comparacao=c, auc_a=r["auc_a"], auc_b=r["auc_b"], delta=r["delta"],
                              ic_baixo=r["ic"][0], ic_alto=r["ic"][1], z=r["z"], p=r["p"], p_holm=r["p_holm"],
                              rejeita=r["rejeita"], degenerado=r["degenerado"])
                         for d, comps in delong.items() for c, r in comps.items()])


def bootstrap_principal(predicoes: pd.DataFrame, ic_resumo: list[float]) -> dict:
    """Reamostras bootstrap da AUC do modelo principal; confere o IC contra o `avaliar` (mesmas reamostragens)."""
    e = escores_por_pessoa(predicoes)          # a mesma ordem das pessoas que `avaliar` usa
    b = bootstrap_roc(e["y"].to_numpy(), e["prob"].to_numpy(), semente=SEMENTE)
    ic = np.percentile(b["aucs"], [2.5, 97.5])
    for nome_ic, calculado, esperado in (("baixo", ic[0], ic_resumo[0]), ("alto", ic[1], ic_resumo[1])):
        conferir(f"bootstrap IC {nome_ic}", float(calculado), float(esperado), 1e-9)
    return dict(reamostragens=len(b["aucs"]), grade=GRADE_FPR.tolist(), ic=ic.tolist(),
                auc_media=float(b["aucs"].mean()), curvas=np.round(b["tpr"][:N_CURVAS_BOOTSTRAP], 4).tolist())


def perdas_folds(caminho: Path) -> list[dict]:
    """Uma entrada por (repetição, fold): perda de treino e de validação interna por época e a época escolhida."""
    t = pd.read_csv(caminho)
    folds = []
    for (r, f), g in t.sort_values("epoca").groupby(["repeticao", "fold"]):
        if (g["epoca"].to_numpy() != np.arange(1, len(g) + 1)).any():
            raise Divergencia(f"{caminho}: repetição {r} fold {f} sem as épocas 1, 2, … em sequência")
        validacao = g["perda_validacao"].to_numpy()
        melhor = g.loc[g["melhor"] == 1, "epoca"].tolist()
        if melhor != [1 + int(np.argmin(validacao))]:
            raise Divergencia(f"{caminho}: repetição {r} fold {f}: a época escolhida {melhor} não é a de menor "
                              f"perda de validação ({1 + int(np.argmin(validacao))})")
        folds.append(dict(repeticao=int(r), fold=int(f), melhor=melhor[0], treino=g["perda_treino"].tolist(),
                          validacao=validacao.tolist()))
    if len(folds) != 50:
        raise Divergencia(f"{caminho}: {len(folds)} folds em vez de 50 (10 repetições × 5 folds)")
    return folds


def calcular(raiz: Path = RESULTADOS) -> dict:
    md = {e: (raiz / "resumos" / f"{e}.md").read_text(encoding="utf-8") for e in ("06_LR_psd", "07_DTSF", "08_classicos")}
    esperado_aucs = tabela_modelos(md["07_DTSF"]) | tabela_modelos(md["08_classicos"]) | tabela_lr_psd(md["06_LR_psd"])
    esperado_deltas = tabela_deltas(md["07_DTSF"], REFERENCIA_ABLACOES) | tabela_deltas(md["08_classicos"], REFERENCIA_BASELINES)

    numeros = dict(modelos={}, comparacoes={}, perdas={}, bootstrap={}, delong={})
    for d in DATASETS:
        pred = {m: ler_previsoes(resolver_arquivo(raiz / "previsoes" / d / f"{m}.csv")) for m in MODELOS}
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
            delta, baixo, alto = esperado_deltas[(d, chave)]
            conferir(f"{d} {chave} Δ", c["delta"], delta)
            conferir(f"{d} {chave} IC baixo", c["ic"][0], baixo)
            conferir(f"{d} {chave} IC alto", c["ic"][1], alto)
            numeros["comparacoes"][d][chave] = _comparacao(c)
        numeros["delong"][d] = delong_comparacoes(pred)

        numeros["bootstrap"][d] = bootstrap_principal(pred[PRINCIPAL], numeros["modelos"][d][PRINCIPAL]["ic"])
        arquivo_perdas = resolver_arquivo(raiz / "perdas" / d / f"{PRINCIPAL}.csv")
        if arquivo_perdas.exists():
            numeros["perdas"][d] = perdas_folds(arquivo_perdas)

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
    tabela_delong(numeros["delong"]).to_csv(args.saida / "delong.csv", index=False)
    n = sum(len(v) for v in numeros["modelos"].values())
    print(f"{destino}: {n} (modelo, dataset) e {sum(len(v) for v in numeros['comparacoes'].values())} comparações, "
          "todos conferidos contra os resumos das etapas")
    for d, comps in numeros["delong"].items():
        print(d, {c: ("rejeita" if r["rejeita"] else "não rejeita") for c, r in comps.items()})


if __name__ == "__main__":
    main()
