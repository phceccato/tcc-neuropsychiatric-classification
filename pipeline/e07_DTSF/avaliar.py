"""Resultados dos modelos DTSF da etapa 07 e as comparações descritivas entre eles.

Sem hipóteses a decidir (docs/avaliacao.md): a etapa reporta, para cada modelo,

- AUC por pessoa (média da probabilidade fora do fold nas 10 repetições), IC por bootstrap estratificado,
  AUC de cada repetição e a média ± desvio de todas as métricas nas 10 repetições; secundárias no limiar 0,5;
- ΔAUC entre modelos nas mesmas pessoas (`comparacao_pareada`): Δ com IC de 95% por bootstrap estratificado e
  pareado, e o Δ de cada repetição. As ablações comparam o DTSF_Lite com cada uma das suas variantes
  (DTSF_Lite_SpectralOnly, DTSF_Lite_TemporalOnly, DTSF_Lite_NoMixup); as outras comparações são DTSF_FA − DTSF (efeito da FA), DTSF − DTSF_Lite (efeito
  do tamanho) e DTSF_FA − DTSF_Lite. Nenhuma delas tem p-valor: o teste estatístico é o DeLong com Holm
  (pipeline/e11_graficos/numeros.py), do DTSF_FA contra os outros 5 modelos principais.

Lê outputs/07_DTSF/resultados/<dataset>/<modelo>/r*_f*.csv; grava outputs/07_DTSF/resultados/resumo_dtsf.md.
O resumo.md da mesma pasta, se existir, não é reescrito.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.avaliacao.metricas import (auc_em_lote, avaliar, delta_por_repeticao, escores_por_pessoa, ic_bootstrap_delta,
                                         parear, tabela_media_desvio)
from pipeline.constantes import SEMENTE
from pipeline.nomes import resolver_pasta
from pipeline.e07_DTSF.redes_dtsf import DTSF, PRINCIPAL, TIPOS_DTSF

ALFA = 0.05
REFERENCIA_ABLACOES = "DTSF_Lite"
ABLACOES = ("DTSF_Lite_SpectralOnly", "DTSF_Lite_TemporalOnly", "DTSF_Lite_NoMixup")
DESCRITIVAS = (("DTSF_FA", "DTSF"), ("DTSF", "DTSF_Lite"), ("DTSF_FA", "DTSF_Lite"))


def completo(predicoes: pd.DataFrame, repeticoes: int = 10) -> bool:
    return bool(len(predicoes)) and predicoes["repeticao"].nunique() == repeticoes \
        and len(predicoes) == repeticoes * predicoes["subject_uid"].nunique()


def comparacao_pareada(predicoes_a: pd.DataFrame, predicoes_b: pd.DataFrame, semente: int = SEMENTE) -> dict:
    """ΔAUC = AUC(a) − AUC(b) num dataset, nas mesmas pessoas: Δ, IC de 95% (bootstrap estratificado e pareado,
    condicional às previsões obtidas) e o Δ de cada repetição."""
    y, a, b = parear(escores_por_pessoa(predicoes_a), escores_por_pessoa(predicoes_b))
    delta = float(auc_em_lote(y, a[None])[0] - auc_em_lote(y, b[None])[0])
    return dict(delta=delta, ic=ic_bootstrap_delta(y, a, b, semente=semente),
                delta_repeticoes=delta_por_repeticao(predicoes_a, predicoes_b))


def ler_modelo(pasta: Path) -> pd.DataFrame:
    """As previsões de um modelo; aceita a pasta de nome antigo de uma rodada anterior à renomeação (pipeline/nomes.py)."""
    arquivos = sorted(resolver_pasta(pasta).glob("r*_f*.csv"))
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True) if arquivos else pd.DataFrame()


def _epocas(pasta: Path) -> str:
    epocas = [json.load(open(a))["melhor_epoca"] for a in resolver_pasta(pasta).glob("r*_f*.json")]
    return f"{np.median(epocas):.0f} ({min(epocas)}–{max(epocas)})" if epocas else "—"


def parametros(modelo: str) -> str:
    """Número de parâmetros do modelo, com ponto de milhar."""
    return f"{sum(p.numel() for p in DTSF(modelo).parameters()):,}".replace(",", ".")


def _linha_delta(dataset: str, x: str, z: str, c: dict) -> str:
    dr = c["delta_repeticoes"]
    return (f"| {dataset} | {x} − {z} | {c['delta']:+.3f} {formatar_ic(c['ic'])} | "
            f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) |")


CABECALHO_DELTA = ["| Dataset | Comparação | ΔAUC [IC 95%] | ΔAUC por repetição: mín–máx (positivos) |",
                   "| --- | --- | --- | --- |"]


def linhas_ablacoes(ablacoes: list[tuple[str, str, dict]]) -> list[str]:
    """DTSF_Lite − cada ablação: Δ e IC descritivos."""
    linhas = ["## Ablações do DTSF_Lite", "",
              f"ΔAUC = {REFERENCIA_ABLACOES} − ablação, nas mesmas pessoas. O IC é o bootstrap estratificado e pareado, "
              "condicional às previsões obtidas. Descritivo: sem p-valor.", ""]
    if not ablacoes:
        return linhas + ["Ainda sem os modelos completos.", ""]
    return linhas + CABECALHO_DELTA + [_linha_delta(d, REFERENCIA_ABLACOES, abl, c) for d, abl, c in ablacoes] + [""]


def linhas_descritivas(descritivas: list[tuple[str, str, str, dict]]) -> list[str]:
    """DTSF_FA − DTSF, DTSF − DTSF_Lite e DTSF_FA − DTSF_Lite: Δ e IC descritivos."""
    linhas = ["## Comparações entre os modelos da família DTSF", "",
              "DTSF_FA − DTSF isola a FA; DTSF − DTSF_Lite isola o tamanho (dimensões do artigo contra as reduzidas). Descritivo: "
              "sem p-valor; o teste estatístico do DTSF_FA é o DeLong com Holm (etapa 11).", ""]
    if not descritivas:
        return linhas + ["Ainda sem os modelos completos.", ""]
    return linhas + CABECALHO_DELTA + [_linha_delta(d, x, z, c) for d, x, z, c in descritivas] + [""]


def formatar_ic(par: tuple[float, float]) -> str:
    return f"[{par[0]:+.3f}, {par[1]:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modelos", type=Path, default=Path("outputs/07_DTSF/resultados"))
    args = parser.parse_args()

    linhas = ["# DTSF — resultados da etapa 07", "",
              f"Modelo principal: {PRINCIPAL}. AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; "
              "IC de 95% por bootstrap estratificado. Secundárias no limiar 0,5, só descritivas. Época: mediana "
              "(mín–máx) da época escolhida pelo early stopping.", ""]
    ablacoes: list[tuple[str, str, dict]] = []
    descritivas: list[tuple[str, str, str, dict]] = []
    for dataset in ("SZ", "MDD"):
        predicoes = {m: ler_modelo(args.modelos / dataset / m) for m in TIPOS_DTSF}
        completos = {m: p for m, p in predicoes.items() if completo(p)}
        if REFERENCIA_ABLACOES in completos:
            ablacoes += [(dataset, abl, comparacao_pareada(completos[REFERENCIA_ABLACOES], completos[abl]))
                         for abl in ABLACOES if abl in completos]
        descritivas += [(dataset, x, z, comparacao_pareada(completos[x], completos[z]))
                        for x, z in DESCRITIVAS if x in completos and z in completos]
        linhas += [f"## {dataset}", "",
                   "| Modelo | Parâmetros | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | "
                   "Sensibilidade | Especificidade | Macro F1 | Época |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m in TIPOS_DTSF:
            if m not in completos:
                linhas.append(f"| {m} | {parametros(m)} | incompleto | | | | | | |")
                continue
            r = avaliar(completos[m], semente=SEMENTE)
            linhas.append(f"| {m} | {parametros(m)} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                          f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
                          f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} | "
                          f"{_epocas(args.modelos / dataset / m)} |")
        linhas += ["", "Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.", ""]
        linhas += tabela_media_desvio([([m], completos.get(m)) for m in TIPOS_DTSF], ["Modelo"])
        linhas.append("")
    linhas += linhas_ablacoes(ablacoes)
    linhas += linhas_descritivas(descritivas)
    (args.modelos / "resumo_dtsf.md").write_text("\n".join(linhas), encoding="utf-8")
    print((args.modelos / "resumo_dtsf.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
