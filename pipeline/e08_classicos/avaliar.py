"""Resultados dos baselines clássicos e a comparação com o modelo principal da etapa 07 (D_FA).

Sem hipóteses a decidir (docs/avaliacao.md):

- AUC por pessoa (média da probabilidade fora do fold nas 10 repetições), IC por bootstrap estratificado,
  média ± desvio nas 10 repetições e secundárias descritivas, como em pipeline/e07_DTSF/avaliar.py;
- ΔAUC = D_FA − baseline, para cada um dos 9 baselines (LR, RF e KNN sobre as características temporais,
  espectrais e combinadas), nas mesmas pessoas, com IC de 95% (bootstrap estratificado e pareado) e o Δ de cada
  repetição (`comparacao_pareada`, pipeline/e07_DTSF/avaliar.py). Sem p-valor: o teste estatístico é o DeLong com
  Holm do D_FA contra os 3 baselines combinados e os outros modelos principais (pipeline/e11_graficos/numeros.py).

Lê outputs/08_classicos/resultados/<dataset>/<modelo>/r*_f*.{csv,json} e
outputs/07_DTSF/resultados/<dataset>/<modelo principal> (D_FA); grava só outputs/08_classicos/resultados/resumo.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from pipeline.avaliacao.metricas import avaliar, tabela_media_desvio
from pipeline.e08_classicos.treino import MODELOS
from pipeline.e07_DTSF.avaliar import CABECALHO_DELTA, SEMENTE, _linha_delta, ler_modelo, comparacao_pareada, completo
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL

REFERENCIA = PRINCIPAL

def comparacoes(predicoes_d: pd.DataFrame, baselines: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """O modelo principal contra cada baseline: Δ, IC e Δ por repetição (`comparacao_pareada`)."""
    return {m: comparacao_pareada(predicoes_d, p) for m, p in baselines.items()}

def _hiperparametros(pasta: Path) -> str:
    escolhas = [json.dumps(json.load(open(a))["parametros"], sort_keys=True) for a in sorted(pasta.glob("r*_f*.json"))]
    if not escolhas:
        return "—"
    contagem = pd.Series(escolhas).value_counts()
    return "; ".join(f"{k} ×{v}" for k, v in contagem.items())

def linhas_comparacoes(resultados: dict[str, dict[str, dict]]) -> list[str]:
    linhas = [f"## {REFERENCIA} contra os baselines clássicos", "",
              f"ΔAUC = {REFERENCIA} − baseline, nas mesmas pessoas, para os {len(MODELOS)} baselines. O IC é o bootstrap "
              "estratificado e pareado, condicional às previsões obtidas. Descritivo: sem p-valor; o teste estatístico "
              "é o DeLong com Holm (etapa 11).", ""] + CABECALHO_DELTA
    for dataset, r in resultados.items():
        linhas += [_linha_delta(dataset, REFERENCIA, m, r[m]) for m in MODELOS]
    return linhas

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--classicos", type=Path, default=Path("outputs/08_classicos/resultados"))
    parser.add_argument("--modelos", type=Path, default=Path("outputs/07_DTSF/resultados"))
    args = parser.parse_args()

    linhas = ["# Baselines clássicos — resultados", "",
              "AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; IC de 95% por bootstrap "
              "estratificado. Secundárias no limiar 0,5, só descritivas. Hiperparâmetros: escolhas da seleção interna "
              "nos 50 folds.", ""]
    resultados: dict[str, dict[str, dict]] = {}
    for dataset in ("SZ", "MDD"):
        predicoes = {m: ler_modelo(args.classicos / dataset / m) for m in MODELOS}
        predicoes[REFERENCIA] = ler_modelo(args.modelos / dataset / REFERENCIA)
        linhas += [f"## {dataset}", "",
                   "| Modelo | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | Sensibilidade | "
                   "Especificidade | Macro F1 | Hiperparâmetros escolhidos |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m in (REFERENCIA, *MODELOS):
            if not completo(predicoes[m]):
                linhas.append(f"| {m} | incompleto | | | | | | |")
                continue
            r = avaliar(predicoes[m], semente=SEMENTE)
            hp = "—" if m == REFERENCIA else _hiperparametros(args.classicos / dataset / m)
            linhas.append(f"| {m} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                          f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
                          f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} | {hp} |")
        if all(completo(predicoes[m]) for m in (REFERENCIA, *MODELOS)):
            resultados[dataset] = comparacoes(predicoes[REFERENCIA], {m: predicoes[m] for m in MODELOS})
        linhas += ["", "Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a população.", ""]
        linhas += tabela_media_desvio([([m], predicoes[m] if completo(predicoes[m]) else None)
                                       for m in (REFERENCIA, *MODELOS)], ["Modelo"])
        linhas.append("")
    if set(resultados) == {"SZ", "MDD"}:
        linhas += linhas_comparacoes(resultados) + [""]
    else:
        linhas += [f"As comparações só são calculadas com o {REFERENCIA} e os {len(MODELOS)} baselines completos nos dois datasets.", ""]
    args.classicos.mkdir(parents=True, exist_ok=True)
    (args.classicos / "resumo.md").write_text("\n".join(linhas), encoding="utf-8")
    print((args.classicos / "resumo.md").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
