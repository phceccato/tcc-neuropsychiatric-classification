"""Resultados dos baselines clássicos e a comparação com o DTSF adaptado (H4).

Plano em docs/00_proposta-tcc.md:

- AUC por pessoa (média da probabilidade fora do fold nas 10 repetições), IC por bootstrap estratificado
  e secundárias descritivas, como em pipeline/e07_DTSF/avaliar.py;
- comparação D − baseline, para cada um dos 9 baselines, sobre os escores por pessoa: permutação pareada
  sobre os ranks (10.000), unilateral D > baseline, com IC do Δ e as sensibilidades (Δ por
  repetição, t corrigido), com `comparacao_pareada` (pipeline/e07_DTSF/avaliar.py);
- H4 num dataset: D supera os 9 baselines (interseção-união; p = maior dos 9 p); decisão por Holm sobre
  {SZ, MDD}, α = 0,05;
- exploratório (sem decisão): interseção-união por modalidade (3 temporais, 3 espectrais, 3 combinados).

Lê outputs/08_classicos/resultados/<dataset>/<modelo>/r*_f*.{csv,json} e
outputs/07_DTSF/resultados/<dataset>/D; grava só outputs/08_classicos/resultados/resumo.md.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.avaliacao.metricas import avaliar, holm, p_interseccao_uniao
from pipeline.e08_classicos.treino import CLASSIFICADORES, MODALIDADES, MODELOS
from pipeline.e07_DTSF.avaliar import ALFA, SEMENTE, formatar_ic, ler_modelo, comparacao_pareada, completo

REFERENCIA = "D"
PLURAL = {"temporal": "temporais", "espectral": "espectrais", "combinado": "combinados"}

def comparacoes(predicoes_d: pd.DataFrame, baselines: dict[str, pd.DataFrame]) -> dict[str, dict]:
    """D contra cada baseline: Δ, IC, p (ranks) e sensibilidades (`comparacao_pareada`)."""
    return {m: comparacao_pareada(predicoes_d, p) for m, p in baselines.items()}

def decidir_h4(resultados: dict[str, dict[str, dict]], alfa: float = ALFA) -> dict[str, dict]:
    """H4 por dataset: p de interseção-união sobre os baselines (`MODELOS`) e Holm sobre os datasets."""
    for dataset, r in resultados.items():
        if set(r) != set(MODELOS):
            raise ValueError(f"{dataset}: a H4 precisa dos {len(MODELOS)} baselines completos, tem {sorted(r)}")
    p_iu = {dataset: p_interseccao_uniao([c["p"] for c in r.values()]) for dataset, r in resultados.items()}
    return holm(p_iu, alfa=alfa)

def _hiperparametros(pasta: Path) -> str:
    escolhas = [json.dumps(json.load(open(a))["parametros"], sort_keys=True) for a in sorted(pasta.glob("r*_f*.json"))]
    if not escolhas:
        return "—"
    contagem = pd.Series(escolhas).value_counts()
    return "; ".join(f"{k} ×{v}" for k, v in contagem.items())

def linhas_h4(resultados: dict[str, dict[str, dict]]) -> list[str]:
    decisao = decidir_h4(resultados)
    linhas = ["## H4: D contra os baselines clássicos (condicional às previsões fora do fold)", "",
              "Cada comparação: permutação pareada sobre os ranks (10.000), unilateral D > baseline. H4 num dataset: "
              f"D supera os {len(MODELOS)} baselines (interseção-união, p = maior p); decisão por Holm sobre SZ e MDD (α = 0,05). "
              "O IC é o bootstrap estratificado e pareado, condicional às previsões obtidas.", "",
              "**Validade.** A interseção-união e o Holm controlam só a multiplicidade. Eles supõem que cada teste "
              "componente tem nível α, e a validação cruzada não garante isso: as pessoas de um fold de teste "
              "compartilham o modelo, os treinos se sobrepõem e o escore é a média de repetições com partições "
              "sobrepostas. A permutação troca os escores dentro de cada pessoa como se elas fossem independentes. "
              "O p vale condicionado às previsões obtidas; para a pergunta sobre o método, o erro tipo I real "
              "pode passar de α (~10–13% na simulação do plano do DTSF, não validado para esta comparação).", "",
              "| Dataset | p interseção-união | p ajustado (Holm) | H4 |", "| --- | --- | --- | --- |"]
    for dataset in resultados:
        d = decisao[dataset]
        linhas.append(f"| {dataset} | {d['p']:.4f} | {d['p_ajustado']:.4f} | "
                      f"{'rejeita H0' if d['rejeita'] else 'não rejeita H0'} |")
    linhas += ["", "Comparações componentes e sensibilidades (as sensibilidades não mudam a decisão):", "",
               "| Dataset | Baseline | ΔAUC D − baseline [IC 95%] | p (ranks) | "
               "ΔAUC por repetição: mín–máx (positivos) | t corrigido p |",
               "| --- | --- | --- | --- | --- | --- |"]
    for dataset, r in resultados.items():
        for m, c in r.items():
            dr = c["delta_repeticoes"]
            linhas.append(f"| {dataset} | {m} | {c['delta']:+.3f} {formatar_ic(c['ic'])} | {c['p']:.4f} | "
                          f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) | "
                          f"{c['t_corrigido_p']:.4f} |")
    linhas += ["", "Exploratório (sem decisão): interseção-união por modalidade.", "",
               "| Dataset | " + " | ".join(f"p D > os {len(CLASSIFICADORES)} {PLURAL[m]}" for m in MODALIDADES) + " |",
               "| --- |" + " --- |" * len(MODALIDADES)]
    for dataset, r in resultados.items():
        ps = [p_interseccao_uniao([r[f"{c}_{m}"]["p"] for c in CLASSIFICADORES]) for m in MODALIDADES]
        linhas.append(f"| {dataset} | " + " | ".join(f"{p:.4f}" for p in ps) + " |")
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
        linhas.append("")
    if set(resultados) == {"SZ", "MDD"}:
        linhas += linhas_h4(resultados) + [""]
    else:
        linhas += [f"A H4 só é calculada com o D e os {len(MODELOS)} baselines completos nos dois datasets.", ""]
    args.classicos.mkdir(parents=True, exist_ok=True)
    (args.classicos / "resumo.md").write_text("\n".join(linhas), encoding="utf-8")
    print((args.classicos / "resumo.md").read_text(encoding="utf-8"))

if __name__ == "__main__":
    main()
