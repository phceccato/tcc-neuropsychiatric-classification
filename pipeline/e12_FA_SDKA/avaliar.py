"""Resultados exploratórios da etapa 12: cada variante (FA, SDKA, FA + SDKA) contra o D da etapa 07.

- AUC por pessoa, IC e secundárias de cada modelo (`avaliar`), como na etapa 07;
- variante − D com `comparacao_pareada` (pipeline/e07_DTSF/avaliar.py): permutação pareada sobre os
  ranks, unilateral variante > D, IC do Δ e as sensibilidades (Δ por repetição, t corrigido);
- p contra o acaso, quando houver permutações em outputs/12_FA_SDKA/permutacoes.

Nada aqui decide hipótese: os p são descritivos, sem Holm nem interseção-união. As variantes com SDKA
usam o EEG das pessoas de teste no treino (transdutivo), então a comparação com o D não é simétrica.

Lê outputs/07_DTSF/resultados/<dataset>/D e outputs/12_FA_SDKA/resultados/<dataset>/<modelo>; grava
outputs/12_FA_SDKA/resultados/resumo.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from pipeline.avaliacao.metricas import avaliar
from pipeline.e07_DTSF.avaliar import (SEMENTE, _epocas, comparacao_pareada, completo, formatar_ic, ler_modelo,
                                       p_acaso)
from pipeline.e07_DTSF.rodar import dispositivos_gravados
from pipeline.e12_FA_SDKA.redes import TIPOS_12


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d", type=Path, default=Path("outputs/07_DTSF/resultados"))
    parser.add_argument("--permutacoes-d", type=Path, default=Path("outputs/07_DTSF/permutacoes"))
    parser.add_argument("--modelos", type=Path, default=Path("outputs/12_FA_SDKA/resultados"))
    parser.add_argument("--permutacoes", type=Path, default=Path("outputs/12_FA_SDKA/permutacoes"))
    args = parser.parse_args()

    linhas = ["# Etapa 12 (exploratória) — FA e SDKA no DTSF adaptado", "",
              "AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; IC de 95% por bootstrap "
              "estratificado. Secundárias no limiar 0,5. Época: mediana (mín–máx) do early stopping. "
              "Os p são descritivos: nenhuma comparação decide hipótese. As variantes com SDKA usam o EEG das "
              "pessoas de teste no treino (transdutivo).", ""]
    disp_d = dispositivos_gravados([args.d / d / "D" for d in ("SZ", "MDD")])
    disp_v = dispositivos_gravados([args.modelos / d / m for d in ("SZ", "MDD") for m in TIPOS_12])
    linhas += [f"Dispositivo dos treinos: D {', '.join(sorted(disp_d)) or '—'}; variantes {', '.join(sorted(disp_v)) or '—'}."
               + (" **Atenção: o D e as variantes foram treinados em dispositivos diferentes; a comparação mistura "
                  "o efeito do dispositivo com o da variante.**" if disp_d and disp_v and disp_d != disp_v else ""), ""]
    comparacoes = []
    for dataset in ("SZ", "MDD"):
        pastas = {"D": (args.d / dataset / "D", args.permutacoes_d / dataset / "D")}
        pastas |= {m: (args.modelos / dataset / m, args.permutacoes / dataset / m) for m in TIPOS_12}
        predicoes = {m: ler_modelo(p) for m, (p, _) in pastas.items()}
        linhas += [f"## {dataset}", "",
                   "| Modelo | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | "
                   "Sensibilidade | Especificidade | Macro F1 | Época | p contra o acaso (N) |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, (pasta, pasta_perm) in pastas.items():
            if not completo(predicoes[m]):
                linhas.append(f"| {m} | incompleto ({len(list(pasta.glob('r*_f*.csv')))}/50 folds) | | | | | | | |")
                continue
            r = avaliar(predicoes[m], semente=SEMENTE)
            p_m = p_acaso(predicoes[m], pasta_perm)
            acaso = "—" if p_m is None else f"{p_m[0]:.3f} ({p_m[1]})"
            linhas.append(f"| {m} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                          f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
                          f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} | "
                          f"{_epocas(pasta)} | {acaso} |")
            if m != "D" and completo(predicoes["D"]):
                comparacoes.append((dataset, m, comparacao_pareada(predicoes[m], predicoes["D"])))
        linhas.append("")
    linhas += ["## Variante − D", "",
               "Permutação pareada sobre os ranks (10.000), unilateral variante > D; IC por bootstrap pareado. "
               "Sensibilidades: ΔAUC por repetição e t corrigido de Nadeau e Bengio.", "",
               "| Dataset | Comparação | ΔAUC [IC 95%] | p (ranks) | ΔAUC por repetição: mín–máx (positivos) | "
               "t corrigido p |", "| --- | --- | --- | --- | --- | --- |"]
    for dataset, m, c in comparacoes:
        dr = c["delta_repeticoes"]
        linhas.append(f"| {dataset} | {m} − D | {c['delta']:+.3f} {formatar_ic(c['ic'])} | {c['p']:.4f} | "
                      f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) | "
                      f"{c['t_corrigido_p']:.4f} |")
    args.modelos.mkdir(parents=True, exist_ok=True)
    (args.modelos / "resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print((args.modelos / "resumo.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
