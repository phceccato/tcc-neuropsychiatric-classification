"""Resultados exploratórios da etapa 14: D_artigo e D_artigo_FA contra o D da etapa 07.

- AUC por pessoa, IC e secundárias de cada modelo (`avaliar`), como na etapa 07;
- comparações com `comparacao_pareada` (pipeline/e07_DTSF/avaliar.py): permutação pareada sobre os ranks,
  unilateral, IC do Δ e as sensibilidades (Δ por repetição, t corrigido):
  - D_artigo − D e D_artigo_FA − D;
  - D_artigo_FA − D_artigo, que isola o efeito da FA no modelo grande;
- p contra o acaso de cada modelo, quando houver permutações.

Nada aqui decide hipótese: os p são descritivos e não corrigem a multiplicidade das variantes exploratórias
comparadas com o mesmo D. O resumo registra o dispositivo dos treinos e avisa se o D e as
variantes foram treinados em dispositivos diferentes.

Lê outputs/07_DTSF/resultados/<dataset>/D e outputs/14_DTSF_artigo/resultados/<dataset>/<modelo>; grava
outputs/14_DTSF_artigo/resultados/resumo.md. Modelos ainda sem previsões completas aparecem como incompletos.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pipeline.avaliacao.metricas import avaliar
from pipeline.e07_DTSF.avaliar import (SEMENTE, _epocas, comparacao_pareada, completo, formatar_ic, ler_modelo,
                                       p_acaso)
from pipeline.e07_DTSF.rodar import dispositivos_gravados
from pipeline.e14_DTSF_artigo.rodar import MODELOS

COMPARACOES = (("D_artigo", "D"), ("D_artigo_FA", "D"), ("D_artigo_FA", "D_artigo"))


def _parametros(pasta: Path) -> str:
    valores = {json.load(open(a)).get("parametros") for a in pasta.glob("r*_f*.json")}
    return ", ".join(f"{v:,}".replace(",", ".") for v in sorted(v for v in valores if v)) or "—"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--d", type=Path, default=Path("outputs/07_DTSF/resultados"))
    parser.add_argument("--permutacoes-d", type=Path, default=Path("outputs/07_DTSF/permutacoes"))
    parser.add_argument("--modelos", type=Path, default=Path("outputs/14_DTSF_artigo/resultados"))
    parser.add_argument("--permutacoes", type=Path, default=Path("outputs/14_DTSF_artigo/permutacoes"))
    args = parser.parse_args()

    disp_d = dispositivos_gravados([args.d / d / "D" for d in ("SZ", "MDD")])
    disp_v = dispositivos_gravados([args.modelos / d / m for d in ("SZ", "MDD") for m in MODELOS])
    linhas = ["# Etapa 14 (exploratória) — DTSF com as dimensões do artigo", "",
              "AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; IC de 95% por bootstrap "
              "estratificado. Secundárias no limiar 0,5. Época: mediana (mín–máx) do early stopping. Os p são "
              "descritivos: nenhuma comparação decide hipótese, e não há correção para as várias variantes "
              "exploratórias comparadas com o mesmo D.", "",
              f"Dispositivo dos treinos: D {', '.join(sorted(disp_d)) or '—'}; variantes {', '.join(sorted(disp_v)) or '—'}."
              + (" **Atenção: dispositivos diferentes; a comparação mistura o efeito do dispositivo com o da variante.**"
                 if disp_d and disp_v and disp_d != disp_v else ""), ""]
    comparacoes = []
    for dataset in ("SZ", "MDD"):
        pastas = {"D": (args.d / dataset / "D", args.permutacoes_d / dataset / "D")}
        pastas |= {m: (args.modelos / dataset / m, args.permutacoes / dataset / m) for m in MODELOS}
        predicoes = {m: ler_modelo(p) for m, (p, _) in pastas.items()}
        linhas += [f"## {dataset}", "",
                   "| Modelo | Parâmetros | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | "
                   "Sensibilidade | Especificidade | Macro F1 | Época | p contra o acaso (N) |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m, (pasta, pasta_perm) in pastas.items():
            parametros = "80.449" if m == "D" else _parametros(pasta)
            if not completo(predicoes[m]):
                linhas.append(f"| {m} | {parametros} | incompleto ({len(list(pasta.glob('r*_f*.csv')))}/50 folds) "
                              "| | | | | | | |")
                continue
            r = avaliar(predicoes[m], semente=SEMENTE)
            p_m = p_acaso(predicoes[m], pasta_perm)
            acaso = "—" if p_m is None else f"{p_m[0]:.3f} ({p_m[1]})"
            linhas.append(f"| {m} | {parametros} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                          f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
                          f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} | "
                          f"{_epocas(pasta)} | {acaso} |")
        for a, b in COMPARACOES:
            if completo(predicoes[a]) and completo(predicoes[b]):
                comparacoes.append((dataset, a, b, comparacao_pareada(predicoes[a], predicoes[b])))
        linhas.append("")
    linhas += ["## Comparações", "",
               "Permutação pareada sobre os ranks (10.000), unilateral (o primeiro modelo maior que o segundo); IC por "
               "bootstrap pareado. Sensibilidades: ΔAUC por repetição e t corrigido de Nadeau e Bengio.", "",
               "| Dataset | Comparação | ΔAUC [IC 95%] | p (ranks) | ΔAUC por repetição: mín–máx (positivos) | "
               "t corrigido p |", "| --- | --- | --- | --- | --- | --- |"]
    for dataset, a, b, c in comparacoes:
        dr = c["delta_repeticoes"]
        linhas.append(f"| {dataset} | {a} − {b} | {c['delta']:+.3f} {formatar_ic(c['ic'])} | {c['p']:.4f} | "
                      f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) | {c['t_corrigido_p']:.4f} |")
    args.modelos.mkdir(parents=True, exist_ok=True)
    (args.modelos / "resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")
    print((args.modelos / "resumo.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
