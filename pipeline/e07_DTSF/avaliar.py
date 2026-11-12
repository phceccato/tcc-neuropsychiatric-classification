"""Resultados dos modelos DTSF da etapa 07 e a comparação pareada por pessoa usada nas hipóteses.

Segue o protocolo estatístico de docs/00_proposta-tcc.md:

- AUC por pessoa (média da probabilidade fora do fold nas 10 repetições), IC por bootstrap
  estratificado e AUC de cada repetição; métricas secundárias só descritivas;
- H1, contra o acaso, sobre o modelo principal (D_FA): AUC da repetição 0 contra as AUCs das 25 permutações
  de rótulo com retreino (outputs/07_DTSF/permutacoes), p = (nº ≥ observado + 1) / (N + 1); decidida em cada
  transtorno a α = 0,05, sem Holm (o menor p possível, 1/26, não passaria no 1º passo do Holm);
- H2 (D_FA > D_sem_T e D_FA > D_sem_E, interseção-união) e H3 (D_FA > D_sem_mixup), sobre o modelo principal,
  de que as ablações derivam: ΔAUC = D_FA − ablação com `comparacao_pareada`; decisão por Holm sobre
  {SZ, MDD}, α = 0,05, separadamente para cada hipótese;
- comparações descritivas, sem decisão: D_FA − D (efeito da FA), D − D_pequeno (efeito do tamanho) e
  D_FA − D_pequeno.
- `comparacao_pareada`: o teste de dois modelos nas mesmas pessoas e partições (permutação pareada
  sobre os ranks, unilateral, com IC do Δ e as sensibilidades Δ por repetição e t corrigido de
  Nadeau e Bengio). É usado por H2, H3 e pela H4 (pipeline/e08_classicos/avaliar.py).

Lê outputs/07_DTSF/resultados/<dataset>/<modelo>/r*_f*.csv; grava outputs/07_DTSF/resultados/resumo_dtsf.md.
O resumo.md da mesma pasta, se existir, não é reescrito.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import rankdata

from pipeline.avaliacao.metricas import (PERMUTACOES_PAREADAS, avaliar, escores_por_pessoa,
                                         delta_por_fold, delta_por_repeticao, holm,
                                         ic_bootstrap_delta, p_interseccao_uniao, p_valor, parear, roc_auc_score,
                                         t_corrigido, teste_pareado)
from pipeline.e07_DTSF.redes_dtsf import DTSF, PRINCIPAL, TIPOS_DTSF

ALFA = 0.05
SEMENTE = 20261108
REFERENCIA_ABLACOES = PRINCIPAL
ABLACOES = ("D_sem_T", "D_sem_E", "D_sem_mixup")
PERMUTACOES_H1 = 25
HIPOTESES = {"H2": ("D_sem_T", "D_sem_E"), "H3": ("D_sem_mixup",)}
DESCRITIVAS = (("D_FA", "D"), ("D", "D_pequeno"), ("D_FA", "D_pequeno"))
ENUNCIADOS = {"H2": f"{PRINCIPAL} contra um ramo só ({PRINCIPAL} > D_sem_T e {PRINCIPAL} > D_sem_E; interseção-união, "
                    "p = maior p)",
              "H3": f"Mixup no treino ({PRINCIPAL} > D_sem_mixup)"}


def completo(predicoes: pd.DataFrame, repeticoes: int = 10) -> bool:
    return bool(len(predicoes)) and predicoes["repeticao"].nunique() == repeticoes \
        and len(predicoes) == repeticoes * predicoes["subject_uid"].nunique()


def teste_ranks(escores_a: pd.DataFrame, escores_b: pd.DataFrame, n: int = PERMUTACOES_PAREADAS,
                semente: int = SEMENTE) -> dict:
    """Modelo a contra modelo b nas mesmas pessoas, unilateral a > b.

    Recebe os escores por pessoa (`escores_por_pessoa`) dos dois modelos e pareia pela pessoa.
    Os escores de cada modelo viram ranks entre todas as pessoas (rank médio nos empates); a
    permutação pareada troca os dois ranks de cada pessoa (`teste_pareado`, n permutações). A AUC
    só depende da ordem, então o ΔAUC é o mesmo dos escores. O IC é o bootstrap estratificado e
    pareado, condicional às previsões obtidas.
    """
    y, a, b = parear(escores_a, escores_b)
    delta, p = teste_pareado(y, rankdata(a, method="average"), rankdata(b, method="average"), n=n, semente=semente)
    return dict(delta=delta, p=p, ic=ic_bootstrap_delta(y, a, b, semente=semente))


def comparacao_pareada(predicoes_a: pd.DataFrame, predicoes_b: pd.DataFrame, n: int = PERMUTACOES_PAREADAS,
                       semente: int = SEMENTE) -> dict:
    """Teste primário (`teste_ranks`) e IC de a − b num dataset, mais as sensibilidades."""
    escores_a, escores_b = escores_por_pessoa(predicoes_a), escores_por_pessoa(predicoes_b)
    r = teste_ranks(escores_a, escores_b, n=n, semente=semente)
    r["delta_repeticoes"] = delta_por_repeticao(predicoes_a, predicoes_b)
    folds = delta_por_fold(predicoes_a, predicoes_b)
    n_pessoas = escores_a["subject_uid"].nunique()
    r["delta_folds"] = folds["delta"].to_numpy()
    r["razao_teste_treino"] = float((folds["n_teste"] / (n_pessoas - folds["n_teste"])).mean())
    r["t_corrigido_p"] = t_corrigido(r["delta_folds"], r["razao_teste_treino"])[2]
    return r


def comparacoes_hipotese(predicoes: dict[str, pd.DataFrame], ablacoes: tuple[str, ...],
                         n: int = PERMUTACOES_PAREADAS, referencia: str = REFERENCIA_ABLACOES) -> dict[str, dict]:
    """O modelo principal contra cada ablação da hipótese num dataset: ΔAUC = principal − ablação."""
    return {abl: comparacao_pareada(predicoes[referencia], predicoes[abl], n=n) for abl in ablacoes}


def decidir_hipotese(resultados: dict[str, dict[str, dict]], ablacoes: tuple[str, ...],
                     alfa: float = ALFA) -> dict[str, dict]:
    """H2 ou H3: interseção-união sobre as ablações em cada dataset e Holm sobre {SZ, MDD}."""
    if set(resultados) != {"SZ", "MDD"}:
        raise ValueError(f"a hipótese precisa dos dois datasets, tem {sorted(resultados)}")
    for dataset, r in resultados.items():
        if set(r) != set(ablacoes):
            raise ValueError(f"{dataset}: a hipótese precisa de {sorted(ablacoes)}, tem {sorted(r)}")
    return holm({d: p_interseccao_uniao([c["p"] for c in r.values()]) for d, r in resultados.items()}, alfa=alfa)


def decidir_h1(acaso: dict[str, tuple[float, int]], alfa: float = ALFA) -> dict[str, dict]:
    """H1 em cada dataset, sem Holm: rejeita se p ≤ α, com as 25 permutações completas."""
    for dataset, (_, n) in acaso.items():
        if n != PERMUTACOES_H1:
            raise ValueError(f"{dataset}: a H1 precisa de {PERMUTACOES_H1} permutações completas, tem {n}")
    return {dataset: dict(p=p, n=n, rejeita=p <= alfa) for dataset, (p, n) in acaso.items()}


_VALIDADE = ("**Validade.** A interseção-união e o Holm controlam só a multiplicidade. A validação cruzada não "
             "garante que cada permutação tenha nível α: as pessoas de um fold de teste compartilham o modelo, os "
             "treinos se sobrepõem e o escore é a média de repetições com partições sobrepostas. O p vale "
             "condicionado às previsões obtidas; para a pergunta sobre o método, o erro tipo I real pode passar "
             "de α (~10–13% na simulação do plano do DTSF).")


def linhas_h1(acaso: dict[str, tuple[float, int] | None]) -> list[str]:
    linhas = [f"## H1: {PRINCIPAL} acima do acaso", "",
              f"AUC da repetição 0 contra {PERMUTACOES_H1} permutações de rótulo com retreino; decidida em cada "
              "transtorno a α = 0,05, sem Holm (o menor p possível é 1/26 ≈ 0,038).", ""]
    if any(v is None or v[1] != PERMUTACOES_H1 for v in acaso.values()):
        return linhas + [f"H1 incompleta: faltam as {PERMUTACOES_H1} permutações completas do {PRINCIPAL} nos dois "
                         "datasets.", ""]
    decisao = decidir_h1(acaso)
    linhas += ["| Dataset | p (N) | H1 |", "| --- | --- | --- |"]
    for dataset, d in decisao.items():
        linhas.append(f"| {dataset} | {d['p']:.3f} ({d['n']}) | {'rejeita H0' if d['rejeita'] else 'não rejeita H0'} |")
    return linhas + [""]


def linhas_hipotese(nome: str, resultados: dict[str, dict[str, dict]]) -> list[str]:
    ablacoes = HIPOTESES[nome]
    decisao = decidir_hipotese(resultados, ablacoes)
    linhas = [f"## {nome}: {ENUNCIADOS[nome]}", "",
              f"Cada comparação: permutação pareada sobre os ranks (10.000), unilateral {REFERENCIA_ABLACOES} > ablação; decisão por Holm "
              "sobre SZ e MDD (α = 0,05). O IC é o bootstrap estratificado e pareado, condicional às previsões obtidas.",
              "", _VALIDADE, "",
              f"| Dataset | p{' interseção-união' if len(ablacoes) > 1 else ''} | p ajustado (Holm) | {nome} |",
              "| --- | --- | --- | --- |"]
    for dataset, d in decisao.items():
        linhas.append(f"| {dataset} | {d['p']:.4f} | {d['p_ajustado']:.4f} | "
                      f"{'rejeita H0' if d['rejeita'] else 'não rejeita H0'} |")
    linhas += ["", "Comparações e sensibilidades (as sensibilidades não mudam a decisão):", "",
               "| Dataset | Comparação | ΔAUC [IC 95%] | p (ranks) | "
               "ΔAUC por repetição: mín–máx (positivos) | t corrigido p |",
               "| --- | --- | --- | --- | --- | --- |"]
    for dataset, r in resultados.items():
        for abl, c in r.items():
            dr = c["delta_repeticoes"]
            linhas.append(f"| {dataset} | {REFERENCIA_ABLACOES} − {abl} | {c['delta']:+.3f} {formatar_ic(c['ic'])} | {c['p']:.4f} | "
                          f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) | "
                          f"{c['t_corrigido_p']:.4f} |")
    return linhas + [""]


def ler_modelo(pasta: Path) -> pd.DataFrame:
    arquivos = sorted(pasta.glob("r*_f*.csv"))
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True) if arquivos else pd.DataFrame()


def aucs_nulas(pasta_permutacoes: Path, folds: int = 5) -> np.ndarray:
    """AUC de cada permutação completa (todas as `folds` gravadas), sobre as previsões fora do fold."""
    por_k: dict[str, list[Path]] = {}
    for a in pasta_permutacoes.glob("p*_f*.csv"):
        por_k.setdefault(a.stem.split("_")[0], []).append(a)
    nulas = []
    for arquivos in por_k.values():
        if len(arquivos) == folds:
            p = pd.concat([pd.read_csv(a) for a in arquivos])
            nulas.append(roc_auc_score(p["y"], p["prob"]))
    return np.array(nulas)


def auc_repeticao0(real: pd.DataFrame) -> float:
    """AUC observada da H1: a da repetição 0, a única retreinada nas permutações."""
    r0 = real[real["repeticao"] == 0]
    return float(roc_auc_score(r0["y"], r0["prob"]))


def p_acaso(real: pd.DataFrame, pasta_permutacoes: Path, folds: int = 5) -> tuple[float, int] | None:
    """p da AUC da repetição 0 contra as permutações completas (todas as `folds` gravadas)."""
    nulas = aucs_nulas(pasta_permutacoes, folds)
    if not len(nulas):
        return None
    return p_valor(auc_repeticao0(real), nulas), len(nulas)


def _epocas(pasta: Path) -> str:
    epocas = [json.load(open(a))["melhor_epoca"] for a in pasta.glob("r*_f*.json")]
    return f"{np.median(epocas):.0f} ({min(epocas)}–{max(epocas)})" if epocas else "—"


def parametros(modelo: str) -> str:
    """Número de parâmetros do modelo, com ponto de milhar."""
    return f"{sum(p.numel() for p in DTSF(modelo).parameters()):,}".replace(",", ".")


def linhas_descritivas(descritivas: list[tuple[str, str, str, dict]]) -> list[str]:
    """D_FA − D, D − D_pequeno e D_FA − D_pequeno: Δ, IC e p descritivos, sem decisão."""
    linhas = ["## Comparações descritivas (sem decisão)", "",
              "D_FA − D isola a FA; D − D_pequeno isola o tamanho (dimensões do artigo contra as reduzidas). Permutação "
              "pareada sobre os ranks (10.000), unilateral (o primeiro maior que o segundo); os p não decidem "
              "hipótese.", ""]
    if not descritivas:
        return linhas + ["Ainda sem os modelos completos.", ""]
    linhas += ["| Dataset | Comparação | ΔAUC [IC 95%] | p (ranks) | ΔAUC por repetição: mín–máx (positivos) | "
               "t corrigido p |", "| --- | --- | --- | --- | --- | --- |"]
    for dataset, x, z, c in descritivas:
        dr = c["delta_repeticoes"]
        linhas.append(f"| {dataset} | {x} − {z} | {c['delta']:+.3f} {formatar_ic(c['ic'])} | {c['p']:.4f} | "
                      f"{dr.min():+.3f} a {dr.max():+.3f} ({int((dr > 0).sum())}/{len(dr)}) | {c['t_corrigido_p']:.4f} |")
    return linhas + [""]


def formatar_ic(par: tuple[float, float]) -> str:
    return f"[{par[0]:+.3f}, {par[1]:+.3f}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--modelos", type=Path, default=Path("outputs/07_DTSF/resultados"))
    parser.add_argument("--permutacoes", type=Path, default=Path("outputs/07_DTSF/permutacoes"))
    args = parser.parse_args()

    linhas = ["# DTSF — resultados da etapa 07", "",
              f"Modelo principal: {PRINCIPAL}. AUC por pessoa: média da probabilidade fora do fold nas 10 repetições; "
              "IC de 95% por bootstrap estratificado. Secundárias no limiar 0,5, só descritivas. Época: mediana "
              "(mín–máx) da época escolhida pelo early stopping.", ""]
    acaso: dict[str, tuple[float, int] | None] = {}
    hipoteses: dict[str, dict[str, dict]] = {h: {} for h in HIPOTESES}
    descritivas: list[tuple[str, str, str, dict]] = []
    for dataset in ("SZ", "MDD"):
        predicoes = {m: ler_modelo(args.modelos / dataset / m) for m in TIPOS_DTSF}
        completos = {m: p for m, p in predicoes.items() if completo(p)}
        acaso[dataset] = (p_acaso(completos[PRINCIPAL], args.permutacoes / dataset / PRINCIPAL)
                          if PRINCIPAL in completos else None)
        for h, ablacoes in HIPOTESES.items():
            if all(m in completos for m in (REFERENCIA_ABLACOES, *ablacoes)):
                hipoteses[h][dataset] = comparacoes_hipotese(completos, ablacoes)
        descritivas += [(dataset, x, z, comparacao_pareada(completos[x], completos[z]))
                        for x, z in DESCRITIVAS if x in completos and z in completos]
        linhas += [f"## {dataset}", "",
                   "| Modelo | Parâmetros | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | "
                   "Sensibilidade | Especificidade | Macro F1 | Época | p contra o acaso (N) |",
                   "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        for m in TIPOS_DTSF:
            if m not in completos:
                linhas.append(f"| {m} | {parametros(m)} | incompleto | | | | | | | |")
                continue
            r = avaliar(completos[m], semente=SEMENTE)
            p_m = p_acaso(completos[m], args.permutacoes / dataset / m)
            acaso_txt = "—" if p_m is None else f"{p_m[0]:.3f} ({p_m[1]})"
            linhas.append(f"| {m} | {parametros(m)} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                          f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
                          f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} | "
                          f"{_epocas(args.modelos / dataset / m)} | {acaso_txt} |")
        linhas.append("")
    linhas += linhas_h1(acaso)
    for h in HIPOTESES:
        if set(hipoteses[h]) == {"SZ", "MDD"}:
            linhas += linhas_hipotese(h, hipoteses[h])
        else:
            linhas += [f"## {h}: {ENUNCIADOS[h]}", "",
                       f"{h} incompleta: precisa do {REFERENCIA_ABLACOES} e de {', '.join(HIPOTESES[h])} completos nos dois datasets.", ""]
    linhas += linhas_descritivas(descritivas)
    (args.modelos / "resumo_dtsf.md").write_text("\n".join(linhas), encoding="utf-8")
    print((args.modelos / "resumo_dtsf.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
