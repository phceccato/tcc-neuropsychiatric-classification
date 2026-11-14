"""Métricas e testes do protocolo estatístico (docs/00_proposta-tcc.md, "Protocolo estatístico").

Comum a todos os modelos (LR de 76 bandas, DTSF e baselines clássicos). As previsões chegam como uma tabela com uma
linha por (repetição, pessoa) e as colunas repeticao, subject_uid, y (1 = paciente) e prob.

- métrica primária: ROC-AUC por pessoa, sobre a média da probabilidade fora do fold nas
  repetições;
- incerteza: IC de 95% por bootstrap estratificado por pessoa e AUC de cada repetição;
- média ± desvio nas repetições (`metricas_por_repeticao`, `tabela_media_desvio`): em cada repetição, as 5
  previsões fora do fold juntas dão AUC e secundárias; reporta-se a média e o desvio (ddof = 1) das 10. Mostra
  quanto o resultado muda com a partição, não a incerteza sobre a população (as repetições reusam as pessoas);
- secundárias (descritivas), no limiar 0,5 e com o paciente como classe positiva: acurácia, acurácia
  balanceada, precisão, sensibilidade (recall), especificidade, F1 e Macro F1;
- permutação: p = (nº de nulas ≥ observado + 1) / (N + 1);
- `teste_pareado`: ΔAUC entre dois modelos sobre os escores pareados de cada pessoa, com
  permutação pareada (troca dos dois escores de cada pessoa), unilateral; as hipóteses H2–H4
  o aplicam sobre ranks (`teste_ranks`, em pipeline/e07_DTSF/avaliar.py).

Extensão DTSF:
- `parear`: alinha os escores por pessoa de dois modelos e recusa pessoas ausentes, repetidas,
  rótulos divergentes ou escores inválidos;
- `p_interseccao_uniao`: p = maior p das comparações componentes (H2 e H4);
- `holm`: correção de Holm sobre {SZ, MDD} de cada hipótese (H2–H4);
- sensibilidades das comparações pareadas (secundárias, não mudam a decisão): ΔAUC por
  repetição e t corrigido de Nadeau e Bengio (2003) sobre os ΔAUC por fold.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import t as dist_t
from sklearn.metrics import f1_score, precision_score, roc_auc_score

LIMIAR = 0.5
REAMOSTRAGENS = 2000
PERMUTACOES_PAREADAS = 10_000


def escores_por_pessoa(predicoes: pd.DataFrame) -> pd.DataFrame:
    """Média da probabilidade de cada pessoa nas repetições."""
    return predicoes.groupby("subject_uid", as_index=False).agg(y=("y", "first"), prob=("prob", "mean"))


def auc_por_repeticao(predicoes: pd.DataFrame) -> np.ndarray:
    return np.array([roc_auc_score(g["y"], g["prob"]) for _, g in predicoes.groupby("repeticao")])


def ic_bootstrap(y: np.ndarray, escore: np.ndarray, reamostragens: int = REAMOSTRAGENS,
                 semente: int = 0) -> tuple[float, float]:
    """IC de 95% (percentil) da AUC, reamostrando pessoas dentro de cada classe."""
    y, escore = np.asarray(y), np.asarray(escore)
    rng = np.random.default_rng(semente)
    pacientes, controles = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    aucs = np.empty(reamostragens)
    for i in range(reamostragens):
        idx = np.concatenate([rng.choice(pacientes, len(pacientes)), rng.choice(controles, len(controles))])
        aucs[i] = roc_auc_score(y[idx], escore[idx])
    baixo, alto = np.percentile(aucs, [2.5, 97.5])
    return float(baixo), float(alto)


def metricas_secundarias(y: np.ndarray, prob: np.ndarray) -> dict[str, float]:
    y, previsto = np.asarray(y), (np.asarray(prob) >= LIMIAR).astype(int)
    sensibilidade = float(np.mean(previsto[y == 1] == 1))
    especificidade = float(np.mean(previsto[y == 0] == 0))
    return dict(acuracia=float(np.mean(previsto == y)), acuracia_balanceada=(sensibilidade + especificidade) / 2,
                precisao=float(precision_score(y, previsto, zero_division=0)), sensibilidade=sensibilidade,
                especificidade=especificidade, f1=float(f1_score(y, previsto, zero_division=0)),
                macro_f1=float(f1_score(y, previsto, average="macro")))


METRICAS_REPETICAO = {"auc": "AUC", "acuracia": "Acurácia", "acuracia_balanceada": "Acurácia balanceada",
                      "precisao": "Precisão", "sensibilidade": "Sensibilidade (recall)",
                      "especificidade": "Especificidade", "f1": "F1", "macro_f1": "Macro F1"}


def metricas_por_repeticao(predicoes: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por repetição: AUC e secundárias (limiar 0,5) sobre as previsões fora do fold dela."""
    linhas = []
    for r, g in predicoes.groupby("repeticao"):
        linhas.append(dict(repeticao=r, auc=roc_auc_score(g["y"], g["prob"]), **metricas_secundarias(g["y"], g["prob"])))
    return pd.DataFrame(linhas).set_index("repeticao")


def media_desvio(predicoes: pd.DataFrame) -> dict[str, tuple[float, float]]:
    """{métrica: (média, desvio)} nas repetições; desvio com ddof = 1 (0 com uma repetição só)."""
    t = metricas_por_repeticao(predicoes)
    return {m: (float(t[m].mean()), float(t[m].std(ddof=1)) if len(t) > 1 else 0.0) for m in METRICAS_REPETICAO}


def tabela_media_desvio(linhas: list[tuple[list[str], pd.DataFrame | None]], rotulos: list[str]) -> list[str]:
    """Tabela markdown com média ± desvio das métricas por repetição; `None` marca um modelo incompleto."""
    saida = ["| " + " | ".join([*rotulos, *METRICAS_REPETICAO.values()]) + " |",
             "| " + " | ".join(["---"] * (len(rotulos) + len(METRICAS_REPETICAO))) + " |"]
    for chave, predicoes in linhas:
        if predicoes is None:
            saida.append("| " + " | ".join([*chave, "incompleto", *[""] * (len(METRICAS_REPETICAO) - 1)]) + " |")
            continue
        md = media_desvio(predicoes)
        saida.append("| " + " | ".join([*chave, *(f"{md[m][0]:.3f} ± {md[m][1]:.3f}" for m in METRICAS_REPETICAO)]) + " |")
    return saida


def p_valor(observado: float, nulas: np.ndarray) -> float:
    nulas = np.asarray(nulas)
    return float((np.sum(nulas >= observado) + 1) / (len(nulas) + 1))


def avaliar(predicoes: pd.DataFrame, semente: int = 0) -> dict[str, float]:
    """Resumo de um modelo num dataset: AUC por pessoa com IC, AUC por repetição e secundárias."""
    escores = escores_por_pessoa(predicoes)
    auc = roc_auc_score(escores["y"], escores["prob"])
    baixo, alto = ic_bootstrap(escores["y"].to_numpy(), escores["prob"].to_numpy(), semente=semente)
    por_repeticao = auc_por_repeticao(predicoes)
    return dict(auc=auc, ic_baixo=baixo, ic_alto=alto, auc_rep_media=float(por_repeticao.mean()),
                auc_rep_desvio=float(por_repeticao.std(ddof=1)) if len(por_repeticao) > 1 else 0.0,
                **metricas_secundarias(escores["y"], escores["prob"]))


def auc_em_lote(y: np.ndarray, escores: np.ndarray) -> np.ndarray:
    """AUC de cada linha de `escores` [n_conjuntos, n_pessoas] (Mann-Whitney, empates valem 1/2)."""
    y = np.asarray(y).astype(bool)
    pos, neg = escores[:, y][:, :, None], escores[:, ~y][:, None, :]
    return ((pos > neg).sum(axis=(1, 2)) + 0.5 * (pos == neg).sum(axis=(1, 2))) / (y.sum() * (~y).sum())


def teste_pareado(y: np.ndarray, escore_a: np.ndarray, escore_b: np.ndarray,
                  n: int = PERMUTACOES_PAREADAS, semente: int = 0) -> tuple[float, float]:
    """ΔAUC = AUC(a) − AUC(b) e o p unilateral (a > b) da permutação pareada."""
    a, b = np.asarray(escore_a, dtype=float), np.asarray(escore_b, dtype=float)
    delta = float(auc_em_lote(y, a[None])[0] - auc_em_lote(y, b[None])[0])
    troca = np.random.default_rng(semente).random((n, len(a))) < 0.5
    nulas = auc_em_lote(y, np.where(troca, b, a)) - auc_em_lote(y, np.where(troca, a, b))
    return delta, p_valor(delta - 1e-12, nulas)


def ic_bootstrap_delta(y: np.ndarray, escore_a: np.ndarray, escore_b: np.ndarray,
                       reamostragens: int = REAMOSTRAGENS, semente: int = 0) -> tuple[float, float]:
    """IC de 95% do ΔAUC (a − b), reamostrando pessoas dentro de cada classe (pares mantidos)."""
    y = np.asarray(y)
    a, b = np.asarray(escore_a, dtype=float), np.asarray(escore_b, dtype=float)
    rng = np.random.default_rng(semente)
    pacientes, controles = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    idx = np.concatenate([rng.choice(pacientes, (reamostragens, len(pacientes))),
                          rng.choice(controles, (reamostragens, len(controles)))], axis=1)
    y_boot = np.concatenate([np.ones(len(pacientes)), np.zeros(len(controles))])
    deltas = auc_em_lote(y_boot, a[idx]) - auc_em_lote(y_boot, b[idx])
    baixo, alto = np.percentile(deltas, [2.5, 97.5])
    return float(baixo), float(alto)


def parear(escores_a: pd.DataFrame, escores_b: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(y, escore_a, escore_b) das mesmas pessoas, em ordem de subject_uid (saída de `escores_por_pessoa`)."""
    a, b = (e.set_index("subject_uid") for e in (escores_a, escores_b))
    if not a.index.is_unique or not b.index.is_unique:
        raise ValueError("pessoa repetida nos escores")
    if set(a.index) != set(b.index):
        raise ValueError(f"pessoas diferentes entre os modelos: {sorted(set(a.index) ^ set(b.index))}")
    a, b = a.sort_index(), b.loc[a.sort_index().index]
    if not np.array_equal(a["y"].to_numpy(), b["y"].to_numpy()):
        raise ValueError("rótulo diferente para a mesma pessoa")
    y, ea, eb = a["y"].to_numpy().astype(int), a["prob"].to_numpy(float), b["prob"].to_numpy(float)
    if not (np.isfinite(ea).all() and np.isfinite(eb).all()):
        raise ValueError("escore ausente ou não finito")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError("os escores precisam ter pacientes (y = 1) e controles (y = 0)")
    return y, ea, eb


def p_interseccao_uniao(ps: list[float]) -> float:
    """p do teste de interseção-união: rejeita só se rejeitar todas as componentes (BERGER, 1982)."""
    if not ps:
        raise ValueError("sem comparações")
    return float(max(ps))


def holm(ps: dict[str, float], alfa: float = 0.05) -> dict[str, dict]:
    """Holm (step-down) sobre a família de p: p ajustado = max_{j ≤ i} min(1, (m − j + 1)·p₍ⱼ₎)."""
    if not ps or not all(np.isfinite(p) and 0 <= p <= 1 for p in ps.values()):
        raise ValueError(f"p inválido: {ps}")
    ordem = sorted(ps, key=ps.get)
    resultado, ajustado, rejeitando = {}, 0.0, True
    for i, nome in enumerate(ordem):
        ajustado = max(ajustado, min(1.0, (len(ps) - i) * ps[nome]))
        rejeitando = rejeitando and ajustado <= alfa
        resultado[nome] = dict(p=ps[nome], p_ajustado=ajustado, rejeita=rejeitando)
    return resultado


def _juntar(predicoes_a: pd.DataFrame, predicoes_b: pd.DataFrame) -> pd.DataFrame:
    """Previsões dos dois modelos lado a lado, uma linha por (repetição, pessoa)."""
    juntas = predicoes_a.merge(predicoes_b, on=["repeticao", "subject_uid"], how="outer", suffixes=("_a", "_b"),
                               indicator=True, validate="one_to_one")
    if (juntas["_merge"] != "both").any() or not (juntas["y_a"] == juntas["y_b"]).all():
        raise ValueError("as previsões dos dois modelos não cobrem as mesmas (repetição, pessoa) com o mesmo rótulo")
    if "fold_a" in juntas and not (juntas["fold_a"] == juntas["fold_b"]).all():
        raise ValueError("a mesma (repetição, pessoa) está em folds diferentes nos dois modelos")
    return juntas.rename(columns={"y_a": "y", "fold_a": "fold"})


def _deltas(juntas: pd.DataFrame, chaves: list[str]) -> pd.DataFrame:
    if (juntas.groupby(chaves)["y"].nunique() < 2).any():
        raise ValueError(f"há grupos ({', '.join(chaves)}) com uma classe só: AUC indefinida")
    return pd.DataFrame([dict(zip(chaves, np.atleast_1d(chave)), n_teste=len(g),
                              delta=roc_auc_score(g["y"], g["prob_a"]) - roc_auc_score(g["y"], g["prob_b"]))
                         for chave, g in juntas.groupby(chaves)])


def delta_por_repeticao(predicoes_a: pd.DataFrame, predicoes_b: pd.DataFrame) -> np.ndarray:
    """ΔAUC (a − b) de cada repetição, sobre as previsões fora do fold de todas as pessoas."""
    return _deltas(_juntar(predicoes_a, predicoes_b), ["repeticao"])["delta"].to_numpy()


def delta_por_fold(predicoes_a: pd.DataFrame, predicoes_b: pd.DataFrame) -> pd.DataFrame:
    """ΔAUC (a − b) em cada (repetição, fold), com o número de pessoas de teste do fold."""
    return _deltas(_juntar(predicoes_a, predicoes_b), ["repeticao", "fold"])


def t_corrigido(deltas: np.ndarray, razao_teste_treino: float) -> tuple[float, float, float]:
    """Média, t e p unilateral (Δ > 0) do t corrigido de Nadeau e Bengio (2003).

    A variância dos J deltas é multiplicada por (1/J + n_teste/n_treino); J − 1 graus de liberdade.
    """
    deltas = np.asarray(deltas, dtype=float)
    if len(deltas) < 2:
        raise ValueError("o t corrigido precisa de pelo menos 2 deltas")
    media, variancia = float(deltas.mean()), float(deltas.var(ddof=1))
    if variancia == 0:
        return media, 0.0, 1.0 if media <= 0 else 0.0
    t = media / np.sqrt((1 / len(deltas) + razao_teste_treino) * variancia)
    return media, float(t), float(dist_t.sf(t, len(deltas) - 1))
