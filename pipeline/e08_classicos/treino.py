"""Treino e previsão de um baseline clássico num fold externo, com seleção interna de hiperparâmetros.

Modelos: {LR, RF, KNN} × {temporal, espectral, combinado} (características de pipeline/e08_classicos/caracteristicas.py;
o combinado concatena as 475 temporais e as 1102 espectrais de cada janela, 1577 no total).

- Pessoas de ajuste: `treino` + `validacao` do fold em outputs/05_folds/folds.csv (o baseline clássico não
  tem early stopping, como pipeline/e06_LR). As pessoas de `teste` só entram na previsão final.
- Janelas de ajuste: até JANELAS_POR_PESSOA (150) janelas aceitas da grade com 50% de overlap de cada
  pessoa, sorteadas uma vez com `dados.sortear` e o gerador da semente da tarefa (determinístico e sem
  rótulos); quem tem menos aceitas entra com todas. O sorteio vale **só para o ajuste**: toda previsão
  (validação interna e teste) usa todas as janelas aceitas sem overlap da pessoa.
- Pesos (`pesos_janelas`): cada janela da pessoa i da classe c pesa N / (2 · P_c · n_i), com N janelas,
  P_c pessoas da classe e n_i janelas da pessoa. Cada pessoa pesa igual e as duas classes pesam igual, na
  proporção das pessoas; a média dos pesos é 1, para o C da LR manter a escala. Com o mesmo n_i
  para todos, é exatamente o `class_weight="balanced"`. O KNN não aceita `sample_weight` e é ajustado sem pesos:
  cada janela pesa igual (no MDD, de 83 a 150 janelas sorteadas por pessoa; na SZ, 150).
- Seleção: StratifiedKFold de 3 partes sobre as pessoas de ajuste (agrupado por pessoa: cada pessoa cai
  inteira numa parte). Cada candidato da grade é ajustado em 2 partes e prevê as pessoas da 3ª; a AUC
  interna é a AUC por pessoa das 3 partes juntas. Vence a maior AUC; no empate, o primeiro candidato da
  grade, que é sempre o mais regularizado. O vencedor é reajustado em todas as pessoas de ajuste.
- Normalização: z-score (StandardScaler) dentro do Pipeline, ajustado só nas janelas de quem está sendo
  ajustado (partes internas de treino ou pessoas de ajuste). O RF não tem normalização.
- KNN: z-score + distância euclidiana, voto uniforme; k = 101, 31 ou 11 (k maior = mais suave).
- Previsão da pessoa, sobre as suas janelas aceitas da grade sem overlap (`dados.janelas_avaliacao`):
  média das probabilidades das janelas, como no DTSF.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.e07_DTSF.dados import janelas_avaliacao, sortear

JANELAS_POR_PESSOA = 150
PARTES_INTERNAS = 3
ARVORES = 300
CLASSIFICADORES = ("LR", "RF", "KNN")
SEM_PESOS = {"KNN"}  # não aceitam sample_weight
MODALIDADES = ("temporal", "espectral", "combinado")
MODELOS = tuple(f"{c}_{m}" for m in MODALIDADES for c in CLASSIFICADORES)
# do mais regularizado para o menos: o empate fica com o primeiro
GRADES = {
    "LR": [dict(C=0.01), dict(C=0.1), dict(C=1.0)],
    "RF": [dict(min_samples_leaf=20), dict(min_samples_leaf=5), dict(min_samples_leaf=1)],
    "KNN": [dict(n_neighbors=101), dict(n_neighbors=31), dict(n_neighbors=11)],
}

def separar(modelo: str) -> tuple[str, str]:
    """'RF_espectral' → ('RF', 'espectral')."""
    classificador, modalidade = modelo.split("_")
    if classificador not in CLASSIFICADORES or modalidade not in MODALIDADES:
        raise ValueError(f"modelo desconhecido: {modelo!r}")
    return classificador, modalidade

def construir(classificador: str, parametros: dict, semente: int):
    if classificador == "LR":
        return Pipeline([("zscore", StandardScaler()),
                         ("clf", LogisticRegression(l1_ratio=0, max_iter=5000, **parametros))])
    if classificador == "RF":
        return RandomForestClassifier(n_estimators=ARVORES, max_features="sqrt", random_state=semente, n_jobs=1,
                                      **parametros)
    if classificador == "KNN":
        return Pipeline([("zscore", StandardScaler()), ("clf", KNeighborsClassifier(n_jobs=1, **parametros))])
    raise ValueError(f"classificador desconhecido: {classificador!r}")

def escore_pessoa(modelo, x: np.ndarray) -> float:
    """Escore de paciente de uma pessoa a partir das suas janelas `x`: média das probabilidades."""
    return float(modelo.predict_proba(x)[:, 1].mean())

def _matriz(x: np.ndarray, linhas: np.ndarray) -> np.ndarray:
    m = np.asarray(x[linhas], dtype=np.float64)
    if not np.isfinite(m).all():
        raise ValueError("característica não finita nas janelas pedidas (janela rejeitada?)")
    return m

def pesos_janelas(contagens: list[int], rotulos: list[int]) -> np.ndarray:
    """Peso de cada janela, na ordem das pessoas: N / (2 · P_c · n_i) (cada pessoa e cada classe pesam igual)."""
    contagens, rotulos = np.asarray(contagens), np.asarray(rotulos)
    pessoas_classe = {c: int((rotulos == c).sum()) for c in (0, 1)}
    if min(pessoas_classe.values()) == 0 or (contagens <= 0).any():
        raise ValueError("o ajuste precisa das duas classes e de janelas para cada pessoa")
    por_pessoa = contagens.sum() / (2 * np.array([pessoas_classe[c] for c in rotulos]) * contagens)
    return np.repeat(por_pessoa, contagens)

def _ajustar(classificador: str, parametros: dict, semente: int, x: np.ndarray, sorteadas: dict, uids, y: pd.Series):
    linhas = np.concatenate([sorteadas[u] for u in uids])
    alvo = np.concatenate([np.full(len(sorteadas[u]), int(y[u])) for u in uids])
    modelo = construir(classificador, parametros, semente)
    if classificador in SEM_PESOS:
        return modelo.fit(_matriz(x, linhas), alvo)
    pesos = pesos_janelas([len(sorteadas[u]) for u in uids], [int(y[u]) for u in uids])
    chave = "clf__sample_weight" if isinstance(modelo, Pipeline) else "sample_weight"
    return modelo.fit(_matriz(x, linhas), alvo, **{chave: pesos})

def prever_pessoas(modelo, x: np.ndarray, avaliacao: dict, uids) -> dict[str, float]:
    """Escore de cada pessoa sobre todas as suas janelas de avaliação (`escore_pessoa`)."""
    return {u: escore_pessoa(modelo, _matriz(x, avaliacao[u])) for u in uids}

@dataclass
class Resultado:
    prob: dict[str, float]
    parametros: dict
    auc_interna: list[float] = field(default_factory=list)
    partes_internas: list[list[str]] = field(default_factory=list)
    janelas_ajuste: int = 0

def partes_internas(uids: list[str], y: pd.Series, semente: int) -> list[tuple[list[str], list[str]]]:
    """Partições internas por pessoa: (ajuste, validação) com as duas classes em cada validação."""
    uids = list(uids)
    rotulos = np.array([int(y[u]) for u in uids])
    divisor = StratifiedKFold(PARTES_INTERNAS, shuffle=True, random_state=semente)
    partes = []
    for a, v in divisor.split(np.zeros(len(uids)), rotulos):
        if len(set(rotulos[v])) < 2 or len(set(rotulos[a])) < 2:
            raise ValueError("partição interna com uma classe só")
        partes.append(([uids[i] for i in a], [uids[i] for i in v]))
    return partes

def treinar_e_prever(modelo: str, x: np.ndarray, indice: pd.DataFrame, ajuste: list[str], teste: list[str],
                     y: pd.Series, semente: int) -> Resultado:
    """`x`: características por linha de janelas.csv; `indice`: linha, subject_uid, rejeitada, sem_overlap."""
    if set(ajuste) & set(teste):
        raise ValueError(f"pessoa no ajuste e no teste: {sorted(set(ajuste) & set(teste))}")
    classificador, _ = separar(modelo)
    grade = GRADES[classificador]
    por_pessoa = {u: g for u, g in indice.groupby("subject_uid")}
    rng = np.random.default_rng(semente)
    sorteadas = {u: sortear(por_pessoa[u], JANELAS_POR_PESSOA, rng) for u in ajuste}
    avaliacao = {u: janelas_avaliacao(por_pessoa[u]) for u in [*ajuste, *teste]}

    partes = partes_internas(ajuste, y, semente)
    aucs = []
    for parametros in grade:
        prob = {}
        for uids_ajuste, uids_validacao in partes:
            m = _ajustar(classificador, parametros, semente, x, sorteadas, uids_ajuste, y)
            prob.update(prever_pessoas(m, x, avaliacao, uids_validacao))
        aucs.append(float(roc_auc_score([int(y[u]) for u in prob], list(prob.values()))))
    melhor = grade[int(np.argmax(aucs))]

    final = _ajustar(classificador, melhor, semente, x, sorteadas, ajuste, y)
    return Resultado(prever_pessoas(final, x, avaliacao, teste), dict(melhor), aucs,
                     [sorted(v) for _, v in partes], int(sum(len(s) for s in sorteadas.values())))
