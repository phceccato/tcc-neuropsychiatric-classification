"""Verificações das premissas metodológicas do teste pareado com ranks — não são testes de implementação.

Os testes de implementação (pareamento, ranks, Holm, etc.) estão em test_metricas.py e
tests/e07_DTSF/test_avaliar.py. O teste é usado na H4 (pipeline/e08_classicos/avaliar.py). Aqui, simulações pequenas e de semente fixa conferem as
propriedades estatísticas que o protocolo (docs/00_proposta-tcc.md) assume,
com pessoas independentes e sem treino:

- sob H0 (AUC igual), a permutação com ranks rejeita perto de α mesmo com calibrações diferentes;
- com escores brutos e calibrações diferentes, a permutação fica conservadora (o motivo dos ranks).

Elas não cobrem a dependência introduzida pelo treino na validação cruzada (erro tipo I de ~10–13%
para a pergunta sobre o método, registrado como limitação na proposta).
"""
import numpy as np
import pandas as pd

from pipeline.avaliacao import metricas as m
from pipeline.e07_DTSF import avaliar as a

REPLICAS, PERMUTACOES = 400, 500


def _taxa_de_rejeicao(usar_ranks: bool, comprimir_b: bool) -> float:
    rng = np.random.default_rng(2026)
    rejeicoes = 0
    for i in range(REPLICAS):
        y = np.r_[np.ones(14), np.zeros(14)].astype(int)
        z = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], 28)
        pa, pb = 1 / (1 + np.exp(-(1.2 * y + z[:, 0]))), 1 / (1 + np.exp(-(1.2 * y + z[:, 1])))
        if comprimir_b:
            pb = 0.4 + 0.2 * pb
        if usar_ranks:
            uids = [f"S{k:02d}" for k in range(28)]
            p = a.teste_ranks(pd.DataFrame(dict(subject_uid=uids, y=y, prob=pa)),
                     pd.DataFrame(dict(subject_uid=uids, y=y, prob=pb)), n=PERMUTACOES, semente=i)["p"]
        else:
            p = m.teste_pareado(y, pa, pb, n=PERMUTACOES, semente=i)[1]
        rejeicoes += p < 0.05
    return rejeicoes / REPLICAS


def test_ranks_mantem_o_erro_tipo_i_perto_de_alfa_com_calibracoes_diferentes():
    # erro de Monte Carlo com 400 réplicas: ~±0,02 em torno de 0,05
    assert 0.02 <= _taxa_de_rejeicao(usar_ranks=True, comprimir_b=True) <= 0.09


def test_escores_brutos_ficam_conservadores_com_calibracoes_diferentes():
    assert _taxa_de_rejeicao(usar_ranks=False, comprimir_b=True) < _taxa_de_rejeicao(usar_ranks=True, comprimir_b=True)
