"""Roda os treinos do DTSF adaptado e das ablações em todas as (repetições, folds), em paralelo e com retomada.

Cada treino é uma tarefa (dataset, modelo, repetição, fold) que grava
outputs/07_DTSF/resultados/<dataset>/<modelo>/r<rep>_f<fold>.csv (previsões das pessoas de
teste) e .json (época escolhida e histórico de perdas). Tarefas já gravadas são puladas, então
o comando pode ser interrompido e rodado de novo. Cada processo usa 1 thread do PyTorch.

Modelos: D, D_sem_T, D_sem_E e D_sem_mixup (pipeline/e07_DTSF/redes_dtsf.py). A semente de
cada tarefa depende só de (repetição, fold): todos os modelos veem as mesmas partições e sementes.

Com --permutacoes N, roda o teste contra o acaso: para cada permutação k, os rótulos das pessoas
do dataset são embaralhados e o modelo é retreinado do zero nas partições da repetição 0; grava
outputs/07_DTSF/permutacoes/<dataset>/<modelo>/p<k>_f<fold>.csv.

Pré-requisitos: python -m pipeline.e07_DTSF.dados (cache do sinal a 125 Hz) e
python -m pipeline.e07_DTSF.dados_dtsf (cache multibanda).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.e07_DTSF.dados_dtsf import FonteDTSF
from pipeline.e07_DTSF.redes_dtsf import TIPOS_DTSF
from pipeline.e07_DTSF.treino import prever, treinar

SEMENTE = 20261108
SEMENTE_PERMUTACAO = 1_108_000
JANELAS_CSV = Path("outputs/03_janelas/janelas.csv")
CACHE_DTSF = Path("outputs/07_DTSF/cache_dtsf")
JANELAS = Path("outputs/03_janelas")

_FONTES: dict[str, FonteDTSF] = {}


def _fonte(dataset: str, folds: pd.DataFrame) -> FonteDTSF:
    if dataset not in _FONTES:
        uids = folds.loc[folds["dataset"] == dataset, "subject_uid"].unique()
        _FONTES[dataset] = FonteDTSF(uids, JANELAS_CSV, CACHE_DTSF, JANELAS)
    return _FONTES[dataset]


def rotulos_permutados(y: pd.Series, k: int) -> pd.Series:
    """Rótulos das pessoas embaralhados (permutação k), com as mesmas contagens por classe."""
    rng = np.random.default_rng(SEMENTE_PERMUTACAO + k)
    return pd.Series(rng.permutation(y.to_numpy()), index=y.index)


def _iniciar() -> None:
    torch.set_num_threads(1)


def rodar_tarefa(dataset: str, modelo: str, repeticao: int, fold: int, folds_csv: str, saida: str,
                 permutacao: int | None = None) -> str:
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == repeticao) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    if permutacao is not None:
        y = rotulos_permutados(y.sort_index(), permutacao)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    fonte = _fonte(dataset, folds)

    inicio = time.time()
    resultado = treinar(modelo, fonte, papel["treino"], papel["validacao"], y, semente=SEMENTE + 1000 * repeticao + fold)
    prob = prever(resultado.modelo, fonte, papel["teste"])

    destino = Path(saida) / dataset / modelo
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / (f"r{repeticao}_f{fold}" if permutacao is None else f"p{permutacao}_f{fold}")
    with open(f"{base}.json", "w") as f:
        json.dump(dict(melhor_epoca=resultado.melhor_epoca, segundos=time.time() - inicio,
                       historico=resultado.historico), f)
    pd.DataFrame([dict(repeticao=repeticao, fold=fold, subject_uid=u, y=int(y[u]), prob=p) for u, p in prob.items()]
                 ).to_csv(f"{base}.csv.tmp", index=False)
    os.replace(f"{base}.csv.tmp", f"{base}.csv")  # o .csv só aparece completo
    rotulo = f"r{repeticao}" if permutacao is None else f"p{permutacao}"
    return f"{dataset} {modelo} {rotulo} f{fold}: época {resultado.melhor_epoca}, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", default="outputs/07_DTSF/resultados")
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--modelos", nargs="+", default=["D"], choices=TIPOS_DTSF)
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--permutacoes", type=int, default=0,
                        help="roda N permutações de rótulo (repetição 0) em vez dos treinos reais")
    args = parser.parse_args()

    if args.permutacoes:
        saida = args.saida if args.saida != parser.get_default("saida") else "outputs/07_DTSF/permutacoes"
        tarefas = [(d, m, 0, f, k) for k in range(args.permutacoes) for m in args.modelos for d in args.datasets
                   for f in args.folds_ids if not (Path(saida) / d / m / f"p{k}_f{f}.csv").exists()]
        args.saida = saida
    else:
        tarefas = [(d, m, r, f, None) for m in args.modelos for d in args.datasets for r in args.repeticoes
                   for f in args.folds_ids if not (Path(args.saida) / d / m / f"r{r}_f{f}.csv").exists()]
    print(f"{len(tarefas)} tarefas pendentes", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar) as executor:
        futuros = [executor.submit(rodar_tarefa, d, m, r, f, args.folds, args.saida, k) for d, m, r, f, k in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
