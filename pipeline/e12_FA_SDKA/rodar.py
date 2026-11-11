"""Roda os treinos das variantes da etapa 12 em todas as (repetições, folds), em paralelo e com retomada.

Cada treino é uma tarefa (dataset, modelo, repetição, fold) que grava
outputs/12_FA_SDKA/resultados/<dataset>/<modelo>/r<rep>_f<fold>.csv (previsões das pessoas de teste) e
.json (época escolhida e histórico de perdas, com o SMMD médio de cada época). Tarefas já gravadas são
puladas. As partições e as sementes de cada (repetição, fold) são as do D (pipeline/e07_DTSF/rodar.py),
então a comparação com o D é pareada.

Com --permutacoes N, roda o teste contra o acaso como na etapa 07: rótulos das pessoas embaralhados
(as mesmas permutações da etapa 07) e retreino nas partições da repetição 0; grava
outputs/12_FA_SDKA/permutacoes/<dataset>/<modelo>/p<k>_f<fold>.csv.

Com --dispositivo cuda, os treinos rodam na GPU, como na etapa 07. Use o mesmo dispositivo do D da etapa 07,
com que as variantes são comparadas; o runner se recusa a completar uma pasta feita em outro dispositivo.

Pré-requisitos: os caches da etapa 07 (python -m pipeline.e07_DTSF.dados e
python -m pipeline.e07_DTSF.dados_dtsf).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import torch

from pipeline.e07_DTSF.rodar import (SEMENTE, _fonte, argumento_dispositivo, conferir_dispositivo, contexto_processos,
                                     rotulos_permutados)
from pipeline.e12_FA_SDKA.redes import TIPOS_12
from pipeline.e12_FA_SDKA.treino import prever, treinar

SAIDA = Path("outputs/12_FA_SDKA/resultados")
SAIDA_PERMUTACOES = Path("outputs/12_FA_SDKA/permutacoes")


def _iniciar() -> None:
    torch.set_num_threads(1)


def rodar_tarefa(dataset: str, modelo: str, repeticao: int, fold: int, folds_csv: str, saida: str,
                 permutacao: int | None = None, dispositivo: str = "cpu") -> str:
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == repeticao) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    if permutacao is not None:
        y = rotulos_permutados(y.sort_index(), permutacao)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    fonte = _fonte(dataset, folds)

    inicio = time.time()
    resultado = treinar(modelo, fonte, papel["treino"], papel["validacao"], papel["teste"], y,
                        semente=SEMENTE + 1000 * repeticao + fold, dispositivo=dispositivo)
    prob = prever(resultado.modelo, fonte, papel["teste"])

    destino = Path(saida) / dataset / modelo
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / (f"r{repeticao}_f{fold}" if permutacao is None else f"p{permutacao}_f{fold}")
    with open(f"{base}.json", "w") as f:
        json.dump(dict(melhor_epoca=resultado.melhor_epoca, segundos=time.time() - inicio, dispositivo=dispositivo,
                       historico=resultado.historico), f)
    pd.DataFrame([dict(repeticao=repeticao, fold=fold, subject_uid=u, y=int(y[u]), prob=p) for u, p in prob.items()]
                 ).to_csv(f"{base}.csv.tmp", index=False)
    os.replace(f"{base}.csv.tmp", f"{base}.csv")
    rotulo = f"r{repeticao}" if permutacao is None else f"p{permutacao}"
    return f"{dataset} {modelo} {rotulo} f{fold}: época {resultado.melhor_epoca}, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--modelos", nargs="+", default=list(TIPOS_12), choices=TIPOS_12)
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--permutacoes", type=int, default=0,
                        help="roda N permutações de rótulo (repetição 0) em vez dos treinos reais")
    argumento_dispositivo(parser)
    args = parser.parse_args()

    if args.permutacoes:
        saida = args.saida or SAIDA_PERMUTACOES
        tarefas = [(d, m, 0, f, k) for k in range(args.permutacoes) for m in args.modelos for d in args.datasets
                   for f in args.folds_ids if not (saida / d / m / f"p{k}_f{f}.csv").exists()]
    else:
        saida = args.saida or SAIDA
        tarefas = [(d, m, r, f, None) for m in args.modelos for d in args.datasets for r in args.repeticoes
                   for f in args.folds_ids if not (saida / d / m / f"r{r}_f{f}.csv").exists()]
    conferir_dispositivo(parser, args.dispositivo, [saida / d / m for m in args.modelos for d in args.datasets])
    print(f"{len(tarefas)} tarefas pendentes ({args.dispositivo})", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar, mp_context=contexto_processos(args.dispositivo)) as executor:
        futuros = [executor.submit(rodar_tarefa, d, m, r, f, args.folds, str(saida), k, args.dispositivo)
                   for d, m, r, f, k in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
