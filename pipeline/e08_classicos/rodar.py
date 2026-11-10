"""Roda os baselines clássicos em todas as (repetições, folds), em paralelo e com retomada.

Mesma interface e mesmo formato de saída de pipeline/e07_DTSF/rodar.py, em outra pasta: cada tarefa
(dataset, modelo, repetição, fold) grava outputs/08_classicos/resultados/<dataset>/<modelo>/r<rep>_f<fold>.csv
(previsões das pessoas de teste) e .json (hiperparâmetros escolhidos, AUC interna de cada candidato,
partes internas e tempo). Tarefas já gravadas são puladas. Cada processo usa 1 thread.

A semente da tarefa é a do DTSF: SEMENTE + 1000 · repetição + fold.

Com --permutacoes N, os rótulos das pessoas do dataset são embaralhados (`rotulos_permutados`, o mesmo
de pipeline/e07_DTSF/rodar.py) e os modelos são retreinados nas partições da repetição 0; grava
outputs/08_classicos/permutacoes/<dataset>/<modelo>/p<k>_f<fold>.csv. Serve como teste contra o
acaso e como piloto sem ver resultados reais.

Pré-requisitos: as etapas 03 e 05, os caches da 07 (python -m pipeline.e07_DTSF.dados e
python -m pipeline.e07_DTSF.dados_dtsf) e python -m pipeline.e08_classicos.caracteristicas.
As permutações gravadas por --permutacoes ainda não são avaliadas por nenhum código.
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
import sklearn
from threadpoolctl import threadpool_limits

from pipeline.e08_classicos.caracteristicas import carregar
from pipeline.e08_classicos.treino import MODELOS, separar, treinar_e_prever
from pipeline.e07_DTSF.rodar import JANELAS_CSV, SEMENTE, rotulos_permutados

CARACTERISTICAS = Path("outputs/08_classicos/caracteristicas")
SAIDA = Path("outputs/08_classicos/resultados")
SAIDA_PERMUTACOES = Path("outputs/08_classicos/permutacoes")

_DADOS: dict = {}


def _dados(caracteristicas: str, janelas_csv: str) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    if (caracteristicas, janelas_csv) not in _DADOS:
        temporais, espectrais = carregar(Path(caracteristicas), Path(janelas_csv))
        indice = pd.read_csv(janelas_csv)
        indice["linha"] = np.arange(len(indice))
        combinado = np.concatenate([temporais, espectrais], axis=1)  # [linhas, 475 + 1102]
        _DADOS[(caracteristicas, janelas_csv)] = (dict(temporal=temporais, espectral=espectrais, combinado=combinado),
                                                  indice)
    return _DADOS[(caracteristicas, janelas_csv)]


def _iniciar() -> None:
    threadpool_limits(1)


def rodar_tarefa(dataset: str, modelo: str, repeticao: int, fold: int, folds_csv: str, saida: str,
                 permutacao: int | None = None, caracteristicas: str = str(CARACTERISTICAS),
                 janelas_csv: str = str(JANELAS_CSV)) -> str:
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == repeticao) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    if permutacao is not None:
        y = rotulos_permutados(y.sort_index(), permutacao)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    x, indice = _dados(caracteristicas, janelas_csv)
    indice = indice[indice["dataset"] == dataset]

    inicio = time.time()
    r = treinar_e_prever(modelo, x[separar(modelo)[1]], indice, papel["treino"] + papel["validacao"],
                         papel["teste"], y, semente=SEMENTE + 1000 * repeticao + fold)

    destino = Path(saida) / dataset / modelo
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / (f"r{repeticao}_f{fold}" if permutacao is None else f"p{permutacao}_f{fold}")
    with open(f"{base}.json", "w") as f:
        json.dump(dict(parametros=r.parametros, auc_interna=r.auc_interna, partes_internas=r.partes_internas,
                       janelas_ajuste=r.janelas_ajuste, segundos=time.time() - inicio,
                       sklearn=sklearn.__version__), f)
    pd.DataFrame([dict(repeticao=repeticao, fold=fold, subject_uid=u, y=int(y[u]), prob=p) for u, p in r.prob.items()]
                 ).to_csv(f"{base}.csv.tmp", index=False)
    os.replace(f"{base}.csv.tmp", f"{base}.csv")  # o .csv só aparece completo
    rotulo = f"r{repeticao}" if permutacao is None else f"p{permutacao}"
    return f"{dataset} {modelo} {rotulo} f{fold}: {r.parametros}, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", default=str(SAIDA))
    parser.add_argument("--caracteristicas", default=str(CARACTERISTICAS))
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=MODELOS)
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--permutacoes", type=int, default=0,
                        help="roda N permutações de rótulo (repetição 0) em vez dos treinos reais")
    args = parser.parse_args()

    if args.permutacoes:
        saida = args.saida if args.saida != parser.get_default("saida") else str(SAIDA_PERMUTACOES)
        tarefas = [(d, m, 0, f, k) for k in range(args.permutacoes) for m in args.modelos for d in args.datasets
                   for f in args.folds_ids if not (Path(saida) / d / m / f"p{k}_f{f}.csv").exists()]
        args.saida = saida
    else:
        tarefas = [(d, m, r, f, None) for m in args.modelos for d in args.datasets for r in args.repeticoes
                   for f in args.folds_ids if not (Path(args.saida) / d / m / f"r{r}_f{f}.csv").exists()]
    print(f"{len(tarefas)} tarefas pendentes", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar) as executor:
        futuros = [executor.submit(rodar_tarefa, d, m, r, f, args.folds, args.saida, k, args.caracteristicas)
                   for d, m, r, f, k in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
