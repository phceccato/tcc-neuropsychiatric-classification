"""Roda os treinos dos modelos DTSF da etapa 07 em todas as (repetições, folds), em paralelo e com retomada.

Cada treino é uma tarefa (dataset, modelo, repetição, fold) que grava
outputs/07_DTSF/resultados/<dataset>/<modelo>/r<rep>_f<fold>.csv (previsões das pessoas de
teste) e .json (época escolhida, número de parâmetros e histórico de perdas). Tarefas já gravadas são puladas, então
o comando pode ser interrompido e rodado de novo. Cada processo usa 1 thread do PyTorch.

Modelos (pipeline/e07_DTSF/redes_dtsf.py): D_FA (o principal), D_pequeno, D, D_sem_mixup, D_sem_T e D_sem_E;
por padrão, os seis. A semente de cada tarefa depende só de (repetição, fold): todos os modelos veem as mesmas
partições e sementes.

Com --permutacoes N, roda o teste contra o acaso: para cada permutação k, os rótulos das pessoas
do dataset são embaralhados e o modelo é retreinado do zero nas partições da repetição 0; grava
outputs/07_DTSF/permutacoes/<dataset>/<modelo>/p<k>_f<fold>.csv. Nas permutações, o padrão é só o
modelo principal (a H1).

Com --dispositivo cuda, cada processo treina na GPU (pipeline/e07_DTSF/treino.py): os processos dividem a
mesma GPU e continuam preparando os lotes na CPU. O .json de cada tarefa registra o dispositivo, e o runner
se recusa a completar uma pasta que já tem tarefas feitas em outro dispositivo (use outra --saida).

Pré-requisitos: python -m pipeline.e07_DTSF.dados (cache do sinal a 125 Hz) e
python -m pipeline.e07_DTSF.dados_dtsf (cache multibanda).
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.e07_DTSF.dados_dtsf import FonteDTSF
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL, TIPOS_DTSF
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


def dispositivos_gravados(pastas: list[Path]) -> set[str]:
    """Dispositivos registrados nos .json já gravados (os antigos, sem o campo, foram feitos na CPU)."""
    return {json.load(open(a)).get("dispositivo", "cpu") for p in pastas for a in p.glob("*.json")}


def argumento_dispositivo(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cpu",
                        help="cpu (padrão, como as rodadas registradas) ou cuda (GPU dividida entre os processos)")


def conferir_dispositivo(parser: argparse.ArgumentParser, dispositivo: str, pastas: list[Path]) -> None:
    """Para com erro se pediram cuda sem GPU ou se as pastas já têm tarefas feitas em outro dispositivo."""
    if dispositivo == "cuda" and not torch.cuda.is_available():
        parser.error("--dispositivo cuda, mas o PyTorch não encontra GPU (torch.cuda.is_available() é False)")
    outros = dispositivos_gravados(pastas) - {dispositivo}
    if outros:
        parser.error(f"já há tarefas feitas em {sorted(outros)} em {', '.join(map(str, pastas))}; para rodar em "
                     f"{dispositivo}, use outra --saida ou apague as tarefas antigas (não misture CPU e GPU)")


def contexto_processos(dispositivo: str):
    """A GPU não sobrevive a fork: com cuda, os processos são criados com spawn."""
    return multiprocessing.get_context("spawn" if dispositivo == "cuda" else None)


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
    resultado = treinar(modelo, fonte, papel["treino"], papel["validacao"], y, semente=SEMENTE + 1000 * repeticao + fold,
                        dispositivo=dispositivo)
    prob = prever(resultado.modelo, fonte, papel["teste"])

    destino = Path(saida) / dataset / modelo
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / (f"r{repeticao}_f{fold}" if permutacao is None else f"p{permutacao}_f{fold}")
    with open(f"{base}.json", "w") as f:
        json.dump(dict(melhor_epoca=resultado.melhor_epoca, segundos=time.time() - inicio, dispositivo=dispositivo,
                       parametros=sum(p.numel() for p in resultado.modelo.parameters()),
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
    parser.add_argument("--modelos", nargs="+", default=None, choices=TIPOS_DTSF,
                        help=f"padrão: os seis modelos; com --permutacoes, só o principal ({PRINCIPAL})")
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--permutacoes", type=int, default=0,
                        help="roda N permutações de rótulo (repetição 0) em vez dos treinos reais")
    argumento_dispositivo(parser)
    args = parser.parse_args()
    if args.modelos is None:
        args.modelos = [PRINCIPAL] if args.permutacoes else list(TIPOS_DTSF)

    if args.permutacoes:
        saida = args.saida if args.saida != parser.get_default("saida") else "outputs/07_DTSF/permutacoes"
        tarefas = [(d, m, 0, f, k) for k in range(args.permutacoes) for m in args.modelos for d in args.datasets
                   for f in args.folds_ids if not (Path(saida) / d / m / f"p{k}_f{f}.csv").exists()]
        args.saida = saida
    else:
        tarefas = [(d, m, r, f, None) for m in args.modelos for d in args.datasets for r in args.repeticoes
                   for f in args.folds_ids if not (Path(args.saida) / d / m / f"r{r}_f{f}.csv").exists()]
    conferir_dispositivo(parser, args.dispositivo, [Path(args.saida) / d / m for m in args.modelos for d in args.datasets])
    print(f"{len(tarefas)} tarefas pendentes ({args.dispositivo})", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar, mp_context=contexto_processos(args.dispositivo)) as executor:
        futuros = [executor.submit(rodar_tarefa, d, m, r, f, args.folds, args.saida, k, args.dispositivo)
                   for d, m, r, f, k in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
