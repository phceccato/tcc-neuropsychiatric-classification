"""Roda o D_artigo (o D com as dimensões do artigo original do DTSF) e o D_artigo_FA em todas as (repetições, folds).

Exploratório. O D da etapa 07 tem ~80 mil parâmetros; o artigo (WANG et al., 2026, Tabela 1) usa GRU de 128,
fusão de dimensão 128, Transformer de 2 blocos com 4 cabeças e convoluções de 64 filtros. O D_artigo é o D
com essas dimensões (702.721 parâmetros) e todo o resto igual: entradas, kernels, 16 passos por ramo, dropout,
Mixup, Adam, lr, lotes de 64, early stopping, partições e sementes de cada (repetição, fold). A única diferença
para o D é a capacidade, e a comparação com ele é pareada.

Escolhas fixadas antes de rodar, sem busca:
- filtros = 64: a Tabela 1 traz "Conv2d(64, …)" na coluna de parâmetros e 256 canais na coluna de formas; vale
  a coluna de parâmetros;
- o resto do treino é o do D, e não o do artigo (SGD, lotes de 128, 50 épocas), para isolar o tamanho.

D_artigo_FA: o D_artigo com a Frequency Attention (pipeline/e07_DTSF/redes_dtsf.py) depois do 1º bloco
temporal; com 64 filtros, são 4 grupos de 16, com os componentes DCT fixos (704.769 parâmetros). É o D_FA da
etapa 07. Começa com os mesmos pesos do D_artigo, e a comparação com ele isola o efeito da FA.

Grava outputs/14_DTSF_artigo/resultados/<dataset>/<modelo>/r<rep>_f<fold>.{csv,json}; com --permutacoes N,
o teste contra o acaso como na etapa 07, em outputs/14_DTSF_artigo/permutacoes/. Retoma: tarefas gravadas são
puladas. --dispositivo cuda treina na GPU; use o mesmo dispositivo do D da etapa 07.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

from pipeline.e07_DTSF.rodar import (SEMENTE, _fonte, _iniciar, argumento_dispositivo, conferir_dispositivo,
                                     contexto_processos, rotulos_permutados)
from pipeline.e07_DTSF.treino import ConfigTreino, prever, treinar
from pipeline.e07_DTSF.redes_dtsf import DTSF

MODELO = "D_artigo"
ARTIGO = ConfigTreino(filtros=64, oculto=128, cabecas=4, camadas=2)
MODELOS = ("D_artigo", "D_artigo_FA")


def construtor(modelo: str):
    """Função que cria a rede do `modelo` (None: o `DTSF` padrão com a configuração `ARTIGO`)."""
    if modelo == "D_artigo":
        return None
    if modelo == "D_artigo_FA":
        a = ARTIGO
        return lambda: DTSF("D_FA", dropout=a.dropout, dropout_encoder=a.dropout_encoder, filtros=a.filtros,
                              oculto=a.oculto, cabecas=a.cabecas, camadas=a.camadas)
    raise ValueError(f"modelo desconhecido: {modelo!r}")
SAIDA = Path("outputs/14_DTSF_artigo/resultados")
SAIDA_PERMUTACOES = Path("outputs/14_DTSF_artigo/permutacoes")


def rodar_tarefa(dataset: str, repeticao: int, fold: int, folds_csv: str, saida: str,
                 permutacao: int | None = None, dispositivo: str = "cpu", modelo: str = MODELO) -> str:
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == repeticao) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    if permutacao is not None:
        y = rotulos_permutados(y.sort_index(), permutacao)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    fonte = _fonte(dataset, folds)

    inicio = time.time()
    resultado = treinar("D", fonte, papel["treino"], papel["validacao"], y, semente=SEMENTE + 1000 * repeticao + fold,
                        config=ARTIGO, dispositivo=dispositivo, construir=construtor(modelo))
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
    os.replace(f"{base}.csv.tmp", f"{base}.csv")
    rotulo = f"r{repeticao}" if permutacao is None else f"p{permutacao}"
    return f"{dataset} {modelo} {rotulo} f{fold}: época {resultado.melhor_epoca}, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--modelos", nargs="+", default=list(MODELOS), choices=MODELOS)
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--permutacoes", type=int, default=0,
                        help="roda N permutações de rótulo (repetição 0) em vez dos treinos reais")
    argumento_dispositivo(parser)
    args = parser.parse_args()

    if args.permutacoes:
        saida = args.saida or SAIDA_PERMUTACOES
        tarefas = [(m, d, 0, f, k) for k in range(args.permutacoes) for m in args.modelos for d in args.datasets
                   for f in args.folds_ids if not (saida / d / m / f"p{k}_f{f}.csv").exists()]
    else:
        saida = args.saida or SAIDA
        tarefas = [(m, d, r, f, None) for m in args.modelos for d in args.datasets for r in args.repeticoes
                   for f in args.folds_ids if not (saida / d / m / f"r{r}_f{f}.csv").exists()]
    conferir_dispositivo(parser, args.dispositivo, [saida / d / m for m in args.modelos for d in args.datasets])
    print(f"{len(tarefas)} tarefas pendentes ({args.dispositivo})", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar, mp_context=contexto_processos(args.dispositivo)) as executor:
        futuros = [executor.submit(rodar_tarefa, d, r, f, args.folds, str(saida), k, args.dispositivo, m)
                   for m, d, r, f, k in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
