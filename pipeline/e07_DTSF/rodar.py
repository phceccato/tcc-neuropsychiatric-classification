"""Roda os treinos dos modelos DTSF da etapa 07 em todas as (repetições, folds), em paralelo e com retomada.

Cada treino é uma tarefa (dataset, modelo, repetição, fold) que grava
outputs/07_DTSF/resultados/<dataset>/<modelo>/r<rep>_f<fold>.csv (previsões das pessoas de
teste) e .json (época escolhida, dispositivo, número de parâmetros, configuração do treino e histórico de perdas). Tarefas já gravadas são puladas, então
o comando pode ser interrompido e rodado de novo. Cada processo usa 1 thread do PyTorch.

Modelos (pipeline/e07_DTSF/redes_dtsf.py): D_FA (o principal), D_pequeno, D, D_sem_mixup, D_sem_T e D_sem_E;
por padrão, os seis. A semente de cada tarefa depende só de (repetição, fold): todos os modelos veem as mesmas
partições e sementes.

Checkpoints: o D_FA (o principal; --checkpoints todos para os seis) grava os pesos do melhor
modelo de cada fold em outputs/07_DTSF/checkpoints/<dataset>/<modelo>/r<rep>_f<fold>.pt (treino.salvar_checkpoint).
O .json guarda o histórico de perdas de treino e de validação interna de todas as épocas executadas.

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
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.constantes import SEMENTE
from pipeline.e07_DTSF.dados_dtsf import FonteDTSF
from pipeline.e07_DTSF.redes_dtsf import MODELOS, PRINCIPAL, TIPOS_DTSF
from pipeline.e07_DTSF.treino import JANELAS_POR_PESSOA, PADRAO, prever, salvar_checkpoint, treinar

JANELAS_CSV = Path("outputs/03_janelas/janelas.csv")
CACHE_DTSF = Path("outputs/07_DTSF/cache_dtsf")
CHECKPOINTS = Path("outputs/07_DTSF/checkpoints")
JANELAS = Path("outputs/03_janelas")

_FONTES: dict[str, FonteDTSF] = {}


def _fonte(dataset: str, folds: pd.DataFrame) -> FonteDTSF:
    if dataset not in _FONTES:
        uids = folds.loc[folds["dataset"] == dataset, "subject_uid"].unique()
        _FONTES[dataset] = FonteDTSF(uids, JANELAS_CSV, CACHE_DTSF, JANELAS)
    return _FONTES[dataset]


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


def configuracao_tarefa(modelo: str, semente: int, dispositivo: str) -> dict:
    """O que reproduz o treino de uma tarefa, gravado no .json: modelo, dispositivo, semente, a variante
    (ramos, Mixup, FA e dimensões) e os hiperparâmetros de treino (`ConfigTreino`), mais as versões do PyTorch e da CUDA."""
    return dict(modelo=modelo, dispositivo=dispositivo, semente=semente, variante=asdict(MODELOS[modelo]),
                hiperparametros=asdict(PADRAO), janelas_por_pessoa=JANELAS_POR_PESSOA,
                torch=str(torch.__version__), cuda=torch.version.cuda)


def gravar_json(caminho: Path, conteudo: dict) -> None:
    """Grava o .json inteiro ou não grava: um treino interrompido não deixa um histórico de perdas truncado."""
    temporario = Path(f"{caminho}.tmp")
    with open(temporario, "w") as f:
        json.dump(conteudo, f)
    os.replace(temporario, caminho)


def modelos_com_checkpoint(opcao: str, modelos: list[str]) -> set[str]:
    """`principal` (o padrão: só o D_FA), `todos` ou `nenhum`, entre os modelos pedidos."""
    return {"principal": {PRINCIPAL}, "todos": set(TIPOS_DTSF), "nenhum": set()}[opcao] & set(modelos)


def caminho_checkpoint(pasta: Path | str, dataset: str, modelo: str, repeticao: int, fold: int) -> Path:
    return Path(pasta) / dataset / modelo / f"r{repeticao}_f{fold}.pt"


def rodar_tarefa(dataset: str, modelo: str, repeticao: int, fold: int, folds_csv: str, saida: str,
                 dispositivo: str = "cpu", pasta_checkpoints: str | None = None) -> str:
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == repeticao) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    fonte = _fonte(dataset, folds)

    inicio = time.time()
    semente = SEMENTE + 1000 * repeticao + fold
    resultado = treinar(modelo, fonte, papel["treino"], papel["validacao"], y, semente=semente, dispositivo=dispositivo)
    prob = prever(resultado.modelo, fonte, papel["teste"])

    destino = Path(saida) / dataset / modelo
    destino.mkdir(parents=True, exist_ok=True)
    base = destino / f"r{repeticao}_f{fold}"
    config = configuracao_tarefa(modelo, semente, dispositivo)
    historico = resultado.historico
    gravar_json(Path(f"{base}.json"), dict(
        melhor_epoca=resultado.melhor_epoca, epocas_executadas=len(historico),
        perda_validacao_melhor=historico[resultado.melhor_epoca - 1]["perda_validacao"],
        segundos=time.time() - inicio,
        parametros=sum(p.numel() for p in resultado.modelo.parameters()), **config, historico=historico))
    if pasta_checkpoints is not None:
        salvar_checkpoint(caminho_checkpoint(pasta_checkpoints, dataset, modelo, repeticao, fold), resultado.modelo,
                          resultado.melhor_epoca, semente=semente, repeticao=repeticao, fold=fold, dataset=dataset,
                          configuracao=config)
    pd.DataFrame([dict(repeticao=repeticao, fold=fold, subject_uid=u, y=int(y[u]), prob=p) for u, p in prob.items()]
                 ).to_csv(f"{base}.csv.tmp", index=False)
    os.replace(f"{base}.csv.tmp", f"{base}.csv")  # o .csv só aparece completo
    return f"{dataset} {modelo} r{repeticao} f{fold}: época {resultado.melhor_epoca}, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", default="outputs/07_DTSF/resultados")
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--modelos", nargs="+", default=None, choices=TIPOS_DTSF,
                        help="padrão: os seis modelos")
    parser.add_argument("--repeticoes", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--folds-ids", type=int, nargs="+", default=list(range(5)))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--checkpoints", choices=["principal", "todos", "nenhum"], default="principal",
                        help="quais modelos gravam os pesos do melhor fold em --saida-checkpoints (padrão: só o D_FA)")
    parser.add_argument("--saida-checkpoints", default=str(CHECKPOINTS))
    argumento_dispositivo(parser)
    args = parser.parse_args()
    if args.modelos is None:
        args.modelos = list(TIPOS_DTSF)

    com_checkpoint = modelos_com_checkpoint(args.checkpoints, args.modelos)
    # um treino só conta como feito se o .csv existe e, nos modelos com checkpoint, o .pt também
    tarefas = [(d, m, r, f) for m in args.modelos for d in args.datasets for r in args.repeticoes
               for f in args.folds_ids
               if not (Path(args.saida) / d / m / f"r{r}_f{f}.csv").exists()
               or (m in com_checkpoint and not caminho_checkpoint(args.saida_checkpoints, d, m, r, f).exists())]
    conferir_dispositivo(parser, args.dispositivo, [Path(args.saida) / d / m for m in args.modelos for d in args.datasets])
    print(f"{len(tarefas)} tarefas pendentes ({args.dispositivo})", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar, mp_context=contexto_processos(args.dispositivo)) as executor:
        futuros = [executor.submit(rodar_tarefa, d, m, r, f, args.folds, args.saida, args.dispositivo,
                                   args.saida_checkpoints if m in com_checkpoint else None)
                   for d, m, r, f in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
