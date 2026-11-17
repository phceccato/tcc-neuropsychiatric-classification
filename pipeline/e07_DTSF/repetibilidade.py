"""Repetibilidade do treino num dispositivo: treina o mesmo fold duas vezes e compara as probabilidades (ME-05).

Serve para medir, na máquina da GPU, quanto duas execuções do mesmo (modelo, repetição, fold) diferem. O treino usa
`cudnn.deterministic`, mas não `torch.use_deterministic_algorithms`, e há operações de CUDA sem versão determinística.
Na CPU com 1 thread, a diferença esperada é 0 (docs/07_DTSF.md). Não grava nada em outputs/.

    venv/bin/python -m pipeline.e07_DTSF.repetibilidade --dispositivo cuda --dataset SZ --repeticao 0 --fold 0
"""
from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import pandas as pd

from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL, TIPOS_DTSF
from pipeline.e07_DTSF.rodar import rodar_tarefa


def comparar(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    """Diferença entre as probabilidades de duas execuções do mesmo fold (mesmas pessoas, mesma ordem de chave)."""
    chave = ["repeticao", "fold", "subject_uid"]
    a, b = a.sort_values(chave).reset_index(drop=True), b.sort_values(chave).reset_index(drop=True)
    if len(a) != len(b) or not (a[chave] == b[chave]).all().all():
        raise ValueError("as duas execuções não preveem as mesmas pessoas")
    dif = (a["prob"] - b["prob"]).abs()
    return dict(pessoas=len(a), max_abs=float(dif.max()), media_abs=float(dif.mean()), identicas=bool((dif == 0).all()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dispositivo", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dataset", choices=["SZ", "MDD"], default="SZ")
    parser.add_argument("--modelo", choices=TIPOS_DTSF, default=PRINCIPAL)
    parser.add_argument("--repeticao", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    args = parser.parse_args()

    execucoes = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in (1, 2):
            saida = Path(tmp) / f"execucao{i}"
            print(rodar_tarefa(args.dataset, args.modelo, args.repeticao, args.fold, args.folds, str(saida),
                               dispositivo=args.dispositivo), flush=True)
            execucoes.append(pd.read_csv(saida / args.dataset / args.modelo / f"r{args.repeticao}_f{args.fold}.csv"))
    r = comparar(*execucoes)
    print(f"{args.dispositivo} · {args.modelo} · {args.dataset} r{args.repeticao} f{args.fold}: {r['pessoas']} pessoas, "
          f"diferença máxima {r['max_abs']:.3g}, média {r['media_abs']:.3g}, "
          f"{'idênticas' if r['identicas'] else 'NÃO idênticas'}")


if __name__ == "__main__":
    main()
