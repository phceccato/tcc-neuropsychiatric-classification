"""Partições da validação cruzada, salvas antes de qualquer treino.

Segue o "Protocolo estatístico" de docs/00_proposta-tcc.md: StratifiedGroupKFold por pessoa,
k = 5, repetido 10 vezes, gerado separadamente em cada dataset; as mesmas partições servem
para o DTSF, os baselines clássicos e a LR de 76 bandas. Dentro do treino de cada fold, 20% das pessoas (estratificado)
formam o split interno do early stopping.

Lê outputs/02_harmonizacao/manifesto.csv e grava outputs/05_folds/folds.csv, com uma linha por
(dataset, repetição, fold, pessoa) e o papel da pessoa: treino, validacao ou teste.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

K, REPETICOES = 5, 10
FRACAO_VALIDACAO = 0.2
SEMENTE = 20261108


def gerar(pessoas: pd.DataFrame, repeticoes: int = REPETICOES, k: int = K,
          semente: int = SEMENTE) -> pd.DataFrame:
    """Papéis de cada pessoa em cada (repetição, fold). `pessoas` tem subject_uid e grupo."""
    pessoas = pessoas.reset_index(drop=True)
    linhas = []
    for r in range(repeticoes):
        divisor = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=semente + r)
        for fold, (treino, teste) in enumerate(divisor.split(pessoas, pessoas["grupo"], pessoas["subject_uid"])):
            n_validacao = round(FRACAO_VALIDACAO * len(treino))
            treino, validacao = train_test_split(treino, test_size=n_validacao,
                                                 stratify=pessoas.loc[treino, "grupo"],
                                                 random_state=semente + 1000 * r + fold)
            for papel, idx in (("treino", treino), ("validacao", validacao), ("teste", teste)):
                for i in idx:
                    linhas.append(dict(repeticao=r, fold=fold, subject_uid=pessoas.at[i, "subject_uid"],
                                       grupo=pessoas.at[i, "grupo"], papel=papel))
    return pd.DataFrame(linhas)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/05_folds"))
    args = parser.parse_args()

    manifesto = pd.read_csv(args.manifesto)
    tabelas = []
    for dataset, pessoas in manifesto.groupby("dataset", sort=False):
        particoes = gerar(pessoas[["subject_uid", "grupo"]])
        particoes.insert(0, "dataset", dataset)
        tabelas.append(particoes)
        teste = particoes[particoes["papel"] == "teste"].groupby(["repeticao", "fold"]).size()
        print(f"{dataset}: {pessoas['subject_uid'].nunique()} pessoas, teste com {teste.min()}–{teste.max()} pessoas por fold")

    args.saida.mkdir(parents=True, exist_ok=True)
    pd.concat(tabelas, ignore_index=True).to_csv(args.saida / "folds.csv", index=False)
    print(f"→ {args.saida / 'folds.csv'}")


if __name__ == "__main__":
    main()
