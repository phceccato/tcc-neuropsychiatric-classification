"""LR: regressão logística L2 (C = 1), com peso de classe, sobre a média das 76 features por pessoa.

Segue docs/00_proposta-tcc.md ("Baselines clássicos por modalidade", referência adicional, e "Protocolo estatístico"):

- features: log10 da potência absoluta por canal × banda (outputs/04_features/psd.npz), média
  das janelas aceitas de cada pessoa; as pessoas de treino usam a grade com 50% de overlap e
  as de teste a grade sem overlap, como no DTSF;
- z-score por feature com estatísticas só do treino de cada fold;
- mesmas partições de outputs/05_folds/folds.csv; a LR não tem early stopping, então
  treino e validacao entram juntos no ajuste;
- sem teste contra o acaso: a incerteza é o IC de 95% da AUC por pessoa (bootstrap) e a média ± desvio nas 10 repetições.

Checagem no MDD: o mesmo modelo com potência relativa (banda ÷ potência de 2–30 Hz). Ela só afasta o ganho
do aparelho: não elimina diferenças de ritmo entre sessões (pipeline/e06_LR_psd/checagem_lote.py).

Grava em outputs/06_LR_psd/ as previsões e resumo.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.avaliacao.metricas import avaliar, tabela_media_desvio
from pipeline.constantes import SEMENTE



def media_por_pessoa(x: np.ndarray, janelas: pd.DataFrame, so_sem_overlap: bool) -> pd.DataFrame:
    """Média das features das janelas aceitas de cada pessoa (linhas de x alinhadas a janelas)."""
    usar = ~janelas["rejeitada"].to_numpy()
    if so_sem_overlap:
        usar &= janelas["sem_overlap"].to_numpy()
    return pd.DataFrame(x[usar]).groupby(janelas["subject_uid"].to_numpy()[usar]).mean()


def _modelo():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, l1_ratio=0, class_weight="balanced", max_iter=1000))


def prever_cv(x_treino: pd.DataFrame, x_teste: pd.DataFrame, y: pd.Series, folds: pd.DataFrame) -> pd.DataFrame:
    """Probabilidade fora do fold de cada pessoa, em cada repetição de folds."""
    linhas = []
    for (r, fold), g in folds.groupby(["repeticao", "fold"]):
        treino = g.loc[g["papel"] != "teste", "subject_uid"].to_numpy()
        teste = g.loc[g["papel"] == "teste", "subject_uid"].to_numpy()
        modelo = _modelo().fit(x_treino.loc[treino].to_numpy(), y.loc[treino].to_numpy())
        prob = modelo.predict_proba(x_teste.loc[teste].to_numpy())[:, 1]
        linhas += [dict(repeticao=r, fold=fold, subject_uid=u, y=int(y[u]), prob=float(p))
                   for u, p in zip(teste, prob)]
    return pd.DataFrame(linhas)


def _formatar(nome: str, dataset: str, r: dict) -> str:
    return (f"| {dataset} | {nome} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
            f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {r['acuracia_balanceada']:.3f} | "
            f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("outputs/04_features/psd.npz"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/05_folds/folds.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/06_LR_psd"))
    args = parser.parse_args()

    features = np.load(args.features)
    janelas = pd.read_csv(args.janelas)
    if not (features["subject_uid"] == janelas["subject_uid"].to_numpy()).all():
        raise ValueError("psd.npz e janelas.csv não estão alinhados")
    folds = pd.read_csv(args.folds)
    args.saida.mkdir(parents=True, exist_ok=True)

    linhas = ["# LR_psd — resultados", "",
              "Regressão logística L2 (C = 1), com peso de classe, sobre a média das features por pessoa; z-score com estatísticas "
              "do treino de cada fold; 5 folds × 10 repetições. IC de 95% por bootstrap estratificado (2000 reamostragens). "
              "Sem teste contra o acaso.", "",
              "| Conjunto | Features | AUC por pessoa [IC 95%] | AUC por repetição | Acurácia balanceada | "
              "Sensibilidade | Especificidade | Macro F1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    predicoes = []
    for dataset, folds_ds in folds.groupby("dataset", sort=False):
        rotulos = folds_ds.drop_duplicates("subject_uid").set_index("subject_uid")["grupo"]
        y = (rotulos != "HC").astype(int)
        for nome in ("absolutas", "relativas"):
            x_treino = media_por_pessoa(features[nome], janelas, so_sem_overlap=False).loc[y.index]
            x_teste = media_por_pessoa(features[nome], janelas, so_sem_overlap=True).loc[y.index]
            pred = prever_cv(x_treino, x_teste, y, folds_ds)
            pred.insert(0, "features", nome)
            pred.insert(0, "dataset", dataset)
            predicoes.append(pred)
            resultado = avaliar(pred, semente=SEMENTE)
            linhas.append(_formatar(nome, dataset, resultado))
            print(f"{dataset} {nome}: AUC {resultado['auc']:.3f}")

    pd.concat(predicoes, ignore_index=True).to_csv(args.saida / "predicoes.csv", index=False)
    linhas += ["", "Média ± desvio nas 10 repetições: em cada repetição, as previsões fora do fold dos 5 folds juntas; "
               "limiar 0,5 nas secundárias. O desvio mostra a variação com a partição, não a incerteza sobre a "
               "população.", ""]
    linhas += tabela_media_desvio([([p["dataset"].iat[0], p["features"].iat[0]], p) for p in predicoes],
                                  ["Conjunto", "Features"])
    linhas += ["", "A potência relativa é a checagem do ganho do aparelho no MDD (docs/04_features.md). Ela não elimina "
               "diferenças de ritmo entre sessões de gravação: ver docs/06_LR_psd.md e checagem_lote.py."]
    (args.saida / "resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
