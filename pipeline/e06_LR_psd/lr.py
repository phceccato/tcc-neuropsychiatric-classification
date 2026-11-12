"""LR: regressão logística L2 (C = 1), com peso de classe, sobre a média das 76 features por pessoa.

Segue docs/00_proposta-tcc.md ("Baselines clássicos por modalidade", referência adicional, e "Protocolo estatístico"):

- features: log10 da potência absoluta por canal × banda (outputs/04_features/psd.npz), média
  das janelas aceitas de cada pessoa; as pessoas de treino usam a grade com 50% de overlap e
  as de teste a grade sem overlap, como no DTSF;
- z-score por feature com estatísticas só do treino de cada fold;
- mesmas partições de outputs/05_folds/folds.csv; a LR não tem early stopping, então
  treino e validacao entram juntos no ajuste;
- teste contra o acaso: 1000 permutações de rótulo com retreino, na repetição 0.

Checagem no MDD: o mesmo modelo com potência relativa (banda ÷ potência de 2–30 Hz).

Grava em outputs/06_LR_psd/ as previsões, as AUCs nulas e resumo.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline.avaliacao.metricas import avaliar, p_valor, roc_auc_score

PERMUTACOES = 1000
SEMENTE = 20261108


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


def permutacao(x_treino: pd.DataFrame, x_teste: pd.DataFrame, y: pd.Series, folds: pd.DataFrame,
               n: int = PERMUTACOES, semente: int = SEMENTE) -> np.ndarray:
    """AUCs com os rótulos das pessoas embaralhados e retreino completo nas mesmas partições."""
    rng = np.random.default_rng(semente)
    nulas = np.empty(n)
    for i in range(n):
        y_perm = pd.Series(rng.permutation(y.to_numpy()), index=y.index)
        pred = prever_cv(x_treino, x_teste, y_perm, folds)
        nulas[i] = roc_auc_score(pred["y"], pred["prob"])
    return nulas


def _formatar(nome: str, dataset: str, r: dict, p: float | None) -> str:
    p_txt = "—" if p is None else f"{p:.3f}"
    return (f"| {dataset} | {nome} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
            f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} | {p_txt} | {r['acuracia_balanceada']:.3f} | "
            f"{r['sensibilidade']:.3f} | {r['especificidade']:.3f} | {r['macro_f1']:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("outputs/04_features/psd.npz"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/05_folds/folds.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/06_LR_psd"))
    parser.add_argument("--permutacoes", type=int, default=PERMUTACOES)
    args = parser.parse_args()

    features = np.load(args.features)
    janelas = pd.read_csv(args.janelas)
    if not (features["subject_uid"] == janelas["subject_uid"].to_numpy()).all():
        raise ValueError("psd.npz e janelas.csv não estão alinhados")
    folds = pd.read_csv(args.folds)
    args.saida.mkdir(parents=True, exist_ok=True)

    linhas = ["# LR — resultados", "",
              "Regressão logística L2 (C = 1), com peso de classe, sobre a média das features por pessoa; z-score com estatísticas "
              f"do treino de cada fold; 5 folds × 10 repetições. p: {args.permutacoes} permutações de rótulo "
              "com retreino, na repetição 0. IC de 95% por bootstrap estratificado (2000 reamostragens).", "",
              "| Conjunto | Features | AUC por pessoa [IC 95%] | AUC por repetição | p | Acurácia balanceada | "
              "Sensibilidade | Especificidade | Macro F1 |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    predicoes, nulas_todas = [], []
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
            p = None
            if nome == "absolutas":
                rep0 = folds_ds[folds_ds["repeticao"] == 0]
                observado = roc_auc_score(*pred.loc[pred["repeticao"] == 0, ["y", "prob"]].to_numpy().T)
                nulas = permutacao(x_treino, x_teste, y, rep0, n=args.permutacoes)
                nulas_todas.append(pd.DataFrame(dict(dataset=dataset, auc_nula=nulas)))
                p = p_valor(observado, nulas)
            resultado = avaliar(pred, semente=SEMENTE)
            linhas.append(_formatar(nome, dataset, resultado, p))
            print(f"{dataset} {nome}: AUC {resultado['auc']:.3f}" + ("" if p is None else f", p = {p:.3f}"))

    pd.concat(predicoes, ignore_index=True).to_csv(args.saida / "predicoes.csv", index=False)
    pd.concat(nulas_todas, ignore_index=True).to_csv(args.saida / "permutacoes.csv", index=False)
    linhas += ["", "A potência relativa é a checagem do MDD (docs/04_features.md): se a absoluta "
               "vencer muito só no MDD, a suspeita é o ganho do aparelho."]
    (args.saida / "resumo.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
