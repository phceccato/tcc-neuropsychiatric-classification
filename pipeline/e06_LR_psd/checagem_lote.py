"""Checagem da potência relativa no MDD, estratificada por sessão de gravação (post-hoc, descritiva).

A etapa 06 compara a LR com potência absoluta e com potência relativa (banda ÷ soma das 4 bandas) para
suspeitar do ganho do amplificador. Esta análise pergunta o que essa comparação não responde: a
normalização também remove a diferença de ritmo entre sessões?

Sessão = datas consecutivas do cabeçalho EDF; uma lacuna de mais de 30 dias abre outra
(pipeline/e09_investigacao/datas.py). No MDD são três: S1 (jul–ago/2011, só HC), S2 (12–14/jan/2013,
só HC) e S3 (abr–jul/2013, 29 pacientes e 4 HC).

Calcula, com as features de outputs/04_features/psd.npz (média das janelas aceitas, depois dos 19 canais):
- média ± desvio de cada banda, absoluta e relativa, por sessão e classe;
- AUC de cada banda sozinha como escore de paciente, com todos, sem a S2 e só na S3;
- AUC cruzada da LR (mesmas partições da etapa 06, sem as pessoas da sessão excluída), com potência
  absoluta e relativa.

Não altera dados nem protocolo: a LR da etapa 06 continua com todas as pessoas. Grava
outputs/06_LR_psd/checagem_lote.md.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.avaliacao.metricas import avaliar
from pipeline.e06_LR_psd.lr import SEMENTE, media_por_pessoa, prever_cv
from pipeline.e09_investigacao.datas import pessoas_por_dataset

BANDAS = ("delta", "teta", "alfa", "beta")
SESSAO_JAN2013 = 2


def bandas_por_pessoa(x: np.ndarray, nomes: np.ndarray, janelas: pd.DataFrame) -> pd.DataFrame:
    """Média das janelas aceitas de cada pessoa e, depois, dos canais de cada banda (nomes `<canal>_<banda>`)."""
    por_pessoa = media_por_pessoa(x, janelas, so_sem_overlap=False)
    nomes = [str(n) for n in nomes]
    return pd.DataFrame({b: por_pessoa.iloc[:, [i for i, n in enumerate(nomes) if n.endswith("_" + b)]].mean(axis=1)
                         for b in BANDAS})


def resumo_por_sessao(bandas: pd.DataFrame, pessoas: pd.DataFrame) -> pd.DataFrame:
    """Média, desvio e n de cada banda por (sessão, grupo). `pessoas` é indexada por subject_uid."""
    juntas = bandas.join(pessoas[["sessao", "grupo"]], how="inner")
    saida = juntas.groupby(["sessao", "grupo"]).agg(["mean", "std", "count"])
    return saida


def auc_univariada(y: pd.Series, escore: pd.Series, manter: pd.Series | None = None) -> float:
    """AUC do escore como previsor de paciente (< 0,5: pacientes têm valores menores). NaN se faltar classe."""
    if manter is not None:
        y, escore = y[manter], escore[manter]
    if y.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(y, escore))


def tabela_auc_univariada(bandas: pd.DataFrame, pessoas: pd.DataFrame) -> pd.DataFrame:
    """AUC de cada banda sozinha: todos, sem a sessão de jan/2013 e só na sessão de abr–jul/2013."""
    pessoas = pessoas.loc[bandas.index]
    y, sessao = pessoas["y"], pessoas["sessao"]
    linhas = []
    for b in BANDAS:
        linhas.append(dict(banda=b,
                           todos=auc_univariada(y, bandas[b]),
                           sem_jan2013=auc_univariada(y, bandas[b], sessao != SESSAO_JAN2013),
                           so_sessao_3=auc_univariada(y, bandas[b], sessao == 3)))
    return pd.DataFrame(linhas)


def lr_sem_sessao(x: np.ndarray, janelas: pd.DataFrame, y: pd.Series, folds_ds: pd.DataFrame,
                  excluir: set[str]) -> dict[str, float]:
    """LR da etapa 06 (mesmas partições) sem as pessoas de `excluir`, no treino e no teste."""
    folds = folds_ds[~folds_ds["subject_uid"].isin(excluir)]
    y = y[~y.index.isin(excluir)]
    x_treino = media_por_pessoa(x, janelas, so_sem_overlap=False).loc[y.index]
    x_teste = media_por_pessoa(x, janelas, so_sem_overlap=True).loc[y.index]
    return avaliar(prever_cv(x_treino, x_teste, y, folds), semente=SEMENTE)


def _md_resumo(resumo: pd.DataFrame, titulo: str) -> list[str]:
    linhas = [f"**{titulo}**", "", "| Sessão | Classe | n | Delta | Teta | Alfa | Beta |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for (sessao, grupo), r in resumo.iterrows():
        n = int(r[(BANDAS[0], "count")])
        celulas = " | ".join(f"{r[(b, 'mean')]:.2f} ± {r[(b, 'std')]:.2f}" if n > 1 else f"{r[(b, 'mean')]:.2f}"
                             for b in BANDAS)
        linhas.append(f"| S{sessao} | {grupo} | {n} | {celulas} |")
    return linhas + [""]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--features", type=Path, default=Path("outputs/04_features/psd.npz"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--folds", type=Path, default=Path("outputs/05_folds/folds.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/06_LR_psd/checagem_lote.md"))
    args = parser.parse_args()

    features = np.load(args.features)
    janelas = pd.read_csv(args.janelas)
    if not (features["subject_uid"] == janelas["subject_uid"].to_numpy()).all():
        raise ValueError("psd.npz e janelas.csv não estão alinhados")
    manifesto = pd.read_csv(args.manifesto)
    # o manifesto inteiro (INCLUIR e SINALIZAR), como a etapa 05 gera as partições
    pessoas = pessoas_por_dataset(manifesto)["MDD"].set_index("subject_uid")
    folds_mdd = pd.read_csv(args.folds).query("dataset == 'MDD'")
    y = pessoas["y"]
    jan = set(pessoas.index[pessoas["sessao"] == SESSAO_JAN2013])

    saida = ["# LR_psd — checagem da potência relativa por sessão (MDD)", "",
             "Análise post-hoc e descritiva; não muda a LR da etapa 06. Sessões reconstruídas pela data do cabeçalho EDF "
             "(lacuna > 30 dias): S1 jul–ago/2011 (só HC), S2 12–14/jan/2013 (só HC), S3 abr–jul/2013 (pacientes e 4 HC).", ""]
    univariadas = {}
    for tipo, titulo in (("absolutas", "log10 da potência absoluta (µV²)"), ("relativas", "Potência relativa")):
        bandas = bandas_por_pessoa(features[tipo], features["nomes"], janelas).loc[y.index]
        saida += _md_resumo(resumo_por_sessao(bandas, pessoas), titulo)
        univariadas[tipo] = tabela_auc_univariada(bandas, pessoas)
    for tipo, titulo in (("absolutas", "AUC de cada banda sozinha, potência absoluta"),
                         ("relativas", "AUC de cada banda sozinha, potência relativa")):
        saida += [f"**{titulo}** (< 0,5: pacientes com valores menores)", "",
                  "| Banda | Todos | Sem a S2 (jan/2013) | Só S3 |", "| --- | --- | --- | --- |"]
        saida += [f"| {r.banda} | {r.todos:.3f} | {r.sem_jan2013:.3f} | {r.so_sessao_3:.3f} |"
                  for r in univariadas[tipo].itertuples()] + [""]

    saida += ["**LR de 76 bandas (mesmas partições da etapa 06)**", "",
              "| Features | Pessoas | AUC por pessoa [IC 95%] | AUC por repetição |", "| --- | --- | --- | --- |"]
    for tipo in ("absolutas", "relativas"):
        for rotulo, excluir in (("todas (etapa 06)", set()), ("sem as 14 HC da S2", jan)):
            r = lr_sem_sessao(features[tipo], janelas, y, folds_mdd, excluir)
            saida.append(f"| {tipo} | {rotulo} | {r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}] | "
                         f"{r['auc_rep_media']:.3f} ± {r['auc_rep_desvio']:.3f} |")
            print(f"{tipo} {rotulo}: AUC {r['auc']:.3f}")
    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text("\n".join(saida) + "\n", encoding="utf-8")
    print(f"→ {args.saida}")


if __name__ == "__main__":
    main()
