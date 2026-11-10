"""Experimento 4: influência da sessão de gravação nas AUCs.

Duas análises, com as sessões reconstruídas pela data de gravação (pipeline/e09_investigacao/datas.py):

1. **Subconjuntos, sem retreino.** As previsões fora do fold já gravadas (D da etapa 07, os 9
   baselines da 08 e a LR de 76 bandas da 06) são reavaliadas só nas pessoas de um subconjunto em
   que a data se confunde menos com o diagnóstico. O modelo continua treinado com todas as pessoas.
   - SZ: os 3 pacientes gravados em 2003 contra os 14 HC (todos de 2003), e os 11 pacientes de
     2004–2008 contra os mesmos HC.
   - MDD: sem o lote de jan/2013; só a sessão de abr–jul/2013, a única com as duas classes (4 HC);
     e cada sessão de HC separada contra todos os pacientes.
2. **Retreino sem o lote de jan/2013 (MDD).** Novas partições 5 × 10 (mesmo procedimento de
   pipeline/e05_folds) nas 42 pessoas restantes e o D treinado do zero com
   pipeline/e07_DTSF/rodar.py. Na SZ não há retreino equivalente: os 14 HC estão numa sessão só.

Validação por sessão (deixar uma sessão de fora) não é possível: quase todas as sessões têm uma
classe só (ver outputs/09_investigacao/datas/resumo.md).

Grava em outputs/09_investigacao/lote/: subconjuntos.csv, folds_mdd_sem_jan2013.csv,
resultados/MDD/D/r<rep>_f<fold>.{csv,json} e resumo.md.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

from pipeline.avaliacao.metricas import escores_por_pessoa, ic_bootstrap
from pipeline.e05_folds.folds import gerar
from pipeline.e07_DTSF.avaliar import SEMENTE, completo, ler_modelo
from pipeline.e07_DTSF.rodar import _iniciar, rodar_tarefa
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS  # os 9 baselines da etapa 08 (LR, RF e KNN × 3 modalidades)
from pipeline.e09_investigacao.datas import pessoas_por_dataset


def subconjuntos(pessoas: dict[str, pd.DataFrame]) -> dict[tuple[str, str], set[str]]:
    """{(dataset, nome): subject_uids} das análises da docstring."""
    sz, mdd = pessoas["SZ"], pessoas["MDD"]
    ano_sz = pd.to_datetime(sz["data_gravacao"]).dt.year
    hc_sz = set(sz.loc[sz["y"] == 0, "subject_uid"])
    saida = {("SZ", "todas as pessoas"): set(sz["subject_uid"]),
             ("SZ", "pacientes de 2003 × HC"): hc_sz | set(sz.loc[(sz["y"] == 1) & (ano_sz == 2003), "subject_uid"]),
             ("SZ", "pacientes de 2004–2008 × HC"): hc_sz | set(sz.loc[(sz["y"] == 1) & (ano_sz > 2003), "subject_uid"])}
    pacientes = set(mdd.loc[mdd["y"] == 1, "subject_uid"])
    saida[("MDD", "todas as pessoas")] = set(mdd["subject_uid"])
    lote = set(mdd.loc[(mdd["y"] == 0) & (mdd["data_gravacao"].str.startswith("2013-01")), "subject_uid"])
    saida[("MDD", "sem o lote de jan/2013")] = set(mdd["subject_uid"]) - lote
    mista = mdd.loc[mdd["sessao_mista"], "subject_uid"]
    saida[("MDD", "só a sessão de abr–jul/2013")] = set(mista)
    for sessao, g in mdd[(mdd["y"] == 0) & ~mdd["sessao_mista"]].groupby("sessao"):
        inicio = pd.to_datetime(g["data_gravacao"]).min()
        saida[("MDD", f"HC de {inicio:%m/%Y} × pacientes")] = set(g["subject_uid"]) | pacientes
    return saida


def formatar_ic(par: tuple[float, float]) -> str:
    return f"[{par[0]:.3f}, {par[1]:.3f}]"


def auc_subconjunto(predicoes: pd.DataFrame, uids: set[str], semente: int = SEMENTE) -> dict:
    escores = escores_por_pessoa(predicoes[predicoes["subject_uid"].isin(uids)])
    y, p = escores["y"].to_numpy(), escores["prob"].to_numpy()
    return dict(hc=int((y == 0).sum()), pacientes=int(y.sum()), auc=float(roc_auc_score(y, p)),
                ic=ic_bootstrap(y, p, semente=semente))


def predicoes_por_modelo(dataset: str, lr: pd.DataFrame) -> dict[str, pd.DataFrame]:
    modelos = {"D": ler_modelo(Path("outputs/07_DTSF/resultados") / dataset / "D")}
    modelos |= {c: ler_modelo(Path("outputs/08_classicos/resultados") / dataset / c) for c in CLASSICOS}
    modelos["LR de 76 bandas"] = lr[(lr["dataset"] == dataset) & (lr["features"] == "absolutas")]
    return modelos


def folds_sem_lote(manifesto: pd.DataFrame, lote: set[str]) -> pd.DataFrame:
    pessoas = manifesto[(manifesto["dataset"] == "MDD") & ~manifesto["subject_uid"].isin(lote)]
    particoes = gerar(pessoas[["subject_uid", "grupo"]])
    particoes.insert(0, "dataset", "MDD")
    return particoes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/09_investigacao/lote"))
    parser.add_argument("--processos", type=int, default=10)
    parser.add_argument("--sem-retreino", action="store_true", help="só a análise de subconjuntos")
    args = parser.parse_args()
    args.saida.mkdir(parents=True, exist_ok=True)

    manifesto = pd.read_csv(args.manifesto)
    pessoas = pessoas_por_dataset(manifesto)
    subs = subconjuntos(pessoas)
    lr = pd.read_csv("outputs/06_LR/predicoes.csv")

    linhas = []
    for dataset in ("SZ", "MDD"):
        modelos = predicoes_por_modelo(dataset, lr)
        for (ds, nome), uids in subs.items():
            if ds != dataset:
                continue
            for modelo, pred in modelos.items():
                r = auc_subconjunto(pred, uids)
                linhas.append(dict(dataset=ds, subconjunto=nome, modelo=modelo, hc=r["hc"], pacientes=r["pacientes"],
                                   auc=r["auc"], ic_baixo=r["ic"][0], ic_alto=r["ic"][1]))
    tabela = pd.DataFrame(linhas)
    tabela.to_csv(args.saida / "subconjuntos.csv", index=False)

    lote = subs[("MDD", "todas as pessoas")] - subs[("MDD", "sem o lote de jan/2013")]
    folds_csv = args.saida / "folds_mdd_sem_jan2013.csv"
    if not folds_csv.exists():
        folds_sem_lote(manifesto, lote).to_csv(folds_csv, index=False)
    resultados = args.saida / "resultados"
    if not args.sem_retreino:
        tarefas = [(r, f) for r in range(10) for f in range(5) if not (resultados / "MDD" / "D" / f"r{r}_f{f}.csv").exists()]
        print(f"{len(tarefas)} treinos pendentes", flush=True)
        with ProcessPoolExecutor(args.processos, initializer=_iniciar) as executor:
            futuros = [executor.submit(rodar_tarefa, "MDD", "D", r, f, str(folds_csv), str(resultados)) for r, f in tarefas]
            for i, futuro in enumerate(as_completed(futuros), 1):
                print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)

    texto = "\n".join(linhas_resumo(tabela, ler_modelo(resultados / "MDD" / "D"), subs)) + "\n"
    (args.saida / "resumo.md").write_text(texto)
    print(texto)


def linhas_resumo(tabela: pd.DataFrame, retreino: pd.DataFrame, subs: dict) -> list[str]:
    linhas = ["# Influência da sessão de gravação", "",
              "## Subconjuntos, sem retreino", "",
              "AUC por pessoa (média das 10 repetições fora do fold) só nas pessoas do subconjunto; IC de 95% por bootstrap "
              "estratificado. Os modelos foram treinados com todas as pessoas.", ""]
    modelos = list(dict.fromkeys(tabela["modelo"]))
    for dataset in ("SZ", "MDD"):
        t = tabela[tabela["dataset"] == dataset]
        linhas += [f"### {dataset}", "", "| Subconjunto | HC | Pacientes | " + " | ".join(modelos) + " |",
                   "| --- | --- | --- | " + " | ".join("---" for _ in modelos) + " |"]
        for nome, g in t.groupby("subconjunto", sort=False):
            g = g.set_index("modelo")
            linhas.append(f"| {nome} | {g['hc'].iloc[0]} | {g['pacientes'].iloc[0]} | "
                          + " | ".join(f"{g.at[m, 'auc']:.3f}" for m in modelos) + " |")
        linhas += ["", "IC de 95% do D:", ""]
        for nome, g in t[t["modelo"] == "D"].groupby("subconjunto", sort=False):
            linhas.append(f"- {nome}: {g['auc'].iloc[0]:.3f} {formatar_ic((g['ic_baixo'].iloc[0], g['ic_alto'].iloc[0]))}")
        linhas.append("")
    linhas += ["## Retreino do D no MDD sem o lote de jan/2013", ""]
    if retreino.empty or not completo(retreino):
        return linhas + ["Incompleto: faltam previsões em `resultados/MDD/D`."]
    r = auc_subconjunto(retreino, subs[("MDD", "sem o lote de jan/2013")])
    posthoc = tabela[(tabela["dataset"] == "MDD") & (tabela["subconjunto"] == "sem o lote de jan/2013") & (tabela["modelo"] == "D")].iloc[0]
    linhas += ["| Treino | HC | Pacientes | AUC por pessoa [IC 95%] |", "| --- | --- | --- | --- |",
               f"| Com o lote (etapa 07), avaliado sem ele | {posthoc['hc']} | {posthoc['pacientes']} | "
               f"{posthoc['auc']:.3f} {formatar_ic((posthoc['ic_baixo'], posthoc['ic_alto']))} |",
               f"| Sem o lote, partições novas | {r['hc']} | {r['pacientes']} | {r['auc']:.3f} {formatar_ic(r['ic'])} |"]
    return linhas


if __name__ == "__main__":
    main()
