"""Experimento 1: quanto a data de gravação está associada ao diagnóstico.

Lê a data de gravação do cabeçalho EDF (`data_gravacao` em outputs/02_harmonizacao/manifesto.csv) e,
em cada dataset:
- cruza o grupo (HC × paciente) com o ano e o mês de gravação;
- agrupa as gravações em sessões de aquisição: datas consecutivas separadas por mais de
  `INTERVALO_SESSAO_DIAS` dias abrem uma sessão nova. O EDF não traz lote nem sessão, então a sessão
  é uma reconstrução a partir das datas;
- mede a associação: V de Cramér (grupo × sessão), quantas pessoas de cada classe estão em sessões
  com as duas classes e a AUC da própria data como escore de paciente, que mostra quanto um
  classificador que só soubesse a data acertaria.

Grava outputs/09_investigacao/datas/{ano_mes.csv, sessoes.csv, pessoas.csv, resumo.md}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

INTERVALO_SESSAO_DIAS = 30


def sessoes(datas: pd.Series, intervalo_dias: int = INTERVALO_SESSAO_DIAS) -> pd.Series:
    """Número da sessão (1, 2, ...) de cada data: uma lacuna maior que `intervalo_dias` abre outra."""
    datas = pd.to_datetime(datas)
    ordenadas = datas.sort_values()
    nova = ordenadas.diff().dt.days.fillna(0) > intervalo_dias
    return nova.cumsum().add(1).reindex(datas.index)


def v_cramer(tabela: pd.DataFrame) -> float:
    """V de Cramér de uma tabela de contingência (sem correção de continuidade)."""
    chi2 = chi2_contingency(tabela, correction=False)[0]
    n, menor = tabela.to_numpy().sum(), min(tabela.shape) - 1
    return float(np.sqrt(chi2 / (n * menor))) if menor > 0 else 0.0


def auc_da_data(y: pd.Series, datas: pd.Series) -> float:
    """AUC da data (dias desde 1970) como escore de paciente: > 0,5 quando os pacientes são gravados depois."""
    return float(roc_auc_score(y, pd.to_datetime(datas).map(pd.Timestamp.toordinal)))


def pessoas_por_dataset(manifesto: pd.DataFrame) -> dict[str, pd.DataFrame]:
    tabelas = {}
    for dataset, g in manifesto.groupby("dataset", sort=False):
        g = g[["subject_uid", "grupo", "data_gravacao"]].copy()
        g["y"] = (g["grupo"] != "HC").astype(int)
        datas = pd.to_datetime(g["data_gravacao"])
        g["ano_mes"] = datas.dt.strftime("%Y-%m")
        g["sessao"] = sessoes(g["data_gravacao"])
        g["sessao_mista"] = g.groupby("sessao")["y"].transform("nunique") > 1
        tabelas[dataset] = g.sort_values("data_gravacao").reset_index(drop=True)
    return tabelas


def resumo_sessoes(g: pd.DataFrame) -> pd.DataFrame:
    datas = pd.to_datetime(g["data_gravacao"])
    return (g.assign(data=datas).groupby("sessao")
            .agg(inicio=("data", "min"), fim=("data", "max"), hc=("y", lambda y: int((y == 0).sum())),
                 pacientes=("y", "sum"))
            .assign(inicio=lambda t: t["inicio"].dt.strftime("%d/%m/%Y"), fim=lambda t: t["fim"].dt.strftime("%d/%m/%Y")))


def associacao(g: pd.DataFrame) -> dict[str, float]:
    mistas = g[g["sessao_mista"]]
    return dict(v_cramer=v_cramer(pd.crosstab(g["sessao"], g["y"])), auc_data=auc_da_data(g["y"], g["data_gravacao"]),
                hc_em_sessao_mista=int((mistas["y"] == 0).sum()), pacientes_em_sessao_mista=int(mistas["y"].sum()),
                hc=int((g["y"] == 0).sum()), pacientes=int(g["y"].sum()))


def linhas_resumo(tabelas: dict[str, pd.DataFrame]) -> list[str]:
    linhas = ["# Data de gravação × diagnóstico", "",
              f"Sessão: datas separadas por mais de {INTERVALO_SESSAO_DIAS} dias abrem uma sessão nova "
              "(reconstrução a partir do cabeçalho EDF; o dataset não informa lote nem sessão).", "",
              "## Associação", "",
              "| Dataset | Sessões | V de Cramér (grupo × sessão) | AUC da data como escore "
              "| HC em sessões mistas | Pacientes em sessões mistas |",
              "| --- | --- | --- | --- | --- | --- |"]
    for dataset, g in tabelas.items():
        a = associacao(g)
        linhas.append(f"| {dataset} | {g['sessao'].nunique()} | {a['v_cramer']:.2f} | {a['auc_data']:.3f} "
                      f"| {a['hc_em_sessao_mista']} de {a['hc']} | {a['pacientes_em_sessao_mista']} de {a['pacientes']} |")
    for dataset, g in tabelas.items():
        linhas += ["", f"## {dataset}: sessões", "", "| Sessão | Início | Fim | HC | Pacientes |", "| --- | --- | --- | --- | --- |"]
        for s, r in resumo_sessoes(g).iterrows():
            linhas.append(f"| {s} | {r['inicio']} | {r['fim']} | {r['hc']} | {r['pacientes']} |")
        linhas += ["", f"## {dataset}: ano e mês", "", "| Ano-mês | HC | Pacientes |", "| --- | --- | --- |"]
        for mes, r in pd.crosstab(g["ano_mes"], g["y"]).reindex(columns=[0, 1], fill_value=0).iterrows():
            linhas.append(f"| {mes} | {r[0]} | {r[1]} |")
    return linhas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/09_investigacao/datas"))
    args = parser.parse_args()

    tabelas = pessoas_por_dataset(pd.read_csv(args.manifesto))
    args.saida.mkdir(parents=True, exist_ok=True)
    pd.concat([g.assign(dataset=d) for d, g in tabelas.items()]).to_csv(args.saida / "pessoas.csv", index=False)
    pd.concat([pd.crosstab(g["ano_mes"], g["grupo"]).reset_index().assign(dataset=d) for d, g in tabelas.items()]
              ).to_csv(args.saida / "ano_mes.csv", index=False)
    pd.concat([resumo_sessoes(g).reset_index().assign(dataset=d) for d, g in tabelas.items()]
              ).to_csv(args.saida / "sessoes.csv", index=False)
    texto = "\n".join(linhas_resumo(tabelas)) + "\n"
    (args.saida / "resumo.md").write_text(texto)
    print(texto)


if __name__ == "__main__":
    main()
