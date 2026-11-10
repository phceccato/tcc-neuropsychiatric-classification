"""Mapas topográficos de todas as pessoas, organizados por dataset, classe e sessão de gravação.

Para cada pessoa de outputs/02_harmonizacao/manifesto.csv, gera a figura de `topomapa.py` (uma pessoa por
figura, mesma metodologia e mesma escala) em

    outputs/10_topomapa/<dataset>/<classe>/<sessão>/<absoluta|relativa>/topomapa_<pessoa>_<escala>.png

- dataset: `Schizophrenia_dataset` (SZ) ou `Depression_dataset` (MDD);
- classe: `HC` ou o diagnóstico (`SZ`, `MDD`), da coluna `grupo` do manifesto;
- sessão: as sessões reconstruídas na etapa 09 (`pessoas_por_dataset`, pipeline/e09_investigacao/datas.py:
  sessão nova quando duas datas consecutivas distam mais de 30 dias), com o número e o período, por exemplo
  `sessao_2_jun-jul2003`. O número é o da sessão no dataset inteiro, então uma mesma sessão tem o mesmo nome
  nas pastas de HC e de pacientes;
- subpasta: `absoluta` para as escalas `log` (log10 da potência absoluta) e `absoluta` (µV²), que se distinguem
  pelo sufixo do arquivo; `relativa` para a potência relativa.

Escala de cor comum: em cada banda, a barra vai do mínimo ao máximo dos 19 canais de todas as pessoas
(`limites_comuns`), igual em todas as figuras, para as cores serem comparáveis entre pessoas.

Grava também, em outputs/10_topomapa/:
- `indice_<escala>.csv`: pessoa, dataset, classe, data, sessão, janelas e arquivo;
- `limites_<escala>.csv`: os limites da barra de cada banda;
- em cada subpasta de escala, `media_<sessão>_<escala>.png`: a média das pessoas da pasta (`medias_por_pasta`),
  na mesma escala comum; sempre refeita.
Retoma: pula as figuras que já existem, a menos que os limites tenham mudado (aí refaz todas) ou com
`--refazer`.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.e04_features.psd import BANDAS
from pipeline.e09_investigacao.datas import pessoas_por_dataset
from pipeline.e10_topomapa.topomapa import ESCALAS, figura, limites_comuns, valores_de_todos

NOMES_DATASETS = {"SZ": "Schizophrenia_dataset", "MDD": "Depression_dataset"}
SUBPASTAS = {"log": "absoluta", "absoluta": "absoluta", "relativa": "relativa"}  # log10 da absoluta fica em absoluta/
MESES = ("jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez")


def nome_sessao(numero: int, datas: pd.Series) -> str:
    """`sessao_<n>_<período>`: um mês (mar2003), meses do mesmo ano (jun-jul2003) ou de anos diferentes
    (dez2005-jan2006). Sem "/" nem espaço, para servir de nome de pasta."""
    d = pd.to_datetime(datas)
    ini, fim = d.min(), d.max()
    if (ini.year, ini.month) == (fim.year, fim.month):
        periodo = f"{MESES[ini.month - 1]}{ini.year}"
    elif ini.year == fim.year:
        periodo = f"{MESES[ini.month - 1]}-{MESES[fim.month - 1]}{ini.year}"
    else:
        periodo = f"{MESES[ini.month - 1]}{ini.year}-{MESES[fim.month - 1]}{fim.year}"
    return f"sessao_{numero}_{periodo}"


def plano(manifesto: Path, saida: Path, escala: str) -> pd.DataFrame:
    """Uma linha por pessoa, com a pasta de destino."""
    linhas = []
    for dataset, pessoas in pessoas_por_dataset(pd.read_csv(manifesto)).items():
        nomes = {s: nome_sessao(s, g["data_gravacao"]) for s, g in pessoas.groupby("sessao")}
        for p in pessoas.itertuples():
            pasta = saida / NOMES_DATASETS[dataset] / p.grupo / nomes[p.sessao] / SUBPASTAS[escala]
            linhas.append(dict(subject_uid=p.subject_uid, dataset=dataset, classe=p.grupo, data_gravacao=p.data_gravacao,
                               sessao=nomes[p.sessao], arquivo=pasta / f"topomapa_{p.subject_uid}_{escala}.png"))
    return pd.DataFrame(linhas)


def medias_por_pasta(tabela: pd.DataFrame, valores: dict[str, np.ndarray]) -> list[dict]:
    """Uma média por pasta (dataset, classe, sessão): média, entre as pessoas da pasta, dos valores [19, 4] de
    cada pessoa na escala escolhida (na escala `log`, média do log10, ou seja, média geométrica da potência).
    Cada pessoa pesa o mesmo, qualquer que seja o número de janelas."""
    saida = []
    for (dataset, classe, sessao), g in tabela.groupby(["dataset", "classe", "sessao"], sort=False):
        pasta = Path(g["arquivo"].iloc[0]).parent
        saida.append(dict(dataset=dataset, classe=classe, sessao=sessao, pasta=pasta, n=len(g),
                          valores=np.mean([valores[u] for u in g["subject_uid"]], axis=0)))
    return saida


def gerar(pessoa: str, valores: np.ndarray, janelas: int, escala: str, destino: str, limites: np.ndarray,
          n_pessoas: int) -> str:
    figura(valores, escala, f"{pessoa} ({janelas} janelas de 4 s)", Path(destino), limites, n_pessoas)
    return pessoa


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--escala", choices=tuple(ESCALAS), default="log")
    parser.add_argument("--manifesto", type=Path, default=Path("outputs/02_harmonizacao/manifesto.csv"))
    parser.add_argument("--psd", type=Path, default=Path("outputs/04_features/psd.npz"))
    parser.add_argument("--janelas", type=Path, default=Path("outputs/03_janelas/janelas.csv"))
    parser.add_argument("--saida", type=Path, default=Path("outputs/10_topomapa"))
    parser.add_argument("--processos", type=int, default=8)
    parser.add_argument("--refazer", action="store_true")
    args = parser.parse_args()

    tabela = plano(args.manifesto, args.saida, args.escala)
    valores = valores_de_todos(args.escala, args.psd, args.janelas)
    faltam = sorted(set(tabela["subject_uid"]) - set(valores))
    if faltam:
        raise ValueError(f"pessoas sem janela aceita: {faltam}")
    limites = limites_comuns(valores)
    arquivo_limites = args.saida / f"limites_{args.escala}.csv"
    novos = pd.DataFrame(dict(banda=BANDAS, minimo=limites[:, 0], maximo=limites[:, 1], unidade=ESCALAS[args.escala]))
    mudou = not arquivo_limites.exists() or not np.allclose(pd.read_csv(arquivo_limites)[["minimo", "maximo"]].to_numpy(), limites)
    refazer = args.refazer or mudou
    pendentes = tabela if refazer else tabela[~tabela["arquivo"].map(Path.exists)]
    print(f"{len(tabela)} pessoas, {len(pendentes)} figuras a gerar{' (limites novos)' if mudou else ''}", flush=True)
    print(novos.to_string(index=False), flush=True)

    indice = pd.read_csv(args.janelas)
    aceitas = indice[~indice["rejeitada"]].groupby("subject_uid").size()
    with ProcessPoolExecutor(args.processos) as executor:
        futuros = [executor.submit(gerar, p.subject_uid, valores[p.subject_uid], int(aceitas[p.subject_uid]), args.escala,
                                   str(p.arquivo), limites, len(valores))
                   for p in pendentes.itertuples()]
        for i, futuro in enumerate(as_completed(futuros), 1):
            pessoa = futuro.result()
            if i % 10 == 0 or i == len(futuros):
                print(f"[{i}/{len(futuros)}] {pessoa}", flush=True)
        medias = medias_por_pasta(tabela, valores)
        futuros = [executor.submit(figura, m["valores"], args.escala,
                                   f"Média de {m['n']} pessoa{'s' if m['n'] > 1 else ''}: {NOMES_DATASETS[m['dataset']]} / "
                                   f"{m['classe']} / {m['sessao']}",
                                   m["pasta"] / f"media_{m['sessao']}_{args.escala}.png", limites, len(valores))
                   for m in medias]
        for futuro in as_completed(futuros):
            futuro.result()
    print(f"{len(medias)} médias por sessão", flush=True)
    novos.to_csv(arquivo_limites, index=False)

    tabela["janelas_aceitas"] = tabela["subject_uid"].map(aceitas).astype("Int64")
    tabela["arquivo"] = tabela["arquivo"].map(lambda a: str(Path(a).relative_to(args.saida)))
    tabela.to_csv(args.saida / f"indice_{args.escala}.csv", index=False)
    print(tabela.groupby(["dataset", "classe", "sessao"]).size().to_string())


if __name__ == "__main__":
    main()
