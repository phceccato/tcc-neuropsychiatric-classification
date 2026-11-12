"""Exporta as previsões de todos os modelos para uma pasta versionada, para que qualquer pessoa reproduza os resultados.

Lê as saídas das etapas 05 (partições), 06 (LR_psd), 07 (DTSF) e 08 (clássicos) e grava em outputs/10_resultados/
(a única pasta de outputs/ versionada):

- previsoes/<dataset>/<modelo>.csv: uma linha por (repetição, fold, pessoa de teste), com y_true (1 = paciente),
  y_prob (probabilidade de paciente, fora do fold) e y_pred (y_prob ≥ 0,5);
- permutacoes/<dataset>/<modelo>.csv: o mesmo para as permutações de rótulo com retreino (coluna permutacao em vez
  de repeticao; y_true é o rótulo permutado), base da H1; e permutacoes/<dataset>/LR_psd_absolutas_nulas.csv, as
  AUCs nulas da etapa 06 (ela grava só as AUCs);
- folds.csv: as partições (dataset, repetição, fold, pessoa, grupo, papel), cópia de outputs/05_folds/folds.csv;
- metricas.csv: AUC por pessoa com IC, média ± desvio nas repetições e secundárias, recalculadas das previsões
  exportadas com pipeline/avaliacao/metricas.py;
- resumos/: cópia dos resumos das etapas 06, 07 e 08;
- manifesto.json: semente, versão do código (commit), dispositivo e número de parâmetros de cada modelo, e o que
  ficou de fora por estar incompleto;
- LEIAME.md: o que é cada arquivo e como recalcular a AUC por pessoa.

Só entram modelos com as 10 repetições completas; os incompletos são listados no manifesto. Antes de gravar, a AUC
por pessoa recalculada de cada modelo é conferida contra o resumo da etapa (|diferença| ≤ 0,0005): se não bater, o
script para sem gravar.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.avaliacao.metricas import LIMIAR, avaliar, media_desvio
from pipeline.e07_DTSF.avaliar import SEMENTE, completo, ler_modelo
from pipeline.e07_DTSF.redes_dtsf import TIPOS_DTSF
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS

DATASETS = ("SZ", "MDD")
SAIDA = Path("outputs/10_resultados")
TOLERANCIA = 5e-4
COLUNAS = ["repeticao", "fold", "subject_uid", "y_true", "y_pred", "y_prob"]


class Divergencia(ValueError):
    """A métrica recalculada das previsões exportadas não bate com o resumo da etapa."""


def padronizar(predicoes: pd.DataFrame, chave: str = "repeticao") -> pd.DataFrame:
    """Colunas y_true, y_pred (y_prob ≥ 0,5) e y_prob, ordenadas por (chave, fold, pessoa)."""
    t = pd.DataFrame({chave: predicoes[chave].astype(int), "fold": predicoes["fold"].astype(int),
                      "subject_uid": predicoes["subject_uid"], "y_true": predicoes["y"].astype(int),
                      "y_pred": (predicoes["prob"] >= LIMIAR).astype(int), "y_prob": predicoes["prob"]})
    return t.sort_values([chave, "fold", "subject_uid"]).reset_index(drop=True)


def para_avaliacao(exportado: pd.DataFrame) -> pd.DataFrame:
    """Volta ao formato de pipeline/avaliacao (y, prob): é o que um leitor faria para recalcular."""
    return exportado.rename(columns={"y_true": "y", "y_prob": "prob"})


def ler_permutacoes(pasta: Path) -> pd.DataFrame:
    partes = []
    for a in sorted(pasta.glob("p*_f*.csv")):
        p = pd.read_csv(a)
        p["permutacao"] = int(a.stem.split("_")[0][1:])
        partes.append(p)
    return pd.concat(partes, ignore_index=True) if partes else pd.DataFrame()


def _jsons(pasta: Path) -> list[dict]:
    return [json.load(open(a)) for a in sorted(pasta.glob("r*_f*.json"))]


def aucs_do_resumo(md: str) -> dict[tuple[str, str], float]:
    """{(dataset, modelo): AUC por pessoa} das tabelas "## SZ" / "## MDD" de um resumo (etapas 07 e 08)."""
    saida, dataset = {}, None
    for linha in md.splitlines():
        if linha.startswith("## "):
            dataset = linha[3:].strip() if linha[3:].strip() in DATASETS else None
        m = re.match(r"^\| (\S+) \|(?: [\d.]+ \|)? (\d\.\d{3}) \[\d\.\d{3}, \d\.\d{3}\] \|", linha)
        if dataset and m:
            saida[(dataset, m.group(1))] = float(m.group(2))
    return saida


def aucs_do_resumo_06(md: str) -> dict[tuple[str, str], float]:
    saida = {}
    for m in re.finditer(r"^\| (SZ|MDD) \| (absolutas|relativas) \| (\d\.\d{3}) \[", md, re.M):
        saida[(m.group(1), f"LR_psd_{m.group(2)}")] = float(m.group(3))
    return saida


def coletar(raiz: Path) -> dict:
    """Previsões, permutações e metadados de cada (dataset, modelo) completo."""
    modelos, incompletos = {}, []
    lr = pd.read_csv(raiz / "06_LR_psd/predicoes.csv")
    for d in DATASETS:
        fontes = {m: ("07", raiz / "07_DTSF/resultados" / d / m) for m in TIPOS_DTSF}
        fontes |= {m: ("08", raiz / "08_classicos/resultados" / d / m) for m in CLASSICOS}
        for m, (etapa, pasta) in fontes.items():
            p = ler_modelo(pasta)
            if not completo(p):
                incompletos.append(f"{d}/{m}")
                continue
            js = _jsons(pasta)
            dispositivos = sorted({j.get("dispositivo", "cpu") for j in js}) if etapa == "07" else ["cpu"]
            parametros = sorted({j["parametros"] for j in js if isinstance(j.get("parametros"), int)})
            modelos[(d, m)] = dict(etapa=etapa, previsoes=p, dispositivo=dispositivos, parametros=parametros,
                                   permutacoes=ler_permutacoes(raiz / "07_DTSF/permutacoes" / d / m)
                                   if etapa == "07" else pd.DataFrame())
        for nome in ("absolutas", "relativas"):
            g = lr[(lr["dataset"] == d) & (lr["features"] == nome)]
            p = g[["repeticao", "fold", "subject_uid", "y", "prob"]].reset_index(drop=True)
            if not completo(p):
                incompletos.append(f"{d}/LR_psd_{nome}")
                continue
            modelos[(d, f"LR_psd_{nome}")] = dict(etapa="06", previsoes=p, dispositivo=["cpu"], parametros=[],
                                                   permutacoes=pd.DataFrame())
    return dict(modelos=modelos, incompletos=incompletos)


def conferir(modelos: dict, raiz: Path) -> None:
    """A AUC por pessoa recalculada do arquivo exportado bate com a do resumo de cada etapa."""
    esperado = {}
    for caminho, leitor in (("07_DTSF/resultados/resumo_dtsf.md", aucs_do_resumo),
                            ("08_classicos/resultados/resumo.md", aucs_do_resumo),
                            ("06_LR_psd/resumo.md", aucs_do_resumo_06)):
        if (raiz / caminho).exists():
            esperado |= leitor((raiz / caminho).read_text(encoding="utf-8"))
    for (d, m), info in modelos.items():
        if (d, m) not in esperado:
            raise Divergencia(f"{d} {m}: não aparece no resumo da etapa {info['etapa']}; rode o avaliar da etapa antes")
        auc = avaliar(para_avaliacao(padronizar(info["previsoes"])), semente=SEMENTE)["auc"]
        if abs(auc - esperado[(d, m)]) > TOLERANCIA:
            raise Divergencia(f"{d} {m}: AUC {auc:.4f} recalculada, {esperado[(d, m)]:.3f} no resumo")


def metricas(modelos: dict) -> pd.DataFrame:
    linhas = []
    for (d, m), info in modelos.items():
        p = para_avaliacao(padronizar(info["previsoes"]))
        r, md = avaliar(p, semente=SEMENTE), media_desvio(p)
        linha = dict(dataset=d, modelo=m, etapa=info["etapa"], auc_pessoa=r["auc"], ic_baixo=r["ic_baixo"],
                     ic_alto=r["ic_alto"], acuracia_balanceada_pessoa=r["acuracia_balanceada"],
                     sensibilidade_pessoa=r["sensibilidade"], especificidade_pessoa=r["especificidade"],
                     macro_f1_pessoa=r["macro_f1"])
        for k, (media, desvio) in md.items():
            linha[f"{k}_rep_media"], linha[f"{k}_rep_desvio"] = media, desvio
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _commit() -> str:
    r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else "desconhecido"


LEIAME = """# Resultados exportados

Previsões fora do fold de todos os modelos do TCC (SZ × HC e MDD × HC), para recalcular as métricas e os testes sem
treinar nada. Gerado por `python -m pipeline.e10_resultados.exportar`; o método está em docs/10_resultados.md.

| Arquivo | Conteúdo |
| --- | --- |
| `previsoes/<dataset>/<modelo>.csv` | uma linha por (repetição, fold, pessoa de teste): `y_true` (1 = paciente), `y_prob` (probabilidade de paciente), `y_pred` (`y_prob` ≥ 0,5) |
| `permutacoes/<dataset>/<modelo>.csv` | o mesmo nas permutações de rótulo com retreino (repetição 0); `y_true` é o rótulo permutado |
| `permutacoes/<dataset>/LR_psd_absolutas_nulas.csv` | as AUCs nulas da LR_psd (1000 permutações) |
| `folds.csv` | as partições: 5 folds × 10 repetições, por pessoa, com o papel (treino, validação, teste) |
| `metricas.csv` | AUC por pessoa com IC de 95%, secundárias e média ± desvio nas 10 repetições, recalculadas destes arquivos |
| `resumos/` | os resumos das etapas 06, 07 e 08, com os testes das hipóteses |
| `manifesto.json` | semente, commit do código, dispositivo e parâmetros de cada modelo, modelos incompletos |

AUC por pessoa: média de `y_prob` de cada pessoa nas 10 repetições, e uma AUC sobre as pessoas:

```python
import pandas as pd
from sklearn.metrics import roc_auc_score
p = pd.read_csv("previsoes/SZ/D_FA.csv")
pessoa = p.groupby("subject_uid").agg(y=("y_true", "first"), prob=("y_prob", "mean"))
print(roc_auc_score(pessoa["y"], pessoa["prob"]))
```
"""


def exportar(raiz: Path = Path("outputs"), saida: Path = SAIDA) -> dict:
    dados = coletar(raiz)
    modelos = dados["modelos"]
    if not modelos:
        raise ValueError("nenhum modelo completo para exportar")
    conferir(modelos, raiz)

    if saida.exists():
        shutil.rmtree(saida)
    for (d, m), info in modelos.items():
        (saida / "previsoes" / d).mkdir(parents=True, exist_ok=True)
        padronizar(info["previsoes"])[COLUNAS].to_csv(saida / "previsoes" / d / f"{m}.csv", index=False)
        if len(info["permutacoes"]):
            (saida / "permutacoes" / d).mkdir(parents=True, exist_ok=True)
            padronizar(info["permutacoes"], "permutacao").to_csv(saida / "permutacoes" / d / f"{m}.csv", index=False)
    nulas = raiz / "06_LR_psd/permutacoes.csv"
    if nulas.exists():
        n = pd.read_csv(nulas)
        for d, g in n.groupby("dataset"):
            (saida / "permutacoes" / d).mkdir(parents=True, exist_ok=True)
            g[["auc_nula"]].to_csv(saida / "permutacoes" / d / "LR_psd_absolutas_nulas.csv", index=False)
    shutil.copy(raiz / "05_folds/folds.csv", saida / "folds.csv")
    metricas(modelos).to_csv(saida / "metricas.csv", index=False)
    (saida / "resumos").mkdir()
    for origem, destino in (("06_LR_psd/resumo.md", "06_LR_psd.md"), ("07_DTSF/resultados/resumo_dtsf.md", "07_DTSF.md"),
                            ("08_classicos/resultados/resumo.md", "08_classicos.md")):
        if (raiz / origem).exists():
            shutil.copy(raiz / origem, saida / "resumos" / destino)
    manifesto = dict(semente=SEMENTE, commit=_commit(), limiar_y_pred=LIMIAR,
                     modelos={f"{d}/{m}": dict(etapa=i["etapa"], dispositivo=i["dispositivo"], parametros=i["parametros"],
                                               linhas=len(i["previsoes"]), permutacoes=int(i["permutacoes"]["permutacao"].nunique())
                                               if len(i["permutacoes"]) else 0)
                              for (d, m), i in modelos.items()},
                     incompletos=dados["incompletos"])
    (saida / "LEIAME.md").write_text(LEIAME, encoding="utf-8")
    (saida / "manifesto.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--saida", type=Path, default=SAIDA)
    args = parser.parse_args()
    manifesto = exportar(args.outputs, args.saida)
    print(f"{len(manifesto['modelos'])} (dataset, modelo) exportados em {args.saida}")
    if manifesto["incompletos"]:
        print(f"fora por estarem incompletos: {', '.join(manifesto['incompletos'])}")


if __name__ == "__main__":
    main()
