"""Exporta as previsões de todos os modelos para uma pasta versionada, para que qualquer pessoa reproduza os resultados.

Lê as saídas das etapas 05 (partições), 06 (LR_psd), 07 (DTSF) e 08 (clássicos) e grava em outputs/10_resultados/
(a única pasta de outputs/ versionada):

- previsoes/<dataset>/<modelo>.csv: uma linha por (repetição, fold, pessoa de teste), com y_true (1 = paciente),
  y_prob (probabilidade de paciente, fora do fold) e y_pred (y_prob ≥ 0,5);
- perdas/<dataset>/<modelo>.csv, só do modelo principal (D_FA): a perda de treino e a de validação interna de cada
  época de cada (repetição, fold), com a época escolhida (`melhor`); a validação é o split interno do treino, nunca o
  teste, e a perda de treino é a do lote com Mixup;
- folds.csv: as partições (dataset, repetição, fold, pessoa, grupo, papel), cópia de outputs/05_folds/folds.csv;
- metricas.csv: AUC por pessoa com IC, média ± desvio nas repetições e secundárias, recalculadas das previsões
  exportadas com pipeline/avaliacao/metricas.py;
- resumos/: cópia dos resumos das etapas 06, 07 e 08;
- manifesto.json: semente, versão do código (commit), dispositivo e número de parâmetros de cada modelo, e o que
  ficou de fora por estar incompleto;
- LEIAME.md: o que é cada arquivo e como recalcular a AUC por pessoa.

Só entram modelos com as 10 repetições completas; os incompletos são listados no manifesto. Antes de gravar, a AUC
por pessoa recalculada de cada modelo é conferida contra o resumo da etapa (|diferença| ≤ 0,0005): se não bater, o
script para sem gravar. Também para, antes de apagar a pasta de saída, se o D_FA não estiver completo nos dois
datasets ou se os DTSF vierem de dispositivos diferentes (`--permitir-parcial` desliga essa trava).
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from pathlib import Path

import pandas as pd

from pipeline.avaliacao.metricas import LIMIAR, METRICAS_REPETICAO, avaliar, media_desvio
from pipeline.e07_DTSF.avaliar import SEMENTE, completo, ler_modelo
from pipeline.e07_DTSF.redes_dtsf import PRINCIPAL, TIPOS_DTSF
from pipeline.e08_classicos.treino import MODELOS as CLASSICOS

DATASETS = ("SZ", "MDD")
SAIDA = Path("outputs/10_resultados")
TOLERANCIA = 5e-4
COLUNAS = ["repeticao", "fold", "subject_uid", "y_true", "y_pred", "y_prob"]


class Divergencia(ValueError):
    """A métrica recalculada das previsões exportadas não bate com o resumo da etapa."""


class RodadaIncompleta(ValueError):
    """As saídas não são uma rodada completa e única: falta o modelo principal ou os DTSF vêm de dispositivos diferentes."""


def verificar_rodada(modelos: dict) -> None:
    """Recusa exportar uma rodada parcial, que apagaria a exportação completa que já está no git.

    Exige o modelo principal (D_FA) completo nos dois datasets e todos os DTSF da etapa 07 no mesmo dispositivo
    (uma rodada em CPU e outra em GPU não se misturam: docs/07_DTSF.md).
    """
    faltam = [d for d in DATASETS if (d, PRINCIPAL) not in modelos]
    if faltam:
        raise RodadaIncompleta(f"{PRINCIPAL} ausente ou incompleto em {', '.join(faltam)}; a exportação recriaria "
                               "outputs/10_resultados sem ele. Rode a rodada completa ou use --permitir-parcial.")
    dispositivos = {tuple(i["dispositivo"]) for i in modelos.values() if i["etapa"] == "07"}
    if len(dispositivos) > 1:
        raise RodadaIncompleta("os DTSF da etapa 07 vêm de dispositivos diferentes "
                               f"({sorted(dispositivos)}); não misture CPU e GPU. Use --permitir-parcial para ignorar.")


def padronizar(predicoes: pd.DataFrame, chave: str = "repeticao") -> pd.DataFrame:
    """Colunas y_true, y_pred (y_prob ≥ 0,5) e y_prob, ordenadas por (chave, fold, pessoa)."""
    t = pd.DataFrame({chave: predicoes[chave].astype(int), "fold": predicoes["fold"].astype(int),
                      "subject_uid": predicoes["subject_uid"], "y_true": predicoes["y"].astype(int),
                      "y_pred": (predicoes["prob"] >= LIMIAR).astype(int), "y_prob": predicoes["prob"]})
    return t.sort_values([chave, "fold", "subject_uid"]).reset_index(drop=True)


def para_avaliacao(exportado: pd.DataFrame) -> pd.DataFrame:
    """Volta ao formato de pipeline/avaliacao (y, prob): é o que um leitor faria para recalcular."""
    return exportado.rename(columns={"y_true": "y", "y_prob": "prob"})


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


def ler_perdas(pasta: Path) -> pd.DataFrame:
    """Histórico de perdas de cada fold de um modelo: uma linha por (repetição, fold, época)."""
    linhas = []
    for a in sorted(pasta.glob("r*_f*.json")):
        j = json.load(open(a))
        if "historico" not in j:
            continue
        r, f = (int(x[1:]) for x in a.stem.split("_"))
        linhas += [dict(repeticao=r, fold=f, epoca=h["epoca"], perda_treino=h["perda_treino"],
                        perda_validacao=h["perda_validacao"], melhor=int(h["epoca"] == j["melhor_epoca"]))
                   for h in j["historico"]]
    return pd.DataFrame(linhas, columns=["repeticao", "fold", "epoca", "perda_treino", "perda_validacao", "melhor"])


def coletar(raiz: Path) -> dict:
    """Previsões e metadados de cada (dataset, modelo) completo."""
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
            perdas = ler_perdas(pasta) if etapa == "07" and m == PRINCIPAL else pd.DataFrame()
            modelos[(d, m)] = dict(etapa=etapa, previsoes=p, dispositivo=dispositivos, parametros=parametros, perdas=perdas)
        for nome in ("absolutas", "relativas"):
            g = lr[(lr["dataset"] == d) & (lr["features"] == nome)]
            p = g[["repeticao", "fold", "subject_uid", "y", "prob"]].reset_index(drop=True)
            if not completo(p):
                incompletos.append(f"{d}/LR_psd_{nome}")
                continue
            modelos[(d, f"LR_psd_{nome}")] = dict(etapa="06", previsoes=p, dispositivo=["cpu"], parametros=[],
                                                   perdas=pd.DataFrame())
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
        if abs(auc - esperado[(d, m)]) > TOLERANCIA + 1e-9:   # 0,0005 exato (arredondamento do resumo) passa
            raise Divergencia(f"{d} {m}: AUC {auc:.4f} recalculada, {esperado[(d, m)]:.3f} no resumo")


def metricas(modelos: dict) -> pd.DataFrame:
    linhas = []
    for (d, m), info in modelos.items():
        p = para_avaliacao(padronizar(info["previsoes"]))
        r, md = avaliar(p, semente=SEMENTE), media_desvio(p)
        linha = dict(dataset=d, modelo=m, etapa=info["etapa"], auc_pessoa=r["auc"], ic_baixo=r["ic_baixo"],
                     ic_alto=r["ic_alto"])
        linha |= {f"{k}_pessoa": r[k] for k in METRICAS_REPETICAO if k != "auc"}
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
| `perdas/<dataset>/D_FA.csv` | perda de treino e de validação interna por época, em cada (repetição, fold), do modelo principal; `melhor` marca a época escolhida |
| `folds.csv` | as partições: 5 folds × 10 repetições, por pessoa, com o papel (treino, validação, teste) |
| `metricas.csv` | AUC por pessoa com IC de 95%; acurácia, acurácia balanceada, precisão, sensibilidade (recall), especificidade, F1 e Macro F1 no limiar 0,5; e a média ± desvio de todas nas 10 repetições, recalculadas destes arquivos |
| `resumos/` | os resumos das etapas 06, 07 e 08, com as comparações entre modelos |
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


def exportar(raiz: Path = Path("outputs"), saida: Path = SAIDA, permitir_parcial: bool = False) -> dict:
    dados = coletar(raiz)
    modelos = dados["modelos"]
    if not modelos:
        raise ValueError("nenhum modelo completo para exportar")
    if not permitir_parcial:
        verificar_rodada(modelos)
    conferir(modelos, raiz)

    if saida.exists():
        shutil.rmtree(saida)
    for (d, m), info in modelos.items():
        (saida / "previsoes" / d).mkdir(parents=True, exist_ok=True)
        padronizar(info["previsoes"])[COLUNAS].to_csv(saida / "previsoes" / d / f"{m}.csv", index=False)
        if len(info["perdas"]):
            (saida / "perdas" / d).mkdir(parents=True, exist_ok=True)
            info["perdas"].to_csv(saida / "perdas" / d / f"{m}.csv", index=False)
    shutil.copy(raiz / "05_folds/folds.csv", saida / "folds.csv")
    metricas(modelos).to_csv(saida / "metricas.csv", index=False)
    (saida / "resumos").mkdir()
    for origem, destino in (("06_LR_psd/resumo.md", "06_LR_psd.md"), ("07_DTSF/resultados/resumo_dtsf.md", "07_DTSF.md"),
                            ("08_classicos/resultados/resumo.md", "08_classicos.md")):
        if (raiz / origem).exists():
            shutil.copy(raiz / origem, saida / "resumos" / destino)
    manifesto = dict(semente=SEMENTE, commit=_commit(), limiar_y_pred=LIMIAR,
                     modelos={f"{d}/{m}": dict(etapa=i["etapa"], dispositivo=i["dispositivo"], parametros=i["parametros"],
                                               linhas=len(i["previsoes"]))
                              for (d, m), i in modelos.items()},
                     incompletos=dados["incompletos"])
    (saida / "LEIAME.md").write_text(LEIAME, encoding="utf-8")
    (saida / "manifesto.json").write_text(json.dumps(manifesto, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return manifesto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--permitir-parcial", action="store_true",
                        help="exporta mesmo sem o D_FA completo ou com dispositivos misturados (apaga a pasta de saída)")
    args = parser.parse_args()
    manifesto = exportar(args.outputs, args.saida, args.permitir_parcial)
    print(f"{len(manifesto['modelos'])} (dataset, modelo) exportados em {args.saida}")
    if manifesto["incompletos"]:
        print(f"fora por estarem incompletos: {', '.join(manifesto['incompletos'])}")


if __name__ == "__main__":
    main()
