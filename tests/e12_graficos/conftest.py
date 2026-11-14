"""Uma pasta outputs/10_resultados sintética, com previsões, permutações e resumos coerentes, e os números dela."""
import functools
import json

import numpy as np
import pandas as pd
import pytest

from pipeline.e12_graficos import numeros as n

PESSOAS = 16


def _previsoes(semente: int, sinal: float, chave: str = "repeticao", n_chave: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    y = np.r_[np.ones(PESSOAS // 2), np.zeros(PESSOAS // 2)].astype(int)
    pessoa = rng.normal(size=PESSOAS)
    linhas = []
    for r in range(n_chave):
        ordem = np.random.default_rng(100 + r).permutation(PESSOAS // 2)
        yy = rng.permutation(y) if chave == "permutacao" else y
        for i in range(PESSOAS):
            prob = 1 / (1 + np.exp(-(sinal * (2 * yy[i] - 1) + pessoa[i] + 0.4 * rng.normal())))
            linhas.append({chave: r, "fold": int(ordem[i % (PESSOAS // 2)] % 4), "subject_uid": f"S{i:02d}",
                           "y_true": int(yy[i]), "y_pred": int(prob >= 0.5), "y_prob": prob})
    return pd.DataFrame(linhas)


def _resultados_sinteticos(raiz, comparar) -> None:
    previsoes = {}
    for k, d in enumerate(n.DATASETS):
        (raiz / "previsoes" / d).mkdir(parents=True)
        (raiz / "permutacoes" / d).mkdir(parents=True)
        for i, m in enumerate(n.MODELOS):
            p = _previsoes(10 * k + i, 0.4 + 0.05 * (i % 5))
            p.to_csv(raiz / "previsoes" / d / f"{m}.csv", index=False)
            previsoes[(d, m)] = n.ler_previsoes(raiz / "previsoes" / d / f"{m}.csv")
        _previsoes(500 + k, 0.0, "permutacao", 25).to_csv(raiz / "permutacoes" / d / "D_FA.csv", index=False)

    md07, md08, md06 = ["# 07"], ["# 08"], ["# 06"]
    for d in n.DATASETS:
        md07 += [f"## {d}", ""]
        md08 += [f"## {d}", ""]
        for m in n.MODELOS:
            r = n.avaliar(previsoes[(d, m)], semente=n.SEMENTE)
            linha = f"{r['auc']:.3f} [{r['ic_baixo']:.3f}, {r['ic_alto']:.3f}]"
            if m.startswith("LR_psd"):
                md06.append(f"| {d} | {m.split('_')[-1]} | {linha} | x |")
            else:
                (md07 if m in n.TIPOS_DTSF else md08).append(f"| {m} | {linha} | x |")
    for d in n.DATASETS:
        for a, b in n.comparacoes():
            c = comparar(previsoes[(d, a)], previsoes[(d, b)])
            texto = f"{c['delta']:+.3f} [{c['ic'][0]:+.3f}, {c['ic'][1]:+.3f}] | {c['p']:.4f} | x |"
            if b in n.CLASSICOS:
                md08.append(f"| {d} | {b} | {texto}")
            else:
                md07.append(f"| {d} | {a} − {b} | {texto}")
    (raiz / "resumos").mkdir()
    for nome_md, linhas in (("07_DTSF", md07), ("08_classicos", md08), ("06_LR_psd", md06)):
        (raiz / "resumos" / f"{nome_md}.md").write_text("\n".join(linhas) + "\n", encoding="utf-8")


@pytest.fixture(scope="session")
def numeros_sinteticos(tmp_path_factory):
    raiz = tmp_path_factory.mktemp("resultados")
    comparar = functools.partial(n.comparacao_pareada, n=200)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(n, "comparacao_pareada", comparar)
        _resultados_sinteticos(raiz, comparar)
        return n.calcular(raiz)




@pytest.fixture(scope="session")
def numeros_json(numeros_sinteticos):
    """Os números como saem de numeros.json (tuplas viram listas)."""
    return json.loads(json.dumps(numeros_sinteticos))
