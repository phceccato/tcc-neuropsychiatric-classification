"""Experimento 3: explicações SHAP do DTSF (expected gradients) nas pessoas de teste da repetição 0.

As previsões da etapa 07 não guardam os pesos. Aqui o D é retreinado nos 5 folds da repetição 0 de
cada dataset, com as mesmas partições e sementes de pipeline/e07_DTSF/rodar.py, e os pesos ficam
gravados. Para cada pessoa de teste:
- a probabilidade de cada janela aceita sem overlap (as mesmas que formam a previsão da pessoa);
- os valores SHAP do logit em relação às duas entradas, calculados por expected gradients
  (ERION et al., 2021), o algoritmo do `shap.GradientExplainer`:
  φ = E[(x − x') ⊙ ∇f(x' + α(x − x'))], com x' sorteado do fundo e α ~ U(0, 1).
  O fundo tem `FUNDO_POR_CLASSE` janelas aceitas de treino de cada classe, sorteadas entre as pessoas
  de treino do fold, e cada janela usa `AMOSTRAS` pares (x', α). A soma dos φ aproxima
  f(x) − E[f(x')] (completude), conferida no resumo.

Os φ não são gravados inteiros. Por pessoa, a média nas janelas de:
- ramo temporal [19 canais, 5 entradas (sinal base, δ, θ, α, β), 500 amostras]: soma de |φ| e de φ no
  tempo por (canal, entrada), e soma de |φ| por amostra;
- ramo espectral [19 canais, 29 frequências (2–30 Hz), 16 quadros]: soma de |φ| e de φ nos quadros por
  (canal, frequência), e soma de |φ| por quadro.
Por janela: o logit, a probabilidade e a soma de φ de cada ramo.

Grava em outputs/09_investigacao/explicacao/:
- modelos/<dataset>/f<fold>.pt e .json (época escolhida);
- janelas/<dataset>/f<fold>.csv (uma linha por janela de teste);
- atribuicoes/<dataset>/f<fold>.npz (agregados por pessoa).
Tarefas já gravadas são puladas. Com --dispositivo cuda, só o treino do D roda na GPU; o modelo volta para
a CPU antes do SHAP, que continua na CPU. O .json de cada modelo registra o dispositivo do treino. O resumo e as figuras ficam em pipeline/e09_investigacao/relatorio_explicacao.py.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from pipeline.e07_DTSF.dados import janelas_avaliacao
from pipeline.e07_DTSF.redes_dtsf import DTSF
from pipeline.e07_DTSF.rodar import (SEMENTE, _fonte, _iniciar, argumento_dispositivo, conferir_dispositivo,
                                     contexto_processos)
from pipeline.e07_DTSF.treino import treinar

MODELO, REPETICAO = "D", 0
FUNDO_POR_CLASSE = 100
AMOSTRAS = 50
JANELAS_POR_PASSO = 4
SEMENTE_SHAP = 1108


def gradientes_esperados(modelo: torch.nn.Module, x_t: torch.Tensor, x_e: torch.Tensor, fundo_t: torch.Tensor,
                         fundo_e: torch.Tensor, amostras: int, rng: np.random.Generator,
                         janelas_por_passo: int = JANELAS_POR_PASSO) -> tuple[torch.Tensor, torch.Tensor]:
    """φ de cada janela [n, ...] para as duas entradas, por expected gradients sobre o fundo [m, ...]."""
    phi_t, phi_e = torch.zeros_like(x_t), torch.zeros_like(x_e)
    for i in range(0, len(x_t), janelas_por_passo):
        bt, be = x_t[i:i + janelas_por_passo], x_e[i:i + janelas_por_passo]
        k = len(bt)
        idx = torch.from_numpy(rng.integers(0, len(fundo_t), (k, amostras)))
        alfa = torch.from_numpy(rng.random((k, amostras))).float()
        rt, re = fundo_t[idx], fundo_e[idx]                       # [k, amostras, ...]
        dt, de = bt[:, None] - rt, be[:, None] - re
        pt = (rt + alfa.view(k, amostras, 1, 1, 1) * dt).reshape(-1, *bt.shape[1:]).requires_grad_(True)
        pe = (re + alfa.view(k, amostras, 1, 1, 1) * de).reshape(-1, *be.shape[1:]).requires_grad_(True)
        gt, ge = torch.autograd.grad(modelo(pt, pe).sum(), (pt, pe))
        phi_t[i:i + k] = (dt * gt.view_as(dt)).mean(dim=1)
        phi_e[i:i + k] = (de * ge.view_as(de)).mean(dim=1)
    return phi_t, phi_e


def agregar(phi_t: torch.Tensor, phi_e: torch.Tensor) -> dict[str, np.ndarray]:
    """Médias nas janelas de uma pessoa (ver a docstring do módulo)."""
    at, ae = phi_t.abs(), phi_e.abs()
    return dict(t_abs_canal_entrada=at.sum(-1).mean(0), t_sinal_canal_entrada=phi_t.sum(-1).mean(0),
                t_abs_tempo=at.sum((1, 2)).mean(0), e_abs_canal_freq=ae.sum(-1).mean(0),
                e_sinal_canal_freq=phi_e.sum(-1).mean(0), e_abs_quadro=ae.sum((1, 2)).mean(0))


def sortear_fundo(fonte, treino: list[str], y: pd.Series, por_classe: int, rng: np.random.Generator) -> np.ndarray:
    """`por_classe` janelas aceitas de treino de cada classe (pessoas sorteadas, janelas sem repetição)."""
    linhas = []
    for classe in (0, 1):
        aceitas = fonte.indice[fonte.indice["subject_uid"].isin([u for u in treino if y[u] == classe])
                               & ~fonte.indice["rejeitada"]]["linha"].to_numpy()
        linhas.append(rng.choice(aceitas, size=min(por_classe, len(aceitas)), replace=False))
    return np.concatenate(linhas)


def rodar_tarefa(dataset: str, fold: int, folds_csv: str, saida: str, dispositivo: str = "cpu") -> str:
    inicio = time.time()
    folds = pd.read_csv(folds_csv)
    g = folds[(folds["dataset"] == dataset) & (folds["repeticao"] == REPETICAO) & (folds["fold"] == fold)]
    y = (g.set_index("subject_uid")["grupo"] != "HC").astype(int)
    papel = g.groupby("papel")["subject_uid"].apply(list)
    fonte = _fonte(dataset, folds)
    saida = Path(saida)

    caminho_modelo = saida / "modelos" / dataset / f"f{fold}.pt"
    caminho_modelo.parent.mkdir(parents=True, exist_ok=True)
    if caminho_modelo.exists():
        modelo = DTSF(MODELO)
        modelo.load_state_dict(torch.load(caminho_modelo))
        modelo.eval()
    else:
        resultado = treinar(MODELO, fonte, papel["treino"], papel["validacao"], y, semente=SEMENTE + 1000 * REPETICAO + fold,
                            dispositivo=dispositivo)
        modelo = resultado.modelo.cpu()
        torch.save(modelo.state_dict(), caminho_modelo)
        caminho_modelo.with_suffix(".json").write_text(json.dumps(dict(melhor_epoca=resultado.melhor_epoca,
                                                                        dispositivo=dispositivo)))
    for p in modelo.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng([SEMENTE_SHAP, fold, dataset == "MDD"])
    fundo_t, fundo_e = fonte.lote(sortear_fundo(fonte, papel["treino"], y, FUNDO_POR_CLASSE, rng))
    with torch.no_grad():
        logit_fundo = float(modelo(fundo_t, fundo_e).mean())

    por_pessoa = {u: gp for u, gp in fonte.indice.groupby("subject_uid")}
    janelas, agregados = [], {}
    for uid in papel["teste"]:
        linhas = janelas_avaliacao(por_pessoa[uid])
        x_t, x_e = fonte.lote(linhas)
        with torch.no_grad():
            logit = modelo(x_t, x_e)
        phi_t, phi_e = gradientes_esperados(modelo, x_t, x_e, fundo_t, fundo_e, AMOSTRAS, rng)
        agregados[uid] = {k: v.numpy() for k, v in agregar(phi_t, phi_e).items()}
        inicio_s = por_pessoa[uid].set_index("linha").loc[linhas, "inicio_s"].to_numpy()
        janelas.append(pd.DataFrame(dict(subject_uid=uid, y=int(y[uid]), linha=linhas, inicio_s=inicio_s,
                                         logit=logit.numpy(), prob=torch.sigmoid(logit).numpy(),
                                         phi_temporal=phi_t.sum((1, 2, 3)).numpy(), phi_espectral=phi_e.sum((1, 2, 3)).numpy(),
                                         logit_fundo=logit_fundo)))

    for pasta in ("janelas", "atribuicoes"):
        (saida / pasta / dataset).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(saida / "atribuicoes" / dataset / f"f{fold}.npz", uids=np.array(list(agregados)),
                        **{k: np.stack([a[k] for a in agregados.values()]) for k in next(iter(agregados.values()))})
    pd.concat(janelas).to_csv(saida / "janelas" / dataset / f"f{fold}.csv", index=False)  # gravado por último: marca a tarefa como feita
    return f"{dataset} f{fold}: {sum(len(j) for j in janelas)} janelas, {time.time() - inicio:.0f} s"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folds", default="outputs/05_folds/folds.csv")
    parser.add_argument("--saida", default="outputs/09_investigacao/explicacao")
    parser.add_argument("--datasets", nargs="+", default=["SZ", "MDD"])
    parser.add_argument("--processos", type=int, default=10)
    argumento_dispositivo(parser)
    args = parser.parse_args()

    tarefas = [(d, f) for d in args.datasets for f in range(5)
               if not (Path(args.saida) / "janelas" / d / f"f{f}.csv").exists()]
    conferir_dispositivo(parser, args.dispositivo, [Path(args.saida) / "modelos" / d for d in args.datasets])
    print(f"{len(tarefas)} tarefas pendentes ({args.dispositivo})", flush=True)
    with ProcessPoolExecutor(args.processos, initializer=_iniciar,
                             mp_context=contexto_processos(args.dispositivo)) as executor:
        futuros = [executor.submit(rodar_tarefa, d, f, args.folds, args.saida, args.dispositivo) for d, f in tarefas]
        for i, futuro in enumerate(as_completed(futuros), 1):
            print(f"[{i}/{len(tarefas)}] {futuro.result()}", flush=True)


if __name__ == "__main__":
    main()
