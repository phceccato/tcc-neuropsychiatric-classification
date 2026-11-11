"""Treino das variantes da etapa 12 (pipeline/e12_FA_SDKA/redes.py) num fold.

O laço é o do D (pipeline/e07_DTSF/treino.py): mesmo sorteio de janelas, Mixup, Adam, BCE com peso
de classe, early stopping na BCE das pessoas de validação e previsão pela média das janelas. As
variantes com SDKA somam à perda de cada lote α · SMMD (XU et al., 2026, Eqs. 10–14):

- fonte: as janelas do lote **antes** do Mixup, com o rótulo verdadeiro como peso de classe;
- alvo: 64 janelas aceitas de **uma** pessoa de teste (a de cada lote segue uma ordem sorteada das
  pessoas de teste, repetida enquanto houver lotes), como o sujeito-alvo único do artigo; o peso de
  classe é a probabilidade prevista pela própria rede (sem gradiente), nunca o rótulo;
- fonte e alvo passam juntos pela rede, em modo de treino, e o SMMD é calculado na representação do
  CLS. A BatchNorm vê, então, também as janelas do alvo;
- α = 0,1 (o valor do artigo no MODMA), fixo, sem ajuste.

O treino com SDKA é **transdutivo**: usa o EEG das pessoas de teste, sem os rótulos. A validação e o
early stopping não mudam. Os sorteios do alvo vêm de um gerador próprio (`gerador_alvo`), para que o
sorteio das janelas de treino e o Mixup sejam iguais aos do D na mesma semente.

Com `dispositivo="cuda"`, o modelo e os lotes vão para a GPU, como no D (pipeline/e07_DTSF/treino.py); os
sorteios, o Mixup e a STFT continuam na CPU.
"""
from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pipeline.e07_DTSF.dados import sortear
from pipeline.e07_DTSF.dados_dtsf import estatisticas_stft
from pipeline.e07_DTSF.redes_dtsf import misturar, sortear_mixup
from pipeline.e07_DTSF.treino import (JANELAS_POR_PESSOA, LR, MAX_EPOCAS, PACIENCIA, TAMANHO_LOTE, WEIGHT_DECAY,
                                      Resultado, _para, _perda_validacao, gerador_mixup, prever)
from pipeline.e12_FA_SDKA.redes import DTSF12

ALFA_SMMD = 0.1
LOTE_ALVO = 64
N_NUCLEOS, MULT_NUCLEO = 5, 2.0

__all__ = ["ALFA_SMMD", "LOTE_ALVO", "gerador_alvo", "pesos_classe", "nucleo_gaussiano", "smmd", "treinar", "prever"]


def gerador_alvo(semente: int) -> np.random.Generator:
    """Gerador dos lotes do alvo: fluxo diferente do sorteio de treino e do Mixup."""
    return np.random.default_rng([semente, 2])


def pesos_classe(p_positivo: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Pesos w_i^k da Eq. 12 para k ∈ {controle, paciente}: [n, 2], cada coluna somando 1, e a máscara
    das classes presentes (soma > 0). `p_positivo` é o rótulo (fonte) ou a probabilidade (alvo)."""
    y = torch.stack([1 - p_positivo, p_positivo], dim=1)
    soma = y.sum(0)
    presente = soma > 1e-8
    return y / soma.clamp_min(1e-8), presente


def nucleo_gaussiano(a: torch.Tensor, b: torch.Tensor, n_nucleos: int = N_NUCLEOS,
                     mult: float = MULT_NUCLEO) -> torch.Tensor:
    """Soma de n núcleos gaussianos sobre [a; b] (como no DAN/DSAN): larguras em progressão geométrica
    em torno da distância quadrática média entre os pares (sem gradiente na largura)."""
    z = torch.cat([a, b])
    d2 = torch.cdist(z, z).pow(2)
    n = len(z)
    largura = d2.detach().sum() / max(n * n - n, 1)
    largura = largura.clamp_min(1e-8) / mult ** (n_nucleos // 2)
    return sum(torch.exp(-d2 / (largura * mult ** i)) for i in range(n_nucleos))


def smmd(z_fonte: torch.Tensor, y_fonte: torch.Tensor, z_alvo: torch.Tensor, p_alvo: torch.Tensor) -> torch.Tensor:
    """Eq. 13: média, nas classes presentes na fonte, de ‖Σ w^s φ(z^s) − Σ w^t φ(z^t)‖² no RKHS."""
    w_s, presente_s = pesos_classe(y_fonte)
    w_t, presente_t = pesos_classe(p_alvo)
    k = nucleo_gaussiano(z_fonte, z_alvo)
    n = len(z_fonte)
    k_ss, k_tt, k_st = k[:n, :n], k[n:, n:], k[:n, n:]
    termos = (w_s * (k_ss @ w_s)).sum(0) + (w_t * (k_tt @ w_t)).sum(0) - 2 * (w_s * (k_st @ w_t)).sum(0)
    presente = presente_s & presente_t
    return termos[presente].sum() / presente.sum().clamp_min(1)


def _lote_alvo(por_pessoa: dict, alvo: list[str], i_lote: int, ordem: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uid = alvo[ordem[i_lote % len(ordem)]]
    return sortear(por_pessoa[uid], LOTE_ALVO, rng)


def treinar(tipo: str, fonte, treino: list[str], validacao: list[str], alvo: list[str], y: pd.Series, semente: int,
            max_epocas: int = MAX_EPOCAS, paciencia: int = PACIENCIA, alfa_smmd: float = ALFA_SMMD,
            dispositivo: str = "cpu") -> Resultado:
    """Treina uma variante; `alvo` são as pessoas de teste (só os sinais são usados, e só com SDKA)."""
    dispositivo = torch.device(dispositivo)
    if dispositivo.type == "cuda":
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    por_pessoa = {uid: g for uid, g in fonte.indice.groupby("subject_uid")}
    modelo = DTSF12(tipo)
    rng_mixup, rng_alvo = gerador_mixup(semente), gerador_alvo(semente)

    linhas_treino = np.concatenate([por_pessoa[u].loc[~por_pessoa[u]["rejeitada"], "linha"].to_numpy() for u in treino])
    modelo.definir_normalizacao(*estatisticas_stft(fonte, linhas_treino))
    modelo.to(dispositivo)

    n_pos = int(sum(y[u] for u in treino))
    peso_positivo = torch.tensor((len(treino) - n_pos) / n_pos).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    val = {u: por_pessoa[u] for u in validacao}

    melhor, melhor_epoca, melhor_estado, historico = np.inf, 0, None, []
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        sorteadas = [sortear(por_pessoa[u], JANELAS_POR_PESSOA, rng) for u in treino]
        linhas = np.concatenate(sorteadas)
        rotulos = np.concatenate([np.full(len(s), float(y[u])) for s, u in zip(sorteadas, treino)])
        ordem = rng.permutation(len(linhas))
        linhas, rotulos = linhas[ordem], rotulos[ordem]
        ordem_alvo = rng_alvo.permutation(len(alvo)) if modelo.usa_sdka else None
        perdas, perdas_smmd = [], []
        for i_lote, i in enumerate(range(0, len(linhas), TAMANHO_LOTE)):
            x_t, x_e = fonte.lote(linhas[i:i + TAMANHO_LOTE])
            rotulo = torch.from_numpy(rotulos[i:i + TAMANHO_LOTE]).float()
            xm_t, xm_e, alvo_mix = _para(dispositivo, *misturar(x_t, x_e, rotulo, *sortear_mixup(rng_mixup, len(rotulo))))
            perda = F.binary_cross_entropy_with_logits(modelo(xm_t, xm_e), alvo_mix, pos_weight=peso_positivo)
            if modelo.usa_sdka:
                a_t, a_e = fonte.lote(_lote_alvo(por_pessoa, alvo, i_lote, ordem_alvo, rng_alvo))
                c_t, c_e, rotulo = _para(dispositivo, torch.cat([x_t, a_t]), torch.cat([x_e, a_e]), rotulo)
                z = modelo.representacao(c_t, c_e)
                z_fonte, z_alvo = z[:len(rotulo)], z[len(rotulo):]
                p_alvo = torch.sigmoid(modelo.saida(z_alvo).squeeze(-1)).detach()
                perda_smmd = smmd(z_fonte, rotulo, z_alvo, p_alvo)
                perda = perda + alfa_smmd * perda_smmd
                perdas_smmd.append(perda_smmd.item())
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())
        perda_val = _perda_validacao(modelo, fonte, val, y, peso_positivo)
        historico.append(dict(epoca=epoca, perda_treino=float(np.mean(perdas)), perda_validacao=perda_val,
                              smmd=float(np.mean(perdas_smmd)) if perdas_smmd else None))
        if perda_val < melhor:
            melhor, melhor_epoca, melhor_estado = perda_val, epoca, copy.deepcopy(modelo.state_dict())
        elif epoca - melhor_epoca >= paciencia:
            break

    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    return Resultado(modelo, melhor_epoca, historico)
