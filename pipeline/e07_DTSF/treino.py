"""Treino e previsão do DTSF adaptado (pipeline/e07_DTSF/redes_dtsf.py) num fold.

- a cada época, 60 janelas aceitas sorteadas de cada pessoa de treino (semente fixa);
- Adam, lr 1e-3, weight decay 1e-4, lotes de 64, BCE com peso de classe (pessoas);
- até 100 épocas, early stopping (paciência 10) na loss das pessoas de validação, com as
  janelas aceitas da grade sem overlap; volta aos pesos da melhor época;
- previsão da pessoa: média das probabilidades das janelas aceitas sem overlap.

A STFT é normalizada com média e desvio por (canal, frequência) das janelas aceitas das pessoas de
treino. Com Mixup, cada lote de treino é misturado com λ e permutação de um gerador próprio
(`gerador_mixup`, derivado da semente da tarefa e independente do gerador do sorteio), para que o
sorteio das janelas seja igual ao do modelo sem Mixup. Validação e teste nunca são misturados.

Dispositivo: por padrão tudo roda na CPU (é como as rodadas registradas foram feitas). Com
`dispositivo="cuda"`, o modelo e os lotes vão para a GPU; o sorteio das janelas, o Mixup e a STFT continuam
na CPU, com os mesmos geradores, e só depois o lote é copiado para a GPU. A cuDNN fica em modo determinístico,
mas a GPU não reproduz bit a bit as contas da CPU: os resultados de uma rodada em GPU não devem ser misturados
com os de uma rodada em CPU.

`fonte` precisa ter `indice` (subject_uid, linha, sem_overlap, rejeitada) e `lote(linhas) -> (x_t, x_e)`
(`dados_dtsf.FonteDTSF`).
"""
from __future__ import annotations

import copy
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

from pipeline.e07_DTSF.dados import janelas_avaliacao, sortear
from pipeline.e07_DTSF.dados_dtsf import estatisticas_stft
from pipeline.e07_DTSF.redes_dtsf import ALFA_MIXUP, DROPOUT, DROPOUT_ENCODER, DTSF, misturar, sortear_mixup

MAX_EPOCAS, PACIENCIA = 100, 10
JANELAS_POR_PESSOA = 60
TAMANHO_LOTE = 64
LR, WEIGHT_DECAY = 1e-3, 1e-4
LOTE_INFERENCIA = 256


@dataclass(frozen=True)
class ConfigTreino:
    """Hiperparâmetros do DTSF: os do treino, iguais para todos os modelos da etapa 07, e as dimensões da rede.
    Dimensões None (o padrão) usam as do modelo (`MODELOS` em redes_dtsf.py)."""
    lr: float = LR
    weight_decay: float = WEIGHT_DECAY
    tamanho_lote: int = TAMANHO_LOTE
    paciencia: int = PACIENCIA
    max_epocas: int = MAX_EPOCAS
    dropout: float = DROPOUT
    dropout_encoder: float = DROPOUT_ENCODER
    alfa_mixup: float = ALFA_MIXUP
    filtros: int | None = None
    oculto: int | None = None
    cabecas: int | None = None
    camadas: int | None = None


PADRAO = ConfigTreino()


def gerador_mixup(semente: int) -> np.random.Generator:
    """Gerador do Mixup: derivado da semente da tarefa, com fluxo diferente do gerador do sorteio."""
    return np.random.default_rng([semente, 1])


@dataclass
class Resultado:
    modelo: DTSF
    melhor_epoca: int
    historico: list[dict] = field(default_factory=list)


def dispositivo_do(modelo: torch.nn.Module) -> torch.device:
    return next(modelo.parameters()).device


def _para(dispositivo: torch.device, *tensores: torch.Tensor) -> tuple[torch.Tensor, ...]:
    return tuple(t.to(dispositivo, non_blocking=True) for t in tensores)


def _logits(modelo: DTSF, fonte, linhas: np.ndarray) -> torch.Tensor:
    """Logits das janelas, no dispositivo do modelo; devolvidos na CPU."""
    dispositivo = dispositivo_do(modelo)
    partes = [modelo(*_para(dispositivo, *fonte.lote(linhas[i:i + LOTE_INFERENCIA]))).cpu()
              for i in range(0, len(linhas), LOTE_INFERENCIA)]
    return torch.cat(partes) if partes else torch.empty(0)


def _perda_validacao(modelo: DTSF, fonte, por_pessoa: dict, y: pd.Series, peso_positivo: torch.Tensor) -> float:
    """Média, entre as pessoas, da BCE média das suas janelas (cada pessoa pesa igual)."""
    modelo.eval()
    perdas = []
    with torch.no_grad():
        for uid, g in por_pessoa.items():
            logits = _logits(modelo, fonte, janelas_avaliacao(g))
            alvo = torch.full_like(logits, float(y[uid]))
            perdas.append(F.binary_cross_entropy_with_logits(logits, alvo, pos_weight=peso_positivo.cpu()).item())
    return float(np.mean(perdas))


def treinar(tipo: str, fonte, treino: list[str], validacao: list[str], y: pd.Series, semente: int,
            max_epocas: int | None = None, paciencia: int | None = None, config: ConfigTreino = PADRAO,
            dispositivo: str = "cpu", construir: Callable[[], DTSF] | None = None) -> Resultado:
    """Treina o DTSF `tipo` com `config`; `construir`, se dado, cria a rede no lugar de `DTSF(tipo, …)`
    (as etapas 12 e 14 usam para as suas variantes). A rede é criada depois da semente, como o D."""
    max_epocas = config.max_epocas if max_epocas is None else max_epocas
    paciencia = config.paciencia if paciencia is None else paciencia
    dispositivo = torch.device(dispositivo)
    if dispositivo.type == "cuda":
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.manual_seed(semente)
    rng = np.random.default_rng(semente)
    por_pessoa = {uid: g for uid, g in fonte.indice.groupby("subject_uid")}
    modelo = construir() if construir is not None else \
        DTSF(tipo, dropout=config.dropout, dropout_encoder=config.dropout_encoder, filtros=config.filtros,
             oculto=config.oculto, cabecas=config.cabecas, camadas=config.camadas)
    rng_mixup = gerador_mixup(semente)

    linhas_treino = np.concatenate([por_pessoa[u].loc[~por_pessoa[u]["rejeitada"], "linha"].to_numpy() for u in treino])
    modelo.definir_normalizacao(*estatisticas_stft(fonte, linhas_treino))
    modelo.to(dispositivo)

    n_pos = int(sum(y[u] for u in treino))
    peso_positivo = torch.tensor((len(treino) - n_pos) / n_pos).to(dispositivo)
    otimizador = torch.optim.Adam(modelo.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    val = {u: por_pessoa[u] for u in validacao}

    melhor, melhor_epoca, melhor_estado, historico = np.inf, 0, None, []
    for epoca in range(1, max_epocas + 1):
        modelo.train()
        sorteadas = [sortear(por_pessoa[u], JANELAS_POR_PESSOA, rng) for u in treino]
        linhas = np.concatenate(sorteadas)
        rotulos = np.concatenate([np.full(len(s), float(y[u])) for s, u in zip(sorteadas, treino)])
        ordem = rng.permutation(len(linhas))
        linhas, rotulos = linhas[ordem], rotulos[ordem]
        perdas = []
        for i in range(0, len(linhas), config.tamanho_lote):
            x_t, x_e = fonte.lote(linhas[i:i + config.tamanho_lote])
            alvo = torch.from_numpy(rotulos[i:i + config.tamanho_lote]).float()
            if modelo.mixup:
                x_t, x_e, alvo = misturar(x_t, x_e, alvo, *sortear_mixup(rng_mixup, len(alvo), config.alfa_mixup))
            x_t, x_e, alvo = _para(dispositivo, x_t, x_e, alvo)
            perda = modelo.perda_treino(x_t, x_e, alvo, peso_positivo)
            otimizador.zero_grad()
            perda.backward()
            otimizador.step()
            perdas.append(perda.item())
        perda_val = _perda_validacao(modelo, fonte, val, y, peso_positivo)
        historico.append(dict(epoca=epoca, perda_treino=float(np.mean(perdas)), perda_validacao=perda_val))
        if perda_val < melhor:
            melhor, melhor_epoca, melhor_estado = perda_val, epoca, copy.deepcopy(modelo.state_dict())
        elif epoca - melhor_epoca >= paciencia:
            break

    modelo.load_state_dict(melhor_estado)
    modelo.eval()
    return Resultado(modelo, melhor_epoca, historico)


def prever(modelo: DTSF, fonte, uids: list[str]) -> dict[str, float]:
    """Probabilidade de paciente de cada pessoa: média das janelas aceitas sem overlap."""
    modelo.eval()
    por_pessoa = {uid: g for uid, g in fonte.indice.groupby("subject_uid")}
    with torch.no_grad():
        return {u: float(torch.sigmoid(_logits(modelo, fonte, janelas_avaliacao(por_pessoa[u]))).mean())
                for u in uids}
