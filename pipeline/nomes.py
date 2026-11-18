"""Nomes dos 17 modelos do TCC e a ponte com os nomes antigos (padronização de 17 e 18/11/2026).

Os identificadores usados para ler e gravar resultados (pastas, arquivos, colunas `modelo`) são os nomes novos.
Os nomes antigos continuam reconhecidos para que as pastas de uma rodada feita antes da renomeação (por exemplo,
`outputs/07_DTSF/resultados/SZ/D_FA/`) sigam sendo lidas e retomadas sem copiar nem treinar nada de novo.

Todos os identificadores estão em inglês.
Etapa 07 (DTSF): DTSF_FA (principal, com Frequency Attention), DTSF (sem FA), DTSF_Lite (versão compacta) e as
ablações do DTSF_Lite: DTSF_Lite_NoMixup, DTSF_Lite_SpectralOnly (só o ramo espectral) e DTSF_Lite_TemporalOnly
(só o ramo temporal). Etapa 08: LR, RF e KNN × {Temporal, Spectral, Combined}. Etapa 06: LR_PSD_Absolute e
LR_PSD_Relative.

Aqui `DTSF` é o nome do modelo sem FA com as dimensões do artigo. A arquitetura em geral (a família dos seis modelos
da etapa 07) continua sendo chamada de "DTSF" no texto; no código o modelo é sempre escrito `DTSF`.
"""
from __future__ import annotations

from pathlib import Path

# nome antigo -> nome novo
NOVOS = {
    "D_FA": "DTSF_FA",
    "D": "DTSF",
    "D_pequeno": "DTSF_Lite",
    "D_sem_mixup": "DTSF_Lite_NoMixup",
    "D_sem_T": "DTSF_Lite_SpectralOnly",
    "D_sem_E": "DTSF_Lite_TemporalOnly",
    "LR_psd_absolutas": "LR_PSD_Absolute",
    "LR_psd_relativas": "LR_PSD_Relative",
}
# modalidade das características dos clássicos (chaves internas, em português) -> sufixo do nome do modelo
MODALIDADES_EN = {"temporal": "Temporal", "espectral": "Spectral", "combinado": "Combined"}
MODALIDADE_DE_NOME = {en: pt for pt, en in MODALIDADES_EN.items()}
for _c in ("LR", "RF", "KNN"):
    for _pt, _en in MODALIDADES_EN.items():
        NOVOS[f"{_c}_{_pt}"] = f"{_c}_{_en}"          # os clássicos antes tinham o sufixo em português
ANTIGOS = {novo: antigo for antigo, novo in NOVOS.items()}

# conjunto de características da LR de 76 bandas (coluna `features` do resumo da etapa 06) -> nome do modelo
LR_PSD = {"absolutas": "LR_PSD_Absolute", "relativas": "LR_PSD_Relative"}


def nome_classico(classificador: str, modalidade: str) -> str:
    """O nome de um baseline clássico: `nome_classico("LR", "espectral")` → "LR_Spectral"."""
    return f"{classificador}_{MODALIDADES_EN[modalidade]}"


def nome_novo(nome: str) -> str:
    """O nome atual de um modelo, dado o nome novo ou o antigo (os 9 clássicos não mudaram)."""
    return NOVOS.get(nome, nome)


def resolver_pasta(pasta: Path) -> Path:
    """A pasta de um modelo: a de nome novo se existir; senão a de nome antigo, de uma rodada anterior à renomeação."""
    pasta = Path(pasta)
    antigo = ANTIGOS.get(pasta.name)
    if not pasta.exists() and antigo is not None and (pasta.parent / antigo).exists():
        return pasta.parent / antigo
    return pasta


def resolver_arquivo(arquivo: Path) -> Path:
    """O mesmo para um arquivo `<modelo>.csv` (por exemplo, `previsoes/SZ/DTSF_FA.csv`)."""
    arquivo = Path(arquivo)
    antigo = ANTIGOS.get(arquivo.stem)
    if not arquivo.exists() and antigo is not None and (arquivo.parent / (antigo + arquivo.suffix)).exists():
        return arquivo.parent / (antigo + arquivo.suffix)
    return arquivo
