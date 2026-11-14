"""Gráficos de resultados do TCC, em PDF vetorial (para `\\includegraphics`) e PNG de conferência.

Lê outputs/11_graficos/numeros.json (pipeline/e11_graficos/numeros.py, já conferido contra os resumos das
etapas) e grava em duas pastas de outputs/11_graficos/.

principais/ — os gráficos do texto:
  p1_roc_principais: curvas ROC por pessoa dos 6 modelos principais (D_FA, D, D_pequeno e LR, RF e KNN combinados);
  p2_auc_modelos: AUC por pessoa com IC de 95% de todos os modelos (etapas 06, 07 e 08);
  p3_metricas_principais: AUC, acurácia, precisão, recall e F1 dos 6 principais, média ± desvio nas 10 repetições;
  p4_sz_mdd: AUC por pessoa com IC de 95% de cada modelo em SZ e em MDD, lado a lado.

suplementares/ — o material suplementar:
  s1_roc_ablacoes: curvas ROC do D_pequeno e das suas 3 ablações;
  s2_confusao_sz, s2_confusao_mdd: matriz de confusão por pessoa (limiar 0,5) de todos os modelos;
  s3_temporal_espectral: AUC só com o ramo ou as características temporais, só com as espectrais e com as duas,
     no DTSF (ablações do D_pequeno) e nos clássicos;
  s4_absolutas_relativas: LR de 76 bandas com potência absoluta e relativa (AUC por pessoa e por repetição);
  s5_auc_repeticoes: AUC de cada uma das 10 repetições e a média, mesmos modelos de p2;
  s6_delta_hipoteses: ΔAUC = referência − comparado, com IC de 95% e p, agrupado por hipótese, com a decisão,
     e as comparações descritivas;
  s7_testes_estatisticos: distribuição nula da H1 e tabela com p, p de Holm e decisão de H1 a H4;
  s8_barras_ablacoes, s9_barras_classicos (coloridas): AUC por pessoa em barras do D_pequeno e das ablações e do
     D_FA e dos 9 baselines.

As figuras de pontos (p2, p4 sem cor de modelo, s5, s6) são pensadas para impressão: preto e cinza; os grupos se
distinguem por marcador (LR círculo, RF quadrado, KNN triângulo, LR de 76 bandas losango; D_FA círculo, D quadrado e
D_pequeno triângulo, preenchidos; as ablações, círculos vazados). As curvas ROC diferenciam os modelos também pelo
tracejado, para a leitura em P&B. As figuras de barras usam a paleta categórica de referência, validada para
daltonismo nos pares vizinhos; a cor nunca é a única pista (legenda e eixo nomeiam o modelo). Vírgula decimal nos
eixos e nos textos; largura de 16 cm, para caber em `\\linewidth` com margens ABNT.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SAIDA = Path("outputs/11_graficos")
PRINCIPAIS, SUPLEMENTARES = "principais", "suplementares"
LARGURA = 16 / 2.54
PRETO, CINZA, CINZA_CLARO = "#000000", "#6e6e6e", "#bdbdbd"
DATASETS = ("SZ", "MDD")
TITULOS = {"SZ": "Esquizofrenia × controles (14 + 14)", "MDD": "Depressão maior × controles (29 + 27)"}
CURTOS = {"SZ": "SZ × HC", "MDD": "MDD × HC"}

PRINCIPAL, REFERENCIA_ABLACOES = "D_FA", "D_pequeno"
CLASSIFICADORES = ("LR", "RF", "KNN")
MODELOS_PRINCIPAIS = ["D_FA", "D", "D_pequeno", "LR_combinado", "RF_combinado", "KNN_combinado"]
ABLACOES = ["D_sem_T", "D_sem_E", "D_sem_mixup"]
GRUPOS = [
    ("DTSF (etapa 07)", ["D_FA", "D", "D_pequeno"]),
    ("Ablações do D_pequeno (etapa 07)", ABLACOES),
    ("Clássicos temporais (etapa 08)", [f"{c}_temporal" for c in CLASSIFICADORES]),
    ("Clássicos espectrais (etapa 08)", [f"{c}_espectral" for c in CLASSIFICADORES]),
    ("Clássicos combinados (etapa 08)", [f"{c}_combinado" for c in CLASSIFICADORES]),
    ("LR de 76 bandas (etapa 06)", ["LR_psd_absolutas", "LR_psd_relativas"]),
]
NOMES = {"D_FA": "D_FA (principal)", "D": "D (artigo, sem FA)", "D_pequeno": "D_pequeno",
         "D_sem_T": "sem ramo temporal", "D_sem_E": "sem ramo espectral", "D_sem_mixup": "sem Mixup",
         "LR_psd_absolutas": "LR, potência absoluta", "LR_psd_relativas": "LR, potência relativa"}
NOMES |= {f"{c}_{m}": f"{c} {m}" for c in CLASSIFICADORES for m in ("temporal", "espectral", "combinado")}
NOMES_CURTOS = NOMES | {"D_FA": "D_FA", "D": "D", "D_sem_T": "D_peq. sem T", "D_sem_E": "D_peq. sem E",
                        "D_sem_mixup": "D_peq. sem Mixup", "LR_psd_absolutas": "LR absolutas",
                        "LR_psd_relativas": "LR relativas"}
METRICAS_FIG = [("auc", "AUC"), ("acuracia", "Acurácia"), ("precisao", "Precisão"), ("sensibilidade", "Recall"),
                ("f1", "F1")]


def nome(m: str) -> str:
    return NOMES[m]


def virgula(x: float, casas: int = 3, sinal: bool = False) -> str:
    texto = f"{x:+.{casas}f}" if sinal else f"{x:.{casas}f}"
    return texto.replace(".", ",").replace("-", "−")


def estilo(m: str) -> dict:
    """Marcador e cor de cada modelo (sem cor: preto para o DTSF, cinza para os demais)."""
    dtsf = {"D_FA": ("o", 5.5), "D": ("s", 4.5), "D_pequeno": ("^", 5)}
    if m in dtsf:
        marcador, tamanho = dtsf[m]
        return dict(marker=marcador, color=PRETO, mfc=PRETO, ms=tamanho)
    if m.startswith("D_"):
        return dict(marker="o", color=PRETO, mfc="white", ms=5)
    if m.startswith("LR_psd"):
        return dict(marker="D", color=CINZA, mfc="white" if "relativas" in m else CINZA, ms=4.5)
    marcador = {"LR": "o", "RF": "s", "KNN": "^"}[m.split("_")[0]]
    return dict(marker=marcador, color=CINZA, mfc=CINZA, ms=4.5 if marcador != "^" else 5)


def configurar() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["DejaVu Serif"], "mathtext.fontset": "dejavuserif",
        "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.axisbelow": True, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0, "ytick.major.size": 0,
        "pdf.fonttype": 42, "savefig.dpi": 300, "legend.fontsize": 7.5, "legend.frameon": False,
    })


def _eixo_virgula(ax, casas: int = 1, sinal: bool = False, eixo: str = "x") -> None:
    formato = FuncFormatter(lambda v, _: virgula(v, casas, sinal and abs(v) > 1e-9))
    (ax.xaxis if eixo == "x" else ax.yaxis).set_major_formatter(formato)


def _linhas(grupos) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Posições y (de cima para baixo) dos modelos e dos títulos de grupo."""
    modelos, titulos, y = [], [], 0.0
    for titulo, ms in grupos:
        titulos.append((titulo, y))
        y -= 1
        for m in ms:
            modelos.append((m, y))
            y -= 1
        y -= 0.4
    return modelos, titulos


def _rotulos_y(ax, modelos, titulos, rotulo=nome) -> None:
    ax.set_yticks([y for _, y in modelos] + [y for _, y in titulos])
    ax.set_yticklabels([rotulo(m) for m, _ in modelos] + [t for t, _ in titulos])
    for r in ax.get_yticklabels()[len(modelos):]:
        r.set_fontweight("bold")
    ax.set_ylim(min(y for _, y in modelos) - 0.7, 0.7)


def _gravar(fig, destino: Path, base: str) -> list[Path]:
    destino.mkdir(parents=True, exist_ok=True)
    caminhos = [destino / f"{base}.pdf", destino / f"{base}.png"]
    for c in caminhos:
        fig.savefig(c, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)
    return caminhos


# ---------------------------------------------------------------- curvas ROC

def curva_roc(escores: dict) -> tuple[np.ndarray, np.ndarray]:
    """1 − especificidade e sensibilidade por pessoa (escore médio nas 10 repetições)."""
    fpr, tpr, _ = roc_curve(escores["y"], escores["prob"])
    return fpr, tpr


ESTILO_ROC = {
    "D_FA": dict(color=PRETO, ls="-", lw=1.8), "D": dict(color=PRETO, ls="--", lw=1.1),
    "D_pequeno": dict(color=PRETO, ls=":", lw=1.3),
    "LR_combinado": dict(color="#eb6834", ls="-", lw=1.1), "RF_combinado": dict(color="#eb6834", ls="--", lw=1.1),
    "KNN_combinado": dict(color="#eb6834", ls=":", lw=1.3),
    "D_sem_T": dict(color="#2a78d6", ls="--", lw=1.1), "D_sem_E": dict(color="#2a78d6", ls=":", lw=1.3),
    "D_sem_mixup": dict(color="#2a78d6", ls="-.", lw=1.1),
}


def _fig_roc(numeros: dict, modelos: list[str], destino: Path, base: str, rotulos=None, estilos=None) -> list[Path]:
    rotulos, estilos = rotulos or {}, ESTILO_ROC | (estilos or {})
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 4.6), gridspec_kw=dict(wspace=0.25))
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        ax.plot([0, 1], [0, 1], color=CINZA_CLARO, lw=0.8, zorder=0)
        for m in modelos:
            fpr, tpr = curva_roc(r[m]["escores"])
            ax.plot(fpr, tpr, drawstyle="steps-post", label=f"{rotulos.get(m, nome(m))} ({virgula(r[m]['auc'])})",
                    **estilos[m])
        ax.set_xlim(-0.01, 1.01)
        ax.set_ylim(-0.01, 1.01)
        ax.set_aspect("equal")
        for eixo in ("x", "y"):
            _eixo_virgula(ax, eixo=eixo)
        ax.tick_params(axis="y", length=3, width=0.6)
        ax.grid(color="#ebebe8", lw=0.5)
        ax.set_xlabel("1 − especificidade")
        ax.set_ylabel("Sensibilidade" if d == "SZ" else "")
        ax.set_title(TITULOS[d], loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=6.6, handlelength=2.6,
                  title="AUC por pessoa", title_fontsize=6.6)
    return _gravar(fig, destino, base)


def fig_roc_principais(numeros: dict, saida: Path) -> list[Path]:
    return _fig_roc(numeros, MODELOS_PRINCIPAIS, saida / PRINCIPAIS, "p1_roc_principais")


def fig_roc_ablacoes(numeros: dict, saida: Path) -> list[Path]:
    rotulos = {m: f"D_pequeno {nome(m)}" for m in ABLACOES}
    return _fig_roc(numeros, [REFERENCIA_ABLACOES, *ABLACOES], saida / SUPLEMENTARES, "s1_roc_ablacoes", rotulos,
                    {REFERENCIA_ABLACOES: dict(color=PRETO, ls="-", lw=1.8)})


# ---------------------------------------------------------------- AUC de todos os modelos

def fig_auc_modelos(numeros: dict, saida: Path) -> list[Path]:
    modelos, titulos = _linhas(GRUPOS)
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 5.6), sharey=True, gridspec_kw=dict(wspace=0.32))
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        ax.axvline(0.5, color=CINZA_CLARO, lw=0.8, ls="--", zorder=0)
        ax.axvline(r[PRINCIPAL]["auc"], color=PRETO, lw=0.6, ls=":", zorder=0)
        for m, y in modelos:
            e = estilo(m)
            ax.plot(r[m]["ic"], [y, y], color=e["color"], lw=1.1, solid_capstyle="round", zorder=1)
            ax.plot(r[m]["auc"], y, ls="none", marker=e["marker"], ms=e["ms"], mfc=e["mfc"], mec=e["color"],
                    mew=0.9, zorder=2)
            ax.text(1.025, y, virgula(r[m]["auc"]), transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=7, fontweight="bold" if m == PRINCIPAL else "normal")
        ax.text(1.025, titulos[0][1], "AUC", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7, fontweight="bold")
        ax.set_xlim(0.3, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        _eixo_virgula(ax)
        ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
        ax.set_title(TITULOS[d], loc="left")
        ax.set_xlabel("AUC por pessoa e IC de 95%")
    _rotulos_y(eixos[0], modelos, titulos)
    eixos[1].tick_params(labelleft=False)
    return _gravar(fig, saida / PRINCIPAIS, "p2_auc_modelos")


# ---------------------------------------------------------------- métricas dos principais

CORES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")  # azul, laranja, água, amarelo (validadas em pares vizinhos)
CORES_6 = ("#1f4e8c", "#2a78d6", "#8fb8ea", "#eb6834", "#1baf7a", "#eda100")  # 3 DTSF em azuis, 3 clássicos
TEXTO, TEXTO_2, EIXO = "#0b0b0b", "#52514e", "#c9c8c2"


def _eixo_barras(ax, topo: float = 1.1) -> None:
    ax.axhline(0.5, color=TEXTO_2, lw=0.7, ls="--", zorder=1)
    ax.set_ylim(0, topo)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    _eixo_virgula(ax, 1, eixo="y")
    ax.tick_params(axis="y", length=3, width=0.6)
    ax.grid(axis="y", color="#ebebe8", lw=0.5, zorder=0)
    ax.spines["left"].set_color(EIXO)
    ax.spines["bottom"].set_color(EIXO)


def fig_metricas_principais(numeros: dict, saida: Path) -> list[Path]:
    fig, eixos = plt.subplots(2, 1, figsize=(LARGURA, 5.4), gridspec_kw=dict(hspace=0.45))
    n = len(MODELOS_PRINCIPAIS)
    largura = 0.8 / n
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        for i, m in enumerate(MODELOS_PRINCIPAIS):
            x = np.arange(len(METRICAS_FIG)) + (i - (n - 1) / 2) * largura
            medias = [r[m]["media_desvio"][k][0] for k, _ in METRICAS_FIG]
            desvios = [r[m]["media_desvio"][k][1] for k, _ in METRICAS_FIG]
            ax.bar(x, medias, width=largura * 0.92, color=CORES_6[i], zorder=2, label=nome(m))
            ax.errorbar(x, medias, yerr=desvios, fmt="none", ecolor=TEXTO, elinewidth=0.7, capsize=1.6, zorder=3)
        _eixo_barras(ax)
        ax.set_xticks(range(len(METRICAS_FIG)), [t for _, t in METRICAS_FIG])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("Média ± desvio (10 repetições)")
        ax.set_title(TITULOS[d], loc="left")
    eixos[0].legend(loc="upper center", bbox_to_anchor=(0.5, -1.62), ncol=3, handlelength=1.2, columnspacing=1.4)
    return _gravar(fig, saida / PRINCIPAIS, "p3_metricas_principais")


# ---------------------------------------------------------------- SZ × MDD

def fig_sz_mdd(numeros: dict, saida: Path) -> list[Path]:
    modelos, titulos = _linhas(GRUPOS)
    cor = {"SZ": "#eb6834", "MDD": "#2a78d6"}
    marcador = {"SZ": "s", "MDD": "o"}
    fig, ax = plt.subplots(figsize=(LARGURA * 0.75, 5.6))
    ax.axvline(0.5, color=CINZA_CLARO, lw=0.8, ls="--", zorder=0)
    for m, y in modelos:
        for d in DATASETS:
            r = numeros["modelos"][d][m]
            yy = y + (0.17 if d == "SZ" else -0.17)
            ax.plot(r["ic"], [yy, yy], color=cor[d], lw=1.0, solid_capstyle="round", zorder=1)
            ax.plot(r["auc"], yy, ls="none", marker=marcador[d], ms=4.2, mfc=cor[d], mec=cor[d], zorder=2)
    ax.set_xlim(0.3, 1.0)
    ax.xaxis.set_major_locator(MultipleLocator(0.1))
    _eixo_virgula(ax)
    ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
    ax.set_xlabel("AUC por pessoa e IC de 95%")
    _rotulos_y(ax, modelos, titulos)
    legenda = [Line2D([], [], color=cor[d], marker=marcador[d], ms=4.2, lw=1.0, label=TITULOS[d]) for d in DATASETS]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.35, -0.08), ncol=1)
    return _gravar(fig, saida / PRINCIPAIS, "p4_sz_mdd")


# ---------------------------------------------------------------- matrizes de confusão

def _matriz(ax, c: dict, titulo: str, rotulos: bool = True) -> None:
    matriz = np.array([[c["vn"], c["fp"]], [c["fn"], c["vp"]]])
    por_linha = matriz / matriz.sum(1, keepdims=True)
    ax.imshow(por_linha, cmap="Greys", vmin=0, vmax=1.25)
    for i in range(2):
        for j in range(2):
            escuro = por_linha[i, j] > 0.55
            ax.text(j, i, f"{matriz[i, j]}\n({virgula(100 * por_linha[i, j], 0)}%)", ha="center", va="center",
                    fontsize=6.2, color="white" if escuro else PRETO)
    ax.set_xticks([0, 1], ["C", "P"] if rotulos else ["", ""])
    ax.set_yticks([0, 1], ["C", "P"] if rotulos else ["", ""])
    ax.tick_params(axis="both", length=0, labelsize=6.5)
    for lado in ax.spines.values():
        lado.set_visible(False)
    ax.set_title(titulo, fontsize=6.8, pad=3)


def fig_confusao_todos(numeros: dict, saida: Path) -> list[Path]:
    modelos = [m for _, ms in GRUPOS for m in ms]
    colunas = 5
    linhas = -(-len(modelos) // colunas)
    caminhos = []
    for d in DATASETS:
        fig, eixos = plt.subplots(linhas, colunas, figsize=(LARGURA, LARGURA * 0.92),
                                  gridspec_kw=dict(wspace=0.35, hspace=0.55))
        for k, ax in enumerate(eixos.flat):
            if k >= len(modelos):
                ax.axis("off")
                continue
            m = modelos[k]
            _matriz(ax, numeros["modelos"][d][m]["confusao"], NOMES_CURTOS[m], rotulos=True)
        fig.suptitle(f"{TITULOS[d]}: linhas = classe real, colunas = prevista (C controle, P paciente; limiar 0,5)",
                     fontsize=7.5, y=0.995)
        caminhos += _gravar(fig, saida / SUPLEMENTARES, f"s2_confusao_{d.lower()}")
    return caminhos


# ---------------------------------------------------------------- barras com IC

def _barras(ax, x, alturas, ics, cores, larguras=0.8, valores: bool = True) -> None:
    """Barras com IC de 95% e o valor acima do IC; base em 0, linha do acaso em 0,5."""
    ics = np.asarray(ics)
    ax.bar(x, alturas, width=larguras, color=cores, edgecolor="white", linewidth=1.0, zorder=2)
    ax.errorbar(x, alturas, yerr=[np.asarray(alturas) - ics[:, 0], ics[:, 1] - np.asarray(alturas)], fmt="none",
                ecolor=TEXTO, elinewidth=0.8, capsize=2.2, capthick=0.8, zorder=3)
    if valores:
        for xi, a, hi in zip(x, alturas, ics[:, 1]):
            ax.text(xi, hi + 0.015, virgula(a), ha="center", va="bottom", fontsize=6.4, color=TEXTO, zorder=4,
                    bbox=dict(fc="white", ec="none", pad=0.5))
    _eixo_barras(ax)


MODALIDADES = [("temporal", "Só temporal"), ("espectral", "Só espectral"), ("combinado", "Temporal + espectral")]
DTSF_MODALIDADE = {"temporal": "D_sem_E", "espectral": "D_sem_T", "combinado": "D_pequeno"}


def fig_temporal_espectral(numeros: dict, saida: Path) -> list[Path]:
    familias = [("DTSF (D_pequeno)", lambda mod: DTSF_MODALIDADE[mod])] + \
               [(c, lambda mod, c=c: f"{c}_{mod}") for c in CLASSIFICADORES]
    fig, eixos = plt.subplots(2, 1, figsize=(LARGURA, 5.2), gridspec_kw=dict(hspace=0.45))
    largura = 0.26
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        for j, (mod, _) in enumerate(MODALIDADES):
            x = np.arange(len(familias)) + (j - 1) * (largura + 0.01)
            ms = [f(mod) for _, f in familias]
            _barras(ax, x, [r[m]["auc"] for m in ms], [r[m]["ic"] for m in ms], [CORES[j + 1]] * len(ms), largura)
        ax.set_xticks(range(len(familias)), [t for t, _ in familias])
        ax.tick_params(axis="x", length=0)
        ax.set_ylabel("AUC por pessoa")
        ax.set_title(TITULOS[d], loc="left")
    legenda = [plt.Rectangle((0, 0), 1, 1, color=CORES[j + 1], label=t) for j, (_, t) in enumerate(MODALIDADES)]
    legenda.append(Line2D([], [], color=TEXTO_2, lw=0.7, ls="--", label="acaso (0,5)"))
    fig.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, 0.04), ncol=4, handlelength=1.4)
    return _gravar(fig, saida / SUPLEMENTARES, "s3_temporal_espectral")


def fig_absolutas_relativas(numeros: dict, saida: Path) -> list[Path]:
    modelos = ["LR_psd_absolutas", "LR_psd_relativas"]
    fig, ax = plt.subplots(figsize=(LARGURA * 0.7, 2.9))
    largura = 0.34
    rng = np.random.default_rng(0)
    for i, m in enumerate(modelos):
        x = np.arange(len(DATASETS)) + (i - 0.5) * (largura + 0.02)
        r = [numeros["modelos"][d][m] for d in DATASETS]
        _barras(ax, x, [v["auc"] for v in r], [v["ic"] for v in r], [CORES[i]] * len(DATASETS), largura)
        for xi, v in zip(x, r):
            reps = np.asarray(v["auc_repeticoes"])
            ax.plot(xi + rng.uniform(-0.08, 0.08, len(reps)), reps, ls="none", marker="o", ms=2.6, mfc="white",
                    mec=TEXTO, mew=0.5, zorder=4)
    ax.set_xticks(range(len(DATASETS)), [CURTOS[d] for d in DATASETS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("AUC")
    legenda = [plt.Rectangle((0, 0), 1, 1, color=CORES[i], label=f"{nome(m)} (por pessoa, IC de 95%)")
               for i, m in enumerate(modelos)]
    legenda.append(Line2D([], [], ls="none", marker="o", ms=2.6, mfc="white", mec=TEXTO, mew=0.5,
                          label="AUC de cada repetição"))
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1)
    return _gravar(fig, saida / SUPLEMENTARES, "s4_absolutas_relativas")


# ---------------------------------------------------------------- repetições

def fig_auc_repeticoes(numeros: dict, saida: Path) -> list[Path]:
    modelos, titulos = _linhas(GRUPOS)
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 5.6), sharey=True, gridspec_kw=dict(wspace=0.12))
    rng = np.random.default_rng(0)
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        ax.axvline(0.5, color=CINZA_CLARO, lw=0.8, ls="--", zorder=0)
        for m, y in modelos:
            e = estilo(m)
            aucs = np.asarray(r[m]["auc_repeticoes"])
            jitter = rng.uniform(-0.18, 0.18, len(aucs))
            ax.plot(aucs, y + jitter, ls="none", marker=e["marker"], ms=3, mfc="white", mec=e["color"], mew=0.6,
                    alpha=0.9, zorder=1)
            ax.plot([aucs.mean()] * 2, [y - 0.32, y + 0.32], color=e["color"], lw=1.6, zorder=2)
        ax.set_xlim(0.3, 1.0)
        ax.xaxis.set_major_locator(MultipleLocator(0.1))
        _eixo_virgula(ax)
        ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
        ax.set_title(TITULOS[d], loc="left")
        ax.set_xlabel("AUC de cada repetição (10) e média")
    _rotulos_y(eixos[0], modelos, titulos)
    eixos[1].tick_params(labelleft=False)
    legenda = [Line2D([], [], ls="none", marker="o", ms=3, mfc="white", mec=CINZA, mew=0.6, label="uma repetição"),
               Line2D([], [], color=CINZA, lw=1.6, label="média das 10")]
    eixos[1].legend(handles=legenda, loc="lower left", bbox_to_anchor=(0.0, 0.0))
    return _gravar(fig, saida / SUPLEMENTARES, "s5_auc_repeticoes")


# ---------------------------------------------------------------- ΔAUC

UMA_MODALIDADE = [f"{c}_{m}" for m in ("temporal", "espectral") for c in CLASSIFICADORES]
# (título do grupo, hipótese decidida ou None, [(comparação "A − B", modelo que dá o marcador, rótulo)])
HIPOTESES_FIG = [
    ("H2: D_pequeno contra", "H2", [(f"{REFERENCIA_ABLACOES} − {m}", m, rotulo) for m, rotulo in
                                    (("D_sem_T", "sem ramo temporal"), ("D_sem_E", "sem ramo espectral"))]),
    ("H3: D_pequeno contra", "H3", [(f"{REFERENCIA_ABLACOES} − D_sem_mixup", "D_sem_mixup", "sem Mixup")]),
    ("H4: D_FA contra", "H4", [(f"{PRINCIPAL} − {c}_combinado", f"{c}_combinado", f"{c} combinado")
                              for c in CLASSIFICADORES]),
    ("Descritivo: entre os DTSF", None, [("D_FA − D", "D", "D_FA − D (FA)"),
                                         ("D − D_pequeno", "D_pequeno", "D − D_pequeno (tamanho)"),
                                         ("D_FA − D_pequeno", "D_pequeno", "D_FA − D_pequeno")]),
    ("Descritivo: D_FA contra", None, [(f"{PRINCIPAL} − {m}", m, m.replace("_", " ")) for m in UMA_MODALIDADE]),
]


def _decisao(numeros: dict, h: str, d: str) -> str:
    dec = numeros["hipoteses"][h][d]
    return (f"{'p IU' if h in ('H2', 'H4') else 'p'} = {virgula(dec['p'])}; Holm = {virgula(dec['p_ajustado'])}; "
            f"{'rejeita' if dec['rejeita'] else 'não rejeita'} H₀")


def fig_delta_hipoteses(numeros: dict, saida: Path) -> list[Path]:
    modelos, titulos = _linhas([(t, itens) for t, _, itens in HIPOTESES_FIG])
    decide = [h for _, h, _ in HIPOTESES_FIG]
    fig, eixos = plt.subplots(1, 2, figsize=(LARGURA, 6.6), sharey=True, gridspec_kw=dict(wspace=0.42))
    for ax, d in zip(eixos, DATASETS):
        c = numeros["comparacoes"][d]
        ax.axvline(0, color=PRETO, lw=0.7, zorder=0)
        for (chave, m, _), y in modelos:
            e = estilo(m)
            ax.plot(c[chave]["ic"], [y, y], color=e["color"], lw=1.1, solid_capstyle="round", zorder=1)
            ax.plot(c[chave]["delta"], y, ls="none", marker=e["marker"], ms=e["ms"], mfc=e["mfc"], mec=e["color"],
                    mew=0.9, zorder=2)
            ax.text(1.03, y, virgula(c[chave]["p"]), transform=ax.get_yaxis_transform(), va="center", ha="left",
                    fontsize=7)
        ax.text(1.03, titulos[0][1] + 0.9, "p", transform=ax.get_yaxis_transform(), va="center", ha="left",
                fontsize=7, fontstyle="italic", fontweight="bold")
        for h, (_, y) in zip(decide, titulos):
            texto = "sem decisão" if h is None else _decisao(numeros, h, d)
            ax.text(0.01, y, texto, transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.8,
                    color=PRETO, fontstyle="italic", bbox=dict(fc="white", ec="none", pad=0.6))
        ax.set_xlim(-0.2, 0.55)
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.1))
        _eixo_virgula(ax, sinal=True)
        ax.grid(axis="x", color="#e6e6e6", lw=0.5, zorder=0)
        ax.set_title(TITULOS[d], loc="left")
    fig.supxlabel("ΔAUC = AUC(referência) − AUC(comparado), com IC de 95%", fontsize=8, y=0.035)
    eixos[0].set_yticks([y for _, y in modelos] + [y for _, y in titulos])
    eixos[0].set_yticklabels([rotulo for (_, _, rotulo), _ in modelos] + [t for t, _ in titulos])
    for rotulo in eixos[0].get_yticklabels()[len(modelos):]:
        rotulo.set_fontweight("bold")
    eixos[0].set_ylim(min(y for _, y in modelos) - 0.7, 0.7)
    eixos[1].tick_params(labelleft=False)
    return _gravar(fig, saida / SUPLEMENTARES, "s6_delta_hipoteses")


# ---------------------------------------------------------------- testes estatísticos

ENUNCIADOS = {"H1": "D_FA > acaso", "H2": "D_pequeno > sem T, sem E",
              "H3": "D_pequeno > sem Mixup", "H4": "D_FA > os 3 combinados"}
TESTES = {"H1": "permutação de rótulos", "H2": "ranks pareados, IU", "H3": "ranks pareados",
          "H4": "ranks pareados, IU"}


def linhas_testes(numeros: dict) -> list[list[str]]:
    """Uma linha por (hipótese, transtorno): enunciado, teste, p, p de Holm e decisão."""
    linhas = []
    for h in ("H1", "H2", "H3", "H4"):
        for d in DATASETS:
            dec = numeros["h1"][d] if h == "H1" else numeros["hipoteses"][h][d]
            holm = "—" if h == "H1" else virgula(dec["p_ajustado"])
            linhas.append([h, ENUNCIADOS[h], TESTES[h], d, virgula(dec["p"]), holm,
                           "rejeita H₀" if dec["rejeita"] else "não rejeita H₀"])
    return linhas


def fig_testes_estatisticos(numeros: dict, saida: Path) -> list[Path]:
    fig = plt.figure(figsize=(LARGURA, 5.6))
    grade = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.45, wspace=0.25)
    for k, d in enumerate(DATASETS):
        ax = fig.add_subplot(grade[0, k])
        dist = numeros["h1_distribuicao"][d]
        ax.hist(dist["nulas"], bins=np.arange(0.2, 1.0001, 0.04), color=CINZA_CLARO, edgecolor="white", zorder=2,
                label="AUC com rótulos permutados")
        ax.axvline(dist["observada"], color=PRETO, lw=1.4, zorder=3, label="AUC observada (repetição 0)")
        ax.axvline(0.5, color=TEXTO_2, lw=0.7, ls="--", zorder=1, label="acaso (0,5)")
        ax.text(dist["observada"], ax.get_ylim()[1] * 0.95, f" {virgula(dist['observada'])}\n p = {virgula(dist['p'])}",
                va="top", ha="right" if dist["observada"] > 0.8 else "left", fontsize=6.8)
        ax.set_xlim(0.2, 1.0)
        _eixo_virgula(ax)
        ax.tick_params(axis="y", length=3, width=0.6)
        ax.set_xlabel("AUC")
        ax.set_ylabel("Permutações" if k == 0 else "")
        ax.set_title(f"H1, {CURTOS[d]}", loc="left")
    fig.axes[0].legend(loc="upper center", bbox_to_anchor=(1.1, -0.28), ncol=3, fontsize=6.8)
    ax = fig.add_subplot(grade[1, :])
    ax.axis("off")
    tabela = ax.table(cellText=linhas_testes(numeros),
                      colLabels=["", "Hipótese", "Teste", "", "p", "p (Holm)", "Decisão"],
                      colWidths=[0.04, 0.33, 0.23, 0.07, 0.07, 0.1, 0.16], loc="center", cellLoc="left")
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(6.8)
    tabela.scale(1, 1.25)
    for (i, _), celula in tabela.get_celld().items():
        celula.set_edgecolor(EIXO)
        celula.set_linewidth(0.4)
        if i == 0:
            celula.set_text_props(fontweight="bold")
    return _gravar(fig, saida / SUPLEMENTARES, "s7_testes_estatisticos")


# ---------------------------------------------------------------- barras (coloridas)

def fig_barras_ablacoes(numeros: dict, saida: Path) -> list[Path]:
    modelos = [REFERENCIA_ABLACOES, *ABLACOES]
    fig, ax = plt.subplots(figsize=(LARGURA, 2.9))
    largura = 0.19
    for i, m in enumerate(modelos):
        x = np.arange(len(DATASETS)) + (i - 1.5) * (largura + 0.012)
        r = [numeros["modelos"][d][m] for d in DATASETS]
        _barras(ax, x, [v["auc"] for v in r], [v["ic"] for v in r], [CORES[i]] * len(DATASETS), largura)
    ax.set_xticks(range(len(DATASETS)), [TITULOS[d] for d in DATASETS])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("AUC por pessoa")
    legenda = [plt.Rectangle((0, 0), 1, 1, color=CORES[i], label=nome(m) if i == 0 else f"D_pequeno {nome(m)}")
               for i, m in enumerate(modelos)]
    ax.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, handlelength=1.2,
              columnspacing=1.4)
    ax.text(1.0, 0.5, " acaso", transform=ax.get_yaxis_transform(), va="center", ha="left", fontsize=6.6,
            color=TEXTO_2)
    return _gravar(fig, saida / SUPLEMENTARES, "s8_barras_ablacoes")


def fig_barras_classicos(numeros: dict, saida: Path) -> list[Path]:
    fig, eixos = plt.subplots(2, 1, figsize=(LARGURA, 5.6), gridspec_kw=dict(hspace=0.42))
    for ax, d in zip(eixos, DATASETS):
        r = numeros["modelos"][d]
        x, alturas, ics, cores, rotulos = [0.0], [r[PRINCIPAL]["auc"]], [r[PRINCIPAL]["ic"]], [CORES[0]], ["D_FA"]
        posicao = 0.6
        for j, (modalidade, _) in enumerate(MODALIDADES):
            posicao += 0.6
            for c in CLASSIFICADORES:
                m = f"{c}_{modalidade}"
                x.append(posicao)
                alturas.append(r[m]["auc"])
                ics.append(r[m]["ic"])
                cores.append(CORES[j + 1])
                rotulos.append(c)
                posicao += 1
        _barras(ax, x, alturas, ics, cores, 0.82)
        ax.axhline(r[PRINCIPAL]["auc"], color=CORES[0], lw=0.8, ls=":", zorder=1)
        ax.set_xticks(x, rotulos)
        ax.tick_params(axis="x", length=0, labelsize=7.5)
        ax.set_ylabel("AUC por pessoa")
        ax.set_xlim(-0.7, x[-1] + 0.7)
        ax.set_title(TITULOS[d], loc="left")
    legenda = [plt.Rectangle((0, 0), 1, 1, color=CORES[0], label="DTSF (D_FA)")]
    legenda += [plt.Rectangle((0, 0), 1, 1, color=CORES[j + 1], label=f"Clássicos: {t.lower()}")
                for j, (_, t) in enumerate(MODALIDADES)]
    legenda += [Line2D([], [], color=CORES[0], lw=0.8, ls=":", label="AUC do D_FA"),
                Line2D([], [], color=TEXTO_2, lw=0.7, ls="--", label="acaso (0,5)")]
    fig.legend(handles=legenda, loc="upper center", bbox_to_anchor=(0.5, 0.045), ncol=3, handlelength=1.4,
               columnspacing=1.6)
    return _gravar(fig, saida / SUPLEMENTARES, "s9_barras_classicos")


FIGURAS = dict(
    roc_principais=fig_roc_principais, auc_modelos=fig_auc_modelos, metricas_principais=fig_metricas_principais,
    sz_mdd=fig_sz_mdd,
    roc_ablacoes=fig_roc_ablacoes, confusao_todos=fig_confusao_todos, temporal_espectral=fig_temporal_espectral,
    absolutas_relativas=fig_absolutas_relativas, auc_repeticoes=fig_auc_repeticoes,
    delta_hipoteses=fig_delta_hipoteses, testes_estatisticos=fig_testes_estatisticos,
    barras_ablacoes=fig_barras_ablacoes, barras_classicos=fig_barras_classicos,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--numeros", type=Path, default=SAIDA / "numeros.json")
    parser.add_argument("--saida", type=Path, default=SAIDA)
    parser.add_argument("--figuras", nargs="+", default=list(FIGURAS), choices=list(FIGURAS))
    args = parser.parse_args()
    configurar()
    numeros = json.loads(args.numeros.read_text(encoding="utf-8"))
    for f in args.figuras:
        for caminho in FIGURAS[f](numeros, args.saida):
            print(caminho)


if __name__ == "__main__":
    main()
