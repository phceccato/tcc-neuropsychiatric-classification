"""
Análise do Dataset EEG BRMH — Classificação de Transtornos Neuropsiquiátricos
SMG-SNU Boramae Medical Center (Coreia do Sul, 2019)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")

# Configuração
DATA_PATH = "EEG.machinelearing_data_BRMH.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COL = "main.disorder"
BANDS = ["delta", "theta", "alpha", "beta", "highbeta", "gamma"]
BAND_COLORS = {
    "delta":   "#4e79a7",
    "theta":   "#f28e2b",
    "alpha":   "#59a14f",
    "beta":    "#e15759",
    "highbeta":"#b07aa1",
    "gamma":   "#76b7b2",
}

CLASS_LABELS = {
    "Mood disorder":                        "Humor",
    "Addictive disorder":                   "Adição",
    "Trauma and stress related disorder":   "Trauma/Estresse",
    "Schizophrenia":                        "Esquizofrenia",
    "Anxiety disorder":                     "Ansiedade",
    "Healthy control":                      "Controle",
    "Obsessive compulsive disorder":        "TOC",
}

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
})


# 1. CARREGAMENTO E PRÉ-PROCESSAMENTO

def load_and_preprocess(path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    print("=" * 70)
    print("1. CARREGAMENTO E PRÉ-PROCESSAMENTO")
    print("=" * 70)

    df = pd.read_csv(path)
    print(f"Shape bruto: {df.shape}")

    drop_cols = ["no.", "eeg.date", "specific.disorder"]
    drop_cols += [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    print(f"Shape após remoção de colunas irrelevantes: {df.shape}")

    df["sex"] = (df["sex"] == "M").astype(int)

    labels = df[LABEL_COL].map(CLASS_LABELS)
    df = df.drop(columns=[LABEL_COL])

    eeg_cols  = [c for c in df.columns if c.startswith(("AB.", "COH."))]
    meta_cols = ["sex", "age", "education", "IQ"]
    feature_cols = meta_cols + eeg_cols

    X = df[feature_cols].copy()

    # Imputa valores ausentes com a mediana de cada coluna
    imputer = SimpleImputer(strategy="median")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols)

    print(f"\nFeatures de metadados : {len(meta_cols)}")
    print(f"Features de PSD (AB)  : {len([c for c in eeg_cols if c.startswith('AB.')])}")
    print(f"Features de COH       : {len([c for c in eeg_cols if c.startswith('COH.')])}")
    print(f"Total de features     : {X_imp.shape[1]}")
    print(f"\nDistribuição de classes:")
    vc = labels.value_counts()
    for cls, n in vc.items():
        print(f"  {cls:<22} {n:>4} ({100*n/len(labels):.1f}%)")

    return df, X_imp, labels


# 2. ANÁLISE EXPLORATÓRIA (EDA)

# Imagem: barras horizontais + gráfico de pizza mostrando quantas amostras
# existem por diagnóstico e qual a proporção de cada classe no total.
def plot_class_distribution(labels: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Distribuição das Classes Diagnósticas", fontsize=13, fontweight="bold")

    vc = labels.value_counts()
    palette = sns.color_palette("tab10", len(vc))

    ax = axes[0]
    bars = ax.barh(vc.index, vc.values, color=palette)
    ax.bar_label(bars, fmt="%d", padding=3)
    ax.set_xlabel("Número de Amostras")
    ax.set_title("Contagem por Diagnóstico")
    ax.invert_yaxis()

    ax = axes[1]
    ax.pie(
        vc.values, labels=vc.index, autopct="%1.1f%%",
        colors=palette, startangle=140,
        textprops={"fontsize": 8},
    )
    ax.set_title("Proporção (%)")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/01_distribuicao_classes.png", bbox_inches="tight")
    plt.close()
    print("\n[Salvo] 01_distribuicao_classes.png")


# Imagem: quatro painéis com boxplots de idade e QI por diagnóstico,
# histograma geral de idade e barras de proporção de sexo por classe.
def plot_demographics(df_raw: pd.DataFrame, labels: pd.Series) -> None:
    df_plot = df_raw[["sex", "age", "education", "IQ"]].copy()
    df_plot["sex"] = df_raw["sex"].map({1: "M", 0: "F"}) if df_raw["sex"].dtype == int \
        else df_raw["sex"]
    df_plot["Diagnóstico"] = labels.values

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Análise Demográfica por Diagnóstico", fontsize=13, fontweight="bold")

    order = labels.value_counts().index.tolist()

    sns.boxplot(data=df_plot, x="Diagnóstico", y="age", order=order,
                palette="tab10", ax=axes[0, 0])
    axes[0, 0].set_title("Distribuição de Idade")
    axes[0, 0].tick_params(axis="x", rotation=30)
    axes[0, 0].set_xlabel("")

    axes[0, 1].hist(df_plot["age"].dropna(), bins=20, color="#4e79a7", edgecolor="white")
    axes[0, 1].set_title("Distribuição Geral de Idade")
    axes[0, 1].set_xlabel("Idade")

    sns.boxplot(data=df_plot, x="Diagnóstico", y="IQ", order=order,
                palette="tab10", ax=axes[1, 0])
    axes[1, 0].set_title("Distribuição de QI")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].set_xlabel("")

    sex_df = df_plot.groupby(["Diagnóstico", "sex"]).size().unstack(fill_value=0)
    sex_df = sex_df.loc[[o for o in order if o in sex_df.index]]
    sex_df.plot(kind="bar", ax=axes[1, 1], color=["#e15759", "#4e79a7"], edgecolor="white")
    axes[1, 1].set_title("Distribuição por Sexo")
    axes[1, 1].set_xlabel("")
    axes[1, 1].tick_params(axis="x", rotation=30)
    axes[1, 1].legend(title="Sexo")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/02_demograficos.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 02_demograficos.png")


# 3. ANÁLISE POR BANDA DE FREQUÊNCIA (PSD)

def get_band_cols(X: pd.DataFrame) -> dict[str, list[str]]:
    ab_cols = [c for c in X.columns if c.startswith("AB.")]
    return {
        band: [c for c in ab_cols if f".{band}." in c.lower()]
        for band in BANDS
    }


# Imagem: dois heatmaps lado a lado — valores brutos de PSD (μV²) e
# z-score normalizado — mostrando quais diagnósticos têm mais ou menos
# potência em cada banda de frequência.
def plot_band_power_by_class(X: pd.DataFrame, labels: pd.Series) -> None:
    band_cols = get_band_cols(X)
    order = labels.value_counts().index.tolist()

    matrix = pd.DataFrame(index=order, columns=BANDS, dtype=float)
    for band, cols in band_cols.items():
        for cls in order:
            mask = labels == cls
            matrix.loc[cls, band] = X.loc[mask, cols].values.mean()

    matrix_norm = (matrix - matrix.mean()) / matrix.std()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Potência Espectral Média por Banda e Diagnóstico", fontsize=13, fontweight="bold")

    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f",
                cmap="YlOrRd", ax=axes[0], linewidths=0.5)
    axes[0].set_title("Valores Brutos (μV²)")
    axes[0].set_xlabel("Banda de Frequência")

    sns.heatmap(matrix_norm.astype(float), annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=axes[1], linewidths=0.5)
    axes[1].set_title("Normalizado (Z-score entre classes)")
    axes[1].set_xlabel("Banda de Frequência")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/03_psd_por_banda_classe.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 03_psd_por_banda_classe.png")


# Imagem: grade 2×4 com um mapa de calor por diagnóstico. Cada mapa mostra
# os 19 canais EEG (linhas) × 6 bandas (colunas) em z-score relativo à
# média global — vermelho indica potência acima da média, azul abaixo.
def plot_topographic_band_power(X: pd.DataFrame, labels: pd.Series) -> None:
    ab_cols = [c for c in X.columns if c.startswith("AB.")]

    # Uppercase para evitar mismatch com nomes como Fz → FZ nos nomes de coluna
    CHANNELS_UP  = ["FP1","FP2","F7","F3","FZ","F4","F8",
                    "T3","C3","CZ","C4","T4","T5","P3","PZ","P4","T6","O1","O2"]
    CHANNEL_DISP = ["FP1","FP2","F7","F3","Fz","F4","F8",
                    "T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]

    records = []
    for col in ab_cols:
        parts = col.split(".")
        if len(parts) < 3:
            continue
        band = parts[2].lower()
        ch   = parts[-1].upper()
        if band in BANDS and ch in CHANNELS_UP:
            records.append((band, ch, col))

    if not records:
        print("[AVISO] plot_topographic_band_power: nenhuma feature AB encontrada.")
        return

    order = labels.value_counts().index.tolist()
    col_idx = {b: i for i, b in enumerate(BANDS)}
    row_idx = {c: i for i, c in enumerate(CHANNELS_UP)}

    col_mean = {col: float(X[col].mean()) for _, _, col in records}
    col_std  = {col: float(X[col].std()) or 1.0 for _, _, col in records}

    matrices = {}
    for cls in order:
        mask = labels.values == cls
        mat = np.full((len(CHANNELS_UP), len(BANDS)), np.nan)
        for band, ch, col in records:
            cls_mean = float(X.loc[mask, col].mean())
            mat[row_idx[ch], col_idx[band]] = (cls_mean - col_mean[col]) / col_std[col]
        matrices[cls] = mat

    all_z = np.concatenate([m.flatten() for m in matrices.values()])
    all_z = all_z[~np.isnan(all_z)]
    vlim  = float(np.percentile(np.abs(all_z), 95)) or 1.0

    BAND_SHORT = ["δ\n1-4", "θ\n4-8", "α\n8-12", "β\n12-25", "hβ\n25-30", "γ\n30-40"]
    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-vlim, vlim)

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    fig.suptitle(
        "PSD por Canal × Banda — Z-score em relação à média global\n"
        "(vermelho = acima da média | azul = abaixo)",
        fontsize=12, fontweight="bold",
    )
    axes_flat = axes.flatten()

    for i, cls in enumerate(order):
        ax = axes_flat[i]
        n  = int((labels == cls).sum())

        im = ax.imshow(matrices[cls], aspect="auto", cmap=cmap,
                       norm=norm, interpolation="nearest")

        ax.set_title(f"{cls}  (n={n})", fontsize=10, fontweight="bold")
        ax.set_xticks(range(len(BANDS)))
        ax.set_xticklabels(BAND_SHORT, fontsize=8)
        ax.set_yticks(range(len(CHANNELS_UP)))
        ax.set_yticklabels(CHANNEL_DISP, fontsize=7)
        ax.set_xlabel("Banda (Hz)", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("Canal EEG", fontsize=8)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="z-score" if i % 4 == 3 else "")

    axes_flat[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/04_psd_canal_banda_classe.png", bbox_inches="tight", dpi=130)
    plt.close()
    print("[Salvo] 04_psd_canal_banda_classe.png")


# Imagem: seis violin plots (um por banda) mostrando a distribuição
# da potência média entre diagnósticos, com box interno indicando mediana
# e quartis. Permite ver sobreposição e assimetria entre grupos.
def plot_band_violin(X: pd.DataFrame, labels: pd.Series) -> None:
    band_cols = get_band_cols(X)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribuição da Potência Espectral por Banda", fontsize=13, fontweight="bold")
    axes = axes.flatten()

    order = labels.value_counts().index.tolist()

    for i, band in enumerate(BANDS):
        cols = band_cols[band]
        means = X[cols].mean(axis=1)
        plot_df = pd.DataFrame({"Potência": means, "Diagnóstico": labels.values})

        sns.violinplot(data=plot_df, x="Diagnóstico", y="Potência",
                       order=order, palette="tab10", ax=axes[i],
                       inner="box", cut=0)
        axes[i].set_title(f"Banda {band.capitalize()} ({_band_freq(band)})")
        axes[i].set_xlabel("")
        axes[i].tick_params(axis="x", rotation=35, labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/05_violin_bandas.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 05_violin_bandas.png")


def _band_freq(band: str) -> str:
    return {"delta":"1-4 Hz","theta":"4-8 Hz","alpha":"8-12 Hz",
            "beta":"12-25 Hz","highbeta":"25-30 Hz","gamma":"30-40 Hz"}.get(band,"")


# 4. TESTES ESTATÍSTICOS

def run_kruskal_wallis(X: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    print("\n" + "=" * 70)
    print("4. TESTES DE KRUSKAL-WALLIS POR BANDA")
    print("=" * 70)

    band_cols = get_band_cols(X)
    classes = labels.unique()
    results = []

    for band, cols in band_cols.items():
        means = X[cols].mean(axis=1)
        groups = [means[labels == cls].values for cls in classes]
        h, p = stats.kruskal(*groups)
        results.append({"Banda": band, "H-statistic": round(h, 3), "p-value": p,
                         "Significativo (α=0.05)": "Sim" if p < 0.05 else "Não"})

    res_df = pd.DataFrame(results).sort_values("p-value")
    print(res_df.to_string(index=False))
    return res_df


# Imagem: barras com -log10(p-value) por banda. A linha vermelha tracejada
# marca o limiar α=0,05. Bandas com barra acima da linha têm diferença
# estatisticamente significativa entre os diagnósticos.
def plot_kruskal_results(kw_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = [BAND_COLORS.get(b, "#888") for b in kw_df["Banda"]]
    bars = ax.bar(kw_df["Banda"], -np.log10(kw_df["p-value"]), color=colors, edgecolor="white")
    ax.axhline(-np.log10(0.05), color="red", linestyle="--", label="α = 0.05")
    ax.bar_label(bars, labels=[f'p={p:.3g}' for p in kw_df["p-value"]], padding=3, fontsize=8)
    ax.set_ylabel("-log₁₀(p-value)")
    ax.set_title("Teste de Kruskal-Wallis por Banda de Frequência")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/06_kruskal_wallis.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 06_kruskal_wallis.png")


# 5. REDUÇÃO DE DIMENSIONALIDADE

# Imagem: três painéis — variância explicada por componente, variância
# acumulada com marcador em 90%, e scatter plot PC1 × PC2 com cada
# diagnóstico em uma cor diferente.
def plot_pca(X: pd.DataFrame, labels: pd.Series) -> None:
    print("\n" + "=" * 70)
    print("5. REDUÇÃO DE DIMENSIONALIDADE")
    print("=" * 70)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X_sc)

    var_exp = np.cumsum(pca.explained_variance_ratio_)
    n90 = np.searchsorted(var_exp, 0.90) + 1
    print(f"PCA: componentes para 90% da variância: {n90}")

    classes = labels.unique()
    palette = dict(zip(classes, sns.color_palette("tab10", len(classes))))

    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0])
    ax0.plot(range(1, 51), pca.explained_variance_ratio_ * 100, "o-", ms=4, color="#4e79a7")
    ax0.set_xlabel("Componente Principal")
    ax0.set_ylabel("Variância Explicada (%)")
    ax0.set_title("Variância por Componente")

    ax1 = fig.add_subplot(gs[1])
    ax1.plot(range(1, 51), var_exp * 100, "o-", ms=4, color="#59a14f")
    ax1.axhline(90, color="red", linestyle="--", label="90%")
    ax1.axvline(n90, color="red", linestyle="--")
    ax1.set_xlabel("Componente Principal")
    ax1.set_ylabel("Variância Acumulada (%)")
    ax1.set_title("Variância Acumulada")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[2])
    for cls in classes:
        mask = labels == cls
        ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                    label=cls, alpha=0.5, s=18, color=palette[cls])
    ax2.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax2.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax2.set_title("PCA — PC1 × PC2")
    ax2.legend(fontsize=6, markerscale=1.5)

    plt.suptitle("Análise de Componentes Principais (PCA)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/07_pca.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 07_pca.png")

    return X_pca


# Imagem: scatter 2D com os 945 pacientes projetados pelo t-SNE (aplicado
# sobre os 50 primeiros PCs). Cada cor representa um diagnóstico. Grupos
# visualmente separados indicam padrões EEG distintos.
def plot_tsne(X: pd.DataFrame, labels: pd.Series) -> None:
    print("Calculando t-SNE (pode levar 1-2 min)…")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    pca50 = PCA(n_components=50, random_state=42)
    X_pca50 = pca50.fit_transform(X_sc)

    tsne = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42)
    X_tsne = tsne.fit_transform(X_pca50)

    classes = labels.unique()
    palette = dict(zip(classes, sns.color_palette("tab10", len(classes))))

    fig, ax = plt.subplots(figsize=(9, 7))
    for cls in classes:
        mask = labels == cls
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                   label=cls, alpha=0.6, s=22, color=palette[cls])
    ax.set_title("t-SNE (PCA 50 → 2D)", fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend(fontsize=8, markerscale=1.5, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/08_tsne.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 08_tsne.png")


# 6. CLASSIFICAÇÃO E COMPARAÇÃO DE MODELOS

def build_pipelines() -> dict:
    return {
        "Regressão Logística": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    LogisticRegression(max_iter=1000, C=0.1, random_state=42)),
        ]),
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    SVC(kernel="rbf", C=10, gamma="scale", random_state=42)),
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    RandomForestClassifier(n_estimators=200, max_depth=None,
                                              random_state=42, n_jobs=-1)),
        ]),
        "Gradient Boosting": Pipeline([
            ("scaler", StandardScaler()),
            ("clf",    GradientBoostingClassifier(n_estimators=200, learning_rate=0.1,
                                                  max_depth=4, random_state=42)),
        ]),
    }


def evaluate_models(X: pd.DataFrame, labels: pd.Series,
                    use_pca: bool = True) -> dict:
    print("\n" + "=" * 70)
    print("6. CLASSIFICAÇÃO — VALIDAÇÃO CRUZADA (5-fold stratified)")
    print("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    Xn = X.values
    if use_pca:
        sc = StandardScaler()
        pca = PCA(n_components=100, random_state=42)
        Xn = pca.fit_transform(sc.fit_transform(Xn))
        print(f"Usando PCA (100 componentes) para acelerar avaliação.")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipelines = build_pipelines()

    results = {}
    for name, pipe in pipelines.items():
        if use_pca:
            clf = pipe.named_steps["clf"]
            scores = cross_val_score(clf, Xn, y, cv=cv,
                                     scoring="balanced_accuracy", n_jobs=-1)
        else:
            scores = cross_val_score(pipe, Xn, y, cv=cv,
                                     scoring="balanced_accuracy", n_jobs=-1)

        mean, std = scores.mean(), scores.std()
        results[name] = {"scores": scores, "mean": mean, "std": std}
        print(f"  {name:<25} Balanced Acc: {mean:.4f} ± {std:.4f}")

    return results, le, y, Xn


# Imagem: barras com a Balanced Accuracy média de cada modelo (5-fold CV),
# com barra de erro indicando o desvio entre folds. A linha tracejada cinza
# marca a acurácia por acaso (1/7 ≈ 14%).
def plot_model_comparison(results: dict) -> None:
    names = list(results.keys())
    means = [results[n]["mean"] for n in names]
    stds  = [results[n]["std"]  for n in names]

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = sns.color_palette("tab10", len(names))
    bars = ax.bar(names, means, yerr=stds, capsize=6, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{m:.3f}" for m in means], padding=6, fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Balanced Accuracy (5-fold CV)")
    ax.set_title("Comparação de Modelos de Classificação", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=15)
    ax.axhline(1 / 7, color="grey", linestyle="--", label="Chance (1/7 ≈ 14%)")
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/09_comparacao_modelos.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 09_comparacao_modelos.png")


# Imagem: duas matrizes de confusão do melhor modelo (5-fold CV predict) —
# à esquerda em contagens absolutas, à direita em proporção por linha
# (recall por classe). Ideal para ver quais diagnósticos são confundidos.
def plot_confusion_matrix_best(X_transformed, y, le, results: dict) -> None:
    best_name = max(results, key=lambda n: results[n]["mean"])
    print(f"\nMelhor modelo: {best_name} (Balanced Acc = {results[best_name]['mean']:.4f})")

    clf = {
        "Regressão Logística": LogisticRegression(max_iter=1000, C=0.1, random_state=42),
        "SVM (RBF)":            SVC(kernel="rbf", C=10, gamma="scale", random_state=42),
        "Random Forest":        RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=200, random_state=42),
    }[best_name]

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(clf, X_transformed, y, cv=cv)

    class_names = le.classes_
    cm = confusion_matrix(y, y_pred)
    cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Matriz de Confusão — {best_name}", fontsize=12, fontweight="bold")

    ConfusionMatrixDisplay(cm, display_labels=class_names).plot(
        ax=axes[0], colorbar=False, cmap="Blues")
    axes[0].set_title("Contagens")
    axes[0].tick_params(axis="x", rotation=35, labelsize=8)
    axes[0].tick_params(axis="y", labelsize=8)

    ConfusionMatrixDisplay(cm_pct, display_labels=class_names).plot(
        ax=axes[1], colorbar=False, cmap="Blues",
        values_format=".2f")
    axes[1].set_title("Proporção (por linha)")
    axes[1].tick_params(axis="x", rotation=35, labelsize=8)
    axes[1].tick_params(axis="y", labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/10_matriz_confusao.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 10_matriz_confusao.png")

    print("\nRelatório de Classificação (5-fold CV predict):")
    print(classification_report(y, y_pred, target_names=class_names))


# 7. IMPORTÂNCIA DAS FEATURES (Random Forest)

# Imagem: barras horizontais com as 30 features de maior importância Gini
# no Random Forest. Cores indicam o tipo: PSD por banda (6 cores) ou
# coerência (laranja). Abaixo do gráfico, a importância é somada por grupo.
def plot_feature_importance(X: pd.DataFrame, labels: pd.Series) -> None:
    print("\n" + "=" * 70)
    print("7. IMPORTÂNCIA DAS FEATURES (Random Forest)")
    print("=" * 70)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(X_sc, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns)
    top30 = importances.nlargest(30)

    def color_for(col):
        if col.startswith("AB."):
            for b in BANDS:
                if f".{b}." in col.lower():
                    return BAND_COLORS.get(b, "#aaa")
        if col.startswith("COH."):
            return "#d4a373"
        return "#264653"

    colors = [color_for(c) for c in top30.index]

    fig, ax = plt.subplots(figsize=(10, 9))
    ax.barh(range(len(top30)), top30.values, color=colors)
    ax.set_yticks(range(len(top30)))
    ax.set_yticklabels([c.replace("AB.A.", "").replace("AB.B.", "")
                        .replace("COH.A.", "COH.").replace("COH.B.", "COH.")
                        for c in top30.index], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Importância (Gini)")
    ax.set_title("Top 30 Features — Random Forest", fontsize=12, fontweight="bold")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=BAND_COLORS[b], label=f"PSD {b}") for b in BANDS]
    legend_elements.append(Patch(facecolor="#d4a373", label="Coerência (COH)"))
    legend_elements.append(Patch(facecolor="#264653", label="Metadados"))
    ax.legend(handles=legend_elements, fontsize=8, loc="lower right")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/11_feature_importance.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] 11_feature_importance.png")

    band_imp = {}
    for band in BANDS:
        cols = [c for c in X.columns if f".{band}." in c.lower() and c.startswith("AB.")]
        band_imp[band] = importances[cols].sum() if cols else 0
    coh_imp  = importances[[c for c in X.columns if c.startswith("COH.")]].sum()
    meta_imp = importances[["sex","age","education","IQ"]].sum()

    print("\nImportância agregada por grupo:")
    for k, v in sorted({**{f"PSD {b}": band_imp[b] for b in BANDS},
                         "COH total": coh_imp,
                         "Metadados": meta_imp}.items(), key=lambda x: -x[1]):
        print(f"  {k:<18} {v:.4f}")


# 8. ANÁLISE DE COERÊNCIA FUNCIONAL

# Imagem: grade 2×4 com uma matriz 19×19 por diagnóstico. Cada célula
# representa o z-score da coerência entre um par de eletrodos na banda
# escolhida — vermelho indica sincronização acima da média global, azul
# abaixo. A diagonal é zero (sem auto-coerência no dataset).
def plot_coherence_heatmap(X: pd.DataFrame, labels: pd.Series,
                           band: str = "alpha") -> None:
    coh_cols = [c for c in X.columns
                if c.startswith("COH.") and f".{band}." in c.lower()]

    if not coh_cols:
        print(f"[AVISO] Nenhuma coluna COH.{band} encontrada.")
        return

    # Uppercase para parear corretamente com os nomes de canal nas colunas COH
    CHANS_UP   = ["FP1","FP2","F7","F3","FZ","F4","F8",
                  "T3","C3","CZ","C4","T4","T5","P3","PZ","P4","T6","O1","O2"]
    CHANS_DISP = ["FP1","FP2","F7","F3","Fz","F4","F8",
                  "T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2"]
    N = len(CHANS_UP)
    ch_idx = {c: i for i, c in enumerate(CHANS_UP)}

    pairs = []
    for col in coh_cols:
        parts = col.split(".")
        if len(parts) < 7:
            continue
        ch1 = parts[4].upper()
        ch2 = parts[6].upper()
        if ch1 in ch_idx and ch2 in ch_idx:
            pairs.append((ch_idx[ch1], ch_idx[ch2], col))

    if not pairs:
        print("[AVISO] Nenhum par de eletrodos COH mapeado.")
        return

    col_mean = {col: float(X[col].mean()) for _, _, col in pairs}
    col_std  = {col: float(X[col].std()) or 1.0 for _, _, col in pairs}

    order = labels.value_counts().index.tolist()

    matrices = {}
    for cls in order:
        mask = labels.values == cls
        mat = np.zeros((N, N))
        for i, j, col in pairs:
            z = (float(X.loc[mask, col].mean()) - col_mean[col]) / col_std[col]
            mat[i, j] = z
            mat[j, i] = z  # matriz simétrica
        matrices[cls] = mat

    all_z = np.concatenate([m.flatten() for m in matrices.values()])
    vlim  = float(np.percentile(np.abs(all_z), 95)) or 0.3

    cmap = plt.cm.RdBu_r
    norm = plt.Normalize(-vlim, vlim)

    freq_label = {"delta":"1-4 Hz","theta":"4-8 Hz","alpha":"8-12 Hz",
                  "beta":"12-25 Hz","highbeta":"25-30 Hz","gamma":"30-40 Hz"}.get(band, band)

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(
        f"Coerência Funcional — Banda {band.capitalize()} ({freq_label})\n"
        f"Z-score em relação à média global  (vermelho = acima | azul = abaixo)",
        fontsize=12, fontweight="bold",
    )
    axes_flat = axes.flatten()

    tick_pos = list(range(N))
    for i, cls in enumerate(order):
        ax = axes_flat[i]
        n  = int((labels == cls).sum())

        im = ax.imshow(matrices[cls], cmap=cmap, norm=norm,
                       interpolation="nearest", aspect="equal")

        ax.set_title(f"{cls}  (n={n})", fontsize=10, fontweight="bold")
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(CHANS_DISP, rotation=90, fontsize=6)
        ax.set_yticks(tick_pos)
        ax.set_yticklabels(CHANS_DISP, fontsize=6)

        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                     label="z-score" if i % 4 == 3 else "")

    axes_flat[-1].set_visible(False)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/12_coerencia_{band}.png", bbox_inches="tight", dpi=130)
    plt.close()
    print(f"[Salvo] 12_coerencia_{band}.png")


# 9. SUMÁRIO FINAL

def print_summary(results: dict) -> None:
    print("\n" + "=" * 70)
    print("SUMÁRIO DOS RESULTADOS")
    print("=" * 70)
    print(f"{'Modelo':<28} {'Balanced Acc':>14}  {'Std':>8}")
    print("-" * 55)
    for name, r in sorted(results.items(), key=lambda x: -x[1]["mean"]):
        print(f"{name:<28} {r['mean']:>14.4f}  {r['std']:>8.4f}")
    best = max(results, key=lambda n: results[n]["mean"])
    print(f"\nMelhor modelo: {best} ({results[best]['mean']:.4f})")
    print(f"\nArquivos gerados em: ./{OUTPUT_DIR}/")


# MAIN

def main():
    # 1. Dados
    df_raw, X, labels = load_and_preprocess(DATA_PATH)

    # 2. EDA
    print("\n" + "=" * 70)
    print("2-3. ANÁLISE EXPLORATÓRIA E VISUALIZAÇÕES")
    print("=" * 70)
    plot_class_distribution(labels)
    plot_demographics(df_raw, labels)

    # 3. PSD por banda
    plot_band_power_by_class(X, labels)
    plot_topographic_band_power(X, labels)
    plot_band_violin(X, labels)

    # 4. Testes estatísticos
    kw_df = run_kruskal_wallis(X, labels)
    plot_kruskal_results(kw_df)

    # 5. Redução de dimensionalidade
    plot_pca(X, labels)
    plot_tsne(X, labels)

    # 6. Classificação
    results, le, y, X_pca = evaluate_models(X, labels, use_pca=True)
    plot_model_comparison(results)
    plot_confusion_matrix_best(X_pca, y, le, results)

    # 7. Importância das features
    plot_feature_importance(X, labels)

    # 8. Coerência
    plot_coherence_heatmap(X, labels)

    # 9. Sumário
    print_summary(results)


if __name__ == "__main__":
    main()
