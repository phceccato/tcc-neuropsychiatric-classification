"""
Pré-processamento do Dataset EEG BRMH
Etapas: Imputação → Balanceamento → Normalização → PCA
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

DATA_PATH  = "EEG.machinelearing_data_BRMH.csv"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

LABEL_COL = "main.disorder"
CLASS_LABELS = {
    "Mood disorder":                        "Humor",
    "Addictive disorder":                   "Adição",
    "Trauma and stress related disorder":   "Trauma/Estresse",
    "Schizophrenia":                        "Esquizofrenia",
    "Anxiety disorder":                     "Ansiedade",
    "Healthy control":                      "Controle",
    "Obsessive compulsive disorder":        "TOC",
}

PCA_VARIANCE_THRESHOLD = 0.95  # componentes que explicam 95% da variância


# ─────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO E LIMPEZA INICIAL
# ─────────────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> tuple[pd.DataFrame, pd.Series]:
    print("=" * 65)
    print("CARREGAMENTO E LIMPEZA INICIAL")
    print("=" * 65)

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
    print(f"Features totais: {X.shape[1]}  |  Amostras: {X.shape[0]}")
    return X, labels


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 1 — IMPUTAÇÃO DE DADOS FALTANTES
# ─────────────────────────────────────────────────────────────────────────────

def impute(X: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("ETAPA 1 — IMPUTAÇÃO (KNN Imputer, k=5)")
    print("=" * 65)

    missing = X.isnull().sum()
    missing = missing[missing > 0]
    print("Valores ausentes por coluna antes da imputação:")
    for col, n in missing.items():
        print(f"  {col:<12} {n:>3} NaN  ({100*n/len(X):.1f}%)")

    # KNNImputer: estima o valor de cada célula faltante como a média
    # ponderada dos k vizinhos mais próximos no espaço das demais features.
    # É mais preciso que a mediana simples pois considera correlações entre variáveis.
    imputer = KNNImputer(n_neighbors=5, weights="distance")
    X_imp = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print(f"NaN restantes após imputação: {X_imp.isnull().sum().sum()}")
    return X_imp, imputer


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 2 — BALANCEAMENTO DE CLASSES (SMOTE)
# ─────────────────────────────────────────────────────────────────────────────

def balance(X: pd.DataFrame, labels: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    print("\n" + "=" * 65)
    print("ETAPA 2 — BALANCEAMENTO (SMOTE)")
    print("=" * 65)

    le = LabelEncoder()
    y = le.fit_transform(labels)

    print("Distribuição de classes ANTES do SMOTE:")
    vc_before = labels.value_counts().sort_values()
    for cls, n in vc_before.items():
        bar = "█" * (n // 5)
        print(f"  {cls:<26} {n:>4}  {bar}")

    # SMOTE (Synthetic Minority Over-sampling Technique): gera amostras sintéticas
    # para as classes minoritárias interpolando entre amostras reais no espaço
    # de features. Evita simples duplicação (que causaria overfitting).
    # k_neighbors=5 é o padrão; deve ser menor que a menor classe (46 amostras → ok).
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_res, y_res = smote.fit_resample(X.values, y)

    labels_res = pd.Series(le.inverse_transform(y_res), name=labels.name)
    X_res = pd.DataFrame(X_res, columns=X.columns)

    print("\nDistribuição de classes APÓS o SMOTE:")
    vc_after = labels_res.value_counts().sort_values()
    for cls, n in vc_after.items():
        bar = "█" * (n // 5)
        print(f"  {cls:<26} {n:>4}  {bar}")

    print(f"\nAmostras antes: {len(labels)}  →  depois: {len(labels_res)}")
    return X_res, labels_res, le


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 3 — NORMALIZAÇÃO (StandardScaler)
# ─────────────────────────────────────────────────────────────────────────────

def normalize(X: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    print("\n" + "=" * 65)
    print("ETAPA 3 — NORMALIZAÇÃO (StandardScaler)")
    print("=" * 65)

    # StandardScaler: subtrai a média e divide pelo desvio padrão de cada feature,
    # resultando em média ≈ 0 e desvio ≈ 1. Necessário antes do PCA, pois features
    # com escalas maiores dominariam as componentes principais.
    scaler = StandardScaler()
    X_sc = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    sample = X_sc[["age", "IQ", "education"]].describe().loc[["mean", "std"]]
    print("Média e desvio após normalização (amostra — metadados):")
    print(sample.round(4).to_string())
    return X_sc, scaler


# ─────────────────────────────────────────────────────────────────────────────
# ETAPA 4 — REDUÇÃO DE DIMENSIONALIDADE (PCA)
# ─────────────────────────────────────────────────────────────────────────────

def apply_pca(X: pd.DataFrame, variance_threshold: float = PCA_VARIANCE_THRESHOLD
              ) -> tuple[np.ndarray, PCA]:
    print("\n" + "=" * 65)
    print(f"ETAPA 4 — PCA (variância retida ≥ {variance_threshold*100:.0f}%)")
    print("=" * 65)

    # Fit completo para inspecionar a curva de variância acumulada
    pca_full = PCA(random_state=42)
    pca_full.fit(X.values)

    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    n_components = int(np.searchsorted(cumvar, variance_threshold)) + 1

    print(f"Features originais : {X.shape[1]}")
    print(f"Componentes PCA    : {n_components}  (retendo {cumvar[n_components-1]*100:.2f}% da variância)")

    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X.values)

    col_names = [f"PC{i+1}" for i in range(n_components)]
    X_pca_df  = pd.DataFrame(X_pca, columns=col_names)
    return X_pca_df, pca, cumvar


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def plot_nan_before_imputation(X_raw: pd.DataFrame) -> None:
    missing = X_raw.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(missing.index, missing.values, color="#e15759")
    ax.bar_label(ax.containers[0], padding=3)
    ax.set_ylabel("Quantidade de NaN")
    ax.set_title("Valores Ausentes por Feature (antes da imputação)")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pp_01_nan_antes_imputacao.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] pp_01_nan_antes_imputacao.png")


def plot_class_balance(labels_before: pd.Series, labels_after: pd.Series) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Balanceamento de Classes", fontsize=13, fontweight="bold")

    palette = sns.color_palette("tab10", labels_before.nunique())

    for ax, labels, title in zip(
        axes,
        [labels_before, labels_after],
        ["Antes do SMOTE", "Após o SMOTE"],
    ):
        vc = labels.value_counts()
        bars = ax.barh(vc.index, vc.values, color=palette[: len(vc)])
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_xlabel("Número de Amostras")
        ax.set_title(title)
        ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pp_02_balanceamento.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] pp_02_balanceamento.png")


def plot_pca_variance(cumvar: np.ndarray, n_components: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Análise de Componentes Principais (PCA)", fontsize=13, fontweight="bold")

    max_plot = min(100, len(cumvar))

    ax = axes[0]
    ax.plot(range(1, max_plot + 1),
            cumvar[:max_plot] * 100, "o-", ms=3, color="#4e79a7")
    ax.axhline(PCA_VARIANCE_THRESHOLD * 100, color="red", linestyle="--",
               label=f"{PCA_VARIANCE_THRESHOLD*100:.0f}%")
    ax.axvline(n_components, color="red", linestyle="--")
    ax.annotate(
        f"n={n_components}",
        xy=(n_components, cumvar[n_components - 1] * 100),
        xytext=(n_components + 3, cumvar[n_components - 1] * 100 - 8),
        arrowprops=dict(arrowstyle="->", color="red"),
        color="red", fontsize=9,
    )
    ax.set_xlabel("Número de Componentes")
    ax.set_ylabel("Variância Acumulada (%)")
    ax.set_title("Variância Acumulada")
    ax.legend()

    ax = axes[1]
    first_n = 30
    ax.bar(range(1, first_n + 1),
           cumvar[:first_n] * 100 - np.concatenate([[0], cumvar[:first_n - 1] * 100]),
           color="#59a14f")
    ax.set_xlabel("Componente Principal")
    ax.set_ylabel("Variância Explicada (%)")
    ax.set_title(f"Variância por Componente (primeiros {first_n})")

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pp_03_pca_variancia.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] pp_03_pca_variancia.png")


def plot_pca_scatter(X_pca: pd.DataFrame, labels: pd.Series, pca: PCA) -> None:
    classes = labels.unique()
    palette = dict(zip(classes, sns.color_palette("tab10", len(classes))))

    fig, ax = plt.subplots(figsize=(9, 6))
    for cls in classes:
        mask = labels == cls
        ax.scatter(
            X_pca.loc[mask, "PC1"], X_pca.loc[mask, "PC2"],
            label=cls, alpha=0.45, s=14, color=palette[cls],
        )
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.set_title("PCA — PC1 × PC2 (dados balanceados)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=7, markerscale=1.8, framealpha=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pp_04_pca_scatter.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] pp_04_pca_scatter.png")


# ─────────────────────────────────────────────────────────────────────────────
# SALVAR ARTEFATOS
# ─────────────────────────────────────────────────────────────────────────────

def save_artifacts(X_pca: pd.DataFrame, labels: pd.Series, le: LabelEncoder,
                   imputer: KNNImputer, scaler: StandardScaler, pca: PCA) -> None:
    print("\n" + "=" * 65)
    print("SALVANDO ARTEFATOS")
    print("=" * 65)

    X_pca.to_csv(f"{OUTPUT_DIR}/X_preprocessed.csv", index=False)
    print(f"[Salvo] X_preprocessed.csv  — shape {X_pca.shape}")

    labels.to_csv(f"{OUTPUT_DIR}/y_labels.csv", index=False)
    print(f"[Salvo] y_labels.csv        — {len(labels)} amostras, {labels.nunique()} classes")

    y_encoded = pd.Series(le.transform(labels), name="label_encoded")
    y_encoded.to_csv(f"{OUTPUT_DIR}/y_encoded.csv", index=False)
    print(f"[Salvo] y_encoded.csv       — mapeamento: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    artifacts = {"imputer": imputer, "scaler": scaler, "pca": pca, "label_encoder": le}
    with open(f"{OUTPUT_DIR}/preprocessors.pkl", "wb") as f:
        pickle.dump(artifacts, f)
    print(f"[Salvo] preprocessors.pkl   — imputer, scaler, pca, label_encoder")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    # Carregamento
    X_raw, labels = load_raw(DATA_PATH)

    # Visualização dos NaN antes da imputação
    plot_nan_before_imputation(X_raw)

    # Etapa 1 — Imputação
    X_imp, imputer = impute(X_raw)

    # Etapa 2 — Balanceamento
    labels_before = labels.copy()
    X_bal, labels_bal, le = balance(X_imp, labels)
    plot_class_balance(labels_before, labels_bal)

    # Etapa 3 — Normalização
    X_norm, scaler = normalize(X_bal)

    # Etapa 4 — PCA
    X_pca, pca, cumvar = apply_pca(X_norm)
    plot_pca_variance(cumvar, X_pca.shape[1])
    plot_pca_scatter(X_pca, labels_bal, pca)

    # Salvar
    save_artifacts(X_pca, labels_bal, le, imputer, scaler, pca)

    print("\n" + "=" * 65)
    print("PRÉ-PROCESSAMENTO CONCLUÍDO")
    print("=" * 65)
    print(f"  Dataset final : {X_pca.shape[0]} amostras × {X_pca.shape[1]} componentes PCA")
    print(f"  Arquivos em   : ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
