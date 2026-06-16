import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings("ignore")

DATA_PATH  = "EEG.machinelearing_data_BRMH.csv"
OUTPUT_DIR = "outputs"

CLASS_LABELS_MAIN = {
    "Mood disorder":                       "Humor",
    "Addictive disorder":                  "Adição",
    "Trauma and stress related disorder":  "Trauma/Estresse",
    "Schizophrenia":                       "Esquizofrenia",
    "Anxiety disorder":                    "Ansiedade",
    "Healthy control":                     "Controle",
    "Obsessive compulsive disorder":       "TOC",
}

plt.rcParams.update({"figure.dpi": 130, "font.size": 9})


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    drop = ["no.", "eeg.date"] + [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=[c for c in drop if c in df.columns])
    df["main_disorder"]     = df["main.disorder"].map(CLASS_LABELS_MAIN)
    df["specific_disorder"] = df["specific.disorder"]
    df = df.drop(columns=["main.disorder", "specific.disorder"])
    df["sex_label"] = df["sex"].map({"M": "Masculino", "F": "Feminino"})
    df["age_group"] = pd.cut(
        df["age"],
        bins=[17, 30, 40, 50, 60, 100],
        labels=["18-30", "31-40", "41-50", "51-60", "61+"],
    )
    edu_med = df["education"].median()
    df["edu_group"] = pd.cut(
        df["education"].fillna(edu_med),
        bins=[-np.inf, 11, 13, 15, np.inf],
        labels=["≤11 anos", "12-13 anos", "14-15 anos", "≥16 anos"],
    )
    return df


def _equity_heatmap(df: pd.DataFrame, demo_col: str, disorder_col: str,
                    ax: plt.Axes, title: str) -> None:
    ct      = pd.crosstab(df[demo_col], df[disorder_col])
    ct_norm = ct.div(ct.sum(axis=1), axis=0)
    sns.heatmap(
        ct_norm, annot=True, fmt=".0%", cmap="YlOrRd",
        ax=ax, linewidths=0.3, cbar_kws={"shrink": 0.7},
        vmin=0, vmax=ct_norm.values.max(),
    )
    ax.set_title(title, fontsize=8, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="x", rotation=40, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)


def plot_eda(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(22, 8), layout="constrained")
    gs = gridspec.GridSpec(1, 6, figure=fig)

    ax_main = fig.add_subplot(gs[0, 0:2])
    vc = df["main_disorder"].value_counts()
    bars = ax_main.barh(vc.index, vc.values, color=sns.color_palette("tab10", len(vc)))
    ax_main.bar_label(bars, fmt="%d", padding=3, fontsize=7)
    ax_main.set_xlim(0, vc.values.max() * 1.20)
    ax_main.set_xlabel("N amostras")
    ax_main.tick_params(labelsize=7)
    ax_main.invert_yaxis()

    ax_spec = fig.add_subplot(gs[0, 2:4])
    vc2 = df["specific_disorder"].value_counts()
    bars2 = ax_spec.barh(vc2.index, vc2.values, color=sns.color_palette("tab20", len(vc2)))
    ax_spec.bar_label(bars2, fmt="%d", padding=3, fontsize=7)
    ax_spec.set_xlim(0, vc2.values.max() * 1.20)
    ax_spec.set_xlabel("N amostras")
    ax_spec.tick_params(labelsize=7)
    ax_spec.invert_yaxis()

    ax_age = fig.add_subplot(gs[0, 4])
    ax_age.hist(df["age"].dropna(), bins=20, color="#4e79a7", edgecolor="white")
    ax_age.set_xlabel("Idade")
    ax_age.set_ylabel("N")
    ax_age.tick_params(labelsize=7)

    ax_sex = fig.add_subplot(gs[0, 5])
    sc = df["sex_label"].value_counts()
    ax_sex.pie(sc.values, labels=sc.index, autopct="%1.1f%%",
               colors=["#4e79a7", "#e15759"], startangle=90,
               textprops={"fontsize": 8})

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "eda.png")
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"[Salvo] {path}")


def plot_equidade(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(22, 18))
    gs = gridspec.GridSpec(3, 6, figure=fig, hspace=0.65, wspace=0.45)

    ax_sex_main = fig.add_subplot(gs[0, 0:3])
    _equity_heatmap(df, "sex_label", "main_disorder", ax_sex_main,
                    "Sexo × Main Disorder — P(transtorno | sexo)")

    ax_sex_spec = fig.add_subplot(gs[0, 3:6])
    _equity_heatmap(df, "sex_label", "specific_disorder", ax_sex_spec,
                    "Sexo × Specific Disorder — P(transtorno | sexo)")

    ax_age_main = fig.add_subplot(gs[1, 0:3])
    _equity_heatmap(df, "age_group", "main_disorder", ax_age_main,
                    "Faixa Etária × Main Disorder — P(transtorno | faixa etária)")

    ax_age_spec = fig.add_subplot(gs[1, 3:6])
    _equity_heatmap(df, "age_group", "specific_disorder", ax_age_spec,
                    "Faixa Etária × Specific Disorder — P(transtorno | faixa etária)")

    ax_edu_main = fig.add_subplot(gs[2, 0:3])
    _equity_heatmap(df, "edu_group", "main_disorder", ax_edu_main,
                    "Educação × Main Disorder — P(transtorno | escolaridade)")

    ax_edu_spec = fig.add_subplot(gs[2, 3:6])
    _equity_heatmap(df, "edu_group", "specific_disorder", ax_edu_spec,
                    "Educação × Specific Disorder — P(transtorno | escolaridade)")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, "equidade.png")
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"[Salvo] {path}")


def main() -> None:
    df = load_data(DATA_PATH)
    plot_eda(df)
    plot_equidade(df)


if __name__ == "__main__":
    main()
