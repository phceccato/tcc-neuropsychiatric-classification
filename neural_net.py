"""
Experimentos de Linha de Base — Cenário A
Classificação de Transtornos Neuropsiquiátricos (EEG BRMH)
Park et al. (2021) — https://osf.io/8bsvr

Pipeline por fold: KNNImputer → ADASYN → StandardScaler → (PCA em A4)
"""

import json
import os
import random
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")

# Reprodutibilidade
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Configuração global
DATA_PATH  = "EEG.machinelearing_data_BRMH.csv"
OUTPUT_DIR = "outputs/baseline"
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_FOLDS   = 5
TEST_SIZE = 0.25   # split 75/25 estratificado por specific_disorder
PCA_VAR   = 0.95   # variância retida no PCA (A4/B4)

DEMO_COLS = ["sex", "age", "education", "IQ"]

CLASS_LABELS_MAIN = {
    "Mood disorder":                       "Humor",
    "Addictive disorder":                  "Adição",
    "Trauma and stress related disorder":  "Trauma/Estresse",
    "Schizophrenia":                       "Esquizofrenia",
    "Anxiety disorder":                    "Ansiedade",
    "Healthy control":                     "Controle",
    "Obsessive compulsive disorder":       "TOC",
}

# sub_id → (nome do conjunto de features, aplicar PCA)
SUBCENARIOS: dict[str, tuple[str, bool]] = {
    "1": ("PSD",    False),
    "2": ("FC",     False),
    "3": ("PSD+FC", False),
    "4": ("PSD+FC", True),
}

# Cenário A — hiperparâmetros fixos
A_HP: dict = {
    "hidden_dims":  [1024, 512, 256, 128, 64],
    "dropout":      0.30,
    "lr":           1e-5,
    "batch_size":   32,
    "weight_decay": 1e-4,
    "optimizer":    "Adam",
    "epochs":       200,
    "patience":     20,
}

print(f"Device: {DEVICE}")


# Versionamento de bibliotecas

def save_library_versions(out_dir: str = OUTPUT_DIR) -> None:
    import sklearn, imblearn, optuna
    versions = {
        "python":            sys.version,
        "numpy":             np.__version__,
        "pandas":            pd.__version__,
        "torch":             torch.__version__,
        "scikit-learn":      sklearn.__version__,
        "imbalanced-learn":  imblearn.__version__,
        "optuna":            optuna.__version__,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "library_versions.json")
    with open(path, "w") as f:
        json.dump(versions, f, indent=2)
    print(f"[Salvo] library_versions.json")


# Carregamento

def load_raw() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Retorna (X, y_main, y_specific) sem qualquer pré-processamento."""
    df = pd.read_csv(DATA_PATH)

    drop = ["no.", "eeg.date"] + [c for c in df.columns if c.startswith("Unnamed")]
    df = df.drop(columns=[c for c in drop if c in df.columns])

    df["sex"] = (df["sex"] == "M").astype(int)

    y_main     = df["main.disorder"].map(CLASS_LABELS_MAIN)
    y_specific = df["specific.disorder"]

    df = df.drop(columns=["main.disorder", "specific.disorder"])

    eeg_cols     = [c for c in df.columns if c.startswith(("AB.", "COH."))]
    feature_cols = DEMO_COLS + eeg_cols
    X = df[feature_cols].copy()

    print(f"\nDataset carregado: {X.shape[0]} pacientes × {X.shape[1]} atributos")
    print(f"  PSD  : {sum(c.startswith('AB.')  for c in eeg_cols)} features")
    print(f"  FC   : {sum(c.startswith('COH.') for c in eeg_cols)} features")
    print(f"  Demo : {len(DEMO_COLS)} features")
    print(f"  main.disorder     : {y_main.nunique()} classes")
    print(f"  specific.disorder : {y_specific.nunique()} classes")

    return X, y_main, y_specific


# Seleção de features

def select_features(X: pd.DataFrame, feat_set: str) -> pd.DataFrame:
    psd = [c for c in X.columns if c.startswith("AB.")]
    fc  = [c for c in X.columns if c.startswith("COH.")]

    match feat_set:
        case "PSD":
            cols = psd
        case "FC":
            cols = fc
        case "PSD+FC":
            cols = psd + fc
        case _:
            raise ValueError(f"Conjunto de features desconhecido: {feat_set}")

    return X[[c for c in cols if c in X.columns]]


# Pré-processamento dentro do fold (sem leakage)

def _safe_balance(
    X_tr: np.ndarray, y_tr: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """ADASYN com fallback para SMOTE. Loga quando o método principal falha."""
    min_count = int(np.bincount(y_tr).min())
    try:
        n_neighbors = min(5, min_count - 1)
        sampler = ADASYN(random_state=SEED, n_neighbors=n_neighbors)
        return sampler.fit_resample(X_tr, y_tr)
    except Exception as e:
        print(f"    [AVISO] ADASYN falhou ({e}), tentando SMOTE...")
    try:
        k = max(1, min(5, min_count - 1))
        sampler = SMOTE(random_state=SEED, k_neighbors=k)
        result = sampler.fit_resample(X_tr, y_tr)
        print("    [AVISO] SMOTE utilizado no lugar de ADASYN.")
        return result
    except Exception as e:
        print(f"    [AVISO] SMOTE também falhou ({e}). Sem balanceamento neste fold.")
        return X_tr, y_tr


_ES_VAL_FRAC = 0.15   # fração do fold-treino reservada para early stopping


def _preprocess(
    X_tr: np.ndarray, y_tr: np.ndarray,
    extra_sets: list[np.ndarray],
    apply_pca: bool = False,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """
    Ajusta o pipeline em X_tr e aplica em todos os conjuntos de extra_sets.
    KNNImputer(k=5, weights='uniform') → ADASYN → StandardScaler → PCA (opcional).
    Retorna (X_tr_proc, y_tr_bal, [X_extra_proc, ...]).
    """
    # 1. Imputação KNN — weights='uniform' conforme especificação
    imputer = KNNImputer(n_neighbors=5)
    X_tr = imputer.fit_transform(X_tr)
    extra_sets = [imputer.transform(X) for X in extra_sets]

    # 2. Balanceamento ADASYN — somente no treino
    X_tr, y_tr = _safe_balance(X_tr, y_tr)

    # 3. Normalização z-score
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    extra_sets = [scaler.transform(X) for X in extra_sets]

    # 4. PCA — apenas subcenários *4
    if apply_pca:
        pca = PCA(n_components=PCA_VAR, svd_solver="full", random_state=SEED)
        X_tr = pca.fit_transform(X_tr)
        extra_sets = [pca.transform(X) for X in extra_sets]

    return X_tr, y_tr, extra_sets


# Modelo (MLP)

class EEGClassifier(nn.Module):
    """MLP tronco-compartilhável: Linear → BatchNorm → ReLU → Dropout × N + head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: list[int],
        n_classes: int,
        dropout: float,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# DataLoaders

def _make_loader(
    X: np.ndarray, y: np.ndarray,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    ds  = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.int64)),
    )
    gen = torch.Generator().manual_seed(SEED) if shuffle else None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      generator=gen, drop_last=drop_last)


def make_loaders(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    batch_size: int,
) -> tuple[DataLoader, DataLoader]:
    return (
        _make_loader(X_tr, y_tr, batch_size, shuffle=True),
        _make_loader(X_te, y_te, batch_size),
    )


# Otimizador

def build_optimizer(
    name: str,
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    params = model.parameters()
    match name:
        case "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        case "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        case _:
            raise ValueError(f"Otimizador desconhecido: {name}")


# Treinamento

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    patience: int,
) -> tuple[list[float], list[float]]:
    """Treina com early stopping. Retorna (train_losses, val_losses) por época."""
    best_val_loss = float("inf")
    patience_count = 0
    best_state: dict | None = None
    train_losses: list[float] = []
    val_losses:   list[float] = []

    for _ in range(1, epochs + 1):
        # treino
        model.train()
        tl, tn = 0.0, 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            optimizer.step()
            tl += loss.item() * len(yb)
            tn += len(yb)
        train_losses.append(tl / tn)

        # validação
        model.eval()
        vl, n = 0.0, 0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                vl += criterion(model(Xb), yb).item() * len(yb)
                n  += len(yb)
        vl /= n
        val_losses.append(vl)
        scheduler.step(vl)

        if vl < best_val_loss:
            best_val_loss = vl
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return train_losses, val_losses


# Métricas

@torch.no_grad()
def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    n_classes: int,
) -> dict:
    model.eval()
    preds, probs, targets = [], [], []

    for Xb, yb in loader:
        logits = model(Xb.to(DEVICE))
        probs.extend(torch.softmax(logits, dim=1).cpu().numpy())
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(yb.numpy())

    y_true = np.array(targets)
    y_pred = np.array(preds)
    y_prob = np.array(probs)
    labels = list(range(n_classes))

    acc      = accuracy_score(y_true, y_pred)
    bal_acc  = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    kappa    = cohen_kappa_score(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred, labels=labels, normalize="true")
    per_cls  = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )

    try:
        auc = roc_auc_score(
            y_true, y_prob, multi_class="ovr", average="macro", labels=labels
        )
    except ValueError:
        auc = float("nan")

    return {
        "accuracy":          float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1":          float(macro_f1),
        "auc_roc":           float(auc),
        "kappa":             float(kappa),
        "confusion_matrix":  cm,
        "per_class":         per_cls,
    }


# Execução de um fold completo

def run_fold(
    X_tr_raw: np.ndarray,
    y_tr_raw: np.ndarray,
    X_te_raw: np.ndarray,
    y_te_raw: np.ndarray,
    n_classes: int,
    hp: dict,
    apply_pca: bool = False,
) -> dict:
    """
    Pipeline completo de um fold sem data leakage.

    Estrutura interna:
        fold-treino (X_tr_raw)
        ├─ sub-treino (85%) → treinamento + ADASYN
        └─ es-val     (15%) → early stopping (único uso)
        fold-teste (X_te_raw)  → métricas finais (nunca visto pelo modelo)
    """
    # ── Split interno: 85% sub-treino / 15% early-stopping val ───────────────
    try:
        X_sub, X_es, y_sub, y_es = train_test_split(
            X_tr_raw, y_tr_raw,
            test_size=_ES_VAL_FRAC, stratify=y_tr_raw, random_state=SEED,
        )
    except ValueError:
        # Fallback sem estratificação se alguma classe tiver < 2 amostras
        X_sub, X_es, y_sub, y_es = train_test_split(
            X_tr_raw, y_tr_raw,
            test_size=_ES_VAL_FRAC, random_state=SEED,
        )

    # ── Pré-processamento: fit em X_sub, apply em X_es e X_te ────────────────
    X_sub, y_sub, (X_es, X_te) = _preprocess(
        X_sub, y_sub, [X_es, X_te_raw], apply_pca
    )

    train_ld = _make_loader(X_sub, y_sub,    hp["batch_size"], shuffle=True, drop_last=True)
    es_ld    = _make_loader(X_es,  y_es,     hp["batch_size"])
    test_ld  = _make_loader(X_te,  y_te_raw, hp["batch_size"])

    model = EEGClassifier(
        X_sub.shape[1], hp["hidden_dims"], n_classes, hp["dropout"]
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = build_optimizer(hp["optimizer"], model, hp["lr"], hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.9, patience=3, min_lr=1e-6
    )

    # Early stopping monitorado em es_ld; test_ld nunca influencia o treino
    train_losses, val_losses = train_model(
        model, train_ld, es_ld, criterion, optimizer, scheduler,
        hp["epochs"], hp["patience"],
    )

    metrics = compute_metrics(model, test_ld, n_classes)
    metrics["train_losses"] = train_losses
    metrics["val_losses"]   = val_losses
    return metrics


# Agregação de métricas dos folds

def aggregate_metrics(fold_metrics: list[dict]) -> dict:
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "auc_roc", "kappa"]
    agg: dict = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if not np.isnan(m[k])]
        agg[k] = {
            "mean":   float(np.mean(vals)),
            "std":    float(np.std(vals)),
            "values": [float(v) for v in vals],
        }
    return agg


# Visualizações

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names,
    label: str,
    out_dir: str,
) -> None:
    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, linewidths=0.4, vmin=0.0, vmax=1.0,
    )
    ax.set_title(f"Matriz de Confusão (normalizada) — {label}",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    ax.tick_params(axis="x", rotation=35, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label}_cm.png")
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [Salvo] {label}_cm.png")


def plot_loss_curves(
    fold_metrics: list[dict],
    label: str,
    out_dir: str,
) -> None:
    n_folds = len(fold_metrics)
    fig, axes = plt.subplots(1, n_folds, figsize=(4 * n_folds, 4), sharey=False)
    if n_folds == 1:
        axes = [axes]

    for i, (ax, m) in enumerate(zip(axes, fold_metrics), start=1):
        tl = m["train_losses"]
        vl = m["val_losses"]
        epochs = range(1, len(tl) + 1)
        ax.plot(epochs, tl, label="Train", linewidth=1.2)
        ax.plot(epochs, vl, label="Val",   linewidth=1.2)
        ax.set_title(f"Fold {i}", fontsize=9)
        ax.set_xlabel("Época", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=7)

    fig.suptitle(f"Loss curves — {label}", fontsize=11, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(out_dir, f"{label}_loss.png")
    plt.savefig(path, bbox_inches="tight", dpi=130)
    plt.close()
    print(f"  [Salvo] {label}_loss.png")


def _json_serial(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"Não serializável: {type(obj)}")


# Execução de um subcenário completo

def run_subcenario(
    scenario: str,         # "A" ou "B"
    sub_id: str,           # "1" … "4"
    target_name: str,      # "main" ou "specific"
    X_all: pd.DataFrame,
    y_main: pd.Series,
    y_specific: pd.Series,
    hp: dict,
    out_dir: str,
) -> dict:
    feat_name, apply_pca = SUBCENARIOS[sub_id]
    label = f"{scenario}{sub_id}_{target_name}"
    feat_label = f"{feat_name}+PCA" if apply_pca else feat_name

    print(f"\n{'='*65}")
    print(f"  {label}  |  features: {feat_label}  |  target: {target_name}")
    print(f"{'='*65}")

    # Encoding do target
    y_raw = y_main if target_name == "main" else y_specific
    le = LabelEncoder()
    y_enc = le.fit_transform(y_raw)
    n_classes = len(le.classes_)

    # Estratificador para o split externo: sempre specific_disorder
    y_strat = LabelEncoder().fit_transform(y_specific)

    X_feat = select_features(X_all, feat_name)
    X_np   = X_feat.values.astype(np.float32)

    print(f"\n  Dataset: {X_np.shape[0]} amostras × {X_np.shape[1]} features")
    print(f"  Distribuição das classes ({target_name}):")
    counts = pd.Series(y_enc).value_counts().sort_index()
    for idx, cnt in counts.items():
        print(f"    [{idx}] {le.classes_[idx]:<40} {cnt:>4} amostras")

    # Split 75/25 estratificado por specific_disorder
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_enc,
        test_size=TEST_SIZE,
        stratify=y_strat,
        random_state=SEED,
    )

    # ── Validação cruzada 5-fold no conjunto de treino ────────────────────────
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_metrics: list[dict] = []

    for fold_i, (tr_idx, va_idx) in enumerate(
        skf.split(X_train, y_train), start=1
    ):
        print(f"  Fold {fold_i}/{N_FOLDS}", end="  ", flush=True)
        m = run_fold(
            X_train[tr_idx], y_train[tr_idx],
            X_train[va_idx], y_train[va_idx],
            n_classes=n_classes, hp=hp, apply_pca=apply_pca,
        )
        fold_metrics.append(m)
        print(
            f"BalAcc={m['balanced_accuracy']:.3f}  "
            f"F1={m['macro_f1']:.3f}  "
            f"AUC={m['auc_roc']:.3f}  "
            f"κ={m['kappa']:.3f}"
        )

    cv_agg = aggregate_metrics(fold_metrics)
    os.makedirs(out_dir, exist_ok=True)
    plot_loss_curves(fold_metrics, label, out_dir)

    # ── Avaliação final: treino completo → conjunto de teste ──────────────────
    print(f"  Teste holdout...", end="  ", flush=True)
    test_m = run_fold(
        X_train, y_train, X_test, y_test,
        n_classes=n_classes, hp=hp, apply_pca=apply_pca,
    )
    print(
        f"BalAcc={test_m['balanced_accuracy']:.3f}  "
        f"F1={test_m['macro_f1']:.3f}  "
        f"AUC={test_m['auc_roc']:.3f}  "
        f"κ={test_m['kappa']:.3f}"
    )

    # ── Saídas ────────────────────────────────────────────────────────────────
    os.makedirs(out_dir, exist_ok=True)
    plot_confusion_matrix(test_m["confusion_matrix"], le.classes_, label, out_dir)

    result = {
        "scenario":    label,
        "feature_set": feat_label,
        "target":      target_name,
        "n_classes":   n_classes,
        "classes":     list(le.classes_),
        "hp":          {k: v for k, v in hp.items() if k not in ("epochs", "patience")},
        "epochs":      hp["epochs"],
        "patience":    hp["patience"],
        "cv_metrics":  cv_agg,
        "test_metrics": {k: v for k, v in test_m.items()
                         if k not in ("confusion_matrix", "per_class",
                                      "train_losses", "val_losses")},
        "test_confusion_matrix": test_m["confusion_matrix"].tolist(),
        "test_per_class":        test_m["per_class"],
    }

    with open(os.path.join(out_dir, f"{label}_results.json"), "w") as f:
        json.dump(result, f, indent=2, default=_json_serial)

    _print_cv_summary(label, cv_agg)
    return result


def _print_cv_summary(label: str, cv: dict) -> None:
    print(f"\n  Resumo CV — {label}:")
    for k, name in [
        ("balanced_accuracy", "Balanced Accuracy"),
        ("macro_f1",          "Macro F1        "),
        ("auc_roc",           "AUC-ROC         "),
        ("kappa",             "Cohen's κ       "),
    ]:
        print(f"    {name}: {cv[k]['mean']:.4f} ± {cv[k]['std']:.4f}")


# Tabela resumo

def save_summary(
    results: list[dict],
    target: str,
    out_dir: str,
    filename: str | None = None,
) -> pd.DataFrame:
    rows = []
    for r in results:
        cv = r["cv_metrics"]
        tm = r["test_metrics"]
        rows.append({
            "Cenário":          r["scenario"],
            "Features":         r["feature_set"],
            "Target":           r["target"],
            "CV Acc":           f"{cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}",
            "CV BalAcc":        f"{cv['balanced_accuracy']['mean']:.4f} ± {cv['balanced_accuracy']['std']:.4f}",
            "CV Macro F1":      f"{cv['macro_f1']['mean']:.4f} ± {cv['macro_f1']['std']:.4f}",
            "CV AUC-ROC":       f"{cv['auc_roc']['mean']:.4f} ± {cv['auc_roc']['std']:.4f}",
            "CV κ":             f"{cv['kappa']['mean']:.4f} ± {cv['kappa']['std']:.4f}",
            "Test Acc":         f"{tm['accuracy']:.4f}",
            "Test BalAcc":      f"{tm['balanced_accuracy']:.4f}",
            "Test Macro F1":    f"{tm['macro_f1']:.4f}",
            "Test AUC-ROC":     f"{tm['auc_roc']:.4f}",
            "Test κ":           f"{tm['kappa']:.4f}",
        })

    df = pd.DataFrame(rows)
    fname = filename or f"summary_{target}.csv"
    path  = os.path.join(out_dir, fname)
    df.to_csv(path, index=False)

    print(f"\n{'='*65}")
    print(f"TABELA RESUMO — {target.upper()}")
    print(f"{'='*65}")
    print(df.to_string(index=False))
    print(f"\n[Salvo] {fname}")
    return df


# Cenário A

def run_scenario_a(
    X_all: pd.DataFrame,
    y_main: pd.Series,
    y_specific: pd.Series,
) -> None:
    print("\n" + "=" * 65)
    print("CENÁRIO A — HIPERPARÂMETROS FIXOS")
    print(f"  Arquitetura : {A_HP['hidden_dims']}")
    print(f"  Otimizador  : {A_HP['optimizer']}  lr={A_HP['lr']}  wd={A_HP['weight_decay']}")
    print(f"  Batch       : {A_HP['batch_size']}  Dropout: {A_HP['dropout']}")
    print("=" * 65)

    out_base = os.path.join(OUTPUT_DIR, "scenario_a")
    results_main:     list[dict] = []
    results_specific: list[dict] = []

    for sub_id in ("1", "2", "3", "4"):
        for target_name, res_list in (
            ("main",     results_main),
            ("specific", results_specific),
        ):
            out_dir = os.path.join(out_base, target_name)
            r = run_subcenario(
                "A", sub_id, target_name,
                X_all, y_main, y_specific,
                A_HP, out_dir,
            )
            res_list.append(r)

    save_summary(results_main,     "main",     out_base)
    save_summary(results_specific, "specific", out_base)


# Entry point

def main() -> None:
    print("=" * 65)
    print("EXPERIMENTOS DE LINHA DE BASE — CENÁRIO A")
    print(f"Device: {DEVICE}  |  Seed: {SEED}")
    print("=" * 65)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_library_versions()

    X_all, y_main, y_specific = load_raw()
    run_scenario_a(X_all, y_main, y_specific)

    print("\n" + "=" * 65)
    print("CENÁRIO A CONCLUÍDO")
    print(f"Resultados em: ./{OUTPUT_DIR}/scenario_a/")
    print("=" * 65)


if __name__ == "__main__":
    main()
