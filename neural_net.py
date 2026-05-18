"""
Classificação de Transtornos Neuropsiquiátricos com Rede Neural (PyTorch)
Entrada: dados pré-processados (PCA 88 componentes, SMOTE balanceado)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score,
)

warnings.filterwarnings("ignore")

OUTPUT_DIR  = "outputs"
MODEL_PATH  = f"{OUTPUT_DIR}/neural_net.pt"
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hiperparâmetros
BATCH_SIZE    = 64
EPOCHS        = 200
LR            = 1e-3
WEIGHT_DECAY  = 1e-4
DROPOUT       = 0.3
PATIENCE      = 20        # early stopping: épocas sem melhora na val loss

HIDDEN_DIMS   = [256, 128, 64]  # camadas ocultas

print(f"Dispositivo: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# DADOS
# ─────────────────────────────────────────────────────────────────────────────

def load_data():
    X = pd.read_csv(f"{OUTPUT_DIR}/X_preprocessed.csv").values.astype(np.float32)
    y_raw = pd.read_csv(f"{OUTPUT_DIR}/y_labels.csv").iloc[:, 0]

    le = LabelEncoder()
    y = le.fit_transform(y_raw).astype(np.int64)

    # Split estratificado: 70% treino · 15% validação · 15% teste
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=42
    )

    print(f"Treino : {X_train.shape[0]} amostras")
    print(f"Val    : {X_val.shape[0]} amostras")
    print(f"Teste  : {X_test.shape[0]} amostras")
    print(f"Classes: {list(le.classes_)}")
    return X_train, X_val, X_test, y_train, y_val, y_test, le


def make_loaders(X_train, X_val, X_test, y_train, y_val, y_test):
    def to_tensor(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y))

    train_ds = to_tensor(X_train, y_train)
    val_ds   = to_tensor(X_val,   y_val)
    test_ds  = to_tensor(X_test,  y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)
    return train_loader, val_loader, test_loader


# ─────────────────────────────────────────────────────────────────────────────
# ARQUITETURA
# ─────────────────────────────────────────────────────────────────────────────

class EEGClassifier(nn.Module):
    """
    MLP com BatchNorm + ReLU + Dropout em cada camada oculta.
    Arquitetura: 88 → 256 → 128 → 64 → 7
    """
    def __init__(self, input_dim: int, hidden_dims: list[int],
                 n_classes: int, dropout: float):
        super().__init__()

        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, n_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# TREINAMENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        total_loss += loss.item() * len(y_batch)
        correct    += (logits.argmax(1) == y_batch).sum().item()
        n          += len(y_batch)
    return total_loss / n, correct / n


def train(model, train_loader, val_loader, criterion, optimizer, scheduler):
    print("\n" + "=" * 65)
    print("TREINAMENTO")
    print("=" * 65)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = eval_epoch(model, val_loader, criterion)
        scheduler.step(vl_loss)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)

        if epoch % 20 == 0 or epoch == 1:
            print(f"Época {epoch:>3}/{EPOCHS}  "
                  f"loss treino {tr_loss:.4f}  val {vl_loss:.4f}  "
                  f"acc treino {tr_acc:.3f}  val {vl_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            patience_count = 0
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                print(f"\nEarly stopping na época {epoch} "
                      f"(sem melhora por {PATIENCE} épocas).")
                break

    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"\nMelhor modelo salvo em: {MODEL_PATH}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
# AVALIAÇÃO
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(model, loader):
    model.eval()
    preds, targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(y_batch.numpy())
    return np.array(preds), np.array(targets)


def evaluate(model, test_loader, le):
    print("\n" + "=" * 65)
    print("AVALIAÇÃO — CONJUNTO DE TESTE")
    print("=" * 65)

    y_pred, y_true = predict(model, test_loader)
    class_names = le.classes_

    acc     = (y_pred == y_true).mean()
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")

    print(f"Acurácia         : {acc:.4f}")
    print(f"Balanced Accuracy: {bal_acc:.4f}")
    print(f"Macro F1         : {macro_f1:.4f}")
    print()
    print(classification_report(y_true, y_pred, target_names=class_names))

    return y_pred, y_true


# ─────────────────────────────────────────────────────────────────────────────
# VISUALIZAÇÕES
# ─────────────────────────────────────────────────────────────────────────────

def plot_learning_curves(history: dict, stopped_at: int) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Curvas de Aprendizado", fontsize=13, fontweight="bold")

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="Treino")
    ax.plot(epochs, history["val_loss"],   label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss (Cross-Entropy)")
    ax.set_title("Loss")
    ax.legend()

    ax = axes[1]
    ax.plot(epochs, history["train_acc"], label="Treino")
    ax.plot(epochs, history["val_acc"],   label="Validação")
    ax.set_xlabel("Época")
    ax.set_ylabel("Acurácia")
    ax.set_title("Acurácia")
    ax.set_ylim(0, 1)
    ax.legend()

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nn_01_learning_curves.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] nn_01_learning_curves.png")


def plot_confusion_matrix(y_true, y_pred, class_names) -> None:
    cm      = confusion_matrix(y_true, y_pred)
    cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Matriz de Confusão — Rede Neural", fontsize=13, fontweight="bold")

    for ax, data, fmt, title in zip(
        axes,
        [cm, cm_pct],
        ["d", ".2f"],
        ["Contagens", "Proporção por linha (Recall)"],
    ):
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.4,
        )
        ax.set_title(title)
        ax.set_xlabel("Predito")
        ax.set_ylabel("Real")
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.tick_params(axis="y", rotation=0,  labelsize=8)

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nn_02_confusion_matrix.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] nn_02_confusion_matrix.png")


def plot_per_class_f1(y_true, y_pred, class_names) -> None:
    report = classification_report(y_true, y_pred, target_names=class_names,
                                   output_dict=True)
    f1_scores = [report[c]["f1-score"] for c in class_names]
    colors = ["#e15759" if f < 0.6 else "#59a14f" for f in f1_scores]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(class_names, f1_scores, color=colors, edgecolor="white")
    ax.bar_label(bars, labels=[f"{v:.2f}" for v in f1_scores], padding=4, fontsize=9)
    ax.axhline(0.6, color="grey", linestyle="--", linewidth=0.8, label="F1 = 0.60")
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-score")
    ax.set_title("F1-score por Classe — Conjunto de Teste", fontsize=12, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nn_03_f1_por_classe.png", bbox_inches="tight")
    plt.close()
    print("[Salvo] nn_03_f1_por_classe.png")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("REDE NEURAL — CLASSIFICAÇÃO NEUROPSIQUIÁTRICA (EEG BRMH)")
    print("=" * 65)

    # Dados
    X_train, X_val, X_test, y_train, y_val, y_test, le = load_data()
    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test
    )

    n_classes   = len(le.classes_)
    input_dim   = X_train.shape[1]

    # Modelo
    model = EEGClassifier(input_dim, HIDDEN_DIMS, n_classes, DROPOUT).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nArquitetura : {input_dim} → {' → '.join(str(h) for h in HIDDEN_DIMS)} → {n_classes}")
    print(f"Parâmetros  : {total_params:,}")

    criterion  = nn.CrossEntropyLoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )

    # Treino
    history = train(model, train_loader, val_loader, criterion, optimizer, scheduler)
    stopped_at = len(history["train_loss"])

    # Visualizações de treino
    plot_learning_curves(history, stopped_at)

    # Avaliação
    y_pred, y_true = evaluate(model, test_loader, le)
    plot_confusion_matrix(y_true, y_pred, le.classes_)
    plot_per_class_f1(y_true, y_pred, le.classes_)

    print("\n" + "=" * 65)
    print("CONCLUÍDO")
    print("=" * 65)
    print(f"Modelo salvo em : {MODEL_PATH}")
    print(f"Gráficos em     : ./{OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
