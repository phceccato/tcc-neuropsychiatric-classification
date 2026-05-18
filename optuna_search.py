"""
Otimização de Hiperparâmetros com Optuna — Classificação Neuropsiquiátrica (EEG)
Reutiliza load_data, make_loaders e EEGClassifier de neural_net.py
"""

import json
import os
import warnings
import numpy as np
import torch
import torch.nn as nn
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_param_importances,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, balanced_accuracy_score

from neural_net import (
    EEGClassifier,
    load_data,
    make_loaders,
    evaluate,
    DEVICE,
)

warnings.filterwarnings("ignore")

# ─── Constantes configuráveis ────────────────────────────────────────────────

SEED          = 42
N_TRIALS      = 50
EPOCHS        = 100
PATIENCE      = 15

N_LAYERS_MIN  = 2
N_LAYERS_MAX  = 4
UNITS_MIN     = 64
UNITS_MAX     = 512
LR_MIN        = 1e-5
LR_MAX        = 1e-2
DROPOUT_MIN   = 0.1
DROPOUT_MAX   = 0.6
WD_MIN        = 1e-6
WD_MAX        = 1e-2
BATCH_CHOICES = [16, 32, 64, 128]
OPTIMIZERS    = ["Adam", "AdamW", "RMSprop"]

OUTPUT_DIR    = os.path.join("outputs", "optuna")
os.makedirs(OUTPUT_DIR, exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

optuna.logging.set_verbosity(optuna.logging.WARNING)


# ─── Utilidades de treino (trial) ────────────────────────────────────────────

def build_optimizer(name: str, model: nn.Module, lr: float,
                    weight_decay: float) -> torch.optim.Optimizer:
    params = model.parameters()
    match name:
        case "Adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        case "AdamW":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        case "RMSprop":
            return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Optimizer desconhecido: {name}")


def _run_epoch(model: nn.Module, loader, criterion,
               optimizer=None) -> tuple[float, float]:
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()

    with ctx:
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            if training:
                optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            if training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y_batch)
            correct    += (logits.argmax(1) == y_batch).sum().item()
            n          += len(y_batch)

    return total_loss / n, correct / n


def train_trial(model: nn.Module, train_loader, val_loader,
                optimizer, trial: optuna.Trial) -> float:
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    best_val_loss = float("inf")
    patience_count = 0

    for epoch in range(1, EPOCHS + 1):
        _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        # Pruning: encerra trials ruins cedo
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    return best_val_loss


@torch.no_grad()
def macro_f1_on_loader(model: nn.Module, loader) -> tuple[float, float, float]:
    model.eval()
    preds, targets = [], []
    for X_batch, y_batch in loader:
        logits = model(X_batch.to(DEVICE))
        preds.extend(logits.argmax(1).cpu().numpy())
        targets.extend(y_batch.numpy())
    y_pred = np.array(preds)
    y_true = np.array(targets)
    return (
        f1_score(y_true, y_pred, average="macro"),
        (y_pred == y_true).mean(),
        balanced_accuracy_score(y_true, y_pred),
    )


# ─── Objective ───────────────────────────────────────────────────────────────

def make_objective(X_train, X_val, X_test, y_train, y_val, y_test,
                   n_classes: int, input_dim: int):

    def objective(trial: optuna.Trial) -> float:
        # Hiperparâmetros sugeridos pelo Optuna
        lr           = trial.suggest_float("lr",           LR_MIN,      LR_MAX,  log=True)
        dropout      = trial.suggest_float("dropout",      DROPOUT_MIN, DROPOUT_MAX)
        weight_decay = trial.suggest_float("weight_decay", WD_MIN,      WD_MAX,  log=True)
        batch_size   = trial.suggest_categorical("batch_size", BATCH_CHOICES)
        optimizer_name = trial.suggest_categorical("optimizer", OPTIMIZERS)
        n_layers     = trial.suggest_int("n_layers", N_LAYERS_MIN, N_LAYERS_MAX)

        hidden_dims = [
            trial.suggest_int(f"units_l{i}", UNITS_MIN, UNITS_MAX, step=32)
            for i in range(n_layers)
        ]

        train_loader, val_loader, _ = make_loaders(
            X_train, X_val, X_test, y_train, y_val, y_test,
            batch_size=batch_size,
        )

        model = EEGClassifier(input_dim, hidden_dims, n_classes, dropout).to(DEVICE)
        optimizer = build_optimizer(optimizer_name, model, lr, weight_decay)

        train_trial(model, train_loader, val_loader, optimizer, trial)

        macro_f1, _, _ = macro_f1_on_loader(model, val_loader)
        return macro_f1

    return objective


# ─── Salvamento dos artefatos ─────────────────────────────────────────────────

def save_best_model(study: optuna.Study, X_train, X_val, X_test,
                    y_train, y_val, y_test, n_classes: int,
                    input_dim: int) -> tuple[nn.Module, dict]:
    p = study.best_params
    hidden_dims = [p[f"units_l{i}"] for i in range(p["n_layers"])]

    train_loader, val_loader, test_loader = make_loaders(
        X_train, X_val, X_test, y_train, y_val, y_test,
        batch_size=p["batch_size"],
    )

    model = EEGClassifier(input_dim, hidden_dims, n_classes, p["dropout"]).to(DEVICE)
    optimizer = build_optimizer(p["optimizer"], model, p["lr"], p["weight_decay"])
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7
    )

    best_val_loss = float("inf")
    patience_count = 0
    best_state = None

    for _ in range(EPOCHS):
        _run_epoch(model, train_loader, criterion, optimizer)
        val_loss, _ = _run_epoch(model, val_loader, criterion)
        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_count = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pt"))

    params_to_save = {k: (int(v) if isinstance(v, np.integer) else v)
                      for k, v in p.items()}
    params_to_save["hidden_dims"] = hidden_dims
    params_to_save["best_val_macro_f1"] = float(study.best_value)

    with open(os.path.join(OUTPUT_DIR, "best_params.json"), "w") as f:
        json.dump(params_to_save, f, indent=2)

    return model, test_loader


def save_trials_csv(study: optuna.Study) -> None:
    df = study.trials_dataframe()
    df.to_csv(os.path.join(OUTPUT_DIR, "trials.csv"), index=False)
    print(f"[Salvo] trials.csv  ({len(df)} trials)")


def save_plots(study: optuna.Study) -> None:
    ax = plot_optimization_history(study)
    ax.set_title("Evolução da Otimização (Macro F1 na validação)")
    ax.figure.tight_layout()
    ax.figure.savefig(os.path.join(OUTPUT_DIR, "optimization_history.png"), bbox_inches="tight")
    plt.close(ax.figure)
    print("[Salvo] optimization_history.png")

    try:
        ax = plot_param_importances(study)
        ax.set_title("Importância dos Hiperparâmetros (fANOVA)")
        ax.figure.tight_layout()
        ax.figure.savefig(os.path.join(OUTPUT_DIR, "param_importance.png"), bbox_inches="tight")
        plt.close(ax.figure)
        print("[Salvo] param_importance.png")
    except Exception:
        print("[AVISO] Importância dos hiperparâmetros não disponível (trials insuficientes).")


# ─── Relatório final ──────────────────────────────────────────────────────────

def print_report(study: optuna.Study, macro_f1: float,
                 accuracy: float, bal_acc: float) -> None:
    p = study.best_params
    hidden_dims = [p[f"units_l{i}"] for i in range(p["n_layers"])]
    arch = " → ".join(str(h) for h in hidden_dims)

    print("\n" + "=" * 65)
    print("RESULTADO FINAL — MELHOR MODELO (conjunto de teste)")
    print("=" * 65)
    print(f"  Macro F1          : {macro_f1:.4f}")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Balanced Accuracy : {bal_acc:.4f}")
    print("\nMelhores hiperparâmetros:")
    print(f"  Arquitetura       : 88 → {arch} → 7")
    print(f"  Optimizer         : {p['optimizer']}")
    print(f"  Learning rate     : {p['lr']:.2e}")
    print(f"  Weight decay      : {p['weight_decay']:.2e}")
    print(f"  Dropout           : {p['dropout']:.3f}")
    print(f"  Batch size        : {p['batch_size']}")

    completed = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials
                 if t.state == optuna.trial.TrialState.PRUNED]
    print(f"\n  Trials completos  : {len(completed)}")
    print(f"  Trials podados    : {len(pruned)}")
    print(f"\n  Arquivos em       : ./{OUTPUT_DIR}/")

    print("\n" + "=" * 65)
    print("ANÁLISE E PRÓXIMOS PASSOS")
    print("=" * 65)
    print("""
Hiperparâmetros mais influentes (tipicamente neste tipo de problema):
  1. Learning rate   — controla convergência; valores extremos causam instabilidade
                       ou convergência lenta. Faixa log-scale é essencial.
  2. Arquitetura     — número de camadas e neurônios define capacidade do modelo.
  3. Dropout         — principal regulador; datasets pequenos exigem valores altos.
  4. Weight decay    — regularização L2 complementar ao dropout.
  5. Optimizer       — Adam/AdamW geralmente superam RMSprop em problemas tabulares.
  6. Batch size      — afeta curvatura do gradiente; batches menores = mais ruído,
                       o que pode ajudar a escapar de mínimos locais.

Por que Optuna é melhor que Grid Search aqui:
  • Grid Search com 6 hiperparâmetros e 4-5 valores cada = milhares de combinações.
  • Optuna usa TPE (Tree-structured Parzen Estimator): aprende quais regiões do
    espaço são promissoras e concentra trials lá — muito mais eficiente.
  • Pruning automático elimina trials ruins cedo, economizando tempo de treino.
  • Neste problema (1862 amostras × 88 features), Grid Search seria inviável.

Possíveis próximos passos:
  1. CNN 1D  — tratar os 88 PCs como sequência temporal para capturar padrões locais.
  2. Ensemble — combinar MLP + SVM + Random Forest com soft voting para ganho de robustez.
  3. Feature selection — SHAP ou permutation importance antes do PCA para remover
     features EEG irrelevantes e reduzir ruído.
  4. Nested cross-validation — k-fold externo para avaliação imparcial do pipeline
     completo (SMOTE dentro de cada fold para evitar data leakage).
""")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 65)
    print("OPTUNA — OTIMIZAÇÃO DE HIPERPARÂMETROS")
    print(f"Trials: {N_TRIALS}  |  Épocas máx./trial: {EPOCHS}  |  Device: {DEVICE}")
    print("=" * 65)

    X_train, X_val, X_test, y_train, y_val, y_test, le = load_data()
    n_classes = len(le.classes_)
    input_dim = X_train.shape[1]

    sampler = TPESampler(seed=SEED)
    pruner  = MedianPruner(n_startup_trials=5, n_warmup_steps=10)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name="eeg_mlp_optimization",
    )

    objective = make_objective(
        X_train, X_val, X_test, y_train, y_val, y_test, n_classes, input_dim
    )

    print(f"\nIniciando {N_TRIALS} trials...\n")
    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
        gc_after_trial=True,
    )

    print(f"\nMelhor Macro F1 (validação): {study.best_value:.4f}")

    print("\nRe-treinando melhor configuração no conjunto completo de treino...")
    best_model, test_loader = save_best_model(
        study, X_train, X_val, X_test, y_train, y_val, y_test, n_classes, input_dim
    )
    print("[Salvo] best_model.pt")
    print("[Salvo] best_params.json")

    save_trials_csv(study)
    save_plots(study)

    macro_f1, accuracy, bal_acc = macro_f1_on_loader(best_model, test_loader)
    print_report(study, macro_f1, accuracy, bal_acc)


if __name__ == "__main__":
    main()
