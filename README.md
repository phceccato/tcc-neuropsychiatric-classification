# tcc-neuropsychiatric-classification

Projeto de TCC voltado à classificação de transtornos neuropsiquiátricos utilizando sinais de EEG e técnicas de aprendizado profundo, com foco na análise de padrões neurais e apoio ao diagnóstico por meio de Inteligência Artificial.

---

## Dataset

**EEG BRMH** — SMG-SNU Boramae Medical Center (Coreia do Sul, 2019)
- 945 pacientes · 7 diagnósticos · 1144 features (PSD + coerência EEG + metadados)
- Arquivo: `EEG.machinelearing_data_BRMH.csv`

---

## Ordem de execução

```bash
# 1. Análise exploratória dos dados brutos
python analise_eeg.py

# 2. Pré-processamento (imputação → SMOTE → normalização → PCA)
python preprocess.py

# 3. Treinamento da rede neural base
python neural_net.py

# 4. Otimização de hiperparâmetros com Optuna
python optuna_search.py
```

---

## Descrição dos scripts

| Script | Descrição |
|---|---|
| `analise_eeg.py` | Análise exploratória: distribuição de classes, demografia, PSD por banda, coerência, PCA/t-SNE, comparação de modelos clássicos |
| `preprocess.py` | Pipeline de pré-processamento: KNN imputation, SMOTE, StandardScaler, PCA (95% de variância) |
| `neural_net.py` | MLP em PyTorch (`88 → 256 → 128 → 64 → 7`) com early stopping e ReduceLROnPlateau |
| `optuna_search.py` | Busca automática de hiperparâmetros com Optuna (TPE + MedianPruner, 50 trials) |

---

## Saídas geradas

```
outputs/
├── X_preprocessed.csv          # features PCA prontas para treino
├── y_labels.csv                # labels textuais
├── y_encoded.csv               # labels codificadas
├── preprocessors.pkl           # imputer, scaler, pca, label_encoder
├── neural_net.pt               # pesos do modelo base
├── pp_01_nan_antes_imputacao.png
├── pp_02_balanceamento.png
├── pp_03_pca_variancia.png
├── pp_04_pca_scatter.png
├── nn_01_learning_curves.png
├── nn_02_confusion_matrix.png
├── nn_03_f1_por_classe.png
└── optuna/
    ├── best_model.pt           # melhor modelo encontrado pelo Optuna
    ├── best_params.json        # hiperparâmetros do melhor trial
    ├── trials.csv              # histórico dos 50 trials
    ├── optimization_history.png
    └── param_importance.png
```

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
