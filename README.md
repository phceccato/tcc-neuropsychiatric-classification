# tcc-neuropsychiatric-classification

Projeto de TCC voltado à classificação de transtornos neuropsiquiátricos utilizando sinais de EEG e técnicas de aprendizado profundo, com foco na análise de padrões neurais e apoio ao diagnóstico por meio de Inteligência Artificial.

---

## Dataset

**EEG BRMH** — SMG-SNU Boramae Medical Center (Coreia do Sul, 2019)  
Park et al. (2021) — https://osf.io/8bsvr  
- 945 pacientes · 7 diagnósticos · features PSD (AB.*) + coerência (COH.*) + metadados demográficos
- Arquivo: `EEG.machinelearing_data_BRMH.csv`

---

## Ordem de execução

```bash
# 1. Análise de paridade demográfica (por sexo)
python parity_analysis.py

# 2. Experimentos de linha de base — Cenário A (4 sub-cenários × 2 targets)
python neural_net.py
```

---

## Descrição dos scripts

| Script | Descrição |
|---|---|
| `parity_analysis.py` | Análise de paridade demográfica por sexo: calcula P(C=1 \| A=sexo) para `main.disorder` e `specific.disorder`; gera dot-plot estilo "equality of odds", gráfico de barras e heatmap |
| `neural_net.py` | Experimentos de linha de base (Cenário A): MLP em PyTorch com pipeline por fold sem data leakage — KNNImputer → ADASYN → StandardScaler → (PCA opcional); validação cruzada 5-fold + holdout 75/25; 4 sub-cenários de features (PSD, FC, PSD+FC, PSD+FC+PCA) × 2 targets (main/specific); métricas: Balanced Accuracy, Macro F1, AUC-ROC, Cohen's κ |

### Sub-cenários (`neural_net.py`)

| Sub-cenário | Features | PCA |
|---|---|---|
| A1 | PSD (`AB.*`) | Não |
| A2 | Coerência (`COH.*`) | Não |
| A3 | PSD + Coerência | Não |
| A4 | PSD + Coerência | Sim (95% var.) |

### Arquitetura MLP (Cenário A — hiperparâmetros fixos)

```
Input → 1024 → 512 → 256 → 128 → 64 → n_classes
(BatchNorm + GELU + Dropout 0.30 em cada camada oculta)
Otimizador: Adam  lr=1e-5  weight_decay=1e-4
Early stopping: patience=20  |  ReduceLROnPlateau (factor=0.9, patience=3)
```

---

## Saídas geradas

```
parity_dotplot.png              # dot-plot equality of odds (main + specific)
parity_bars.png                 # barras P(C=1 | A=sexo) por transtorno
parity_heatmap.png              # heatmap de paridade demográfica

outputs/baseline/
├── library_versions.json
└── scenario_a/
    ├── summary_main.csv        # tabela de resultados agregados (main.disorder)
    ├── summary_specific.csv    # tabela de resultados agregados (specific.disorder)
    ├── main/
    │   ├── A{1..4}_main_results.json
    │   ├── A{1..4}_main_cm.png       # matriz de confusão (normalizada)
    │   └── A{1..4}_main_loss.png     # curvas de loss por fold
    └── specific/
        ├── A{1..4}_specific_results.json
        ├── A{1..4}_specific_cm.png
        └── A{1..4}_specific_loss.png
```

---

## Instalação

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
