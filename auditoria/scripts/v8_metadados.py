"""V8: quanto do rótulo os metadados de aquisição preveem, com as partições por pessoa da etapa 05 (5 × 10).
Mesma métrica do TCC: média da probabilidade fora do fold de cada pessoa nas 10 repetições → AUC."""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
sys.path.insert(0, sys.argv[1])
from pipeline.avaliacao.metricas import ic_bootstrap
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
folds = pd.read_csv(f"{R}/outputs/05_folds/folds.csv")
bruto = pd.read_csv(sys.argv[2] + "/v4_espectro_bruto.csv").set_index("uid")
PASTA = {"SZ": "Schizophrenia_dataset", "MDD": "Depression_dataset"}
def cab(uid):
    r = man.loc[uid]
    b = open(f"{R}/dados/{PASTA[r.dataset]}/{r.arquivo}", "rb").read()
    ns = int(b[252:256])
    pmax = float(b[256 + ns * 112: 256 + ns * 112 + 8])
    return dict(n_sinais=ns, fisico_max=pmax, data_ord=pd.Timestamp(r.data_gravacao).toordinal(), duracao_s=r.duracao_s)
meta = pd.DataFrame({u: cab(u) for u in man.index}).T.astype(float)
tec = bruto[["hp_db", "lp_db", "acima45_db", "rede50_db"]]
X = meta.join(tec)
CONJ = {"só data": ["data_ord"],
        "cabeçalho sem data (nº de sinais, faixa física, duração)": ["n_sinais", "fisico_max", "duracao_s"],
        "técnico fora de 2–30 Hz (passa-alta, passa-baixa, >45 Hz, rede)": ["hp_db", "lp_db", "acima45_db", "rede50_db"],
        "cabeçalho sem data + técnico": ["n_sinais", "fisico_max", "duracao_s", "hp_db", "lp_db", "acima45_db", "rede50_db"]}
MODELOS = {"LR": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=1000)),
           "árvore prof. 2": lambda: DecisionTreeClassifier(max_depth=2, class_weight="balanced", random_state=0)}
linhas = []
for d in ("SZ", "MDD"):
    fd = folds[folds.dataset == d]
    y = (fd.drop_duplicates("subject_uid").set_index("subject_uid").grupo != "HC").astype(int)
    for nome, cols in CONJ.items():
        for mnome, fabrica in MODELOS.items():
            prev = []
            for (r, f), g in fd.groupby(["repeticao", "fold"]):
                ajuste = g[g.papel != "teste"].subject_uid; teste = g[g.papel == "teste"].subject_uid
                m = fabrica().fit(X.loc[ajuste, cols].to_numpy(), y.loc[ajuste].to_numpy())
                prev += list(zip(teste, m.predict_proba(X.loc[teste, cols].to_numpy())[:, 1]))
            p = pd.DataFrame(prev, columns=["uid", "prob"]).groupby("uid").prob.mean()
            yy = y.loc[p.index]
            auc = roc_auc_score(yy, p); ic = ic_bootstrap(yy.to_numpy(), p.to_numpy(), semente=20261108)
            linhas.append(dict(dataset=d, metadados=nome, modelo=mnome, auc=round(auc, 3), ic=f"[{ic[0]:.3f}; {ic[1]:.3f}]"))
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(pd.DataFrame(linhas).to_string(index=False))
print("\nTabela cruzada SZ: faixa física × classe"); s = man[man.dataset == "SZ"]
print(pd.crosstab(meta.loc[s.index, "fisico_max"].round(0), s.grupo).to_string())
print("\nTabela cruzada MDD: nº de sinais no EDF × classe"); m_ = man[man.dataset == "MDD"]
print(pd.crosstab(meta.loc[m_.index, "n_sinais"], m_.grupo).to_string())
