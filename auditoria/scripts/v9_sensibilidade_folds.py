"""V9: sensibilidade da AUC da LR_psd à partição (sementes alternativas) e reprodutibilidade dos folds entre versões."""
import sys, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, sklearn
sys.path.insert(0, sys.argv[1])
from pipeline.e05_folds.folds import gerar
from pipeline.e06_LR_psd.lr import media_por_pessoa, prever_cv
from pipeline.avaliacao.metricas import avaliar
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
feat = np.load(f"{R}/outputs/04_features/psd.npz"); jan = pd.read_csv(f"{R}/outputs/03_janelas/janelas.csv")
if len(sys.argv) > 2 and sys.argv[2] == "folds":
    f = pd.concat([gerar(g[["subject_uid", "grupo"]]).assign(dataset=d) for d, g in man.groupby("dataset", sort=False)])
    f.to_csv(sys.argv[3], index=False); print("sklearn", sklearn.__version__, "folds gravados"); sys.exit()
res = []
for semente in [20261108, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
    for d, g in man.groupby("dataset", sort=False):
        folds = gerar(g[["subject_uid", "grupo"]], semente=semente).assign(dataset=d)
        y = (folds.drop_duplicates("subject_uid").set_index("subject_uid").grupo != "HC").astype(int)
        for nome in ("absolutas",):
            xt = media_por_pessoa(feat[nome], jan, False).loc[y.index]; xe = media_por_pessoa(feat[nome], jan, True).loc[y.index]
            r = avaliar(prever_cv(xt, xe, y, folds))
            res.append(dict(semente=semente, dataset=d, auc=r["auc"], auc_rep=r["auc_rep_media"]))
t = pd.DataFrame(res)
print(t.pivot(index="semente", columns="dataset", values="auc").round(3).to_string())
print(t.groupby("dataset")[["auc", "auc_rep"]].agg(["min", "median", "max"]).round(3).to_string())
