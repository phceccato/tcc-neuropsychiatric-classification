"""V10: dependência da LR_psd (etapa 06) dos lotes de aquisição: subconjuntos sem retreino e retreino sem o lote."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, sys.argv[1])
from pipeline.e05_folds.folds import gerar
from pipeline.e06_LR_psd.lr import media_por_pessoa, prever_cv
from pipeline.avaliacao.metricas import avaliar, escores_por_pessoa, ic_bootstrap
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
feat = np.load(f"{R}/outputs/04_features/psd.npz"); jan = pd.read_csv(f"{R}/outputs/03_janelas/janelas.csv")
folds = pd.read_csv(f"{R}/outputs/05_folds/folds.csv")
def auc_sub(pred, uids):
    e = escores_por_pessoa(pred[pred.subject_uid.isin(uids)])
    ic = ic_bootstrap(e.y.to_numpy(), e.prob.to_numpy(), semente=20261108)
    return f"{roc_auc_score(e.y, e.prob):.3f} [{ic[0]:.3f}; {ic[1]:.3f}] (HC {int((e.y==0).sum())}, pac {int(e.y.sum())})"
data = pd.to_datetime(man.data_gravacao)
for nome in ("absolutas", "relativas"):
    print(f"\n===== LR_psd {nome}")
    for d in ("SZ", "MDD"):
        fd = folds[folds.dataset == d]
        y = (fd.drop_duplicates("subject_uid").set_index("subject_uid").grupo != "HC").astype(int)
        xt = media_por_pessoa(feat[nome], jan, False).loc[y.index]; xe = media_por_pessoa(feat[nome], jan, True).loc[y.index]
        pred = prever_cv(xt, xe, y, fd)
        todos = set(y.index)
        print(f"{d} todas as pessoas: {auc_sub(pred, todos)}")
        if d == "SZ":
            hc = {u for u in todos if y[u] == 0}
            print(f"SZ pacientes de 2003 × HC: {auc_sub(pred, hc | {u for u in todos if y[u] == 1 and data[u].year == 2003})}")
            print(f"SZ pacientes de 2004–2008 × HC: {auc_sub(pred, hc | {u for u in todos if y[u] == 1 and data[u].year > 2003})}")
        else:
            lote = {u for u in todos if y[u] == 0 and man.at[u, 'data_gravacao'].startswith('2013-01')}
            pac = {u for u in todos if y[u] == 1}
            print(f"MDD avaliado sem o lote jan/2013 (treinado com ele): {auc_sub(pred, todos - lote)}")
            print(f"MDD só HC jan/2013 × pacientes: {auc_sub(pred, lote | pac)}")
            print(f"MDD só HC 2011 × pacientes: {auc_sub(pred, {u for u in todos if y[u]==0 and data[u].year==2011} | pac)}")
            print(f"MDD só sessão abr–jul/2013 (4 HC × 29 pac): {auc_sub(pred, {u for u in todos if data[u] >= pd.Timestamp('2013-04-01')})}")
            # retreino sem o lote, partições novas como em pipeline/e09_investigacao/lote.py
            p2 = man.loc[sorted(todos - lote)].reset_index()[["subject_uid", "grupo"]]
            f2 = gerar(p2).assign(dataset="MDD")
            y2 = y.loc[p2.subject_uid]
            pr2 = prever_cv(xt.loc[y2.index], xe.loc[y2.index], y2, f2)
            print(f"MDD RETREINADO sem o lote jan/2013 (partições novas): {auc_sub(pr2, set(y2.index))}")
