"""V14: as mesmas características (76 potências de banda, 2–30 Hz) decodificam a SESSÃO dentro de uma classe só?
LR L2 (C = 1) + z-score, StratifiedKFold 5 × 10 sobre pessoas (uma gravação por pessoa), AUC sobre a média fora do fold."""
import sys, warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
sys.path.insert(0, sys.argv[1])
from pipeline.e06_LR_psd.lr import media_por_pessoa
from pipeline.avaliacao.metricas import ic_bootstrap
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
feat = np.load(f"{R}/outputs/04_features/psd.npz"); jan = pd.read_csv(f"{R}/outputs/03_janelas/janelas.csv")
def decodificar(uids, alvo, nome):
    for tipo in ("absolutas", "relativas"):
        x = media_por_pessoa(feat[tipo], jan, False).loc[uids].to_numpy(); y = alvo.loc[uids].to_numpy()
        prob = np.zeros((10, len(uids)))
        for r in range(10):
            for tr, te in StratifiedKFold(5, shuffle=True, random_state=r).split(x, y):
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=2000)).fit(x[tr], y[tr])
                prob[r, te] = m.predict_proba(x[te])[:, 1]
        p = prob.mean(0); ic = ic_bootstrap(y, p, semente=20261108)
        print(f"{nome} [{tipo}]: AUC {roc_auc_score(y, p):.3f} [{ic[0]:.3f}; {ic[1]:.3f}] (n = {int(y.sum())} × {int((y == 0).sum())})")
hc_mdd = man[(man.dataset == "MDD") & (man.grupo == "HC")].index
decodificar(hc_mdd, pd.Series(man.loc[hc_mdd, "data_gravacao"].str.startswith("2013-01").astype(int), index=hc_mdd), "MDD, só HC: jan/2013 × outras sessões")
hc_out = [u for u in hc_mdd if not man.at[u, "data_gravacao"].startswith("2013-01")]
decodificar(hc_out, pd.Series([int(man.at[u, "data_gravacao"].startswith("2011")) for u in hc_out], index=hc_out), "MDD, só HC fora de jan/2013: 2011 × abr–jul/2013")
sz = man[(man.dataset == "SZ") & (man.grupo == "SZ")].index
decodificar(sz, pd.Series([int(pd.Timestamp(man.at[u, "data_gravacao"]).year >= 2006) for u in sz], index=sz), "SZ, só pacientes: 2006–2008 (faixa física ±5862) × 2003–2005")
