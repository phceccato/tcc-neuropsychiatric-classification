"""V12: previsões fora do fold do D_FA (só repetição 0, rodada reduzida desta auditoria) por sessão de aquisição."""
import sys, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, sys.argv[1])
from pipeline.avaliacao.metricas import ic_bootstrap
R, D = sys.argv[1], sys.argv[2]
MODELO = sys.argv[4] if len(sys.argv) > 4 else "D_FA"
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
v6 = pd.read_csv(sys.argv[3] + "/v6_assinaturas.csv").set_index("uid")
for d in ("SZ", "MDD"):
    arquivos = sorted(glob.glob(f"{D}/{d}/{MODELO}/r0_f*.csv"))
    p = pd.concat([pd.read_csv(a) for a in arquivos])
    epocas = [json.load(open(a.replace(".csv", ".json"))).get("melhor_epoca", json.load(open(a.replace(".csv", ".json"))).get("parametros")) for a in arquivos]
    data = pd.to_datetime(p.subject_uid.map(man.data_gravacao))
    if d == "MDD":
        p["sessao"] = np.where(p.y == 1, "MDD (abr–jul/2013)", np.where(data.dt.year == 2011, "HC 2011", np.where(data.dt.month == 1, "HC jan/2013", "HC abr–jul/2013")))
    else:
        p["sessao"] = np.where(p.y == 0, "HC 2003", np.where(data.dt.year == 2003, "SZ 2003", "SZ 2004–2008"))
    ic = ic_bootstrap(p.y.to_numpy(), p.prob.to_numpy(), semente=20261108)
    print(f"\n== {MODELO} {d}: {len(arquivos)} folds, épocas escolhidas {epocas}; AUC repetição 0 = {roc_auc_score(p.y, p.prob):.3f} [{ic[0]:.3f}; {ic[1]:.3f}]")
    print(p.groupby("sessao").prob.agg(["count", "median", "min", "max"]).round(3).to_string())
    if d == "MDD":
        pac = p[p.y == 1]
        for s in ("HC 2011", "HC jan/2013", "HC abr–jul/2013"):
            q = pd.concat([pac, p[p.sessao == s]])
            print(f"   AUC {s} × pacientes: {roc_auc_score(q.y, q.prob):.3f}")
        sem = p[p.sessao != "HC jan/2013"]
        print(f"   AUC sem o lote jan/2013 (treinado com ele): {roc_auc_score(sem.y, sem.prob):.3f}")
        hc = p[p.y == 0].set_index("subject_uid").join(v6[["frac_pico"]])
        from scipy.stats import spearmanr
        print(f"   Spearman(prob, fração de potência no pico estreito) entre os HC: {spearmanr(hc.prob, hc.frac_pico).statistic:.2f} (n = {len(hc)})")
    else:
        hc = p[p.y == 0]
        for s in ("SZ 2003", "SZ 2004–2008"):
            q = pd.concat([hc, p[p.sessao == s]])
            print(f"   AUC {s} × HC: {roc_auc_score(q.y, q.prob):.3f}")
