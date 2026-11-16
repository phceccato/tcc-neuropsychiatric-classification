"""V13: o componente de ~5,4 Hz do lote MDD jan/2013 tem a topografia do alfa? E a periodicidade cardíaca, em todos os canais?"""
import sys
import numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt
sys.path.insert(0, sys.argv[1])
from pipeline.e01_auditoria.auditar_ec import ler_edf, CANAIS
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
mdd = man[man.dataset == "MDD"]
v6 = pd.read_csv(sys.argv[2] + "/v6_assinaturas.csv").set_index("uid")
topo, hr = {}, []
for r in mdd.itertuples():
    x = np.load(f"{R}/outputs/02_harmonizacao/{r.subject_uid}.npz")["x"].astype(np.float64)   # 1–40 Hz, CAR, 250 Hz
    f, p = welch(x, fs=250, nperseg=1000, axis=1)
    pk = v6.at[r.subject_uid, "pico_hz"]
    banda = (f >= pk - 0.5) & (f <= pk + 0.5) if r.data_gravacao.startswith("2013-01") else (f >= 8) & (f <= 13)
    rel = p[:, banda].sum(1) / p[:, (f >= 2) & (f <= 30)].sum(1)
    topo[r.subject_uid] = rel / rel.mean()
    edf = ler_edf(f"{R}/dados/Depression_dataset/{r.arquivo}")
    fs = edf.fs[0]; melhor = (0, np.nan, "")
    for i, rot in enumerate(edf.rotulos):
        if "Annotations" in rot: continue
        s = edf.microvolts(i)[: int(120 * fs)]
        s = sosfiltfilt(butter(4, [8, 40], btype="band", fs=fs, output="sos"), s - s.mean())
        env = sosfiltfilt(butter(2, 3, fs=fs, output="sos"), np.abs(s)); env -= env.mean()
        ac = np.correlate(env, env, "full")[len(env) - 1:]; ac /= ac[0]
        lo, hi = int(fs * 60 / 150), int(fs * 60 / 20)                 # 20–150 bpm
        k = lo + int(np.argmax(ac[lo:hi]))
        if ac[k] > melhor[0]: melhor = (float(ac[k]), 60 * fs / k, rot)
    hr.append(dict(uid=r.subject_uid, grupo=r.grupo, lote="jan/2013" if r.data_gravacao.startswith("2013-01") else ("2011" if r.data_gravacao.startswith("2011") else "abr–jul/2013"),
                   bpm=round(melhor[1], 1), autocorr=round(melhor[0], 2), canal=melhor[2]))
T = pd.DataFrame(topo, index=CANAIS).T
lote = [u for u in T.index if man.set_index("subject_uid").at[u, "data_gravacao"].startswith("2013-01")]
hc_outros = [u for u in T.index if u.startswith("MDD-HC") and u not in lote]
pac = [u for u in T.index if u.startswith("MDD-MDD")]
tab = pd.DataFrame({"jan/2013: pico ~5,4 Hz": T.loc[lote].median(), "outros HC: alfa 8–13 Hz": T.loc[hc_outros].median(), "MDD: alfa 8–13 Hz": T.loc[pac].median()})
print("Topografia relativa (potência na banda ÷ 2–30 Hz, normalizada pela média dos canais; mediana entre pessoas):")
print(tab.round(2).T.to_string())
print("correlação de Spearman entre as topografias (jan/2013 × outros HC):", round(tab.iloc[:, 0].corr(tab.iloc[:, 1], method="spearman"), 2))
h = pd.DataFrame(hr)
print("\nPeriodicidade cardíaca (melhor canal, 20–150 bpm):")
print(h.groupby("lote").apply(lambda g: pd.Series({"n": len(g), "bpm_mediana": g.bpm.median(), "bpm_min": g.bpm.min(), "bpm_max": g.bpm.max(),
                                                   "autocorr_mediana": g.autocorr.median(), "n_autocorr≥0,4": int((g.autocorr >= 0.4).sum())}), include_groups=False).to_string())
print(h[h.lote == "jan/2013"].sort_values("uid").to_string(index=False))
