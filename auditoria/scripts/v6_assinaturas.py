"""V6: assinaturas técnicas na banda de 2–30 Hz que os modelos recebem (outputs/03_janelas, µV, janelas aceitas sem
overlap) e a AUC de cada uma, sozinha, para o rótulo (paciente = 1)."""
import sys
import numpy as np, pandas as pd
from scipy.signal import welch
from sklearn.metrics import roc_auc_score
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
jan = pd.read_csv(f"{R}/outputs/03_janelas/janelas.csv")
linhas, espectros = [], {}
for r in man.itertuples():
    x = np.load(f"{R}/outputs/03_janelas/{r.subject_uid}.npz")["x"]
    ini = jan[(jan.subject_uid == r.subject_uid) & ~jan.rejeitada & jan.sem_overlap].inicio_amostra.to_numpy()
    w = np.stack([x[:, i:i + 1000] for i in ini])                     # [n, 19, 1000]
    f, p = welch(w, fs=250, nperseg=1000, axis=-1)                     # 0,25 Hz por janela
    pm = p.mean(axis=(0, 1))                                           # média nas janelas e canais
    espectros[r.subject_uid] = pm
    def b(a, c): return pm[(f >= a) & (f < c)].mean()
    faixa = (f >= 2) & (f <= 30)
    k = np.argmax(np.where(faixa, pm, 0))
    meia, e, d = pm[k] / 2, k, k
    while pm[e] > meia: e -= 1
    while pm[d] > meia: d += 1
    linhas.append(dict(uid=r.subject_uid, dataset=r.dataset, y=int(r.grupo != "HC"), data=r.data_gravacao,
                       lote_jan2013=r.data_gravacao.startswith("2013-01"),
                       pico_hz=f[k], largura_hz=(d - e) * 0.25, frac_pico=b(f[k] - 0.25, f[k] + 0.5) / b(2, 30.01),
                       pico_5hz_db=10 * np.log10(b(5.0, 5.75) / ((b(4.0, 4.5) + b(6.25, 6.75)) / 2)),
                       hp_in_db=10 * np.log10(b(2.0, 2.5) / b(3.0, 3.5)),
                       lp_in_db=10 * np.log10(b(28, 30) / b(22, 25)),
                       log_pot_total=np.log10(b(2, 30.01))))
t = pd.DataFrame(linhas)
t.to_csv(sys.argv[2] + "/v6_assinaturas.csv", index=False)
np.savez(sys.argv[2] + "/v6_espectros.npz", f=f, **espectros)
pd.set_option("display.width", 220)
print("Lote MDD jan/2013 (14 HC): pico, largura e fração da potência de 2–30 Hz no pico")
print(t[t.lote_jan2013][["uid", "pico_hz", "largura_hz", "frac_pico", "pico_5hz_db"]].round(3).to_string(index=False))
print("\nAUC de cada assinatura sozinha (paciente = 1; < 0,5 = aponta para HC; reportamos max(AUC, 1−AUC)):")
cols = ["pico_5hz_db", "hp_in_db", "lp_in_db", "log_pot_total", "largura_hz", "frac_pico"]
res = []
for nome, g in (("SZ", t[t.dataset == "SZ"]), ("MDD", t[t.dataset == "MDD"]), ("MDD sem jan/2013", t[(t.dataset == "MDD") & ~t.lote_jan2013])):
    linha = {"conjunto": nome, "n_hc": int((g.y == 0).sum()), "n_pac": int(g.y.sum())}
    for c in cols:
        a = roc_auc_score(g.y, g[c]); linha[c] = f"{max(a, 1 - a):.3f}{'+' if a >= 0.5 else '−'}"
    res.append(linha)
print(pd.DataFrame(res).to_string(index=False))
print("\nMedianas por classe:")
print(t.groupby(["dataset", "y", "lote_jan2013"])[cols].median().round(2).to_string())
