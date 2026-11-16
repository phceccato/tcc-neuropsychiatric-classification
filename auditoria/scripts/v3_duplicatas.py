"""V3: trechos de sinal compartilhados entre pessoas diferentes (correlação cruzada normalizada, 125 Hz)."""
import sys
import numpy as np, pandas as pd
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
CANAIS = [0, 8, 11, 18]           # Fp1, C3, T4, O2 (ordem de CANAIS)
P = 2500                          # sonda de 20 s a 125 Hz
sinais = {u: np.load(f"{R}/outputs/07_DTSF/cache/{u}.npy")[CANAIS].astype(np.float64) for u in man.subject_uid}
resultados = []
for d, g in man.groupby("dataset"):
    uids = list(g.subject_uid)
    sondas = []
    for u in uids:
        x = sinais[u]; L = x.shape[1]
        for pos in (L // 10, L // 2, L - L // 10 - P):
            s = x[:, pos:pos + P]; s = s - s.mean(axis=1, keepdims=True)
            sondas.append((u, pos, s))
    for alvo in uids:
        b = sinais[alvo]; L = b.shape[1]
        nfft = 1 << int(np.ceil(np.log2(L + P)))
        B = np.fft.rfft(b, nfft, axis=1)
        c2 = np.concatenate([np.zeros((b.shape[0], 1)), np.cumsum(b ** 2, axis=1)], axis=1)
        energia = (c2[:, P:] - c2[:, :-P]).sum(axis=0)            # energia local nos 4 canais, por deslocamento
        for u, pos, s in sondas:
            if u == alvo:
                continue
            S = np.fft.rfft(s[:, ::-1], nfft, axis=1)
            corr = np.fft.irfft(B * S, nfft, axis=1)[:, P - 1:L].sum(axis=0)   # soma nos canais, deslocamentos válidos
            ncc = corr / np.sqrt((s ** 2).sum() * np.maximum(energia, 1e-12))
            k = int(np.argmax(np.abs(ncc)))
            resultados.append(dict(dataset=d, sonda=u, pos_s=pos / 125, alvo=alvo, ncc_max=float(abs(ncc[k])), desloc_s=k / 125))
t = pd.DataFrame(resultados)
print("pares (sonda, alvo) comparados:", len(t))
print("NCC máximo por dataset (quantis):")
print(t.groupby("dataset").ncc_max.describe(percentiles=[.5, .99, .999]).to_string())
print("\n10 maiores NCC:")
print(t.sort_values("ncc_max", ascending=False).head(10).to_string(index=False))
print("\npares com NCC > 0,5:", int((t.ncc_max > 0.5).sum()))
lote = set(man[man.data_gravacao.str.startswith("2013-01")].subject_uid)
alto = t[t.ncc_max > 0.5]
print("  dos quais com sonda e alvo no lote MDD jan/2013:", int((alto.sonda.isin(lote) & alto.alvo.isin(lote)).sum()),
      "| na SZ:", int((alto.dataset == "SZ").sum()))
fora = t[~(t.sonda.isin(lote) & t.alvo.isin(lote))]
print("  NCC máximo entre pares que não são ambos do lote jan/2013: %.3f" % fora.ncc_max.max())
