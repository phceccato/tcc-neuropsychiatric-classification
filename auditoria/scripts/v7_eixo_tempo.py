"""V7: o eixo de tempo do lote MDD jan/2013 está certo? Frequência cardíaca (canais extras e EEG) e picos espectrais
no EDF bruto, por sessão."""
import sys
import numpy as np, pandas as pd
from scipy.signal import welch, butter, sosfiltfilt, find_peaks
sys.path.insert(0, sys.argv[1])
from pipeline.e01_auditoria.auditar_ec import ler_edf
R = sys.argv[1]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
mdd = man[man.dataset == "MDD"].copy()
def sessao(d):
    return "HC 2011" if d.startswith("2011") else ("HC jan/2013" if d.startswith("2013-01") else "abr–jul/2013")
linhas = []
for r in mdd.itertuples():
    edf = ler_edf(f"{R}/dados/Depression_dataset/{r.arquivo}")
    fs = edf.fs[0]
    extras = {}
    for i, rot in enumerate(edf.rotulos):
        if "23A" in rot or "24A" in rot or "A2-A1" in rot:
            extras[rot] = edf.microvolts(i)
    # frequência dominante de 0,3–3,5 Hz na autocorrelação do canal extra mais "periódico" (candidato a ECG)
    def periodo_cardiaco(x):
        x = sosfiltfilt(butter(4, [5, 30], btype="band", fs=fs, output="sos"), x - x.mean())
        env = np.abs(x)
        env = sosfiltfilt(butter(2, 4, fs=fs, output="sos"), env - env.mean())
        n = len(env); ac = np.correlate(env[: int(60 * fs)], env[: int(60 * fs)], "full")[int(60 * fs) - 1:]
        ac /= ac[0]
        lo, hi = int(fs / 3.5), int(fs / 0.3)                      # períodos de 0,29 s a 3,3 s
        k = lo + int(np.argmax(ac[lo:hi]))
        return 60 * fs / k, float(ac[k])
    melhor = None
    for rot, x in extras.items():
        bpm, forca = periodo_cardiaco(x)
        if melhor is None or forca > melhor[2]:
            melhor = (rot, bpm, forca)
    # espectro bruto de um canal EEG (Cz) até Nyquist: picos > 10 dB acima da mediana local
    cz = [i for i, rot in enumerate(edf.rotulos) if "Cz" in rot][0]
    f, p = welch(edf.microvolts(cz), fs=fs, nperseg=int(8 * fs))
    lp = 10 * np.log10(p)
    base = pd.Series(lp).rolling(41, center=True, min_periods=1).median().to_numpy()
    pk, _ = find_peaks(lp - base, height=10, distance=8)
    picos = ", ".join(f"{f[k]:.2f}" for k in pk if f[k] > 1)
    linhas.append(dict(uid=r.subject_uid, grupo=r.grupo, sessao=sessao(r.data_gravacao), dur_s=round(r.duracao_s),
                       canal_cardiaco=melhor[0] if melhor else "", bpm=round(melhor[1], 1) if melhor else np.nan,
                       autocorr=round(melhor[2], 2) if melhor else np.nan, picos_cz_hz=picos[:70]))
t = pd.DataFrame(linhas).sort_values(["sessao", "uid"])
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 80)
print(t.to_string(index=False))
print("\nMediana de bpm (só autocorr ≥ 0,3) por sessão:")
print(t[t.autocorr >= 0.3].groupby("sessao").bpm.describe()[["count", "min", "50%", "max"]].to_string())
