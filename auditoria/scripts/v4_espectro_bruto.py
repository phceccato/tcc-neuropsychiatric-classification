"""V4: espectro do EDF bruto (µV, zeros do fim cortados, CAR, sem filtro) por gravação: pico dominante, largura,
rede elétrica e assinatura dos filtros de hardware; cruzado com classe e data de gravação."""
import sys
import numpy as np, pandas as pd
from scipy.signal import welch
sys.path.insert(0, sys.argv[1])
from pipeline.e02_harmonizacao.harmonizar import carregar_microvolts, PADRAO_ARQUIVOS
R = sys.argv[1]
aud = pd.read_csv(f"{R}/outputs/01_auditoria/auditoria_ec.csv")
aud = aud[aud.decisao != "EXCLUIR"]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
linhas = []
for r in aud.itertuples():
    n = int(round(r.duracao_util_s * r.fs))
    x = carregar_microvolts(f"{R}/dados/{PADRAO_ARQUIVOS[r.dataset][0]}/{r.arquivo}", n)
    x = x - x.mean(axis=0, keepdims=True)                         # CAR, sem filtro
    f, p = welch(x, fs=r.fs, nperseg=int(8 * r.fs), axis=1)       # 0,125 Hz
    pm = np.median(p, axis=0)                                     # mediana nos 19 canais
    def banda(a, b): return pm[(f >= a) & (f < b)].mean()
    faixa = (f >= 2) & (f <= 30)
    k = np.argmax(np.where(faixa, pm, 0)); fpk = f[k]
    meia = pm[k] / 2                                              # largura a meia altura do pico dominante
    esq = k
    while esq > 0 and pm[esq] > meia: esq -= 1
    dir_ = k
    while dir_ < len(pm) - 1 and pm[dir_] > meia: dir_ += 1
    linhas.append(dict(uid=r.subject_uid, dataset=r.dataset, grupo=r.grupo, data=man.at[r.subject_uid, "data_gravacao"],
                       fs=r.fs, dur_s=r.duracao_util_s, pico_hz=fpk, largura_hz=f[dir_] - f[esq],
                       pico_sobre_2_30=pm[k] / np.median(pm[faixa]),
                       rede50_db=10 * np.log10(banda(49.5, 50.5) / ((banda(46, 48) + banda(52, 54)) / 2)),
                       pico25_db=10 * np.log10(banda(24.75, 25.25) / ((banda(23, 24.5) + banda(25.5, 27)) / 2)),
                       hp_db=10 * np.log10(banda(0.5, 1.0) / banda(2.0, 3.0)),
                       lp_db=10 * np.log10(banda(38, 42) / banda(26, 30)),
                       acima45_db=10 * np.log10(banda(55, 60) / banda(26, 30))))
t = pd.DataFrame(linhas)
t["sessao"] = t.data.str[:7]
t.to_csv(sys.argv[2] + "/v4_espectro_bruto.csv", index=False)
pd.set_option("display.width", 250)
for d in ("MDD", "SZ"):
    g = t[t.dataset == d].sort_values("data")
    print(f"\n==== {d}: mediana por (classe, ano-mês)")
    print(g.groupby(["grupo", "sessao"])[["pico_hz", "largura_hz", "pico_sobre_2_30", "rede50_db", "pico25_db", "hp_db", "lp_db", "acima45_db"]]
          .median().round(2).assign(n=g.groupby(["grupo", "sessao"]).size()).to_string())
