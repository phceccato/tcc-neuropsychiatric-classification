"""V5: campos técnicos do cabeçalho EDF (sem campo de paciente) por classe e sessão."""
import sys, collections
import numpy as np, pandas as pd
R = sys.argv[1]
aud = pd.read_csv(f"{R}/outputs/01_auditoria/auditoria_ec.csv")
aud = aud[aud.decisao != "EXCLUIR"]
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv").set_index("subject_uid")
PASTA = {"SZ": "Schizophrenia_dataset", "MDD": "Depression_dataset"}
linhas = []
for r in aud.itertuples():
    b = open(f"{R}/dados/{PASTA[r.dataset]}/{r.arquivo}", "rb").read()
    ns = int(b[252:256])
    def campo(desl, larg):
        ini = 256 + ns * desl
        return [b[ini + i * larg: ini + (i + 1) * larg].decode("latin-1").strip() for i in range(ns)]
    rot, transd, unid = campo(0, 16), campo(16, 80), campo(96, 8)
    pmin, pmax, dmin, dmax = campo(104, 8), campo(112, 8), campo(120, 8), campo(128, 8)
    pref, amostras = campo(136, 80), campo(216, 8)
    eeg = [i for i, x in enumerate(rot) if x.lower().removeprefix("eeg ").split("-")[0] in
           {c.lower() for c in ("Fp1","Fp2","F7","F3","Fz","F4","F8","T3","C3","Cz","C4","T4","T5","P3","Pz","P4","T6","O1","O2")}]
    linhas.append(dict(uid=r.subject_uid, dataset=r.dataset, grupo=r.grupo, sessao=man.at[r.subject_uid, "data_gravacao"][:7],
                       ns=ns, reserved=b[192:236].decode("latin-1").strip()[:10], dur_registro=b[244:252].decode().strip(),
                       n_registros=int(b[236:244]), rotulos="|".join(rot[i] for i in eeg[:3]) + "…",
                       extras="|".join(x for i, x in enumerate(rot) if i not in eeg),
                       unidade=",".join(sorted(set(unid[i] for i in eeg))),
                       fisico=",".join(sorted(set(f"{pmin[i]}/{pmax[i]}" for i in eeg))),
                       digital=",".join(sorted(set(f"{dmin[i]}/{dmax[i]}" for i in eeg))),
                       prefiltro=",".join(sorted(set(pref[i] for i in eeg)))[:60],
                       transdutor=",".join(sorted(set(transd[i] for i in eeg)))[:30],
                       amostras_reg=",".join(sorted(set(amostras[i] for i in eeg))),
                       inicio=b[176:184].decode("latin-1").strip()))
t = pd.DataFrame(linhas)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
for d in ("SZ", "MDD"):
    g = t[t.dataset == d]
    print(f"\n==== {d}")
    for col in ("ns", "reserved", "dur_registro", "rotulos", "extras", "unidade", "fisico", "digital", "prefiltro", "transdutor", "amostras_reg"):
        tab = pd.crosstab(g[col], [g.grupo, g.sessao.str[:4]])
        if len(tab) > 1 or True:
            print(f"-- {col}:"); print(tab.to_string())
