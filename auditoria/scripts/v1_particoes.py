"""V1/V2: partições por pessoa e vínculo janela→pessoa (somente leitura sobre outputs/ do clone)."""
import sys
import numpy as np, pandas as pd
R = sys.argv[1]
folds = pd.read_csv(f"{R}/outputs/05_folds/folds.csv")
man = pd.read_csv(f"{R}/outputs/02_harmonizacao/manifesto.csv")
jan = pd.read_csv(f"{R}/outputs/03_janelas/janelas.csv")
print("IDs únicos no manifesto:", man.subject_uid.is_unique, "| n por dataset:", man.groupby("dataset").size().to_dict())
print("IDs compartilhados entre datasets:", set(man[man.dataset=="SZ"].subject_uid) & set(man[man.dataset=="MDD"].subject_uid))
# arquivos de origem distintos por pessoa
print("arquivo de origem repetido entre pessoas:", man.arquivo.duplicated().any())
problemas = 0
for (d, r, f), g in folds.groupby(["dataset", "repeticao", "fold"]):
    papeis = {p: set(gg.subject_uid) for p, gg in g.groupby("papel")}
    t, v, te = papeis.get("treino", set()), papeis.get("validacao", set()), papeis.get("teste", set())
    if (t & v) or (t & te) or (v & te):
        problemas += 1; print("INTERSEÇÃO", d, r, f)
    if (t | v | te) != set(man[man.dataset == d].subject_uid):
        problemas += 1; print("COBERTURA", d, r, f)
    for p, s in (("teste", te), ("validacao", v)):
        if g[g.papel == p].grupo.nunique() < 2:
            problemas += 1; print("UMA CLASSE SÓ", d, r, f, p)
print("folds com problema:", problemas, "de", folds.groupby(["dataset","repeticao","fold"]).ngroups)
# cada pessoa é teste exatamente 1 vez por repetição
c = folds[folds.papel == "teste"].groupby(["dataset", "repeticao", "subject_uid"]).size()
print("teste por (rep, pessoa): min", c.min(), "max", c.max(), "| pessoas×rep esperadas", len(c), "=", 10*len(man))
# grupo coerente entre folds, manifesto e janelas
g1 = folds.drop_duplicates("subject_uid").set_index("subject_uid").grupo
g2 = man.set_index("subject_uid").grupo
g3 = jan.drop_duplicates("subject_uid").set_index("subject_uid").grupo
print("grupo coerente folds=manifesto=janelas:", (g1.sort_index() == g2.sort_index()).all() and (g2.sort_index() == g3.sort_index()).all())
# rótulo derivado do nome do arquivo
def rot(r):
    a = r.arquivo
    if r.dataset == "SZ":
        return "SZ" if a.startswith("s") else "HC"
    return "MDD" if a.startswith("MDD") else "HC"
print("grupo = rótulo do nome do arquivo:", (man.apply(rot, axis=1) == man.grupo).all())
# tamanhos por papel e classe
tab = folds.groupby(["dataset", "papel", "grupo"]).size() / 50
print("média de pessoas por fold (papel × classe):"); print(tab.round(2).to_string())
# janelas: cabem na gravação, grade sem overlap em múltiplos de 1000
n = man.set_index("subject_uid").n_amostras
fora = (jan.inicio_amostra + 1000 > jan.subject_uid.map(n)).sum()
print("janelas que ultrapassam a gravação:", fora)
print("sem_overlap ⇔ início múltiplo de 1000:", ((jan.inicio_amostra % 1000 == 0) == jan.sem_overlap).all())
print("passo de 500 amostras em todas as pessoas:", jan.groupby("subject_uid").inicio_amostra.apply(lambda s: (np.diff(s) == 500).all()).all())
# contagens por dataset/classe/partição em pessoas, gravações e janelas
ac = jan[~jan.rejeitada]
print("\npessoas, gravações e janelas por dataset × classe:")
t = man.groupby(["dataset","grupo"]).agg(pessoas=("subject_uid","nunique"), gravacoes=("arquivo","nunique"))
t["janelas_50"] = jan.groupby(["dataset","grupo"]).size()
t["aceitas_50"] = ac.groupby(["dataset","grupo"]).size()
t["aceitas_sem_overlap"] = ac[ac.sem_overlap].groupby(["dataset","grupo"]).size()
print(t.to_string())
# peso por pessoa: participantes longos recebem mais janelas?
pp = ac[ac.sem_overlap].groupby(["dataset","grupo","subject_uid"]).size()
print("\njanelas de avaliação por pessoa (min/mediana/max):")
print(pp.groupby(["dataset","grupo"]).agg(["min","median","max"]).to_string())
