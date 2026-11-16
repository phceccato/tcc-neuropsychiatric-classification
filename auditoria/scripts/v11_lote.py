"""V11: um lote real pelo DTSF: formatos, eixos, valores, rótulos, STFT independente, FA, gradientes, eval e campo receptivo."""
import sys
import numpy as np, pandas as pd, torch
from scipy.signal import stft as sp_stft, welch
sys.path.insert(0, sys.argv[1])
from pipeline.e07_DTSF.dados_dtsf import FonteDTSF
from pipeline.e07_DTSF.dados import sortear, janelas_avaliacao
from pipeline.e07_DTSF.redes_dtsf import DTSF, MODELOS
from pipeline.e07_DTSF.dados_dtsf import estatisticas_stft
from pathlib import Path
R = Path(sys.argv[1])
folds = pd.read_csv(R / "outputs/05_folds/folds.csv")
g = folds[(folds.dataset == "MDD") & (folds.repeticao == 0) & (folds.fold == 0)]
y = (g.set_index("subject_uid").grupo != "HC").astype(int)
treino = g[g.papel == "treino"].subject_uid.tolist()
fonte = FonteDTSF(folds[folds.dataset == "MDD"].subject_uid.unique(), R / "outputs/03_janelas/janelas.csv",
                  R / "outputs/07_DTSF/cache_dtsf", R / "outputs/03_janelas")
por_pessoa = {u: q for u, q in fonte.indice.groupby("subject_uid")}
rng = np.random.default_rng(0)
linhas = np.concatenate([sortear(por_pessoa[u], 4, rng) for u in treino[:16]])
x_t, x_e = fonte.lote(linhas)
print("x_t", tuple(x_t.shape), x_t.dtype, "| x_e", tuple(x_e.shape), x_e.dtype)
print("finitos:", bool(torch.isfinite(x_t).all()), bool(torch.isfinite(x_e).all()),
      "| x_t base: média %.3f desvio %.3f | x_e: min %.2f max %.2f" % (x_t[:, :, 0].mean(), x_t[:, :, 0].std(), x_e.min(), x_e.max()))
# rótulo de cada janela = rótulo da pessoa
pess = fonte.indice.set_index("linha").loc[linhas, "subject_uid"]
print("rótulo por janela coerente com a pessoa:", bool((pess.map(y).notna()).all()))
# eixo 2 de x_t: representações base, δ, θ, α, β → frequência de pico de cada uma
f, p = welch(x_t.numpy(), fs=125, nperseg=250, axis=-1)
pm = p.mean(axis=(0, 1))                                 # [5, freqs]
cent = (pm * f).sum(-1) / pm.sum(-1)
print("centroide espectral (Hz) por representação [base, δ, θ, α, β]:", np.round(cent, 1))
# STFT independente (scipy) do sinal em µV reconstruído a partir do .npz de 250 Hz
u0, l0 = pess.iloc[0], linhas[0]
z = np.load(R / f"outputs/03_janelas/{u0}.npz"); ini = int(fonte.indice.set_index("linha").at[l0, "inicio_amostra"])
from scipy.signal import resample_poly
uv = resample_poly(z["x"][:, ini:ini + 1000].astype(np.float64), 1, 2, axis=1)
fz, tz, Z = sp_stft(uv, fs=125, window="hann", nperseg=125, noverlap=100, boundary=None, padded=False, scaling="spectrum")
ref = np.log10((np.abs(Z * 125 * 0.5) ** 2)[:, 2:31] + 1e-12)  # escala da torch.stft: soma(janela)=62,5 → |X|=|Z|·62,5
print("STFT independente: quadros", Z.shape[-1], "| erro máx log10 vs x_e (sem efeito de borda do resample):",
      float(np.abs(ref[:, :, 2:-2] - x_e[0].numpy()[:, :, 2:-2]).max()).__round__(3))
# rede: formatos por estágio
torch.manual_seed(0)
m = DTSF("D_FA")
m.definir_normalizacao(*estatisticas_stft(fonte, linhas))
formas = {}
for nome, mod in [("conv_t[0]", m.conv_t[0]), ("fa", m.fa), ("conv_t[1]", m.conv_t[1]), ("gru_t", m.gru_t),
                  ("conv_e", m.conv_e), ("gru_e", m.gru_e), ("embedding", m.embedding), ("saida", m.saida)]:
    mod.register_forward_hook(lambda mod, i, o, nome=nome: formas.__setitem__(nome, tuple((o[0] if isinstance(o, tuple) else o).shape)))
m.train(); logit = m(x_t, x_e)
print("formas:", formas, "| logit", tuple(logit.shape))
alvo = torch.tensor(pess.map(y).to_numpy(), dtype=torch.float32)
perda = m.perda_treino(x_t, x_e, alvo, torch.tensor(1.0)); perda.backward()
gfa = [p.grad.abs().sum().item() for p in m.fa.parameters()]
print("perda %.4f | |grad| FA (fc1, fc2): %s | |grad| conv_t[0]: %.3g | conv_e: %.3g" % (perda.item(), np.round(gfa, 5),
      m.conv_t[0][0].weight.grad.abs().sum(), m.conv_e[0][0].weight.grad.abs().sum()))
m.eval()
with torch.no_grad():
    w = m.fa.pesos(m.conv_t[0](x_t))
    print("pesos da FA (sigmoid) no lote: média %.3f, desvio entre janelas por filtro (mediana) %.4f, faixa [%.3f, %.3f]" %
          (w.mean(), w.std(0).median(), w.min(), w.max()))
    a, b = m(x_t, x_e), m(x_t, x_e)
print("eval determinístico:", bool(torch.equal(a, b)))
# parâmetros
for k in MODELOS:
    print(f"  {k}: {sum(p.numel() for p in DTSF(k).parameters()):,}".replace(",", "."))
# D e D_FA com a mesma semente: pesos comuns iguais?
torch.manual_seed(123); d1 = DTSF("D").state_dict(); torch.manual_seed(123); d2 = DTSF("D_FA").state_dict()
print("D_FA começa igual ao D nas chaves comuns:", all(torch.equal(d1[k], d2[k]) for k in d1))
# campo receptivo temporal de uma posição do mapa antes do AdaptiveMaxPool (gradiente da saída da 2ª conv)
m2 = DTSF("D").eval()
x = torch.zeros(1, 19, 5, 500, requires_grad=True)
h = x
for camada in list(m2.conv_t[0])[:4]:          # conv (5,13) → BN → GELU → MaxPool(1,2)
    h = camada(h)
h = list(m2.conv_t[1])[0](h)                   # conv (1,25) a 62,5 Hz
h[0, :, 0, 125].sum().backward()
nz = torch.nonzero(x.grad[0].abs().sum((0, 1))).flatten()
n = int(nz.max() - nz.min() + 1)
print("campo receptivo de 1 posição do mapa: %d amostras = %.3f s a 125 Hz; cada token temporal resume ~%.1f posições (~%.2f s de sinal com o campo)"
      % (n, n / 125, 250 / 16, (250 / 16 * 2 + n) / 125))
