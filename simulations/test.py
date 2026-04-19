"""
CTEN 522: Adaptive Digital Communication System with BER Optimisation
======================================================================
Reproduces exactly the figures and tables from the simulation report:

  Figure 1 — BER vs SNR: All Modulations, All FEC Modes (3x2 grid)
  Figure 2 — Adaptive Modulation Selection Map (2x3 grid)
  Figure 3 — FEC Comparison: Post-FEC BER per Modulation & Channel (2x4 grid)
  Figure 4 — Effective Throughput vs SNR (2x3 grid)
  Figure 5 — AWGN vs Rayleigh Channel Impact on Post-FEC BER (1x3 grid)

  Table 1  — Best Modulation Scheme per SNR
  Table 2  — AWGN Channel Post-FEC BER at key SNR points
  Table 3  — Rayleigh Channel Post-FEC BER at key SNR points

Output folder: results/plots/

Dependencies:
    pip install numpy scipy matplotlib pandas

Usage:
    python generate_results.py
"""

import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
from scipy.special import erfc

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUT = os.path.join("results", "plots")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# SIMULATION PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N_BITS    = 10_000
bits_orig = np.random.randint(0, 2, N_BITS, dtype=int)
SNR_RANGE = np.arange(0, 21, 2)       # 0, 2, 4, … 20 dB
SNAP_SNRS = [0, 4, 8, 12, 16, 20]     # columns for Tables 2 & 3

MODS       = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
BPS        = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}
FEC_MODES  = ["none", "hamming", "conv"]
FEC_LABELS = {"none": "No FEC", "hamming": "Hamming(7,4)", "conv": "Conv R=1/2"}
CODE_RATES = {"none": 1.0, "hamming": 4/7, "conv": 0.5}

MOD_COLORS = {"BPSK": "#1f77b4", "QPSK": "#9467bd",
              "16-QAM": "#2ca02c", "64-QAM": "#d62728"}
FEC_COLORS = {"none": "#1f77b4", "hamming": "#2ca02c", "conv": "#d62728"}

print(f"Input     : {N_BITS:,} random bits (seed=42)")
print(f"SNR range : {SNR_RANGE[0]}–{SNR_RANGE[-1]} dB  ({len(SNR_RANGE)} points)")
print(f"Output    : {OUT}/\n")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#111111", "grid.color": "#cccccc",
    "grid.linewidth": 0.5, "legend.framealpha": 0.9,
    "legend.edgecolor": "#aaaaaa", "legend.fontsize": 6, "font.size": 8,
})


# ══════════════════════════════════════════════════════════════════════════════
# MODULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def bpsk_mod(b):   return (2*b - 1).astype(float)
def bpsk_demod(x): return (x > 0).astype(int)

def qpsk_mod(b):
    b = b[:len(b)//2*2].reshape(-1, 2)
    return (2*b[:,0]-1 + 1j*(2*b[:,1]-1)) / np.sqrt(2)
def qpsk_demod(s):
    return np.column_stack([np.real(s)>0, np.imag(s)>0]).flatten().astype(int)

def qam16_mod(b):
    b = b[:len(b)//4*4].reshape(-1, 4)
    lut = {(0,0):-3,(0,1):-1,(1,1):1,(1,0):3}
    I = np.array([lut[tuple(r[:2])] for r in b])
    Q = np.array([lut[tuple(r[2:])] for r in b])
    return (I + 1j*Q) / np.sqrt(10)
def qam16_demod(s):
    I = np.real(s)*np.sqrt(10); Q = np.imag(s)*np.sqrt(10)
    return np.column_stack([(I>0),(np.abs(I)<2),(Q>0),(np.abs(Q)<2)]).flatten().astype(int)

def qam64_mod(b):
    b = b[:len(b)//6*6].reshape(-1, 6)
    lv = [-7,-5,-3,-1,1,3,5,7]
    def mv(x): return lv[int("".join(map(str,x)),2)]
    I = np.array([mv(r[:3]) for r in b])
    Q = np.array([mv(r[3:]) for r in b])
    return (I + 1j*Q) / np.sqrt(42)
def qam64_demod(s):
    out = []
    for x in s:
        i = int(np.clip(np.real(x)*np.sqrt(42),-7,7))
        q = int(np.clip(np.imag(x)*np.sqrt(42),-7,7))
        out += list(map(int, format((i+7)//2,'03b')))
        out += list(map(int, format((q+7)//2,'03b')))
    return np.array(out, dtype=int)

MOD_FN = {
    "BPSK":   (bpsk_mod,  bpsk_demod),
    "QPSK":   (qpsk_mod,  qpsk_demod),
    "16-QAM": (qam16_mod, qam16_demod),
    "64-QAM": (qam64_mod, qam64_demod),
}


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def awgn(x, snr_db):
    lin = 10**(snr_db/10)
    n = ((np.random.randn(len(x))+1j*np.random.randn(len(x)))/np.sqrt(2*lin)
         if np.iscomplexobj(x) else np.random.randn(len(x))/np.sqrt(2*lin))
    return x + n

def rayleigh(x, snr_db):
    h = (np.random.randn(len(x))+1j*np.random.randn(len(x)))/np.sqrt(2)
    return awgn(x*h, snr_db) / h

CHANNELS = {"AWGN": awgn, "Rayleigh": rayleigh}


# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL BER (AWGN, no FEC)
# ══════════════════════════════════════════════════════════════════════════════

def Q(x): return 0.5 * erfc(x / np.sqrt(2))

def theoretical_ber(mod, snr_db_arr):
    g = 10**(np.array(snr_db_arr)/10)
    if mod in ("BPSK","QPSK"): return Q(np.sqrt(2*g))
    if mod == "16-QAM":        return (3/4)*erfc(np.sqrt(g/5)/np.sqrt(2))*2
    if mod == "64-QAM":        return (7/6)*erfc(np.sqrt(g/21)/np.sqrt(2))*2
    return np.zeros_like(g)


# ══════════════════════════════════════════════════════════════════════════════
# FEC — HAMMING (7,4)
# ══════════════════════════════════════════════════════════════════════════════
G_HAM = np.array([[1,0,0,0,1,1,0],[0,1,0,0,1,0,1],
                  [0,0,1,0,0,1,1],[0,0,0,1,1,1,1]], dtype=int)
H_HAM = np.array([[1,1,0,1,1,0,0],[1,0,1,1,0,1,0],[0,1,1,1,0,0,1]], dtype=int)

def ham_enc(bits):
    pad = (4-len(bits)%4)%4
    b   = np.concatenate([bits, np.zeros(pad,dtype=int)])
    return np.array([(blk@G_HAM)%2 for blk in b.reshape(-1,4)]).flatten(), pad

def ham_dec(coded, orig_len):
    pad = (7-len(coded)%7)%7
    coded = np.concatenate([coded, np.zeros(pad,dtype=int)])
    dec = []; ec = 0
    for cw in coded.reshape(-1,7).copy():
        syn = (H_HAM@cw)%2
        idx = int("".join(map(str,syn)),2)
        if 0 < idx <= 7: cw[idx-1] ^= 1; ec += 1
        dec.extend(cw[:4])
    return np.array(dec[:orig_len],dtype=int), ec


# ══════════════════════════════════════════════════════════════════════════════
# FEC — CONVOLUTIONAL (K=3, R=1/2, Viterbi hard-decision)
# ══════════════════════════════════════════════════════════════════════════════
G1 = [1,1,1]; G2 = [1,0,1]

def conv_enc(bits):
    enc = []; state = [0,0]
    for b in bits:
        reg = [b]+state
        enc += [(reg[0]*G1[0]+reg[1]*G1[1]+reg[2]*G1[2])%2,
                (reg[0]*G2[0]+reg[1]*G2[1]+reg[2]*G2[2])%2]
        state = [b,state[0]]
    for _ in range(2):
        reg = [0]+state
        enc += [(reg[0]*G1[0]+reg[1]*G1[1]+reg[2]*G1[2])%2,
                (reg[0]*G2[0]+reg[1]*G2[1]+reg[2]*G2[2])%2]
        state = [0,state[0]]
    return np.array(enc,dtype=int)

def conv_dec(received, orig_len):
    outs={}; nxt={}
    for s in range(4):
        for inp in range(2):
            sb=[(s>>i)&1 for i in range(1,-1,-1)]; reg=[inp]+sb
            outs[(s,inp)]=((reg[0]*G1[0]+reg[1]*G1[1]+reg[2]*G1[2])%2,
                           (reg[0]*G2[0]+reg[1]*G2[1]+reg[2]*G2[2])%2)
            nb=[inp]+sb[:-1]; nxt[(s,inp)]=nb[0]*2+nb[1]
    pm=np.full(4,1e9); pm[0]=0; paths={s:[] for s in range(4)}
    rx=np.array(received,dtype=int)
    for t in range(len(rx)//2):
        r=rx[t*2:t*2+2]; npm=np.full(4,1e9); npth={s:[] for s in range(4)}
        for s in range(4):
            for inp in range(2):
                ns=nxt[(s,inp)]; o=outs[(s,inp)]
                c=pm[s]+int(r[0]!=o[0])+int(r[1]!=o[1])
                if c<npm[ns]: npm[ns]=c; npth[ns]=paths[s]+[inp]
        pm=npm; paths=npth
    bs=int(np.argmin(pm))
    dec=np.array(paths[bs][:orig_len],dtype=int)
    if len(dec)<orig_len:
        dec=np.concatenate([dec,np.zeros(orig_len-len(dec),dtype=int)])
    return dec


# ══════════════════════════════════════════════════════════════════════════════
# BER COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ber(bits, mod, ch_fn, snr_db, fec="none"):
    mf, md = MOD_FN[mod]
    if   fec == "none":    tx = bits
    elif fec == "hamming": tx, _ = ham_enc(bits)
    else:                  tx = conv_enc(bits)

    rx_bits = md(ch_fn(mf(tx), snr_db))
    L       = min(len(tx), len(rx_bits))
    pre_ber = float(np.mean(tx[:L] != rx_bits[:L]))

    if fec == "none":
        return pre_ber, pre_ber
    elif fec == "hamming":
        dec, _ = ham_dec(rx_bits[:L], len(bits))
        return pre_ber, float(np.mean(bits != dec))
    else:
        dec = conv_dec(rx_bits[:L], len(bits))
        return pre_ber, float(np.mean(bits != dec))

def adaptive_score(mod, post_ber, fec):
    return BPS[mod] * CODE_RATES[fec] * (1-post_ber) / (post_ber + 1e-9)


# ══════════════════════════════════════════════════════════════════════════════
# RUN FULL SWEEP
# ══════════════════════════════════════════════════════════════════════════════
ber_data  = {ch:{f:{m:([],[]) for m in MODS} for f in FEC_MODES} for ch in CHANNELS}
best_data = {ch:{f:[] for f in FEC_MODES} for ch in CHANNELS}

total = len(CHANNELS)*len(FEC_MODES)*len(MODS)*len(SNR_RANGE)
done  = 0
print(f"Running {total} simulation jobs…")

for ch_name, ch_fn in CHANNELS.items():
    for fec in FEC_MODES:
        for mod in MODS:
            pre_l, post_l = [], []
            for snr in SNR_RANGE:
                pre, post = compute_ber(bits_orig, mod, ch_fn, snr, fec)
                pre_l.append(pre); post_l.append(post); done += 1
            ber_data[ch_name][fec][mod] = (np.array(pre_l), np.array(post_l))
            print(f"  [{done:>3}/{total}]  {ch_name:<9}  {FEC_LABELS[fec]:<14}  {mod}")

for ch_name in CHANNELS:
    for fec in FEC_MODES:
        row = []
        for k in range(len(SNR_RANGE)):
            sc = {m: adaptive_score(m, ber_data[ch_name][fec][m][1][k], fec) for m in MODS}
            row.append(max(sc, key=sc.get))
        best_data[ch_name][fec] = row

print("\nSimulations complete. Generating figures and tables…\n")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — BER vs SNR  (3 rows × 2 cols)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(3, 2, figsize=(12, 13))
snr_fine  = np.linspace(0, 20, 300)

for row, fec in enumerate(FEC_MODES):
    for col, ch_name in enumerate(["AWGN", "Rayleigh"]):
        ax = axes[row][col]
        for mod in MODS:
            c = MOD_COLORS[mod]
            pre, post = ber_data[ch_name][fec][mod]
            if fec == "none":
                ax.semilogy(SNR_RANGE, np.clip(pre,1e-6,1),
                            color=c, lw=1.5, marker="o", ms=4, label=mod)
                if ch_name == "AWGN":
                    th = np.clip(theoretical_ber(mod, snr_fine), 1e-6, 1)
                    ax.semilogy(snr_fine, th, color=c, lw=0.8, ls=":", alpha=0.6)
            else:
                ax.semilogy(SNR_RANGE, np.clip(pre,1e-6,1),
                            color=c, lw=1.5, marker="o", ms=4,
                            label=f"{mod} (pre-FEC)")
                ax.semilogy(SNR_RANGE, np.clip(post,1e-6,1),
                            color=c, lw=1.5, ls="--", marker="s", ms=3,
                            alpha=0.85, label=f"{mod} (post-FEC)")
        ax.set_xlabel("SNR (dB)", fontsize=8)
        ax.set_ylabel("BER", fontsize=8)
        ax.set_title(f"{FEC_LABELS[fec]} — {ch_name} Channel",
                     fontsize=9, fontweight="bold")
        ax.legend(fontsize=6, ncol=2)
        ax.set_ylim(1e-6, 1.5)
        ax.grid(True, which="both", ls="--", alpha=0.4)
        if fec == "none" and ch_name == "AWGN":
            ax.text(0.02, 0.03, "Dotted = theoretical",
                    transform=ax.transAxes, fontsize=6, color="#666666")

fig.suptitle(
    f"Figure 1 — BER vs SNR: All Modulations, All FEC Modes\n"
    f"({N_BITS:,} random bits, seed=42)",
    fontsize=10, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "fig1_ber_vs_snr.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Adaptive Modulation Selection Map  (2 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
MOD_IDX  = {m:i for i,m in enumerate(MODS)}
CELL_CLR = matplotlib.colors.ListedColormap(["#aec7e8","#c5b0d5","#98df8a","#ff9896"])

fig, axes = plt.subplots(2, 3, figsize=(14, 5))
for col, fec in enumerate(FEC_MODES):
    for row, ch_name in enumerate(["AWGN","Rayleigh"]):
        ax  = axes[row][col]
        seq = best_data[ch_name][fec]
        mat = np.array([[MOD_IDX[b] for b in seq]])
        ax.imshow(mat, aspect="auto", cmap=CELL_CLR, vmin=0, vmax=3,
                  extent=[SNR_RANGE[0]-1, SNR_RANGE[-1]+1, -0.5, 0.5])
        ax.set_xticks(SNR_RANGE); ax.set_yticks([])
        ax.set_xlabel("SNR (dB)", fontsize=8)
        ax.set_title(f"{ch_name} | {FEC_LABELS[fec]}", fontsize=9, fontweight="bold")
        for k, snr in enumerate(SNR_RANGE):
            ax.text(snr, 0, seq[k].replace("-","\n"),
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", color="#111111")

handles = [mpatches.Patch(facecolor=CELL_CLR(i/3), label=MODS[i]) for i in range(4)]
fig.legend(handles=handles, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5,-0.05))
fig.suptitle(
    "Figure 2 — Adaptive Modulation Selection: Best Scheme per SNR\n"
    "(Score = Effective Throughput / BER)",
    fontsize=10, fontweight="bold")
fig.tight_layout(rect=[0,0.06,1,1])
p = os.path.join(OUT, "fig2_selection_map.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — FEC Comparison  (2 rows × 4 cols)
# ══════════════════════════════════════════════════════════════════════════════
LS   = {"none":"-", "hamming":"--", "conv":"-."}
MKRS = {"none":"o", "hamming":"s",  "conv":"^"}

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for row, ch_name in enumerate(["AWGN","Rayleigh"]):
    for col, mod in enumerate(MODS):
        ax = axes[row][col]
        for fec in FEC_MODES:
            _, post = ber_data[ch_name][fec][mod]
            ax.semilogy(SNR_RANGE, np.clip(post,1e-6,1),
                        ls=LS[fec], lw=1.5, color=FEC_COLORS[fec],
                        marker=MKRS[fec], ms=4, label=FEC_LABELS[fec])
        ax.set_title(f"{mod} | {ch_name}", fontsize=9, fontweight="bold")
        ax.set_xlabel("SNR (dB)", fontsize=8)
        ax.set_ylabel("Post-FEC BER", fontsize=8)
        ax.legend(fontsize=6.5)
        ax.set_ylim(1e-6, 1.5)
        ax.grid(True, which="both", ls="--", alpha=0.4)

fig.suptitle(
    "Figure 3 — FEC Scheme Comparison: Post-FEC BER per Modulation & Channel",
    fontsize=10, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "fig3_fec_comparison.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Effective Throughput vs SNR  (2 rows × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
for row, ch_name in enumerate(["AWGN","Rayleigh"]):
    for col, fec in enumerate(FEC_MODES):
        ax = axes[row][col]
        for mod in MODS:
            _, post = ber_data[ch_name][fec][mod]
            tp = BPS[mod] * CODE_RATES[fec] * (1-post)
            ax.plot(SNR_RANGE, tp, color=MOD_COLORS[mod],
                    lw=1.5, marker="o", ms=4, label=mod)
        ax.set_xlabel("SNR (dB)", fontsize=8)
        ax.set_ylabel("Eff. Throughput (bits/symbol)", fontsize=8)
        ax.set_title(f"{ch_name} | {FEC_LABELS[fec]}", fontsize=9, fontweight="bold")
        ax.legend(fontsize=7)
        ax.set_ylim(0, 6.5)
        ax.grid(True, ls="--", alpha=0.4)

fig.suptitle(
    "Figure 4 — Effective Throughput vs SNR\n"
    "Throughput = bits/symbol × code rate × (1 − post-FEC BER)",
    fontsize=10, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "fig4_throughput.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — AWGN vs Rayleigh  (1 row × 3 cols)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
for col, fec in enumerate(FEC_MODES):
    ax = axes[col]
    for mod in MODS:
        c = MOD_COLORS[mod]
        _, pa = ber_data["AWGN"][fec][mod]
        _, pr = ber_data["Rayleigh"][fec][mod]
        ax.semilogy(SNR_RANGE, np.clip(pa,1e-6,1),
                    color=c, lw=1.5, marker="o", ms=4, label=f"{mod} AWGN")
        ax.semilogy(SNR_RANGE, np.clip(pr,1e-6,1),
                    color=c, lw=1.5, ls="--", marker="s", ms=3,
                    alpha=0.8, label=f"{mod} Rayleigh")
    ax.set_xlabel("SNR (dB)", fontsize=8)
    ax.set_ylabel("Post-FEC BER", fontsize=8)
    ax.set_title(f"{FEC_LABELS[fec]}\nSolid = AWGN  ·  Dashed = Rayleigh",
                 fontsize=9, fontweight="bold")
    ax.legend(fontsize=6, ncol=2)
    ax.set_ylim(1e-6, 1.5)
    ax.grid(True, which="both", ls="--", alpha=0.4)

fig.suptitle(
    "Figure 5 — AWGN vs Rayleigh: Channel Impact on Post-FEC BER",
    fontsize=10, fontweight="bold")
fig.tight_layout()
p = os.path.join(OUT, "fig5_awgn_vs_rayleigh.png")
fig.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 1 — Best Modulation Scheme per SNR
# ══════════════════════════════════════════════════════════════════════════════
rows = []
for k, snr in enumerate(SNR_RANGE):
    rows.append({
        "SNR (dB)":         int(snr),
        "AWGN No FEC":      best_data["AWGN"]["none"][k],
        "AWGN Hamming":     best_data["AWGN"]["hamming"][k],
        "AWGN Conv":        best_data["AWGN"]["conv"][k],
        "Rayleigh No FEC":  best_data["Rayleigh"]["none"][k],
        "Rayleigh Hamming": best_data["Rayleigh"]["hamming"][k],
        "Rayleigh Conv":    best_data["Rayleigh"]["conv"][k],
    })
df_t1 = pd.DataFrame(rows)
p = os.path.join(OUT, "table1_best_modulation.csv")
df_t1.to_csv(p, index=False)
print(f"Saved: {p}  ({len(df_t1)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# TABLE 2 & 3 — Post-FEC BER snapshot at key SNR points
# ══════════════════════════════════════════════════════════════════════════════
def build_ber_snapshot(ch_name):
    rows = []
    for fec in FEC_MODES:
        for mod in MODS:
            _, post = ber_data[ch_name][fec][mod]
            row = {"FEC Mode": FEC_LABELS[fec], "Modulation": mod}
            for snr in SNAP_SNRS:
                k = list(SNR_RANGE).index(snr)
                row[f"{snr} dB"] = round(float(post[k]), 4)
            rows.append(row)
    return pd.DataFrame(rows)

df_t2 = build_ber_snapshot("AWGN")
p = os.path.join(OUT, "table2_awgn_ber.csv")
df_t2.to_csv(p, index=False)
print(f"Saved: {p}  ({len(df_t2)} rows)")

df_t3 = build_ber_snapshot("Rayleigh")
p = os.path.join(OUT, "table3_rayleigh_ber.csv")
df_t3.to_csv(p, index=False)
print(f"Saved: {p}  ({len(df_t3)} rows)")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
All outputs saved to: {OUT}/

Figures:
  fig1_ber_vs_snr.png        — BER vs SNR (Fig 1 in report)
  fig2_selection_map.png     — Adaptive modulation selection map (Fig 2)
  fig3_fec_comparison.png    — FEC scheme comparison post-FEC BER (Fig 3)
  fig4_throughput.png        — Effective throughput vs SNR (Fig 4)
  fig5_awgn_vs_rayleigh.png  — AWGN vs Rayleigh channel comparison (Fig 5)

Tables:
  table1_best_modulation.csv — Table 1: Best modulation per SNR (11 rows)
  table2_awgn_ber.csv        — Table 2: AWGN post-FEC BER snapshot (12 rows)
  table3_rayleigh_ber.csv    — Table 3: Rayleigh post-FEC BER snapshot (12 rows)
""")