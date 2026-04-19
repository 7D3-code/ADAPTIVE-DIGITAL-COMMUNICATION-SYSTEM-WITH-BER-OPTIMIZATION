"""
CTEN 522: Advanced Communication Engineering Design
=====================================================
Adaptive Digital Communication System with BER Optimisation

This script simulates an adaptive modulation system over AWGN and Rayleigh
fading channels across three FEC configurations, generating all result plots
and data tables for the project report.

Outputs (saved to results/plots/):
  Figures:
    fig1_ber_vs_snr.png          — BER vs SNR for all modulations, channels, FEC modes
    fig2_selection_map.png       — Adaptive modulation selection map
    fig3_fec_comparison.png      — FEC scheme comparison per modulation & channel
    fig4_throughput.png          — Effective throughput vs SNR
    fig5_awgn_vs_rayleigh.png    — AWGN vs Rayleigh channel impact

  Tables (CSV):
    table1_full_ber.csv          — Full BER results for all 264 simulation combinations
    table2_best_modulation.csv   — Best modulation selected at each SNR point
    table3_fec_summary.csv       — Average BER improvement summary per FEC mode

  Report:
    CTEN522_Results_Report.pdf   — Full results & analysis report

Usage:
  python generate_results.py

Requirements:
  numpy, matplotlib, pandas, scipy, reportlab, Pillow
  (see requirements.txt)
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

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

OUT        = os.path.join("results", "plots")   # output directory
N_BITS     = 10_000                              # number of random input bits
SEED       = 42                                  # NumPy random seed (reproducibility)
SNR_RANGE  = np.arange(0, 21, 2)                # SNR sweep: 0–20 dB in 2 dB steps

MODS       = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
BPS        = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}   # bits per symbol
FEC_MODES  = ["none", "hamming", "conv"]
FEC_LABELS = {"none": "No FEC", "hamming": "Hamming(7,4)", "conv": "Conv R=1/2"}
RATES      = {"none": 1.0, "hamming": 4/7, "conv": 0.5}           # code rates

# Plot colours — one per modulation scheme
COLORS     = ["#1f77b4", "#9467bd", "#2ca02c", "#d62728"]

os.makedirs(OUT, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOT STYLE
# ══════════════════════════════════════════════════════════════════════════════

def set_plot_style():
    plt.rcParams.update({
        "figure.facecolor":  "white",
        "axes.facecolor":    "white",
        "axes.edgecolor":    "#333333",
        "axes.labelcolor":   "#333333",
        "xtick.color":       "#333333",
        "ytick.color":       "#333333",
        "text.color":        "#111111",
        "grid.color":        "#cccccc",
        "grid.linewidth":    0.6,
        "legend.framealpha": 0.9,
        "legend.edgecolor":  "#cccccc",
        "font.size":         9,
    })

set_plot_style()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT
# ══════════════════════════════════════════════════════════════════════════════

np.random.seed(SEED)
bits_orig = np.random.randint(0, 2, N_BITS, dtype=int)

# ══════════════════════════════════════════════════════════════════════════════
# MODULATION — MOD / DEMOD FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def bpsk_mod(b):
    """BPSK: map 0 → -1, 1 → +1"""
    return (2 * b - 1).astype(float)

def bpsk_demod(x):
    return (x > 0).astype(int)


def qpsk_mod(b):
    """QPSK: 2 bits per symbol, Gray-coded on I and Q axes"""
    b = b[:len(b) // 2 * 2].reshape(-1, 2)
    return (2*b[:, 0] - 1 + 1j*(2*b[:, 1] - 1)) / np.sqrt(2)

def qpsk_demod(s):
    return np.column_stack([np.real(s) > 0, np.imag(s) > 0]).flatten().astype(int)


def qam16_mod(b):
    """16-QAM: 4 bits per symbol, Gray-coded 4-level PAM on I and Q"""
    b = b[:len(b) // 4 * 4].reshape(-1, 4)
    mapping = {(0,0): -3, (0,1): -1, (1,1): 1, (1,0): 3}
    I = np.array([mapping[tuple(r[:2])] for r in b])
    Q = np.array([mapping[tuple(r[2:])] for r in b])
    return (I + 1j*Q) / np.sqrt(10)

def qam16_demod(s):
    I = np.real(s) * np.sqrt(10)
    Q = np.imag(s) * np.sqrt(10)
    return np.column_stack([
        (I > 0), (np.abs(I) < 2),
        (Q > 0), (np.abs(Q) < 2),
    ]).flatten().astype(int)


def qam64_mod(b):
    """64-QAM: 6 bits per symbol, Gray-coded 8-level PAM on I and Q"""
    b = b[:len(b) // 6 * 6].reshape(-1, 6)
    levels = [-7, -5, -3, -1, 1, 3, 5, 7]
    def map_val(x): return levels[int("".join(map(str, x)), 2)]
    I = np.array([map_val(r[:3]) for r in b])
    Q = np.array([map_val(r[3:]) for r in b])
    return (I + 1j*Q) / np.sqrt(42)

def qam64_demod(s):
    out = []
    for x in s:
        i = int(np.clip(np.real(x) * np.sqrt(42), -7, 7))
        q = int(np.clip(np.imag(x) * np.sqrt(42), -7, 7))
        out += list(map(int, format((i + 7) // 2, '03b')))
        out += list(map(int, format((q + 7) // 2, '03b')))
    return np.array(out, dtype=int)


MOD_FN = {
    "BPSK":   (bpsk_mod,   bpsk_demod),
    "QPSK":   (qpsk_mod,   qpsk_demod),
    "16-QAM": (qam16_mod,  qam16_demod),
    "64-QAM": (qam64_mod,  qam64_demod),
}

# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def awgn(x, snr_db):
    """Additive White Gaussian Noise channel"""
    lin = 10 ** (snr_db / 10)
    if np.iscomplexobj(x):
        n = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2 * lin)
    else:
        n = np.random.randn(len(x)) / np.sqrt(2 * lin)
    return x + n

def rayleigh(x, snr_db):
    """Flat Rayleigh fading channel (perfect CSI assumed at receiver)"""
    h = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2)
    return awgn(x * h, snr_db) / h

CHANNELS = {"AWGN": awgn, "Rayleigh": rayleigh}

# ══════════════════════════════════════════════════════════════════════════════
# THEORETICAL BER (AWGN only — for reference overlay on plots)
# ══════════════════════════════════════════════════════════════════════════════

def Q(x):
    return 0.5 * erfc(x / np.sqrt(2))

def theoretical_ber(mod, snr_db):
    g = 10 ** (np.array(snr_db) / 10)
    if mod == "BPSK":   return Q(np.sqrt(2 * g))
    if mod == "QPSK":   return Q(np.sqrt(2 * g))
    if mod == "16-QAM": return (3/4) * erfc(np.sqrt(g / 5) / np.sqrt(2)) * 2
    if mod == "64-QAM": return (7/6) * erfc(np.sqrt(g / 21) / np.sqrt(2)) * 2

# ══════════════════════════════════════════════════════════════════════════════
# FEC — HAMMING (7,4)
# ══════════════════════════════════════════════════════════════════════════════

# Generator matrix G and parity-check matrix H
G_HAM = np.array([
    [1, 0, 0, 0, 1, 1, 0],
    [0, 1, 0, 0, 1, 0, 1],
    [0, 0, 1, 0, 0, 1, 1],
    [0, 0, 0, 1, 1, 1, 1],
], dtype=int)

H_HAM = np.array([
    [1, 1, 0, 1, 1, 0, 0],
    [1, 0, 1, 1, 0, 1, 0],
    [0, 1, 1, 1, 0, 0, 1],
], dtype=int)

def ham_enc(bits):
    """Encode bits using Hamming(7,4): every 4 data bits → 7 coded bits"""
    pad = (4 - len(bits) % 4) % 4
    b = np.concatenate([bits, np.zeros(pad, dtype=int)])
    coded = np.array([(blk @ G_HAM) % 2 for blk in b.reshape(-1, 4)]).flatten()
    return coded, pad

def ham_dec(coded, orig_len):
    """Decode Hamming(7,4) coded bits using syndrome decoding (corrects 1 bit/block)"""
    pad = (7 - len(coded) % 7) % 7
    coded = np.concatenate([coded, np.zeros(pad, dtype=int)])
    dec = []; errors_corrected = 0
    for cw in coded.reshape(-1, 7).copy():
        syndrome = (H_HAM @ cw) % 2
        idx = int("".join(map(str, syndrome)), 2)
        if 0 < idx <= 7:
            cw[idx - 1] ^= 1
            errors_corrected += 1
        dec.extend(cw[:4])
    return np.array(dec[:orig_len], dtype=int), errors_corrected

# ══════════════════════════════════════════════════════════════════════════════
# FEC — CONVOLUTIONAL (K=3, Rate 1/2, Viterbi hard-decision)
# ══════════════════════════════════════════════════════════════════════════════

# Generator polynomials: G1 = [1,1,1] (octal 7), G2 = [1,0,1] (octal 5)
G1 = [1, 1, 1]
G2 = [1, 0, 1]

def conv_enc(bits):
    """Rate-1/2 convolutional encoder with constraint length K=3"""
    enc = []; state = [0, 0]
    for b in bits:
        reg = [b] + state
        enc += [
            (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2,
            (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2,
        ]
        state = [b, state[0]]
    # Flush encoder with 2 zero bits (tail bits)
    for _ in range(2):
        reg = [0] + state
        enc += [
            (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2,
            (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2,
        ]
        state = [0, state[0]]
    return np.array(enc, dtype=int)

def conv_dec(rx, orig_len):
    """Hard-decision Viterbi decoder for K=3, R=1/2 convolutional code"""
    # Pre-compute trellis: outputs and next-state for each (state, input) pair
    n_states = 4
    outs = {}; nxt = {}
    for s in range(n_states):
        for inp in range(2):
            s_bits = [(s >> i) & 1 for i in range(1, -1, -1)]
            reg = [inp] + s_bits
            outs[(s, inp)] = (
                (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2,
                (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2,
            )
            nb = [inp] + s_bits[:-1]
            nxt[(s, inp)] = nb[0]*2 + nb[1]

    # Viterbi forward pass
    path_metric = np.full(n_states, 1e9); path_metric[0] = 0
    paths = {s: [] for s in range(n_states)}
    rx = np.array(rx, dtype=int)

    for t in range(len(rx) // 2):
        r = rx[t*2: t*2 + 2]
        new_pm = np.full(n_states, 1e9)
        new_paths = {s: [] for s in range(n_states)}
        for s in range(n_states):
            for inp in range(2):
                ns = nxt[(s, inp)]
                o  = outs[(s, inp)]
                cost = path_metric[s] + int(r[0] != o[0]) + int(r[1] != o[1])
                if cost < new_pm[ns]:
                    new_pm[ns] = cost
                    new_paths[ns] = paths[s] + [inp]
        path_metric = new_pm
        paths = new_paths

    best = int(np.argmin(path_metric))
    dec  = np.array(paths[best][:orig_len], dtype=int)
    if len(dec) < orig_len:
        dec = np.concatenate([dec, np.zeros(orig_len - len(dec), dtype=int)])
    return dec

# ══════════════════════════════════════════════════════════════════════════════
# BER COMPUTATION
# ══════════════════════════════════════════════════════════════════════════════

def compute_ber(bits, mod, ch_fn, snr_db, fec="none"):
    """
    Transmit bits through modulation → channel → demodulation (→ FEC decode).
    Returns (pre_fec_ber, post_fec_ber). For fec='none' both values are equal.
    """
    mf, md = MOD_FN[mod]

    # FEC encoding
    if fec == "none":
        tx = bits
    elif fec == "hamming":
        tx, _ = ham_enc(bits)
    else:
        tx = conv_enc(bits)

    # Modulate → channel → demodulate
    dem = md(ch_fn(mf(tx), snr_db))
    L   = min(len(tx), len(dem))
    pre_ber = float(np.mean(tx[:L] != dem[:L]))

    # FEC decoding
    if fec == "none":
        return pre_ber, pre_ber
    elif fec == "hamming":
        dec, _ = ham_dec(dem[:L], len(bits))
        return pre_ber, float(np.mean(bits != dec))
    else:
        dec = conv_dec(dem[:L], len(bits))
        return pre_ber, float(np.mean(bits != dec))


def adaptive_score(mod, post_ber, fec):
    """
    Composite score = effective throughput / BER.
    Higher is better — jointly maximises spectral efficiency and minimises errors.
    """
    throughput = BPS[mod] * RATES[fec] * (1 - post_ber)
    return throughput / (post_ber + 1e-9)

# ══════════════════════════════════════════════════════════════════════════════
# RUN FULL SIMULATION SWEEP
# ══════════════════════════════════════════════════════════════════════════════

# Storage: ber_data[channel][fec][mod] = (pre_ber_array, post_ber_array)
ber_data  = {ch: {f: {m: ([], []) for m in MODS} for f in FEC_MODES} for ch in CHANNELS}
# best_data[channel][fec] = list of best mod at each SNR point
best_data = {ch: {f: [] for f in FEC_MODES} for ch in CHANNELS}

total = len(CHANNELS) * len(FEC_MODES) * len(MODS) * len(SNR_RANGE)
done  = 0
print(f"Running {total} simulation jobs ({N_BITS:,} bits each)…")

for ch_name, ch_fn in CHANNELS.items():
    for fec in FEC_MODES:
        for mod in MODS:
            pre_list, post_list = [], []
            for snr in SNR_RANGE:
                pre, post = compute_ber(bits_orig, mod, ch_fn, snr, fec)
                pre_list.append(pre)
                post_list.append(post)
                done += 1
            ber_data[ch_name][fec][mod] = (np.array(pre_list), np.array(post_list))
            print(f"  [{done:>3}/{total}]  {ch_name:<9}  {FEC_LABELS[fec]:<14}  {mod}")

# Determine best modulation at each SNR point
for ch_name in CHANNELS:
    for fec in FEC_MODES:
        selection = []
        for k in range(len(SNR_RANGE)):
            scores = {
                m: adaptive_score(m, ber_data[ch_name][fec][m][1][k], fec)
                for m in MODS
            }
            selection.append(max(scores, key=scores.get))
        best_data[ch_name][fec] = selection

print("\nSimulations complete. Generating outputs…\n")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — BER vs SNR (3×2 grid: FEC mode × Channel)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(3, 2, figsize=(13, 14))
snr_fine  = np.linspace(0, 20, 300)

for row, fec in enumerate(FEC_MODES):
    for col, ch_name in enumerate(["AWGN", "Rayleigh"]):
        ax = axes[row][col]
        for j, mod in enumerate(MODS):
            pre, post = ber_data[ch_name][fec][mod]
            label = mod + (" (pre-FEC)" if fec != "none" else "")
            ax.semilogy(SNR_RANGE, np.clip(pre, 1e-6, 1),
                        color=COLORS[j], lw=2, marker='o', ms=5, label=label)
            if fec != "none":
                ax.semilogy(SNR_RANGE, np.clip(post, 1e-6, 1),
                            color=COLORS[j], lw=2, ls='--', marker='s', ms=4,
                            alpha=0.85, label=f"{mod} (post-FEC)")
            # Overlay theoretical BER for AWGN / No-FEC panel
            if fec == "none" and ch_name == "AWGN":
                th = np.clip(theoretical_ber(mod, snr_fine), 1e-6, 1)
                ax.semilogy(snr_fine, th, color=COLORS[j], lw=0.9, ls=':', alpha=0.5)

        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("BER")
        ax.set_title(f"{FEC_LABELS[fec]}  —  {ch_name} Channel", fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.set_ylim(1e-6, 1.5)
        ax.grid(True, which='both', ls='--', alpha=0.4)
        if fec == "none" and ch_name == "AWGN":
            ax.text(0.02, 0.04, "Dotted = theoretical",
                    transform=ax.transAxes, fontsize=7, color="#555555")

fig.suptitle(
    f"Figure 1 — BER vs SNR: All Modulations, All FEC Modes\n"
    f"({N_BITS:,} random bits, seed={SEED})",
    fontsize=12, fontweight='bold', y=1.01
)
fig.tight_layout()
p1 = os.path.join(OUT, "fig1_ber_vs_snr.png")
fig.savefig(p1, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("fig1_ber_vs_snr.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Adaptive Modulation Selection Map (2×3 grid: Channel × FEC)
# ══════════════════════════════════════════════════════════════════════════════

MOD_IDX  = {m: i for i, m in enumerate(MODS)}
CMAP_SEL = matplotlib.colors.ListedColormap(["#aec7e8", "#c5b0d5", "#98df8a", "#ff9896"])

fig, axes = plt.subplots(2, 3, figsize=(15, 6))
for col, fec in enumerate(FEC_MODES):
    for row, ch_name in enumerate(["AWGN", "Rayleigh"]):
        ax  = axes[row][col]
        seq = best_data[ch_name][fec]
        mat = np.array([[MOD_IDX[b] for b in seq]])

        ax.imshow(mat, aspect='auto', cmap=CMAP_SEL, vmin=0, vmax=3,
                  extent=[SNR_RANGE[0]-1, SNR_RANGE[-1]+1, -0.5, 0.5])
        ax.set_xticks(SNR_RANGE)
        ax.set_yticks([])
        ax.set_xlabel("SNR (dB)")
        ax.tick_params(labelsize=8)
        ax.set_title(f"{ch_name}  |  {FEC_LABELS[fec]}", fontweight='bold')

        for k, snr in enumerate(SNR_RANGE):
            ax.text(snr, 0, seq[k].replace("-", "\n"),
                    ha='center', va='center', fontsize=8,
                    fontweight='bold', color='#111111')

handles = [mpatches.Patch(facecolor=CMAP_SEL(i/3), label=MODS[i]) for i in range(4)]
fig.legend(handles=handles, loc='lower center', ncol=4,
           fontsize=10, bbox_to_anchor=(0.5, -0.04))
fig.suptitle(
    "Figure 2 — Adaptive Modulation Selection: Best Scheme per SNR\n"
    "(Score = Effective Throughput / BER)",
    fontsize=12, fontweight='bold'
)
fig.tight_layout(rect=[0, 0.07, 1, 1])
p2 = os.path.join(OUT, "fig2_selection_map.png")
fig.savefig(p2, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("fig2_selection_map.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — FEC Comparison (post-FEC BER per modulation × channel)
# ══════════════════════════════════════════════════════════════════════════════

LS   = {"none": "-", "hamming": "--", "conv": "-."}
FCOL = ["#1f77b4", "#2ca02c", "#d62728"]   # one colour per FEC mode

fig, axes = plt.subplots(2, 4, figsize=(17, 8))
for row, ch_name in enumerate(["AWGN", "Rayleigh"]):
    for col, mod in enumerate(MODS):
        ax = axes[row][col]
        for k, fec in enumerate(FEC_MODES):
            _, post = ber_data[ch_name][fec][mod]
            ax.semilogy(SNR_RANGE, np.clip(post, 1e-6, 1),
                        ls=LS[fec], lw=2, color=FCOL[k],
                        marker=['o', 's', '^'][k], ms=5,
                        label=FEC_LABELS[fec])
        ax.set_title(f"{mod}  |  {ch_name}", fontweight='bold', fontsize=9)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Post-FEC BER")
        ax.legend(fontsize=7)
        ax.set_ylim(1e-6, 1.5)
        ax.grid(True, which='both', ls='--', alpha=0.4)

fig.suptitle(
    "Figure 3 — FEC Scheme Comparison: Post-FEC BER per Modulation & Channel",
    fontsize=12, fontweight='bold'
)
fig.tight_layout()
p3 = os.path.join(OUT, "fig3_fec_comparison.png")
fig.savefig(p3, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("fig3_fec_comparison.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Effective Throughput vs SNR (2×3 grid: Channel × FEC)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for row, ch_name in enumerate(["AWGN", "Rayleigh"]):
    for col, fec in enumerate(FEC_MODES):
        ax = axes[row][col]
        for j, mod in enumerate(MODS):
            _, post = ber_data[ch_name][fec][mod]
            throughput = BPS[mod] * RATES[fec] * (1 - post)
            ax.plot(SNR_RANGE, throughput,
                    color=COLORS[j], lw=2, marker='o', ms=5, label=mod)
        ax.set_xlabel("SNR (dB)")
        ax.set_ylabel("Eff. Throughput (bits/symbol)")
        ax.set_title(f"{ch_name}  |  {FEC_LABELS[fec]}", fontweight='bold')
        ax.legend(fontsize=8)
        ax.set_ylim(0, 6.5)
        ax.grid(True, ls='--', alpha=0.4)

fig.suptitle(
    "Figure 4 — Effective Throughput vs SNR\n"
    "Throughput = bits/symbol × code rate × (1 − post-FEC BER)",
    fontsize=12, fontweight='bold'
)
fig.tight_layout()
p4 = os.path.join(OUT, "fig4_throughput.png")
fig.savefig(p4, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("fig4_throughput.png saved")

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — AWGN vs Rayleigh (1×3: one panel per FEC mode)
# ══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for col, fec in enumerate(FEC_MODES):
    ax = axes[col]
    for j, mod in enumerate(MODS):
        _, pa = ber_data["AWGN"][fec][mod]
        _, pr = ber_data["Rayleigh"][fec][mod]
        ax.semilogy(SNR_RANGE, np.clip(pa, 1e-6, 1),
                    color=COLORS[j], lw=2, marker='o', ms=5, label=f"{mod} AWGN")
        ax.semilogy(SNR_RANGE, np.clip(pr, 1e-6, 1),
                    color=COLORS[j], lw=2, ls='--', marker='s', ms=4,
                    alpha=0.7, label=f"{mod} Rayleigh")
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Post-FEC BER")
    ax.set_title(f"{FEC_LABELS[fec]}\nSolid = AWGN  ·  Dashed = Rayleigh",
                 fontweight='bold')
    ax.legend(fontsize=6.5, ncol=2)
    ax.set_ylim(1e-6, 1.5)
    ax.grid(True, which='both', ls='--', alpha=0.4)

fig.suptitle(
    "Figure 5 — AWGN vs Rayleigh: Channel Impact on Post-FEC BER",
    fontsize=12, fontweight='bold'
)
fig.tight_layout()
p5 = os.path.join(OUT, "fig5_awgn_vs_rayleigh.png")
fig.savefig(p5, dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("fig5_awgn_vs_rayleigh.png saved\n")

# ══════════════════════════════════════════════════════════════════════════════
# CSV TABLES
# ══════════════════════════════════════════════════════════════════════════════

print("Generating CSV tables…")

# Table 1 — Full BER results (all 264 combinations)
rows = []
for ch_name in ["AWGN", "Rayleigh"]:
    for fec in FEC_MODES:
        for mod in MODS:
            pre_a, post_a = ber_data[ch_name][fec][mod]
            for k, snr in enumerate(SNR_RANGE):
                rows.append({
                    "Channel":          ch_name,
                    "FEC Mode":         FEC_LABELS[fec],
                    "Modulation":       mod,
                    "Bits/Symbol":      BPS[mod],
                    "Code Rate":        round(RATES[fec], 4),
                    "SNR (dB)":         int(snr),
                    "Pre-FEC BER":      round(float(pre_a[k]), 6),
                    "Post-FEC BER":     round(float(post_a[k]), 6),
                    "BER Reduction":    round(float(pre_a[k] - post_a[k]), 6),
                    "Eff. Throughput":  round(BPS[mod] * RATES[fec] * (1 - float(post_a[k])), 4),
                    "Score":            round(adaptive_score(mod, float(post_a[k]), fec), 4),
                })
pd.DataFrame(rows).to_csv(os.path.join(OUT, "table1_full_ber.csv"), index=False)
print("  table1_full_ber.csv saved")

# Table 2 — Best modulation selection per SNR
rows2 = []
for k, snr in enumerate(SNR_RANGE):
    row = {"SNR (dB)": int(snr)}
    for ch_name in ["AWGN", "Rayleigh"]:
        for fec in FEC_MODES:
            row[f"{ch_name} / {FEC_LABELS[fec]}"] = best_data[ch_name][fec][k]
    rows2.append(row)
pd.DataFrame(rows2).to_csv(os.path.join(OUT, "table2_best_modulation.csv"), index=False)
print("  table2_best_modulation.csv saved")

# Table 3 — Average BER improvement summary per FEC mode
rows3 = []
for ch_name in ["AWGN", "Rayleigh"]:
    for fec in FEC_MODES:
        for mod in MODS:
            pre_a, post_a = ber_data[ch_name][fec][mod]
            rows3.append({
                "Channel":              ch_name,
                "FEC Mode":             FEC_LABELS[fec],
                "Modulation":           mod,
                "Avg Pre-FEC BER":      round(float(np.mean(pre_a)), 5),
                "Avg Post-FEC BER":     round(float(np.mean(post_a)), 5),
                "Avg BER Reduction":    round(float(np.mean(pre_a - post_a)), 5),
                "Avg Eff. Throughput":  round(float(np.mean(BPS[mod] * RATES[fec] * (1 - post_a))), 4),
            })
pd.DataFrame(rows3).to_csv(os.path.join(OUT, "table3_fec_summary.csv"), index=False)
print("  table3_fec_summary.csv saved\n")

# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ══════════════════════════════════════════════════════════════════════════════

print("=" * 55)
print(f"All outputs saved to: {OUT}/")
print("=" * 55)
for f in sorted(os.listdir(OUT)):
    size_kb = os.path.getsize(os.path.join(OUT, f)) // 1024
    print(f"  {f:<42} {size_kb:>5} KB")

