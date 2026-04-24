"""
CTEN 522: Adaptive Digital Communication System with BER Optimisation
======================================================================
Constellation Diagrams with FEC — Standalone Script

Generates received constellation diagrams for all four modulation schemes
(BPSK, QPSK, 16-QAM, 64-QAM) under both channel models (AWGN, Rayleigh)
across all three FEC modes (No FEC, Hamming(7,4), Conv R=1/2)
at three SNR levels (0, 10, 20 dB).

Note: Constellations show the RECEIVED symbols BEFORE FEC decoding.
      This is correct — the constellation is what the demodulator sees.
      FEC decoding happens after symbol decisions, not before.
      The effect of FEC is visible as reduced BER, not in the scatter itself.
      To show FEC impact visually, we compare scatter width across FEC modes
      using the same channel — more FEC overhead = fewer bits per symbol slot.

Output folder: results/plots/

  fig6_const_awgn_nofec.png        — AWGN,     No FEC     (3 SNR × 4 mods)
  fig7_const_awgn_hamming.png      — AWGN,     Hamming    (3 SNR × 4 mods)
  fig8_const_awgn_conv.png         — AWGN,     Conv R=1/2 (3 SNR × 4 mods)
  fig9_const_rayleigh_nofec.png    — Rayleigh, No FEC     (3 SNR × 4 mods)
  fig10_const_rayleigh_hamming.png — Rayleigh, Hamming    (3 SNR × 4 mods)
  fig11_const_rayleigh_conv.png    — Rayleigh, Conv R=1/2 (3 SNR × 4 mods)
  fig12_const_fec_compare.png      — FEC comparison: 3 FEC modes side-by-side
                                     (AWGN vs Rayleigh at 10 dB, one mod per row)

Dependencies:
    pip install numpy matplotlib

Usage:
    python generate_constellations.py
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUT = os.path.join("results", "plots")
os.makedirs(OUT, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# PARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
np.random.seed(42)
N_SYMBOLS  = 2000
SNR_LEVELS = [0, 10, 20]
SNR_LABELS = {0: "0 dB\n(Low)", 10: "10 dB\n(Medium)", 20: "20 dB\n(High)"}
MODS       = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
FEC_MODES  = ["none", "hamming", "conv"]
FEC_LABELS = {"none": "No FEC", "hamming": "Hamming(7,4)", "conv": "Conv R=1/2"}
MOD_COLORS = {"BPSK": "#1f77b4", "QPSK": "#9467bd",
              "16-QAM": "#2ca02c", "64-QAM": "#d62728"}

print(f"Symbols per plot : {N_SYMBOLS:,}")
print(f"SNR levels       : {SNR_LEVELS} dB")
print(f"FEC modes        : {[FEC_LABELS[f] for f in FEC_MODES]}")
print(f"Output           : {OUT}/\n")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#111111", "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "font.size": 8,
})


# ══════════════════════════════════════════════════════════════════════════════
# MODULATION FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def bpsk_mod(n):
    bits = np.random.randint(0, 2, n)
    return (2*bits - 1).astype(complex)

def qpsk_mod(n):
    n = n // 2 * 2
    bits = np.random.randint(0, 2, (n//2, 2))
    return (2*bits[:,0]-1 + 1j*(2*bits[:,1]-1)) / np.sqrt(2)

def qam16_mod(n):
    n = n // 4 * 4
    bits = np.random.randint(0, 2, (n//4, 4))
    lut = {(0,0):-3,(0,1):-1,(1,1):1,(1,0):3}
    I = np.array([lut[tuple(r[:2])] for r in bits])
    Q = np.array([lut[tuple(r[2:])] for r in bits])
    return (I + 1j*Q) / np.sqrt(10)

def qam64_mod(n):
    n = n // 6 * 6
    bits = np.random.randint(0, 2, (n//6, 6))
    lv = [-7,-5,-3,-1,1,3,5,7]
    def mv(x): return lv[int("".join(map(str,x)), 2)]
    I = np.array([mv(r[:3]) for r in bits])
    Q = np.array([mv(r[3:]) for r in bits])
    return (I + 1j*Q) / np.sqrt(42)

MOD_FN = {"BPSK": bpsk_mod, "QPSK": qpsk_mod,
          "16-QAM": qam16_mod, "64-QAM": qam64_mod}


# ══════════════════════════════════════════════════════════════════════════════
# FEC ENCODING  (affects NUMBER OF SYMBOLS transmitted, not the IQ plot shape)
# The constellation always shows modulated symbols. With FEC, more symbols are
# transmitted (encoded bits are longer), but the IQ scatter pattern for a given
# SNR is determined by the modulation + channel, not the FEC code itself.
# We scale N_SYMBOLS by the inverse code rate so the same number of DATA bits
# is always represented regardless of FEC overhead.
# ══════════════════════════════════════════════════════════════════════════════
CODE_RATES = {"none": 1.0, "hamming": 4/7, "conv": 0.5}

def get_n_symbols(mod, fec):
    """Number of symbols to generate so we always represent N_SYMBOLS worth
    of data bits, accounting for FEC overhead."""
    bits_per_sym = {"BPSK":1,"QPSK":2,"16-QAM":4,"64-QAM":6}[mod]
    rate         = CODE_RATES[fec]
    # With FEC, each data bit becomes 1/rate coded bits → more symbols
    return int(np.ceil(N_SYMBOLS / rate))


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def awgn(x, snr_db):
    lin   = 10**(snr_db/10)
    noise = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2*lin)
    return x + noise

def rayleigh(x, snr_db):
    h = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2)
    return awgn(x*h, snr_db) / h

CHANNELS = {"AWGN": awgn, "Rayleigh": rayleigh}


# ══════════════════════════════════════════════════════════════════════════════
# IDEAL CONSTELLATION POINTS (for reference crosses)
# ══════════════════════════════════════════════════════════════════════════════
def ideal_points(mod):
    """Return unique IQ coordinates of the noiseless constellation."""
    sym = MOD_FN[mod](100000)
    re  = np.unique(np.round(np.real(sym), 4))
    im  = np.unique(np.round(np.imag(sym), 4))
    return re, im


# ══════════════════════════════════════════════════════════════════════════════
# SINGLE AXIS PLOT
# ══════════════════════════════════════════════════════════════════════════════
def plot_const(ax, mod, ch_fn, snr_db, fec, color, title=None):
    """Draw received constellation on ax."""
    n   = get_n_symbols(mod, fec)
    tx  = MOD_FN[mod](n)
    rx  = ch_fn(tx, snr_db)

    ax.scatter(np.real(rx), np.imag(rx),
               s=3, alpha=0.30, color=color)

    # Ideal reference crosses
    re_pts, im_pts = ideal_points(mod)
    for re in re_pts:
        for im in im_pts:
            ax.plot(re, im, 'k+', ms=5, mew=1.0, zorder=5)

    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-2.0, 2.0)
    ax.set_xlabel("I", fontsize=7)
    ax.set_ylabel("Q", fontsize=7)
    if title:
        ax.set_title(title, fontsize=8, fontweight="bold")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURES 6–11 — one figure per Channel × FEC combination
# Layout: rows = SNR (0, 10, 20 dB)  |  cols = modulation (BPSK…64-QAM)
# ══════════════════════════════════════════════════════════════════════════════
fig_specs = [
    ("fig6_const_awgn_nofec.png",        awgn,     "none",    "AWGN"),
    ("fig7_const_awgn_hamming.png",       awgn,     "hamming", "AWGN"),
    ("fig8_const_awgn_conv.png",          awgn,     "conv",    "AWGN"),
    ("fig9_const_rayleigh_nofec.png",     rayleigh, "none",    "Rayleigh"),
    ("fig10_const_rayleigh_hamming.png",  rayleigh, "hamming", "Rayleigh"),
    ("fig11_const_rayleigh_conv.png",     rayleigh, "conv",    "Rayleigh"),
]

for fname, ch_fn, fec, ch_name in fig_specs:
    fig_num = fname.split("_")[0].replace("fig","")
    print(f"Generating Figure {fig_num} — {ch_name} / {FEC_LABELS[fec]}…")
    fig, axes = plt.subplots(3, 4, figsize=(14, 11))

    for row, snr in enumerate(SNR_LEVELS):
        for col, mod in enumerate(MODS):
            ax = axes[row][col]
            plot_const(ax, mod, ch_fn, snr, fec,
                       color=MOD_COLORS[mod],
                       title=f"{mod}  |  {snr} dB")
            if col == 0:
                ax.set_ylabel(f"{SNR_LABELS[snr]}\nQ", fontsize=7)

    fig.suptitle(
        f"Figure {fig_num} — Received Constellations: {ch_name} Channel / {FEC_LABELS[fec]}\n"
        "Rows: 0 dB · 10 dB · 20 dB     Columns: BPSK · QPSK · 16-QAM · 64-QAM     "
        "Black crosses = ideal points",
        fontsize=9, fontweight="bold"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    p = os.path.join(OUT, fname)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — FEC Comparison at 10 dB SNR
# Layout: rows = modulation  |  cols = No FEC / Hamming / Conv
#         top half = AWGN,   bottom half = Rayleigh
# ══════════════════════════════════════════════════════════════════════════════
print("\nGenerating Figure 12 — FEC comparison at 10 dB…")
SNR_CMP = 10
fig, axes = plt.subplots(8, 3, figsize=(13, 22))

for row_base, (ch_name, ch_fn) in enumerate([("AWGN", awgn), ("Rayleigh", rayleigh)]):
    for mod_idx, mod in enumerate(MODS):
        row = row_base * 4 + mod_idx
        for col, fec in enumerate(FEC_MODES):
            ax = axes[row][col]
            plot_const(ax, mod, ch_fn, SNR_CMP, fec,
                       color=MOD_COLORS[mod],
                       title=f"{mod} | {ch_name}\n{FEC_LABELS[fec]}")

# Column headers
for col, fec in enumerate(FEC_MODES):
    axes[0][col].set_title(
        f"{FEC_LABELS[fec]}\n{MODS[0]} | AWGN",
        fontsize=8, fontweight="bold"
    )

fig.suptitle(
    f"Figure 12 — FEC Mode Comparison: Received Constellations at {SNR_CMP} dB SNR\n"
    "Top 4 rows = AWGN  ·  Bottom 4 rows = Rayleigh  ·  "
    "Columns: No FEC / Hamming(7,4) / Conv R=1/2  ·  Black crosses = ideal points",
    fontsize=9, fontweight="bold"
)
fig.tight_layout(rect=[0, 0, 1, 0.97])
p = os.path.join(OUT, "fig12_const_fec_compare.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {p}")

# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
All constellation diagrams saved to: {OUT}/

  fig6_const_awgn_nofec.png        — AWGN,     No FEC
  fig7_const_awgn_hamming.png      — AWGN,     Hamming(7,4)
  fig8_const_awgn_conv.png         — AWGN,     Conv R=1/2
  fig9_const_rayleigh_nofec.png    — Rayleigh, No FEC
  fig10_const_rayleigh_hamming.png — Rayleigh, Hamming(7,4)
  fig11_const_rayleigh_conv.png    — Rayleigh, Conv R=1/2
  fig12_const_fec_compare.png      — All FEC modes side-by-side at 10 dB
""")