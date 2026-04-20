"""
CTEN 522: Adaptive Digital Communication System with BER Optimisation
======================================================================
Constellation Diagrams — Standalone Script

Generates received constellation diagrams for all four modulation schemes
(BPSK, QPSK, 16-QAM, 64-QAM) under both channel models (AWGN, Rayleigh)
at three SNR levels (low, medium, high) to show how noise scatters symbols.

Output folder: results/plots/

  fig6_constellations_awgn.png     — 3 SNR levels × 4 modulations (AWGN)
  fig7_constellations_rayleigh.png — 3 SNR levels × 4 modulations (Rayleigh)
  fig8_constellations_snr_compare.png — Side-by-side SNR comparison per mod

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
N_SYMBOLS  = 2000          # symbols per constellation plot
SNR_LEVELS = [0, 10, 20]   # low / medium / high SNR (dB)
SNR_LABELS = {0: "Low SNR\n(0 dB)", 10: "Medium SNR\n(10 dB)", 20: "High SNR\n(20 dB)"}
MODS       = ["BPSK", "QPSK", "16-QAM", "64-QAM"]
MOD_COLORS = {"BPSK": "#1f77b4", "QPSK": "#9467bd",
              "16-QAM": "#2ca02c", "64-QAM": "#d62728"}

print(f"Symbols per plot : {N_SYMBOLS:,}")
print(f"SNR levels       : {SNR_LEVELS} dB")
print(f"Output           : {OUT}/\n")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "#f8f8f8",
    "axes.edgecolor": "#333333", "axes.labelcolor": "#333333",
    "xtick.color": "#333333", "ytick.color": "#333333",
    "text.color": "#111111", "grid.color": "#dddddd",
    "grid.linewidth": 0.5, "font.size": 8,
})


# ══════════════════════════════════════════════════════════════════════════════
# MODULATION FUNCTIONS  (return complex symbols)
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

MOD_FN = {
    "BPSK":   bpsk_mod,
    "QPSK":   qpsk_mod,
    "16-QAM": qam16_mod,
    "64-QAM": qam64_mod,
}

# Ideal (noiseless) constellation points for reference overlay
def ideal_points(mod):
    return MOD_FN[mod](100000)   # large sample covers all points


# ══════════════════════════════════════════════════════════════════════════════
# CHANNEL MODELS
# ══════════════════════════════════════════════════════════════════════════════

def awgn(x, snr_db):
    lin = 10**(snr_db / 10)
    noise = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2*lin)
    return x + noise

def rayleigh(x, snr_db):
    h = (np.random.randn(len(x)) + 1j*np.random.randn(len(x))) / np.sqrt(2)
    return awgn(x * h, snr_db) / h   # equalized

CHANNELS = {"AWGN": awgn, "Rayleigh": rayleigh}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — draw one constellation axis
# ══════════════════════════════════════════════════════════════════════════════

def plot_constellation(ax, tx_sym, rx_sym, mod, snr_db, channel_name, color):
    """
    Scatter plot of received symbols with ideal reference points overlaid.
    tx_sym — transmitted (noiseless) symbols
    rx_sym — received (noisy) symbols
    """
    # Received symbols (noisy cloud)
    ax.scatter(np.real(rx_sym), np.imag(rx_sym),
               s=3, alpha=0.35, color=color, label="Received")

    # Ideal constellation points (black crosses)
    ideal = ideal_points(mod)
    unique_re = np.unique(np.round(np.real(ideal), 4))
    unique_im = np.unique(np.round(np.imag(ideal), 4))
    for re in unique_re:
        for im in unique_im:
            ax.plot(re, im, 'k+', ms=6, mew=1.2, zorder=5)

    # Formatting
    ax.set_title(f"{mod}\n{channel_name} | {snr_db} dB", fontsize=8, fontweight="bold")
    ax.set_xlabel("In-Phase (I)", fontsize=7)
    ax.set_ylabel("Quadrature (Q)", fontsize=7)
    ax.axhline(0, color="#999999", lw=0.5, ls="--")
    ax.axvline(0, color="#999999", lw=0.5, ls="--")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")

    # Dynamic axis limits based on modulation
    lim = {"BPSK": 2.0, "QPSK": 2.0, "16-QAM": 2.0, "64-QAM": 2.0}[mod]
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Constellations: AWGN Channel
# Layout: rows = SNR level (0, 10, 20 dB)  |  cols = modulation
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 6 — AWGN constellations…")
fig, axes = plt.subplots(3, 4, figsize=(14, 11))

for row, snr in enumerate(SNR_LEVELS):
    for col, mod in enumerate(MODS):
        ax = axes[row][col]
        tx = MOD_FN[mod](N_SYMBOLS)
        rx = awgn(tx, snr)
        plot_constellation(ax, tx, rx, mod, snr, "AWGN", MOD_COLORS[mod])
        if col == 0:
            ax.set_ylabel(f"{SNR_LABELS[snr]}\nQuadrature (Q)", fontsize=7)

fig.suptitle(
    "Figure 6 — Received Constellation Diagrams: AWGN Channel\n"
    "Black crosses = ideal points  ·  Coloured dots = received symbols",
    fontsize=10, fontweight="bold"
)
fig.tight_layout()
p = os.path.join(OUT, "fig6_constellations_awgn.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Constellations: Rayleigh Channel
# Same layout as Figure 6
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 7 — Rayleigh constellations…")
fig, axes = plt.subplots(3, 4, figsize=(14, 11))

for row, snr in enumerate(SNR_LEVELS):
    for col, mod in enumerate(MODS):
        ax = axes[row][col]
        tx = MOD_FN[mod](N_SYMBOLS)
        rx = rayleigh(tx, snr)
        plot_constellation(ax, tx, rx, mod, snr, "Rayleigh", MOD_COLORS[mod])
        if col == 0:
            ax.set_ylabel(f"{SNR_LABELS[snr]}\nQuadrature (Q)", fontsize=7)

fig.suptitle(
    "Figure 7 — Received Constellation Diagrams: Rayleigh Channel\n"
    "Black crosses = ideal points  ·  Coloured dots = received symbols",
    fontsize=10, fontweight="bold"
)
fig.tight_layout()
p = os.path.join(OUT, "fig7_constellations_rayleigh.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — AWGN vs Rayleigh side-by-side per modulation at 10 dB SNR
# Layout: rows = modulation  |  cols = AWGN / Rayleigh
# ══════════════════════════════════════════════════════════════════════════════
print("Generating Figure 8 — AWGN vs Rayleigh side-by-side…")
SNR_COMPARE = 10   # fixed SNR for this comparison

fig, axes = plt.subplots(4, 2, figsize=(8, 16))

for row, mod in enumerate(MODS):
    for col, (ch_name, ch_fn) in enumerate(CHANNELS.items()):
        ax = axes[row][col]
        tx = MOD_FN[mod](N_SYMBOLS)
        rx = ch_fn(tx, SNR_COMPARE)
        plot_constellation(ax, tx, rx, mod, SNR_COMPARE, ch_name, MOD_COLORS[mod])

fig.suptitle(
    f"Figure 8 — AWGN vs Rayleigh Constellation Comparison at {SNR_COMPARE} dB SNR\n"
    "Left = AWGN  ·  Right = Rayleigh  ·  Black crosses = ideal points",
    fontsize=6, fontweight="bold"
)
fig.tight_layout()
p = os.path.join(OUT, "fig8_constellations_snr_compare.png")
fig.savefig(p, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {p}")


# ══════════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════════
print(f"""
All constellation diagrams saved to: {OUT}/

  fig6_constellations_awgn.png        — AWGN channel (3 SNR levels × 4 mods)
  fig7_constellations_rayleigh.png    — Rayleigh channel (3 SNR levels × 4 mods)
  fig8_constellations_snr_compare.png — AWGN vs Rayleigh side-by-side at {SNR_COMPARE} dB
""")