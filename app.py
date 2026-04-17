import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.facecolor'] = '#0d0d1a'
matplotlib.rcParams['axes.facecolor'] = '#0d0d1a'
matplotlib.rcParams['axes.edgecolor'] = '#334155'
matplotlib.rcParams['text.color'] = '#e2e8f0'
matplotlib.rcParams['axes.labelcolor'] = '#94a3b8'
matplotlib.rcParams['xtick.color'] = '#94a3b8'
matplotlib.rcParams['ytick.color'] = '#94a3b8'
matplotlib.rcParams['grid.color'] = '#1e293b'
import pandas as pd

st.set_page_config(layout="wide", page_title="📡 Comm Dashboard", page_icon="📡")

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #060612;
    color: #e2e8f0;
}
h1, h2, h3 { font-family: 'Syne', sans-serif; }
code, .stCode { font-family: 'Space Mono', monospace !important; font-size: 0.75rem; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d1a 0%, #0a0a16 100%);
    border-right: 1px solid #1e293b;
}
[data-testid="stSidebar"] .stMarkdown h2 { color: #38bdf8; letter-spacing: 0.05em; }

.stTabs [data-baseweb="tab-list"] {
    background: #0d0d1a;
    border-bottom: 1px solid #1e293b;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.85rem;
    letter-spacing: 0.06em;
    color: #475569;
    padding: 0.75rem 1.5rem;
    border-bottom: 2px solid transparent;
    transition: all 0.2s ease;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #94a3b8; }

div[data-testid="metric-container"] {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.75rem;
}

.stSuccess {
    background: #052e16 !important;
    border: 1px solid #166534 !important;
    border-radius: 8px !important;
    color: #4ade80 !important;
}

[data-testid="stDataFrame"] { border: 1px solid #1e293b; border-radius: 8px; }

h2 { border-left: 3px solid #38bdf8; padding-left: 0.6rem; margin-top: 1.5rem; }
h3 { color: #7dd3fc; }

.stInfo { background: #082f49 !important; border: 1px solid #0369a1 !important; border-radius:8px!important; }
.stWarning { background: #3d2000 !important; border: 1px solid #92400e !important; border-radius:8px!important; }
</style>
""", unsafe_allow_html=True)

# ===============================
# INPUT / OUTPUT
# ===============================
def input_to_bits(user_input):
    try:
        num = int(float(user_input))
        return np.array(list(map(int, format(num, '016b')))), "number"
    except:
        bits = [int(b) for c in user_input for b in format(ord(c), '08b')]
        return np.array(bits), "text"

def bits_to_output(bits, dtype):
    if dtype == "number":
        try:
            return int("".join(map(str, bits[:16])), 2)
        except:
            return "?"
    else:
        chars = []
        for i in range(0, len(bits), 8):
            byte = bits[i:i+8]
            if len(byte) == 8:
                try:
                    chars.append(chr(int("".join(map(str, byte)), 2)))
                except:
                    chars.append('?')
        return "".join(chars)

# ===============================
# MODULATION
# ===============================
def bpsk_mod(b): return 2*b - 1
def bpsk_demod(x): return (x > 0).astype(int)

def qpsk_mod(b):
    b = b[:len(b)//2*2]
    p = b.reshape(-1, 2)
    return (2*p[:,0]-1 + 1j*(2*p[:,1]-1))/np.sqrt(2)

def qpsk_demod(s):
    out = []
    for x in s:
        out += [int(np.real(x)>0), int(np.imag(x)>0)]
    return np.array(out)

def qam16_mod(b):
    b = b[:len(b)//4*4]
    s = b.reshape(-1, 4)
    m = {(0,0):-3,(0,1):-1,(1,1):1,(1,0):3}
    I = [m[tuple(x[:2])] for x in s]
    Q = [m[tuple(x[2:])] for x in s]
    return (np.array(I)+1j*np.array(Q))/np.sqrt(10)

def qam16_demod(s):
    out = []
    for x in s:
        i, q = np.real(x)*np.sqrt(10), np.imag(x)*np.sqrt(10)
        out += [int(i>0), int(abs(i)<2)]
        out += [int(q>0), int(abs(q)<2)]
    return np.array(out)

def qam64_mod(b):
    b = b[:len(b)//6*6]
    s = b.reshape(-1, 6)
    levels = [-7,-5,-3,-1,1,3,5,7]
    def m(x): return levels[int("".join(map(str,x)),2)]
    I = [m(x[:3]) for x in s]
    Q = [m(x[3:]) for x in s]
    return (np.array(I)+1j*np.array(Q))/np.sqrt(42)

def qam64_demod(s):
    out = []
    for x in s:
        i = int(np.clip(np.real(x)*np.sqrt(42),-7,7))
        q = int(np.clip(np.imag(x)*np.sqrt(42),-7,7))
        out += list(map(int, format((i+7)//2,'03b')))
        out += list(map(int, format((q+7)//2,'03b')))
    return np.array(out)

# ===============================
# CHANNEL
# ===============================
def awgn(x, snr):
    snr_lin = 10**(snr/10)
    if np.iscomplexobj(x):
        n = (np.random.randn(len(x))+1j*np.random.randn(len(x)))/np.sqrt(2*snr_lin)
    else:
        n = np.random.randn(len(x))/np.sqrt(2*snr_lin)
    return x + n

def rayleigh(x, snr):
    h = (np.random.randn(len(x))+1j*np.random.randn(len(x)))/np.sqrt(2)
    return awgn(x*h, snr)/h

# ===============================
# BER
# ===============================
def ber(a, b):
    L = min(len(a), len(b))
    return np.mean(a[:L] != b[:L])

# ===============================
# MODULATE / DEMODULATE HELPER
# ===============================
def mod_tx_rx(bits, mod, channel, snr):
    if mod == "BPSK":
        tx = bpsk_mod(bits); rx = channel(tx, snr); dem = bpsk_demod(rx)
    elif mod == "QPSK":
        tx = qpsk_mod(bits); rx = channel(tx, snr); dem = qpsk_demod(rx)
    elif mod == "16-QAM":
        tx = qam16_mod(bits); rx = channel(tx, snr); dem = qam16_demod(rx)
    elif mod == "64-QAM":
        tx = qam64_mod(bits); rx = channel(tx, snr); dem = qam64_demod(rx)
    return dem, rx

BITS_PER_SYMBOL = {"BPSK": 1, "QPSK": 2, "16-QAM": 4, "64-QAM": 6}

# ===============================
# SIMULATION — ADAPTIVE (NO FEC)
# ===============================
def run_scheme(bits, mod, channel, snr, dtype):
    dem, rx = mod_tx_rx(bits, mod, channel, snr)
    ber_val = ber(bits, dem)
    bps = BITS_PER_SYMBOL[mod]
    throughput = bps * (1 - ber_val)
    score = throughput / (ber_val + 1e-6)
    return {
        "received_bits": dem,
        "decoded": bits_to_output(dem, dtype),
        "ber": ber_val,
        "bps": bps,
        "throughput": throughput,
        "score": score,
        "symbols_rx": rx
    }

# ===============================
# FEC — HAMMING (7,4)
# ===============================
G_HAM = np.array([
    [1,0,0,0, 1,1,0],
    [0,1,0,0, 1,0,1],
    [0,0,1,0, 0,1,1],
    [0,0,0,1, 1,1,1],
], dtype=int)

H_HAM = np.array([
    [1,1,0,1,1,0,0],
    [1,0,1,1,0,1,0],
    [0,1,1,1,0,0,1],
], dtype=int)

def hamming_encode(bits):
    bits = np.array(bits, dtype=int)
    pad = (4 - len(bits) % 4) % 4
    bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    blocks = bits.reshape(-1, 4)
    coded = []
    for b in blocks:
        cw = (b @ G_HAM) % 2
        coded.extend(cw)
    return np.array(coded, dtype=int), pad

def hamming_decode(coded_bits, original_len):
    coded_bits = np.array(coded_bits, dtype=int)
    pad_to = (7 - len(coded_bits) % 7) % 7
    coded_bits = np.concatenate([coded_bits, np.zeros(pad_to, dtype=int)])
    blocks = coded_bits.reshape(-1, 7)
    decoded = []
    errors_corrected = 0
    for cw in blocks:
        syndrome = (H_HAM @ cw) % 2
        idx = int("".join(map(str, syndrome)), 2)
        if idx != 0 and idx <= 7:
            cw[idx-1] ^= 1
            errors_corrected += 1
        decoded.extend(cw[:4])
    decoded = np.array(decoded[:original_len], dtype=int)
    return decoded, errors_corrected

def run_hamming(bits, mod, channel, snr, dtype):
    original_len = len(bits)
    encoded_bits, pad = hamming_encode(bits)
    dem, rx = mod_tx_rx(encoded_bits, mod, channel, snr)
    L = min(len(dem), len(encoded_bits))
    dem = dem[:L]
    ber_before = ber(encoded_bits[:L], dem)
    decoded, errors_corrected = hamming_decode(dem, original_len)
    ber_after = ber(bits, decoded)
    bps = BITS_PER_SYMBOL[mod]
    rate = 4/7
    throughput = bps * rate * (1 - ber_after)
    score = throughput / (ber_after + 1e-6)
    return {
        "received_bits": decoded,
        "decoded": bits_to_output(decoded, dtype),
        "ber_before": ber_before,
        "ber_after": ber_after,
        "errors_corrected": errors_corrected,
        "bps": bps,
        "throughput": throughput,
        "score": score,
        "symbols_rx": rx,
        "encoded_bits": encoded_bits,
        "received_coded": dem,
    }

# ===============================
# FEC — CONVOLUTIONAL (Rate 1/2, K=3)
# ===============================
G1 = [1, 1, 1]
G2 = [1, 0, 1]

def conv_encode(bits):
    bits = np.array(bits, dtype=int)
    encoded = []
    state = [0, 0]
    for b in bits:
        reg = [b] + state
        o1 = (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2
        o2 = (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2
        encoded.extend([o1, o2])
        state = [b, state[0]]
    for _ in range(2):
        b = 0
        reg = [b] + state
        o1 = (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2
        o2 = (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2
        encoded.extend([o1, o2])
        state = [b, state[0]]
    return np.array(encoded, dtype=int)

def conv_decode_viterbi(received, original_len):
    received = np.array(received, dtype=int)
    n_states = 4
    n_symbols = len(received) // 2

    outputs = {}
    next_state = {}
    for s in range(n_states):
        for inp in range(2):
            s_bits = [(s >> i) & 1 for i in range(1, -1, -1)]
            reg = [inp] + s_bits
            o1 = (reg[0]*G1[0] + reg[1]*G1[1] + reg[2]*G1[2]) % 2
            o2 = (reg[0]*G2[0] + reg[1]*G2[1] + reg[2]*G2[2]) % 2
            outputs[(s, inp)] = (o1, o2)
            ns_bits = [inp] + s_bits[:-1]
            ns = ns_bits[0]*2 + ns_bits[1]
            next_state[(s, inp)] = ns

    INF = 1e9
    pm = np.full(n_states, INF)
    pm[0] = 0
    paths = {s: [] for s in range(n_states)}

    for t in range(n_symbols):
        r = received[t*2:(t*2)+2]
        new_pm = np.full(n_states, INF)
        new_paths = {s: [] for s in range(n_states)}
        for s in range(n_states):
            for inp in range(2):
                ns = next_state[(s, inp)]
                o = outputs[(s, inp)]
                hd = int(r[0] != o[0]) + int(r[1] != o[1])
                cost = pm[s] + hd
                if cost < new_pm[ns]:
                    new_pm[ns] = cost
                    new_paths[ns] = paths[s] + [inp]
        pm = new_pm
        paths = new_paths

    best_state = int(np.argmin(pm))
    decoded = np.array(paths[best_state][:original_len], dtype=int)
    if len(decoded) < original_len:
        decoded = np.concatenate([decoded, np.zeros(original_len - len(decoded), dtype=int)])
    return decoded

def run_conv(bits, mod, channel, snr, dtype):
    original_len = len(bits)
    encoded_bits = conv_encode(bits)
    dem, rx = mod_tx_rx(encoded_bits, mod, channel, snr)
    L = min(len(dem), len(encoded_bits))
    dem = dem[:L]
    ber_before = ber(encoded_bits[:L], dem)
    decoded = conv_decode_viterbi(dem, original_len)
    ber_after = ber(bits, decoded)
    bps = BITS_PER_SYMBOL[mod]
    rate = 0.5
    throughput = bps * rate * (1 - ber_after)
    score = throughput / (ber_after + 1e-6)
    return {
        "received_bits": decoded,
        "decoded": bits_to_output(decoded, dtype),
        "ber_before": ber_before,
        "ber_after": ber_after,
        "bps": bps,
        "throughput": throughput,
        "score": score,
        "symbols_rx": rx,
        "encoded_bits": encoded_bits,
        "received_coded": dem,
    }

# ===============================
# SHARED PLOT HELPERS
# ===============================
COLORS = ['#38bdf8','#a78bfa','#34d399','#fb923c']

def plot_ber_vs_snr(bits, mods, channel, snr, run_fn, dtype):
    snr_range = np.linspace(0, snr, 10)
    fig, ax = plt.subplots(figsize=(8, 4))
    for j, m in enumerate(mods):
        bers_b, bers_a = [], []
        for s in snr_range:
            r = run_fn(bits=bits, mod=m, channel=channel, snr=s, dtype=dtype)
            if "ber_before" in r:
                bers_b.append(r["ber_before"])
                bers_a.append(r["ber_after"])
            else:
                bers_b.append(r["ber"])
        ax.plot(snr_range, bers_b, marker='o', label=f"{m}", color=COLORS[j], linewidth=2)
        if bers_a:
            ax.plot(snr_range, bers_a, marker='s', linestyle='--',
                    label=f"{m} (post-FEC)", color=COLORS[j], alpha=0.6)
    ax.set_xlabel("SNR (dB)", fontsize=11)
    ax.set_ylabel("BER", fontsize=11)
    ax.set_title("BER vs SNR", fontsize=13, fontweight='bold', color='#e2e8f0')
    ax.legend(fontsize=8, facecolor='#0d1117', edgecolor='#1e293b', labelcolor='#94a3b8')
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig

def plot_constellations(results, mods):
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5))
    for i, m in enumerate(mods):
        ax = axes[i]
        sym = results[m]["symbols_rx"]
        ax.scatter(np.real(sym), np.imag(sym), s=6, alpha=0.7, color=COLORS[i])
        ax.set_title(m, fontsize=11, color='#e2e8f0')
        ax.grid(True, alpha=0.3)
        ax.set_facecolor('#0d0d1a')
    fig.patch.set_facecolor('#0d0d1a')
    fig.tight_layout()
    return fig

# ===============================
# SIDEBAR
# ===============================
st.sidebar.title("⚙️ Settings")
user_input = st.sidebar.text_input("Message / Number", "HELLO")
snr = st.sidebar.slider("SNR (dB)", 1, 30, 10)
channel_type = st.sidebar.selectbox("Channel", ["AWGN", "Rayleigh"])
channel = awgn if channel_type == "AWGN" else rayleigh

st.sidebar.markdown("---")
st.sidebar.markdown("**FEC Reference**")
st.sidebar.info(
    "**Hamming(7,4)**\n"
    "Rate: 4/7 ≈ 0.571\n"
    "Corrects: 1 bit/block\n\n"
    "**Conv. (K=3, R=1/2)**\n"
    "Rate: 0.5\n"
    "Decoder: Viterbi (hard)"
)

# ===============================
# MAIN
# ===============================
st.title("📡 Adaptive Communication Dashboard")

bits, dtype = input_to_bits(user_input)

st.subheader("📤 Transmitted Data")
col_a, col_b = st.columns([1, 3])
with col_a:
    st.metric("Total Bits", len(bits))
    st.metric("Data Type", dtype.capitalize())
with col_b:
    st.code("".join(map(str, bits))[:200] + ("..." if len(bits) > 200 else ""), language=None)

mods = ["BPSK", "QPSK", "16-QAM", "64-QAM"]

# ─── NAVIGATION TABS ─────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔵  Adaptive (No FEC)",
    "🟢  Hamming(7,4) FEC",
    "🟠  Convolutional FEC"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — ADAPTIVE (NO FEC)
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    results = {m: run_scheme(bits, m, channel, snr, dtype) for m in mods}
    best = max(results, key=lambda x: results[x]["score"])

    st.success(f"🏆 Best Modulation: **{best}** — Score: {results[best]['score']:.2f}")

    st.subheader("📊 Performance Comparison")
    data = []
    for m in mods:
        r = results[m]
        data.append({"Modulation": m, "BER": f"{r['ber']:.6f}",
                     "Bits/Symbol": r["bps"], "Throughput": f"{r['throughput']:.3f}",
                     "Score": f"{r['score']:.2f}"})
    df = pd.DataFrame(data)
    def hi(row): return ['background-color:#052e16']*len(row) if row.Modulation==best else ['']*len(row)
    st.dataframe(df.style.apply(hi, axis=1), use_container_width=True)

    st.subheader("📥 Received Data & Decoding")
    cols = st.columns(4)
    for i, m in enumerate(mods):
        with cols[i]:
            r = results[m]
            st.markdown(f"### {m}")
            st.metric("BER", f"{r['ber']:.6f}")
            st.write("🔢 Received Bits:")
            st.code("".join(map(str, r["received_bits"][:80]))+"...", language=None)
            st.write("🧾 Decoded Output:")
            st.code(str(r["decoded"]), language=None)

    st.subheader("📈 BER vs SNR")
    st.pyplot(plot_ber_vs_snr(bits, mods, channel, snr, run_scheme, dtype))

    st.subheader("🌌 Constellation Diagrams")
    st.pyplot(plot_constellations(results, mods))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — HAMMING(7,4) FEC
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    results_h = {m: run_hamming(bits, m, channel, snr, dtype) for m in mods}
    best_h = max(results_h, key=lambda x: results_h[x]["score"])

    st.success(f"🏆 Best Modulation with Hamming FEC: **{best_h}** — Score: {results_h[best_h]['score']:.2f}")
    st.info("**Hamming(7,4):** Every 4 data bits → 7 transmitted bits (3 parity added). "
            "Can detect and **correct 1-bit errors** per 7-bit block. Code rate = 4/7 ≈ 0.571.")

    # ── Message Before / After ────────────────────────────────────────────────
    st.subheader("💬 Message: Before & After Hamming FEC Decoding")
    cols_msg = st.columns(2)
    with cols_msg[0]:
        st.markdown("#### 📤 Original Message (Before Encoding)")
        st.write(f"**Text/Value:** `{bits_to_output(bits, dtype)}`")
        st.write(f"**Bit length:** {len(bits)} bits")
        st.code("".join(map(str, bits))[:160]+("..." if len(bits)>160 else ""), language=None)
    with cols_msg[1]:
        r_best_h = results_h[best_h]
        match_h = str(r_best_h["decoded"]) == str(bits_to_output(bits, dtype))
        st.markdown(f"#### 📥 Decoded Message (After Hamming Decode — {best_h})")
        st.write(f"**Text/Value:** `{r_best_h['decoded']}`  {'✅ Match' if match_h else '⚠️ Mismatch'}")
        st.write(f"**Errors corrected:** {r_best_h['errors_corrected']} block(s)")
        st.code("".join(map(str, r_best_h["received_bits"]))[:160]+("..." if len(r_best_h["received_bits"])>160 else ""), language=None)

    # ── Performance Table ─────────────────────────────────────────────────────
    st.subheader("📊 Hamming FEC Performance Comparison")
    data_h = []
    for m in mods:
        r = results_h[m]
        data_h.append({
            "Modulation": m,
            "BER (pre-FEC)": f"{r['ber_before']:.6f}",
            "BER (post-FEC)": f"{r['ber_after']:.6f}",
            "Errors Corrected": r["errors_corrected"],
            "Bits/Symbol": r["bps"],
            "Code Rate": "4/7",
            "Throughput": f"{r['throughput']:.3f}",
            "Score": f"{r['score']:.2f}"
        })
    df_h = pd.DataFrame(data_h)
    def hi_h(row): return ['background-color:#052e16']*len(row) if row.Modulation==best_h else ['']*len(row)
    st.dataframe(df_h.style.apply(hi_h, axis=1), use_container_width=True)

    # ── Per-Modulation Detail ─────────────────────────────────────────────────
    st.subheader("📥 Per-Modulation: Bit Pipeline Detail")
    cols_h = st.columns(4)
    for i, m in enumerate(mods):
        with cols_h[i]:
            r = results_h[m]
            st.markdown(f"### {m}")
            c1, c2 = st.columns(2)
            c1.metric("Pre-FEC BER", f"{r['ber_before']:.5f}")
            c2.metric("Post-FEC BER", f"{r['ber_after']:.5f}",
                      delta=f"{r['ber_after']-r['ber_before']:+.5f}", delta_color="inverse")
            st.write(f"🔧 Errors corrected: **{r['errors_corrected']}**")
            st.markdown("**📨 Encoded bits (sent):**")
            st.code("".join(map(str, r["encoded_bits"][:56]))+"...", language=None)
            st.markdown("**📬 Received coded (noisy):**")
            st.code("".join(map(str, r["received_coded"][:56]))+"...", language=None)
            st.markdown("**✅ After Hamming decode:**")
            st.code("".join(map(str, r["received_bits"][:40]))+"...", language=None)
            st.write(f"💬 `{r['decoded']}`")

    st.subheader("📈 BER vs SNR  (solid = pre-FEC · dashed = post-FEC)")
    st.pyplot(plot_ber_vs_snr(bits, mods, channel, snr, run_hamming, dtype))

    st.subheader("🌌 Constellation Diagrams")
    st.pyplot(plot_constellations(results_h, mods))

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CONVOLUTIONAL FEC
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    results_c = {m: run_conv(bits, m, channel, snr, dtype) for m in mods}
    best_c = max(results_c, key=lambda x: results_c[x]["score"])

    st.success(f"🏆 Best Modulation with Convolutional FEC: **{best_c}** — Score: {results_c[best_c]['score']:.2f}")
    st.info("**Convolutional Code (K=3, Rate 1/2):** Every 1 input bit → 2 coded bits. "
            "Generators G1=111, G2=101. Decoded with **Viterbi algorithm** (hard-decision). "
            "Excellent performance against burst errors.")

    # ── Message Before / After ────────────────────────────────────────────────
    st.subheader("💬 Message: Before & After Convolutional FEC Decoding")
    cols_msg2 = st.columns(2)
    with cols_msg2[0]:
        enc_preview = conv_encode(bits)
        st.markdown("#### 📤 Original Message (Before Encoding)")
        st.write(f"**Text/Value:** `{bits_to_output(bits, dtype)}`")
        st.write(f"**Bit length:** {len(bits)} bits → **{len(enc_preview)} coded bits** after R=1/2 encoding")
        st.code("".join(map(str, bits))[:160]+("..." if len(bits)>160 else ""), language=None)
    with cols_msg2[1]:
        r_best_c = results_c[best_c]
        match_c = str(r_best_c["decoded"]) == str(bits_to_output(bits, dtype))
        st.markdown(f"#### 📥 Decoded Message (After Viterbi Decode — {best_c})")
        st.write(f"**Text/Value:** `{r_best_c['decoded']}`  {'✅ Match' if match_c else '⚠️ Partial recovery'}")
        st.write(f"**BER improvement:** {r_best_c['ber_before']:.5f} → {r_best_c['ber_after']:.5f}")
        st.code("".join(map(str, r_best_c["received_bits"]))[:160]+("..." if len(r_best_c["received_bits"])>160 else ""), language=None)

    # ── Performance Table ─────────────────────────────────────────────────────
    st.subheader("📊 Convolutional FEC Performance Comparison")
    data_c = []
    for m in mods:
        r = results_c[m]
        data_c.append({
            "Modulation": m,
            "BER (pre-FEC)": f"{r['ber_before']:.6f}",
            "BER (post-FEC)": f"{r['ber_after']:.6f}",
            "Bits/Symbol": r["bps"],
            "Code Rate": "1/2",
            "Throughput": f"{r['throughput']:.3f}",
            "Score": f"{r['score']:.2f}"
        })
    df_c = pd.DataFrame(data_c)
    def hi_c(row): return ['background-color:#052e16']*len(row) if row.Modulation==best_c else ['']*len(row)
    st.dataframe(df_c.style.apply(hi_c, axis=1), use_container_width=True)

    # ── Per-Modulation Detail ─────────────────────────────────────────────────
    st.subheader("📥 Per-Modulation: Bit Pipeline Detail")
    cols_c = st.columns(4)
    for i, m in enumerate(mods):
        with cols_c[i]:
            r = results_c[m]
            st.markdown(f"### {m}")
            c1, c2 = st.columns(2)
            c1.metric("Pre-FEC BER", f"{r['ber_before']:.5f}")
            c2.metric("Post-FEC BER", f"{r['ber_after']:.5f}",
                      delta=f"{r['ber_after']-r['ber_before']:+.5f}", delta_color="inverse")
            st.markdown("**📨 Encoded bits (sent):**")
            st.code("".join(map(str, r["encoded_bits"][:80]))+"...", language=None)
            st.markdown("**📬 Received coded (noisy):**")
            st.code("".join(map(str, r["received_coded"][:80]))+"...", language=None)
            st.markdown("**✅ After Viterbi decode:**")
            st.code("".join(map(str, r["received_bits"][:40]))+"...", language=None)
            st.write(f"💬 `{r['decoded']}`")

    st.subheader("📈 BER vs SNR  (solid = pre-FEC · dashed = post-Viterbi)")
    st.pyplot(plot_ber_vs_snr(bits, mods, channel, snr, run_conv, dtype))

    st.subheader("🌌 Constellation Diagrams")
    st.pyplot(plot_constellations(results_c, mods))