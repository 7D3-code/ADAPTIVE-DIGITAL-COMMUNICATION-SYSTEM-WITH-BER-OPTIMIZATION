# 📡 Adaptive Communication Dashboard (Streamlit)

A full interactive simulation dashboard for digital communication systems with:
- Multiple modulation schemes (BPSK, QPSK, 16-QAM, 64-QAM)
- Channel models (AWGN and Rayleigh fading)
- BER analysis and SNR sweeps
- Adaptive modulation ranking
- Forward Error Correction (Hamming (7,4) and Convolutional coding + Viterbi decoding)
- Constellation visualization and decoded message recovery

The app is implemented in `app.py` and runs locally with Streamlit.

---

## 1) What this system does

This dashboard simulates an end-to-end digital communication chain:
1. Convert user input (number/text) into bits.
2. Encode bits (optional FEC depending on selected tab).
3. Modulate bits into symbols.
4. Transmit through a noisy channel.
5. Demodulate received symbols back to bits.
6. Decode FEC (if enabled).
7. Compute BER/throughput/score.
8. Compare modulation options and display best choice.
9. Plot BER vs SNR and constellation diagrams.

---

## 2) Project contents

For this app-based setup:
- `app.py` → Streamlit application (UI + simulation engine)
- `README.md` → documentation (this file)

---

## 3) Requirements

- Python 3.10+ (3.11 recommended)
- Packages:
  - `streamlit`
  - `numpy`
  - `matplotlib`
  - `pandas`

---

## 4) Quick start (Windows PowerShell)

> Important: run each command as a separate line.

1. Open PowerShell in the folder containing `app.py`.
2. Create virtual environment:
   - `python -m venv .venv`
3. Activate virtual environment:
   - `.\.venv\Scripts\Activate.ps1`
4. Install dependencies:
   - `pip install streamlit numpy matplotlib pandas`
5. Run app:
   - `streamlit run app.py`

Expected URL: `http://localhost:8501`

### If activation is blocked
Run:
- `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`

Then activate again:
- `.\.venv\Scripts\Activate.ps1`

### If `streamlit` is not recognized
Run with Python module mode:
- `python -m streamlit run app.py`

---

## 5) User workflow in the dashboard

### Sidebar inputs
- **Message / Number**: payload source
- **SNR (dB)**: channel quality control
- **Channel**: AWGN or Rayleigh

### Tabs
1. **Adaptive (No FEC)**
   - Compares BER, throughput, score for all modulations.
   - Chooses best modulation for current settings.

2. **Hamming(7,4) FEC**
   - Adds block coding before modulation.
   - Corrects single-bit errors per 7-bit codeword.
   - Shows BER pre-FEC and post-FEC.

3. **Convolutional FEC**
   - Uses rate 1/2, K=3 encoder.
   - Uses hard-decision Viterbi decoder.
   - Shows BER pre-FEC and post-decoding.

---

## 6) End-to-end system flowchart (full)

```mermaid
flowchart TD
    A[Start App] --> B[Load UI + Theme + Parameters]
    B --> C[User enters message/number]
    C --> D[input_to_bits]
    D --> E{Data type?}
    E -->|Number| F[16-bit numeric representation]
    E -->|Text| G[ASCII bytes -> bitstream]
    F --> H[Base bitstream]
    G --> H

    H --> I[Read controls: SNR + Channel]
    I --> J{Selected tab}

    %% ================= TAB 1 =================
    J -->|Adaptive No FEC| K1[For each modulation: BPSK/QPSK/16-QAM/64-QAM]
    K1 --> L1[Modulate bits -> symbols]
    L1 --> M1[Pass through channel: AWGN or Rayleigh]
    M1 --> N1[Demodulate symbols -> bits]
    N1 --> O1[Compute BER]
    O1 --> P1[Compute throughput = bits_per_symbol x 1-BER]
    P1 --> Q1[Compute score = throughput/(BER + eps)]
    Q1 --> R1[Decode output bits to text/number]
    R1 --> S1[Aggregate per-modulation results]
    S1 --> T1[Select best score]
    T1 --> U1[Render table + received bits + decoded output]
    U1 --> V1[Plot BER vs SNR]
    V1 --> W1[Plot constellation per modulation]

    %% ================= TAB 2 =================
    J -->|Hamming 7,4 FEC| K2[For each modulation]
    K2 --> L2[hamming_encode: 4-bit blocks -> 7-bit codewords]
    L2 --> M2[Modulate encoded bits]
    M2 --> N2[Transmit via channel]
    N2 --> O2[Demodulate received coded bits]
    O2 --> P2[Compute BER pre-FEC]
    P2 --> Q2[hamming_decode with syndrome correction]
    Q2 --> R2[Compute BER post-FEC]
    R2 --> S2[Compute throughput with code rate 4/7]
    S2 --> T2[Compute score]
    T2 --> U2[Decode recovered bits to original datatype]
    U2 --> V2[Show corrected errors + message comparison]
    V2 --> W2[Render BER curves and constellations]

    %% ================= TAB 3 =================
    J -->|Convolutional FEC| K3[For each modulation]
    K3 --> L3[conv_encode: rate 1/2, K=3, G1=111, G2=101]
    L3 --> M3[Modulate encoded bits]
    M3 --> N3[Transmit via channel]
    N3 --> O3[Demodulate hard bits]
    O3 --> P3[Compute BER pre-decoding]
    P3 --> Q3[Viterbi hard-decision decode]
    Q3 --> R3[Compute BER post-decoding]
    R3 --> S3[Compute throughput with code rate 1/2]
    S3 --> T3[Compute score]
    T3 --> U3[Decode bits to text/number]
    U3 --> V3[Show BER improvement + pipeline detail]
    V3 --> W3[Render BER curves and constellations]

    W1 --> X[End Render]
    W2 --> X
    W3 --> X
```

---

## 7) Core algorithms and formulas

### BER
- `BER = (# bit errors) / (total compared bits)`

### Throughput metric (dashboard scoring)
- No FEC: `throughput = bits_per_symbol x (1 - BER)`
- Hamming: `throughput = bits_per_symbol x (4/7) x (1 - BER_post)`
- Convolutional: `throughput = bits_per_symbol x (1/2) x (1 - BER_post)`

### Score used for adaptive decision
- `score = throughput / (BER + 1e-6)`

This score rewards high payload efficiency and low error probability.

---

## 8) Module-by-module explanation of `app.py`

### Input/Output conversion
- `input_to_bits(user_input)`
  - Tries numeric conversion first.
  - If numeric: uses fixed 16-bit binary.
  - Else: treats as text and converts each char to 8-bit ASCII.
- `bits_to_output(bits, dtype)`
  - Reconstructs integer (16 bits) or text (8-bit chunks).

### Modulation / demodulation
- BPSK: 1 bit/symbol
- QPSK: 2 bits/symbol (I/Q)
- 16-QAM: 4 bits/symbol
- 64-QAM: 6 bits/symbol

### Channel models
- `awgn(x, snr)` adds white Gaussian noise based on SNR.
- `rayleigh(x, snr)` multiplies by fading coefficient then applies AWGN and equalizes.

### Simulation engines
- `run_scheme(...)` for adaptive without FEC.
- `run_hamming(...)` for Hamming(7,4) path.
- `run_conv(...)` for convolutional + Viterbi path.

### Plotting helpers
- `plot_ber_vs_snr(...)`: BER trend over SNR for selected run mode.
- `plot_constellations(...)`: received symbol scatter diagrams.

---

## 9) Practical interpretation guide

- At low SNR:
  - Lower-order modulation (BPSK/QPSK) usually gives lower BER.
  - FEC shows stronger benefit.

- At high SNR:
  - Higher-order modulation (16/64-QAM) can win by throughput.
  - BER naturally drops, improving score.

- Rayleigh channels generally produce worse BER than AWGN for same SNR.

---

## 10) Troubleshooting

### Problem: command failed after `python -m venv .venv .\.venv\Scripts\Activate.ps1`
Cause: two commands were combined into one line incorrectly.

Fix:
1. `python -m venv .venv`
2. `.\.venv\Scripts\Activate.ps1`

### Problem: app does not start
- Verify environment is active.
- Reinstall dependencies.
- Try module mode: `python -m streamlit run app.py`

### Problem: strange decoded text
- Noise can corrupt bits at low SNR.
- Increase SNR and/or use FEC tabs.

---

## 11) Suggested future upgrades

- Add soft-decision Viterbi decoding
- Add LDPC or Turbo coding modes
- Add OFDM framework (pilot-assisted channel estimation)
- Export results to CSV/JSON
- Add reproducibility control (random seed)

---

## 12) License / usage

Use for coursework, simulation experiments, and BER/FEC demonstrations.
If sharing publicly, include author attribution and simulation assumptions.
