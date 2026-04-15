import dash
from dash import dcc, html, Input, Output
import plotly.graph_objects as go
import serial
import threading
import time
from collections import deque

# --- CONFIGURATION ---
RX_PORT = 'COM4'   # RX Arduino USB port
TX_PORT = 'COM3'   # TX Arduino USB port  
BAUD_RATE = 9600
AVG_WINDOW = 15

# --- GLOBAL DATA STORAGE ---
raw_ber_history = deque([0.0] * 100, maxlen=100)
avg_ber_history = deque([0.0] * 100, maxlen=100)
recent_samples = deque(maxlen=AVG_WINDOW)

error_rate_history = deque([0.0] * 100, maxlen=100)

stats = {
    "tx_data":        "---",       
    "tx_typing":      "",          
    "tx_raw_bits":    "0",         
    "tx_mod":         "16QAM",
    "tx_fec":         "HAM(7,4)",
    "tx_packets":     "0",
    "tx_last_seen":   0,
    "rx_data":        "WAITING...",
    "rx_data_corrected": "WAITING...",
    "rx_raw":         "0",
    "mod":            "16QAM",
    "ber":            "0.0000",
    "raw_bits":       "0",
    "error_bits":     "0",
    "corrected_bits": "0",
    "fec_type":       "HAM(7,4)",
    "status":         "OFFLINE",
    "rx_last_seen":   0,
    "tx_binary":      "",
    "rx_binary_before": "",
    "rx_binary_after": "",
}
lock = threading.Lock()

# ---------------------------------------------------------------
# SERIAL CONNECTIONS
# ---------------------------------------------------------------
def open_serial(port, baud):
    try:
        s = serial.Serial(port, baud, timeout=0.1)
        print(f"[OK] Connected: {port}")
        return s
    except Exception as e:
        print(f"[ERR] Cannot open {port}: {e}")
        return None

ser_rx = open_serial(RX_PORT, BAUD_RATE)
ser_tx = open_serial(TX_PORT, BAUD_RATE)

# ---------------------------------------------------------------
# Helper: Convert string to binary
# ---------------------------------------------------------------
def string_to_binary(s):
    if not s or s == "---" or s == "WAITING..." or s == "(garbled)":
        return ""
    binary_str = ""
    for char in s:
        binary_str += format(ord(char), '08b') + " "
    return binary_str.strip()

# ---------------------------------------------------------------
# LISTENER: TX Arduino (COM3)
# ---------------------------------------------------------------
def tx_listener():
    """
    TX Arduino sends:
      TX_STATUS:READY              — on boot
      TX_STATUS_LIVE:<mod>|<fec>|<pkts>  — heartbeat every 2s
      TX_TYPING:<buffer>           — on every keypress
      TX_BITS:<n>                  — sent before TX_DATA on '#'
      TX_DATA:<message>            — on '#' press
      TX_CLEARED                   — on '*' press
      BER:<float>                  — from RX via BT, forwarded to dashboard
    """
    buf = b""
    while True:
        if ser_tx is None:
            time.sleep(1)
            continue
        try:
            waiting = ser_tx.in_waiting
            if waiting:
                buf += ser_tx.read(waiting)
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    print(f"[TX] {line}")

                    with lock:
                        stats["tx_last_seen"] = time.time()

                        if line.startswith("TX_DATA:"):
                            msg = line[8:].strip()
                            if msg:
                                stats["tx_data"] = msg
                                stats["tx_binary"] = string_to_binary(msg)
                            stats["tx_typing"] = ""
                            stats["status"] = "TRANSMITTING"

                        elif line.startswith("TX_BITS:"):
                            try:
                                stats["tx_raw_bits"] = str(int(line[8:].strip()))
                            except ValueError:
                                pass

                        elif line.startswith("TX_TYPING:"):
                            stats["tx_typing"] = line[10:].strip()

                        elif line == "TX_CLEARED":
                            stats["tx_typing"] = ""

                        elif line.startswith("TX_STATUS_LIVE:"):
                            parts = line[15:].split("|")
                            if len(parts) >= 3:
                                stats["tx_mod"] = parts[0]
                                stats["tx_fec"] = parts[1]
                                try:
                                    stats["tx_packets"] = parts[2].strip()
                                except Exception:
                                    pass

                        elif line.startswith("BER:"):
                            try:
                                ber_val = float(line[4:].strip())
                                with lock:
                                    raw_ber_history.append(ber_val)
                                    recent_samples.append(ber_val)
                                    avg_val = sum(recent_samples) / len(recent_samples)
                                    avg_ber_history.append(avg_val)
                                    stats["ber"] = f"{avg_val:.6f}"
                                    error_rate_history.append(avg_val * 100)
                            except ValueError:
                                pass

                        elif line.startswith("TX_STATUS:"):
                            if line[10:].strip() in ("READY", "ONLINE"):
                                stats["status"] = "ONLINE"

        except serial.SerialException as e:
            print(f"[TX LISTENER ERR] {e}")
            time.sleep(1)
        except Exception as e:
            print(f"[TX LISTENER ERR] {e}")

        time.sleep(0.005)


# ---------------------------------------------------------------
# LISTENER: RX Arduino (COM4)
# ---------------------------------------------------------------
def rx_listener():
    """
    RX Arduino sends:
      RX_CONTROLLER:READY
      BER:<float>
      MOD_DECISION:<modulation>
      RX_DATA:<errored>|<total_bits>|<error_bits>|<corrected_bits>
      RX_CORRECTED:<corrected>
      RX_RAW:<original_message>
      FEC:<fec_type>
    """
    buf = b""
    while True:
        if ser_rx is None:
            time.sleep(1)
            continue
        try:
            waiting = ser_rx.in_waiting
            if waiting:
                buf += ser_rx.read(waiting)
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode("utf-8", errors="replace").strip()
                    if not line:
                        continue

                    print(f"[RX] {line}")

                    with lock:
                        stats["rx_last_seen"] = time.time()
                        stats["status"] = "ACTIVE"

                        if line.startswith("BER:"):
                            try:
                                val = float(line[4:])
                                raw_ber_history.append(val)
                                recent_samples.append(val)
                                avg_val = sum(recent_samples) / len(recent_samples)
                                avg_ber_history.append(avg_val)
                                stats["ber"] = f"{avg_val:.6f}"
                                error_rate_history.append(avg_val * 100)
                            except ValueError:
                                pass

                        elif line.startswith("MOD_DECISION:"):
                            new_mod = line[13:].strip()
                            stats["mod"] = new_mod
                            if ser_tx:
                                try:
                                    ser_tx.write(f"SET_MOD:{new_mod}\n".encode())
                                    ser_tx.flush()
                                    print(f"[→TX] SET_MOD:{new_mod}")
                                except Exception as e:
                                    print(f"[FORWARD ERR] {e}")

                        elif line.startswith("RX_DATA:"):
                            parts = line[8:].split("|")
                            if len(parts) >= 4:
                                # RX_DATA contains the CORRECTED data (after FEC)
                                stats["rx_data_corrected"] = parts[0] if parts[0] else "(garbled)"
                                stats["rx_binary_after"] = string_to_binary(parts[0]) if parts[0] else ""
                                stats["raw_bits"] = parts[1]
                                stats["error_bits"] = parts[2]
                                stats["corrected_bits"] = parts[3]
                                try:
                                    eb = float(parts[2])
                                    tb = float(parts[1]) if float(parts[1]) > 0 else 1
                                    error_rate_history.append((eb / tb) * 100)
                                except (ValueError, ZeroDivisionError):
                                    pass

                        elif line.startswith("RX_CORRECTED:"):
                            # RX_CORRECTED actually contains the RAW/ERRORED data (before FEC)
                            raw_msg = line[13:].strip()
                            stats["rx_data"] = raw_msg
                            stats["rx_binary_before"] = string_to_binary(raw_msg)

                        elif line.startswith("RX_RAW:"):
                            original_msg = line[7:].strip()
                            stats["rx_raw"] = original_msg

                        elif line.startswith("FEC:"):
                            stats["fec_type"] = line[4:].strip()

                        elif line.startswith("RX_CONTROLLER:"):
                            stats["status"] = "ONLINE"

        except serial.SerialException as e:
            print(f"[RX LISTENER ERR] {e}")
            time.sleep(1)
        except Exception as e:
            print(f"[RX LISTENER ERR] {e}")

        time.sleep(0.005)


threading.Thread(target=tx_listener, daemon=True).start()
threading.Thread(target=rx_listener, daemon=True).start()

# ---------------------------------------------------------------
# DASH LAYOUT
# ---------------------------------------------------------------
app = dash.Dash(__name__)
MOD_COLORS = {'BPSK': '#ff1744', 'QPSK': '#ffab40', '16QAM': '#00e5ff'}

def card(label, elem_id, label_color='#aaa', border=None, font_size='20px', value_color='#64ffda'):
    style = {
        'background': '#1c252e', 'padding': '15px',
        'textAlign': 'center', 'borderRadius': '8px'
    }
    if border:
        style['border'] = f'1px solid {border}'
    return html.Div([
        html.P(label, style={'color': label_color, 'margin': '0 0 5px 0', 'fontSize': '11px'}),
        html.H2(id=elem_id, style={'margin': 0, 'fontSize': font_size, 'color': value_color})
    ], style=style)


def binary_card(label, elem_id):
    return html.Div(
        style={
            'background': '#0b1015', 'padding': '12px', 'borderRadius': '6px',
            'border': '1px solid #1e2d3d', 'marginBottom': '10px'
        },
        children=[
            html.P(label, style={'color': '#aaa', 'margin': '0 0 8px 0', 'fontSize': '11px'}),
            html.Pre(id=elem_id, style={
                'color': '#64ffda', 'fontSize': '10px', 'fontFamily': 'monospace',
                'margin': 0, 'wordBreak': 'break-all', 'whiteSpace': 'pre-wrap'
            })
        ]
    )


app.layout = html.Div(
    style={
        'backgroundColor': '#0b1015', 'color': 'white',
        'fontFamily': 'monospace', 'padding': '20px', 'minHeight': '100vh'
    },
    children=[
        html.H1("2.4GHz ADAPTIVE BER OPTIMIZATION COMMUNICATION SYSTEM",
                style={'textAlign': 'center', 'color': '#00e5ff', 'letterSpacing': '4px',
                       'fontSize': '18px', 'marginBottom': '20px'}),

        # --- Connection status bar ---
        html.Div(
            style={'display': 'flex', 'gap': '10px', 'marginBottom': '15px'},
            children=[
                html.Div(id='conn-rx', style={'flex': 1, 'padding': '8px', 'textAlign': 'center',
                                              'borderRadius': '6px', 'border': '1px solid #333',
                                              'fontSize': '12px'}),
                html.Div(id='conn-tx', style={'flex': 1, 'padding': '8px', 'textAlign': 'center',
                                              'borderRadius': '6px', 'border': '1px solid #333',
                                              'fontSize': '12px'}),
            ]
        ),

        # --- Top row: controls + stats grid ---
        html.Div(style={'display': 'flex', 'gap': '20px', 'marginBottom': '20px'}, children=[

            # Controls panel
            html.Div(
                style={
                    'flex': '1', 'background': '#161d24', 'padding': '20px',
                    'borderRadius': '10px', 'border': '1px solid #1e2d3d'
                },
                children=[
                    html.H3("ENVIRONMENT CONTROL", style={'color': '#00e5ff', 'marginTop': 0, 'fontSize': '13px'}),

                    html.Label("Channel Model:", style={'color': '#aaa', 'fontSize': '12px'}),
                    dcc.RadioItems(
                        id='ch-selector',
                        options=[
                            {'label': '  AWGN',     'value': 'CH:AWG'},
                            {'label': '  RAYLEIGH', 'value': 'CH:RAY'}
                        ],
                        value='CH:AWG',
                        style={'margin': '8px 0', 'color': 'white', 'fontSize': '13px'}
                    ),

                    html.Br(),
                    html.Label("Simulated SNR (dB):", style={'color': '#aaa', 'fontSize': '12px'}),
                    dcc.Slider(
                        id='snr-slider', min=1, max=25, step=1, value=15,
                        marks={1:'1', 5:'5', 10:'10', 15:'15', 20:'20', 25:'25'},
                        tooltip={"placement": "bottom", "always_visible": True}
                    ),

                    html.Br(),
                    html.Label("FEC Scheme:", style={'color': '#aaa', 'fontSize': '12px'}),
                    dcc.RadioItems(
                        id='fec-selector',
                        options=[
                            {'label': '  Hamming (7,4)',    'value': 'FEC:HAM(7,4)'},
                            {'label': '  Repetition (3x)', 'value': 'FEC:REP3X'},
                            {'label': '  None',             'value': 'FEC:NONE'}
                        ],
                        value='FEC:HAM(7,4)',
                        style={'margin': '8px 0', 'color': 'white', 'fontSize': '13px'}
                    ),

                    html.Div(
                        style={'background': '#0b1015', 'padding': '10px',
                               'borderRadius': '5px', 'marginTop': '12px'},
                        children=[
                            html.P("KEYPAD USAGE", style={'color': '#00e5ff', 'margin': '0 0 4px 0',
                                                          'fontWeight': 'bold', 'fontSize': '11px'}),
                            html.P("Type → # to transmit → * to clear",
                                   style={'color': '#aaa', 'margin': '2px 0', 'fontSize': '11px'}),
                        ]
                    ),
                ]
            ),

            # Stats grid
            html.Div(
                style={
                    'flex': '2', 'display': 'grid',
                    'gridTemplateColumns': '1fr 1fr', 'gap': '8px'
                },
                children=[
                    card("TRANSMITTED (keypad)",  'disp-tx-data',    border='#64ffda', value_color='#64ffda'),
                    card("RECEIVED (before FEC)",   'disp-rx-data',    border='#ffab40', value_color='#ffab40'),
                    card("RECEIVED (after FEC)",    'disp-rx-corrected', border='#00e5ff', value_color='#00e5ff'),
                    card("TYPING BUFFER",          'disp-tx-typing',  border='#ffab40', value_color='#ffab40'),
                    card("ACTIVE MODULATION",      'disp-mod',        border='#00e5ff', value_color='#00e5ff'),
                    card("CURRENT BER",            'disp-ber',        border='#ff1744', value_color='#ff1744'),
                    card("TX RAW BITS",            'disp-tx-bits',    value_color='#64ffda', font_size='18px'),
                    card("RX RAW BITS",            'disp-raw',        value_color='#ffab40', font_size='18px'),
                    card("ERROR BITS",             'disp-errors',     value_color='#ff1744', font_size='18px'),
                    card("BITS CORRECTED",         'disp-corrected',  value_color='#64ffda', font_size='18px'),
                    card("ACTIVE FEC",             'disp-fec',        value_color='#00e5ff', font_size='18px'),
                    card("TX PACKETS SENT",        'disp-tx-pkts',    value_color='#64ffda', font_size='18px'),
                    card("LINK STATUS",            'disp-status',     value_color='#64ffda', font_size='18px'),
                ]
            )
        ]),

        # Binary data section
        html.Div(
            style={
                'display': 'grid', 'gridTemplateColumns': '1fr 1fr 1fr',
                'gap': '15px', 'marginBottom': '20px', 'marginTop': '10px'
            },
            children=[
                binary_card("TX RAW BINARY", "disp-tx-binary"),
                binary_card("RX RAW BINARY (before correction)", "disp-rx-binary-before"),
                binary_card("RX BINARY (after bit correction)", "disp-rx-binary-after"),
            ]
        ),

        dcc.Graph(id='live-graph'),
        dcc.Graph(id='error-graph'),

        # Modulation pill indicators
        html.Div(
            style={'display': 'flex', 'justifyContent': 'center', 'gap': '10px', 'margin': '15px 0'},
            children=[
                html.Div(id=f'mod-ind-{m}', children=m,
                         style={'padding': '8px 24px', 'borderRadius': '6px', 'fontWeight': 'bold',
                                'fontSize': '14px', 'letterSpacing': '2px', 'border': '1px solid #333',
                                'color': '#555', 'background': '#161d24'})
                for m in ['BPSK', 'QPSK', '16QAM']
            ]
        ),

        html.Div(id='debug-out',
                 style={'marginTop': '10px', 'fontSize': '10px', 'color': '#444', 'textAlign': 'center'}),
        dcc.Interval(id='graph-update', interval=500),
    ]
)

# ---------------------------------------------------------------
# CALLBACK
# ---------------------------------------------------------------
@app.callback(
    [
        Output('live-graph',      'figure'),
        Output('error-graph',     'figure'),
        Output('disp-tx-data',    'children'),
        Output('disp-tx-bits',    'children'),
        Output('disp-tx-typing',  'children'),
        Output('disp-rx-data',    'children'),
        Output('disp-rx-corrected', 'children'),
        Output('disp-mod',        'children'),
        Output('disp-ber',        'children'),
        Output('disp-raw',        'children'),
        Output('disp-errors',     'children'),
        Output('disp-corrected',  'children'),
        Output('disp-fec',        'children'),
        Output('disp-tx-pkts',    'children'),
        Output('disp-status',     'children'),
        Output('debug-out',       'children'),
        Output('mod-ind-BPSK',    'style'),
        Output('mod-ind-QPSK',    'style'),
        Output('mod-ind-16QAM',   'style'),
        Output('conn-rx',         'children'),
        Output('conn-rx',         'style'),
        Output('conn-tx',         'children'),
        Output('conn-tx',         'style'),
        Output('disp-tx-binary',      'children'),
        Output('disp-rx-binary-before', 'children'),
        Output('disp-rx-binary-after',  'children'),
    ],
    [
        Input('graph-update', 'n_intervals'),
        Input('snr-slider',   'value'),
        Input('ch-selector',  'value'),
        Input('fec-selector', 'value'),
    ]
)
def update_ui(n, snr_val, ch_val, fec_val):
    
    # FIX: Send SNR command to BOTH TX and RX Arduinos
    if ser_rx:
        try:
            ser_rx.write(f"SNR:{snr_val}\n".encode())
            ser_rx.flush()
            print(f"[→RX] SNR:{snr_val}")
        except Exception as e:
            print(f"[RX SNR ERR] {e}")
    
    if ser_tx:
        try:
            ser_tx.write(f"SNR:{snr_val}\n".encode())
            ser_tx.flush()
            print(f"[→TX] SNR:{snr_val}")
        except Exception as e:
            print(f"[TX SNR ERR] {e}")
    
    # FIX: Send Channel command to BOTH TX and RX Arduinos
    if ser_rx:
        try:
            ser_rx.write(f"{ch_val}\n".encode())
            ser_rx.flush()
            print(f"[→RX] {ch_val}")
        except Exception as e:
            print(f"[RX CH ERR] {e}")
    
    if ser_tx:
        try:
            ser_tx.write(f"{ch_val}\n".encode())
            ser_tx.flush()
            print(f"[→TX] {ch_val}")
        except Exception as e:
            print(f"[TX CH ERR] {e}")
    
    # Send FEC commands to both Arduinos
    fec_value = fec_val.replace('FEC:', '')
    
    if ser_rx:
        try:
            ser_rx.write(f"FEC:{fec_value}\n".encode())
            ser_rx.flush()
            print(f"[→RX] FEC:{fec_value}")
        except Exception as e:
            print(f"[RX FEC ERR] {e}")
    
    if ser_tx:
        try:
            ser_tx.write(f"FEC:{fec_value}\n".encode())
            ser_tx.flush()
            print(f"[→TX] FEC:{fec_value}")
        except Exception as e:
            print(f"[TX FEC ERR] {e}")

    with lock:
        now = time.time()

        # --- BER graph ---
        ber_fig = go.Figure()
        ber_fig.add_trace(go.Scatter(
            y=list(raw_ber_history), name="Raw BER",
            line=dict(color='rgba(255,23,68,0.25)'), mode='lines'
        ))
        ber_fig.add_trace(go.Scatter(
            y=list(avg_ber_history), name="Smoothed BER",
            line=dict(color='#ff1744', width=2.5), mode='lines'
        ))
        
        # Add threshold lines
        thresholds = [
            (0.018, '#ff1744', '→ BPSK', 'top left'),
            (0.012, '#ff8a65', '← Exit BPSK', 'top right'),
            (0.006, '#ffab40', '→ QPSK', 'top left'),
            (0.003, '#fff176', '← Exit QPSK', 'top right'),
        ]
        for y, color, label, pos in thresholds:
            ber_fig.add_hline(y=y, line_dash="dot", line_color=color,
                              annotation_text=label, annotation_position=pos)
        
        ber_fig.update_layout(
            template="plotly_dark", height=320, title="Bit Error Rate (BER)",
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 0.05], title="BER"),
            xaxis=dict(title="Samples"),
            legend=dict(orientation='h', y=1.1)
        )

        # --- Error rate graph ---
        error_fig = go.Figure()
        error_fig.add_trace(go.Scatter(
            y=list(error_rate_history), name="Error Rate %",
            line=dict(color='#ffab40', width=2), mode='lines+markers',
            marker=dict(size=3)
        ))
        error_fig.update_layout(
            template="plotly_dark", height=260,
            title="Error Rate % (derived from BER / packet data)",
            margin=dict(l=20, r=20, t=40, b=20),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            yaxis=dict(range=[0, 10], title="Error %"),
            xaxis=dict(title="Samples")
        )

        # --- Modulation pill styles ---
        active_mod = stats["mod"].strip()
        def pill_style(m):
            if m == active_mod:
                c = MOD_COLORS.get(m, '#fff')
                return {
                    'padding': '8px 24px', 'borderRadius': '6px', 'fontWeight': 'bold',
                    'fontSize': '14px', 'letterSpacing': '2px',
                    'border': f'2px solid {c}', 'color': c,
                    'background': '#1c252e', 'boxShadow': f'0 0 10px {c}'
                }
            return {
                'padding': '8px 24px', 'borderRadius': '6px', 'fontWeight': 'bold',
                'fontSize': '14px', 'letterSpacing': '2px',
                'border': '1px solid #333', 'color': '#555', 'background': '#161d24'
            }

        # --- Connection health ---
        rx_age = now - stats["rx_last_seen"] if stats["rx_last_seen"] > 0 else 999
        tx_age = now - stats["tx_last_seen"] if stats["tx_last_seen"] > 0 else 999

        rx_alive = ser_rx is not None and rx_age < 3
        tx_alive = ser_tx is not None and tx_age < 5

        def conn_style(alive):
            base = {
                'flex': 1, 'padding': '8px', 'textAlign': 'center',
                'borderRadius': '6px', 'fontSize': '12px', 'fontWeight': 'bold'
            }
            if alive:
                base['border'] = '1px solid #1a7a2a'
                base['background'] = '#0d2010'
                base['color'] = '#64ffda'
            else:
                base['border'] = '1px solid #7a1a1a'
                base['background'] = '#200d0d'
                base['color'] = '#ff5555'
            return base

        rx_text = (f"RX {RX_PORT}  ●  LIVE  ({rx_age:.1f}s ago)"
                   if rx_alive else f"RX {RX_PORT}  ○  NO DATA  ({rx_age:.0f}s)")
        tx_text = (f"TX {TX_PORT}  ●  LIVE  ({tx_age:.1f}s ago)"
                   if tx_alive else f"TX {TX_PORT}  ○  NO DATA  ({tx_age:.0f}s)")

        # --- Status color ---
        status_color = "#64ffda" if stats["status"] not in ("OFFLINE", "WAITING...") else "#ff1744"

        # --- Typing buffer display ---
        typing_display = (stats["tx_typing"] + " ▌") if stats["tx_typing"] else "—"

        # --- Binary display values ---
        tx_binary_display = stats["tx_binary"] if stats["tx_binary"] else "—"
        rx_binary_before_display = stats["rx_binary_before"] if stats["rx_binary_before"] else "—"
        rx_binary_after_display = stats["rx_binary_after"] if stats["rx_binary_after"] else "—"

        debug = (f"SNR={snr_val}dB | {ch_val} | {fec_val} | "
                 f"Mod={stats['mod']} | BER={stats['ber']} | "
                 f"TX mod={stats['tx_mod']} | TX FEC={stats['tx_fec']} | "
                 f"RX age={rx_age:.1f}s  TX age={tx_age:.1f}s")

        return (
            ber_fig, error_fig,
            stats["tx_data"],
            stats["tx_raw_bits"],
            typing_display,
            stats["rx_data"],
            stats["rx_data_corrected"],
            stats["mod"],
            stats["ber"],
            stats["raw_bits"],
            stats["error_bits"],
            stats["corrected_bits"],
            stats["fec_type"],
            stats["tx_packets"],
            html.Span(stats["status"], style={'color': status_color, 'fontWeight': 'bold'}),
            debug,
            pill_style('BPSK'), pill_style('QPSK'), pill_style('16QAM'),
            rx_text, conn_style(rx_alive),
            tx_text, conn_style(tx_alive),
            tx_binary_display,
            rx_binary_before_display,
            rx_binary_after_display,
        )


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("2.GHz ADAPTIVE LINK CALIBRATION — CLOSED LOOP")
    print(f"RX port : {RX_PORT}  (Receiver with FEC decode)")
    print(f"TX port : {TX_PORT}  (Transmitter with keypad)")
    print("Dashboard : http://localhost:8050")
    print("=" * 60 + "\n")
    app.run(debug=False, port=8050)