"""
Order Book Visualisation
========================
Run from Tools/data_visualisation/:
    python orderbook.py

Reads all CSVs from ../../Files/Prices/

Charts:
  1. Depth heatmap       — bid/ask volume at each price level over time
  2. Order book imbalance — bid vs ask volume pressure over time
  3. Spread regime        — spread width + rolling volatility overlay
  4. Queue depth          — volume at best bid/ask over time
  5. Large order detector — flags abnormally large orders
"""

import os
import glob
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

# ── Config ────────────────────────────────────────────────────────────────────

PRICES_DIR       = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'Files', 'Prices')
LARGE_ORDER_ZSCORE = 2.0   # flag orders this many std devs above mean volume
ROLLING_VOL_WIN    = 20

COLORS = {
    'bg':      '#0d1117',
    'panel':   '#161b22',
    'border':  '#30363d',
    'text':    '#e6edf3',
    'subtext': '#8b949e',
    'green':   '#3fb950',
    'red':     '#f85149',
    'blue':    '#58a6ff',
    'orange':  '#d29922',
    'purple':  '#bc8cff',
    'teal':    '#39d0d8',
    'yellow':  '#e3b341',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['panel'],
    font=dict(color=COLORS['text'], family='monospace'),
    xaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    yaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode='x unified',
)

# ── Data loading ──────────────────────────────────────────────────────────────

def discover_files():
    return sorted(glob.glob(os.path.join(PRICES_DIR, '*.csv')))


def load_selected(filenames):
    frames = []
    for name in filenames:
        path = os.path.join(PRICES_DIR, name)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
            df['source_file'] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values('timestamp').reset_index(drop=True)


def make_empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── Derived metrics ───────────────────────────────────────────────────────────

def compute_metrics(df):
    df = df.copy()

    # Spread
    df['spread'] = df['ask_price_1'] - df['bid_price_1']

    # Total bid / ask volume across all 3 levels
    bid_cols = ['bid_volume_1', 'bid_volume_2', 'bid_volume_3']
    ask_cols = ['ask_volume_1', 'ask_volume_2', 'ask_volume_3']
    df['total_bid_vol'] = df[bid_cols].fillna(0).sum(axis=1)
    df['total_ask_vol'] = df[ask_cols].fillna(0).sum(axis=1)

    # Imbalance: ranges from -1 (all ask) to +1 (all bid)
    total = df['total_bid_vol'] + df['total_ask_vol']
    df['imbalance'] = np.where(
        total > 0,
        (df['total_bid_vol'] - df['total_ask_vol']) / total,
        0.0
    )

    # Rolling volatility on mid price
    df['mid_return'] = df['mid_price'].pct_change()
    df['rolling_vol'] = df['mid_return'].rolling(ROLLING_VOL_WIN).std() * np.sqrt(ROLLING_VOL_WIN)

    # Queue depth at best bid/ask
    df['best_bid_vol'] = df['bid_volume_1'].fillna(0)
    df['best_ask_vol'] = df['ask_volume_1'].fillna(0)

    # Large order detection — z-score on total volume each side
    for col in ['total_bid_vol', 'total_ask_vol']:
        mean = df[col].mean()
        std  = df[col].std()
        df[f'{col}_z'] = (df[col] - mean) / std if std > 0 else 0.0

    df['large_bid'] = df['total_bid_vol_z'] >= LARGE_ORDER_ZSCORE
    df['large_ask'] = df['total_ask_vol_z'] >= LARGE_ORDER_ZSCORE

    df['tick'] = range(len(df))
    return df


def build_depth_heatmap(df):
    """
    Build a heatmap of bid/ask volume at each price level over time.
    Bids shown in green (positive), asks in red (negative), stacked by level.
    Uses a sample of ticks if data is large to keep rendering fast.
    """
    # Sample down if too many ticks
    max_ticks = 500
    if len(df) > max_ticks:
        step = len(df) // max_ticks
        df   = df.iloc[::step].reset_index(drop=True)

    ticks      = df['tick']
    mid_prices = df['mid_price']

    # Collect all unique price levels present
    price_levels = set()
    for i in range(1, 4):
        price_levels.update(df[f'bid_price_{i}'].dropna().unique())
        price_levels.update(df[f'ask_price_{i}'].dropna().unique())
    price_levels = sorted(price_levels)

    if not price_levels:
        return make_empty_fig("No price level data available")

    # Build volume matrix: rows=price levels, cols=ticks
    # Bids positive, asks negative
    vol_matrix = np.zeros((len(price_levels), len(df)))
    price_idx  = {p: i for i, p in enumerate(price_levels)}

    for col_idx, (_, row) in enumerate(df.iterrows()):
        for level in range(1, 4):
            bp = row.get(f'bid_price_{level}')
            bv = row.get(f'bid_volume_{level}', 0)
            ap = row.get(f'ask_price_{level}')
            av = row.get(f'ask_volume_{level}', 0)
            if pd.notna(bp) and bp in price_idx:
                vol_matrix[price_idx[bp], col_idx] += float(bv or 0)
            if pd.notna(ap) and ap in price_idx:
                vol_matrix[price_idx[ap], col_idx] -= float(av or 0)

    fig = go.Figure()

    # Heatmap layer
    fig.add_trace(go.Heatmap(
        z=vol_matrix,
        x=ticks.tolist(),
        y=[str(p) for p in price_levels],
        colorscale=[
            [0.0,  COLORS['red']],
            [0.45, '#1a1a2e'],
            [0.5,  COLORS['panel']],
            [0.55, '#1a2e1a'],
            [1.0,  COLORS['green']],
        ],
        zmid=0,
        hovertemplate='Tick: %{x}<br>Price: %{y}<br>Volume: %{z:.0f}<extra></extra>',
        showscale=True,
        colorbar=dict(
            title=dict(text='Vol', font=dict(color=COLORS['subtext'], size=10)),
            tickfont=dict(color=COLORS['subtext'], size=10),
        ),
    ))

    # Mid price line overlay
    fig.add_trace(go.Scatter(
        x=ticks.tolist(),
        y=[str(round(p)) for p in mid_prices],
        mode='lines',
        name='Mid Price',
        line=dict(color=COLORS['yellow'], width=1.5),
    ))

    fig.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
        xaxis=dict(gridcolor=COLORS['border'], title='Tick'),
        yaxis=dict(gridcolor=COLORS['border'], title='Price Level',
                   type='category', categoryorder='array',
                   categoryarray=[str(p) for p in price_levels]),
        legend=dict(font=dict(color=COLORS['text'])),
    )
    return fig


# ── Startup ───────────────────────────────────────────────────────────────────

try:
    all_files = [os.path.basename(f) for f in discover_files()]
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {PRICES_DIR}")
    _sample    = load_selected(all_files)
    products   = sorted(_sample['product'].unique().tolist()) if not _sample.empty else []
    LOAD_ERROR = None
except Exception as e:
    all_files, products = [], []
    LOAD_ERROR = str(e)

# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Order Book Visualiser"

file_options    = [{'label': 'All Files', 'value': '__all__'}] + \
                  [{'label': f, 'value': f} for f in all_files]
product_options = [{'label': p, 'value': p} for p in products]

app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': 'monospace',
    'color': COLORS['text'],
    'padding': '24px',
}, children=[

    # Header
    html.Div(style={'marginBottom': '24px'}, children=[
        html.H1("ORDER BOOK VISUALISER", style={
            'color': COLORS['teal'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · Depth · Imbalance · Spread Regime · Queue · Large Orders", style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'margin': 0,
            'letterSpacing': '2px',
        }),
    ]),

    # Controls
    html.Div(style={'display': 'flex', 'gap': '16px', 'marginBottom': '24px', 'flexWrap': 'wrap'},
    children=[
        html.Div(children=[
            html.Label("PRODUCT", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                          'letterSpacing': '2px', 'display': 'block', 'marginBottom': '6px'}),
            dcc.Dropdown(
                id='product-selector',
                options=product_options,
                value=products[0] if products else None,
                clearable=False,
                style={'width': '200px', 'backgroundColor': COLORS['panel'],
                       'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}"},
            ),
        ]),
        html.Div(children=[
            html.Label("FILE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                       'letterSpacing': '2px', 'display': 'block', 'marginBottom': '6px'}),
            dcc.Dropdown(
                id='file-selector',
                options=file_options,
                value='__all__',
                clearable=False,
                style={'width': '360px', 'backgroundColor': COLORS['panel'],
                       'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}"},
            ),
        ]),
    ]),

    html.Div(id='error-banner', style={'color': COLORS['red'], 'marginBottom': '16px', 'fontSize': '13px'}),

    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[

        # Depth heatmap
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("DEPTH HEATMAP  —  green = bid volume · red = ask volume · yellow = mid price",
                   style={'color': COLORS['subtext'], 'fontSize': '11px',
                           'letterSpacing': '2px', 'margin': '0 0 4px 0'}),
            html.P("Shows support/resistance levels and where large orders sit. "
                   "Persistent green bands = strong bid support. Persistent red bands = ask resistance.",
                   style={'color': COLORS['subtext'], 'fontSize': '11px', 'margin': '0 0 8px 0',
                           'opacity': '0.6'}),
            dcc.Graph(id='depth-chart', config={'displayModeBar': True}, style={'height': '400px'}),
        ]),

        # Imbalance
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("ORDER BOOK IMBALANCE  —  +1 = all bids · -1 = all asks",
                   style={'color': COLORS['subtext'], 'fontSize': '11px',
                           'letterSpacing': '2px', 'margin': '0 0 4px 0'}),
            html.P("Sustained positive imbalance = buying pressure, price likely to rise. "
                   "Use this as a directional skew signal.",
                   style={'color': COLORS['subtext'], 'fontSize': '11px', 'margin': '0 0 8px 0',
                           'opacity': '0.6'}),
            dcc.Graph(id='imbalance-chart', config={'displayModeBar': False}, style={'height': '240px'}),
        ]),

        # Spread regime + vol
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("SPREAD REGIME  —  spread width vs rolling volatility",
                   style={'color': COLORS['subtext'], 'fontSize': '11px',
                           'letterSpacing': '2px', 'margin': '0 0 4px 0'}),
            html.P("Wide spread + high vol = dangerous to quote tight. "
                   "Narrow spread + low vol = safe to quote inside.",
                   style={'color': COLORS['subtext'], 'fontSize': '11px', 'margin': '0 0 8px 0',
                           'opacity': '0.6'}),
            dcc.Graph(id='regime-chart', config={'displayModeBar': False}, style={'height': '260px'}),
        ]),

        # Queue depth side by side with large order detector
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[

            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("QUEUE DEPTH AT BEST BID / ASK",
                       style={'color': COLORS['subtext'], 'fontSize': '11px',
                               'letterSpacing': '2px', 'margin': '0 0 4px 0'}),
                html.P("Large volume at best bid/ask = you'll be at the back of the queue if you join. "
                       "Consider quoting one tick inside.",
                       style={'color': COLORS['subtext'], 'fontSize': '11px', 'margin': '0 0 8px 0',
                               'opacity': '0.6'}),
                dcc.Graph(id='queue-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),

            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P(f"LARGE ORDER DETECTOR  —  flagged at >{LARGE_ORDER_ZSCORE}σ above mean volume",
                       style={'color': COLORS['subtext'], 'fontSize': '11px',
                               'letterSpacing': '2px', 'margin': '0 0 4px 0'}),
                html.P("Large orders act as support/resistance. "
                       "Price often bounces rather than trading through them.",
                       style={'color': COLORS['subtext'], 'fontSize': '11px', 'margin': '0 0 8px 0',
                               'opacity': '0.6'}),
                dcc.Graph(id='large-order-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
        ]),
    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('depth-chart',       'figure'),
    Output('imbalance-chart',   'figure'),
    Output('regime-chart',      'figure'),
    Output('queue-chart',       'figure'),
    Output('large-order-chart', 'figure'),
    Output('error-banner',      'children'),
    Input('product-selector',   'value'),
    Input('file-selector',      'value'),
)
def update_charts(product, selected_file):
    if LOAD_ERROR:
        e = make_empty_fig(LOAD_ERROR)
        return e, e, e, e, e, f"Error: {LOAD_ERROR}"

    files = all_files if selected_file == '__all__' else [selected_file]
    raw   = load_selected(files)
    df    = raw[raw['product'] == product].copy() if not raw.empty else pd.DataFrame()

    if df.empty:
        e = make_empty_fig("No data for selection")
        return e, e, e, e, e, ""

    df = compute_metrics(df)
    t  = df['tick']

    # ── 1. Depth heatmap ──────────────────────────────────────────────────────
    fig_depth = build_depth_heatmap(df)

    # ── 2. Imbalance ──────────────────────────────────────────────────────────
    fig_imb = go.Figure()
    fig_imb.add_trace(go.Scatter(
        x=t, y=df['imbalance'],
        mode='lines', name='Imbalance',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(63,185,80,0.15)',
    ))
    # Colour above/below zero separately
    fig_imb.add_trace(go.Scatter(
        x=t, y=df['imbalance'].clip(upper=0),
        mode='lines', name='Ask pressure',
        line=dict(width=0),
        fill='tozeroy',
        fillcolor='rgba(248,81,73,0.15)',
        showlegend=True,
    ))
    fig_imb.add_trace(go.Scatter(
        x=t, y=df['imbalance'],
        mode='lines', name='Imbalance',
        line=dict(color=COLORS['blue'], width=0.8),
        showlegend=False,
    ))
    fig_imb.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
    fig_imb.update_layout(**PLOTLY_LAYOUT)
    fig_imb.update_yaxes(title_text='Imbalance', range=[-1.1, 1.1])
    fig_imb.update_xaxes(title_text='Tick')

    # ── 3. Spread regime ──────────────────────────────────────────────────────
    fig_reg = make_subplots(specs=[[{"secondary_y": True}]])
    fig_reg.add_trace(go.Scatter(
        x=t, y=df['spread'],
        mode='lines', name='Spread',
        line=dict(color=COLORS['orange'], width=1.0),
        fill='tozeroy', fillcolor='rgba(210,153,34,0.08)',
    ), secondary_y=False)
    fig_reg.add_trace(go.Scatter(
        x=t, y=df['rolling_vol'],
        mode='lines', name='Rolling Vol',
        line=dict(color=COLORS['purple'], width=1.0, dash='dot'),
    ), secondary_y=True)
    fig_reg.update_layout(
        **{k: v for k, v in PLOTLY_LAYOUT.items() if k not in ('xaxis', 'yaxis')},
        xaxis=dict(gridcolor=COLORS['border'], title='Tick'),
        yaxis=dict(gridcolor=COLORS['border'], title='Spread'),
        yaxis2=dict(title='Rolling Vol', overlaying='y', side='right',
                    gridcolor=COLORS['border'], showgrid=False),
        legend=dict(font=dict(color=COLORS['text'])),
    )

    # ── 4. Queue depth ────────────────────────────────────────────────────────
    fig_queue = go.Figure()
    fig_queue.add_trace(go.Scatter(
        x=t, y=df['best_bid_vol'],
        mode='lines', name='Best Bid Vol',
        line=dict(color=COLORS['green'], width=1.0),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.1)',
    ))
    fig_queue.add_trace(go.Scatter(
        x=t, y=-df['best_ask_vol'],
        mode='lines', name='Best Ask Vol',
        line=dict(color=COLORS['red'], width=1.0),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.1)',
    ))
    fig_queue.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
    fig_queue.update_layout(**PLOTLY_LAYOUT)
    fig_queue.update_yaxes(title_text='Volume (bids +ve, asks -ve)')
    fig_queue.update_xaxes(title_text='Tick')

    # ── 5. Large order detector ───────────────────────────────────────────────
    fig_large = go.Figure()

    # Background volume bars
    fig_large.add_trace(go.Bar(
        x=t, y=df['total_bid_vol'],
        name='Bid Vol', marker_color=COLORS['green'], opacity=0.3,
    ))
    fig_large.add_trace(go.Bar(
        x=t, y=-df['total_ask_vol'],
        name='Ask Vol', marker_color=COLORS['red'], opacity=0.3,
    ))

    # Flag large bid orders
    large_bids = df[df['large_bid']]
    if not large_bids.empty:
        fig_large.add_trace(go.Scatter(
            x=large_bids['tick'], y=large_bids['total_bid_vol'],
            mode='markers', name='Large Bid',
            marker=dict(color=COLORS['green'], size=10, symbol='triangle-up',
                        line=dict(color='white', width=1)),
        ))

    # Flag large ask orders
    large_asks = df[df['large_ask']]
    if not large_asks.empty:
        fig_large.add_trace(go.Scatter(
            x=large_asks['tick'], y=-large_asks['total_ask_vol'],
            mode='markers', name='Large Ask',
            marker=dict(color=COLORS['red'], size=10, symbol='triangle-down',
                        line=dict(color='white', width=1)),
        ))

    fig_large.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
    fig_large.update_layout(**PLOTLY_LAYOUT, barmode='overlay')
    fig_large.update_yaxes(title_text='Volume')
    fig_large.update_xaxes(title_text='Tick')

    return fig_depth, fig_imb, fig_reg, fig_queue, fig_large, ""


if __name__ == '__main__':
    print("Starting Order Book Visualiser → http://127.0.0.1:8054")
    app.run(debug=True, port=8054)