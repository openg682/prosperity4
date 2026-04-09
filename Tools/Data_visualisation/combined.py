"""
Combined Analysis Visualisation
================================
Run from Tools/data_visualisation/:
    python combined.py

Reads:
  - ../../Files/Prices/  → price CSVs
  - ../../Files/Trades/  → trade CSVs
  - ../backtest/backtest_output.csv → backtester output (run backtester first)
"""

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output

# ── Config ────────────────────────────────────────────────────────────────────

PRICES_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'Files', 'Prices')
TRADES_DIR    = os.path.join(os.path.dirname(__file__), '..', '..', 'Files', 'Trades')
BACKTEST_FILE = os.path.join(os.path.dirname(__file__), '..', 'backtest', 'backtest_output.csv')

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

PRODUCT_COLORS = ['#58a6ff', '#d29922', '#3fb950', '#bc8cff', '#39d0d8']

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

def load_all_prices():
    files = sorted(glob.glob(os.path.join(PRICES_DIR, '*.csv')))
    if not files:
        return pd.DataFrame(), []
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=';')
        df['source_file'] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(['day', 'timestamp']).reset_index(drop=True), \
           [os.path.basename(f) for f in files]


def load_all_trades():
    files = sorted(glob.glob(os.path.join(TRADES_DIR, '*.csv')))
    if not files:
        return pd.DataFrame(), []
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=';')
        df['source_file'] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values('timestamp').reset_index(drop=True), \
           [os.path.basename(f) for f in files]


def load_backtest():
    if not os.path.exists(BACKTEST_FILE):
        return None
    return pd.read_csv(BACKTEST_FILE)


def make_empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ── Load everything at startup ────────────────────────────────────────────────

try:
    price_df, price_files = load_all_prices()
    trade_df, trade_files = load_all_trades()
    bt_df = load_backtest()

    products = sorted(price_df['product'].unique().tolist()) if not price_df.empty else []

    # Add spread to price data
    if not price_df.empty:
        price_df['spread'] = price_df['ask_price_1'] - price_df['bid_price_1']

    # Add trade frequency per timestamp per symbol to trade data
    if not trade_df.empty:
        trade_df['trade_freq'] = trade_df.groupby(
            ['source_file', 'timestamp', 'symbol']
        )['quantity'].transform('count')

    LOAD_ERROR = None

except Exception as e:
    price_df, trade_df, bt_df, products, price_files, trade_files = \
        pd.DataFrame(), pd.DataFrame(), None, [], [], []
    LOAD_ERROR = str(e)

# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "Combined Analysis"

file_options = [{'label': 'All Files', 'value': '__all__'}] + \
               [{'label': f, 'value': f} for f in price_files]

app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': 'monospace',
    'color': COLORS['text'],
    'padding': '24px',
}, children=[

    # Header
    html.Div(style={'marginBottom': '24px'}, children=[
        html.H1("COMBINED ANALYSIS", style={
            'color': COLORS['yellow'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · PnL · Positions · Spread vs Frequency", style={
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

    # Backtest required notice
    html.Div(id='bt-notice', style={'color': COLORS['orange'], 'marginBottom': '16px', 'fontSize': '12px',
                                     'letterSpacing': '1px'}),

    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[

        # PnL over time
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("PnL OVER TIME", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                            'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='pnl-chart', config={'displayModeBar': False}, style={'height': '280px'}),
        ]),

        # Position history per product
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("POSITION HISTORY vs PRICE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                         'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='position-chart', config={'displayModeBar': False}, style={'height': '360px'}),
        ]),

        # PnL attribution
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("PnL ATTRIBUTION PER PRODUCT", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                           'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='attribution-chart', config={'displayModeBar': False}, style={'height': '260px'}),
        ]),

        # Spread vs trade frequency
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("SPREAD WIDTH vs TRADE FREQUENCY", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                               'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='spread-freq-chart', config={'displayModeBar': False}, style={'height': '300px'}),
        ]),

    ]),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('pnl-chart',         'figure'),
    Output('position-chart',    'figure'),
    Output('attribution-chart', 'figure'),
    Output('spread-freq-chart', 'figure'),
    Output('error-banner',      'children'),
    Output('bt-notice',         'children'),
    Input('file-selector', 'value'),
)
def update_charts(selected_file):
    if LOAD_ERROR:
        empty = make_empty_fig(LOAD_ERROR)
        return empty, empty, empty, empty, f"Error: {LOAD_ERROR}", ""

    # Filter price/trade data by file if needed
    if selected_file == '__all__':
        p_df = price_df.copy()
        t_df = trade_df.copy()
    else:
        p_df = price_df[price_df['source_file'] == selected_file].copy()
        trade_file = selected_file.replace('prices_', 'trades_')
        t_df = trade_df[trade_df['source_file'] == trade_file].copy() \
               if trade_file in trade_df['source_file'].values else trade_df.copy()

    bt_notice = "" if bt_df is not None else \
        "⚠  No backtest_output.csv found — run backtester.py first to see PnL and position charts."

    # ── PnL over time ─────────────────────────────────────────────────────────
    if bt_df is not None:
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=bt_df['tick'], y=bt_df['pnl'],
            mode='lines', name='PnL',
            line=dict(color=COLORS['green'], width=1.2),
            fill='tozeroy', fillcolor='rgba(63,185,80,0.07)',
        ))
        fig_pnl.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
        fig_pnl.update_layout(**PLOTLY_LAYOUT)
        fig_pnl.update_yaxes(title_text='PnL')
        fig_pnl.update_xaxes(title_text='Tick')
    else:
        fig_pnl = make_empty_fig("Run backtester.py to generate PnL data")

    # ── Position history vs price ─────────────────────────────────────────────
    if bt_df is not None and not p_df.empty:
        pos_cols = [c for c in bt_df.columns if c.startswith('position_')]
        n = len(pos_cols)
        fig_pos = make_subplots(
            rows=n, cols=1,
            shared_xaxes=True,
            subplot_titles=[c.replace('position_', '') for c in pos_cols],
            vertical_spacing=0.08,
        )
        for idx, col in enumerate(pos_cols):
            product = col.replace('position_', '')
            color   = PRODUCT_COLORS[idx % len(PRODUCT_COLORS)]
            row     = idx + 1

            # Position line
            fig_pos.add_trace(go.Scatter(
                x=bt_df['tick'], y=bt_df[col],
                mode='lines', name=f'{product} pos',
                line=dict(color=color, width=1.0),
                fill='tozeroy', fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)',
            ), row=row, col=1)

            # Mid price on secondary y
            prod_price = p_df[p_df['product'] == product].copy()
            prod_price = prod_price.reset_index(drop=True)
            prod_price['tick'] = range(len(prod_price))
            if not prod_price.empty:
                fig_pos.add_trace(go.Scatter(
                    x=prod_price['tick'], y=prod_price['mid_price'],
                    mode='lines', name=f'{product} mid',
                    line=dict(color=COLORS['subtext'], width=0.8, dash='dot'),
                    yaxis=f'y{(idx*2)+2}',
                    opacity=0.5,
                ), row=row, col=1)

        fig_pos.update_layout(
            **{k: v for k, v in PLOTLY_LAYOUT.items() if k != 'xaxis' and k != 'yaxis'},
            height=180 * n,
            showlegend=True,
        )
        fig_pos.update_xaxes(gridcolor=COLORS['border'])
        fig_pos.update_yaxes(gridcolor=COLORS['border'])
    else:
        fig_pos = make_empty_fig("Run backtester.py to generate position data")

    # ── PnL attribution ───────────────────────────────────────────────────────
    if bt_df is not None:
        pos_cols = [c for c in bt_df.columns if c.startswith('position_')]
        # Approximate per-product PnL as fraction of total based on position exposure
        total_exposure = bt_df[[c for c in pos_cols]].abs().sum().sum()
        fig_attr = go.Figure()
        for idx, col in enumerate(pos_cols):
            product = col.replace('position_', '')
            color   = PRODUCT_COLORS[idx % len(PRODUCT_COLORS)]

            if not p_df.empty and product in p_df['product'].values:
                prod_p = p_df[p_df['product'] == product].reset_index(drop=True)
                # Align tick lengths
                n_ticks = min(len(bt_df), len(prod_p))
                pos_series   = bt_df[col].values[:n_ticks]
                price_series = prod_p['mid_price'].values[:n_ticks]
                price_diff   = np.diff(price_series, prepend=price_series[0])
                product_pnl  = np.cumsum(pos_series * price_diff)
            else:
                n_ticks     = len(bt_df)
                product_pnl = np.zeros(n_ticks)

            fig_attr.add_trace(go.Scatter(
                x=np.arange(n_ticks), y=product_pnl,
                mode='lines', name=product,
                line=dict(color=color, width=1.2),
                fill='tozeroy',
                fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.07)',
            ))

        fig_attr.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
        fig_attr.update_layout(**PLOTLY_LAYOUT)
        fig_attr.update_yaxes(title_text='Cumulative PnL')
        fig_attr.update_xaxes(title_text='Tick')
    else:
        fig_attr = make_empty_fig("Run backtester.py to generate attribution data")

    # ── Spread vs trade frequency ─────────────────────────────────────────────
    fig_sf = go.Figure()
    if not p_df.empty and not t_df.empty:
        for idx, product in enumerate(products):
            color    = PRODUCT_COLORS[idx % len(PRODUCT_COLORS)]
            pp       = p_df[p_df['product'] == product][['timestamp', 'spread']].dropna()
            tp       = t_df[t_df['symbol'] == product].groupby('timestamp').size().reset_index(name='freq')
            merged   = pd.merge(pp, tp, on='timestamp', how='inner')
            if merged.empty:
                continue
            fig_sf.add_trace(go.Scatter(
                x=merged['spread'], y=merged['freq'],
                mode='markers', name=product,
                marker=dict(color=color, size=4, opacity=0.5),
            ))
        fig_sf.update_layout(**PLOTLY_LAYOUT, showlegend=True)
        fig_sf.update_xaxes(title_text='Spread Width')
        fig_sf.update_yaxes(title_text='Trades per Timestamp')
    else:
        fig_sf = make_empty_fig("No price or trade data available")

    return fig_pnl, fig_pos, fig_attr, fig_sf, "", bt_notice


if __name__ == '__main__':
    print("Starting Combined Analysis → http://127.0.0.1:8052")
    app.run(debug=True, port=8052)