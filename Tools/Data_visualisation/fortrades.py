"""
Trade Data Visualisation
========================
Run from Tools/data_visualisation/:
    python fortrades.py

Reads all CSVs from ../../Files/Trades/
"""

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import dash
from dash import dcc, html, Input, Output

# ── Config ────────────────────────────────────────────────────────────────────

TRADES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Files', 'Trades')
PRICES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Files', 'Prices')
ROLLING_WINDOW = 20

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

def discover_files(directory):
    return sorted(glob.glob(os.path.join(directory, '*.csv')))


def load_selected(filenames, directory):
    frames = []
    for name in filenames:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            df = pd.read_csv(path, sep=';')
            df['source_file'] = name
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def prepare(df):
    if df.empty:
        return df
    df = df.copy()

    if 'buyer' in df.columns and 'seller' in df.columns:
        df['direction'] = df.apply(
            lambda r: 'BUY'  if str(r['buyer']).strip().upper()  == 'SUBMISSION'
            else     ('SELL' if str(r['seller']).strip().upper() == 'SUBMISSION'
            else      'MARKET'),
            axis=1
        )
    else:
        df['direction'] = 'MARKET'

    df = df.sort_values('timestamp').reset_index(drop=True)
    df['tick'] = df.groupby('symbol').cumcount()

    df['rolling_avg_size'] = (
        df.groupby('symbol')['quantity']
          .transform(lambda x: x.rolling(ROLLING_WINDOW, min_periods=1).mean())
    )
    df['cumulative_volume'] = df.groupby('symbol')['quantity'].cumsum()

    return df


def make_empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ── Discover available files ──────────────────────────────────────────────────

try:
    trade_files = [os.path.basename(f) for f in discover_files(TRADES_DIR)]
    price_files = [os.path.basename(f) for f in discover_files(PRICES_DIR)]

    if not trade_files:
        raise FileNotFoundError(f"No CSV files found in {TRADES_DIR}")

    _all = prepare(load_selected(trade_files, TRADES_DIR))
    products = sorted(_all['symbol'].unique().tolist()) if not _all.empty else []
    LOAD_ERROR = None

except Exception as e:
    trade_files, price_files, products = [], [], []
    LOAD_ERROR = str(e)

# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__)
app.title = "Trade Visualiser"

file_options    = [{'label': 'All Files', 'value': '__all__'}] + \
                  [{'label': f, 'value': f} for f in trade_files]
product_options = [{'label': p, 'value': p} for p in products]

app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': 'monospace',
    'color': COLORS['text'],
    'padding': '24px',
}, children=[

    html.Div(style={'marginBottom': '24px'}, children=[
        html.H1("TRADE DATA VISUALISER", style={
            'color': COLORS['teal'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · Trade Analysis", style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'margin': 0,
            'letterSpacing': '2px',
        }),
    ]),

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

        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("TRADE PRICE vs MID PRICE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                        'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='price-chart', config={'displayModeBar': False}, style={'height': '300px'}),
        ]),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("TRADE VOLUME OVER TIME", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                          'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='volume-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P(f"ROLLING AVG TRADE SIZE (window={ROLLING_WINDOW})",
                       style={'color': COLORS['subtext'], 'fontSize': '11px',
                               'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='avg-size-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
        ]),

        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("BUY vs SELL VOLUME", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                      'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='buysell-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("TRADE FREQUENCY PER TIMESTAMP", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                                  'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='freq-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
        ]),

        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("CUMULATIVE VOLUME", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='cumvol-chart', config={'displayModeBar': False}, style={'height': '240px'}),
        ]),

    ]),
])

# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('price-chart',    'figure'),
    Output('volume-chart',   'figure'),
    Output('avg-size-chart', 'figure'),
    Output('buysell-chart',  'figure'),
    Output('freq-chart',     'figure'),
    Output('cumvol-chart',   'figure'),
    Output('error-banner',   'children'),
    Input('product-selector', 'value'),
    Input('file-selector',    'value'),
)
def update_charts(product, selected_file):
    if LOAD_ERROR:
        empty = make_empty_fig(LOAD_ERROR)
        return empty, empty, empty, empty, empty, empty, f"Error: {LOAD_ERROR}"

    t_files = trade_files if selected_file == '__all__' else [selected_file]
    p_files = price_files if selected_file == '__all__' else [
        pf for pf in price_files
        if selected_file.replace('trades_', 'prices_') == pf
    ] or price_files

    raw = prepare(load_selected(t_files, TRADES_DIR))
    df  = raw[raw['symbol'] == product].copy() if not raw.empty else pd.DataFrame()

    if df.empty:
        empty = make_empty_fig("No data for selection")
        return empty, empty, empty, empty, empty, empty, ""

    tick = df['tick']

    # ── Trade price vs mid price ──────────────────────────────────────────────
    fig_price = go.Figure()

    price_raw = load_selected(p_files, PRICES_DIR)
    if not price_raw.empty and 'product' in price_raw.columns:
        pdf = price_raw[price_raw['product'] == product].sort_values('timestamp').reset_index(drop=True)
        pdf['tick'] = range(len(pdf))
        if not pdf.empty:
            fig_price.add_trace(go.Scatter(
                x=pdf['tick'], y=pdf['mid_price'],
                mode='lines', name='Mid Price',
                line=dict(color=COLORS['blue'], width=1.0, dash='dot'),
                opacity=0.6,
            ))

    for direction, color, symbol in [
        ('BUY',    COLORS['green'],   'triangle-up'),
        ('SELL',   COLORS['red'],     'triangle-down'),
        ('MARKET', COLORS['subtext'], 'circle'),
    ]:
        subset = df[df['direction'] == direction]
        if not subset.empty:
            fig_price.add_trace(go.Scatter(
                x=subset['tick'], y=subset['price'], mode='markers', name=direction,
                marker=dict(color=color, size=5, symbol=symbol),
            ))

    fig_price.update_layout(**PLOTLY_LAYOUT)
    fig_price.update_yaxes(title_text='Price')
    fig_price.update_xaxes(title_text='Tick')

    # ── Volume over time ──────────────────────────────────────────────────────
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Bar(x=tick, y=df['quantity'], name='Volume',
                             marker_color=COLORS['teal'], opacity=0.7))
    fig_vol.update_layout(**PLOTLY_LAYOUT, bargap=0.1)
    fig_vol.update_yaxes(title_text='Quantity')
    fig_vol.update_xaxes(title_text='Tick')

    # ── Rolling avg trade size ────────────────────────────────────────────────
    fig_avg = go.Figure()
    fig_avg.add_trace(go.Scatter(x=tick, y=df['rolling_avg_size'], mode='lines',
                                  name='Avg Size', line=dict(color=COLORS['orange'], width=1.2),
                                  fill='tozeroy', fillcolor='rgba(210,153,34,0.08)'))
    fig_avg.update_layout(**PLOTLY_LAYOUT)
    fig_avg.update_yaxes(title_text='Avg Trade Size')
    fig_avg.update_xaxes(title_text='Tick')

    # ── Buy vs sell volume ────────────────────────────────────────────────────
    fig_bs = go.Figure()
    fig_bs.add_trace(go.Bar(
        x=['BUY', 'SELL', 'MARKET'],
        y=[df[df['direction'] == d]['quantity'].sum() for d in ['BUY', 'SELL', 'MARKET']],
        marker_color=[COLORS['green'], COLORS['red'], COLORS['subtext']],
    ))
    fig_bs.update_layout(**PLOTLY_LAYOUT, showlegend=False)
    fig_bs.update_yaxes(title_text='Total Volume')

    # ── Trade frequency ───────────────────────────────────────────────────────
    freq = df.groupby('timestamp').size().reset_index(name='count')
    fig_freq = go.Figure()
    fig_freq.add_trace(go.Bar(x=freq['timestamp'], y=freq['count'],
                               name='Trades/Tick', marker_color=COLORS['purple'], opacity=0.8))
    fig_freq.update_layout(**PLOTLY_LAYOUT, bargap=0.05)
    fig_freq.update_yaxes(title_text='# Trades')
    fig_freq.update_xaxes(title_text='Timestamp')

    # ── Cumulative volume ─────────────────────────────────────────────────────
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(x=tick, y=df['cumulative_volume'], mode='lines',
                                  name='Cumulative Volume',
                                  line=dict(color=COLORS['blue'], width=1.2),
                                  fill='tozeroy', fillcolor='rgba(88,166,255,0.06)'))
    fig_cum.update_layout(**PLOTLY_LAYOUT)
    fig_cum.update_yaxes(title_text='Cumulative Volume')
    fig_cum.update_xaxes(title_text='Tick')

    return fig_price, fig_vol, fig_avg, fig_bs, fig_freq, fig_cum, ""


if __name__ == '__main__':
    print("Starting Trade Visualiser → http://127.0.0.1:8051")
    app.run(debug=True, port=8051)