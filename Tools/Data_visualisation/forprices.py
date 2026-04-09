"""
Price Data Visualisation
========================
Run from Tools/data_visualisation/:
    python forprices.py

Reads all CSVs from ../../Files/Prices/
"""

import os
import glob
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, callback

# ── Config ────────────────────────────────────────────────────────────────────

PRICES_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'Files', 'Prices')
ROLLING_VOL_WINDOW = 20

COLORS = {
    'bg':        '#0d1117',
    'panel':     '#161b22',
    'border':    '#30363d',
    'text':      '#e6edf3',
    'subtext':   '#8b949e',
    'green':     '#3fb950',
    'red':       '#f85149',
    'blue':      '#58a6ff',
    'orange':    '#d29922',
    'purple':    '#bc8cff',
}

# ── Data loading ──────────────────────────────────────────────────────────────

def load_prices():
    files = sorted(glob.glob(os.path.join(PRICES_DIR, '*.csv')))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {PRICES_DIR}")
    frames = []
    for f in files:
        df = pd.read_csv(f, sep=';')
        df['source_file'] = os.path.basename(f)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(['day', 'timestamp']).reset_index(drop=True)
    combined['tick'] = combined.groupby('product').cumcount()
    return combined, [os.path.basename(f) for f in files]


def compute_spread(df):
    df = df.copy()
    df['spread'] = df['ask_price_1'] - df['bid_price_1']
    return df


def compute_rolling_vol(df, window=ROLLING_VOL_WINDOW):
    df = df.copy()
    df['mid_returns'] = df.groupby('product')['mid_price'].pct_change()
    df['rolling_vol'] = df.groupby('product')['mid_returns'].transform(
        lambda x: x.rolling(window).std() * np.sqrt(window)
    )
    return df


def prepare(df):
    df = compute_spread(df)
    df = compute_rolling_vol(df)
    return df

# ── Layout helpers ────────────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['panel'],
    font=dict(color=COLORS['text'], family='monospace'),
    xaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    yaxis=dict(gridcolor=COLORS['border'], zerolinecolor=COLORS['border']),
    margin=dict(l=60, r=20, t=40, b=40),
    hovermode='x unified',
)

def make_empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

# ── App ───────────────────────────────────────────────────────────────────────

try:
    raw_df, file_list = load_prices()
    raw_df = prepare(raw_df)
    products = sorted(raw_df['product'].unique().tolist())
    days = sorted(raw_df['day'].unique().tolist())
    LOAD_ERROR = None
except Exception as e:
    raw_df, file_list, products, days = None, [], [], []
    LOAD_ERROR = str(e)

app = dash.Dash(__name__)
app.title = "Price Visualiser"

day_options = [{'label': 'All Days', 'value': 'all'}] + \
              [{'label': f'Day {d}', 'value': d} for d in days]

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
        html.H1("PRICE DATA VISUALISER", style={
            'color': COLORS['green'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · Price Analysis", style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'margin': 0,
            'letterSpacing': '2px',
        }),
    ]),

    # Controls
    html.Div(style={
        'display': 'flex',
        'gap': '16px',
        'marginBottom': '24px',
        'flexWrap': 'wrap',
    }, children=[
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
            html.Label("DAY / FILE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                             'letterSpacing': '2px', 'display': 'block', 'marginBottom': '6px'}),
            dcc.Dropdown(
                id='day-selector',
                options=day_options,
                value='all',
                clearable=False,
                style={'width': '200px', 'backgroundColor': COLORS['panel'],
                       'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}"},
            ),
        ]),
    ]),

    # Error banner
    html.Div(id='error-banner', style={'color': COLORS['red'], 'marginBottom': '16px',
                                        'fontSize': '13px'}),

    # Charts
    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[

        # Mid price
        html.Div(style={'backgroundColor': COLORS['panel'],
                        'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("MID PRICE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                        'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='mid-price-chart', config={'displayModeBar': False},
                      style={'height': '280px'}),
        ]),

        # Spread + Vol (side by side)
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'],
                            'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("BID-ASK SPREAD", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                  'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='spread-chart', config={'displayModeBar': False},
                          style={'height': '240px'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'],
                            'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P(f"ROLLING VOLATILITY (window={ROLLING_VOL_WINDOW})",
                       style={'color': COLORS['subtext'], 'fontSize': '11px',
                               'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='vol-chart', config={'displayModeBar': False},
                          style={'height': '240px'}),
            ]),
        ]),

        # Order book depth
        html.Div(style={'backgroundColor': COLORS['panel'],
                        'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("ORDER BOOK DEPTH (bid vs ask volume)", style={
                'color': COLORS['subtext'], 'fontSize': '11px',
                'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='depth-chart', config={'displayModeBar': False},
                      style={'height': '280px'}),
        ]),

    ]),
])


# ── Callbacks ─────────────────────────────────────────────────────────────────

@app.callback(
    Output('mid-price-chart', 'figure'),
    Output('spread-chart', 'figure'),
    Output('vol-chart', 'figure'),
    Output('depth-chart', 'figure'),
    Output('error-banner', 'children'),
    Input('product-selector', 'value'),
    Input('day-selector', 'value'),
)
def update_charts(product, day):
    if LOAD_ERROR:
        empty = make_empty_fig(LOAD_ERROR)
        return empty, empty, empty, empty, f"Error loading data: {LOAD_ERROR}"

    df = raw_df[raw_df['product'] == product].copy()
    if day != 'all':
        df = df[df['day'] == day]

    if df.empty:
        empty = make_empty_fig("No data for selection")
        return empty, empty, empty, empty, ""

    tick = df['tick']

    # ── Mid price ────────────────────────────────────────────────────────────
    fig_mid = go.Figure()
    fig_mid.add_trace(go.Scatter(
        x=tick, y=df['mid_price'],
        mode='lines', name='Mid Price',
        line=dict(color=COLORS['blue'], width=1.2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.05)',
    ))
    fig_mid.update_layout(**PLOTLY_LAYOUT, title=None)
    fig_mid.update_yaxes(title_text='Price')
    fig_mid.update_xaxes(title_text='Tick')

    # ── Spread ────────────────────────────────────────────────────────────────
    fig_spread = go.Figure()
    fig_spread.add_trace(go.Scatter(
        x=tick, y=df['spread'],
        mode='lines', name='Spread',
        line=dict(color=COLORS['orange'], width=1.0),
        fill='tozeroy', fillcolor='rgba(210,153,34,0.08)',
    ))
    fig_spread.update_layout(**PLOTLY_LAYOUT)
    fig_spread.update_yaxes(title_text='Spread')
    fig_spread.update_xaxes(title_text='Tick')

    # ── Rolling vol ───────────────────────────────────────────────────────────
    fig_vol = go.Figure()
    fig_vol.add_trace(go.Scatter(
        x=tick, y=df['rolling_vol'],
        mode='lines', name='Vol',
        line=dict(color=COLORS['purple'], width=1.0),
        fill='tozeroy', fillcolor='rgba(188,140,255,0.08)',
    ))
    fig_vol.update_layout(**PLOTLY_LAYOUT)
    fig_vol.update_yaxes(title_text='Volatility')
    fig_vol.update_xaxes(title_text='Tick')

    # ── Order book depth ──────────────────────────────────────────────────────
    bid_vol = df[['bid_volume_1','bid_volume_2','bid_volume_3']].sum(axis=1)
    ask_vol = df[['ask_volume_1','ask_volume_2','ask_volume_3']].sum(axis=1)

    fig_depth = go.Figure()
    fig_depth.add_trace(go.Scatter(
        x=tick, y=bid_vol,
        mode='lines', name='Total Bid Vol',
        line=dict(color=COLORS['green'], width=1.0),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.1)',
    ))
    fig_depth.add_trace(go.Scatter(
        x=tick, y=-ask_vol,
        mode='lines', name='Total Ask Vol',
        line=dict(color=COLORS['red'], width=1.0),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.1)',
    ))
    fig_depth.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
    fig_depth.update_layout(**PLOTLY_LAYOUT)
    fig_depth.update_yaxes(title_text='Volume (bids +ve, asks -ve)')
    fig_depth.update_xaxes(title_text='Tick')

    return fig_mid, fig_spread, fig_vol, fig_depth, ""


if __name__ == '__main__':
    print("Starting Price Visualiser → http://127.0.0.1:8050")
    app.run(debug=True)