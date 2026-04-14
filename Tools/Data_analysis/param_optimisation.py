"""
Parameter Calibration Tool
===========================
Run from Tools/data_analysis/:
    python param_calibration.py

Sweeps key strategy parameters and scores each combination by PnL.
Supports two modes:
  - All Data:          train and score on all files combined
  - Cross Validation:  for each param combo, train on each day and test on
                       the held-out day. Shows in-sample, out-of-sample, and
                       robustness (min of both) heatmaps side by side.
"""

import os
import sys
import glob
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State

# ── Path setup ────────────────────────────────────────────────────────────────

_HERE     = os.path.dirname(os.path.abspath(__file__))
_BACKTEST = os.path.join(_HERE, '..', 'backtest')
_FILES    = os.path.join(_HERE, '..', '..', 'Files')
sys.path.insert(0, _BACKTEST)

from backtester import Backtester, TradingAlgorithm, Order

# ── Data files ────────────────────────────────────────────────────────────────

PRICE_FILES = sorted(glob.glob(os.path.join(_FILES, 'Prices', '*.csv')))
TRADE_FILES = sorted(glob.glob(os.path.join(_FILES, 'Trades', '*.csv')))

# ── Product type mapping ──────────────────────────────────────────────────────

# Stable products use fixed fair-value strategy (default_edge × take_width sweep)
# All others use EMA-based strategy (alpha × default_edge sweep)
STABLE_PRODUCTS = {'ASH_COATED_OSMIUM'}

# ── Colors ────────────────────────────────────────────────────────────────────

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
    'yellow':  '#e3b341',
    'purple':  '#bc8cff',
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor=COLORS['bg'],
    plot_bgcolor=COLORS['panel'],
    font=dict(color=COLORS['text'], family='monospace'),
    margin=dict(l=60, r=20, t=50, b=60),
)

# ── Algorithms ────────────────────────────────────────────────────────────────

class StableSweep(TradingAlgorithm):
    """Fixed fair-value market maker sweep — for ASH_COATED_OSMIUM and EMERALDS."""
    def __init__(self, default_edge, take_width, fair_value=10000,
                 disregard_edge=6, join_edge=2, product='ASH_COATED_OSMIUM'):
        super().__init__()
        self.product        = product
        self.fair_value     = fair_value
        self.default_edge   = default_edge
        self.take_width     = take_width
        self.disregard_edge = disregard_edge
        self.join_edge      = join_edge
        self.LIMIT          = 80

    def get_orders(self, order_books, positions):
        book = order_books.get(self.product)
        if not book: return []
        pos     = positions.get(self.product, 0)
        orders  = []
        fv      = self.fair_value
        buy_vol = sell_vol = 0

        if book.asks:
            best_ask = min(p for p, _ in book.asks)
            ask_vol  = next(v for p, v in book.asks if p == best_ask)
            if best_ask <= fv - self.take_width:
                qty = min(ask_vol, self.LIMIT - pos)
                if qty > 0:
                    orders.append(Order(self.product, 'BUY', best_ask, qty))
                    buy_vol += qty

        if book.bids:
            best_bid = max(p for p, _ in book.bids)
            bid_vol  = next(v for p, v in book.bids if p == best_bid)
            if best_bid >= fv + self.take_width:
                qty = min(bid_vol, self.LIMIT + pos)
                if qty > 0:
                    orders.append(Order(self.product, 'SELL', best_bid, qty))
                    sell_vol += qty

        asks_outside = [p for p, _ in book.asks if p > fv + self.disregard_edge]
        bids_outside = [p for p, _ in book.bids if p < fv - self.disregard_edge]

        ask = fv + self.default_edge
        if asks_outside:
            ba  = min(asks_outside)
            ask = ba if abs(ba - fv) <= self.join_edge else ba - 1

        bid = fv - self.default_edge
        if bids_outside:
            bb  = max(bids_outside)
            bid = bb if abs(fv - bb) <= self.join_edge else bb + 1

        buy_qty  = self.LIMIT - (pos + buy_vol)
        sell_qty = self.LIMIT + (pos - sell_vol)
        if buy_qty  > 0: orders.append(Order(self.product, 'BUY',  round(bid), buy_qty))
        if sell_qty > 0: orders.append(Order(self.product, 'SELL', round(ask), sell_qty))
        return orders


class EMASweep(TradingAlgorithm):
    """EMA-based market maker sweep — for INTARIAN_PEPPER_ROOT and TOMATOES."""
    def __init__(self, alpha, default_edge, take_width=5, vol_scale_factor=0.0,
                 vol_window=20, product='INTARIAN_PEPPER_ROOT'):
        super().__init__()
        self.product          = product
        self.alpha            = alpha
        self.default_edge     = default_edge
        self.take_width       = take_width
        self.vol_scale_factor = vol_scale_factor
        self.vol_window       = vol_window
        self.ema              = None
        self._mid_history     = []
        self.LIMIT            = 80

    def _rolling_std(self, mid):
        self._mid_history.append(mid)
        if len(self._mid_history) > self.vol_window + 1:
            self._mid_history.pop(0)
        if len(self._mid_history) < self.vol_window + 1:
            return 0.0
        changes  = [self._mid_history[i] - self._mid_history[i - 1]
                    for i in range(1, len(self._mid_history))]
        mean     = sum(changes) / len(changes)
        variance = sum((c - mean) ** 2 for c in changes) / len(changes)
        return variance ** 0.5

    def get_orders(self, order_books, positions):
        book = order_books.get(self.product)
        if not book: return []
        mid      = book.mid_price
        self.ema = mid if self.ema is None else self.alpha * mid + (1 - self.alpha) * self.ema
        fv       = round(self.ema)
        pos      = positions.get(self.product, 0)
        orders   = []
        buy_vol  = sell_vol = 0

        rolling_std    = self._rolling_std(mid)
        effective_edge = self.default_edge + round(rolling_std * self.vol_scale_factor)

        # Aggressive — walk up to 2 ask levels through fair value
        levels_hit = 0
        for ask_px, ask_vol in sorted(book.asks, key=lambda x: x[0]):
            if levels_hit >= 2 or ask_px > fv - self.take_width: break
            qty = min(abs(ask_vol), self.LIMIT - pos - buy_vol)
            if qty > 0:
                orders.append(Order(self.product, 'BUY', ask_px, qty))
                buy_vol += qty; levels_hit += 1

        # Aggressive — walk up to 2 bid levels through fair value
        levels_hit = 0
        for bid_px, bid_vol in sorted(book.bids, key=lambda x: -x[0]):
            if levels_hit >= 2 or bid_px < fv + self.take_width: break
            qty = min(bid_vol, self.LIMIT + pos - sell_vol)
            if qty > 0:
                orders.append(Order(self.product, 'SELL', bid_px, qty))
                sell_vol += qty; levels_hit += 1

        # Passive quotes with vol-scaled edge
        if self.LIMIT - pos - buy_vol > 0:
            orders.append(Order(self.product, 'BUY',  fv - effective_edge,
                                self.LIMIT - pos - buy_vol))
        if self.LIMIT + pos - sell_vol > 0:
            orders.append(Order(self.product, 'SELL', fv + effective_edge,
                                self.LIMIT + pos - sell_vol))
        return orders


# ── Backtester runner ─────────────────────────────────────────────────────────

def run_bt(algo, price_files, trade_files):
    """Run backtester silently, return (final_pnl, pnl_history)."""
    if not price_files or not trade_files:
        return 0.0, []
    bt = Backtester(price_files, trade_files, algo)
    bt._plot        = lambda products: None
    bt._save_output = lambda products: None
    bt.run()
    return (bt.pnl_history[-1] if bt.pnl_history else 0.0), bt.pnl_history


def make_algo(product, x_val, y_val, sweep_axes='default', fixed_params=None):
    fp = fixed_params or {}
    if product in STABLE_PRODUCTS:
        return StableSweep(default_edge=int(x_val), take_width=int(y_val), product=product)
    else:
        # EMA sweep axis options:
        # 'edge_alpha', 'edge_tw', 'alpha_tw', 'edge_vol', 'alpha_vol', 'tw_vol'
        kwargs = dict(product=product)
        if sweep_axes == 'edge_tw':
            return EMASweep(alpha=fp.get('alpha', 0.25), default_edge=int(x_val),
                            take_width=int(y_val),
                            vol_scale_factor=fp.get('vol_scale_factor', 0.0), **kwargs)
        elif sweep_axes == 'alpha_tw':
            return EMASweep(alpha=round(x_val / 100, 4),
                            default_edge=fp.get('default_edge', 5),
                            take_width=int(y_val),
                            vol_scale_factor=fp.get('vol_scale_factor', 0.0), **kwargs)
        elif sweep_axes == 'edge_vol':
            return EMASweep(alpha=fp.get('alpha', 0.25), default_edge=int(x_val),
                            take_width=fp.get('take_width', 5),
                            vol_scale_factor=round(y_val / 100, 4), **kwargs)
        elif sweep_axes == 'alpha_vol':
            return EMASweep(alpha=round(x_val / 100, 4),
                            default_edge=fp.get('default_edge', 5),
                            take_width=fp.get('take_width', 5),
                            vol_scale_factor=round(y_val / 100, 4), **kwargs)
        elif sweep_axes == 'tw_vol':
            return EMASweep(alpha=fp.get('alpha', 0.25),
                            default_edge=fp.get('default_edge', 5),
                            take_width=int(x_val),
                            vol_scale_factor=round(y_val / 100, 4), **kwargs)
        else:  # default: edge_alpha
            return EMASweep(alpha=round(y_val / 100, 4), default_edge=int(x_val),
                            take_width=fp.get('take_width', 5),
                            vol_scale_factor=fp.get('vol_scale_factor', 0.0), **kwargs)


# ── Sweep functions ───────────────────────────────────────────────────────────

def sweep_all(product, x_vals, y_vals, sweep_axes='default', fixed_params=None):
    """Single sweep over all data."""
    z = np.zeros((len(y_vals), len(x_vals)))
    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            pnl, _ = run_bt(make_algo(product, xv, yv, sweep_axes, fixed_params),
                            PRICE_FILES, TRADE_FILES)
            z[i, j] = pnl
    return z


def sweep_crossval(product, x_vals, y_vals, sweep_axes='default', fixed_params=None):
    """
    Cross validation sweep.
    For each param combo, trains on each fold and tests on the held-out fold.
    Returns:
      z_in  — average in-sample PnL
      z_out — average out-of-sample PnL
      z_rob — robustness score = min(out-of-sample PnLs across folds)
    """
    n_files = min(len(PRICE_FILES), len(TRADE_FILES))
    if n_files < 2:
        z = sweep_all(product, x_vals, y_vals, sweep_axes, fixed_params)
        return z, z, z

    z_in  = np.zeros((len(y_vals), len(x_vals)))
    z_out = np.zeros((len(y_vals), len(x_vals)))
    z_rob = np.zeros((len(y_vals), len(x_vals)))

    for i, yv in enumerate(y_vals):
        for j, xv in enumerate(x_vals):
            in_scores  = []
            out_scores = []

            for fold in range(n_files):
                test_p  = [PRICE_FILES[fold]]
                test_t  = [TRADE_FILES[fold]]
                train_p = [f for k, f in enumerate(PRICE_FILES[:n_files]) if k != fold]
                train_t = [f for k, f in enumerate(TRADE_FILES[:n_files]) if k != fold]

                if not train_p:
                    continue

                in_pnl, _  = run_bt(make_algo(product, xv, yv, sweep_axes, fixed_params),
                                    train_p, train_t)
                out_pnl, _ = run_bt(make_algo(product, xv, yv, sweep_axes, fixed_params),
                                    test_p,  test_t)

                in_scores.append(in_pnl)
                out_scores.append(out_pnl)

            z_in[i, j]  = np.mean(in_scores)  if in_scores  else 0.0
            z_out[i, j] = np.mean(out_scores) if out_scores else 0.0
            z_rob[i, j] = np.min(out_scores)  if out_scores else 0.0

    return z_in, z_out, z_rob


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


def make_heatmap(z, x_labels, y_labels, x_title, y_title, title, colorscale='RdYlGn'):
    fig = go.Figure(go.Heatmap(
        z=z,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        hoverongaps=False,
        hovertemplate=f'{x_title}: %{{x}}<br>{y_title}: %{{y}}<br>PnL: %{{z:,.0f}}<extra></extra>',
    ))
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title=dict(text=title, font=dict(color=COLORS['subtext'], size=11), x=0.01),
        xaxis_title=x_title,
        yaxis_title=y_title,
    )
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Parameter Calibration"

app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': 'monospace',
    'color': COLORS['text'],
    'padding': '24px',
}, children=[

    # Header
    html.Div(style={'marginBottom': '24px'}, children=[
        html.H1("PARAMETER CALIBRATION", style={
            'color': COLORS['yellow'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · Parameter Sweep · Cross Validation", style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'margin': 0,
            'letterSpacing': '2px',
        }),
    ]),

    # Product + mode selectors
    html.Div(style={'display': 'flex', 'gap': '40px', 'marginBottom': '24px', 'flexWrap': 'wrap'},
    children=[
        html.Div(children=[
            html.Label("PRODUCT", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                          'letterSpacing': '2px', 'display': 'block',
                                          'marginBottom': '8px'}),
            dcc.RadioItems(
                id='product-selector',
                options=[
                    {'label': '  ASH_COATED_OSMIUM      (default_edge × take_width)', 'value': 'ASH_COATED_OSMIUM'},
                    {'label': '  INTARIAN_PEPPER_ROOT   (alpha × default_edge)',       'value': 'INTARIAN_PEPPER_ROOT'},
                ],
                value='ASH_COATED_OSMIUM',
                inline=True,
                style={'color': COLORS['text'], 'fontSize': '13px'},
                inputStyle={'marginRight': '6px'},
                labelStyle={'marginRight': '32px'},
            ),
        ]),
        html.Div(children=[
            html.Label("MODE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                       'letterSpacing': '2px', 'display': 'block',
                                       'marginBottom': '8px'}),
            dcc.RadioItems(
                id='mode-selector',
                options=[
                    {'label': '  All Data',         'value': 'all'},
                    {'label': '  Cross Validation', 'value': 'crossval'},
                ],
                value='all',
                inline=True,
                style={'color': COLORS['text'], 'fontSize': '13px'},
                inputStyle={'marginRight': '6px'},
                labelStyle={'marginRight': '32px'},
            ),
        ]),
    ]),

    # Cross-val explainer (shown only in crossval mode)
    html.Div(id='cv-explainer', style={'marginBottom': '16px'}),

    # Parameter range controls
    html.Div(id='param-controls', style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '20px',
        'marginBottom': '24px',
    }),

    # Run button
    html.Div(style={'marginBottom': '24px'}, children=[
        html.Button("RUN SWEEP", id='run-btn', n_clicks=0, style={
            'backgroundColor': COLORS['green'],
            'color': '#0d1117',
            'border': 'none',
            'padding': '10px 32px',
            'fontSize': '13px',
            'letterSpacing': '2px',
            'fontWeight': 'bold',
            'fontFamily': 'monospace',
            'cursor': 'pointer',
            'borderRadius': '4px',
        }),
        html.Span(id='status-msg', style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'marginLeft': '16px',
            'letterSpacing': '1px',
        }),
    ]),

    # Results area
    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[
        html.Div(id='heatmap-area'),
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 2fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'],
                            'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '20px'}, children=[
                html.P("BEST PARAMETERS", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                   'letterSpacing': '2px', 'margin': '0 0 16px 0'}),
                html.Div(id='best-params'),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'],
                            'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P(id='pnl-curve-title', style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                      'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='best-pnl', config={'displayModeBar': False}, style={'height': '280px'}),
            ]),
        ]),
    ]),
])


# ── Dynamic controls ──────────────────────────────────────────────────────────

@app.callback(
    Output('param-controls', 'children'),
    Output('cv-explainer',   'children'),
    Input('product-selector', 'value'),
    Input('mode-selector',    'value'),
)
def update_controls(product, mode):
    label_style = {'color': COLORS['subtext'], 'fontSize': '11px',
                   'letterSpacing': '2px', 'display': 'block', 'marginBottom': '8px'}
    panel_style = {'backgroundColor': COLORS['panel'],
                   'border': f"1px solid {COLORS['border']}",
                   'borderRadius': '6px', 'padding': '16px'}
    input_style = {'width': '80px', 'backgroundColor': COLORS['bg'],
                   'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}",
                   'padding': '6px', 'fontFamily': 'monospace', 'fontSize': '13px'}

    def slider_block(label, id_prefix, step, default_min, default_max):
        return html.Div(style=panel_style, children=[
            html.Label(label, style=label_style),
            html.Div(style={'display': 'flex', 'gap': '12px'}, children=[
                html.Div(children=[
                    html.Label("MIN",  style={**label_style, 'fontSize': '10px'}),
                    dcc.Input(id=f'{id_prefix}-min',  type='number', value=default_min,
                              step=step, style=input_style),
                ]),
                html.Div(children=[
                    html.Label("MAX",  style={**label_style, 'fontSize': '10px'}),
                    dcc.Input(id=f'{id_prefix}-max',  type='number', value=default_max,
                              step=step, style=input_style),
                ]),
                html.Div(children=[
                    html.Label("STEP", style={**label_style, 'fontSize': '10px'}),
                    dcc.Input(id=f'{id_prefix}-step', type='number', value=step,
                              step=step, style=input_style),
                ]),
            ]),
        ])

    # EMA-only controls are always rendered so their IDs always exist in the DOM.
    # When a stable product is selected they are hidden via display:none — Dash
    # requires every State/Input ID to be present in the layout at callback time.
    ema_hidden = product in STABLE_PRODUCTS
    ema_style  = {**panel_style, 'display': 'none'} if ema_hidden else panel_style

    ema_controls = html.Div(style=ema_style, children=[
        html.Label("SWEEP AXES", style=label_style),
        dcc.RadioItems(
            id='tomatoes-axes',
            options=[
                {'label': '  default_edge × alpha',            'value': 'edge_alpha'},
                {'label': '  default_edge × take_width',       'value': 'edge_tw'},
                {'label': '  alpha × take_width',              'value': 'alpha_tw'},
                {'label': '  default_edge × vol_scale_factor', 'value': 'edge_vol'},
                {'label': '  alpha × vol_scale_factor',        'value': 'alpha_vol'},
                {'label': '  take_width × vol_scale_factor',   'value': 'tw_vol'},
            ],
            value='edge_alpha',
            style={'color': COLORS['text'], 'fontSize': '12px'},
            inputStyle={'marginRight': '6px'},
            labelStyle={'display': 'block', 'marginBottom': '6px'},
        ),
        html.Div(style={'marginTop': '12px', 'display': 'flex', 'gap': '16px', 'flexWrap': 'wrap'},
        children=[
            html.Div(children=[
                html.Label("FIXED ALPHA (×0.01)", style={**label_style, 'fontSize': '10px'}),
                dcc.Input(id='fixed-alpha', type='number', value=25, step=1,
                          style={**input_style, 'width': '80px'}),
            ]),
            html.Div(children=[
                html.Label("FIXED DEFAULT EDGE", style={**label_style, 'fontSize': '10px'}),
                dcc.Input(id='fixed-edge', type='number', value=5, step=1,
                          style={**input_style, 'width': '80px'}),
            ]),
            html.Div(children=[
                html.Label("FIXED TAKE WIDTH", style={**label_style, 'fontSize': '10px'}),
                dcc.Input(id='fixed-tw', type='number', value=5, step=1,
                          style={**input_style, 'width': '80px'}),
            ]),
            html.Div(children=[
                html.Label("FIXED VOL SCALE (×0.01)", style={**label_style, 'fontSize': '10px'}),
                dcc.Input(id='fixed-vol', type='number', value=50, step=1,
                          style={**input_style, 'width': '80px'}),
            ]),
        ]),
    ])

    if product in STABLE_PRODUCTS:
        controls = [
            slider_block('X AXIS — DEFAULT EDGE', 'x', 1, 2, 12),
            slider_block('Y AXIS — TAKE WIDTH',   'y', 1, 1, 6),
            ema_controls,   # hidden, but present so IDs exist in the DOM
        ]
    else:
        controls = [
            ema_controls,
            html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '12px'}, children=[
                slider_block('X AXIS RANGE', 'x', 1, 4, 16),
                slider_block('Y AXIS RANGE', 'y', 1, 4, 30),
            ]),
        ]

    if mode == 'crossval':
        n = min(len(PRICE_FILES), len(TRADE_FILES))
        explainer = html.Div(style={
            'backgroundColor': COLORS['panel'],
            'border': f"1px solid {COLORS['orange']}",
            'borderRadius': '6px',
            'padding': '12px 16px',
            'fontSize': '12px',
            'color': COLORS['subtext'],
            'letterSpacing': '1px',
        }, children=[
            html.Span("CROSS VALIDATION MODE  ",
                      style={'color': COLORS['orange'], 'fontWeight': 'bold'}),
            html.Span(
                f"Using {n} file(s). For each param combo, each file is held out in turn as the test set "
                f"and the rest are used to train. Shows 3 heatmaps: IN-SAMPLE, OUT-OF-SAMPLE, and "
                f"ROBUSTNESS (worst out-of-sample fold). Pick params that score well on all three."
            ),
        ])
    else:
        explainer = html.Div()

    return controls, explainer


# ── Sweep callback ────────────────────────────────────────────────────────────

@app.callback(
    Output('heatmap-area',    'children'),
    Output('best-params',     'children'),
    Output('best-pnl',        'figure'),
    Output('pnl-curve-title', 'children'),
    Output('status-msg',      'children'),
    Input('run-btn', 'n_clicks'),
    State('product-selector', 'value'),
    State('mode-selector',    'value'),
    State('x-min',  'value'),
    State('x-max',  'value'),
    State('x-step', 'value'),
    State('y-min',  'value'),
    State('y-max',  'value'),
    State('y-step', 'value'),
    State('tomatoes-axes', 'value'),
    State('fixed-alpha',   'value'),
    State('fixed-edge',    'value'),
    State('fixed-tw',      'value'),
    State('fixed-vol',     'value'),
    prevent_initial_call=True,
)
def run_sweep(n_clicks, product, mode, x_min, x_max, x_step, y_min, y_max, y_step,
              ema_axes, fixed_alpha, fixed_edge, fixed_tw, fixed_vol):
    empty_fig = make_empty_fig()

    if not PRICE_FILES or not TRADE_FILES:
        return html.Div(), "No data files.", empty_fig, "", "Error: no data files."

    try:
        x_vals = list(np.arange(float(x_min), float(x_max) + float(x_step) / 2, float(x_step)))
        y_vals = list(np.arange(float(y_min), float(y_max) + float(y_step) / 2, float(y_step)))
    except Exception:
        return html.Div(), "Invalid inputs.", empty_fig, "", "Error: bad inputs."

    total = len(x_vals) * len(y_vals)
    limit = 200 if mode == 'all' else 100
    if total > limit:
        return html.Div(), "", empty_fig, "", \
               f"⚠ Too many combinations ({total}). Reduce range or increase step."

    sweep_axes   = ema_axes or 'edge_alpha'
    fixed_params = {
        'alpha':            round((fixed_alpha or 25) / 100, 4),
        'default_edge':     int(fixed_edge or 5),
        'take_width':       int(fixed_tw   or 5),
        'vol_scale_factor': round((fixed_vol or 50) / 100, 4),
    }

    if product in STABLE_PRODUCTS:
        x_label = 'default_edge'
        y_label = 'take_width'
        x_tick  = [str(int(v)) for v in x_vals]
        y_tick  = [str(int(v)) for v in y_vals]
    else:
        axis_labels = {
            'edge_alpha': ('default_edge', 'alpha'),
            'edge_tw':    ('default_edge', 'take_width'),
            'alpha_tw':   ('alpha',        'take_width'),
            'edge_vol':   ('default_edge', 'vol_scale_factor'),
            'alpha_vol':  ('alpha',        'vol_scale_factor'),
            'tw_vol':     ('take_width',   'vol_scale_factor'),
        }
        x_label, y_label = axis_labels.get(sweep_axes, ('default_edge', 'alpha'))
        x_tick = [str(round(v / 100, 2)) for v in x_vals] \
                 if x_label == 'alpha' else [str(int(v)) for v in x_vals]
        y_tick = [str(round(v / 100, 2)) for v in y_vals] \
                 if y_label in ('alpha', 'vol_scale_factor') else [str(int(v)) for v in y_vals]

    # ── Run the sweep ─────────────────────────────────────────────────────────
    if mode == 'all':
        z = sweep_all(product, x_vals, y_vals, sweep_axes, fixed_params)
        best_z = z
        heatmap_area = html.Div(style={
            'backgroundColor': COLORS['panel'],
            'border': f"1px solid {COLORS['border']}",
            'borderRadius': '6px', 'padding': '16px',
        }, children=[
            html.P("PnL HEATMAP — ALL DATA", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                      'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(figure=make_heatmap(z, x_tick, y_tick, x_label, y_label, ''),
                      config={'displayModeBar': False}, style={'height': '420px'}),
        ])
        pnl_title = "PnL CURVE — BEST PARAMS (ALL DATA)"

    else:
        z_in, z_out, z_rob = sweep_crossval(product, x_vals, y_vals, sweep_axes, fixed_params)
        best_z = z_rob

        fig_in  = make_heatmap(z_in,  x_tick, y_tick, x_label, y_label, 'IN-SAMPLE')
        fig_out = make_heatmap(z_out, x_tick, y_tick, x_label, y_label, 'OUT-OF-SAMPLE')
        fig_rob = make_heatmap(z_rob, x_tick, y_tick, x_label, y_label,
                               'ROBUSTNESS (worst out-of-sample fold)', colorscale='Blues')

        heatmap_area = html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '16px'},
        children=[
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '16px'}, children=[
                html.Div(style={'backgroundColor': COLORS['panel'],
                                'border': f"1px solid {COLORS['border']}",
                                'borderRadius': '6px', 'padding': '16px'}, children=[
                    html.P("IN-SAMPLE PnL", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                    'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                    dcc.Graph(figure=fig_in, config={'displayModeBar': False},
                              style={'height': '340px'}),
                ]),
                html.Div(style={'backgroundColor': COLORS['panel'],
                                'border': f"1px solid {COLORS['border']}",
                                'borderRadius': '6px', 'padding': '16px'}, children=[
                    html.P("OUT-OF-SAMPLE PnL", style={'color': COLORS['orange'], 'fontSize': '11px',
                                                        'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                    dcc.Graph(figure=fig_out, config={'displayModeBar': False},
                              style={'height': '340px'}),
                ]),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'],
                            'border': f"1px solid {COLORS['purple']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("ROBUSTNESS SCORE — worst out-of-sample fold (higher = more generalisable)",
                       style={'color': COLORS['purple'], 'fontSize': '11px',
                               'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(figure=fig_rob, config={'displayModeBar': False},
                          style={'height': '340px'}),
            ]),
        ])
        pnl_title = "PnL CURVE — BEST ROBUST PARAMS (ALL DATA)"

    # ── Best params ───────────────────────────────────────────────────────────
    best_idx = np.unravel_index(np.argmax(best_z), best_z.shape)
    best_x   = x_vals[best_idx[1]]
    best_y   = y_vals[best_idx[0]]
    best_pnl = best_z[best_idx]

    if product in STABLE_PRODUCTS:
        best_labels = [(x_label, str(int(best_x))), (y_label, str(int(best_y)))]
        best_algo   = StableSweep(default_edge=int(best_x), take_width=int(best_y), product=product)
    else:
        best_labels = [
            (x_label, str(round(best_x / 100, 4) if x_label == 'alpha' else int(best_x))),
            (y_label, str(round(best_y / 100, 4) if y_label == 'alpha' else int(best_y))),
        ]
        best_algo = make_algo(product, best_x, best_y, sweep_axes, fixed_params)

    score_color = COLORS['purple'] if mode == 'crossval' else COLORS['yellow']
    score_label = "ROBUSTNESS SCORE" if mode == 'crossval' else "BEST PnL"

    param_div = html.Div(children=[
        *[html.Div(style={'marginBottom': '12px'}, children=[
            html.Span(f"{k}  ", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                        'letterSpacing': '2px'}),
            html.Span(v, style={'color': COLORS['green'], 'fontSize': '20px',
                                 'fontWeight': 'bold'}),
        ]) for k, v in best_labels],
        html.Div(style={'marginTop': '16px', 'paddingTop': '16px',
                        'borderTop': f"1px solid {COLORS['border']}"}, children=[
            html.Span(f"{score_label}  ", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                                   'letterSpacing': '2px'}),
            html.Span(f"{best_pnl:,.0f}", style={'color': score_color, 'fontSize': '22px',
                                                   'fontWeight': 'bold'}),
        ]),
    ])

    # ── PnL curve for best params on full data ────────────────────────────────
    _, pnl_hist = run_bt(best_algo, PRICE_FILES, TRADE_FILES)
    fig_pnl = go.Figure()
    fig_pnl.add_trace(go.Scatter(
        x=list(range(len(pnl_hist))), y=pnl_hist,
        mode='lines', name='PnL',
        line=dict(color=COLORS['green'], width=1.2),
        fill='tozeroy', fillcolor='rgba(63,185,80,0.07)',
    ))
    fig_pnl.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
    fig_pnl.update_layout(**PLOTLY_LAYOUT)
    fig_pnl.update_yaxes(title_text='PnL')
    fig_pnl.update_xaxes(title_text='Tick')

    multiplier = min(len(PRICE_FILES), len(TRADE_FILES)) if mode == 'crossval' else 1
    status = f"✓ Swept {total} combinations × {multiplier} fold(s) = {total * multiplier} runs."

    return heatmap_area, param_div, fig_pnl, pnl_title, status


if __name__ == '__main__':
    print("Starting Parameter Calibration → http://127.0.0.1:8053")
    app.run(debug=True, port=8053)