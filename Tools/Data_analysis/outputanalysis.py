"""
Platform Log Analyser
======================
Run from Tools/data_visualisation/:
    python log_analysis.py

Drag and drop (or point to) your .log and .json files from the IMC platform.
Analyses:
  - PnL curve from graphLog
  - Fill analysis (size, price, direction, frequency) from tradeHistory
  - Position over time from activitiesLog
  - Spread capture — avg sell price vs avg buy price
  - Fill size distribution
  - Activity heatmap — when are fills happening
"""

import os
import json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State

# ── Config ────────────────────────────────────────────────────────────────────

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

# ── Parsers ───────────────────────────────────────────────────────────────────

def parse_log(raw):
    data = json.loads(raw) if isinstance(raw, str) else raw

    # Activities log
    lines   = data['activitiesLog'].strip().split('\n')
    headers = lines[0].split(';')
    acts    = pd.DataFrame([dict(zip(headers, l.split(';'))) for l in lines[1:]])
    for col in acts.columns:
        try:
            acts[col] = pd.to_numeric(acts[col])
        except Exception:
            pass

    # Trade history
    trades = pd.DataFrame(data.get('tradeHistory', []))

    # Per-tick logs
    logs = pd.DataFrame(data.get('logs', []))

    return acts, trades, logs


def parse_json(raw):
    data = json.loads(raw) if isinstance(raw, str) else raw

    # Graph log
    gl_lines = data['graphLog'].strip().split('\n')
    gl_rows  = [l.split(';') for l in gl_lines[1:]]
    graph_df = pd.DataFrame(gl_rows, columns=['timestamp', 'value'])
    graph_df['timestamp'] = pd.to_numeric(graph_df['timestamp'])
    graph_df['value']     = pd.to_numeric(graph_df['value'])

    positions = data.get('positions', [])
    profit    = data.get('profit', None)

    return graph_df, positions, profit


def make_empty_fig(msg="Upload log and json files to begin"):
    fig = go.Figure()
    fig.add_annotation(text=msg, xref="paper", yref="paper",
                       x=0.5, y=0.5, showarrow=False,
                       font=dict(color=COLORS['subtext'], size=14))
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


# ── App ───────────────────────────────────────────────────────────────────────

app = dash.Dash(__name__, suppress_callback_exceptions=True)
app.title = "Log Analyser"

app.layout = html.Div(style={
    'backgroundColor': COLORS['bg'],
    'minHeight': '100vh',
    'fontFamily': 'monospace',
    'color': COLORS['text'],
    'padding': '24px',
}, children=[

    # Header
    html.Div(style={'marginBottom': '24px'}, children=[
        html.H1("PLATFORM LOG ANALYSER", style={
            'color': COLORS['purple'],
            'fontSize': '20px',
            'letterSpacing': '4px',
            'margin': '0 0 4px 0',
            'fontWeight': 'bold',
        }),
        html.P("Prosperity · Fill Analysis · PnL · Position · Spread Capture", style={
            'color': COLORS['subtext'],
            'fontSize': '12px',
            'margin': 0,
            'letterSpacing': '2px',
        }),
    ]),

    # File upload
    html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr',
        'gap': '16px',
        'marginBottom': '24px',
    }, children=[
        html.Div(style={
            'backgroundColor': COLORS['panel'],
            'border': f"2px dashed {COLORS['border']}",
            'borderRadius': '6px',
            'padding': '16px',
        }, children=[
            html.Label(".LOG FILE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                            'letterSpacing': '2px', 'display': 'block',
                                            'marginBottom': '8px'}),
            dcc.Upload(
                id='upload-log',
                children=html.Div(['Drag & drop or ', html.A('select .log file',
                    style={'color': COLORS['purple']})]),
                style={'cursor': 'pointer', 'fontSize': '13px', 'color': COLORS['subtext']},
                multiple=False,
            ),
            html.Div(id='log-filename', style={'color': COLORS['green'], 'fontSize': '11px',
                                                'marginTop': '6px'}),
        ]),
        html.Div(style={
            'backgroundColor': COLORS['panel'],
            'border': f"2px dashed {COLORS['border']}",
            'borderRadius': '6px',
            'padding': '16px',
        }, children=[
            html.Label(".JSON FILE", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                             'letterSpacing': '2px', 'display': 'block',
                                             'marginBottom': '8px'}),
            dcc.Upload(
                id='upload-json',
                children=html.Div(['Drag & drop or ', html.A('select .json file',
                    style={'color': COLORS['purple']})]),
                style={'cursor': 'pointer', 'fontSize': '13px', 'color': COLORS['subtext']},
                multiple=False,
            ),
            html.Div(id='json-filename', style={'color': COLORS['green'], 'fontSize': '11px',
                                                 'marginTop': '6px'}),
        ]),
    ]),

    # Product selector
    html.Div(style={'marginBottom': '24px'}, children=[
        html.Label("PRODUCT", style={'color': COLORS['subtext'], 'fontSize': '11px',
                                      'letterSpacing': '2px', 'display': 'block',
                                      'marginBottom': '6px'}),
        dcc.Dropdown(
            id='product-selector',
            options=[],
            value=None,
            clearable=False,
            style={'width': '200px', 'backgroundColor': COLORS['panel'],
                   'color': COLORS['text'], 'border': f"1px solid {COLORS['border']}"},
        ),
    ]),

    html.Div(id='error-banner', style={'color': COLORS['red'], 'marginBottom': '16px',
                                        'fontSize': '13px'}),

    # Summary stats bar
    html.Div(id='summary-bar', style={'marginBottom': '24px'}),

    # Charts
    html.Div(style={'display': 'flex', 'flexDirection': 'column', 'gap': '20px'}, children=[

        # PnL curve
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("PnL CURVE (from platform graphLog)", style={
                'color': COLORS['subtext'], 'fontSize': '11px',
                'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='pnl-chart', config={'displayModeBar': False}, style={'height': '260px'}),
        ]),

        # Fill price + mid price
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("FILL PRICES vs MID PRICE", style={
                'color': COLORS['subtext'], 'fontSize': '11px',
                'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='fill-price-chart', config={'displayModeBar': False}, style={'height': '300px'}),
        ]),

        # Fill size dist + cumulative fills side by side
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("FILL SIZE DISTRIBUTION", style={
                    'color': COLORS['subtext'], 'fontSize': '11px',
                    'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='fill-dist-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("CUMULATIVE FILLS OVER TIME", style={
                    'color': COLORS['subtext'], 'fontSize': '11px',
                    'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='cumfill-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
        ]),

        # Position over time + fill frequency side by side
        html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '20px'}, children=[
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("POSITION OVER TIME", style={
                    'color': COLORS['subtext'], 'fontSize': '11px',
                    'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='position-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
            html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                            'borderRadius': '6px', 'padding': '16px'}, children=[
                html.P("FILL FREQUENCY OVER TIME", style={
                    'color': COLORS['subtext'], 'fontSize': '11px',
                    'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
                dcc.Graph(id='fill-freq-chart', config={'displayModeBar': False}, style={'height': '240px'}),
            ]),
        ]),

        # Spread capture
        html.Div(style={'backgroundColor': COLORS['panel'], 'border': f"1px solid {COLORS['border']}",
                        'borderRadius': '6px', 'padding': '16px'}, children=[
            html.P("SPREAD CAPTURE — buy vs sell fill prices over time", style={
                'color': COLORS['subtext'], 'fontSize': '11px',
                'letterSpacing': '2px', 'margin': '0 0 8px 0'}),
            dcc.Graph(id='spread-capture-chart', config={'displayModeBar': False}, style={'height': '260px'}),
        ]),

    ]),

    # Hidden stores
    dcc.Store(id='log-store'),
    dcc.Store(id='json-store'),
])


# ── Upload callbacks ──────────────────────────────────────────────────────────

@app.callback(
    Output('log-store',    'data'),
    Output('log-filename', 'children'),
    Input('upload-log',    'contents'),
    State('upload-log',    'filename'),
)
def store_log(contents, filename):
    if contents is None:
        return None, ''
    import base64
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    return decoded, f"✓ {filename}"


@app.callback(
    Output('json-store',    'data'),
    Output('json-filename', 'children'),
    Input('upload-json',    'contents'),
    State('upload-json',    'filename'),
)
def store_json(contents, filename):
    if contents is None:
        return None, ''
    import base64
    _, content_string = contents.split(',')
    decoded = base64.b64decode(content_string).decode('utf-8')
    return decoded, f"✓ {filename}"


@app.callback(
    Output('product-selector', 'options'),
    Output('product-selector', 'value'),
    Input('log-store', 'data'),
)
def update_products(log_data):
    if not log_data:
        return [], None
    try:
        acts, _, _ = parse_log(log_data)
        products   = sorted(acts['product'].unique().tolist())
        options    = [{'label': p, 'value': p} for p in products]
        return options, products[0] if products else None
    except Exception:
        return [], None


# ── Main chart callback ───────────────────────────────────────────────────────

@app.callback(
    Output('pnl-chart',            'figure'),
    Output('fill-price-chart',     'figure'),
    Output('fill-dist-chart',      'figure'),
    Output('cumfill-chart',        'figure'),
    Output('position-chart',       'figure'),
    Output('fill-freq-chart',      'figure'),
    Output('spread-capture-chart', 'figure'),
    Output('summary-bar',          'children'),
    Output('error-banner',         'children'),
    Input('log-store',         'data'),
    Input('json-store',        'data'),
    Input('product-selector',  'value'),
)
def update_charts(log_data, json_data, product):
    empty = make_empty_fig()
    empties = [empty] * 7

    if not log_data or not product:
        return *empties, html.Div(), ""

    try:
        acts, trades, logs = parse_log(log_data)
    except Exception as e:
        return *empties, html.Div(), f"Error parsing log: {e}"

    # Filter to product
    prod_acts   = acts[acts['product'] == product].copy().reset_index(drop=True)
    prod_trades = trades[trades['symbol'] == product].copy() if not trades.empty else pd.DataFrame()

    our_buys  = prod_trades[prod_trades['buyer']  == 'SUBMISSION'] if not prod_trades.empty else pd.DataFrame()
    our_sells = prod_trades[prod_trades['seller'] == 'SUBMISSION'] if not prod_trades.empty else pd.DataFrame()
    mkt       = prod_trades[(prod_trades['buyer'] != 'SUBMISSION') &
                             (prod_trades['seller'] != 'SUBMISSION')] if not prod_trades.empty else pd.DataFrame()

    # ── PnL curve ─────────────────────────────────────────────────────────────
    if json_data:
        try:
            graph_df, positions, profit = parse_json(json_data)
            fig_pnl = go.Figure()
            fig_pnl.add_trace(go.Scatter(
                x=graph_df['timestamp'], y=graph_df['value'],
                mode='lines', name='PnL',
                line=dict(color=COLORS['green'], width=1.2),
                fill='tozeroy', fillcolor='rgba(63,185,80,0.07)',
            ))
            fig_pnl.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
            fig_pnl.update_layout(**PLOTLY_LAYOUT)
            fig_pnl.update_yaxes(title_text='PnL')
            fig_pnl.update_xaxes(title_text='Timestamp')
        except Exception as e:
            fig_pnl = make_empty_fig(f"graphLog error: {e}")
    else:
        # Fall back to activitiesLog PnL
        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=prod_acts['timestamp'], y=prod_acts['profit_and_loss'],
            mode='lines', name='PnL',
            line=dict(color=COLORS['green'], width=1.2),
            fill='tozeroy', fillcolor='rgba(63,185,80,0.07)',
        ))
        fig_pnl.add_hline(y=0, line_color=COLORS['border'], line_width=0.8)
        fig_pnl.update_layout(**PLOTLY_LAYOUT)
        fig_pnl.update_yaxes(title_text='PnL')
        fig_pnl.update_xaxes(title_text='Timestamp')

    # ── Fill prices vs mid price ───────────────────────────────────────────────
    fig_fp = go.Figure()
    fig_fp.add_trace(go.Scatter(
        x=prod_acts['timestamp'], y=prod_acts['mid_price'],
        mode='lines', name='Mid Price',
        line=dict(color=COLORS['blue'], width=1.0, dash='dot'),
        opacity=0.6,
    ))
    if not our_buys.empty:
        fig_fp.add_trace(go.Scatter(
            x=our_buys['timestamp'], y=our_buys['price'],
            mode='markers', name='Our Buys',
            marker=dict(color=COLORS['green'], size=6, symbol='triangle-up'),
        ))
    if not our_sells.empty:
        fig_fp.add_trace(go.Scatter(
            x=our_sells['timestamp'], y=our_sells['price'],
            mode='markers', name='Our Sells',
            marker=dict(color=COLORS['red'], size=6, symbol='triangle-down'),
        ))
    if not mkt.empty:
        fig_fp.add_trace(go.Scatter(
            x=mkt['timestamp'], y=mkt['price'],
            mode='markers', name='Market',
            marker=dict(color=COLORS['subtext'], size=4, symbol='circle', opacity=0.4),
        ))
    fig_fp.update_layout(**PLOTLY_LAYOUT)
    fig_fp.update_yaxes(title_text='Price')
    fig_fp.update_xaxes(title_text='Timestamp')

    # ── Fill size distribution ────────────────────────────────────────────────
    fig_dist = go.Figure()
    if not our_buys.empty:
        fig_dist.add_trace(go.Histogram(
            x=our_buys['quantity'], name='Buy sizes',
            marker_color=COLORS['green'], opacity=0.7, nbinsx=20,
        ))
    if not our_sells.empty:
        fig_dist.add_trace(go.Histogram(
            x=our_sells['quantity'], name='Sell sizes',
            marker_color=COLORS['red'], opacity=0.7, nbinsx=20,
        ))
    fig_dist.update_layout(**PLOTLY_LAYOUT, barmode='overlay',
                            xaxis_title='Fill Size', yaxis_title='Count')

    # ── Cumulative fills ──────────────────────────────────────────────────────
    fig_cum = go.Figure()
    if not our_buys.empty:
        cum_buys = our_buys.sort_values('timestamp').copy()
        cum_buys['cum_qty'] = cum_buys['quantity'].cumsum()
        fig_cum.add_trace(go.Scatter(
            x=cum_buys['timestamp'], y=cum_buys['cum_qty'],
            mode='lines', name='Cumulative Buys',
            line=dict(color=COLORS['green'], width=1.2),
        ))
    if not our_sells.empty:
        cum_sells = our_sells.sort_values('timestamp').copy()
        cum_sells['cum_qty'] = cum_sells['quantity'].cumsum()
        fig_cum.add_trace(go.Scatter(
            x=cum_sells['timestamp'], y=cum_sells['cum_qty'],
            mode='lines', name='Cumulative Sells',
            line=dict(color=COLORS['red'], width=1.2),
        ))
    fig_cum.update_layout(**PLOTLY_LAYOUT)
    fig_cum.update_yaxes(title_text='Cumulative Quantity')
    fig_cum.update_xaxes(title_text='Timestamp')

    # ── Position over time ────────────────────────────────────────────────────
    fig_pos = go.Figure()
    fig_pos.add_trace(go.Scatter(
        x=prod_acts['timestamp'], y=prod_acts['profit_and_loss'] * 0,  # placeholder x-axis
        mode='lines', showlegend=False, line=dict(width=0),
    ))
    # Reconstruct position from trade history
    if not prod_trades.empty:
        all_our = pd.concat([
            our_buys.assign(delta=our_buys['quantity']),
            our_sells.assign(delta=-our_sells['quantity']),
        ]).sort_values('timestamp')
        all_our['position'] = all_our['delta'].cumsum()
        fig_pos.add_trace(go.Scatter(
            x=all_our['timestamp'], y=all_our['position'],
            mode='lines+markers', name='Position',
            line=dict(color=COLORS['orange'], width=1.2),
            marker=dict(size=4),
        ))
        fig_pos.add_hline(y=80,  line_color=COLORS['red'], line_dash='dash', line_width=0.8)
        fig_pos.add_hline(y=-80, line_color=COLORS['red'], line_dash='dash', line_width=0.8)
        fig_pos.add_hline(y=0,   line_color=COLORS['border'], line_width=0.5)
    fig_pos.update_layout(**PLOTLY_LAYOUT)
    fig_pos.update_yaxes(title_text='Position')
    fig_pos.update_xaxes(title_text='Timestamp')

    # ── Fill frequency over time ──────────────────────────────────────────────
    fig_freq = go.Figure()
    if not prod_trades.empty:
        # Bin timestamps into buckets
        bucket = 5000
        prod_trades_copy = prod_trades.copy()
        prod_trades_copy['bucket'] = (prod_trades_copy['timestamp'] // bucket) * bucket
        our_only = prod_trades_copy[
            (prod_trades_copy['buyer']  == 'SUBMISSION') |
            (prod_trades_copy['seller'] == 'SUBMISSION')
        ]
        if not our_only.empty:
            freq = our_only.groupby('bucket').size().reset_index(name='count')
            fig_freq.add_trace(go.Bar(
                x=freq['bucket'], y=freq['count'],
                name='Fills per 5k timestamps',
                marker_color=COLORS['teal'], opacity=0.8,
            ))
    fig_freq.update_layout(**PLOTLY_LAYOUT, bargap=0.1)
    fig_freq.update_yaxes(title_text='Fill Count')
    fig_freq.update_xaxes(title_text='Timestamp')

    # ── Spread capture ────────────────────────────────────────────────────────
    fig_sc = go.Figure()
    if not our_buys.empty:
        # Rolling avg buy price
        rb = our_buys.sort_values('timestamp').copy()
        rb['rolling_buy'] = rb['price'].rolling(5, min_periods=1).mean()
        fig_sc.add_trace(go.Scatter(
            x=rb['timestamp'], y=rb['rolling_buy'],
            mode='lines', name='Avg Buy Price (rolling 5)',
            line=dict(color=COLORS['green'], width=1.2),
        ))
    if not our_sells.empty:
        rs = our_sells.sort_values('timestamp').copy()
        rs['rolling_sell'] = rs['price'].rolling(5, min_periods=1).mean()
        fig_sc.add_trace(go.Scatter(
            x=rs['timestamp'], y=rs['rolling_sell'],
            mode='lines', name='Avg Sell Price (rolling 5)',
            line=dict(color=COLORS['red'], width=1.2),
        ))
    fig_sc.add_trace(go.Scatter(
        x=prod_acts['timestamp'], y=prod_acts['mid_price'],
        mode='lines', name='Mid Price',
        line=dict(color=COLORS['blue'], width=0.8, dash='dot'),
        opacity=0.5,
    ))
    fig_sc.update_layout(**PLOTLY_LAYOUT)
    fig_sc.update_yaxes(title_text='Price')
    fig_sc.update_xaxes(title_text='Timestamp')

    # ── Summary bar ───────────────────────────────────────────────────────────
    final_pnl     = float(prod_acts['profit_and_loss'].iloc[-1]) if not prod_acts.empty else 0
    total_buys    = int(our_buys['quantity'].sum())    if not our_buys.empty  else 0
    total_sells   = int(our_sells['quantity'].sum())   if not our_sells.empty else 0
    n_buys        = len(our_buys)
    n_sells       = len(our_sells)
    avg_buy_sz    = our_buys['quantity'].mean()        if not our_buys.empty  else 0
    avg_sell_sz   = our_sells['quantity'].mean()       if not our_sells.empty else 0
    avg_buy_px    = our_buys['price'].mean()           if not our_buys.empty  else 0
    avg_sell_px   = our_sells['price'].mean()          if not our_sells.empty else 0
    spread_cap    = avg_sell_px - avg_buy_px           if (not our_buys.empty and not our_sells.empty) else 0

    def stat_box(label, value, color=COLORS['text']):
        return html.Div(style={
            'backgroundColor': COLORS['panel'],
            'border': f"1px solid {COLORS['border']}",
            'borderRadius': '6px',
            'padding': '12px 16px',
            'textAlign': 'center',
        }, children=[
            html.Div(label, style={'color': COLORS['subtext'], 'fontSize': '10px',
                                    'letterSpacing': '2px', 'marginBottom': '6px'}),
            html.Div(value, style={'color': color, 'fontSize': '18px', 'fontWeight': 'bold'}),
        ])

    summary = html.Div(style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(8, 1fr)',
        'gap': '10px',
        'marginBottom': '8px',
    }, children=[
        stat_box("FINAL PnL",      f"{final_pnl:,.1f}",  COLORS['green'] if final_pnl >= 0 else COLORS['red']),
        stat_box("BUY FILLS",      str(n_buys),           COLORS['green']),
        stat_box("SELL FILLS",     str(n_sells),          COLORS['red']),
        stat_box("TOTAL BUY QTY",  str(total_buys),       COLORS['green']),
        stat_box("TOTAL SELL QTY", str(total_sells),      COLORS['red']),
        stat_box("AVG BUY SIZE",   f"{avg_buy_sz:.1f}",   COLORS['text']),
        stat_box("AVG SELL SIZE",  f"{avg_sell_sz:.1f}",  COLORS['text']),
        stat_box("SPREAD CAP",     f"{spread_cap:.2f}",   COLORS['yellow']),
    ])

    return fig_pnl, fig_fp, fig_dist, fig_cum, fig_pos, fig_freq, fig_sc, summary, ""


if __name__ == '__main__':
    print("Starting Log Analyser → http://127.0.0.1:8055")
    app.run(debug=True, port=8055)