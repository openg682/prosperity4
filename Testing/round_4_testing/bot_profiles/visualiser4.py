"""
visualise_r4.py  —  local web viewer for trades + mid price
============================================================
    pip install flask pandas
    python visualise_r4.py
Then open http://localhost:5000
"""

import json
import os
import pandas as pd
from flask import Flask, jsonify, render_template_string

# ── CONFIG ────────────────────────────────────────────────────────────────────

PRICE_FILES = [
    "../../../Files/Prices/prices_round_4_day_1.csv",
    "../../../Files/Prices/prices_round_4_day_2.csv",
    "../../../Files/Prices/prices_round_4_day_3.csv",
]

TRADE_FILES = [
    "../../../Files/Trades/trades_round_4_day_1.csv",
    "../../../Files/Trades/trades_round_4_day_2.csv",
    "../../../Files/Trades/trades_round_4_day_3.csv",
]
#change bot over here
FOCUS_BOT  = "Mark 67"
DAY_TICKS  = 1_000_000

# ── DATA LOADING ──────────────────────────────────────────────────────────────

def load_prices():
    dfs = []
    for i, f in enumerate(PRICE_FILES, start=1):
        try:
            df = pd.read_csv(f, sep=";")
            if "day" not in df.columns:
                df["day"] = i
            dfs.append(df)
            print(f"  ✓ prices {f} ({len(df):,} rows)")
        except FileNotFoundError:
            print(f"  ✗ missing: {f}")
    if not dfs:
        return pd.DataFrame()
    p = pd.concat(dfs, ignore_index=True)
    p["day"]       = pd.to_numeric(p["day"],       errors="coerce")
    p["timestamp"] = pd.to_numeric(p["timestamp"], errors="coerce")
    p["mid_price"] = pd.to_numeric(p["mid_price"], errors="coerce")
    p["ct"] = (p["day"] - 1) * DAY_TICKS + p["timestamp"]
    # downsample to 3000 pts per product
    parts = []
    for _, grp in p.groupby("product"):
        step = max(1, len(grp) // 3000)
        parts.append(grp.iloc[::step])
    return pd.concat(parts).sort_values("ct").reset_index(drop=True)


def load_trades():
    dfs = []
    for i, f in enumerate(TRADE_FILES, start=1):
        try:
            df = pd.read_csv(f)
            if df.shape[1] == 1:
                df = pd.read_csv(f, sep=";")
            if "day" not in df.columns:
                df["day"] = i
            dfs.append(df)
            print(f"  ✓ trades {f} ({len(df):,} rows)")
        except FileNotFoundError:
            print(f"  ✗ missing: {f}")
    if not dfs:
        return pd.DataFrame()
    t = pd.concat(dfs, ignore_index=True)
    t.columns = t.columns.str.strip()
    if "product" in t.columns and "symbol" not in t.columns:
        t = t.rename(columns={"product": "symbol"})
    t["day"]       = pd.to_numeric(t.get("day"),       errors="coerce")
    t["price"]     = pd.to_numeric(t.get("price"),     errors="coerce")
    t["quantity"]  = pd.to_numeric(t.get("quantity"),  errors="coerce")
    t["timestamp"] = pd.to_numeric(t.get("timestamp"), errors="coerce")
    if "continuous_time" not in t.columns:
        t["continuous_time"] = (t["day"] - 1) * DAY_TICKS + t["timestamp"]
    else:
        t["continuous_time"] = pd.to_numeric(t["continuous_time"], errors="coerce")
    return t


print("Loading data...")
PRICES = load_prices()
TRADES = load_trades()
PRODUCTS = sorted(
    set(PRICES["product"].unique() if not PRICES.empty else []) |
    set(TRADES["symbol"].unique()  if not TRADES.empty else [])
)
print(f"\nProducts: {PRODUCTS}\n")

# ── FLASK APP ─────────────────────────────────────────────────────────────────

app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>Trade Viewer</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.2/dist/chart.umd.min.js"></script>
<style>
  @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');
  *{box-sizing:border-box;margin:0;padding:0}
  body{background:#0d1117;color:#e6edf3;font-family:'JetBrains Mono',monospace;height:100vh;display:flex;flex-direction:column}
  header{display:flex;align-items:center;gap:16px;padding:12px 20px;background:#161b22;border-bottom:1px solid #30363d;flex-shrink:0}
  header h1{font-size:13px;letter-spacing:.12em;color:#58a6ff;text-transform:uppercase}
  select{background:#0d1117;color:#e6edf3;border:1px solid #30363d;border-radius:4px;padding:6px 12px;font-family:inherit;font-size:12px;cursor:pointer;outline:none}
  select:hover{border-color:#58a6ff}
  #stats{margin-left:auto;display:flex;gap:10px;font-size:11px}
  .pill{padding:3px 10px;border-radius:3px}
  .pill.buy{background:rgba(0,229,160,.15);color:#00e5a0}
  .pill.sell{background:rgba(255,77,106,.15);color:#ff4d6a}
  .pill.mid{background:rgba(245,200,66,.15);color:#f5c842}
  #chart-wrap{flex:1;padding:16px;min-height:0;position:relative}
  canvas{width:100%!important;height:100%!important}
  #loading{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:12px;color:#3d4f60;letter-spacing:.1em}
</style>
</head>
<body>
<header>
  <h1>⬡ Trade Viewer</h1>
  <select id="product-select">
    {% for p in products %}
    <option value="{{p}}">{{p}}</option>
    {% endfor %}
  </select>
  <div id="stats">
    <span class="pill mid" id="stat-mid">–</span>
    <span class="pill buy" id="stat-buy">–</span>
    <span class="pill sell" id="stat-sell">–</span>
  </div>
</header>
<div id="chart-wrap">
  <div id="loading">LOADING...</div>
  <canvas id="chart"></canvas>
</div>

<script>
let chart = null;

const cfg = {
  type: 'scatter',
  data: { datasets: [] },
  options: {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    interaction: { mode: 'nearest', intersect: false },
    plugins: {
      legend: {
        labels: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 11 }, boxWidth: 12 }
      },
      tooltip: {
        backgroundColor: '#161b22',
        borderColor: '#30363d',
        borderWidth: 1,
        titleColor: '#58a6ff',
        bodyColor: '#e6edf3',
        bodyFont: { family: 'JetBrains Mono', size: 11 },
        callbacks: {
          title: items => `t = ${items[0].parsed.x.toLocaleString()}`,
          label: item => {
            const d = item.raw;
            let s = ` price: ${d.y}`;
            if (d.qty  != null) s += `  qty: ${d.qty}`;
            if (d.buyer != null) s += `  ${d.buyer} → ${d.seller}`;
            return s;
          }
        }
      }
    },
    scales: {
      x: {
        ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 },
                 callback: v => v >= 1e6 ? (v/1e6).toFixed(1)+'M' : (v/1e3).toFixed(0)+'k' },
        grid: { color: '#21262d' }
      },
      y: {
        ticks: { color: '#8b949e', font: { family: 'JetBrains Mono', size: 10 } },
        grid: { color: '#21262d' }
      }
    }
  }
};

async function load(product) {
  document.getElementById('loading').style.display = 'flex';
  const res  = await fetch(`/data/${encodeURIComponent(product)}`);
  const data = await res.json();

  const midDs = {
    label: 'Mid price',
    data: data.mid.map(r => ({ x: r.ct, y: r.mid })),
    borderColor: '#f5c842', backgroundColor: 'transparent',
    showLine: true, pointRadius: 0, borderWidth: 1.5, order: 3
  };

  const buyDs = {
    label: `${data.focus_bot} BUY`,
    data: data.buys.map(r => ({ x: r.ct, y: r.price, qty: r.qty, buyer: r.buyer, seller: r.seller })),
    backgroundColor: '#00e5a0', pointRadius: 5, pointHoverRadius: 7, showLine: false, order: 1
  };

  const sellDs = {
    label: `${data.focus_bot} SELL`,
    data: data.sells.map(r => ({ x: r.ct, y: r.price, qty: r.qty, buyer: r.buyer, seller: r.seller })),
    backgroundColor: '#ff4d6a', pointRadius: 5, pointHoverRadius: 7, showLine: false, order: 1
  };

  const otherDs = {
    label: 'Other trades',
    data: data.others.map(r => ({ x: r.ct, y: r.price, qty: r.qty, buyer: r.buyer, seller: r.seller })),
    backgroundColor: 'rgba(139,148,158,0.4)', pointRadius: 3, showLine: false, order: 2
  };

  // day dividers as vertical annotations via a plugin
  const dayLines = {
    id: 'dayLines',
    afterDraw(chart) {
      const {ctx, chartArea, scales} = chart;
      ctx.save();
      ctx.strokeStyle = '#30363d';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      for (const day of [2, 3, 4]) {
        const x = scales.x.getPixelForValue((day - 1) * 1_000_000);
        if (x > chartArea.left && x < chartArea.right) {
          ctx.beginPath(); ctx.moveTo(x, chartArea.top); ctx.lineTo(x, chartArea.bottom); ctx.stroke();
          ctx.fillStyle = '#3d4f60';
          ctx.font = '10px JetBrains Mono';
          ctx.fillText(`Day ${day}`, x + 4, chartArea.top + 14);
        }
      }
      ctx.restore();
    }
  };

  if (chart) chart.destroy();
  cfg.data.datasets = [midDs, buyDs, sellDs, otherDs];
  cfg.plugins = [dayLines];
  chart = new Chart(document.getElementById('chart'), JSON.parse(JSON.stringify(cfg)));
  // re-attach plugin (can't JSON clone functions)
  chart.config.plugins = [dayLines];
  chart.update();

  document.getElementById('stat-mid').textContent  = `mid pts: ${data.mid.length}`;
  document.getElementById('stat-buy').textContent  = `buys: ${data.buys.length}`;
  document.getElementById('stat-sell').textContent = `sells: ${data.sells.length}`;
  document.getElementById('loading').style.display = 'none';
}

document.getElementById('product-select').addEventListener('change', e => load(e.target.value));
load(document.getElementById('product-select').value);
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML, products=PRODUCTS)


@app.route("/data/<product>")
def data(product):
    # mid price
    mid_df = PRICES[PRICES["product"] == product][["ct", "mid_price"]].dropna() \
             if not PRICES.empty else pd.DataFrame()
    mid = [{"ct": int(r.ct), "mid": round(r.mid_price, 4)} for r in mid_df.itertuples()]

    # trades
    tr = TRADES[TRADES["symbol"] == product].dropna(subset=["continuous_time", "price"]) \
         if not TRADES.empty else pd.DataFrame()

    def row_to_dict(r):
        return {
            "ct":     int(r.continuous_time),
            "price":  round(float(r.price), 4),
            "qty":    int(r.quantity) if pd.notna(r.quantity) else None,
            "buyer":  getattr(r, "buyer",  None),
            "seller": getattr(r, "seller", None),
        }

    buys, sells, others = [], [], []
    for r in tr.itertuples():
        buyer  = getattr(r, "buyer",  None)
        seller = getattr(r, "seller", None)
        d = row_to_dict(r)
        if buyer == FOCUS_BOT:
            buys.append(d)
        elif seller == FOCUS_BOT:
            sells.append(d)
        else:
            others.append(d)

    return jsonify({"mid": mid, "buys": buys, "sells": sells,
                    "others": others, "focus_bot": FOCUS_BOT})


if __name__ == "__main__":
    print("Open http://localhost:5000\n")
    app.run(debug=False, port=5000)