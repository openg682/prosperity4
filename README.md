# Prosperity Trading Codebase

Algorithmic trading strategies and tooling for the IMC Prosperity simulation platform.

---

## Folder Structure

```
prosperity4/
├── Files/
│   ├── Prices/          # Raw price CSVs from the exchange
│   └── Trades/          # Raw trade CSVs from the exchange
│
├── Tools/
│   ├── backtest/        # Backtester and strategy wrapper
│   └── data_visualisation/  # Interactive Dash dashboards
│   └── data_analysis/   # Parameter calibration and optimisation tools
│
├── Optimisation/        # Isolated per-product/strategy files for tuning
├── Testing/             # Combined strategy files for full simulation runs
└── Final Submissions/   # One file per round — best performer only
```

---

## Setup

**Requirements**

```bash
pip install pandas numpy matplotlib dash plotly (all mentioned in requirements.txt)
```

Python 3.11+ recommended. All simulation platform files are limited to the standard library + `pandas`, `numpy`, `statistics`, `math`, `typing`, and `jsonpickle`.

---

## Files

### `Files/Prices/`
Price CSVs from the exchange, named `prices_round_{r}_day_{d}.csv`. Each row is one tick for one product, with columns:

| Column | Description |
|---|---|
| `day` | Day number |
| `timestamp` | Tick timestamp |
| `product` | Product name |
| `bid_price_1/2/3` | Best bid prices |
| `bid_volume_1/2/3` | Bid volumes at each level |
| `ask_price_1/2/3` | Best ask prices |
| `ask_volume_1/2/3` | Ask volumes at each level |
| `mid_price` | Mid price |

### `Files/Trades/`
Trade CSVs from the exchange, named `trades_round_{r}_day_{d}.csv`. Each row is one executed trade:

| Column | Description |
|---|---|
| `timestamp` | Tick timestamp |
| `symbol` | Product name |
| `price` | Trade price |
| `quantity` | Trade quantity |
| `buyer` | Buyer identifier (`SUBMISSION` = your algo) |
| `seller` | Seller identifier (`SUBMISSION` = your algo) |

---

## Tools

### Backtester — `Tools/backtest/backtester.py`

Simulates strategy execution against historical price and trade data. Supports both aggressive fills (hitting the book) and passive fills (resting orders matched against historical trades).

**Run from `backtest/`:**
```bash
python backtester.py
```

**Run with a custom strategy:**
```bash
python backtester.py wrapper.py:totest.py
```

**Outputs:**
- `Tools/backtest/backtest_output.csv` — tick-by-tick PnL and position history
- `Tools/backtest/backtest_results.png` — PnL and position chart

**Position limits:** 80 for all products. Exceeding this silently rejects orders.

**Built-in strategies:**
- `combined` — runs both EmeraldsMarketMaker and TomatoesEMAMarketMaker together
- `emeralds` — market making on EMERALDS with fixed fair value
- `tomatoes` — market making on TOMATOES with EMA fair value tracking

---

### `Tools/backtest/wrapper.py`

Wraps any Prosperity-format `Trader` class so it can run inside the backtester without modification. Translates between the backtester's `OrderBook` format and Prosperity's `TradingState`/`OrderDepth` format. Also fixes a common passive order bug where `market_make()` orders are silently dropped.

### `Tools/backtest/datamodel.py`

Minimal reimplementation of Prosperity's `datamodel` — provides `OrderDepth`, `TradingState`, and `Order` classes so strategy files import correctly outside the platform.

---

### Data Visualisation — `Tools/data_visualisation/`

All tools are interactive Dash apps. Run from `Tools/data_visualisation/`. Each opens in the browser on a different port.

#### `forprices.py` — Port 8050
Visualises price data per product and file.

| Chart | Description |
|---|---|
| Mid price | Price over time |
| Bid-ask spread | Spread width over time |
| Rolling volatility | Price volatility (window = 20 ticks) |
| Order book depth | Total bid/ask volume mirrored over time |

```bash
python forprices.py
```

#### `fortrades.py` — Port 8051
Visualises trade data per product and file.

| Chart | Description |
|---|---|
| Trade price vs mid price | Buys/sells overlaid on mid price |
| Volume over time | Per-trade quantity as bars |
| Rolling avg trade size | Smoothed trade size |
| Buy vs sell volume | Total directional breakdown |
| Trade frequency | Trades per timestamp |
| Cumulative volume | Running total volume |

```bash
python fortrades.py
```

#### `combined.py` — Port 8052
Cross-analysis combining backtest output, price, and trade data. Requires running the backtester first.

| Chart | Description |
|---|---|
| PnL over time | From `backtest_output.csv` |
| Position history vs price | Position overlaid on mid price per product |
| PnL attribution | Estimated per-product contribution to PnL |
| Spread vs trade frequency | Scatter — how spread width correlates with activity |

```bash
python combined.py
```

#### `orderbook.py` — Port 8054
Order book microstructure analysis.

| Chart | Description |
|---|---|
| Depth heatmap | Bid/ask volume at every price level over time. Green = bids, red = asks, yellow = mid price |
| Imbalance | (bid vol − ask vol) / total vol. Positive = buying pressure |
| Spread regime | Spread width vs rolling volatility on dual axis |
| Queue depth | Volume sitting at best bid/ask |
| Large order detector | Flags orders >2σ above mean volume |

```bash
python orderbook.py
```

---

### Data Analysis — `Tools/data_analysis/`

#### `param_optimisation.py` — Port 8053
Interactive parameter sweep tool. Re-runs the backtester for every parameter combination and scores by PnL.

**Products:**
- `EMERALDS` — sweeps `default_edge` × `take_width`
- `TOMATOES` — sweeps `default_edge` × `alpha`

**Modes:**
- **All Data** — scores on all files combined. Fast but prone to overfitting.
- **Cross Validation** — for each combination, holds out each file in turn as the test set and scores on the held-out data. Shows three heatmaps: in-sample, out-of-sample, and robustness (worst out-of-sample fold). Pick parameters that score well on all three.

```bash
python param_calibration.py
```

**Tip:** When two parameter combinations give equal PnL, prefer the smaller `default_edge` and lower `alpha` — they are more conservative and more likely to generalise to unseen data.

---

## Strategy Files

### `Optimisation/`
One file per product or strategy. Used for isolated tuning before combining. Examples:
- `market_making_emeralds.py`
- `market_making_tomatoes.py`
- `arb_etf_basket.py`

### `Testing/`
Combined files that merge optimised strategies together for a full simulation run. Run through the backtester to get a holistic PnL figure before submitting.

### `Final Submissions/`
One file per round. Keep only the best-performing version. Submitted directly to the IMC Prosperity platform.

---

## Key Concepts

### Fair Value
The price your algorithm believes the product is worth. Orders are placed around this value.

- **EMERALDS** — stable ~10000. Uses a fixed fair value.
- **TOMATOES** — volatile, drifts significantly. Uses an EMA of mid price to track fair value dynamically.

### EMA (Exponential Moving Average)
```
ema = alpha * current_price + (1 - alpha) * previous_ema
```
- High alpha → reacts faster to new prices, noisier
- Low alpha → smoother, slower to track genuine drift
- When in doubt, prefer lower alpha — it generalises better

### Position Limits
Hard limit of **80 units** per product in both directions. Orders that would exceed this are silently rejected by the exchange. The backtester enforces the same limit.

### Edge / Spread
The distance from fair value at which you place your passive quotes. Tighter edge = more fills but more risk. Wider edge = fewer fills but better PnL per fill. Use `param_calibration.py` to find the optimal value.

### Aggressive vs Passive Fills
- **Aggressive** — you hit an existing order in the book immediately
- **Passive** — you rest a quote and wait for someone to fill you

The backtester simulates both. Passive fills are matched against historical trades — if a trade occurred at a price that crosses your resting order, you fill.

---

## Workflow

```
1. Explore data         →  forprices.py, fortrades.py, orderbook.py
2. Develop strategy     →  Optimisation/
3. Tune parameters      →  param_calibration.py (cross validation mode)
4. Combine strategies   →  Testing/
5. Run full backtest    →  backtester.py
6. Analyse results      →  combined.py
7. Submit               →  Final Submissions/
```