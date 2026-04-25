"""
VEV.py
======
Round 3 ML analysis + competition Trader.

PART 1 (offline): Load historical data, engineer features, train Random Forest to find
                  patterns in HYDROGEL_PACK, VELVETFRUIT_EXTRACT, and VEV call options.

PART 2 (competition Trader class): Black-Scholes options market-making, HYDROGEL_PACK
                  mean reversion, VEV underlying tracking.

Black-Scholes calibration (from historical Round 3 data):
  T  = competition days remaining (5.0 at round 3 start, decreases by 1 per day)
  sigma ~= 0.016 per sqrt(competition-day)   [calibrated: VEV_5400@T=5 => C≈23]
  r  = 0

Run `python VEV.py` to perform the offline analysis and print findings.
The Trader class is the competition submission.
"""

import os
import math
import numpy as np
import jsonpickle
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Try competition datamodel (falls back to local Tools path for backtesting)
# ---------------------------------------------------------------------------
try:
    from datamodel import OrderDepth, TradingState, Order
except ImportError:
    try:
        from Tools.Backtest.datamodel import OrderDepth, TradingState, Order
    except ImportError:
        pass  # analysis-only mode

# ---------------------------------------------------------------------------
# Analysis imports (only needed for offline ML section)
# ---------------------------------------------------------------------------
try:
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import StandardScaler
    import warnings
    warnings.filterwarnings("ignore")
    ANALYSIS_AVAILABLE = True
except ImportError:
    ANALYSIS_AVAILABLE = False

# ===========================================================================
# Constants
# ===========================================================================

STRIKES = {
    "VEV_4000": 4000, "VEV_4500": 4500, "VEV_5000": 5000,
    "VEV_5100": 5100, "VEV_5200": 5200, "VEV_5300": 5300,
    "VEV_5400": 5400, "VEV_5500": 5500, "VEV_6000": 6000,
    "VEV_6500": 6500,
}

LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    **{k: 300 for k in STRIKES},
}

UNDERLYING = "VELVETFRUIT_EXTRACT"
HYDROGEL   = "HYDROGEL_PACK"

# TTE at the start of the round 3 simulation
TTE_START = 5.0
# Total ticks per day (observed from historical data: timestamps 0..999900 step 100)
TICKS_PER_DAY = 1_000_000   # timestamp range per day

# Initial vol estimate — calibrated so BS(5250, 5400, T=5) ≈ 23 and BS(5250, 5500, T=5) ≈ 9
INIT_SIGMA = 0.016

# ===========================================================================
# Black-Scholes Utilities  (no scipy — uses math.erfc)
# ===========================================================================

def _norm_cdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2))

def bs_call(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """Black-Scholes European call price."""
    if T <= 1e-9:
        return max(S - K, 0.0)
    sq = sigma * math.sqrt(T)
    if sq < 1e-12:
        return max(S - K * math.exp(-r * T), 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    d2 = d1 - sq
    return S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """BS delta of a call option."""
    if T <= 1e-9:
        return 1.0 if S > K else 0.0
    sq = sigma * math.sqrt(T)
    if sq < 1e-12:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    return _norm_cdf(d1)

def bs_vega(S: float, K: float, T: float, sigma: float, r: float = 0.0) -> float:
    """BS vega (dC/dsigma)."""
    if T <= 1e-9:
        return 0.0
    sq = sigma * math.sqrt(T)
    if sq < 1e-12:
        return 0.0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / sq
    phi = math.exp(-0.5 * d1 ** 2) / math.sqrt(2 * math.pi)
    return S * math.sqrt(T) * phi

def implied_vol(C_market: float, S: float, K: float, T: float, r: float = 0.0,
                tol: float = 1e-6, max_iter: int = 50) -> Optional[float]:
    """Newton-Raphson implied volatility solver. Returns None if solution fails."""
    if C_market <= max(S - K * math.exp(-r * T), 0.0) + 1e-9:
        return None  # price ≤ intrinsic: can't back out a real IV
    sigma = 0.5
    for _ in range(max_iter):
        price = bs_call(S, K, T, sigma, r)
        vega  = bs_vega(S, K, T, sigma, r)
        if abs(vega) < 1e-12:
            return None
        sigma -= (price - C_market) / vega
        if sigma <= 0:
            return None
        if abs(price - C_market) < tol:
            return sigma
    return sigma if 0 < sigma < 10 else None

# ===========================================================================
# PART 1 — Offline ML Analysis
# ===========================================================================

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

PRICE_FILES = [
    os.path.join(DATA_DIR, "prices_round_3_day_0.csv"),
    os.path.join(DATA_DIR, "prices_round_3_day_1.csv"),
    os.path.join(DATA_DIR, "prices_round_3_day_2.csv"),
]

TRADE_FILES = [
    os.path.join(DATA_DIR, "trades_round_3_day_0.csv"),
    os.path.join(DATA_DIR, "trades_round_3_day_1.csv"),
    os.path.join(DATA_DIR, "trades_round_3_day_2.csv"),
]


def load_price_data() -> "pd.DataFrame":
    dfs = []
    for day_idx, f in enumerate(PRICE_FILES):
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f, sep=";")
        df["day_idx"] = day_idx
        # TTE decreases from (TTE_START - day_idx) toward (TTE_START - day_idx - 1)
        df["tte"] = (TTE_START - day_idx) - df["timestamp"] / TICKS_PER_DAY
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _compute_mid(df: "pd.DataFrame") -> "pd.Series":
    bid = df.get("bid_price_1", pd.Series(dtype=float))
    ask = df.get("ask_price_1", pd.Series(dtype=float))
    return (bid + ask) / 2.0


def engineer_features(df_prod: "pd.DataFrame", product: str) -> "pd.DataFrame":
    d = df_prod.sort_values(["day_idx", "timestamp"]).copy()

    d["mid"] = _compute_mid(d)
    d["spread"] = d["ask_price_1"] - d["bid_price_1"]

    bid_vol = d["bid_volume_1"].fillna(0)
    ask_vol = d["ask_volume_1"].fillna(0)
    total_vol = bid_vol + ask_vol
    d["imbalance"] = np.where(total_vol > 0, (bid_vol - ask_vol) / total_vol, 0.0)

    d["log_ret_1"]  = np.log(d["mid"] / d["mid"].shift(1))
    d["mom_5"]      = d["mid"] - d["mid"].shift(5)
    d["mom_20"]     = d["mid"] - d["mid"].shift(20)
    d["mom_100"]    = d["mid"] - d["mid"].shift(100)
    d["vol_20"]     = d["log_ret_1"].rolling(20).std()
    d["vol_100"]    = d["log_ret_1"].rolling(100).std()
    d["sma_100"]    = d["mid"].rolling(100).mean()
    d["sma_500"]    = d["mid"].rolling(500).mean()
    d["dist_mean"]  = (d["mid"] - d["sma_100"]) / (d["vol_20"] * d["mid"] + 1e-9)
    d["spread_rel"] = d["spread"] / d["mid"]

    # ✅ FIX: compute target per day (no leakage)
    d["future_ret_5"] = (
        d.groupby("day_idx")["mid"]
         .shift(-5) / d["mid"] - 1
    )

    # Clean invalid values
    d["future_ret_5"] = d["future_ret_5"].replace([np.inf, -np.inf], np.nan)

    # Drop rows without valid future
    d = d.dropna(subset=["future_ret_5"])

    # Create labels
    d["target_5"] = np.sign(d["future_ret_5"]).astype(int)

    # Remove neutral moves
    d = d[d["target_5"] != 0]

    # Final cleanup
    d = d.replace([np.inf, -np.inf], np.nan)
    return d.dropna()
    
def train_rf(features: "pd.DataFrame", feature_cols: List[str],
             label: str = "target_5", n_splits: int = 3) -> Tuple[float, Dict]:
    """Train RandomForest with TimeSeriesSplit CV. Returns mean accuracy + importances."""
    X = features[feature_cols].values
    y = features[label].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    accs = []
    rf = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=30,
                                 n_jobs=-1, random_state=42)
    for tr_idx, te_idx in tscv.split(X):
        rf.fit(X[tr_idx], y[tr_idx])
        accs.append(accuracy_score(y[te_idx], rf.predict(X[te_idx])))

    rf.fit(X, y)  # final fit on all data
    importances = dict(zip(feature_cols, rf.feature_importances_))
    return float(np.mean(accs)), importances


def analyze_iv_surface(df_all: "pd.DataFrame") -> Dict:
    """Compute average implied vol for each strike across all timestamps."""
    vev_df = df_all[df_all["product"] == UNDERLYING].sort_values(
        ["day_idx", "timestamp"]
    )[["day_idx", "timestamp", "tte", "mid_price"]].rename(columns={"mid_price": "S"})

    results = {}
    for voucher, K in STRIKES.items():
        opt_df = df_all[df_all["product"] == voucher].sort_values(
            ["day_idx", "timestamp"]
        )[["day_idx", "timestamp", "mid_price"]].rename(columns={"mid_price": "C_market"})

        merged = pd.merge(vev_df, opt_df, on=["day_idx", "timestamp"], how="inner")
        merged = merged[merged["tte"] > 0.05]

        ivs = []
        bs_prices = []
        for _, row in merged.iterrows():
            iv = implied_vol(row["C_market"], row["S"], K, row["tte"])
            bp = bs_call(row["S"], K, row["tte"], INIT_SIGMA)
            if iv is not None:
                ivs.append(iv)
            bs_prices.append(bp - row["C_market"])  # positive = market underpriced

        results[voucher] = {
            "K": K,
            "mean_iv": float(np.mean(ivs)) if ivs else None,
            "std_iv":  float(np.std(ivs))  if ivs else None,
            "mean_mispricing": float(np.mean(bs_prices)),  # BS - market
        }
    return results


def run_analysis():
    """Full offline ML analysis. Prints findings to stdout."""
    if not ANALYSIS_AVAILABLE:
        print("pandas / sklearn not available. Install requirements first.")
        return

    print("=" * 60)
    print("VEV.py — Round 3 ML Analysis")
    print("=" * 60)

    print("\nLoading price data...")
    df_all = load_price_data()
    if df_all.empty:
        print("No price data found. Make sure CSV files are in the same directory.")
        return
    print(f"  {len(df_all):,} rows across {df_all['product'].nunique()} products")

    # -----------------------------------------------------------------------
    # 1. HYDROGEL_PACK — mean reversion analysis
    # -----------------------------------------------------------------------
    print("\n--- HYDROGEL_PACK ---")
    hyd_raw = df_all[df_all["product"] == HYDROGEL].copy()
    hyd_features = engineer_features(hyd_raw, HYDROGEL)

    FEATURE_COLS_HYD = ["spread_rel", "imbalance", "mom_5", "mom_20", "mom_100",
                        "vol_20", "vol_100", "dist_mean"]
    acc_hyd, imp_hyd = train_rf(hyd_features, FEATURE_COLS_HYD)
    print(f"  RF direction accuracy (TimeSeriesCV): {acc_hyd:.3f}")
    print("  Feature importances (top 5):")
    for feat, imp in sorted(imp_hyd.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat:<20} {imp:.4f}")

    mid_hyd  = hyd_raw["mid_price"].dropna()
    mean_hyd = float(mid_hyd.mean())
    std_hyd  = float(mid_hyd.std())
    print(f"  Mid-price mean={mean_hyd:.2f}  std={std_hyd:.2f}")

    # Autocorrelation as evidence of mean reversion
    ac1 = float(mid_hyd.diff().autocorr(lag=1))
    print(f"  1-lag price-change autocorrelation: {ac1:.4f}  ({'mean-reverting' if ac1 < -0.05 else 'trending/random'})")

    # -----------------------------------------------------------------------
    # 2. VELVETFRUIT_EXTRACT — directional analysis
    # -----------------------------------------------------------------------
    print("\n--- VELVETFRUIT_EXTRACT ---")
    vev_raw = df_all[df_all["product"] == UNDERLYING].copy()
    vev_features = engineer_features(vev_raw, UNDERLYING)

    FEATURE_COLS_VEV = ["spread_rel", "imbalance", "mom_5", "mom_20", "mom_100",
                        "vol_20", "vol_100", "dist_mean"]
    acc_vev, imp_vev = train_rf(vev_features, FEATURE_COLS_VEV)
    print(f"  RF direction accuracy (TimeSeriesCV): {acc_vev:.3f}")
    print("  Feature importances (top 5):")
    for feat, imp in sorted(imp_vev.items(), key=lambda x: -x[1])[:5]:
        print(f"    {feat:<20} {imp:.4f}")

    mid_vev = vev_raw["mid_price"].dropna()
    print(f"  Mid-price mean={float(mid_vev.mean()):.2f}  std={float(mid_vev.std()):.2f}")
    print(f"  1-lag autocorr: {float(mid_vev.diff().autocorr(lag=1)):.4f}")

    # Realized daily vol in competition-day units (for BS sigma)
    log_rets = np.log(mid_vev / mid_vev.shift(1)).dropna()
    # ticks per day = 10000 (timestamps 0..999900 step 100)
    realized_vol = float(log_rets.std() * math.sqrt(10_000))
    print(f"  Realized vol (per sqrt-competition-day): {realized_vol:.5f}")
    print(f"  (INIT_SIGMA used in BS: {INIT_SIGMA:.4f})")

    # -----------------------------------------------------------------------
    # 3. Implied vol surface and mispricing
    # -----------------------------------------------------------------------
    print("\n--- VEV Options: Implied Vol Surface & Mispricing ---")
    iv_results = analyze_iv_surface(df_all)
    print(f"  {'Voucher':<18} {'Strike':>6}  {'Mean IV':>8}  {'Std IV':>8}  {'BS-mkt bias':>12}")
    for v, res in iv_results.items():
        iv_str  = f"{res['mean_iv']:.5f}"  if res["mean_iv"] else "  N/A   "
        std_str = f"{res['std_iv']:.5f}"   if res["std_iv"]  else "  N/A   "
        print(f"  {v:<18} {res['K']:>6}  {iv_str:>8}  {std_str:>8}  {res['mean_mispricing']:>+12.2f}")

    print("\n  Calibrated sigma recommendation: use mean IV of near-the-money strikes")
    atm = {k: v for k, v in iv_results.items()
           if v["mean_iv"] is not None and abs(v["K"] - 5250) < 400}
    if atm:
        rec_sigma = float(np.mean([v["mean_iv"] for v in atm.values()]))
        print(f"  => sigma = {rec_sigma:.5f} (ATM-average)")

    print("\nAnalysis complete.\n")


# ===========================================================================
# PART 2 — Competition Trader
# ===========================================================================

class Trader:
    """
    Round 3 trading strategy:
      - HYDROGEL_PACK:       mean-reversion market-making around a slow EMA
      - VELVETFRUIT_EXTRACT: mean-reversion + realized-vol tracking for BS sigma
      - VEV_* options:       Black-Scholes fair-value market-making / mispricing capture
    """

    def __init__(self):
        # EMA state
        self.ema: Dict[str, float] = {}
        # Rolling log-return buffer for VEV realized vol
        self.vev_log_rets: List[float] = []
        self.prev_vev_mid: Optional[float] = None
        # Dynamic sigma estimate
        self.sigma: float = INIT_SIGMA
        # Previous mid prices (for autocorr-based signal)
        self.prev_mid: Dict[str, float] = {}

        self._persisted = ["ema", "vev_log_rets", "prev_vev_mid", "sigma", "prev_mid"]

    # ------------------------------------------------------------------
    # State persistence
    # ------------------------------------------------------------------
    def _load(self, trader_data: str):
        if not trader_data:
            return
        try:
            d = jsonpickle.decode(trader_data)
            for k in self._persisted:
                if k in d and d[k] is not None:
                    setattr(self, k, d[k])
        except Exception:
            pass

    def _save(self) -> str:
        return jsonpickle.encode({k: getattr(self, k) for k in self._persisted})

    # ------------------------------------------------------------------
    # Helper: mid price from order depth
    # ------------------------------------------------------------------
    @staticmethod
    def _mid(od: "OrderDepth") -> Optional[float]:
        if od.buy_orders and od.sell_orders:
            return (max(od.buy_orders) + min(od.sell_orders)) / 2.0
        return None

    # ------------------------------------------------------------------
    # EMA updater
    # ------------------------------------------------------------------
    def _update_ema(self, product: str, mid: float, alpha: float) -> float:
        if product not in self.ema:
            self.ema[product] = mid
        else:
            self.ema[product] = alpha * mid + (1.0 - alpha) * self.ema[product]
        return self.ema[product]

    # ------------------------------------------------------------------
    # Realized vol updater (for BS sigma of VEV options)
    # ------------------------------------------------------------------
    def _update_sigma(self, vev_mid: float):
        if self.prev_vev_mid and self.prev_vev_mid > 0:
            lr = math.log(vev_mid / self.prev_vev_mid)
            self.vev_log_rets.append(lr)
            # Keep last 2000 observations (~20% of a day)
            if len(self.vev_log_rets) > 2000:
                self.vev_log_rets = self.vev_log_rets[-2000:]
        self.prev_vev_mid = vev_mid

        if len(self.vev_log_rets) >= 100:
            arr = np.array(self.vev_log_rets)
            realized = float(arr.std()) * math.sqrt(10_000)  # scale to comp-day units
            # Blend with prior: slow update to avoid overreaction
            self.sigma = 0.95 * self.sigma + 0.05 * realized
            # Clamp to sensible range
            self.sigma = max(0.005, min(0.08, self.sigma))

    # ------------------------------------------------------------------
    # TTE at current timestamp (round 3 starts at TTE = 5)
    # ------------------------------------------------------------------
    @staticmethod
    def _tte(timestamp: int) -> float:
        return max(TTE_START - timestamp / TICKS_PER_DAY, 1e-4)

    # ------------------------------------------------------------------
    # Generic "take + make" market-making
    # ------------------------------------------------------------------
    def _trade_product(
        self,
        product: str,
        fair: float,
        od: "OrderDepth",
        position: int,
        take_edge: float,
        make_edge: float,
        limit: int,
        orders: List["Order"],
    ):
        """
        Aggressive takes if market price crosses fair ± take_edge.
        Passive makes quoted at fair ± make_edge (rounded to int).
        """
        # --- Aggressive takes ---
        if od.sell_orders:
            best_ask = min(od.sell_orders)
            ask_qty  = -od.sell_orders[best_ask]  # positive volume
            if best_ask < fair - take_edge:
                buy_qty = min(ask_qty, limit - position)
                if buy_qty > 0:
                    orders.append(Order(product, best_ask, buy_qty))
                    position += buy_qty

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            bid_qty  = od.buy_orders[best_bid]   # positive volume
            if best_bid > fair + take_edge:
                sell_qty = min(bid_qty, limit + position)
                if sell_qty > 0:
                    orders.append(Order(product, best_bid, -sell_qty))
                    position -= sell_qty

        # --- Passive makes ---
        make_bid = math.floor(fair - make_edge)
        make_ask = math.ceil(fair + make_edge)

        if position < limit:
            orders.append(Order(product, make_bid, min(10, limit - position)))
        if position > -limit:
            orders.append(Order(product, make_ask, -min(10, limit + position)))

    # ------------------------------------------------------------------
    # HYDROGEL_PACK strategy
    # ------------------------------------------------------------------
    def _trade_hydrogel(self, state: "TradingState", orders: Dict) -> None:
        od = state.order_depths.get(HYDROGEL)
        if not od:
            return
        mid = self._mid(od)
        if mid is None:
            return

        ema  = self._update_ema(HYDROGEL, mid, alpha=0.05)
        pos  = state.position.get(HYDROGEL, 0)
        lim  = LIMITS[HYDROGEL]

        result: List["Order"] = []
        # Mean-reversion: larger size when deviation is larger
        deviation = mid - ema
        take_edge = 6.0
        make_edge = 3.0

        # Skew passive quotes toward mean reversion
        skew = int(np.clip(deviation / 3, -5, 5))

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            if best_ask < ema - take_edge:
                qty = min(-od.sell_orders[best_ask], lim - pos)
                if qty > 0:
                    result.append(Order(HYDROGEL, best_ask, qty))
                    pos += qty

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            if best_bid > ema + take_edge:
                qty = min(od.buy_orders[best_bid], lim + pos)
                if qty > 0:
                    result.append(Order(HYDROGEL, best_bid, -qty))
                    pos -= qty

        bid_price = math.floor(ema - make_edge - skew)
        ask_price = math.ceil(ema + make_edge - skew)
        if bid_price < ask_price:
            if pos < lim:
                result.append(Order(HYDROGEL, bid_price, min(20, lim - pos)))
            if pos > -lim:
                result.append(Order(HYDROGEL, ask_price, -min(20, lim + pos)))

        orders[HYDROGEL] = result

    # ------------------------------------------------------------------
    # VELVETFRUIT_EXTRACT strategy
    # ------------------------------------------------------------------
    def _trade_underlying(self, state: "TradingState", orders: Dict) -> None:
        od = state.order_depths.get(UNDERLYING)
        if not od:
            return
        mid = self._mid(od)
        if mid is None:
            return

        self._update_sigma(mid)
        ema  = self._update_ema(UNDERLYING, mid, alpha=0.02)
        pos  = state.position.get(UNDERLYING, 0)
        lim  = LIMITS[UNDERLYING]

        result: List["Order"] = []
        take_edge = 4.0
        make_edge = 2.0
        deviation = mid - ema
        skew = int(np.clip(deviation / 2, -4, 4))

        if od.sell_orders:
            best_ask = min(od.sell_orders)
            if best_ask < ema - take_edge:
                qty = min(-od.sell_orders[best_ask], lim - pos)
                if qty > 0:
                    result.append(Order(UNDERLYING, best_ask, qty))
                    pos += qty

        if od.buy_orders:
            best_bid = max(od.buy_orders)
            if best_bid > ema + take_edge:
                qty = min(od.buy_orders[best_bid], lim + pos)
                if qty > 0:
                    result.append(Order(UNDERLYING, best_bid, -qty))
                    pos -= qty

        bid_price = math.floor(ema - make_edge - skew)
        ask_price = math.ceil(ema + make_edge - skew)
        if bid_price < ask_price:
            if pos < lim:
                result.append(Order(UNDERLYING, bid_price, min(15, lim - pos)))
            if pos > -lim:
                result.append(Order(UNDERLYING, ask_price, -min(15, lim + pos)))

        orders[UNDERLYING] = result

    # ------------------------------------------------------------------
    # VEV options strategy
    # ------------------------------------------------------------------
    def _trade_options(self, state: "TradingState", vev_mid: float,
                       tte: float, orders: Dict) -> None:
        """
        For each voucher: compute BS fair value, take mispriced orders,
        and post passive quotes at ±make_edge around fair value.
        """
        for voucher, K in STRIKES.items():
            od = state.order_depths.get(voucher)
            if not od:
                continue

            fair = bs_call(vev_mid, K, tte, self.sigma)
            pos  = state.position.get(voucher, 0)
            lim  = LIMITS[voucher]

            # Near expiry: don't make markets, only clear
            if tte < 0.15:
                result: List["Order"] = []
                if pos > 0 and od.buy_orders:
                    best_bid = max(od.buy_orders)
                    result.append(Order(voucher, best_bid, -pos))
                elif pos < 0 and od.sell_orders:
                    best_ask = min(od.sell_orders)
                    result.append(Order(voucher, best_ask, -pos))
                if result:
                    orders[voucher] = result
                continue

            # For deep OTM (fair < 0.5): skip, too illiquid/noisy
            if fair < 0.5:
                continue

            take_edge = max(0.5, fair * 0.01)  # at least 0.5, or 1% of fair
            make_edge = max(0.3, fair * 0.006)

            result = []

            # Aggressive: buy cheap options
            if od.sell_orders:
                best_ask = min(od.sell_orders)
                if best_ask < fair - take_edge:
                    qty = min(-od.sell_orders[best_ask], lim - pos)
                    if qty > 0:
                        result.append(Order(voucher, best_ask, qty))
                        pos += qty

            # Aggressive: sell expensive options
            if od.buy_orders:
                best_bid = max(od.buy_orders)
                if best_bid > fair + take_edge:
                    qty = min(od.buy_orders[best_bid], lim + pos)
                    if qty > 0:
                        result.append(Order(voucher, best_bid, -qty))
                        pos -= qty

            # Passive quotes
            bid_px = max(1, math.floor(fair - make_edge))
            ask_px = math.ceil(fair + make_edge)
            size   = min(10, max(1, int(lim * 0.05)))

            if pos < lim:
                result.append(Order(voucher, bid_px, min(size, lim - pos)))
            if pos > -lim:
                result.append(Order(voucher, ask_px, -min(size, lim + pos)))

            if result:
                orders[voucher] = result

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------
    def run(self, state: "TradingState"):
        self._load(state.traderData)

        orders: Dict[str, List["Order"]] = {}
        tte = self._tte(state.timestamp)

        # 1. Trade HYDROGEL_PACK
        self._trade_hydrogel(state, orders)

        # 2. Trade VEV underlying (also updates sigma)
        self._trade_underlying(state, orders)

        # 3. Trade VEV options (need current VEV mid for BS)
        vev_od  = state.order_depths.get(UNDERLYING)
        vev_mid = self._mid(vev_od) if vev_od else self.ema.get(UNDERLYING, 5250.0)
        if vev_mid:
            self._trade_options(state, vev_mid, tte, orders)

        return orders, 0, self._save()


# ===========================================================================
# Entry point — run offline analysis
# ===========================================================================
if __name__ == "__main__":
    run_analysis()
