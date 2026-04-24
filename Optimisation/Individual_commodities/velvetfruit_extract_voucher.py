"""
vouchers_trader.py - Self-contained voucher strategy (all 10 strikes).

Submit this file directly to the platform. No external imports beyond datamodel.

Combines two voucher strategies that share a rolling fair-IV state:

  1. NEAR-MONEY VOL ARB  (K in {5000, 5100, 5200, 5300, 5400, 5500})
     - Compute current per-strike IV, update slow rolling-average ('fair IV')
     - For each voucher: BS-fair price = bs_call(S, K, T, fair_IV)
     - TAKE if voucher mid deviates from fair by >= price_edge_take
     - Naturally accumulates LONG K=5400 (-1.16 IV pts persistent) and
       SHORT K=5500 (+0.79 IV pts persistent)

  2. SYNTH-FORWARD ARB  (K in {4000, 4500})
     - Deep ITM, trade at intrinsic (S - K) +/- ~1 tick noise
     - TAKE when |mid - (S - K)| >= edge_threshold (~near arb, delta=1)

NOT TRADED:
    VEV_6000, VEV_6500 are stuck at the 0.5 floor (worthless OTM)

PARAMS to tune:
    NEAR:
      - fair_iv_init        (calibrated 0.241 from history)
      - fair_iv_alpha       smaller = stickier IV anchor
      - price_edge_take     larger = fewer trades, more PnL/trade
      - max_take_size       per-tick take cap per voucher
    SYNTH:
      - edge_threshold      deviation from intrinsic to trigger
      - max_take_size

TIME-TO-EXPIRY:
    START_TTE_DAYS = 5 for live round 3 submission.
    Change to 8 for backtesting historical CSVs (which start at TTE=8).
"""

import json
import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


# ============================================================================
# PARAMETERS  (tune these)
# ============================================================================
NEAR_PARAMS = {
    "fair_iv_init":     0.241,
    "fair_iv_alpha":    0.001,
    "trade_strikes":    [5000, 5100, 5200, 5300, 5400, 5500],
    "price_edge_take":  1.0,
    "max_take_size":    60,
    "iv_min":           0.10,
    "iv_max":           0.60,
}

SYNTH_PARAMS = {
    "strikes":          [4000, 4500],
    "edge_threshold":   2.0,
    "max_take_size":    40,
}

UNDERLYING = "VELVETFRUIT_EXTRACT"
VOUCHER_LIMIT = 300

# Time-to-expiry config — flip to 8 for historical CSV backtests
START_TTE_DAYS  = 5
TS_PER_DAY      = 1_000_000
DAYS_PER_YEAR   = 365.0
TTE_FLOOR_DAYS  = 0.5


# ============================================================================
# BLACK-SCHOLES HELPERS  (inlined)
# ============================================================================
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _norm_cdf(d1)

def bs_vega(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0:
        return 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return S * _norm_pdf(d1) * sqrtT

def implied_vol(price: float, S: float, K: float, T: float,
                sigma0: float = 0.24):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or price >= S - 1e-6:
        return None
    sigma = sigma0
    for _ in range(50):
        diff = bs_call(S, K, T, sigma) - price
        if abs(diff) < 1e-5:
            return sigma
        v = bs_vega(S, K, T, sigma)
        if v < 1e-8:
            return None
        sigma -= diff / v
        if sigma <= 1e-5 or sigma > 5.0:
            return None
    return sigma


# ============================================================================
# TRADER
# ============================================================================
class Trader:
    def __init__(self):
        self._inited = False

    def _init_state(self, trader_data: str):
        self.fair_iv: float = NEAR_PARAMS["fair_iv_init"]
        self.day_count: int = 0
        self.last_ts: int = -1
        if trader_data:
            try:
                d = json.loads(trader_data)
                self.fair_iv   = d.get("fair_iv", NEAR_PARAMS["fair_iv_init"])
                self.day_count = d.get("day_count", 0)
                self.last_ts   = d.get("last_ts", -1)
            except Exception:
                pass
        self._inited = True

    def _serialize_state(self) -> str:
        return json.dumps({
            "fair_iv":   self.fair_iv,
            "day_count": self.day_count,
            "last_ts":   self.last_ts,
        })

    @staticmethod
    def _mid(od: OrderDepth):
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    def _trade_near(self, state: TradingState, result: Dict[str, List[Order]],
                    T_years: float):
        if UNDERLYING not in state.order_depths:
            return
        S = self._mid(state.order_depths[UNDERLYING])
        if S is None:
            return

        # 1) update rolling fair IV
        observed = []
        for K in NEAR_PARAMS["trade_strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            m = self._mid(state.order_depths[sym])
            if m is None:
                continue
            iv = implied_vol(m, S, float(K), T_years, sigma0=self.fair_iv)
            if iv is not None and NEAR_PARAMS["iv_min"] < iv < NEAR_PARAMS["iv_max"]:
                observed.append(iv)
        if observed:
            avg_iv = sum(observed) / len(observed)
            a = NEAR_PARAMS["fair_iv_alpha"]
            self.fair_iv = a * avg_iv + (1.0 - a) * self.fair_iv

        # 2) trade each near-money voucher vs BS-fair
        for K in NEAR_PARAMS["trade_strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders:
                continue
            b = max(od.buy_orders.keys())
            a = min(od.sell_orders.keys())

            fair_px = bs_call(S, float(K), T_years, self.fair_iv)
            edge = NEAR_PARAMS["price_edge_take"]
            pos = state.position.get(sym, 0)
            max_take = NEAR_PARAMS["max_take_size"]
            eff = pos

            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px > fair_px - edge:
                    break
                avail = min(VOUCHER_LIMIT - eff, max_take - max(eff - pos, 0))
                if avail <= 0:
                    break
                qty = min(-od.sell_orders[ask_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, ask_px, qty))
                    eff += qty

            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px < fair_px + edge:
                    break
                avail = min(VOUCHER_LIMIT + eff, max_take - max(pos - eff, 0))
                if avail <= 0:
                    break
                qty = min(od.buy_orders[bid_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, bid_px, -qty))
                    eff -= qty

    def _trade_synth(self, state: TradingState, result: Dict[str, List[Order]]):
        if UNDERLYING not in state.order_depths:
            return
        S = self._mid(state.order_depths[UNDERLYING])
        if S is None:
            return

        for K in SYNTH_PARAMS["strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            if not od.buy_orders or not od.sell_orders:
                continue

            fair_px = S - K
            edge = SYNTH_PARAMS["edge_threshold"]
            pos = state.position.get(sym, 0)
            max_take = SYNTH_PARAMS["max_take_size"]
            eff = pos

            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px > fair_px - edge:
                    break
                avail = min(VOUCHER_LIMIT - eff, max_take - max(eff - pos, 0))
                if avail <= 0:
                    break
                qty = min(-od.sell_orders[ask_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, ask_px, qty))
                    eff += qty

            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px < fair_px + edge:
                    break
                avail = min(VOUCHER_LIMIT + eff, max_take - max(pos - eff, 0))
                if avail <= 0:
                    break
                qty = min(od.buy_orders[bid_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, bid_px, -qty))
                    eff -= qty

    def run(self, state: TradingState):
        if not self._inited:
            self._init_state(getattr(state, "traderData", "") or "")

        # day boundary detection (timestamp wraps to 0 at start of new day)
        if self.last_ts >= 0 and state.timestamp < self.last_ts:
            self.day_count += 1
        self.last_ts = state.timestamp

        elapsed_days = self.day_count + state.timestamp / TS_PER_DAY
        TTE_days = max(START_TTE_DAYS - elapsed_days, TTE_FLOOR_DAYS)
        T_years = TTE_days / DAYS_PER_YEAR

        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}
        self._trade_synth(state, result)
        self._trade_near(state, result, T_years)

        return result, 0, self._serialize_state()