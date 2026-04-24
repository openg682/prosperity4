"""
Round 3 trading strategy
========================

Universe (12 products):
  - HYDROGEL_PACK         (delta-1, MM target: stable mean-reverting around 9990, 16-wide spread)
  - VELVETFRUIT_EXTRACT   (delta-1, the option underlying: mean ~5250, 5-wide spread)
  - VEV_{K} for K in {4000,4500,5000,5100,5200,5300,5400,5500,6000,6500}  (calls on VELVETFRUIT)

Key edges identified from 3 days of historical price data
---------------------------------------------------------
1. HYDROGEL_PACK has a ~16-wide best-bid/ask spread with weak mean reversion
   (lag-1 autocorr -0.13). Naive MM at bid+1/ask-1 yields ~13.7 PnL per round
   trip with 73% win rate. THIS IS THE LARGEST RELIABLE PNL DRIVER.

2. The voucher implied-vol smile is essentially flat at ~24.1% IV, with two
   PERSISTENT outliers across all 3 days:
     - K=5400 trades ~1.16 IV pts BELOW the smile (cheap)  -> bias LONG
     - K=5500 trades ~0.79 IV pts ABOVE the smile (rich)   -> bias SHORT
   We trade vouchers when their BS-fair price (using rolling avg IV) deviates
   from observed mid by more than `price_edge_take`.

3. VEV_4000 and VEV_4500 are deep-ITM and trade essentially at intrinsic
   (S - K, with std ~0.8). Whenever (mid + K - S) deviates by more than
   ~2 ticks we have a near-arb (these have delta=1, so the hedge is just spot).

4. Realized vol (41%) is much higher than implied (24%), so being long
   gamma is +EV in expectation. The vol-arb above naturally captures some
   of this when we end up net long the cheap vouchers.

Position management:
  - Limits enforced by tracking effective_pos after each order placement
  - Inventory skew on HYDROGEL/VELVET passive quotes to stay near flat
  - Voucher net delta is computed and used to bias spot MM toward hedging

State persistence:
  - EMA of mid prices and the rolling fair IV are serialized via traderData
  - day_count is auto-incremented when state.timestamp wraps back to 0
"""

import json
import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


# ============================================================================
# POSITION LIMITS (from round 3 wiki)
# ============================================================================
LIMITS: Dict[str, int] = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
}
VEV_STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]
for _K in VEV_STRIKES:
    LIMITS[f"VEV_{_K}"] = 300


# ============================================================================
# TIME / EXPIRY
# ============================================================================
# At the start of the LIVE round 3 submission, vouchers have TTE = 5 days.
# For BACKTESTING historical CSVs (days 0/1/2 = TTE 8/7/6 at start of each),
# change START_TTE_DAYS to 8.
START_TTE_DAYS = 5         # <-- 5 for live round 3 submission, 8 for historical backtest
TS_PER_DAY = 1_000_000
DAYS_PER_YEAR = 365.0
TTE_FLOOR_DAYS = 0.5       # never let T -> 0 (avoids div-by-zero in BS)


# ============================================================================
# STRATEGY PARAMETERS  (tune these)
# ============================================================================
HYDROGEL_PARAMS = {
    "ema_alpha":        0.005,   # very smooth - mean is rock stable
    "take_edge":        2,       # take if (fair - ask) >= 2 (or symmetric for hits)
    "make_inside":      1,       # post passive 1 tick inside best b/a
    "make_size":        40,      # max passive size posted per side
    "skew_per_unit":    50,      # at +/-50 inv, 1 tick of quote skew
}

VELVET_PARAMS = {
    "ema_alpha":        0.04,    # faster - it drifts mildly
    "take_edge":        1,
    "make_inside":      1,
    "make_size":        30,
    "skew_per_unit":    80,
    "hedge_aggressiveness": 0.5, # fraction of voucher delta to actively hedge per tick
}

VOUCHER_PARAMS = {
    "fair_iv_init":     0.241,   # initial smile-average IV (calibrated from historical data)
    "fair_iv_alpha":    0.001,   # slow update of rolling fair IV
    "trade_strikes":    [5000, 5100, 5200, 5300, 5400, 5500],   # near-money vouchers we actively trade
    "price_edge_take":  1.0,     # take if mid-fair deviation >= 1.0 in price units
    "max_take_size":    60,      # max take size per voucher per tick
    "iv_min":           0.10,    # ignore IV readings outside this band
    "iv_max":           0.60,
}

SYNTH_PARAMS = {
    "strikes":          [4000, 4500],   # deep-ITM vouchers
    "edge_threshold":   2.0,            # take if |mid - (S - K)| >= 2.0
    "max_take_size":    40,
}


# ============================================================================
# BLACK-SCHOLES HELPERS
# ============================================================================
_SQRT_2 = math.sqrt(2.0)
_SQRT_2PI = math.sqrt(2.0 * math.pi)

def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / _SQRT_2))

def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    """European call price (zero rates, zero divs)."""
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

def implied_vol(price: float, S: float, K: float, T: float, sigma0: float = 0.24) -> float:
    """Newton-Raphson implied vol. Returns None on failure."""
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

    # ---- state persistence ----
    def _init_state(self, trader_data: str):
        self.ema: Dict[str, float] = {}
        self.fair_iv: float = VOUCHER_PARAMS["fair_iv_init"]
        self.day_count: int = 0
        self.last_ts: int = -1
        if trader_data:
            try:
                d = json.loads(trader_data)
                self.ema = d.get("ema", {})
                self.fair_iv = d.get("fair_iv", VOUCHER_PARAMS["fair_iv_init"])
                self.day_count = d.get("day_count", 0)
                self.last_ts = d.get("last_ts", -1)
            except Exception:
                pass
        self._inited = True

    def _serialize_state(self) -> str:
        return json.dumps({
            "ema": self.ema,
            "fair_iv": self.fair_iv,
            "day_count": self.day_count,
            "last_ts": self.last_ts,
        })

    # ---- generic helpers ----
    def _update_ema(self, prod: str, value: float, alpha: float) -> float:
        if prod not in self.ema:
            self.ema[prod] = value
        else:
            self.ema[prod] = alpha * value + (1.0 - alpha) * self.ema[prod]
        return self.ema[prod]

    @staticmethod
    def _best_bid_ask(od: OrderDepth):
        bid = max(od.buy_orders.keys()) if od.buy_orders else None
        ask = min(od.sell_orders.keys()) if od.sell_orders else None
        return bid, ask

    @staticmethod
    def _mid(od: OrderDepth):
        b, a = Trader._best_bid_ask(od)
        if b is None or a is None:
            return None
        return (b + a) / 2.0

    # =========================================================================
    # HYDROGEL_PACK : stable mean-reverting MM
    # =========================================================================
    def _trade_hydrogel(self, state: TradingState, result: Dict[str, List[Order]]):
        prod = "HYDROGEL_PACK"
        if prod not in state.order_depths:
            return
        od = state.order_depths[prod]
        b, a = self._best_bid_ask(od)
        if b is None or a is None:
            return
        mid = (b + a) / 2.0
        fair = self._update_ema(prod, mid, HYDROGEL_PARAMS["ema_alpha"])

        pos = state.position.get(prod, 0)
        limit = LIMITS[prod]
        take_edge = HYDROGEL_PARAMS["take_edge"]
        eff = pos  # effective position after orders we place

        # === TAKE: lift cheap asks ===
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px > fair - take_edge:
                break
            avail = limit - eff
            if avail <= 0:
                break
            qty = min(-od.sell_orders[ask_px], avail)
            if qty > 0:
                result[prod].append(Order(prod, ask_px, qty))
                eff += qty

        # === TAKE: hit pricey bids ===
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px < fair + take_edge:
                break
            avail = limit + eff
            if avail <= 0:
                break
            qty = min(od.buy_orders[bid_px], avail)
            if qty > 0:
                result[prod].append(Order(prod, bid_px, -qty))
                eff -= qty

        # === MAKE: post passive 1 tick inside best b/a, skewed by inventory ===
        inside = HYDROGEL_PARAMS["make_inside"]
        # don't cross the inside spread
        make_bid = b + inside
        make_ask = a - inside
        if make_bid >= make_ask:
            # spread too tight - bracket fair
            make_bid = int(math.floor(fair)) - 2
            make_ask = int(math.ceil(fair)) + 2
        # inventory skew: positive pos -> shift quotes down to encourage selling
        skew = -int(round(eff / HYDROGEL_PARAMS["skew_per_unit"]))
        make_bid += skew
        make_ask += skew
        if make_bid >= make_ask:
            return

        size = HYDROGEL_PARAMS["make_size"]
        buy_room = limit - eff
        sell_room = limit + eff
        if buy_room > 0:
            result[prod].append(Order(prod, int(make_bid), min(size, buy_room)))
        if sell_room > 0:
            result[prod].append(Order(prod, int(make_ask), -min(size, sell_room)))

    # =========================================================================
    # VELVETFRUIT_EXTRACT : MM + opportunistic delta hedge
    # =========================================================================
    def _trade_velvetfruit(self, state: TradingState, result: Dict[str, List[Order]],
                           voucher_net_delta: float):
        prod = "VELVETFRUIT_EXTRACT"
        if prod not in state.order_depths:
            return
        od = state.order_depths[prod]
        b, a = self._best_bid_ask(od)
        if b is None or a is None:
            return
        mid = (b + a) / 2.0
        fair = self._update_ema(prod, mid, VELVET_PARAMS["ema_alpha"])

        pos = state.position.get(prod, 0)
        limit = LIMITS[prod]
        take_edge = VELVET_PARAMS["take_edge"]
        eff = pos

        # Effective desired position incorporates delta hedge:
        # voucher_net_delta is signed delta from voucher book; we want spot to
        # offset some fraction of it. target_spot = -voucher_net_delta.
        target_spot = -int(round(voucher_net_delta * VELVET_PARAMS["hedge_aggressiveness"]))
        target_spot = max(-limit, min(limit, target_spot))

        # === TAKE: lift cheap asks ===
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px > fair - take_edge:
                break
            avail = limit - eff
            if avail <= 0:
                break
            qty = min(-od.sell_orders[ask_px], avail)
            if qty > 0:
                result[prod].append(Order(prod, ask_px, qty))
                eff += qty

        # === TAKE: hit pricey bids ===
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px < fair + take_edge:
                break
            avail = limit + eff
            if avail <= 0:
                break
            qty = min(od.buy_orders[bid_px], avail)
            if qty > 0:
                result[prod].append(Order(prod, bid_px, -qty))
                eff -= qty

        # === MAKE: passive inside spread, skewed by (eff - target_spot) ===
        inside = VELVET_PARAMS["make_inside"]
        make_bid = b + inside
        make_ask = a - inside
        if make_bid >= make_ask:
            make_bid = int(math.floor(fair)) - 2
            make_ask = int(math.ceil(fair)) + 2
        # skew so we drift toward target_spot, not zero
        deviation_from_target = eff - target_spot
        skew = -int(round(deviation_from_target / VELVET_PARAMS["skew_per_unit"]))
        make_bid += skew
        make_ask += skew
        if make_bid >= make_ask:
            return

        size = VELVET_PARAMS["make_size"]
        buy_room = limit - eff
        sell_room = limit + eff
        if buy_room > 0:
            result[prod].append(Order(prod, int(make_bid), min(size, buy_room)))
        if sell_room > 0:
            result[prod].append(Order(prod, int(make_ask), -min(size, sell_room)))

    # =========================================================================
    # NEAR-MONEY VOUCHERS : vol smile arb (TAKE only)
    # =========================================================================
    def _trade_near_vouchers(self, state: TradingState, result: Dict[str, List[Order]],
                             T_years: float) -> float:
        """
        Returns net voucher portfolio delta (in units of underlying), so the
        spot MM can hedge.
        """
        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return 0.0
        S = self._mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        if S is None:
            return 0.0

        # 1) Compute current per-strike IVs and update rolling smile-average IV
        observed_ivs = {}
        for K in VOUCHER_PARAMS["trade_strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            m = self._mid(state.order_depths[sym])
            if m is None:
                continue
            iv = implied_vol(m, S, float(K), T_years, sigma0=self.fair_iv)
            if iv is not None and VOUCHER_PARAMS["iv_min"] < iv < VOUCHER_PARAMS["iv_max"]:
                observed_ivs[K] = iv

        if observed_ivs:
            avg_iv = sum(observed_ivs.values()) / len(observed_ivs)
            a = VOUCHER_PARAMS["fair_iv_alpha"]
            self.fair_iv = a * avg_iv + (1.0 - a) * self.fair_iv

        # 2) For each near-money voucher: take if BS-fair vs mid deviates enough
        net_delta = 0.0
        for K in VOUCHER_PARAMS["trade_strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            b, a = self._best_bid_ask(od)
            if b is None or a is None:
                continue

            fair_px = bs_call(S, float(K), T_years, self.fair_iv)
            edge = VOUCHER_PARAMS["price_edge_take"]
            pos = state.position.get(sym, 0)
            limit = LIMITS[sym]
            eff = pos
            max_take = VOUCHER_PARAMS["max_take_size"]

            # voucher trades cheap (asks below fair-edge) -> BUY
            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px > fair_px - edge:
                    break
                avail = min(limit - eff, max_take - max(eff - pos, 0))
                if avail <= 0:
                    break
                qty = min(-od.sell_orders[ask_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, ask_px, qty))
                    eff += qty

            # voucher trades rich (bids above fair+edge) -> SELL
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px < fair_px + edge:
                    break
                avail = min(limit + eff, max_take - max(pos - eff, 0))
                if avail <= 0:
                    break
                qty = min(od.buy_orders[bid_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, bid_px, -qty))
                    eff -= qty

            # accumulate signed delta of this voucher (after our trades)
            d = bs_delta(S, float(K), T_years, self.fair_iv)
            net_delta += eff * d

        return net_delta

    # =========================================================================
    # DEEP-ITM VOUCHERS (4000, 4500) : synthetic-forward arb
    # These trade at intrinsic (S - K) +/- noise. Edge >= 2 ticks => near-arb.
    # =========================================================================
    def _trade_deep_itm_vouchers(self, state: TradingState, result: Dict[str, List[Order]]) -> float:
        if "VELVETFRUIT_EXTRACT" not in state.order_depths:
            return 0.0
        S = self._mid(state.order_depths["VELVETFRUIT_EXTRACT"])
        if S is None:
            return 0.0

        net_delta = 0.0
        for K in SYNTH_PARAMS["strikes"]:
            sym = f"VEV_{K}"
            if sym not in state.order_depths:
                continue
            od = state.order_depths[sym]
            b, a = self._best_bid_ask(od)
            if b is None or a is None:
                continue

            fair_px = S - K  # deep ITM call ~= intrinsic, no time value
            edge = SYNTH_PARAMS["edge_threshold"]
            pos = state.position.get(sym, 0)
            limit = LIMITS[sym]
            max_take = SYNTH_PARAMS["max_take_size"]
            eff = pos

            for ask_px in sorted(od.sell_orders.keys()):
                if ask_px > fair_px - edge:
                    break
                avail = min(limit - eff, max_take - max(eff - pos, 0))
                if avail <= 0:
                    break
                qty = min(-od.sell_orders[ask_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, ask_px, qty))
                    eff += qty

            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                if bid_px < fair_px + edge:
                    break
                avail = min(limit + eff, max_take - max(pos - eff, 0))
                if avail <= 0:
                    break
                qty = min(od.buy_orders[bid_px], avail)
                if qty > 0:
                    result[sym].append(Order(sym, bid_px, -qty))
                    eff -= qty

            # delta = 1 for deep ITM
            net_delta += eff * 1.0

        return net_delta

    # =========================================================================
    # MAIN
    # =========================================================================
    def run(self, state: TradingState):
        if not self._inited:
            self._init_state(getattr(state, "traderData", "") or "")

        # --- track day count: detect timestamp wraparound ---
        if self.last_ts >= 0 and state.timestamp < self.last_ts:
            self.day_count += 1
        self.last_ts = state.timestamp

        # --- compute time-to-expiry (in years) ---
        elapsed_days = self.day_count + state.timestamp / TS_PER_DAY
        TTE_days = max(START_TTE_DAYS - elapsed_days, TTE_FLOOR_DAYS)
        T_years = TTE_days / DAYS_PER_YEAR

        # --- result dict with empty list for every product seen ---
        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}

        # 1) Deep-ITM voucher synthetic-forward arb
        deep_delta = self._trade_deep_itm_vouchers(state, result)

        # 2) Near-money voucher vol arb
        near_delta = self._trade_near_vouchers(state, result, T_years)

        # 3) HYDROGEL MM (independent of everything else)
        self._trade_hydrogel(state, result)

        # 4) VELVETFRUIT MM, biased to hedge total voucher delta
        total_voucher_delta = deep_delta + near_delta
        self._trade_velvetfruit(state, result, total_voucher_delta)

        return result, 0, self._serialize_state()