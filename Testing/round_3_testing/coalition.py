import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
#  Options constants
# ─────────────────────────────────────────────────────────────
SQRT2 = 1.4142135623730951

# Position limits (verified against exchange)
OPT_LIMIT = 300   # all VEV_* vouchers
VFE_LIMIT = 200   # VELVETFRUIT_EXTRACT

# ITM option controls (VEV_5000/5100/5200)
# Per-product limits calibrated so max combined delta stays within VFE hedge capacity:
#   5000 (delta~0.95): 60 × 0.95 = 57
#   5100 (delta~0.85): 50 × 0.85 = 43
#   5200 (delta~0.66): 40 × 0.66 = 26
#   Deep ITM 4000+4500 (delta~1.0): typically 30+30 = 60
#   Total max delta ~186, within 200 VFE capacity
ITM_LIMITS   = {5000: 60, 5100: 50, 5200: 40}  # per-strike caps (was flat 100)
ITM_MM_EDGE  = 1.5   # asks post at fair+1.5 ≈ inside market spread
ITM_SNIPE    = 2.5   # requires genuine 3pt dislocation before sniping
DELTA_CAP    = 130   # was 160 — gate buys before hedge capacity is breached

# Sigma: seed value; overridden each tick by adaptive IV estimate from market
SIGMA_SEED   = 0.0615
IV_EMA_ALPHA = 0.15   # faster tracking of true implied vol
IV_SOURCES   = [("VEV_5200", 5200), ("VEV_5300", 5300)]  # near-ATM: best vol signal
# True market IV is consistently 0.068-0.075. Hard cap at 0.09 to prevent
# stale traderData sigma causing snipe runaway at start of each run.
SIGMA_LO = 0.04
SIGMA_HI = 0.09

# T_TOTAL_TS: calibrated so T reaches T_MIN at end of day 4 (elapsed = 5_000_000)
# Derivation: 0.01 = 0.3256 - 0.1080 * 5_000_000 / T_TOTAL_TS
#             T_TOTAL_TS = 0.1080 * 5_000_000 / (0.3256 - 0.01) = 1_711_027
T_START    = 0.3256
T_DECAY    = 0.1080
T_TOTAL_TS = 1_711_027
T_MIN      = 0.01

# Near-expiry: elapsed > 4_900_000 (day 4, ts > 900_000)
# T < 0.017 — OTM time value almost fully gone.
NEAR_EXPIRY_ELAPSED = 4_900_000

# ─────────────────────────────────────────────────────────────
#  HYDROGEL_PACK parameters
# ─────────────────────────────────────────────────────────────
HYDRO_LIMIT = 200
HYDRO = {
    "ema_alpha":        0.0006,
    "base_take_edge":   6,
    "take_skew_factor": 3,
    "make_inside":      1,
    "make_size":        9,
    "skew_per_unit":    150,
    "trend_gate":       30,
}

# ─────────────────────────────────────────────────────────────
#  Math helpers
# ─────────────────────────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call_delta(S, K, T, sigma):
    if T <= 0: return max(S - K, 0.0), (1.0 if S > K else 0.0)
    sig_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / sig_t
    return (S * norm_cdf(d1) - K * norm_cdf(d1 - sig_t)), norm_cdf(d1)

def implied_vol(S, K, T, market_price):
    """
    Bisection IV solver. Returns None if market_price has no time value to invert
    (deep ITM intrinsic-only, or market price below intrinsic).
    Robust: never diverges, ~50 iterations to 1e-6 precision.
    """
    if T <= 0: return None
    intrinsic = max(S - K, 0.0)
    if market_price <= intrinsic + 0.5: return None  # no meaningful time value
    lo, hi = 1e-4, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        price = bs_call_delta(S, K, T, mid)[0]
        if abs(price - market_price) < 1e-5: return mid
        if price < market_price: lo = mid
        else: hi = mid
    return (lo + hi) / 2.0


# ─────────────────────────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        orders = {}
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        # ── 1. TIME & VOLUME SCALING ──────────────────────────────
        day, prev_ts = td.get("day", 4), td.get("prev_ts", 0)
        if state.timestamp < prev_ts and prev_ts > 800_000: day += 1
        td["day"], td["prev_ts"] = day, state.timestamp

        elapsed = day * 1_000_000 + state.timestamp
        T = max(T_START - T_DECAY * elapsed / T_TOTAL_TS, T_MIN)
        near_expiry = elapsed > NEAR_EXPIRY_ELAPSED

        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = self.get_mid(und_od)
        if S is None: return orders, 0, json.dumps(td)

        # ── ADAPTIVE SIGMA ────────────────────────────────────────
        # Invert BS on near-ATM market mids each tick to extract implied vol,
        # then EMA-smooth. Falls back to SIGMA_SEED if no valid IV available.
        sigma = td.get("sigma_ema", SIGMA_SEED)
        iv_readings = []
        for sym, K in IV_SOURCES:
            if sym in state.order_depths:
                mid_iv = self.get_mid(state.order_depths[sym])
                if mid_iv is not None:
                    iv = implied_vol(S, K, T, mid_iv)
                    if iv is not None:
                        iv_readings.append(iv)
        if iv_readings:
            iv_now = sum(iv_readings) / len(iv_readings)
            # Stale-sigma reset: if stored sigma is >50% away from current market IV,
            # snap directly to current reading (prevents runaway snipes from prior-run data)
            if abs(sigma - iv_now) / iv_now > 0.50:
                sigma = iv_now
            else:
                sigma = IV_EMA_ALPHA * iv_now + (1.0 - IV_EMA_ALPHA) * sigma
            sigma = max(SIGMA_LO, min(SIGMA_HI, sigma))  # hard clamp to realistic range
        td["sigma_ema"] = sigma

        total_delta = 0.0

        # ── HYDROGEL EMA (persistent, updated every tick) ─────────
        # Stored in traderData so the slow EMA (alpha=0.0006) survives
        # across timestamps without resetting.
        hydro_ema = td.get("hydro_ema")
        hydro_od  = state.order_depths.get("HYDROGEL_PACK")
        if hydro_od and hydro_od.buy_orders and hydro_od.sell_orders:
            hydro_mid = self.get_mid(hydro_od)
            hydro_ema = hydro_mid if hydro_ema is None \
                        else HYDRO["ema_alpha"] * hydro_mid + (1.0 - HYDRO["ema_alpha"]) * hydro_ema
        td["hydro_ema"] = hydro_ema

        # ── PRE-DELTA: current portfolio delta before any new orders ──
        # Used to gate buy-side orders when we're already at hedge capacity.
        # Deep-ITM (4000/4500) use tiny T so delta→1; ITM/OTM use full T.
        pre_delta = 0.0
        for _sym, _K, _T in [
            ("VEV_4000", 4000, 0.001), ("VEV_4500", 4500, 0.001),
            ("VEV_5000", 5000, T), ("VEV_5100", 5100, T), ("VEV_5200", 5200, T),
            ("VEV_5300", 5300, T), ("VEV_5400", 5400, T), ("VEV_5500", 5500, T),
        ]:
            _p = state.position.get(_sym, 0)
            if _p != 0:
                pre_delta += _p * bs_call_delta(S, _K, _T, sigma)[1]
        vfe_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        # Unhedged delta = options delta + VFE hedge (vfe_pos is negative when short)
        # This is the REAL constraint: can't exceed VFE_LIMIT because that's all we can hedge
        unhedged_delta  = pre_delta + vfe_pos
        delta_saturated = unhedged_delta >= VFE_LIMIT  # gate when hedge capacity fully used

        # ── 2. GARBAGE COLLECTOR ──────────────────────────────────
        for sym in ["VEV_6000", "VEV_6500"]:
            pos = state.position.get(sym, 0)
            orders[sym] = [Order(sym, 0, 100 - pos), Order(sym, 1, -(100 + pos))]

        # ── 3. ITM MM: VEV_5000 / 5100 / 5200 ───────────────────
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, sigma)
                limit = ITM_LIMITS[K]
                orders[sym] = self.exec_hybrid_itm(
                    state, sym, fair, ITM_MM_EDGE, ITM_SNIPE, limit, delta_saturated
                )
                total_delta += state.position.get(sym, 0) * delta

        # ── 4. DEEP ITM MM: VEV_4000 / 4500 ─────────────────────
        for sym, K in [("VEV_4000", 4000), ("VEV_4500", 4500)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, sigma)
                limit = OPT_LIMIT
                orders[sym] = self.exec_mm_uploaded(
                    state, sym, fair, 9.0, 0, limit, delta_saturated
                )
                total_delta += state.position.get(sym, 0) * delta

        # ── 5. OTM: VEV_5300 / 5400 / 5500 ─────────────────────
        # near_expiry: unwind longs before they expire worthless
        otm_cfg = {
            "VEV_5300": (5300, -3.5,  0.5, "sell"),
            "VEV_5400": (5400, -0.5,  3.5, "buy"),
            "VEV_5500": (5500, -4.0,  0.0, "sell"),
        }
        for sym, (K, b_off, a_off, side) in otm_cfg.items():
            if sym not in state.order_depths:
                continue
            fair, delta = bs_call_delta(S, K, T, sigma)
            limit = OPT_LIMIT
            pos   = state.position.get(sym, 0)

            if near_expiry:
                od = state.order_depths[sym]
                bb = max(od.buy_orders.keys())  if od.buy_orders  else None
                ba = min(od.sell_orders.keys()) if od.sell_orders else None
                sym_orders = []
                if pos > 0 and bb is not None:
                    sym_orders.append(Order(sym, bb, -pos))               # dump longs
                elif pos < 0 and ba is not None:
                    sym_orders.append(Order(sym, ba, min(5, OPT_LIMIT + pos))) # earn residual
                orders[sym] = sym_orders
            else:
                orders[sym] = self.exec_biased_mm(
                    state, sym, fair, b_off, a_off, side, limit, delta_saturated
                )

            total_delta += pos * delta

        # ── 6. HYDROGEL_PACK ──────────────────────────────────────
        if hydro_od is not None and hydro_ema is not None:
            orders["HYDROGEL_PACK"] = self.exec_hydrogel(
                hydro_od, state.position.get("HYDROGEL_PACK", 0), hydro_ema
            )

        # ── 7. DELTA HEDGE ────────────────────────────────────────
        orders["VELVETFRUIT_EXTRACT"] = self.exec_smart_hedge_v3(
            und_od, state.position.get("VELVETFRUIT_EXTRACT", 0),
            total_delta, VFE_LIMIT, 60
        )

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────────────────────
    #  Options engines  (unchanged)
    # ─────────────────────────────────────────────────────────────

    def get_mid(self, od):
        if not od or not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    def exec_hybrid_itm(self, state, sym, fair, mm_edge, snipe_edge, limit, delta_sat=False):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        # Sell snipe: always active
        if bb > fair + snipe_edge and pos > -limit:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        # Buy snipe: suppressed when portfolio delta is at hedge capacity
        if not delta_sat and ba < fair - snipe_edge and pos < limit:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        bid_px = max(int(math.floor(fair - mm_edge)), bb + 1)  # improve inside: above bb
        ask_px = min(int(math.ceil(fair + mm_edge)), ba - 1)   # improve inside: below ba
        # Only post if spread is valid (avoids crossing when market spread is 1pt)
        if ask_px <= bid_px:
            return res
        # Sell side: always post (earn spread, reduce long inventory)
        if limit + pos > 0:
            res.append(Order(sym, ask_px, -min(10, limit + pos)))
        # Buy side: only post when under delta cap
        if not delta_sat and limit - pos > 0 and bid_px > 0:
            res.append(Order(sym, bid_px, min(10, limit - pos)))
        return res

    def exec_mm_uploaded(self, state, sym, fair, edge, take_edge, limit, delta_sat=False):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        # Sell snipe: always active (reduces long inventory)
        if bb > fair + take_edge and pos > -limit:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        # Buy snipe: gated by delta capacity
        if not delta_sat and ba < fair - take_edge and pos < limit:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        bid, ask = int(math.floor(fair - edge)), int(math.ceil(fair + edge))
        # Sell side: always post
        if limit + pos > 0:
            res.append(Order(sym, max(ask, bb + 1), -min(25, limit + pos)))
        # Buy side: only post when under delta cap
        if not delta_sat and limit - pos > 0:
            res.append(Order(sym, min(bid, ba - 1), min(25, limit - pos)))
        return res

    def exec_biased_mm(self, state, sym, fair, b_off, a_off, side, limit, delta_sat=False):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        bid, ask = int(round(fair + b_off)), int(round(fair + a_off))
        if side == "buy": bid = min(max(bid, bb), ba - 1)
        else: ask = max(min(ask, ba), bb + 1)
        # Always post ask side; suppress bid when portfolio delta is at hedge capacity
        if limit + pos > 0:
            res.append(Order(sym, ask, -min(30, limit + pos)))
        if not delta_sat and limit - pos > 0:
            res.append(Order(sym, bid, min(30, limit - pos)))
        return res

    def exec_smart_hedge_v3(self, od, pos, delta, limit, thresh):
        target = max(-limit, min(limit, int(-delta)))
        gap = target - pos
        if abs(gap) < 10: return []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        res = []
        passive = min(abs(gap), 20)
        if gap > 0: res.append(Order("VELVETFRUIT_EXTRACT", bb, passive))
        else: res.append(Order("VELVETFRUIT_EXTRACT", ba, -passive))
        # Aggressive: only send the RESIDUAL beyond the passive order
        residual = abs(gap) - passive
        if residual > thresh:
            qty = residual - thresh
            if gap > 0: res.append(Order("VELVETFRUIT_EXTRACT", ba, qty))
            else: res.append(Order("VELVETFRUIT_EXTRACT", bb, -qty))
        return res

    # ─────────────────────────────────────────────────────────────
    #  HYDROGEL_PACK engine
    # ─────────────────────────────────────────────────────────────

    def exec_hydrogel(self, od, pos, fair):
        res = []
        b   = max(od.buy_orders.keys())
        a   = min(od.sell_orders.keys())
        eff = pos

        # Asymmetric take edge: skewed by current inventory
        take_skew      = (eff / HYDRO_LIMIT) * HYDRO["take_skew_factor"]
        take_edge_buy  = HYDRO["base_take_edge"] + take_skew
        take_edge_sell = HYDRO["base_take_edge"] - take_skew

        # Trend gate: suppress takes that add inventory into a trending move
        ema_diff = (b + a) / 2.0 - fair
        gate = HYDRO["trend_gate"]
        if ema_diff < -gate: take_edge_buy  = 999  # downtrend — no aggressive buys
        if ema_diff >  gate: take_edge_sell = 999  # uptrend   — no aggressive sells

        # Takes: lift cheap asks
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px > fair - take_edge_buy: break
            avail = HYDRO_LIMIT - eff
            if avail <= 0: break
            qty = min(-od.sell_orders[ask_px], avail)
            if qty > 0:
                res.append(Order("HYDROGEL_PACK", ask_px, qty))
                eff += qty

        # Takes: hit pricey bids
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px < fair + take_edge_sell: break
            avail = HYDRO_LIMIT + eff
            if avail <= 0: break
            qty = min(od.buy_orders[bid_px], avail)
            if qty > 0:
                res.append(Order("HYDROGEL_PACK", bid_px, -qty))
                eff -= qty

        # Passive quotes: 1 tick inside best bid/ask, skewed by inventory
        make_bid = b + HYDRO["make_inside"]
        make_ask = a - HYDRO["make_inside"]

        if make_bid >= make_ask:
            make_bid = int(math.floor(fair)) - 2
            make_ask = int(math.ceil(fair))  + 2

        skew      = -int(round(eff / HYDRO["skew_per_unit"]))
        make_bid += skew
        make_ask += skew

        if make_bid >= make_ask:  # circuit breaker
            return res

        size = HYDRO["make_size"]
        if HYDRO_LIMIT - eff > 0:
            res.append(Order("HYDROGEL_PACK", int(make_bid), min(size, HYDRO_LIMIT - eff)))
        if HYDRO_LIMIT + eff > 0:
            res.append(Order("HYDROGEL_PACK", int(make_ask), -min(size, HYDRO_LIMIT + eff)))

        return res