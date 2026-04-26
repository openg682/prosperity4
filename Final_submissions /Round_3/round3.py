import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
#  Options constants
# ─────────────────────────────────────────────────────────────
SQRT2 = 1.4142135623730951
ALPHA = 0.0615

# T_TOTAL_TS: calibrated so T reaches T_MIN at end of day 4 (elapsed = 5_000_000)
# Derivation: 0.01 = 0.3256 - 0.1080 * 5_000_000 / T_TOTAL_TS
#             T_TOTAL_TS = 0.1080 * 5_000_000 / (0.3256 - 0.01) = 1_711_027
T_START    = 0.3256
T_DECAY    = 0.1080
T_TOTAL_TS = 1_711_027   # was 3_000_000 — FIXED for day 4 expiry
T_MIN      = 0.01

# Near-expiry: day 4 ts > 900_000 (elapsed > 4_900_000)
# At this point T < 0.017 — OTM time value almost fully gone.
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


# ─────────────────────────────────────────────────────────────
#  Trader
# ─────────────────────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        orders = {}
        try: td = json.loads(state.traderData) if state.traderData else {}
        except: td = {}

        # ── 1. TIME & VOLUME SCALING ──────────────────────────────
        day, prev_ts = td.get("day", 0), td.get("prev_ts", 0)
        if state.timestamp < prev_ts and prev_ts > 800_000: day += 1
        td["day"], td["prev_ts"] = day, state.timestamp

        elapsed = day * 1_000_000 + state.timestamp
        T = max(T_START - T_DECAY * elapsed / T_TOTAL_TS, T_MIN)

        near_expiry = elapsed > NEAR_EXPIRY_ELAPSED
        time_mult   = 1.0 + (2.5 * (T / T_START))  # 3.5x day 0 → 1.08x day 4 end

        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = self.get_mid(und_od)
        if S is None: return orders, 0, json.dumps(td)

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

        # ── 2. GARBAGE COLLECTOR ──────────────────────────────────
        for sym in ["VEV_6000", "VEV_6500"]:
            pos = state.position.get(sym, 0)
            orders[sym] = [Order(sym, 0, 100 - pos), Order(sym, 1, -(100 + pos))]

        # ── 3. ITM MM: VEV_5000 / 5100 / 5200 ───────────────────
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                limit = int(300 * time_mult)
                s_edge = 1.0 if sym == "VEV_5200" else 1.5
                orders[sym] = self.exec_hybrid_itm(state, sym, fair, 8.0, s_edge, limit)
                total_delta += state.position.get(sym, 0) * delta

        # ── 4. DEEP ITM MM: VEV_4000 / 4500 ─────────────────────
        for sym, K in [("VEV_4000", 4000), ("VEV_4500", 4500)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                limit = int(300 * time_mult)
                orders[sym] = self.exec_mm_uploaded(state, sym, fair, 9.0, 0, limit)
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
            fair, delta = bs_call_delta(S, K, T, ALPHA)
            limit = int(300 * time_mult)
            pos   = state.position.get(sym, 0)

            if near_expiry:
                od = state.order_depths[sym]
                bb = max(od.buy_orders.keys())  if od.buy_orders  else None
                ba = min(od.sell_orders.keys()) if od.sell_orders else None
                sym_orders = []
                if pos > 0 and bb is not None:
                    sym_orders.append(Order(sym, bb, -pos))               # dump longs
                elif pos < 0 and ba is not None:
                    sym_orders.append(Order(sym, ba, min(5, limit + pos))) # earn residual
                orders[sym] = sym_orders
            else:
                orders[sym] = self.exec_biased_mm(state, sym, fair, b_off, a_off, side, limit)

            total_delta += pos * delta

        # ── 6. HYDROGEL_PACK ──────────────────────────────────────
        if hydro_od is not None and hydro_ema is not None:
            orders["HYDROGEL_PACK"] = self.exec_hydrogel(
                hydro_od, state.position.get("HYDROGEL_PACK", 0), hydro_ema
            )

        # ── 7. DELTA HEDGE ────────────────────────────────────────
        orders["VELVETFRUIT_EXTRACT"] = self.exec_smart_hedge_v3(
            und_od, state.position.get("VELVETFRUIT_EXTRACT", 0),
            total_delta, 250, 80
        )

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────────────────────
    #  Options engines  (unchanged)
    # ─────────────────────────────────────────────────────────────

    def get_mid(self, od):
        if not od or not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    def exec_hybrid_itm(self, state, sym, fair, mm_edge, snipe_edge, limit):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        if bb > fair + snipe_edge and pos > -limit:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        if ba < fair - snipe_edge and pos < limit:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        bid, ask = int(math.floor(fair - mm_edge)), int(math.ceil(fair + mm_edge))
        res.append(Order(sym, min(bid, ba - 1), min(10, limit - pos)))
        res.append(Order(sym, max(ask, bb + 1), -min(10, limit + pos)))
        return res

    def exec_mm_uploaded(self, state, sym, fair, edge, take_edge, limit):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        if bb > fair + take_edge and pos > -limit: res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        if ba < fair - take_edge and pos < limit: res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        bid, ask = int(math.floor(fair - edge)), int(math.ceil(fair + edge))
        res.append(Order(sym, min(bid, ba - 1), min(25, limit - pos)))
        res.append(Order(sym, max(ask, bb + 1), -min(25, limit + pos)))
        return res

    def exec_biased_mm(self, state, sym, fair, b_off, a_off, side, limit):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        bid, ask = int(round(fair + b_off)), int(round(fair + a_off))
        if side == "buy": bid = min(max(bid, bb), ba - 1)
        else: ask = max(min(ask, ba), bb + 1)
        res.append(Order(sym, bid, min(30, limit - pos)))
        res.append(Order(sym, ask, -min(30, limit + pos)))
        return res

    def exec_smart_hedge_v3(self, od, pos, delta, limit, thresh):
        target = max(-limit, min(limit, int(-delta)))
        gap = target - pos
        if abs(gap) < 10: return []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        res = []
        if gap > 0: res.append(Order("VELVETFRUIT_EXTRACT", bb, min(gap, 30)))
        else: res.append(Order("VELVETFRUIT_EXTRACT", ba, max(gap, -30)))
        if abs(gap) > thresh:
            res.append(Order("VELVETFRUIT_EXTRACT", ba if gap > 0 else bb, gap))
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