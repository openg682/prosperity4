import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
SQRT2 = 1.4142135623730951
ALPHA = 0.0615

# T_TOTAL_TS: calibrated so T reaches T_MIN at end of day 4 (elapsed = 5_000_000)
# Derivation: 0.01 = 0.3256 - 0.1080 * 5_000_000 / T_TOTAL_TS
#             T_TOTAL_TS = 0.1080 * 5_000_000 / (0.3256 - 0.01) = 1_711_027
T_START   = 0.3256
T_DECAY   = 0.1080
T_TOTAL_TS = 1_711_027   # was 3_000_000 — FIXED for day 4 expiry
T_MIN     = 0.01

# Near-expiry: day 4 ts > 900_000 (elapsed > 4_900_000)
# At this point T < 0.017 — OTM time value almost fully gone,
# no point accumulating more; unwind any longs before exercise.
NEAR_EXPIRY_ELAPSED = 4_900_000

def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call_delta(S, K, T, sigma):
    if T <= 0: return max(S - K, 0.0), (1.0 if S > K else 0.0)
    sig_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / sig_t
    return (S * norm_cdf(d1) - K * norm_cdf(d1 - sig_t)), norm_cdf(d1)

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

        # Near-expiry flag: stop accumulating OTM, unwind longs instead
        near_expiry = elapsed > NEAR_EXPIRY_ELAPSED

        # time_mult: 3.5x on day 0 → 1.08x at end of day 4
        time_mult = 1.0 + (2.5 * (T / T_START))

        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = self.get_mid(und_od)
        if S is None: return orders, 0, json.dumps(td)

        total_delta = 0.0

        # ── 2. GARBAGE COLLECTOR ──────────────────────────────────
        for sym in ["VEV_6000", "VEV_6500"]:
            pos = state.position.get(sym, 0)
            orders[sym] = [Order(sym, 0, 100 - pos), Order(sym, 1, -(100 + pos))]

        # ── 3. ITM MM: VEV_5000 / 5100 / 5200 ───────────────────
        # These exercise at intrinsic at expiry — safe to hold.
        # near_expiry: still MM normally (intrinsic is certain; time value
        # collapse is already priced into lower BS fair → tighter quotes).
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                limit = int(300 * time_mult)
                s_edge = 1.0 if sym == "VEV_5200" else 1.5
                orders[sym] = self.exec_hybrid_itm(state, sym, fair, 8.0, s_edge, limit)
                total_delta += state.position.get(sym, 0) * delta

        # ── 4. DEEP ITM MM: VEV_4000 / 4500 ─────────────────────
        # Delta ≈ 1, time value negligible — hold to expiry is always fine.
        for sym, K in [("VEV_4000", 4000), ("VEV_4500", 4500)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                limit = int(300 * time_mult)
                orders[sym] = self.exec_mm_uploaded(state, sym, fair, 9.0, 0, limit)
                total_delta += state.position.get(sym, 0) * delta

        # ── 5. OTM: VEV_5300 / 5400 / 5500 ─────────────────────
        # EXPIRY RISK: these expire worthless if OTM. BS fair collapses to 0
        # as T → T_MIN. If near_expiry, stop posting bids and unload any longs.
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
            pos = state.position.get(sym, 0)

            if near_expiry:
                # Unwind longs at any available bid; post no new buy orders.
                # If already flat or short, do nothing (short is fine — we
                # collect premium on options that will expire worthless).
                od  = state.order_depths[sym]
                bb  = max(od.buy_orders.keys())  if od.buy_orders  else None
                ba  = min(od.sell_orders.keys()) if od.sell_orders else None
                sym_orders = []
                if pos > 0 and bb is not None:
                    # Sell the full long at best bid (accept the spread cost
                    # to avoid exercising worthless OTM at expiry)
                    sym_orders.append(Order(sym, bb, -pos))
                elif pos < 0 and ba is not None:
                    # Keep short; post a tight ask to earn any residual premium
                    sym_orders.append(Order(sym, ba, min(5, limit + pos)))
                orders[sym] = sym_orders
            else:
                orders[sym] = self.exec_biased_mm(state, sym, fair, b_off, a_off, side, limit)

            total_delta += pos * delta

        # ── 6. DELTA HEDGE ────────────────────────────────────────
        orders["VELVETFRUIT_EXTRACT"] = self.exec_smart_hedge_v3(
            und_od, state.position.get("VELVETFRUIT_EXTRACT", 0),
            total_delta, 250, 80
        )

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────────────────────
    #  Logic Engines  (unchanged from original)
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