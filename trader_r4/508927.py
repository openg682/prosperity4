import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

# ─────────────────────────────────────────────
# Limits
# ─────────────────────────────────────────────
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
    "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
    "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SQRT2 = 1.4142135623730951

OPT_LIMIT = 300
VFE_LIMIT = 200

ITM_LIMITS   = {5000: 60, 5100: 50, 5200: 40}
ITM_MM_EDGE  = 2.5
ITM_SNIPE    = 3.5

DELTA_CAP = 130

SIGMA_SEED   = 0.0615
IV_EMA_ALPHA = 0.24
SIGMA_LO = 0.04
SIGMA_HI = 0.09

T_START = 0.3256
T_DECAY = 0.1080
T_TOTAL_TS = 1_711_027
T_MIN = 0.01

# ─────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call_delta(S, K, T, sigma):
    if T <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)

    sig_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / sig_t

    price = S * norm_cdf(d1) - K * norm_cdf(d1 - sig_t)
    delta = norm_cdf(d1)
    return price, delta

def implied_vol(S, K, T, price):
    if T <= 0:
        return None

    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5:
        return None

    lo, hi = 1e-4, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = bs_call_delta(S, K, T, mid)[0]
        if p < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        orders = {}

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except:
            td = {}

        # ── time
        day = td.get("day", 4)
        prev_ts = td.get("prev_ts", 0)

        if state.timestamp < prev_ts:
            day += 1

        td["day"] = day
        td["prev_ts"] = state.timestamp

        elapsed = day * 1_000_000 + state.timestamp
        T = max(T_START - T_DECAY * elapsed / T_TOTAL_TS, T_MIN)

        # ── underlying
        od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not od:
            return orders, 0, json.dumps(td)

        S = self.mid(od)
        if S is None:
            return orders, 0, json.dumps(td)

        # ─────────────────────────────────────────
        # VOL FIX
        # ─────────────────────────────────────────
        sigma = td.get("sigma", SIGMA_SEED)

        ivs = []
        for sym, K in [("VEV_5200", 5200), ("VEV_5300", 5300)]:
            if sym in state.order_depths:
                mid = self.mid(state.order_depths[sym])
                if mid:
                    iv = implied_vol(S, K, T, mid)
                    if iv:
                        ivs.append(iv)

        if ivs:
            iv_now = sorted(ivs)[len(ivs)//2]

            step = 0.01
            diff = iv_now - sigma
            if abs(diff) > step:
                iv_now = sigma + step * (1 if diff > 0 else -1)

            sigma = IV_EMA_ALPHA * iv_now + (1 - IV_EMA_ALPHA) * sigma
            sigma = max(SIGMA_LO, min(SIGMA_HI, sigma))

        td["sigma"] = sigma

        # ─────────────────────────────────────────
        # DELTA
        # ─────────────────────────────────────────
        total_delta = 0

        # ─────────────────────────────────────────
        # ITM OPTIONS
        # ─────────────────────────────────────────
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            orders[sym] = self.itm_mm(
                sym, od, fair, bb, ba,
                ITM_MM_EDGE, ITM_SNIPE,
                ITM_LIMITS[K],
                state.position.get(sym, 0)
            )

            total_delta += state.position.get(sym, 0) * delta

        # ─────────────────────────────────────────
        # DEEP ITM
        # ─────────────────────────────────────────
        for sym, K in [("VEV_4000", 4000), ("VEV_4500", 4500)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            orders[sym] = self.basic_mm(sym, od, fair, 9.0, 300, state.position.get(sym, 0))
            total_delta += state.position.get(sym, 0) * delta

        # ─────────────────────────────────────────
        # OTM
        # ─────────────────────────────────────────
        for sym, K in [("VEV_5300", 5300), ("VEV_5400", 5400), ("VEV_5500", 5500)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            skew = (S - K) / K
            b_off = -2 + 6 * skew
            a_off =  2 + 6 * skew

            orders[sym] = self.biased_mm(
                sym, od, fair, bb, ba,
                b_off, a_off, 300,
                state.position.get(sym, 0),
                delta_sat=False
            )

            total_delta += state.position.get(sym, 0) * delta

        # ─────────────────────────────────────────
        # HYDROGEL (RESTORED — FIXED)
        # ─────────────────────────────────────────
        hydro_od = state.order_depths.get("HYDROGEL_PACK")
        if hydro_od:
            orders["HYDROGEL_PACK"] = self.exec_hydrogel(
                hydro_od,
                state.position.get("HYDROGEL_PACK", 0),
                S
            )

        # ─────────────────────────────────────────
        # DELTA HEDGE
        # ─────────────────────────────────────────
        pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        gap = -total_delta - pos

        if abs(gap) > 5:
            qty = int(max(-200, min(200, gap)))
            od = state.order_depths["VELVETFRUIT_EXTRACT"]
            bb, ba = self.bbba(od)

            side_px = bb if qty > 0 else ba
            orders["VELVETFRUIT_EXTRACT"] = [Order("VELVETFRUIT_EXTRACT", side_px, qty)]

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────
    # HYDROGEL (FIXED HYBRID VERSION)
    # ─────────────────────────────────────────────
    def exec_hydrogel(self, od, pos, fair):
        res = []

        bb = max(od.buy_orders)
        ba = min(od.sell_orders)

        mid = (bb + ba) / 2
        dev = mid - fair

        # aggressive sweep when mispriced
        if abs(dev) > 2:

            if dev < -2:
                for px in sorted(od.sell_orders):
                    if px > fair - 10:
                        break
                    qty = min(-od.sell_orders[px], 200 - pos)
                    if qty > 0:
                        res.append(Order("HYDROGEL_PACK", px, qty))
                        pos += qty

            elif dev > 2:
                for px in sorted(od.buy_orders, reverse=True):
                    if px < fair + 10:
                        break
                    qty = min(od.buy_orders[px], 200 + pos)
                    if qty > 0:
                        res.append(Order("HYDROGEL_PACK", px, -qty))
                        pos -= qty

        # passive MM layer
        skew = pos / 200 * 4

        bid = int(fair - 2 - skew)
        ask = int(fair + 2 - skew)

        if bid < ask:
            res.append(Order("HYDROGEL_PACK", bid, 10))
            res.append(Order("HYDROGEL_PACK", ask, -10))

        return res

    # ─────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────
    def mid(self, od):
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    def bbba(self, od):
        return max(od.buy_orders), min(od.sell_orders)

    def itm_mm(self, sym, od, fair, bb, ba, edge, snipe, limit, pos):
        res = []

        if bb > fair + snipe:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))

        if ba < fair - snipe:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))

        bid = int(fair - edge)
        ask = int(fair + edge)

        if ask > bid:
            res.append(Order(sym, bid, 5))
            res.append(Order(sym, ask, -5))

        return res

    def basic_mm(self, sym, od, fair, edge, limit, pos):
        bid = int(fair - edge)
        ask = int(fair + edge)

        return [
            Order(sym, bid, min(10, limit - pos)),
            Order(sym, ask, -min(10, limit + pos))
        ]

    def biased_mm(self, sym, od, fair, bb, ba, b_off, a_off, limit, pos, delta_sat):
        bid = int(fair + b_off)
        ask = int(fair + a_off)

        return [
            Order(sym, bid, min(10, limit - pos)),
            Order(sym, ask, -min(10, limit + pos))
        ]