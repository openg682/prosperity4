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

# HYDROGEL params (new)
HYD_LIMIT      = 200
HYD_EDGE       = 2          # quote at micro_price ± 2 (market spread is ~16, so we're well inside)
HYD_SIZE       = 15         # was 10; spread is fat enough to size up
HYD_SKEW_K     = 6          # max ±6 tick shift at full position (pos/limit) * K
HYD_SWEEP_EDGE = 8          # only sweep when book is genuinely dislocated vs micro-price

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
        # HYDROGEL — NEW: fair value from own book, not VFE
        # ─────────────────────────────────────────
        hydro_od = state.order_depths.get("HYDROGEL_PACK")
        if hydro_od:
            orders["HYDROGEL_PACK"] = self.exec_hydrogel(
                hydro_od,
                state.position.get("HYDROGEL_PACK", 0),
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
    # HYDROGEL — own-book micro-price MM
    #
    # The previous version used the VFE mid (~5295) as fair while HYDROGEL
    # trades around 10000, so `dev = mid - fair > 4700` always, and the
    # aggressive-sweep branch dumped HYDROGEL down to position limit every run.
    # Cost: −3550 final P&L (peak −10380 in the original log).
    #
    # Empirical facts about HYDROGEL from the logs:
    #   • Market spread is consistently 16 ticks wide (min 7, max 17)
    #   • Mid std is ~14 ticks, range ~68 ticks over 100k timestamps
    #   • Wide spread + modest vol → textbook target for inside-spread MM
    #
    # New approach:
    #   1. Fair value = volume-weighted top-of-book micro-price
    #      (handles asymmetric depth better than naive (bb+ba)/2)
    #   2. Quote at fair ± HYD_EDGE — well inside the 16-wide market,
    #      so we capture spread when crossed
    #   3. Inventory skew: shift both quotes by −(pos/limit)·HYD_SKEW_K so
    #      we naturally lean toward flattening
    #   4. Aggressive sweep ONLY when a level is HYD_SWEEP_EDGE ticks
    #      worse than micro-price — i.e. someone left a stale quote we
    #      can pick off. No more blind directional dumps.
    #   5. Per-side size capped by remaining capacity (limit ± pos)
    # ─────────────────────────────────────────────
    def exec_hydrogel(self, od, pos):
        res = []

        if not od.buy_orders or not od.sell_orders:
            return res

        bb = max(od.buy_orders)
        ba = min(od.sell_orders)

        if ba - bb < 2:
            # market is already tight, skip — no edge to capture, and our
            # passive quotes would just cross the existing book
            return res

        # micro-price: weight top-of-book by opposing-side volumes so price
        # leans toward where there's less liquidity (less likely to fade)
        bv = od.buy_orders[bb]
        av = -od.sell_orders[ba]
        if bv + av > 0:
            micro = (bb * av + ba * bv) / (bv + av)
        else:
            micro = (bb + ba) / 2

        # ── 1. Sweep stale levels (rare, high-edge)
        # Lift any ask priced below micro − HYD_SWEEP_EDGE
        for px in sorted(od.sell_orders):
            if px >= micro - HYD_SWEEP_EDGE:
                break
            qty = min(-od.sell_orders[px], HYD_LIMIT - pos)
            if qty > 0:
                res.append(Order("HYDROGEL_PACK", px, qty))
                pos += qty

        # Hit any bid priced above micro + HYD_SWEEP_EDGE
        for px in sorted(od.buy_orders, reverse=True):
            if px <= micro + HYD_SWEEP_EDGE:
                break
            qty = min(od.buy_orders[px], HYD_LIMIT + pos)
            if qty > 0:
                res.append(Order("HYDROGEL_PACK", px, -qty))
                pos -= qty

        # ── 2. Passive MM with inventory skew
        skew = (pos / HYD_LIMIT) * HYD_SKEW_K
        bid_px = int(micro - HYD_EDGE - skew)
        ask_px = int(micro + HYD_EDGE - skew)

        # never quote across or onto the market — only inside it
        if bid_px >= ba:
            bid_px = ba - 1
        if ask_px <= bb:
            ask_px = bb + 1

        if bid_px < ask_px:
            bid_size = max(0, min(HYD_SIZE, HYD_LIMIT - pos))
            ask_size = max(0, min(HYD_SIZE, HYD_LIMIT + pos))
            if bid_size > 0:
                res.append(Order("HYDROGEL_PACK", bid_px, bid_size))
            if ask_size > 0:
                res.append(Order("HYDROGEL_PACK", ask_px, -ask_size))

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