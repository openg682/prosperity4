import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List, Optional

# ─────────────────────────────────────────────────────────────
#  Constants
# ─────────────────────────────────────────────────────────────
SQRT2 = 1.4142135623730951
ALPHA = 0.0615 # Locked in based on successful voucher runs

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
        
        # Time management
        day, prev_ts = td.get("day", 0), td.get("prev_ts", 0)
        if state.timestamp < prev_ts and prev_ts > 800_000: day += 1
        td["day"], td["prev_ts"] = day, state.timestamp
        elapsed = day * 1_000_000 + state.timestamp
        T = max(0.3256 - 0.1080 * elapsed / 3_000_000, 0.01)
        
        und_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        S = self.get_mid(und_od)
        if S is None: return orders, 0, json.dumps(td)

        total_delta = 0.0

        # --- Garbage Collector Logic for Illiquid Strikes ---
        for sym in ["VEV_6000", "VEV_6500"]:
            pos = state.position.get(sym, 0)
            limit = 100 # Your desired volume
            
            # Bid 100 at 0 (Buying for free if someone dumps)
            # Ask 100 at 1 (Selling for the minimum possible price)
            orders[sym] = [
                Order(sym, 0, limit - pos),  # Buy up to 100
                Order(sym, 1, -(limit + pos)) # Sell up to 100
            ]
        # 1. THE "MONEY MAKERS": SNIPE + LIGHT MM
        # We allow light passive quoting on 5000-5200 now but with HUGE edges (8.0)
        # to ensure we only get filled on massive noise.
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                # Snipe aggressively (edge 1.5), MM passively (edge 8.0)
                orders[sym] = self.exec_hybrid_itm(state, sym, fair, 8.0, 1.5, 300)
                total_delta += state.position.get(sym, 0) * delta
        for sym, K in [("VEV_5200", 5200)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                # Snipe aggressively (edge 1.5), MM passively (edge 8.0)
                orders[sym] = self.exec_hybrid_itm(state, sym, fair, 8.0, 1.0, 300)
                total_delta += state.position.get(sym, 0) * delta

        # 2. DEEP ITM MM (4000, 4500)
        for sym, K in [("VEV_4000", 4000), ("VEV_4500", 4500)]:
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                orders[sym] = self.exec_mm(state, sym, fair, 9.0, 0, 300)
                total_delta += state.position.get(sym, 0) * delta

        # VEV_5500: We drop the ask offset to 0.0 to be more competitive on sells
        # and push the bid offset to -6.0 to avoid catching falling knives.
        otm_cfg = {
            "VEV_5300": (5300, -3.5, 0.5, "sell"), 
            "VEV_5400": (5400, -0.5, 3.5, "buy"), 
            "VEV_5500": (5500, -4, 0, "sell") 
        }

        # Then, specifically for 5500, increase the limit in the loop:
        for sym, (K, b_off, a_off, side) in otm_cfg.items():
            if sym in state.order_depths:
                fair, delta = bs_call_delta(S, K, T, ALPHA)
                limit = 300 if sym == "VEV_5500" else 300 # Bigger limit for the low-delta lottery ticket
                orders[sym] = self.exec_biased_mm(state, sym, fair, b_off, a_off, side, limit)
                total_delta += state.position.get(sym, 0) * delta


        # 4. SMART HEDGE (The VEF Fix)
        # Increased threshold to 60 to stop over-hedging noise.
        orders["VELVETFRUIT_EXTRACT"] = self.exec_smart_hedge(und_od, state.position.get("VELVETFRUIT_EXTRACT", 0), total_delta, 200, 80)

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────────────────────
    #  Execution Engines
    # ─────────────────────────────────────────────────────────────
    def get_mid(self, od):
        if not od or not od.buy_orders or not od.sell_orders: return None
        return (max(od.buy_orders.keys()) + min(od.sell_orders.keys())) / 2.0

    def exec_hybrid_itm(self, state, sym, fair, mm_edge, snipe_edge, limit):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        # Snipe (Aggressive)
        if bb > fair + snipe_edge and pos > -limit:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        if ba < fair - snipe_edge and pos < limit:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        # Passive (Super Wide to earn fat spreads)
        bid, ask = int(math.floor(fair - mm_edge)), int(math.ceil(fair + mm_edge))
        res.append(Order(sym, min(bid, ba - 1), min(10, limit - pos)))
        res.append(Order(sym, max(ask, bb + 1), -min(10, limit + pos)))
        return res

    def exec_mm(self, state, sym, fair, edge, take_edge, limit):
        od, pos, res = state.order_depths[sym], state.position.get(sym, 0), []
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        # Sniping
        if bb > fair + take_edge and pos > -limit: res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))
        if ba < fair - take_edge and pos < limit: res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))
        # Quoting
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

    def exec_smart_hedge(self, od, pos, delta, limit, thresh):
        target = max(-limit, min(limit, int(-delta)))
        gap = target - pos
        if abs(gap) < 10: return [] # Avoid micro-hedging
        bb, ba = max(od.buy_orders.keys()), min(od.sell_orders.keys())
        res = []
        # Join queue (Tight Passive)
        if gap > 0: res.append(Order("VELVETFRUIT_EXTRACT", bb, min(gap, 30)))
        else: res.append(Order("VELVETFRUIT_EXTRACT", ba, max(gap, -30)))
        # Aggressive Hedge (Only for big imbalances)
        if abs(gap) > thresh:
            px = ba if gap > 0 else bb
            res.append(Order("VELVETFRUIT_EXTRACT", px, gap))
        return res