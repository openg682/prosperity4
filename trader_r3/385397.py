"""
Stripped-down strategy: Trades ONLY VEV_4000 and delta-hedges with VELVETFRUIT_EXTRACT.
Removed smile-fitting and all logic/loops for other strikes.
"""
from datamodel import Order, OrderDepth, TradingState, Symbol
from typing import Dict, List, Optional
import math
import json

# ------------- Product config ---------------------------------------
UNDERLYING = "VELVETFRUIT_EXTRACT"
STRIKE = 4000
SYM_4000 = "VEV_4000"

POS_LIM_UND = 200
POS_LIM_VOU = 300

# ------------- Tunable parameters -----------------------------------
TAKE_EDGE     = 0.0     
MM_EDGE       = 5.0     
MM_QTY        = 15
HEDGE_TOL     = 100
T_START_DAYS  = 5.0     
TICKS_PER_DAY = 1_000_000

IV_EMA_ALPHA  = 0.01   
FALLBACK_SIGMA = 0.1 

# ------------- Black-Scholes helpers --------------------------------
def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))

def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0: return max(S - K, 0.0)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ncdf(d1) - K * _ncdf(d2)

def bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0.0 or sigma <= 0.0: return 1.0 if S > K else 0.0
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * math.sqrt(T))
    return _ncdf(d1)

def implied_vol(price: float, S: float, K: float, T: float) -> Optional[float]:
    if T <= 0.0: return None
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6 or price >= S: return None
    lo, hi = 1e-5, 5.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if bs_call(S, K, T, mid) < price: lo = mid
        else: hi = mid
        if hi - lo < 1e-5: break
    return 0.5 * (lo + hi)

# ------------- Trader -----------------------------------------------
class Trader:
    @staticmethod
    def _best_bid_ask(depth: OrderDepth):
        bb = max(depth.buy_orders.keys()) if depth.buy_orders else None
        ba = min(depth.sell_orders.keys()) if depth.sell_orders else None
        return bb, ba

    def _mid(self, depth: OrderDepth):
        bb, ba = self._best_bid_ask(depth)
        if bb is None or ba is None: return None
        return 0.5 * (bb + ba)

    def run(self, state: TradingState):
        result: Dict[Symbol, List[Order]] = {}

        # 1. Price the underlying
        if UNDERLYING not in state.order_depths: return result, 0, state.traderData or ""
        S = self._mid(state.order_depths[UNDERLYING])
        if S is None: return result, 0, state.traderData or ""

        # 2. Track time across day rollovers safely
        try: td = json.loads(state.traderData) if state.traderData else {}
        except Exception: td = {}
        
        last_ts = td.get("last_ts", 0)
        day_offset = td.get("day_offset", 0)
        if state.timestamp < last_ts: day_offset += 1
        td["last_ts"] = state.timestamp
        td["day_offset"] = day_offset

        total_ticks = day_offset * TICKS_PER_DAY + state.timestamp
        days = T_START_DAYS - total_ticks / TICKS_PER_DAY
        T = max(days, 1e-6) / 365.0

        # 3. Track Implied Volatility just for the 4000 strike
        iv_ema = td.get("iv_ema", FALLBACK_SIGMA)
        sym_depth = state.order_depths.get(SYM_4000)
        
        if sym_depth:
            px = self._mid(sym_depth)
            if px is not None:
                iv_now = implied_vol(px, S, STRIKE, T)
                if iv_now is not None:
                    iv_ema = (1 - IV_EMA_ALPHA) * iv_ema + IV_EMA_ALPHA * iv_now
        td["iv_ema"] = iv_ema

        # This will hold delta change from any aggressive orders we take this tick
        new_trade_delta = 0.0 

        # -------------------- Trade VEV_4000 ---------------------------
        if sym_depth:
            pos = state.position.get(SYM_4000, 0)
            buy_cap, sell_cap = POS_LIM_VOU - pos, POS_LIM_VOU + pos
            
            bb, ba = self._best_bid_ask(sym_depth)
            safe_bb = bb if bb is not None else 0
            safe_ba = ba if ba is not None else 999999

            fair = bs_call(S, STRIKE, T, iv_ema)
            delta_k = bs_delta(S, STRIKE, T, iv_ema)
            
            orders = []

            # --- take overpriced bids ---
            if bb is not None:
                for bid_px in sorted(sym_depth.buy_orders.keys(), reverse=True):
                    if bid_px > fair + TAKE_EDGE and sell_cap > 0:
                        q = min(sym_depth.buy_orders[bid_px], sell_cap)
                        if q > 0:
                            orders.append(Order(SYM_4000, bid_px, -q))
                            pos -= q; sell_cap -= q; buy_cap += q
                            new_trade_delta -= q * delta_k
                    else: break

            # --- take underpriced asks ---
            if ba is not None:
                for ask_px in sorted(sym_depth.sell_orders.keys()):
                    if ask_px < fair - TAKE_EDGE and buy_cap > 0:
                        q = min(-sym_depth.sell_orders[ask_px], buy_cap)
                        if q > 0:
                            orders.append(Order(SYM_4000, ask_px, q))
                            pos += q; buy_cap -= q; sell_cap += q
                            new_trade_delta += q * delta_k
                    else: break

            # --- passive sell inside the spread ---
            if sell_cap > 0:
                tgt_ask = max(safe_bb + 1, int(math.ceil(fair + MM_EDGE)))
                if ba is None or tgt_ask < ba:
                    q = min(MM_QTY, sell_cap)
                    if q > 0: orders.append(Order(SYM_4000, tgt_ask, -q))

            # --- passive buy inside the spread ---
            if buy_cap > 0:
                tgt_bid = min(safe_ba - 1, int(math.floor(fair - MM_EDGE)))
                if bb is None or tgt_bid > bb:
                    q = min(MM_QTY, buy_cap)
                    if q > 0: orders.append(Order(SYM_4000, tgt_bid, q))

            if orders: result[SYM_4000] = orders

        # -------------------- Delta Hedge on VEF --------------------
        # Calculate total delta: Current position delta + delta from trades just fired
        current_delta = state.position.get(SYM_4000, 0) * bs_delta(S, STRIKE, T, iv_ema)
        total_delta = current_delta + new_trade_delta

        vef_pos = state.position.get(UNDERLYING, 0)
        target_vef = max(-POS_LIM_UND, min(POS_LIM_UND, -int(round(total_delta))))
        hedge_need = target_vef - vef_pos

        if abs(hedge_need) > HEDGE_TOL:
            vef_depth = state.order_depths.get(UNDERLYING)
            if vef_depth:
                vef_bb, vef_ba = self._best_bid_ask(vef_depth)
                if hedge_need > 0 and vef_ba is not None:
                    q = min(hedge_need, -vef_depth.sell_orders[vef_ba])
                    if q > 0: result.setdefault(UNDERLYING, []).append(Order(UNDERLYING, vef_ba, int(q)))
                elif hedge_need < 0 and vef_bb is not None:
                    q = min(-hedge_need, vef_depth.buy_orders[vef_bb])
                    if q > 0: result.setdefault(UNDERLYING, []).append(Order(UNDERLYING, vef_bb, -int(q)))

        return result, 0, json.dumps(td)