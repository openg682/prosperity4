"""
WHY THIS PRODUCT IS A MM TARGET:
    - Mean ~9990, std ~32, near-zero correlation with anything else
    - 16-wide best bid/ask spread
    - Lag-1 autocorr -0.13 (weak mean reversion at tick scale)
    - After single-tick moves >8 ticks, next tick mean-reverts 2–4.5 ticks

STRATEGY LAYERS (in order of execution each tick):
    1. Spike mean-reversion takes
       After a large single-tick price jump, take aggressively in the reverse
       direction — exploits the statistically strong next-tick mean reversion.

    2. EMA-based takes (with trend gate)
       Slow EMA (0.0006) as fair value — when asks/bids look cheap/expensive
       relative to it, take aggressively.
       Fast EMA (0.05) as trend detector — suppresses takes that would add
       inventory in the direction of a real short-term trend.

    3. Urgency drain
       When |pos| > 170, place a small aggressive order to start flattening
       before hitting the 200 limit. Prevents being pinned at max inventory
       through large price moves.

    4. Passive make quotes
       1 tick inside best bid/ask, skewed by inventory.
       skew_per_unit=150 gives real lean: 1 tick at pos=150, not just at 200.
"""

import json
import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

# ============================================================================
# PARAMETERS
# ============================================================================
PARAMS = {
    "ema_alpha":        0.0006,  # slow EMA — fair value anchor for makes + takes
    "fast_ema_alpha":   0.05,    # fast EMA — trend signal only (~14-tick memory)
    "base_take_edge":   6,
    "take_skew_factor": 3,
    "make_inside":      1,
    "make_size":        9,
    "skew_per_unit":    150,
    "trend_gate":       5,       # suppress takes when fast_ema lags cur_mid > 5 ticks
    "spike_threshold":  8,       # single-tick move (ticks) that triggers reversion take
    "spike_size":       20,      # units to take on a spike
    "urgency_pos":      170,     # start draining when |pos| exceeds this
    "urgency_size":     5,       # units to drain per tick when urgent
}

PRODUCT = "HYDROGEL_PACK"
LIMIT = 200

# ============================================================================
# TRADER
# ============================================================================
class Trader:
    def __init__(self):
        self._inited = False

    def _init_state(self, trader_data: str):
        self.ema      = None
        self.fast_ema = None
        self.prev_mid = None
        if trader_data:
            try:
                d = json.loads(trader_data)
                self.ema      = d.get("ema")
                self.fast_ema = d.get("fast_ema")
                self.prev_mid = d.get("prev_mid")
            except Exception:
                pass
        self._inited = True

    def _serialize_state(self) -> str:
        return json.dumps({
            "ema":      self.ema,
            "fast_ema": self.fast_ema,
            "prev_mid": self.prev_mid,
        })

    def run(self, state: TradingState):
        if not self._inited:
            self._init_state(getattr(state, "traderData", "") or "")

        result: Dict[str, List[Order]] = {p: [] for p in state.order_depths}

        if PRODUCT not in state.order_depths:
            return result, 0, self._serialize_state()

        od = state.order_depths[PRODUCT]
        if not od.buy_orders or not od.sell_orders:
            return result, 0, self._serialize_state()

        b = max(od.buy_orders.keys())
        a = min(od.sell_orders.keys())
        cur_mid = (b + a) / 2.0

        # ── Update EMAs ───────────────────────────────────────────────────────
        if self.ema is None:
            self.ema = cur_mid
        else:
            self.ema = PARAMS["ema_alpha"] * cur_mid + (1.0 - PARAMS["ema_alpha"]) * self.ema

        if self.fast_ema is None:
            self.fast_ema = cur_mid
        else:
            self.fast_ema = PARAMS["fast_ema_alpha"] * cur_mid + (1.0 - PARAMS["fast_ema_alpha"]) * self.fast_ema

        fair     = self.ema
        prev_mid = self.prev_mid if self.prev_mid is not None else cur_mid

        pos = state.position.get(PRODUCT, 0)
        eff = pos

        # ── 1. Spike mean-reversion takes ─────────────────────────────────────
        # After a large single-tick price jump, the data shows strong mean
        # reversion: >8-tick spike → next tick reverses 2–4.5 ticks on average.
        spike = cur_mid - prev_mid
        spike_size = PARAMS["spike_size"]
        spike_thresh = PARAMS["spike_threshold"]

        if spike < -spike_thresh:
            # Price just dropped hard → buy the dip
            for ask_px in sorted(od.sell_orders.keys()):
                avail = LIMIT - eff
                if avail <= 0:
                    break
                qty = min(-od.sell_orders[ask_px], avail, spike_size)
                if qty > 0:
                    result[PRODUCT].append(Order(PRODUCT, ask_px, qty))
                    eff += qty

        elif spike > spike_thresh:
            # Price just jumped hard → sell the rip
            for bid_px in sorted(od.buy_orders.keys(), reverse=True):
                avail = LIMIT + eff
                if avail <= 0:
                    break
                qty = min(od.buy_orders[bid_px], avail, spike_size)
                if qty > 0:
                    result[PRODUCT].append(Order(PRODUCT, bid_px, -qty))
                    eff -= qty

        # ── 2. EMA-based takes with trend gate ───────────────────────────────
        take_skew = (eff / LIMIT) * PARAMS["take_skew_factor"]
        take_edge_buy  = PARAMS["base_take_edge"] + take_skew
        take_edge_sell = PARAMS["base_take_edge"] - take_skew

        # Trend gate uses fast EMA: fires on real short-term trends, not EMA-lag noise.
        # fast_ema tracks cur_mid closely (~1-2 tick lag) so gate=5 is meaningful.
        fast_diff = cur_mid - self.fast_ema
        gate = PARAMS["trend_gate"]
        if fast_diff < -gate:
            take_edge_buy = 999   # downtrend — don't buy aggressively
        if fast_diff > gate:
            take_edge_sell = 999  # uptrend — don't sell aggressively

        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px > fair - take_edge_buy:
                break
            avail = LIMIT - eff
            if avail <= 0:
                break
            qty = min(-od.sell_orders[ask_px], avail)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, ask_px, qty))
                eff += qty

        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px < fair + take_edge_sell:
                break
            avail = LIMIT + eff
            if avail <= 0:
                break
            qty = min(od.buy_orders[bid_px], avail)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, bid_px, -qty))
                eff -= qty

        # ── 3. Urgency drain ─────────────────────────────────────────────────
        # When near the limit, actively reduce inventory at market rather than
        # wait passively. Prevents holding max inventory through a large move.
        urg_pos  = PARAMS["urgency_pos"]
        urg_size = PARAMS["urgency_size"]

        if eff > urg_pos and od.buy_orders:
            qty = min(eff - urg_pos, urg_size, LIMIT + eff)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, b, -qty))
                eff -= qty

        if eff < -urg_pos and od.sell_orders:
            qty = min(-eff - urg_pos, urg_size, LIMIT - eff)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, a, qty))
                eff += qty

        # ── 4. Passive make quotes ────────────────────────────────────────────
        inside   = PARAMS["make_inside"]
        make_bid = b + inside
        make_ask = a - inside

        if make_bid >= make_ask:
            make_bid = int(math.floor(fair)) - 2
            make_ask = int(math.ceil(fair)) + 2

        skew      = -int(round(eff / PARAMS["skew_per_unit"]))
        make_bid += skew
        make_ask += skew

        # Safe circuit breaker - do not touch!
        if make_bid >= make_ask:
            self.prev_mid = cur_mid
            return result, 0, self._serialize_state()

        size      = PARAMS["make_size"]
        buy_room  = LIMIT - eff
        sell_room = LIMIT + eff

        if buy_room > 0:
            result[PRODUCT].append(Order(PRODUCT, int(make_bid), min(size, buy_room)))
        if sell_room > 0:
            result[PRODUCT].append(Order(PRODUCT, int(make_ask), -min(size, sell_room)))

        self.prev_mid = cur_mid
        return result, 0, self._serialize_state()
