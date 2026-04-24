"""
velvetfruit_trader.py - Self-contained VELVETFRUIT_EXTRACT market-maker.

Submit this file directly to the platform. No external imports beyond datamodel.

WHY THIS PRODUCT IS A MM TARGET:
    - Mean ~5250, std ~16, mild upward drift
    - 5-wide best bid/ask spread (tighter than hydrogel)
    - Lag-1 autocorr -0.16 (mean reversion)
    - Average ~3 PnL per round-trip if both sides fill

STRATEGY:
    - Fair value = faster EMA (alpha=0.04) since price drifts mildly
    - TAKE: lift any ask <= fair-1, hit any bid >= fair+1
    - MAKE: post passive 1 tick inside best b/a
    - Inventory skew: 1 tick per +/-80 units of position

PARAMS to tune:
    - take_edge        : higher = fewer trades, more profitable per trade
    - make_inside      : how aggressively to undercut
    - make_size        : passive size per side
    - skew_per_unit    : smaller = stronger pull toward zero inventory
    - ema_alpha        : larger = faster fair value updates
"""

import json
import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order


# ============================================================================
# PARAMETERS  (tune these)
# ============================================================================
PARAMS = {
    "ema_alpha":     0.04,
    "take_edge":     1,
    "make_inside":   1,
    "make_size":     30,
    "skew_per_unit": 80,
}

PRODUCT = "VELVETFRUIT_EXTRACT"
LIMIT = 200


# ============================================================================
# TRADER
# ============================================================================
class Trader:
    def __init__(self):
        self._inited = False

    def _init_state(self, trader_data: str):
        self.ema = None
        if trader_data:
            try:
                d = json.loads(trader_data)
                self.ema = d.get("ema")
            except Exception:
                pass
        self._inited = True

    def _serialize_state(self) -> str:
        return json.dumps({"ema": self.ema})

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

        # === update EMA fair value ===
        alpha = PARAMS["ema_alpha"]
        if self.ema is None:
            self.ema = cur_mid
        else:
            self.ema = alpha * cur_mid + (1.0 - alpha) * self.ema
        fair = self.ema

        pos = state.position.get(PRODUCT, 0)
        take_edge = PARAMS["take_edge"]
        eff = pos

        # === TAKE: lift cheap asks ===
        for ask_px in sorted(od.sell_orders.keys()):
            if ask_px > fair - take_edge:
                break
            avail = LIMIT - eff
            if avail <= 0:
                break
            qty = min(-od.sell_orders[ask_px], avail)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, ask_px, qty))
                eff += qty

        # === TAKE: hit pricey bids ===
        for bid_px in sorted(od.buy_orders.keys(), reverse=True):
            if bid_px < fair + take_edge:
                break
            avail = LIMIT + eff
            if avail <= 0:
                break
            qty = min(od.buy_orders[bid_px], avail)
            if qty > 0:
                result[PRODUCT].append(Order(PRODUCT, bid_px, -qty))
                eff -= qty

        # === MAKE: passive 1 tick inside best b/a, skewed by inventory ===
        inside = PARAMS["make_inside"]
        make_bid = b + inside
        make_ask = a - inside
        if make_bid >= make_ask:
            make_bid = int(math.floor(fair)) - 2
            make_ask = int(math.ceil(fair)) + 2

        skew = -int(round(eff / PARAMS["skew_per_unit"]))
        make_bid += skew
        make_ask += skew
        if make_bid >= make_ask:
            return result, 0, self._serialize_state()

        size = PARAMS["make_size"]
        buy_room = LIMIT - eff
        sell_room = LIMIT + eff
        if buy_room > 0:
            result[PRODUCT].append(Order(PRODUCT, int(make_bid), min(size, buy_room)))
        if sell_room > 0:
            result[PRODUCT].append(Order(PRODUCT, int(make_ask), -min(size, sell_room)))

        return result, 0, self._serialize_state()