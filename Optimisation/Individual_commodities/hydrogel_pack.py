"""
WHY THIS PRODUCT IS A MM TARGET:
    - Mean ~9990, std ~32, near-zero correlation with anything else
    - 16-wide best bid/ask spread, 73% win rate buying at bid+1
    - Lag-1 autocorr -0.13 (weak mean reversion at tick scale)

OPTIMIZATIONS APPLIED:
    - Slower EMA (0.0005) ensures fair value anchors strongly.
    - Base take_edge (6) prevents paying the 16-wide spread prematurely.
    - Wider skew_per_unit (80) maximizes spread capture before yielding.
    - NEW: Inventory-Asymmetric Takes (Urgency scaling) to safely flatten risk.
"""

import json
import math
from typing import Dict, List
from datamodel import OrderDepth, TradingState, Order

# ============================================================================
# PARAMETERS 
# ============================================================================
PARAMS = {
    "ema_alpha":        0.0006,
    "base_take_edge":   6,
    "take_skew_factor": 3,   # How many ticks to flex the take edge based on inventory
    "make_inside":      1,
    "make_size":        9,  # Increased to 50 to perfectly quarter our 200 limit
    "skew_per_unit":    370,
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
        eff = pos

        # === NEW: Asymmetric Take Edge ===
        # If pos is positive, take_skew is positive.
        take_skew = (eff / LIMIT) * PARAMS["take_skew_factor"]
        
        # If pos > 0: harder to buy (higher edge needed), easier to sell (lower edge needed)
        take_edge_buy = PARAMS["base_take_edge"] + take_skew
        take_edge_sell = PARAMS["base_take_edge"] - take_skew

        # === TAKE: lift cheap asks (BUYING) ===
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

        # === TAKE: hit pricey bids (SELLING) ===
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
        
        # Safe circuit breaker - do not touch!
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