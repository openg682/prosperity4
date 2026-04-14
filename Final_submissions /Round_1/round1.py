from datamodel import OrderDepth, TradingState, Order
from typing import Dict
import jsonpickle
PARAMS = {
    "ASH_COATED_OSMIUM": {
        "fair_value":      10000,
        "take_width":      2,
        "disregard_edge":  6,
        "join_edge":       2,
        "default_edge":    6,
    },
    "INTARIAN_PEPPER_ROOT": {
        "ema_alpha":       0.5,    # increased from 0.25 — tracks the ~1/tick drift much faster
        "take_width":      4,
        "disregard_edge":  10,
        "join_edge":       4,
        "default_edge":    5,
        "skew_factor":     3,
    },
}

POSITION_LIMITS: Dict[str, int] = {
    "ASH_COATED_OSMIUM":    80,
    "INTARIAN_PEPPER_ROOT": 80,
}


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMITS

    def get_fair_value(self, product, order_depth, trader_state):
        if product == "ASH_COATED_OSMIUM":
            return PARAMS["ASH_COATED_OSMIUM"]["fair_value"]

        # INTARIAN_PEPPER_ROOT: EMA of mid price to track the daily drift
        p     = PARAMS["INTARIAN_PEPPER_ROOT"]
        alpha = p["ema_alpha"]

        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            # No market data at all — use last known EMA if available
            return trader_state.get("INTARIAN_PEPPER_ROOT_ema", 10000)

        # KEY FIX: on the very first tick, seed the EMA from the observed mid
        # price rather than the hardcoded constant.  Previously the seed was
        # 10000 while the market was at ~12000 on day 0, causing our ask to be
        # ~1500 below market for the first 20 ticks and pinning us at -80 short.
        if "INTARIAN_PEPPER_ROOT_ema" not in trader_state:
            trader_state["INTARIAN_PEPPER_ROOT_ema"] = mid
            return mid

        prev = trader_state["INTARIAN_PEPPER_ROOT_ema"]
        ema  = alpha * mid + (1 - alpha) * prev
        trader_state["INTARIAN_PEPPER_ROOT_ema"] = ema
        return ema

    def take_best_orders(self, product, fair_value, take_width,
                         orders, order_depth, position):
        lim = self.LIMIT[product]
        bv = sv = 0

        if abs(position) >= lim:
            return 0, 0

        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - take_width:
                qty = min(-order_depth.sell_orders[ba], lim - position)
                if qty > 0:
                    orders.append(Order(product, ba, qty))
                    bv += qty

        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + take_width:
                qty = min(order_depth.buy_orders[bb], lim + position)
                if qty > 0:
                    orders.append(Order(product, bb, -qty))
                    sv += qty

        return bv, sv

    def market_make(self, product, orders, bid, ask,
                    position, buy_volume, sell_volume):
        lim = self.LIMIT[product]
        bq  = lim - (position + buy_volume)
        sq  = lim + (position - sell_volume)
        if bq > 0: orders.append(Order(product, round(bid),  bq))
        if sq > 0: orders.append(Order(product, round(ask), -sq))
        return buy_volume, sell_volume

    def make_orders(self, product, order_depth, fair_value, position,
                    orders, buy_volume, sell_volume):
        p   = PARAMS[product]
        lim = self.LIMIT[product]

        asks_out = [pr for pr in order_depth.sell_orders
                    if pr > fair_value + p["disregard_edge"]]
        bids_out = [pr for pr in order_depth.buy_orders
                    if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + p["default_edge"]
        if asks_out:
            ba  = min(asks_out)
            ask = ba if abs(ba - fair_value) <= p["join_edge"] else ba - 1

        bid = fair_value - p["default_edge"]
        if bids_out:
            bb  = max(bids_out)
            bid = bb if abs(fair_value - bb) <= p["join_edge"] else bb + 1

        # Inventory skewing for INTARIAN_PEPPER_ROOT to prevent drift-induced imbalance
        if product == "INTARIAN_PEPPER_ROOT":
            skew = (position / lim) * p["skew_factor"]
            bid  = round(bid - skew)
            ask  = round(ask - skew)

        return self.market_make(product, orders, bid, ask,
                                position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        for product in ["ASH_COATED_OSMIUM", "INTARIAN_PEPPER_ROOT"]:
            if product not in state.order_depths:
                continue

            orders      = []
            order_depth = state.order_depths[product]
            lim         = self.LIMIT[product]

            # Hard clamp to enforce position limit
            position = state.position.get(product, 0)
            position = max(-lim, min(lim, position))

            p          = PARAMS[product]
            fair_value = self.get_fair_value(product, order_depth, trader_state)
            bv = sv = 0

            b, s = self.take_best_orders(product, fair_value, p["take_width"],
                                         orders, order_depth, position)
            bv += b; sv += s

            b, s = self.make_orders(product, order_depth, fair_value, position,
                                    orders, bv, sv)
            bv += b; sv += s

            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData