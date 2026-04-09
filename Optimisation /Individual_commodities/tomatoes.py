from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

PARAMS = {
    "fair_value_seed": 5000,
    "ema_alpha":       0.25,   # faster than original 0.15 — tracks drift better
    "take_width":      4,      # tighter aggression window
    "default_edge":    5,      # tighter passive quotes than original 7
    "disregard_edge":  10,
    "join_edge":       4,
    "skew_factor":     3,      # shift quotes by up to 3 ticks at position limit
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    def get_fair_value(self, order_depth, trader_state):
        alpha = PARAMS["ema_alpha"]
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            mid = PARAMS["fair_value_seed"]
        prev = trader_state.get("ema", PARAMS["fair_value_seed"])
        ema  = alpha * mid + (1 - alpha) * prev
        trader_state["ema"] = ema
        return ema

    def take_best_orders(self, fair_value, orders, order_depth, position):
        tw = PARAMS["take_width"]; lim = self.LIMIT
        bv = sv = 0
        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - tw:
                qty = min(-order_depth.sell_orders[ba], lim - position)
                if qty > 0:
                    orders.append(Order("TOMATOES", ba, qty))
                    bv += qty
        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + tw:
                qty = min(order_depth.buy_orders[bb], lim + position)
                if qty > 0:
                    orders.append(Order("TOMATOES", bb, -qty))
                    sv += qty
        return bv, sv

    def market_make(self, orders, bid, ask, position, buy_volume, sell_volume):
        lim = self.LIMIT
        bq  = lim - (position + buy_volume)
        sq  = lim + (position - sell_volume)
        if bq > 0: orders.append(Order("TOMATOES", round(bid),  bq))
        if sq > 0: orders.append(Order("TOMATOES", round(ask), -sq))
        return buy_volume, sell_volume

    def make_orders(self, order_depth, fair_value, position,
                    orders, buy_volume, sell_volume):
        p = PARAMS
        asks_out = [pr for pr in order_depth.sell_orders if pr > fair_value + p["disregard_edge"]]
        bids_out = [pr for pr in order_depth.buy_orders  if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + p["default_edge"]
        if asks_out:
            ba  = min(asks_out)
            ask = ba if abs(ba - fair_value) <= p["join_edge"] else ba - 1

        bid = fair_value - p["default_edge"]
        if bids_out:
            bb  = max(bids_out)
            bid = bb if abs(fair_value - bb) <= p["join_edge"] else bb + 1

        # Inventory skewing — shift quotes toward fair value as position builds
        # Long (+pos) → skew down to sell more, short (-pos) → skew up to buy more
        skew = (position / self.LIMIT) * p["skew_factor"]
        bid  = round(bid - skew)
        ask  = round(ask - skew)

        return self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "TOMATOES" in state.order_depths:
            orders      = []
            order_depth = state.order_depths["TOMATOES"]

            # Hard clamp to enforce position limit regardless of exchange behaviour
            position = state.position.get("TOMATOES", 0)
            position = max(-self.LIMIT, min(self.LIMIT, position))

            fair_value = self.get_fair_value(order_depth, trader_state)
            bv = sv = 0

            b, s = self.take_best_orders(fair_value, orders, order_depth, position)
            bv += b; sv += s

            b, s = self.make_orders(order_depth, fair_value, position,
                                    orders, bv, sv)
            bv += b; sv += s

            result["TOMATOES"] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData