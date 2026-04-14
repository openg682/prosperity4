from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


PARAMS = {
    "fair_value":      10000,
    "take_width":      2,
    "disregard_edge":  6,
    "join_edge":       2,
    "default_edge":    6,
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    def take_best_orders(self, fair_value, take_width, orders, order_depth, position):
        bv = sv = 0

        if abs(position) >= self.LIMIT:
            return 0, 0

        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - take_width:
                qty = min(-order_depth.sell_orders[ba], self.LIMIT - position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", ba, qty))
                    bv += qty

        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + take_width:
                qty = min(order_depth.buy_orders[bb], self.LIMIT + position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", bb, -qty))
                    sv += qty

        return bv, sv

    def market_make(self, orders, bid, ask, position, buy_volume, sell_volume):
        bq = self.LIMIT - (position + buy_volume)
        sq = self.LIMIT + (position - sell_volume)
        if bq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(bid),  bq))
        if sq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(ask), -sq))
        return buy_volume, sell_volume

    def make_orders(self, order_depth, fair_value, position, orders, buy_volume, sell_volume):
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

        return self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            orders      = []
            order_depth = state.order_depths["ASH_COATED_OSMIUM"]

            position = state.position.get("ASH_COATED_OSMIUM", 0)
            position = max(-self.LIMIT, min(self.LIMIT, position))

            fair_value = PARAMS["fair_value"]
            bv = sv = 0

            b, s = self.take_best_orders(fair_value, PARAMS["take_width"],
                                         orders, order_depth, position)
            bv += b; sv += s

            b, s = self.make_orders(order_depth, fair_value, position, orders, bv, sv)
            bv += b; sv += s

            result["ASH_COATED_OSMIUM"] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData