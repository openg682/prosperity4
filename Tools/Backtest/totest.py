from Tools.Backtest.datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
#to run this file, use `python3 backtester.py wrapper.py:totest.py` from the root directory of the repo

PARAMS = {
    "fair_value": 10000,
    "take_width": 2,
    "disregard_edge": 6,
    "join_edge": 2,
    "default_edge": 6,
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.params = PARAMS
        self.LIMIT = POSITION_LIMIT

    def take_best_orders(self, fair_value, take_width, orders, order_depth, position):
        buy_volume = sell_volume = 0

        if abs(position) >= self.LIMIT:
            return 0, 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_vol = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                qty = min(ask_vol, self.LIMIT - position)
                if qty > 0:
                    orders.append(Order("EMERALDS", best_ask, qty))
                    buy_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                qty = min(bid_vol, self.LIMIT + position)
                if qty > 0:
                    orders.append(Order("EMERALDS", best_bid, -qty))
                    sell_volume += qty

        return buy_volume, sell_volume

    def market_make(self, orders, bid, ask, position, buy_volume, sell_volume):
        buy_qty = self.LIMIT - (position + buy_volume)
        if buy_qty > 0:
            orders.append(Order("EMERALDS", round(bid), buy_qty))

        sell_qty = self.LIMIT + (position - sell_volume)
        if sell_qty > 0:
            orders.append(Order("EMERALDS", round(ask), -sell_qty))

        return buy_volume, sell_volume

    def make_orders(self, order_depth, fair_value, position,
                    orders, buy_volume, sell_volume):
        p = self.params
        asks_outside = [pr for pr in order_depth.sell_orders if pr > fair_value + p["disregard_edge"]]
        bids_outside = [pr for pr in order_depth.buy_orders  if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + p["default_edge"]
        if asks_outside:
            best_ask = min(asks_outside)
            ask = best_ask if abs(best_ask - fair_value) <= p["join_edge"] else best_ask - 1

        bid = fair_value - p["default_edge"]
        if bids_outside:
            best_bid = max(bids_outside)
            bid = best_bid if abs(fair_value - best_bid) <= p["join_edge"] else best_bid + 1

        return self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "EMERALDS" in state.order_depths:
            orders: List[Order] = []
            order_depth = state.order_depths["EMERALDS"]
            position = state.position.get("EMERALDS", 0)
            fair_value = self.params["fair_value"]

            buy_volume = sell_volume = 0

            b, s = self.take_best_orders(fair_value, self.params["take_width"],
                                         orders, order_depth, position)
            buy_volume += b; sell_volume += s

            b, s = self.make_orders(order_depth, fair_value, position,
                                    orders, buy_volume, sell_volume)
            buy_volume += b; sell_volume += s

            result["EMERALDS"] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData