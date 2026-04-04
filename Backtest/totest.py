from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle
import numpy as np


class Product:
    TOMATOES = "TOMATOES"
    EMERALDS = "EMERALDS"

PARAMS = {
    Product.TOMATOES: {
        "fair_value_seed": 5000,
        "ema_alpha": 0.15,
        "take_width": 6,
        "disregard_edge": 15,
        "join_edge": 5,
        "default_edge": 7,
    },
    Product.EMERALDS: {
        "fair_value": 10000,
        "take_width": 2,
        "disregard_edge": 6,
        "join_edge": 2,
        "default_edge": 6,
    },
}

POSITION_LIMITS: Dict[str, int] = {
    Product.TOMATOES: 80,
    Product.EMERALDS: 80,
}


class Trader:
    def __init__(self):
        self.params = PARAMS
        self.LIMIT = POSITION_LIMITS

    def get_fair_value(self, product, order_depth, trader_state):
        if product == Product.EMERALDS:
            return self.params[product]["fair_value"]

        p = self.params[product]
        alpha = p["ema_alpha"]

        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            mid = p["fair_value_seed"]

        ema_key = f"{product}_ema"
        prev_ema = trader_state.get(ema_key, p["fair_value_seed"])
        new_ema = alpha * mid + (1.0 - alpha) * prev_ema
        trader_state[ema_key] = new_ema
        return new_ema

    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position):
        limit = self.LIMIT[product]
        buy_volume = sell_volume = 0

        if abs(position) >= limit:
            return 0, 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            ask_vol = -order_depth.sell_orders[best_ask]
            if best_ask <= fair_value - take_width:
                qty = min(ask_vol, limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                qty = min(bid_vol, limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_volume += qty

        return buy_volume, sell_volume

    def market_make(self, product, orders, bid, ask, position, buy_volume, sell_volume):
        limit = self.LIMIT[product]
        buy_qty = limit - (position + buy_volume)
        if buy_qty > 0:
            orders.append(Order(product, round(bid), buy_qty))

        sell_qty = limit + (position - sell_volume)
        if sell_qty > 0:
            orders.append(Order(product, round(ask), -sell_qty))

        return buy_volume, sell_volume

    def make_orders(self, product, order_depth, fair_value, position,
                    buy_volume, sell_volume, disregard_edge, join_edge, default_edge):
        orders: List[Order] = []

        asks_outside = [pr for pr in order_depth.sell_orders if pr > fair_value + disregard_edge]
        bids_outside  = [pr for pr in order_depth.buy_orders  if pr < fair_value - disregard_edge]

        ask = fair_value + default_edge
        if asks_outside:
            best_ask = min(asks_outside)
            ask = best_ask if abs(best_ask - fair_value) <= join_edge else best_ask - 1

        bid = fair_value - default_edge
        if bids_outside:
            best_bid = max(bids_outside)
            bid = best_bid if abs(fair_value - best_bid) <= join_edge else best_bid + 1

        return self.market_make(product, orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        for product in [Product.TOMATOES, Product.EMERALDS]:
            if product not in state.order_depths:
                continue

            orders: List[Order] = []
            order_depth = state.order_depths[product]
            position = state.position.get(product, 0)
            p = self.params[product]

            fair_value = self.get_fair_value(product, order_depth, trader_state)

            buy_volume = sell_volume = 0

            b, s = self.take_best_orders(
                product, fair_value, p["take_width"],
                orders, order_depth, position,
            )
            buy_volume += b
            sell_volume += s

            b, s = self.make_orders(
                product, order_depth, fair_value, position,
                buy_volume, sell_volume,
                p["disregard_edge"], p["join_edge"], p["default_edge"],
            )
            buy_volume += b
            sell_volume += s

            result[product] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData