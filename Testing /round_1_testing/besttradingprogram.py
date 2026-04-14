from datamodel import OrderDepth, TradingState, Order
from typing import Dict
import jsonpickle

# Hybrid final submission:
# - keeps robust inventory controls from the improved version
# - borrows public-round aggression from your best visible-round config
PARAMS = {
    "EMERALDS": {
        "fair_value": 10000,
        "take_width": 2,
        "disregard_edge": 6,
        "join_edge": 2,
        "default_edge": 3,
        "skew_factor": 4,
        "base_clip": 34,
    },
    "TOMATOES": {
        "fair_value_seed": 5000,
        "ema_alpha": 0.20,
        "take_width": 3,
        "disregard_edge": 10,
        "join_edge": 3,
        "default_edge": 4,
        "skew_factor": 4,
        "base_clip": 20,
    },
}

POSITION_LIMITS: Dict[str, int] = {
    "EMERALDS": 80,
    "TOMATOES": 80,
}


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMITS

    def get_fair_value(self, product, order_depth, trader_state):
        if product == "EMERALDS":
            return PARAMS[product]["fair_value"]

        p = PARAMS[product]
        alpha = p["ema_alpha"]

        # Hybrid estimator: use midpoint EMA for public friendliness
        # but smoother alpha than pure public optimizer for robustness
        if order_depth.buy_orders and order_depth.sell_orders:
            px = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            px = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            px = float(min(order_depth.sell_orders))
        else:
            px = p["fair_value_seed"]

        prev = trader_state.get(f"{product}_ema", p["fair_value_seed"])
        ema = alpha * px + (1 - alpha) * prev
        trader_state[f"{product}_ema"] = ema
        return ema

    def take_best_orders(self, product, fair_value, take_width, orders, order_depth, position):
        lim = self.LIMIT[product]
        bought = sold = 0

        if abs(position) >= lim:
            return 0, 0

        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - take_width:
                qty = min(-order_depth.sell_orders[ba], lim - position)
                if qty > 0:
                    orders.append(Order(product, ba, qty))
                    bought += qty

        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + take_width:
                qty = min(order_depth.buy_orders[bb], lim + position)
                if qty > 0:
                    orders.append(Order(product, bb, -qty))
                    sold += qty

        return bought, sold

    def market_make(self, product, orders, bid, ask, position, buy_volume, sell_volume):
        lim = self.LIMIT[product]
        clip = PARAMS[product]["base_clip"]

        remaining_buy = max(0, lim - (position + buy_volume))
        remaining_sell = max(0, lim + (position - sell_volume))

        buy_size = min(clip, remaining_buy)
        sell_size = min(clip, remaining_sell)

        # Hybrid inventory control: enough suppression for robustness,
        # but much less conservative than the pure safe version
        inv_ratio = abs(position / lim)
        if position > 0:
            buy_size = max(1, int(buy_size * (1 - 0.50 * inv_ratio)))
        elif position < 0:
            sell_size = max(1, int(sell_size * (1 - 0.50 * inv_ratio)))

        if buy_size > 0:
            orders.append(Order(product, round(bid), buy_size))
        if sell_size > 0:
            orders.append(Order(product, round(ask), -sell_size))

        return buy_size, sell_size

    def make_orders(self, product, order_depth, fair_value, position, orders, buy_volume, sell_volume):
        p = PARAMS[product]
        lim = self.LIMIT[product]

        asks_out = [pr for pr in order_depth.sell_orders if pr > fair_value + p["disregard_edge"]]
        bids_out = [pr for pr in order_depth.buy_orders if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + p["default_edge"]
        if asks_out:
            ba = min(asks_out)
            ask = ba if abs(ba - fair_value) <= p["join_edge"] else ba - 1

        bid = fair_value - p["default_edge"]
        if bids_out:
            bb = max(bids_out)
            bid = bb if abs(fair_value - bb) <= p["join_edge"] else bb + 1

        # Apply controlled skew to BOTH products
        skew = (position / lim) * p["skew_factor"]
        bid -= 0.5 * skew
        ask -= 1.0 * skew

        if round(bid) >= round(ask):
            bid = ask - 1

        return self.market_make(product, orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        for product in ["EMERALDS", "TOMATOES"]:
            if product not in state.order_depths:
                continue

            order_depth = state.order_depths[product]
            lim = self.LIMIT[product]
            position = max(-lim, min(lim, state.position.get(product, 0)))

            p = PARAMS[product]
            fair_value = self.get_fair_value(product, order_depth, trader_state)

            orders = []
            bv = sv = 0

            b, s = self.take_best_orders(product, fair_value, p["take_width"], orders, order_depth, position)
            bv += b
            sv += s

            b, s = self.make_orders(product, order_depth, fair_value, position, orders, bv, sv)
            bv += b
            sv += s

            result[product] = orders

        return result, 0, jsonpickle.encode(trader_state)
