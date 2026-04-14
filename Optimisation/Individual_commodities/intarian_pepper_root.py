from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


PARAMS = {
    "ema_alpha":       0.5,    # higher alpha tracks the ~1/tick upward drift more closely
    "take_width":      4,
    "disregard_edge":  10,
    "join_edge":       4,
    "default_edge":    5,
    "skew_factor":     3,
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
            return trader_state.get("ema", 10000)

        # Seed EMA from first observed mid price to avoid cold-start blowup
        if "ema" not in trader_state:
            trader_state["ema"] = mid
            return mid

        ema = alpha * mid + (1 - alpha) * trader_state["ema"]
        trader_state["ema"] = ema
        return ema

    def take_best_orders(self, fair_value, take_width, orders, order_depth, position):
        bv = sv = 0

        if abs(position) >= self.LIMIT:
            return 0, 0

        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - take_width:
                qty = min(-order_depth.sell_orders[ba], self.LIMIT - position)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", ba, qty))
                    bv += qty

        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + take_width:
                qty = min(order_depth.buy_orders[bb], self.LIMIT + position)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", bb, -qty))
                    sv += qty

        return bv, sv

    def market_make(self, orders, bid, ask, position, buy_volume, sell_volume):
        bq = self.LIMIT - (position + buy_volume)
        sq = self.LIMIT + (position - sell_volume)
        if bq > 0: orders.append(Order("INTARIAN_PEPPER_ROOT", round(bid),  bq))
        if sq > 0: orders.append(Order("INTARIAN_PEPPER_ROOT", round(ask), -sq))
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

        # Inventory skewing — pushes quotes down when long, up when short,
        # to prevent drift pinning us to one side of the book
        skew = (position / self.LIMIT) * p["skew_factor"]
        bid  = round(bid - skew)
        ask  = round(ask - skew)

        return self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "INTARIAN_PEPPER_ROOT" in state.order_depths:
            orders      = []
            order_depth = state.order_depths["INTARIAN_PEPPER_ROOT"]

            position = state.position.get("INTARIAN_PEPPER_ROOT", 0)
            position = max(-self.LIMIT, min(self.LIMIT, position))

            fair_value = self.get_fair_value(order_depth, trader_state)
            bv = sv = 0

            b, s = self.take_best_orders(fair_value, PARAMS["take_width"],
                                         orders, order_depth, position)
            bv += b; sv += s

            b, s = self.make_orders(order_depth, fair_value, position, orders, bv, sv)
            bv += b; sv += s

            result["INTARIAN_PEPPER_ROOT"] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData