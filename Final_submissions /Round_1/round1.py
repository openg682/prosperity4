from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


PARAMS = {
    "ema_alpha":       0.015,   # slow EMA tracks sinusoidal FV (period ~4000 ticks, amp ~4)
    "take_width":      4,
    "disregard_edge":  6,
    "join_edge":       2,
    "default_edge":    3,
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    def get_fair_value(self, order_depth: OrderDepth, trader_state: Dict) -> int:
        """
        Track fair value with a slow EMA instead of fixing it at 10000.

        ASH_COATED_OSMIUM's fair value oscillates sinusoidally with:
          - Period  ~ 4000 ticks
          - Amplitude ~ 3-4 ticks
        A slow EMA (alpha=0.015) tracks this component while ignoring
        the high-frequency noise from the ±8 book-clearing events.

        Cold-start: seed from the first observed mid so we don't quote
        stale on tick 0 (same fix applied to INTARIAN_PEPPER_ROOT).
        """
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            return round(trader_state.get("ema_fv", 10000))

        if "ema_fv" not in trader_state:
            trader_state["ema_fv"] = mid
            return round(mid)

        alpha = PARAMS["ema_alpha"]
        ema   = alpha * mid + (1 - alpha) * trader_state["ema_fv"]
        trader_state["ema_fv"] = ema
        return round(ema)

    def take_best_orders(self, fair_value: int, take_width: int,
                         orders: List[Order], order_depth: OrderDepth,
                         position: int):
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

    def market_make(self, orders: List[Order], bid: int, ask: int,
                    position: int, buy_volume: int, sell_volume: int):
        bq = self.LIMIT - (position + buy_volume)
        sq = self.LIMIT + (position - sell_volume)
        if bq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(bid),  bq))
        if sq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(ask), -sq))
        return buy_volume, sell_volume

    def make_orders(self, order_depth: OrderDepth, fair_value: int,
                    position: int, orders: List[Order],
                    buy_volume: int, sell_volume: int):
        p = PARAMS
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

        return self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "ASH_COATED_OSMIUM" in state.order_depths:
            orders      = []
            order_depth = state.order_depths["ASH_COATED_OSMIUM"]
            position    = state.position.get("ASH_COATED_OSMIUM", 0)
            position    = max(-self.LIMIT, min(self.LIMIT, position))

            fair_value  = self.get_fair_value(order_depth, trader_state)
            bv = sv = 0

            b, s = self.take_best_orders(fair_value, PARAMS["take_width"],
                                         orders, order_depth, position)
            bv += b; sv += s

            b, s = self.make_orders(order_depth, fair_value, position,
                                    orders, bv, sv)
            bv += b; sv += s

            result["ASH_COATED_OSMIUM"] = orders

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData