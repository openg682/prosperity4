from datamodel import OrderDepth, TradingState, Order
from typing import List
import jsonpickle

FAIR_VALUE = 10000
POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    def run(self, state: TradingState):
        result = {}

        if "EMERALDS" not in state.order_depths:
            return result, 0, ""

        od  = state.order_depths["EMERALDS"]
        pos = state.position.get("EMERALDS", 0)
        orders: List[Order] = []

        # Undercut best ask by 1 tick, overcut best bid by 1 tick
        # Defaults to FV ± 6 if book is absent
        best_ask = min(od.sell_orders) if od.sell_orders else FAIR_VALUE + 7
        best_bid = max(od.buy_orders)  if od.buy_orders  else FAIR_VALUE - 7

        bid = best_bid + 1   # e.g. 9992 + 1 = 9993
        ask = best_ask - 1   # e.g. 10008 - 1 = 10007

        # Never cross fair value
        bid = min(bid, FAIR_VALUE - 1)
        ask = max(ask, FAIR_VALUE + 1)

        buy_qty  = self.LIMIT - pos
        sell_qty = self.LIMIT + pos

        if buy_qty  > 0:
            orders.append(Order("EMERALDS", bid,  buy_qty))
        if sell_qty > 0:
            orders.append(Order("EMERALDS", ask, -sell_qty))

        result["EMERALDS"] = orders
        return result, 0, ""