from datamodel import OrderDepth, TradingState, Order
import jsonpickle

PRODUCT = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80
EXIT_TIMESTAMP = 999900


class Trader:
    def __init__(self):
        self.limit = POSITION_LIMIT

    def best_bid(self, order_depth: OrderDepth):
        return max(order_depth.buy_orders) if order_depth.buy_orders else None

    def best_ask(self, order_depth: OrderDepth):
        return min(order_depth.sell_orders) if order_depth.sell_orders else None

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        order_depth = state.order_depths.get(PRODUCT)
        if order_depth is None:
            return result, 0, jsonpickle.encode(trader_state)

        position = max(-self.limit, min(self.limit, state.position.get(PRODUCT, 0)))
        timestamp = state.timestamp
        orders = []

        if timestamp < EXIT_TIMESTAMP:
            buy_qty = self.limit - position
            ask = self.best_ask(order_depth)
            if buy_qty > 0 and ask is not None:
                orders.append(Order(PRODUCT, ask, buy_qty))
        else:
            sell_qty = position
            bid = self.best_bid(order_depth)
            if sell_qty > 0 and bid is not None:
                orders.append(Order(PRODUCT, bid, -sell_qty))

        result[PRODUCT] = orders
        return result, 0, jsonpickle.encode(trader_state)
