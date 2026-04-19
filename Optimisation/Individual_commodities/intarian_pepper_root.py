from datamodel import OrderDepth, TradingState, Order
import jsonpickle

PRODUCT = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80
OPENING_TARGET = 60
FULL_SIZE_TIMESTAMP = 1000


class Trader:
    def __init__(self):
        self.limit = POSITION_LIMIT
        self.anchor = None

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        order_depth = state.order_depths.get(PRODUCT)
        if order_depth is None:
            return result, 0, jsonpickle.encode(trader_state)

        position = state.position.get(PRODUCT, 0)
        timestamp = state.timestamp
        orders = []

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            current_mid = (best_bid + best_ask) / 2.0
            self.anchor = current_mid if self.anchor is None else 0.1 * current_mid + 0.9 * self.anchor
            fair_value = round(self.anchor)

            core_target = OPENING_TARGET if timestamp < FULL_SIZE_TIMESTAMP else 79

            if position < core_target:
                buy_qty = core_target - position
                orders.append(Order(PRODUCT, best_ask, buy_qty))

            if position <= core_target:
                micro_bid = min(fair_value - 1, best_ask - 1)
                orders.append(Order(PRODUCT, micro_bid, 1))
            elif position == core_target + 1:
                micro_ask = max(fair_value + 1, best_bid + 1)
                orders.append(Order(PRODUCT, micro_ask, -1))
            elif position > core_target + 1:
                orders.append(Order(PRODUCT, best_bid, -(position - core_target)))

        result[PRODUCT] = orders
        return result, 0, jsonpickle.encode(trader_state)
