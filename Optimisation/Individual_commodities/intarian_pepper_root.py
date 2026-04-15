from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

EXIT_TIMESTAMP = 999200
STOP_LOSS      = -3000

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    def get_mid(self, order_depth: OrderDepth, fallback: float) -> float:
        has_bid = len(order_depth.buy_orders)  > 0
        has_ask = len(order_depth.sell_orders) > 0
        if has_bid and has_ask:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif has_bid:
            return float(max(order_depth.buy_orders))
        elif has_ask:
            return float(min(order_depth.sell_orders))
        return fallback

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "INTARIAN_PEPPER_ROOT" not in state.order_depths:
            return result, 0, jsonpickle.encode(trader_state)

        orders      = []
        order_depth = state.order_depths["INTARIAN_PEPPER_ROOT"]
        position    = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        position    = max(-self.LIMIT, min(self.LIMIT, position))
        ts          = state.timestamp
        lim         = self.LIMIT

        entry_price = trader_state.get("entry_price", None)
        stopped_out = trader_state.get("stopped_out", False)

        # MTM stop-loss check
        if entry_price is not None and position > 0 and not stopped_out:
            mid = self.get_mid(order_depth, float(entry_price))
            mtm = position * (mid - entry_price)
            if mtm < STOP_LOSS:
                stopped_out = True
                trader_state["stopped_out"] = True

        if stopped_out or ts >= EXIT_TIMESTAMP:
            # Distribution: hit bids to flatten
            want = position
            for px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if want <= 0: break
                qty = min(order_depth.buy_orders[px], want)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", px, -qty))
                    want -= qty

        else:
            # Accumulation: hit asks to go max long
            want = lim - position
            for px in sorted(order_depth.sell_orders.keys()):
                if want <= 0: break
                qty = min(-order_depth.sell_orders[px], want)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", px, qty))
                    want -= qty
                    # Record entry price on first fill
                    if entry_price is None:
                        trader_state["entry_price"] = float(px)
                        entry_price = float(px)

        result["INTARIAN_PEPPER_ROOT"] = orders
        trader_state["stopped_out"]    = stopped_out
        return result, 0, jsonpickle.encode(trader_state)