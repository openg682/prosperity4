import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from datamodel import OrderDepth, TradingState, Order
import jsonpickle

PRODUCT = "INTARIAN_PEPPER_ROOT"
POSITION_LIMIT = 80

class Trader:
    def __init__(self):
        self.limit = POSITION_LIMIT

    def run(self, state: TradingState):
        result = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        order_depth = state.order_depths.get(PRODUCT)
        if order_depth is None:
            return result, 0, jsonpickle.encode(trader_state)

        position = state.position.get(PRODUCT, 0)
        orders = []

        best_bid = max(order_depth.buy_orders) if order_depth.buy_orders else None
        best_ask = min(order_depth.sell_orders) if order_depth.sell_orders else None

        if best_bid is not None and best_ask is not None:
            
            # Phase 1: CORE 79 MAINTENANCE
            # If we somehow drop below 79 (e.g., tick 0, or our top-tick sell got hit),
            # we instantly buy the market to secure our trend exposure.
            if position < 79:
                buy_qty = 79 - position
                orders.append(Order(PRODUCT, best_ask, buy_qty))
                
            # Phase 2: THE ROTATING DOOR (Continuous Buy 1 / Sell 1)
            if position == 79:
                # We skew BOTH orders upward to account for the linear trend.
                our_bid = best_bid + 1
                our_ask = best_ask + 1
                
                # Prevent our bid from crossing the spread if the book gets tight
                if our_bid >= best_ask:
                    our_bid = best_bid
                    
                # Place both sides continuously
                orders.append(Order(PRODUCT, our_bid, 1))   # Buy 1
                orders.append(Order(PRODUCT, our_ask, -1))  # Sell 1
                
            # Phase 3: PROFIT TAKING
            # If our Buy 1 trap above triggered, we now have 80.
            # We immediately sell 1 back at the current best ask to scalp the spread
            # and return to our 79 baseline.
            elif position == 80:
                orders.append(Order(PRODUCT, best_ask, -1))

        result[PRODUCT] = orders
        return result, 0, jsonpickle.encode(trader_state)