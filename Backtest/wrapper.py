"""
wrapper.py
==========
Translates any clean Prosperity strategy file into a backtester-compatible
algorithm — with no modifications to your strategy file required.

Also fixes the common make_orders bug where passive quotes are silently dropped,
so your strategy file runs correctly in both the backtester AND on Prosperity.

Usage:
    python3 backtester.py wrapper.py:totest.py
"""

import sys, types, json

# ── jsonpickle shim (so strategies don't need the package installed) ──────────
if 'jsonpickle' not in sys.modules:
    shim = types.ModuleType('jsonpickle')
    shim.encode = json.dumps
    shim.decode = json.loads
    sys.modules['jsonpickle'] = shim

from backtester import TradingAlgorithm, Order as BacktesterOrder
from datamodel import OrderDepth, TradingState


def wrap(trader_class) -> type:
    """
    Returns a TradingAlgorithm subclass that wraps the given Trader class.

    Includes a fix for the make_orders passive-quote bug:
      - Intercepts market_make() calls to capture passive orders that would
        otherwise be created in a local list and thrown away.
      - Injects those orders back into the result transparently.
      - The strategy file itself is completely unchanged.
    """

    # ── Bug fix: subclass Trader to capture passive orders ────────────────────
    class FixedTrader(trader_class):
        def run(self, state):
            passive_orders = {}
            original_mm   = trader_class.market_make

            # Intercept market_make to capture what it appends to its orders list
            def capturing_mm(self_ref, *args, **kwargs):
                # Find the orders list argument (always a list)
                orders = next(a for a in args if isinstance(a, list))
                before = len(orders)
                result = original_mm(self_ref, *args, **kwargs)
                new_passive = orders[before:]
                # Infer product from the new orders if possible
                if new_passive:
                    product = new_passive[0].symbol
                    passive_orders.setdefault(product, []).extend(new_passive)
                return result

            self.market_make = types.MethodType(capturing_mm, self)
            result, conv, data = super().run(state)
            self.market_make = types.MethodType(original_mm, self)

            # Inject captured passive orders into result
            for product, p_orders in passive_orders.items():
                if product in result:
                    result[product].extend(p_orders)

            return result, conv, data

    # ── Backtester adapter ────────────────────────────────────────────────────
    class StrategyWrapper(TradingAlgorithm):
        def __init__(self):
            super().__init__()
            self.trader      = FixedTrader()
            self.trader_data = ""

        def get_orders(self, order_books, positions):
            # Convert backtester OrderBooks → Prosperity OrderDepth objects
            order_depths = {}
            for product, book in order_books.items():
                od = OrderDepth()
                for price, vol in book.bids:
                    od.buy_orders[price] = vol
                for price, vol in book.asks:
                    od.sell_orders[price] = -vol
                order_depths[product] = od

            state = TradingState(
                traderData   = self.trader_data,
                timestamp    = list(order_books.values())[0].timestamp,
                order_depths = order_depths,
                position     = dict(positions),
            )

            result, _, self.trader_data = self.trader.run(state)

            # Convert Prosperity Orders → backtester Orders
            orders = []
            for product, prod_orders in result.items():
                for o in prod_orders:
                    side = 'BUY' if o.quantity > 0 else 'SELL'
                    orders.append(BacktesterOrder(product, side, o.price, abs(o.quantity)))
            return orders

    return StrategyWrapper


ALGORITHM = None  # set dynamically by backtester when using wrapper.py:strategy.py syntax