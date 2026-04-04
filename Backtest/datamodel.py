"""
Stub datamodel — mirrors the real Prosperity datamodel API
so strategy files import cleanly in the backtester environment.
"""
from typing import Dict, List
from dataclasses import dataclass, field


class Order:
    def __init__(self, symbol: str, price: int, quantity: int):
        self.symbol   = symbol
        self.price    = price
        self.quantity = quantity   # positive = BUY, negative = SELL

    def __repr__(self):
        side = "BUY" if self.quantity > 0 else "SELL"
        return f"Order({self.symbol} {side} {abs(self.quantity)}@{self.price})"


class OrderDepth:
    def __init__(self):
        self.buy_orders:  Dict[int, int] = {}   # price → +volume
        self.sell_orders: Dict[int, int] = {}   # price → -volume (negative!)


@dataclass
class TradingState:
    traderData:   str
    timestamp:    int
    order_depths: Dict[str, OrderDepth]
    position:     Dict[str, int]
    own_trades:   Dict = field(default_factory=dict)
    market_trades: Dict = field(default_factory=dict)
    listings:     Dict = field(default_factory=dict)
    observations: Dict = field(default_factory=dict)