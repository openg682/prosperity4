"""
TOMATOES Market Making — Optimised
====================================
Builds on totest.py with the following additions:
  1. Inverted vol scaling — tighter edge in high vol, wider in calm
  2. Large order sniping — trade aggressively into large orders (fire once, 30% limit)
  3. 1-level aggressive take (same as totest.py)

Retained from totest.py:
  - Inventory skewing
  - Hard position clamp
  - EMA fair value tracking

Parameters (tune via param_calibration.py):
  alpha            = 0.25   EMA speed for fair value tracking
  default_edge     = 4      base passive quote distance from fair value
  take_width       = 4      how far through fair value we'll hit aggressively
  skew_factor      = 3      max quote shift at position limit
  vol_window       = 20     rolling window for volatility estimate (ticks)
  vol_scale_factor = 0.5    edge -= vol_scale_factor * rolling_std
  min_edge         = 1      floor on effective edge
  snipe_fraction   = 0.3    max fraction of position limit to commit per snipe
  large_order_z    = 2.0    z-score threshold to flag a large order
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import jsonpickle
from typing import List
from datamodel import OrderDepth, TradingState, Order

PARAMS = {
    "alpha":            0.25,
    "default_edge":     4,
    "take_width":       4,
    "disregard_edge":   9,
    "join_edge":        4,
    "skew_factor":      3,

    # Inverted vol scaling
    "vol_window":       20,
    "vol_scale_factor": 0.5,
    "min_edge":         1,

    # Large order sniping
    "large_order_z":    2.0,
    "snipe_fraction":   0.3,   # max 30% of position limit per snipe
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.params    = PARAMS
        self.LIMIT     = POSITION_LIMIT
        self.ema       = None

        self._mid_history    = []
        self._vol_history    = []
        self._sniped         = False   # True if we already fired this large order event
        self._large_active   = False   # True while large order is still being tracked

    # ── Fair value ────────────────────────────────────────────────────────────

    def update_ema(self, mid_price):
        alpha    = self.params["alpha"]
        self.ema = mid_price if self.ema is None \
                   else alpha * mid_price + (1 - alpha) * self.ema
        return self.ema

    # ── Rolling volatility ────────────────────────────────────────────────────

    def update_rolling_vol(self, mid_price):
        window = self.params["vol_window"]
        self._mid_history.append(mid_price)
        if len(self._mid_history) > window + 1:
            self._mid_history.pop(0)
        if len(self._mid_history) < window + 1:
            return 0.0
        changes  = [self._mid_history[i] - self._mid_history[i - 1]
                    for i in range(1, len(self._mid_history))]
        mean     = sum(changes) / len(changes)
        variance = sum((c - mean) ** 2 for c in changes) / len(changes)
        return variance ** 0.5

    # ── Large order detection ─────────────────────────────────────────────────

    def check_large_order(self, order_depth: OrderDepth):
        """
        Returns (is_large, large_bid_px, large_ask_px).
        Detects which side the large order is on so we can snipe it.
        """
        bid_vol  = sum(order_depth.buy_orders.values())
        ask_vol  = sum(abs(v) for v in order_depth.sell_orders.values())
        total    = bid_vol + ask_vol

        self._vol_history.append(total)
        if len(self._vol_history) < 20:
            return False, None, None
        if len(self._vol_history) > 50:
            self._vol_history.pop(0)

        mean     = sum(self._vol_history) / len(self._vol_history)
        variance = sum((v - mean) ** 2 for v in self._vol_history) / len(self._vol_history)
        std      = variance ** 0.5
        if std == 0:
            return False, None, None

        if (total - mean) / std < self.params["large_order_z"]:
            return False, None, None

        # Identify which side is large
        bid_mean = mean * (bid_vol / total) if total > 0 else 0
        ask_mean = mean * (ask_vol / total) if total > 0 else 0

        large_bid_px = max(order_depth.buy_orders)  if bid_vol > bid_mean * 1.5 and order_depth.buy_orders  else None
        large_ask_px = min(order_depth.sell_orders) if ask_vol > ask_mean * 1.5 and order_depth.sell_orders else None

        return True, large_bid_px, large_ask_px

    # ── Large order snipe — fire once at 30% of limit ────────────────────────

    def snipe_large_order(self, orders, order_depth, position,
                          large_bid_px, large_ask_px):
        """
        Snipe into the large order:
          large bid → sell into it (they want to buy, give them stock)
          large ask → buy into it (they want to sell, take their stock)
        Commits up to snipe_fraction * LIMIT per snipe, fired once per event.
        """
        snipe_qty = round(self.LIMIT * self.params["snipe_fraction"])
        bv = sv = 0

        # Large ask — buy into it
        if large_ask_px is not None and order_depth.sell_orders:
            available = -order_depth.sell_orders.get(large_ask_px, 0)
            qty       = min(available, snipe_qty, self.LIMIT - position)
            if qty > 0:
                orders.append(Order("TOMATOES", large_ask_px, qty))
                bv += qty

        # Large bid — sell into it
        if large_bid_px is not None and order_depth.buy_orders:
            available = order_depth.buy_orders.get(large_bid_px, 0)
            qty       = min(available, snipe_qty, self.LIMIT + position)
            if qty > 0:
                orders.append(Order("TOMATOES", large_bid_px, -qty))
                sv += qty

        return bv, sv

    # ── Effective edge — inverted vol scaling ─────────────────────────────────

    def compute_effective_edge(self, rolling_std):
        p        = self.params
        vol_edge = p["default_edge"] - round(rolling_std * p["vol_scale_factor"])
        return max(p["min_edge"], vol_edge)

    # ── Aggressive take — 1 level ─────────────────────────────────────────────

    def take_best_orders(self, fair_value, orders, order_depth, position):
        tw  = self.params["take_width"]
        lim = self.LIMIT
        bv  = sv = 0

        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - tw:
                qty = min(-order_depth.sell_orders[ba], lim - position)
                if qty > 0:
                    orders.append(Order("TOMATOES", ba, qty))
                    bv += qty

        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + tw:
                qty = min(order_depth.buy_orders[bb], lim + position)
                if qty > 0:
                    orders.append(Order("TOMATOES", bb, -qty))
                    sv += qty

        return bv, sv

    # ── Passive market making with skew ───────────────────────────────────────

    def market_make(self, orders, bid, ask, position, buy_volume, sell_volume):
        lim = self.LIMIT
        bq  = lim - (position + buy_volume)
        sq  = lim + (position - sell_volume)
        if bq > 0: orders.append(Order("TOMATOES", round(bid),  bq))
        if sq > 0: orders.append(Order("TOMATOES", round(ask), -sq))

    def make_orders(self, order_depth, fair_value, position,
                    orders, buy_volume, sell_volume, effective_edge):
        p = self.params

        asks_outside = [pr for pr in order_depth.sell_orders
                        if pr > fair_value + p["disregard_edge"]]
        bids_outside = [pr for pr in order_depth.buy_orders
                        if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + effective_edge
        if asks_outside:
            ba  = min(asks_outside)
            ask = ba if abs(ba - fair_value) <= p["join_edge"] else ba - 1

        bid = fair_value - effective_edge
        if bids_outside:
            bb  = max(bids_outside)
            bid = bb if abs(fair_value - bb) <= p["join_edge"] else bb + 1

        # Inventory skewing
        skew = (position / self.LIMIT) * p["skew_factor"]
        bid  = bid - skew
        ask  = ask - skew

        self.market_make(orders, bid, ask, position, buy_volume, sell_volume)

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result      = {}
        trader_data = jsonpickle.decode(state.traderData) if state.traderData else {}

        # Restore persisted state
        self.ema              = trader_data.get("ema", None)
        self._mid_history     = trader_data.get("mid_history", [])
        self._vol_history     = trader_data.get("vol_history", [])
        self._sniped          = trader_data.get("sniped", False)
        self._large_active    = trader_data.get("large_active", False)

        if "TOMATOES" in state.order_depths:
            orders      : List[Order] = []
            order_depth  = state.order_depths["TOMATOES"]

            # Hard position clamp
            position = state.position.get("TOMATOES", 0)
            position = max(-self.LIMIT, min(self.LIMIT, position))

            # Mid price
            if order_depth.buy_orders and order_depth.sell_orders:
                mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
            elif order_depth.buy_orders:
                mid = float(max(order_depth.buy_orders))
            elif order_depth.sell_orders:
                mid = float(min(order_depth.sell_orders))
            else:
                mid = self.ema or 5000

            fair_value  = self.update_ema(mid)
            rolling_std = self.update_rolling_vol(mid)

            # Large order detection
            is_large, large_bid_px, large_ask_px = self.check_large_order(order_depth)

            bv = sv = 0

            if is_large:
                if not self._large_active:
                    # New large order event — fire snipe once
                    self._large_active = True
                    self._sniped       = False

                if not self._sniped:
                    bv, sv = self.snipe_large_order(
                        orders, order_depth, position, large_bid_px, large_ask_px
                    )
                    self._sniped = True
            else:
                # Large order gone — reset for next event
                self._large_active = False
                self._sniped       = False

            # Standard aggressive take on remaining capacity
            tb, ts = self.take_best_orders(fair_value, orders, order_depth, position)
            bv += tb; sv += ts

            # Passive quotes with inverted vol edge
            effective_edge = self.compute_effective_edge(rolling_std)
            self.make_orders(order_depth, fair_value, position,
                             orders, bv, sv, effective_edge)

            result["TOMATOES"] = orders

        # Persist state
        trader_data["ema"]           = self.ema
        trader_data["mid_history"]   = self._mid_history
        trader_data["vol_history"]   = self._vol_history
        trader_data["sniped"]        = self._sniped
        trader_data["large_active"]  = self._large_active

        return result, 0, jsonpickle.encode(trader_data)