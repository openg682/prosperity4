"""
TOMATOES — Market Making + Mean Reversion Directional Overlay
==============================================================
Structure:
  - 50 units always reserved for market making
  - 30 units always reserved for directional mean reversion

Directional logic:
  - Price < 4988 → enter long regime
  - Price > 4990 → exit long regime back to neutral
  - Price > 5002 → enter short regime
  - Price < 5000 → exit short regime back to neutral
  - Neutral → unwind directional position to flat
  - Exponential scaling: 0→30 units over 10 ticks outside boundary
  - Trailing stop: cut directional if PnL drops 150 below its peak
  - Cooldown: 25 ticks before re-entering same regime after trailing stop

Market making:
  - Long regime  → only quote asks
  - Short regime → only quote bids
  - Neutral      → normal symmetric quoting
"""

import jsonpickle
from typing import List
from datamodel import OrderDepth, TradingState, Order

PARAMS = {
    "fair_value_seed":  5000,
    "ema_alpha":        0.25,

    # Market making
    "default_edge":     5,
    "take_width":       4,
    "disregard_edge":   10,
    "join_edge":        4,
    "skew_factor":      3,
    "mm_limit":         50,

    # Directional
    "dir_limit":        30,
    "lower_entry":      4988,
    "lower_exit":       4990,
    "upper_entry":      5002,
    "upper_exit":       5000,
    "scale_range":      10,
    "exp_base":         2.5,

    # Trailing stop
    "trail_amount":     150,   # cut if PnL drops 150 below peak directional PnL
    "cooldown_ticks":   25,
}

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    # ── Fair value ────────────────────────────────────────────────────────────

    def get_fair_value(self, order_depth, trader_state):
        alpha = PARAMS["ema_alpha"]
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            mid = PARAMS["fair_value_seed"]
        prev = trader_state.get("ema", PARAMS["fair_value_seed"])
        ema  = alpha * mid + (1 - alpha) * prev
        trader_state["ema"] = ema
        return ema

    # ── Directional target ────────────────────────────────────────────────────

    def directional_target(self, fair_value, regime):
        p     = PARAMS
        base  = p["exp_base"]
        scale = p["scale_range"]
        lim   = p["dir_limit"]

        if regime == "long":
            dist     = max(0, min(p["lower_entry"] - fair_value, scale))
            fraction = (base ** (dist / scale) - 1) / (base - 1) if dist > 0 else 0
            return round(fraction * lim)

        elif regime == "short":
            dist     = max(0, min(fair_value - p["upper_entry"], scale))
            fraction = (base ** (dist / scale) - 1) / (base - 1) if dist > 0 else 0
            return -round(fraction * lim)

        return 0

    # ── Regime ────────────────────────────────────────────────────────────────

    def get_regime(self, fair_value, prev_regime):
        p = PARAMS
        if prev_regime == "long":
            return "long" if fair_value < p["lower_exit"] else "neutral"
        elif prev_regime == "short":
            return "short" if fair_value > p["upper_exit"] else "neutral"
        else:
            if fair_value < p["lower_entry"]:
                return "long"
            elif fair_value > p["upper_entry"]:
                return "short"
            return "neutral"

    # ── Directional orders ────────────────────────────────────────────────────

    def directional_orders(self, order_depth, dir_position, target, total_position):
        orders = []
        delta  = target - dir_position

        if delta > 0:
            if order_depth.sell_orders:
                ba  = min(order_depth.sell_orders)
                qty = min(delta, -order_depth.sell_orders[ba], self.LIMIT - total_position)
                if qty > 0:
                    orders.append(Order("TOMATOES", ba, qty))

        elif delta < 0:
            if order_depth.buy_orders:
                bb  = max(order_depth.buy_orders)
                qty = min(-delta, order_depth.buy_orders[bb], self.LIMIT + total_position)
                if qty > 0:
                    orders.append(Order("TOMATOES", bb, -qty))

        return orders

    # ── Market making ─────────────────────────────────────────────────────────

    def take_best_orders(self, fair_value, orders, order_depth, total_position, regime):
        p   = PARAMS
        lim = self.LIMIT
        bv  = sv = 0

        if regime != "short" and order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - p["take_width"]:
                qty = min(-order_depth.sell_orders[ba], lim - total_position)
                if qty > 0:
                    orders.append(Order("TOMATOES", ba, qty))
                    bv += qty

        if regime != "long" and order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + p["take_width"]:
                qty = min(order_depth.buy_orders[bb], lim + total_position)
                if qty > 0:
                    orders.append(Order("TOMATOES", bb, -qty))
                    sv += qty

        return bv, sv

    def make_orders(self, order_depth, fair_value, total_position,
                    orders, buy_volume, sell_volume, regime):
        p        = PARAMS
        lim      = self.LIMIT
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

        skew = (total_position / lim) * p["skew_factor"]
        bid  = round(bid - skew)
        ask  = round(ask - skew)

        bq = lim - (total_position + buy_volume)
        sq = lim + (total_position - sell_volume)

        if regime != "long"  and bq > 0:
            orders.append(Order("TOMATOES", bid,  bq))
        if regime != "short" and sq > 0:
            orders.append(Order("TOMATOES", ask, -sq))

    # ── Main run ──────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        if "TOMATOES" in state.order_depths:
            orders      = []
            order_depth = state.order_depths["TOMATOES"]

            # Total position
            total_position = state.position.get("TOMATOES", 0)
            total_position = max(-self.LIMIT, min(self.LIMIT, total_position))

            # Fair value and regime
            fair_value  = self.get_fair_value(order_depth, trader_state)
            prev_regime = trader_state.get("regime", "neutral")
            regime      = self.get_regime(fair_value, prev_regime)
            target      = self.directional_target(fair_value, regime)
            trader_state["regime"] = regime

            # Restore directional state
            dir_position    = trader_state.get("dir_position", 0)
            dir_position    = max(-PARAMS["dir_limit"], min(PARAMS["dir_limit"], dir_position))
            dir_entry_price = trader_state.get("dir_entry_price", None)
            cooldown        = trader_state.get("cooldown", 0)
            stopped_regime  = trader_state.get("stopped_regime", None)
            peak_dir_pnl    = trader_state.get("peak_dir_pnl", 0)

            # ── Trailing stop ─────────────────────────────────────────────────
            stopped = False
            if dir_position != 0 and dir_entry_price is not None:
                dir_pnl = dir_position * (fair_value - dir_entry_price)

                # Update peak
                if dir_pnl > peak_dir_pnl:
                    peak_dir_pnl = dir_pnl

                # Fire if dropped 150 below peak
                if peak_dir_pnl - dir_pnl >= PARAMS["trail_amount"]:
                    # Cut directional position aggressively
                    if dir_position > 0 and order_depth.buy_orders:
                        bb  = max(order_depth.buy_orders)
                        qty = min(dir_position,
                                  order_depth.buy_orders[bb],
                                  self.LIMIT + total_position)
                        if qty > 0:
                            orders.append(Order("TOMATOES", bb, -qty))
                            dir_position -= qty
                    elif dir_position < 0 and order_depth.sell_orders:
                        ba  = min(order_depth.sell_orders)
                        qty = min(-dir_position,
                                  -order_depth.sell_orders[ba],
                                  self.LIMIT - total_position)
                        if qty > 0:
                            orders.append(Order("TOMATOES", ba, qty))
                            dir_position += qty
                    stopped         = True
                    stopped_regime  = regime
                    cooldown        = PARAMS["cooldown_ticks"]
                    dir_entry_price = None
                    peak_dir_pnl    = 0

            # ── Cooldown tick ─────────────────────────────────────────────────
            if cooldown > 0:
                cooldown -= 1

            # ── Directional orders ────────────────────────────────────────────
            skip = stopped or (cooldown > 0 and regime == stopped_regime)

            if not skip:
                dir_orders = self.directional_orders(
                    order_depth, dir_position, target, total_position
                )
                orders.extend(dir_orders)
                dir_delta = sum(o.quantity for o in dir_orders)
                new_dir   = dir_position + dir_delta

                # Track entry price
                if dir_position == 0 and new_dir != 0:
                    dir_entry_price = fair_value
                    peak_dir_pnl    = 0
                elif new_dir == 0:
                    dir_entry_price = None
                    peak_dir_pnl    = 0

                dir_position = new_dir

            # ── Market making ─────────────────────────────────────────────────
            bv, sv = self.take_best_orders(
                fair_value, orders, order_depth, total_position, regime
            )
            self.make_orders(
                order_depth, fair_value, total_position, orders, bv, sv, regime
            )

            result["TOMATOES"] = orders

            # Persist all state
            trader_state["dir_position"]    = dir_position
            trader_state["dir_entry_price"] = dir_entry_price
            trader_state["cooldown"]        = cooldown
            trader_state["stopped_regime"]  = stopped_regime
            trader_state["peak_dir_pnl"]    = peak_dir_pnl

        traderData = jsonpickle.encode(trader_state)
        return result, 0, traderData