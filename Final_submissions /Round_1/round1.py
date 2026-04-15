from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle

# ── ASH_COATED_OSMIUM params ──────────────────────────────────────────────────

OSM_PARAMS = {
    "ema_alpha":       0.05,
    "take_width":      1,
    "disregard_edge":  5,
    "join_edge":       2,
    "default_edge":    4,
    "premium_offset":  15,
    "premium_qty":     1,
}

# ── INTARIAN_PEPPER_ROOT params ───────────────────────────────────────────────

IPR_STOP_LOSS = -3000   # unwind immediately if MTM drops below this
                        # worst historical dip = -862; -3000 = clear reversal signal
                        # IMC closes open positions at fair value so no exit needed

POSITION_LIMIT = 80


class Trader:
    def __init__(self):
        self.LIMIT = POSITION_LIMIT

    # ── ASH_COATED_OSMIUM ─────────────────────────────────────────────────────

    def osm_fair_value(self, order_depth: OrderDepth, trader_state: Dict) -> int:
        if order_depth.buy_orders and order_depth.sell_orders:
            mid = (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            mid = float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            mid = float(min(order_depth.sell_orders))
        else:
            return round(trader_state.get("osm_ema_fv", 10000))

        if "osm_ema_fv" not in trader_state:
            trader_state["osm_ema_fv"] = mid
            return round(mid)

        alpha = OSM_PARAMS["ema_alpha"]
        ema   = alpha * mid + (1 - alpha) * trader_state["osm_ema_fv"]
        trader_state["osm_ema_fv"] = ema
        return round(ema)

    def osm_take_best(self, fair_value: int, orders: List[Order],
                      order_depth: OrderDepth, position: int):
        lim = self.LIMIT; bv = sv = 0
        tw  = OSM_PARAMS["take_width"]
        if abs(position) >= lim: return 0, 0
        if order_depth.sell_orders:
            ba = min(order_depth.sell_orders)
            if ba <= fair_value - tw:
                qty = min(-order_depth.sell_orders[ba], lim - position)
                if qty > 0: orders.append(Order("ASH_COATED_OSMIUM", ba, qty)); bv += qty
        if order_depth.buy_orders:
            bb = max(order_depth.buy_orders)
            if bb >= fair_value + tw:
                qty = min(order_depth.buy_orders[bb], lim + position)
                if qty > 0: orders.append(Order("ASH_COATED_OSMIUM", bb, -qty)); sv += qty
        return bv, sv

    def osm_market_make(self, orders: List[Order], bid: int, ask: int,
                        position: int, buy_volume: int, sell_volume: int):
        lim = self.LIMIT
        bq  = lim - (position + buy_volume)
        sq  = lim + (position - sell_volume)
        if bq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(bid),  bq))
        if sq > 0: orders.append(Order("ASH_COATED_OSMIUM", round(ask), -sq))

    def osm_make_orders(self, order_depth: OrderDepth, fair_value: int,
                        position: int, orders: List[Order],
                        buy_volume: int, sell_volume: int):
        p = OSM_PARAMS
        asks_out = [pr for pr in order_depth.sell_orders if pr > fair_value + p["disregard_edge"]]
        bids_out = [pr for pr in order_depth.buy_orders  if pr < fair_value - p["disregard_edge"]]

        ask = fair_value + p["default_edge"]
        if asks_out:
            ba  = min(asks_out)
            ask = ba if abs(ba - fair_value) <= p["join_edge"] else ba - 1

        bid = fair_value - p["default_edge"]
        if bids_out:
            bb  = max(bids_out)
            bid = bb if abs(fair_value - bb) <= p["join_edge"] else bb + 1

        self.osm_market_make(orders, bid, ask, position, buy_volume, sell_volume)

    def run_osmium(self, state: TradingState, trader_state: Dict, result: Dict):
        if "ASH_COATED_OSMIUM" not in state.order_depths:
            return
        orders      = []
        order_depth = state.order_depths["ASH_COATED_OSMIUM"]
        position    = state.position.get("ASH_COATED_OSMIUM", 0)
        position    = max(-self.LIMIT, min(self.LIMIT, position))
        fair_value  = self.osm_fair_value(order_depth, trader_state)

        prem_qty = OSM_PARAMS["premium_qty"]
        prem_off = OSM_PARAMS["premium_offset"]
        if prem_qty > 0:
            if self.LIMIT + position > 0:
                orders.append(Order("ASH_COATED_OSMIUM", fair_value + prem_off,
                                    -min(prem_qty, self.LIMIT + position)))
            if self.LIMIT - position > 0:
                orders.append(Order("ASH_COATED_OSMIUM", fair_value - prem_off,
                                     min(prem_qty, self.LIMIT - position)))

        bv, sv = self.osm_take_best(fair_value, orders, order_depth, position)
        self.osm_make_orders(order_depth, fair_value, position, orders,
                             bv + prem_qty, sv + prem_qty)
        result["ASH_COATED_OSMIUM"] = orders

    # ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────────

    def ipr_get_mid(self, order_depth: OrderDepth, fallback: float) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            return (max(order_depth.buy_orders) + min(order_depth.sell_orders)) / 2.0
        elif order_depth.buy_orders:
            return float(max(order_depth.buy_orders))
        elif order_depth.sell_orders:
            return float(min(order_depth.sell_orders))
        return fallback

    def run_pepper(self, state: TradingState, trader_state: Dict, result: Dict):
        if "INTARIAN_PEPPER_ROOT" not in state.order_depths:
            return
        orders      = []
        order_depth = state.order_depths["INTARIAN_PEPPER_ROOT"]
        position    = state.position.get("INTARIAN_PEPPER_ROOT", 0)
        position    = max(-self.LIMIT, min(self.LIMIT, position))
        lim         = self.LIMIT

        entry_price = trader_state.get("ipr_entry_price", None)
        stopped_out = trader_state.get("ipr_stopped_out", False)

        # Stop-loss: if MTM drops below threshold, unwind immediately
        if entry_price is not None and position > 0 and not stopped_out:
            mid = self.ipr_get_mid(order_depth, float(entry_price))
            if position * (mid - entry_price) < IPR_STOP_LOSS:
                stopped_out = True
                trader_state["ipr_stopped_out"] = True

        if stopped_out:
            # Emergency unwind: hit bids to flatten
            want = position
            for px in sorted(order_depth.buy_orders.keys(), reverse=True):
                if want <= 0: break
                qty = min(order_depth.buy_orders[px], want)
                if qty > 0:
                    orders.append(Order("INTARIAN_PEPPER_ROOT", px, -qty))
                    want -= qty
        else:
            # Accumulate to max long: single order at best ask each tick.
            # Only hits level 1 → cheaper avg entry than walking all levels.
            # IMC closes open positions at fair value so no explicit exit needed.
            want     = lim - position
            best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
            if want > 0 and best_ask is not None:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, want))
                if entry_price is None:
                    trader_state["ipr_entry_price"] = float(best_ask)

        trader_state["ipr_stopped_out"] = stopped_out
        result["INTARIAN_PEPPER_ROOT"] = orders

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result       = {}
        trader_state = jsonpickle.decode(state.traderData) if state.traderData else {}

        self.run_osmium(state, trader_state, result)
        self.run_pepper(state, trader_state, result)

        return result, 0, jsonpickle.encode(trader_state)