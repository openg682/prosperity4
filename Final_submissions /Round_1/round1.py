from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict
import jsonpickle


# ── ASH_COATED_OSMIUM params ──────────────────────────────────────────────────

OSM_PARAMS = {
    "ema_alpha": 0.05,
    "take_width": 1,
    "disregard_edge": 5,
    "join_edge": 2,
    "default_edge": 4,
    "premium_offset": 15,
    "premium_qty": 1,
    "skew_factor": 1.7,
}

# ── Shared limits ─────────────────────────────────────────────────────────────

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
        ema = alpha * mid + (1 - alpha) * trader_state["osm_ema_fv"]
        trader_state["osm_ema_fv"] = ema
        return round(ema)

    def osm_take_best(
        self,
        fair_value: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
    ):
        lim = self.LIMIT
        buy_volume = sell_volume = 0
        tw = OSM_PARAMS["take_width"]

        if abs(position) >= lim:
            return 0, 0

        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            if best_ask <= fair_value - tw:
                qty = min(-order_depth.sell_orders[best_ask], lim - position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", best_ask, qty))
                    buy_volume += qty

        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            if best_bid >= fair_value + tw:
                qty = min(order_depth.buy_orders[best_bid], lim + position)
                if qty > 0:
                    orders.append(Order("ASH_COATED_OSMIUM", best_bid, -qty))
                    sell_volume += qty

        return buy_volume, sell_volume

    def osm_market_make(
        self,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_volume: int,
        sell_volume: int,
    ):
        lim = self.LIMIT

        buy_qty = lim - (position + buy_volume)
        sell_qty = lim + (position - sell_volume)

        if buy_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", round(bid), buy_qty))

        if sell_qty > 0:
            orders.append(Order("ASH_COATED_OSMIUM", round(ask), -sell_qty))

    def osm_make_orders(
        self,
        order_depth: OrderDepth,
        fair_value: int,
        position: int,
        orders: List[Order],
        buy_volume: int,
        sell_volume: int,
    ):
        p = OSM_PARAMS

        asks_out = [
            px
            for px in order_depth.sell_orders
            if px > fair_value + p["disregard_edge"]
        ]
        bids_out = [
            px
            for px in order_depth.buy_orders
            if px < fair_value - p["disregard_edge"]
        ]

        ask = fair_value + p["default_edge"]
        if asks_out:
            best_ask_out = min(asks_out)
            ask = (
                best_ask_out
                if abs(best_ask_out - fair_value) <= p["join_edge"]
                else best_ask_out - 1
            )

        bid = fair_value - p["default_edge"]
        if bids_out:
            best_bid_out = max(bids_out)
            bid = (
                best_bid_out
                if abs(fair_value - best_bid_out) <= p["join_edge"]
                else best_bid_out + 1
            )

        self.osm_market_make(
            orders, bid, ask, position, buy_volume, sell_volume
        )

    def run_osmium(self, state: TradingState, trader_state: Dict, result: Dict):
        PRODUCT = "ASH_COATED_OSMIUM"

        if PRODUCT not in state.order_depths:
            return

        order_depth = state.order_depths[PRODUCT]
        position = state.position.get(PRODUCT, 0)
        position = max(-self.LIMIT, min(self.LIMIT, position))

        orders = []

        base_fair_value = self.osm_fair_value(order_depth, trader_state)

        # Inventory skew
        skew = int(round((position / self.LIMIT) * OSM_PARAMS["skew_factor"]))
        fair_value = base_fair_value - skew

        # Premium resting orders
        prem_qty = OSM_PARAMS["premium_qty"]
        prem_off = OSM_PARAMS["premium_offset"]

        if prem_qty > 0:
            if self.LIMIT + position > 0:
                orders.append(
                    Order(
                        PRODUCT,
                        fair_value + prem_off,
                        -min(prem_qty, self.LIMIT + position),
                    )
                )

            if self.LIMIT - position > 0:
                orders.append(
                    Order(
                        PRODUCT,
                        fair_value - prem_off,
                        min(prem_qty, self.LIMIT - position),
                    )
                )

        buy_volume, sell_volume = self.osm_take_best(
            fair_value, orders, order_depth, position
        )

        self.osm_make_orders(
            order_depth,
            fair_value,
            position,
            orders,
            buy_volume + prem_qty,
            sell_volume + prem_qty,
        )

        result[PRODUCT] = orders

    # ── INTARIAN_PEPPER_ROOT ──────────────────────────────────────────────────

    def run_pepper(self, state: TradingState, trader_state: Dict, result: Dict):
        PRODUCT = "INTARIAN_PEPPER_ROOT"

        if PRODUCT not in state.order_depths:
            return

        od = state.order_depths[PRODUCT]
        position = state.position.get(PRODUCT, 0)
        orders = []

        if not od.buy_orders or not od.sell_orders:
            result[PRODUCT] = orders
            return

        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        spread = best_ask - best_bid

        bid_vol = od.buy_orders[best_bid]
        ask_vol = abs(od.sell_orders[best_ask])

        total = bid_vol + ask_vol
        imbalance = (bid_vol - ask_vol) / total if total > 0 else 0

        # ── PHASE 1: MAINTAIN 79 CORE ─────────────────────────
        if position < 79:
            orders.append(Order(PRODUCT, best_ask, 79 - position))

        # ── PHASE 2: ROTATING DOOR AT 79 ──────────────────────
        if position == 79:
            # Better fill-rate logic
            if spread >= 2:
                buy_px = best_bid + 1
            else:
                buy_px = best_bid

            orders.append(Order(PRODUCT, buy_px, 1))

            # Keep passive ask almost always
            sell_px = best_ask + 1 if spread >= 2 else best_ask
            orders.append(Order(PRODUCT, sell_px, -1))

        # ── PHASE 3: TOP UNIT EXIT AT 80 ──────────────────────
        elif position == 80:
            # only use imbalance to choose price aggressiveness
            if imbalance > 0.6 and spread >= 2:
                sell_px = best_ask + 1
            else:
                sell_px = best_ask

            orders.append(Order(PRODUCT, sell_px, -1))

        result[PRODUCT] = orders

    # ── Main ──────────────────────────────────────────────────────────────────

    def run(self, state: TradingState):
        result = {}

        trader_state = (
            jsonpickle.decode(state.traderData)
            if state.traderData
            else {}
        )

        self.run_osmium(state, trader_state, result)
        self.run_pepper(state, trader_state, result)

        return result, 0, jsonpickle.encode(trader_state)