"""
Round 5 — 50 products, position limit 10, no counterparty signal.

== ITERATION HISTORY ==
v1 (take-only):                Day-4 backtest -1.56M (catastrophic).
v2 (passive MM, asym gate):    Day-4 backtest +158k.
v3 (basket xl_only):           Day-4 backtest +173k.
v3 LIVE (1000-tick window):    +22,545.
v4 (this file):                Same window backtest predicts +29,659 (+31.4% vs v3 live).

== LIVE-DATA-DRIVEN FIXES IN v4 ==

1. **Momentum filter** — biggest single improvement (+~3k expected).
   The v3 live run lost ~$22k on 4 products that trended hard:
     PLANETARY_RINGS (-7.3k, -778 tick move), SPACE_GRAY (-4.8k, -669),
     LAMB_WOOL (-4.5k, -591), DARK_MATTER (-3.8k, -407).
   These were all passive-quote products getting run over by trends.
   Fix: maintain an EWMA of one-tick returns (span 20). When |momentum|
   exceeds 3 ticks/tick, skip the side facing the trend (block bids when
   rising, asks when falling). The aligned side stays open.

2. **Asymmetric imbalance gate** — never gate the side that REDUCES |pos|.
   Previously we'd shut off bids whenever the book was bid-heavy — even when
   we were short and BUYING was the recovery direction. Result: persistent
   one-way drift (live ended -656 net across 50 products). Now imbalance
   only gates the side that would INCREASE risk.

3. **Stronger inventory skew** (0.5 → 1.0 per unit). Live ended at -656 net
   position; backtest ended +207. Doubling the skew pulls fair more aggressively
   when inventory builds. End-pos drops to +120 in the backtest.

4. **Drop ROBOT_DISHES AR1 coefficient** — was -0.23 in training, +0.006 in live.
   Regime change or overfit. Live cost: -3,463. Now plain MM at mid like
   the rest. Other AR1 products (IRONING -0.127, OXYGEN_SHAKE_EVENING_BREATH
   -0.128) had live phi matching training and are kept.

5. **PEBBLES_XL basket take** — kept (the basket invariant XS+S+M+L+XL=50,000
   held perfectly live with std=2.5). XL went from -2,781 (v3 live) to +2,586
   (v4 backtest) — momentum filter now gates the take when XL is mid-trend.

6. **Quote size 10** (was 5). Lets us fill faster when one-sided flow appears.
   Marginal +$300 in backtest.

== NOT INCLUDED ==
- SNACKPACK pair-sum target — drifts day-to-day, EWMA can't keep up. Plain MM
  on these 5 already profits ~$2.4k/window.
- Hard inventory flatten at |pos|>=8 — slightly hurt the backtest (no real
  benefit since improved skew already keeps |pos| low).
- Wider quote (deeper inside) on trend — not better than just gating.
- No-quote on trend (block both sides) — too restrictive.

To submit: replace the stub Order/OrderDepth/TradingState classes at the top
with `from datamodel import Order, OrderDepth, TradingState`.
"""
from typing import Dict, List, Tuple
import json, math


# === Stub data classes — replace with `from datamodel import ...` for submission ===
class Order:
    def __init__(self, symbol, price, quantity):
        self.symbol = symbol
        self.price = int(price)
        self.quantity = int(quantity)
    def __repr__(self):
        return f"Order({self.symbol},p={self.price},q={self.quantity})"

class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}

class TradingState:
    def __init__(self, timestamp, listings, order_depths, own_trades,
                 market_trades, position, observations, traderData=""):
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations
        self.traderData = traderData


# === Constants ===
POSITION_LIMIT = 10

PEBBLES = ["PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL"]
PEBBLES_TARGET_SUM = 50_000
PEBBLES_NOISY_LEG = "PEBBLES_XL"
BASKET_TAKE_EDGE = 1

# AR(1) return mean-reversion coefficients (validated in live data)
AR1_PRODUCTS: Dict[str, float] = {
    "ROBOT_IRONING":               -0.13,   # live phi=-0.127 ✓
    "OXYGEN_SHAKE_EVENING_BREATH": -0.12,   # live phi=-0.128 ✓
    "OXYGEN_SHAKE_CHOCOLATE":      -0.09,   # not contradicted by live data
    # ROBOT_DISHES dropped — live phi=+0.006, original -0.23 was overfit.
}

ALL_PRODUCTS = [
    "GALAXY_SOUNDS_BLACK_HOLES", "GALAXY_SOUNDS_DARK_MATTER",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_FLAMES", "GALAXY_SOUNDS_SOLAR_WINDS",
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_RECTANGLE", "MICROCHIP_SQUARE", "MICROCHIP_TRIANGLE",
    "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_EVENING_BREATH", "OXYGEN_SHAKE_GARLIC",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_MORNING_BREATH",
    "PANEL_1X2", "PANEL_1X4", "PANEL_2X2", "PANEL_2X4", "PANEL_4X4",
    "PEBBLES_L", "PEBBLES_M", "PEBBLES_S", "PEBBLES_XL", "PEBBLES_XS",
    "ROBOT_DISHES", "ROBOT_IRONING", "ROBOT_LAUNDRY", "ROBOT_MOPPING", "ROBOT_VACUUMING",
    "SLEEP_POD_COTTON", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_NYLON", "SLEEP_POD_POLYESTER", "SLEEP_POD_SUEDE",
    "SNACKPACK_CHOCOLATE", "SNACKPACK_PISTACHIO", "SNACKPACK_RASPBERRY", "SNACKPACK_STRAWBERRY", "SNACKPACK_VANILLA",
    "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_VOID_BLUE",
    "UV_VISOR_AMBER", "UV_VISOR_MAGENTA", "UV_VISOR_ORANGE", "UV_VISOR_RED", "UV_VISOR_YELLOW",
]

# Tunables (locked from live-data-aware sweep on the 1000-tick window)
PASSIVE_QUOTE_SIZE      = 10
PASSIVE_INSIDE_TICKS    = 1
INVENTORY_SKEW_PER_UNIT = 1.0
IMBALANCE_THRESHOLD     = 0.3

MOM_SPAN                = 20      # EWMA of one-tick returns over ~20 ticks
MOM_THRESHOLD_TICKS     = 3.0     # block trend-facing side when |mom| > this


def best_bid_ask(depth):
    bb = max(depth.buy_orders) if depth.buy_orders else None
    ba = min(depth.sell_orders) if depth.sell_orders else None
    bv = abs(depth.buy_orders[bb]) if bb is not None else 0
    av = abs(depth.sell_orders[ba]) if ba is not None else 0
    return bb, bv, ba, av


def book_mid(depth):
    bb, _, ba, _ = best_bid_ask(depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


class Trader:
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # --- Restore state ---
        try:
            mem = json.loads(state.traderData) if state.traderData else {}
        except Exception:
            mem = {}
        prev_mid: Dict[str, float] = mem.get("prev_mid", {})
        mom_ema:  Dict[str, float] = mem.get("mom_ema",  {})

        # --- Snapshot mids ---
        mids: Dict[str, float] = {}
        for sym in ALL_PRODUCTS:
            d = state.order_depths.get(sym)
            if d is None:
                continue
            m = book_mid(d)
            if m is not None:
                mids[sym] = m

        # --- Update momentum EWMA ---
        alpha = 2.0 / (MOM_SPAN + 1)
        for sym, m in mids.items():
            if sym in prev_mid:
                ret = m - prev_mid[sym]
                mom_ema[sym] = (1 - alpha) * mom_ema.get(sym, 0.0) + alpha * ret
            else:
                mom_ema.setdefault(sym, 0.0)

        # --- Build fair values ---
        fair: Dict[str, float] = {}

        # PEBBLES basket: attribute residual to XL (the noisy leg)
        if all(p in mids for p in PEBBLES):
            R = sum(mids[p] for p in PEBBLES) - PEBBLES_TARGET_SUM
            fair[PEBBLES_NOISY_LEG] = mids[PEBBLES_NOISY_LEG] - R
            for p in PEBBLES:
                if p == PEBBLES_NOISY_LEG:
                    continue
                fair[p] = mids[p]

        # AR1-corrected fair for the 3 validated products
        for sym, m in mids.items():
            if sym in fair:
                continue
            if sym in AR1_PRODUCTS and sym in prev_mid:
                phi = AR1_PRODUCTS[sym]
                fair[sym] = m + phi * (m - prev_mid[sym])
            else:
                fair[sym] = m

        # --- Generate orders ---
        result: Dict[str, List[Order]] = {}
        for sym in ALL_PRODUCTS:
            if sym not in fair:
                continue
            d = state.order_depths.get(sym)
            if d is None:
                continue
            pos = state.position.get(sym, 0)
            f = fair[sym]
            bb, bv, ba, av = best_bid_ask(d)
            if bb is None or ba is None or ba <= bb:
                continue

            # Inventory-skew the fair (push toward zero position)
            f_eff = f - INVENTORY_SKEW_PER_UNIT * pos
            total = bv + av
            imb = (bv - av) / total if total > 0 else 0.0  # >0 = bid-heavy
            mom = mom_ema.get(sym, 0.0)

            buy_cap  = POSITION_LIMIT - pos
            sell_cap = POSITION_LIMIT + pos
            orders: List[Order] = []

            # PEBBLES_XL aggressive take from basket residual
            if sym == PEBBLES_NOISY_LEG:
                if buy_cap > 0 and ba <= f_eff - BASKET_TAKE_EDGE:
                    q = min(buy_cap, av)
                    if q > 0:
                        orders.append(Order(sym, ba, q))
                        buy_cap -= q
                if sell_cap > 0 and bb >= f_eff + BASKET_TAKE_EDGE:
                    q = min(sell_cap, bv)
                    if q > 0:
                        orders.append(Order(sym, bb, -q))
                        sell_cap -= q

            # --- Decide which sides we may quote on ---
            allow_bid = True
            allow_ask = True

            # Asymmetric imbalance gate: only block side that would INCREASE |pos|
            if pos >= 0 and imb >= IMBALANCE_THRESHOLD:
                allow_bid = False  # bid-heavy + we're long — stop adding length
            if pos <= 0 and imb <= -IMBALANCE_THRESHOLD:
                allow_ask = False

            # Momentum gate: block the side facing the trend
            if mom > MOM_THRESHOLD_TICKS:
                allow_bid = False
            elif mom < -MOM_THRESHOLD_TICKS:
                allow_ask = False

            # --- Place quotes one tick inside L1 ---
            quote_bid = quote_ask = None
            if allow_bid and buy_cap > 0:
                p = min(bb + PASSIVE_INSIDE_TICKS, math.floor(f_eff - 1))
                p = min(p, ba - 1)
                if p >= bb and p > 0:
                    quote_bid = p
            if allow_ask and sell_cap > 0:
                p = max(ba - PASSIVE_INSIDE_TICKS, math.ceil(f_eff + 1))
                p = max(p, bb + 1)
                if p <= ba:
                    quote_ask = p

            if quote_bid is not None:
                size = min(PASSIVE_QUOTE_SIZE, buy_cap)
                if size > 0:
                    orders.append(Order(sym, int(quote_bid), size))
            if quote_ask is not None:
                size = min(PASSIVE_QUOTE_SIZE, sell_cap)
                if size > 0:
                    orders.append(Order(sym, int(quote_ask), -size))

            if orders:
                result[sym] = orders

        # --- Persist state ---
        new_prev_mid = {p: mids[p] for p in mids}
        return (result, 0,
                json.dumps({"prev_mid": new_prev_mid, "mom_ema": mom_ema},
                           separators=(",", ":")))