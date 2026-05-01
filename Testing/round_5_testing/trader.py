"""
Round 5 — v6. Same as v4 + ROBOT_DISHES AR1 restored.

== ML ANALYSIS UPDATE ==

Examined an XGBoost return-prediction model (53 features → 27 selected, 5-fold CV).
Overall OOF IC=0.056, but heavily heterogeneous per product.

Key per-product IC on day 4 (the test surface):
  ROBOT_DISHES                 IC=0.287    sign-hit=0.665    ← star product
  OXYGEN_SHAKE_CHOCOLATE       IC=0.200    sign-hit=0.535
  SNACKPACK_PISTACHIO          IC=0.141
  SNACKPACK_CHOCOLATE          IC=0.137
  SNACKPACK_VANILLA            IC=0.113
  SNACKPACK_STRAWBERRY         IC=0.106
  ROBOT_IRONING                IC=0.106    sign-hit=0.685
  ...
  PEBBLES_*                    IC=0.00-0.03 (basket invariant captures it all)
  MICROCHIP_*                  IC=0.01-0.04 (no exploitable signal)

Comparing simple AR(1) phi against the model IC reveals the model is mostly
re-discovering AR(1) for products where it works — and adding orthogonal
signal mainly for SNACKPACK pair-coupling and OXYGEN_SHAKE_CHOCOLATE:

  ROBOT_DISHES:           AR1 phi=-0.289   model IC=0.287   (just AR1)
  ROBOT_IRONING:          AR1 phi=-0.114   model IC=0.106   (just AR1)
  OXYGEN_EVENING_BREATH:  AR1 phi=-0.077   model IC=0.079   (just AR1)
  OXYGEN_CHOCOLATE:       AR1 phi=-0.101   model IC=0.200   (model > AR1)
  SNACKPACK_PISTACHIO:    AR1 phi=-0.019   model IC=0.141   (model >> AR1)

The 1000-tick live window happened to land on a stretch where ROBOT_DISHES
phi was +0.006 (noise) — that's why I dropped it from v4. But over the FULL
day 4, phi=-0.289. Restoring the AR1 with a moderate phi=-0.23 gains
~+$632 on the 1k window backtest with NO downside on full day.

== WHAT IS NOT IN v6 ==

ML alpha plumbing. Using the model live requires either embedding the XGBoost
model + 27-feature pipeline into trader.py (heavy, risky), or pre-computing
predictions and shipping them as a lookup table (which won't exist for unseen
days). Backtest gain from plumbing alpha into fair: +$1,124 on the 1k window —
not worth the deployment complexity for a 3.7% improvement.

If we DO add ML alpha later, the recommended config is:
  - ALPHA_WEIGHT_GLOBAL = 3.0 to 5.0
  - IC_THRESHOLD = 0.05 (only use alpha for products with at least this IC)
  - High-IC products to prioritize: SNACKPACK family, OXYGEN_CHOCOLATE
  (ROBOT_DISHES already covered by restored AR1)

== ARCHITECTURE (unchanged from v4) ==

For each product: passive MM at one tick inside the L1 book, with
  - inventory skew (1.0 per unit)
  - asymmetric imbalance gate (only blocks side that increases |pos|)
  - momentum filter (block trend-facing side when |EWMA(ret, span 20)| > 3)
  - PEBBLES basket take on XL with edge=1
  - AR(1)-corrected fair for ROBOT_DISHES, ROBOT_IRONING,
    OXYGEN_SHAKE_EVENING_BREATH, OXYGEN_SHAKE_CHOCOLATE

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

# AR(1) return mean-reversion coefficients
# All four phi values match the day-4-wide AR(1) coefficient and the XGBoost
# model's per-product IC (model is essentially relearning AR(1) for these).
AR1_PRODUCTS: Dict[str, float] = {
    "ROBOT_DISHES":                -0.23,   # day-4 phi=-0.289, model IC=0.287
    # ROBOT_IRONING removed: phi=0.0 wins across all days; AR1 correction suppresses profitable fills
    "OXYGEN_SHAKE_EVENING_BREATH": -0.12,   # day-4 phi=-0.077, model IC=0.079
    "OXYGEN_SHAKE_CHOCOLATE":      -0.09,   # day-4 phi=-0.101, model IC=0.200
    "SNACKPACK_CHOCOLATE":         -0.031,  # d2=-0.024 d3=-0.037 d4=-0.032, model IC=0.121
    "SNACKPACK_VANILLA":           -0.027,  # d2=-0.023 d3=-0.027 d4=-0.031, model IC=0.116
    "SNACKPACK_PISTACHIO":         -0.025,  # d2=-0.034 d3=-0.022 d4=-0.019, model IC=0.130
    "SNACKPACK_RASPBERRY":         -0.017,  # d2=-0.007 d3=-0.020 d4=-0.024, model IC=0.104
    "SNACKPACK_STRAWBERRY":        -0.014,  # consistent across days, model IC=0.082
    "MICROCHIP_SQUARE":            -0.022,  # avg phi=-0.022, 16-tick moves → 1.07 ticks at 3σ
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

# Products with persistent intraday drifts that cause consistent losses for passive MM
# (all 3 backtest days negative — slow trends that the EWMA momentum filter cannot catch)
NO_QUOTE_PRODUCTS = {
    "PANEL_1X2",                   # 3/3 days negative; slow drift that EWMA cannot catch
    "SLEEP_POD_LAMB_WOOL",         # 3/3 days negative; consistent upward drift, we accumulate short
    "MICROCHIP_TRIANGLE",          # 2/3 massive losses (-6.5k, -9.7k); zero AR1, large intraday range
    "UV_VISOR_MAGENTA",            # 2/3 negative, progressively worsening (d2=+$198, d3=-$1k, d4=-$2.6k)
    "GALAXY_SOUNDS_SOLAR_FLAMES",  # 1/3 day enormous loss (-10.5k); neg expected PnL vs 2 small gains
    "ROBOT_MOPPING",               # 2/3 negative (-1.5k, -3.8k); neg expected PnL vs d4 gain (+2.5k)
    "ROBOT_IRONING",               # consistent loss 3/3 days; phi=0.0 wins but still net negative
}

# Cross-sectional (rel_ret_1) family correction.
# Multiple regression: next_ret ~ b_own*own_ret + b_fam*family_ret.
# b_fam is the partial coefficient for family mean return, applied additively
# on top of any AR1 correction already in AR1_PRODUCTS.
#
# SNACKPACK: anti-correlated products so family_ret ≈ 0 intra-bar, but fires
# on market-wide SNACKPACK shocks — b_fam ≈ -0.21 to -0.41 (all large-negative).
# OXYGEN_SHAKE: b_fam adds a small correction when the OXYGEN family moves together.
# No family correction for PEBBLES (basket logic already handles it) or ROBOT
# (DISHES is so dominant that family_ret ≈ DISHES_ret/5 and b_fam is negligible).
FAMILY_MEMBERS: Dict[str, List[str]] = {
    "SNACKPACK":   ["SNACKPACK_CHOCOLATE","SNACKPACK_PISTACHIO","SNACKPACK_RASPBERRY",
                    "SNACKPACK_STRAWBERRY","SNACKPACK_VANILLA"],
    "OXYGEN_SHAKE":["OXYGEN_SHAKE_CHOCOLATE","OXYGEN_SHAKE_EVENING_BREATH",
                    "OXYGEN_SHAKE_GARLIC","OXYGEN_SHAKE_MINT","OXYGEN_SHAKE_MORNING_BREATH"],
}

# Maps product → (family_key, b_fam). Tested but backtest showed no gain;
# family correction shifts fair value and suppresses fills more than it helps.
FAM_XS_PHI: Dict[str, Tuple[str, float]] = {}

# Locked tunables
PASSIVE_QUOTE_SIZE      = 10
PASSIVE_INSIDE_TICKS    = 1
INVENTORY_SKEW_PER_UNIT = 1.0
IMBALANCE_THRESHOLD     = 0.41

MOM_SPAN                = 20
MOM_THRESHOLD_TICKS     = 3.0


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
        ewma_a = 2.0 / (MOM_SPAN + 1)
        for sym, m in mids.items():
            if sym in prev_mid:
                ret = m - prev_mid[sym]
                mom_ema[sym] = (1 - ewma_a) * mom_ema.get(sym, 0.0) + ewma_a * ret
            else:
                mom_ema.setdefault(sym, 0.0)

        # --- Build fair values ---
        fair: Dict[str, float] = {}

        # PEBBLES basket: residual on XL leg (XL has 2× return std of others)
        if all(p in mids for p in PEBBLES):
            R = sum(mids[p] for p in PEBBLES) - PEBBLES_TARGET_SUM
            fair[PEBBLES_NOISY_LEG] = mids[PEBBLES_NOISY_LEG] - R
            for p in PEBBLES:
                if p == PEBBLES_NOISY_LEG:
                    continue
                fair[p] = mids[p]

        # Compute family mean returns (needed for cross-sectional correction)
        family_ret: Dict[str, float] = {}
        for fam_key, members in FAMILY_MEMBERS.items():
            rets = [mids[s] - prev_mid[s] for s in members
                    if s in mids and s in prev_mid]
            if len(rets) >= 3:
                family_ret[fam_key] = sum(rets) / len(rets)

        # AR1-corrected fair, then cross-sectional family correction (rel_ret_1)
        for sym, m in mids.items():
            if sym in fair:
                continue
            if sym in AR1_PRODUCTS and sym in prev_mid:
                phi = AR1_PRODUCTS[sym]
                fair[sym] = m + phi * (m - prev_mid[sym])
            else:
                fair[sym] = m

        # Apply family (cross-sectional) correction on top of whatever fair was set
        for sym, (fam_key, b_fam) in FAM_XS_PHI.items():
            if sym in fair and fam_key in family_ret:
                fair[sym] += b_fam * family_ret[fam_key]

        # --- Generate orders ---
        result: Dict[str, List[Order]] = {}
        for sym in ALL_PRODUCTS:
            if sym in NO_QUOTE_PRODUCTS:
                continue
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
            imb = (bv - av) / total if total > 0 else 0.0
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

            # --- Decide which sides we may quote ---
            allow_bid = True
            allow_ask = True

            # Asymmetric imbalance gate
            if pos >= 0 and imb >= IMBALANCE_THRESHOLD:
                allow_bid = False
            if pos <= 0 and imb <= -IMBALANCE_THRESHOLD:
                allow_ask = False

            # Momentum gate: block side facing trend
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