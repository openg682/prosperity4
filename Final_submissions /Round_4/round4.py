import json
import math
from datamodel import OrderDepth, TradingState, Order
from typing import Dict, List

# ─────────────────────────────────────────────
# Limits
# ─────────────────────────────────────────────
POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
    "VEV_4000": 300, "VEV_4500": 300, "VEV_5000": 300,
    "VEV_5100": 300, "VEV_5200": 300, "VEV_5300": 300,
    "VEV_5400": 300, "VEV_5500": 300,
    "VEV_6000": 300, "VEV_6500": 300,
}

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
SQRT2 = 1.4142135623730951

OPT_LIMIT = 300
VFE_LIMIT = 200

ITM_LIMITS   = {5000: 60, 5100: 50, 5200: 40}
ITM_MM_EDGE  = 2.5
ITM_SNIPE    = 3.5

DELTA_CAP = 130

SIGMA_SEED   = 0.0615
IV_EMA_ALPHA = 0.24
SIGMA_LO = 0.04
SIGMA_HI = 0.09

T_START = 0.3256
T_DECAY = 0.1080
T_TOTAL_TS = 1_711_027
T_MIN = 0.01

# HYDROGEL passive MM
HYD_LIMIT      = 200
HYD_EDGE       = 2
HYD_SIZE       = 15
HYD_SKEW_K     = 6
HYD_SWEEP_EDGE = 8

# Mark-take parameters
# From log analysis:
#   Mark 38 (HYDROGEL) sells at fair-2 to fair-4, buys at fair+1 to fair+2
#   Mark 55 (VFE)      sells at fair-2.5 to fair-3, buys at fair+0.5 to fair+3
# Real MMs quote at the outer ±8 levels with size 20-30, so anything inside
# at smaller size is almost certainly a Mark.
HYD_TAKE_ASK_EDGE = 1.5    # take any ask < fair - 1.5 (covers all Mark sells, edge ≥2)
HYD_TAKE_BID_EDGE = 0.5    # hit any bid > fair + 0.5 (covers Mark buys, edge ≥1)
MM_VOLUME_THRESH  = 15     # bid/ask levels with size ≥ this are "real MM"; smaller likely Mark

# VFE passive MM — Mark 55 is a TAKER who hits resting bids/asks, not a poster.
# So take_book can't catch them; we need to BE the resting BBO inside the spread.
# Symmetric to HYDROGEL: post tightly inside the visible spread to capture flow.
VFE_MM_EDGE        = 1      # post bid/ask at robust_fair ± 1 (VFE spread is 5-7 wide)
VFE_MM_SIZE        = 20
VFE_MM_SKEW_K      = 8      # inventory skew at full position
VFE_MM_LIMIT       = 200

# VEV_4000 (deep ITM) — Mark 38 is a taker here too. Replace the basic_mm
# fair±9 with a tighter quote (fair±5) so we compete with Mark 14 for fills.
VEV4K_EDGE         = 5
VEV4K_SIZE         = 10
VEV4K_LIMIT        = 300

# Lottery options: VEV_6000 and VEV_6500 both trade exclusively at price 0
# across all 3 days of historical data. Posting a bid at 0 is risk-free —
# we pay nothing per contract and either earn the strike payout (if VFE
# spikes hard enough to push them ITM) or zero (if they expire OTM).
LOTTERY_STRIKES    = [6000, 6500]
LOTTERY_LIMIT      = 300

# Counterparty-aware adverse selection. With per-trade buyer/seller visibility,
# we can distinguish exploitable counterparties (whose fills we want to keep
# happening) from sharps (whose fills carry adverse information).
#
# Toxic counterparties — when they fill us on a side, that side's price is
# about to move against us. Widen that side temporarily.
#   Mark 49 (VFE):  smart MM around the 5240 anchor — when they sell to us,
#                   mid tends to revert down → widen our bid.
#   Mark 67 (VFE):  momentum buyer with entry alpha — when they buy from us,
#                   mid tends to keep rising → widen our ask.
#
# Everything else is "soft": Mark 38, 55, 14, 22, 01 are our profit source.
# Their fills should NOT widen our quotes — we want to keep being available.
TOXIC_BID_SELLERS = {     # counterparty sells to us (we buy) → widen bid
    "VELVETFRUIT_EXTRACT": {"Mark 49"},
}
TOXIC_ASK_BUYERS  = {     # counterparty buys from us (we sell) → widen ask
    "VELVETFRUIT_EXTRACT": {"Mark 67"},
}
TOXIC_WIDEN_TICKS = 3     # extra ticks added when toxic counterparty hits us

# Late-session mode: 90%+ of Mark 38/55 flow happens in last 10% of session.
# Tighten thresholds and lift limits during this window to maximise capture.
LATE_SESSION_TS   = 90_000   # within a single day
LATE_THRESH_MULT  = 0.6      # multiply ask/bid edge thresholds by this in late session

# ─────────────────────────────────────────────
# Math
# ─────────────────────────────────────────────
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / SQRT2))

def bs_call_delta(S, K, T, sigma):
    if T <= 0:
        return max(S - K, 0.0), (1.0 if S > K else 0.0)

    sig_t = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma**2 * T) / sig_t

    price = S * norm_cdf(d1) - K * norm_cdf(d1 - sig_t)
    delta = norm_cdf(d1)
    return price, delta

def implied_vol(S, K, T, price):
    if T <= 0:
        return None

    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 0.5:
        return None

    lo, hi = 1e-4, 5.0
    for _ in range(40):
        mid = (lo + hi) / 2
        p = bs_call_delta(S, K, T, mid)[0]
        if p < price:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ─────────────────────────────────────────────
# Mark-robust fair value
# Real MMs post deep size (20-30) at outer levels; Marks post smaller (3-7)
# inside that spread. Use mid of largest-volume bid and largest-volume ask.
# Falls back to BBO mid if no level is large.
# ─────────────────────────────────────────────
def robust_fair(od, vol_thresh=MM_VOLUME_THRESH):
    if not od.buy_orders or not od.sell_orders:
        return None

    big_bids = [p for p, v in od.buy_orders.items() if v >= vol_thresh]
    big_asks = [p for p, v in od.sell_orders.items() if -v >= vol_thresh]

    if big_bids and big_asks:
        # use highest "real MM" bid and lowest "real MM" ask
        return (max(big_bids) + min(big_asks)) / 2

    # fallback: largest-volume level on each side
    bid = max(od.buy_orders.items(), key=lambda kv: kv[1])[0]
    ask = min(od.sell_orders.items(), key=lambda kv: kv[1])[0]  # most negative volume
    return (bid + ask) / 2


# ─────────────────────────────────────────────
# Generic Mark taker — walks the book, takes everything mispriced vs fair.
# Returns (orders, new_pos).
# ─────────────────────────────────────────────
def take_book(sym, od, fair, ask_edge, bid_edge, pos, limit):
    res = []

    # Lift mispriced asks: any ask priced below fair - ask_edge
    # Walk from cheapest ask upward.
    for px in sorted(od.sell_orders):
        if px >= fair - ask_edge:
            break
        avail = -od.sell_orders[px]
        room  = limit - pos
        qty = min(avail, room)
        if qty > 0:
            res.append(Order(sym, int(px), qty))
            pos += qty
        if pos >= limit:
            break

    # Hit mispriced bids: any bid priced above fair + bid_edge
    # Walk from richest bid downward.
    for px in sorted(od.buy_orders, reverse=True):
        if px <= fair + bid_edge:
            break
        avail = od.buy_orders[px]
        room  = limit + pos
        qty = min(avail, room)
        if qty > 0:
            res.append(Order(sym, int(px), -qty))
            pos -= qty
        if pos <= -limit:
            break

    return res, pos


# ─────────────────────────────────────────────
# Trader
# ─────────────────────────────────────────────
class Trader:
    def run(self, state: TradingState):
        orders = {}

        try:
            td = json.loads(state.traderData) if state.traderData else {}
        except:
            td = {}

        # ── time
        day = td.get("day", 4)
        prev_ts = td.get("prev_ts", 0)

        if state.timestamp < prev_ts:
            day += 1

        td["day"] = day
        td["prev_ts"] = state.timestamp

        elapsed = day * 1_000_000 + state.timestamp
        T = max(T_START - T_DECAY * elapsed / T_TOTAL_TS, T_MIN)

        # ── underlying
        od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        if not od:
            return orders, 0, json.dumps(td)

        S = self.mid(od)
        if S is None:
            return orders, 0, json.dumps(td)

        # ─────────────────────────────────────────
        # VOL FIX
        # ─────────────────────────────────────────
        sigma = td.get("sigma", SIGMA_SEED)

        ivs = []
        for sym, K in [("VEV_5200", 5200), ("VEV_5300", 5300)]:
            if sym in state.order_depths:
                mid = self.mid(state.order_depths[sym])
                if mid:
                    iv = implied_vol(S, K, T, mid)
                    if iv:
                        ivs.append(iv)

        if ivs:
            iv_now = sorted(ivs)[len(ivs)//2]

            step = 0.01
            diff = iv_now - sigma
            if abs(diff) > step:
                iv_now = sigma + step * (1 if diff > 0 else -1)

            sigma = IV_EMA_ALPHA * iv_now + (1 - IV_EMA_ALPHA) * sigma
            sigma = max(SIGMA_LO, min(SIGMA_HI, sigma))

        td["sigma"] = sigma

        # ─────────────────────────────────────────
        # DELTA
        # ─────────────────────────────────────────
        total_delta = 0

        # ─────────────────────────────────────────
        # ITM OPTIONS
        # ─────────────────────────────────────────
        for sym, K in [("VEV_5000", 5000), ("VEV_5100", 5100), ("VEV_5200", 5200)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            orders[sym] = self.itm_mm(
                sym, od, fair, bb, ba,
                ITM_MM_EDGE, ITM_SNIPE,
                ITM_LIMITS[K],
                state.position.get(sym, 0)
            )

            total_delta += state.position.get(sym, 0) * delta

        # ─────────────────────────────────────────
        # Late-session detector: 92% of Mark 38 / 91% of Mark 55 trades
        # cluster in the last 10k ticks. Boost aggression there.
        # ─────────────────────────────────────────
        late_session = state.timestamp >= LATE_SESSION_TS
        thr_mult = LATE_THRESH_MULT if late_session else 1.0

        # ─────────────────────────────────────────
        # DEEP ITM
        # VEV_4000: Mark 38 also takes here. The original fair±9 quote was
        # too wide — Mark 14 at fair±5 was getting all the flow. Tighten to
        # fair±5 to compete. VEV_4500 stays at the wider edge (no Mark pattern).
        # ─────────────────────────────────────────
        for sym, K, edge in [("VEV_4000", 4000, VEV4K_EDGE),
                             ("VEV_4500", 4500, 9.0)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            sym_pos = state.position.get(sym, 0)
            size = VEV4K_SIZE if sym == "VEV_4000" else 10
            limit = VEV4K_LIMIT if sym == "VEV_4000" else 300
            orders[sym] = self.basic_mm(sym, od, fair, edge, limit, sym_pos)

            total_delta += sym_pos * delta

        # ─────────────────────────────────────────
        # LOTTERY OPTIONS — VEV_6000 and VEV_6500
        # Across 3 days of historical trade data, both products traded
        # exclusively at price 0 (1105 qty each over 3 days). Posting a
        # bid at 0 with full remaining capacity is risk-free: we pay
        # nothing per contract and either earn the strike payout if the
        # underlying spikes high enough to push them ITM, or zero. Free
        # lottery tickets, only constrained by position limit.
        # ─────────────────────────────────────────
        for K in LOTTERY_STRIKES:
            sym = f"VEV_{K}"
            od = state.order_depths.get(sym)
            if od is None:
                continue
            pos = state.position.get(sym, 0)
            room = LOTTERY_LIMIT - pos
            if room > 0:
                orders[sym] = [Order(sym, 0, room)]
            # NB: not adding to total_delta — at price 0 these have effectively
            # zero delta in our pricing model; if they ever start trading at
            # non-zero this needs revisiting.

        # ─────────────────────────────────────────
        # OTM
        # ─────────────────────────────────────────
        for sym, K in [("VEV_5300", 5300), ("VEV_5400", 5400), ("VEV_5500", 5500)]:
            od = state.order_depths.get(sym)
            if not od:
                continue

            bb, ba = self.bbba(od)
            if ba - bb < 2:
                continue

            fair, delta = bs_call_delta(S, K, T, sigma)

            skew = (S - K) / K
            b_off = -2 + 6 * skew
            a_off =  2 + 6 * skew

            orders[sym] = self.biased_mm(
                sym, od, fair, bb, ba,
                b_off, a_off, 300,
                state.position.get(sym, 0),
                delta_sat=False
            )

            total_delta += state.position.get(sym, 0) * delta

        # ─────────────────────────────────────────
        # HYDROGEL — Mark 38 take + own-book MM
        # ─────────────────────────────────────────
        hydro_od = state.order_depths.get("HYDROGEL_PACK")
        if hydro_od:
            bid_widen, ask_widen = adv_widen("HYDROGEL_PACK")
            orders["HYDROGEL_PACK"] = self.exec_hydrogel(
                hydro_od,
                state.position.get("HYDROGEL_PACK", 0),
                thr_mult,
                bid_widen, ask_widen,
            )

        # ─────────────────────────────────────────
        # Counterparty-aware adverse selection.
        # state.own_trades holds the previous tick's fills with named buyer/seller.
        # We only widen quotes after fills from known-toxic counterparties.
        # ─────────────────────────────────────────
        own_trades = getattr(state, "own_trades", {}) or {}

        def adv_widen(sym):
            bid_w = ask_w = 0
            toxic_sellers = TOXIC_BID_SELLERS.get(sym, set())
            toxic_buyers  = TOXIC_ASK_BUYERS.get(sym, set())
            for t in own_trades.get(sym, []) or []:
                # buyer/seller may be set on either field depending on direction
                if getattr(t, "buyer", None) == "SUBMISSION":
                    if getattr(t, "seller", None) in toxic_sellers:
                        bid_w = TOXIC_WIDEN_TICKS
                else:
                    if getattr(t, "buyer", None) in toxic_buyers:
                        ask_w = TOXIC_WIDEN_TICKS
            return bid_w, ask_w

        # ─────────────────────────────────────────
        # VFE passive MM — Mark 55 is a TAKER, so we need to be the resting BBO.
        # This replaces the old take_book layer (which couldn't see Mark's flow
        # because Mark posts and gets filled within the same tick).
        # ─────────────────────────────────────────
        vfe_od = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vfe_orders = []
        vfe_pos = state.position.get("VELVETFRUIT_EXTRACT", 0)
        if vfe_od:
            vfe_fair = robust_fair(vfe_od)
            if vfe_fair is not None:
                bb, ba = self.bbba(vfe_od)
                if ba - bb >= 2:
                    bid_widen, ask_widen = adv_widen("VELVETFRUIT_EXTRACT")
                    skew = (vfe_pos / VFE_MM_LIMIT) * VFE_MM_SKEW_K
                    bid_px = int(vfe_fair - VFE_MM_EDGE - skew - bid_widen)
                    ask_px = int(vfe_fair + VFE_MM_EDGE - skew + ask_widen)

                    # don't cross the visible market
                    if bid_px >= ba: bid_px = ba - 1
                    if ask_px <= bb: ask_px = bb + 1

                    if bid_px < ask_px:
                        bid_size = max(0, min(VFE_MM_SIZE, VFE_MM_LIMIT - vfe_pos))
                        ask_size = max(0, min(VFE_MM_SIZE, VFE_MM_LIMIT + vfe_pos))
                        if bid_size > 0:
                            vfe_orders.append(Order("VELVETFRUIT_EXTRACT", bid_px, bid_size))
                        if ask_size > 0:
                            vfe_orders.append(Order("VELVETFRUIT_EXTRACT", ask_px, -ask_size))

        # ─────────────────────────────────────────
        # DELTA HEDGE (logic unchanged from original)
        # gap uses POST-take position so the hedge fills only the residual
        # after the take layer has captured what it can.
        # ─────────────────────────────────────────
        gap = -total_delta - vfe_pos

        if abs(gap) > 5:
            qty = int(max(-200, min(200, gap)))
            bb, ba = self.bbba(vfe_od)
            side_px = bb if qty > 0 else ba
            vfe_orders.append(Order("VELVETFRUIT_EXTRACT", side_px, qty))

        if vfe_orders:
            orders["VELVETFRUIT_EXTRACT"] = vfe_orders

        return orders, 0, json.dumps(td)

    # ─────────────────────────────────────────────
    # HYDROGEL — Mark 38 take + own-book MM
    # ─────────────────────────────────────────────
    def exec_hydrogel(self, od, pos, thr_mult=1.0, bid_widen=0, ask_widen=0):
        res = []

        if not od.buy_orders or not od.sell_orders:
            return res

        # Robust fair excludes Mark's small flicker quotes
        fair = robust_fair(od)
        if fair is None:
            return res

        # ── 1. Mark 38 taker: aggressive — take everything mispriced vs robust fair
        take_orders, pos = take_book(
            "HYDROGEL_PACK", od, fair,
            HYD_TAKE_ASK_EDGE * thr_mult, HYD_TAKE_BID_EDGE * thr_mult,
            pos, HYD_LIMIT,
        )
        res.extend(take_orders)

        # ── 2. Stale-level sweep (kept from v3) — fires only on really wide
        # dislocations beyond what Mark normally produces. Likely redundant
        # with the taker above, but cheap and harmless.
        # (Skipped: subsumed by the taker which uses tighter thresholds.)

        # ── 3. Passive MM around BBO (post-take book state)
        bb = max(od.buy_orders)
        ba = min(od.sell_orders)
        if ba - bb < 2:
            return res

        # Use BBO mid for passive quoting since we're already inside the spread.
        # Taker prevents us from being adversely selected by Mark.
        bv = od.buy_orders[bb]
        av = -od.sell_orders[ba]
        if bv + av > 0:
            micro = (bb * av + ba * bv) / (bv + av)
        else:
            micro = (bb + ba) / 2

        skew = (pos / HYD_LIMIT) * HYD_SKEW_K
        bid_px = int(micro - HYD_EDGE - skew - bid_widen)
        ask_px = int(micro + HYD_EDGE - skew + ask_widen)

        if bid_px >= ba:
            bid_px = ba - 1
        if ask_px <= bb:
            ask_px = bb + 1

        if bid_px < ask_px:
            bid_size = max(0, min(HYD_SIZE, HYD_LIMIT - pos))
            ask_size = max(0, min(HYD_SIZE, HYD_LIMIT + pos))
            if bid_size > 0:
                res.append(Order("HYDROGEL_PACK", bid_px, bid_size))
            if ask_size > 0:
                res.append(Order("HYDROGEL_PACK", ask_px, -ask_size))

        return res

    # ─────────────────────────────────────────────
    # helpers
    # ─────────────────────────────────────────────
    def mid(self, od):
        if not od.buy_orders or not od.sell_orders:
            return None
        return (max(od.buy_orders) + min(od.sell_orders)) / 2

    def bbba(self, od):
        return max(od.buy_orders), min(od.sell_orders)

    def itm_mm(self, sym, od, fair, bb, ba, edge, snipe, limit, pos):
        res = []

        if bb > fair + snipe:
            res.append(Order(sym, bb, -min(od.buy_orders[bb], limit + pos)))

        if ba < fair - snipe:
            res.append(Order(sym, ba, min(-od.sell_orders[ba], limit - pos)))

        bid = int(fair - edge)
        ask = int(fair + edge)

        if ask > bid:
            res.append(Order(sym, bid, 5))
            res.append(Order(sym, ask, -5))

        return res

    def basic_mm(self, sym, od, fair, edge, limit, pos):
        bid = int(fair - edge)
        ask = int(fair + edge)

        return [
            Order(sym, bid, min(10, limit - pos)),
            Order(sym, ask, -min(10, limit + pos))
        ]

    def biased_mm(self, sym, od, fair, bb, ba, b_off, a_off, limit, pos, delta_sat):
        bid = int(fair + b_off)
        ask = int(fair + a_off)

        return [
            Order(sym, bid, min(10, limit - pos)),
            Order(sym, ask, -min(10, limit + pos))
        ]