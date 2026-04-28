"""
Vectorized backtest with a stricter passive fill model.

Passive fill rule (more conservative than simulate_fills v1):
  - A passive bid at price p_b fills against trades printing at price <= p_b
    that crossed (i.e., their print would imply someone aggressed our quote).
    To approximate that, we say: a market trade at price t with t == p_b counts;
    a market trade at price t with t < p_b is ambiguous and we ignore it
    (the trade may have crossed multiple levels including ours, but we don't
     know our queue position, so be conservative).
  - Similarly passive ask at p_a fills against trades at t == p_a only.

  - Aggressive orders cross the displayed L1 book and fill up to displayed volume.

  - Position limit: we clip orders to remaining headroom before applying fills.
"""
import time
import pandas as pd
import numpy as np
from collections import defaultdict
from trader import Trader, TradingState, OrderDepth, Order, ALL_PRODUCTS, POSITION_LIMIT


def build_order_depth_from_row(r):
    od = OrderDepth()
    for lvl in (1,2,3):
        bp, bv = r.get(f"bid_price_{lvl}"), r.get(f"bid_volume_{lvl}")
        ap, av = r.get(f"ask_price_{lvl}"), r.get(f"ask_volume_{lvl}")
        if pd.notna(bp) and pd.notna(bv): od.buy_orders[int(bp)] = int(bv)
        if pd.notna(ap) and pd.notna(av): od.sell_orders[int(ap)] = int(av)
    return od


def simulate_fills_v2(orders, depth, trades_at_t, mid):
    """
    Fill model:
      Aggressive: orders crossing displayed book consume L1+L2+L3 ask/bid up to vol.
      Passive: classify each market trade as buy- or sell-aggressor by comparison to
        contemporaneous mid:
          trade.price > mid  => buy aggressor (took ask) — fills our passive ASK at p_a if p_a <= trade.price
          trade.price < mid  => sell aggressor (hit bid) — fills our passive BID at p_b if p_b >= trade.price
        Our passive quote has price priority over any same-side displayed book at worse prices,
        so we fill at OUR quote price (the trade.price was where the existing book printed).
      Capped by our quote size and position headroom.
    """
    fills = []
    if not orders: return fills

    asks = dict(depth.sell_orders)
    bids = dict(depth.buy_orders)
    bb = max(bids) if bids else None
    ba = min(asks) if asks else None

    # Classify trades into buy- and sell-aggressor pools
    # buy_pool: trades likely to hit a passive ask (we sell into them)
    # sell_pool: trades likely to hit a passive bid (we buy from them)
    buy_pool = []   # list of (trade_price, qty)
    sell_pool = []
    if trades_at_t is not None and len(trades_at_t) > 0:
        for _, row in trades_at_t.iterrows():
            tp = int(row["price"]); tq = int(row["quantity"])
            if mid is None: continue
            if tp > mid: buy_pool.append([tp, tq])
            elif tp < mid: sell_pool.append([tp, tq])
            else:
                # ambiguous: split
                half = tq // 2
                buy_pool.append([tp, half])
                sell_pool.append([tp, tq - half])

    for o in orders:
        if o.quantity > 0:  # we want to BUY
            remaining = o.quantity
            # Aggressive: cross asks priced <= o.price
            for p in sorted(p for p in asks if p <= o.price):
                avail = asks[p]
                if avail <= 0: continue
                take = min(remaining, avail)
                if take > 0:
                    fills.append((p, +take))
                    asks[p] -= take
                    remaining -= take
                if remaining == 0: break
            # Passive (only if quote sits strictly inside the book)
            if remaining > 0 and bb is not None and ba is not None and bb < o.price < ba:
                # We take from sell-aggressor pool: any trade priced <= our bid would have hit us
                for entry in sell_pool:
                    if remaining == 0: break
                    if entry[1] <= 0: continue
                    if entry[0] <= o.price:
                        take = min(remaining, entry[1])
                        if take > 0:
                            fills.append((o.price, +take))
                            entry[1] -= take
                            remaining -= take
        else:  # we want to SELL, o.quantity < 0
            remaining = -o.quantity
            for p in sorted((p for p in bids if p >= o.price), reverse=True):
                avail = bids[p]
                if avail <= 0: continue
                take = min(remaining, avail)
                if take > 0:
                    fills.append((p, -take))
                    bids[p] -= take
                    remaining -= take
                if remaining == 0: break
            if remaining > 0 and bb is not None and ba is not None and bb < o.price < ba:
                for entry in buy_pool:
                    if remaining == 0: break
                    if entry[1] <= 0: continue
                    if entry[0] >= o.price:
                        take = min(remaining, entry[1])
                        if take > 0:
                            fills.append((o.price, -take))
                            entry[1] -= take
                            remaining -= take
    return fills


def run_one_day(day):
    prices = pd.read_csv(f"/mnt/user-data/uploads/prices_round_5_day_{day}.csv", sep=";")
    trades = pd.read_csv(f"/mnt/user-data/uploads/trades_round_5_day_{day}.csv", sep=";")

    # Index trades by (timestamp, symbol)
    trades_idx = trades.set_index(["timestamp","symbol"]).sort_index()
    has_key = set(zip(trades["timestamp"], trades["symbol"]))

    # Pre-group prices by timestamp; use itertuples for speed
    prices_grouped = {ts: g for ts, g in prices.groupby("timestamp", sort=True)}
    timestamps = sorted(prices_grouped.keys())

    trader = Trader()
    traderData = ""
    position = defaultdict(int)
    cash_per_sym = defaultdict(float)
    last_mid = {}
    fills_per_sym = defaultdict(int)
    vol_per_sym = defaultdict(int)

    t0 = time.time()
    for i, ts in enumerate(timestamps):
        rows = prices_grouped[ts]
        order_depths = {}
        for r in rows.itertuples(index=False):
            sym = r.product
            od = OrderDepth()
            if pd.notna(r.bid_price_1) and pd.notna(r.bid_volume_1):
                od.buy_orders[int(r.bid_price_1)] = int(r.bid_volume_1)
            if pd.notna(r.bid_price_2) and pd.notna(r.bid_volume_2):
                od.buy_orders[int(r.bid_price_2)] = int(r.bid_volume_2)
            if pd.notna(r.bid_price_3) and pd.notna(r.bid_volume_3):
                od.buy_orders[int(r.bid_price_3)] = int(r.bid_volume_3)
            if pd.notna(r.ask_price_1) and pd.notna(r.ask_volume_1):
                od.sell_orders[int(r.ask_price_1)] = int(r.ask_volume_1)
            if pd.notna(r.ask_price_2) and pd.notna(r.ask_volume_2):
                od.sell_orders[int(r.ask_price_2)] = int(r.ask_volume_2)
            if pd.notna(r.ask_price_3) and pd.notna(r.ask_volume_3):
                od.sell_orders[int(r.ask_price_3)] = int(r.ask_volume_3)
            order_depths[sym] = od
            if pd.notna(r.mid_price):
                last_mid[sym] = r.mid_price

        state = TradingState(timestamp=ts, listings={s:None for s in ALL_PRODUCTS},
                             order_depths=order_depths,
                             own_trades={}, market_trades={},
                             position=dict(position), observations=None,
                             traderData=traderData)
        orders_dict, _, traderData = trader.run(state)

        for sym, orders in orders_dict.items():
            depth = order_depths.get(sym)
            if depth is None: continue
            # Pre-clip by remaining position
            buy_room = POSITION_LIMIT - position[sym]
            sell_room = POSITION_LIMIT + position[sym]
            clipped = []
            for o in orders:
                if o.quantity > 0:
                    q = min(o.quantity, buy_room)
                    if q > 0:
                        clipped.append(Order(o.symbol, o.price, q))
                        buy_room -= q
                else:
                    q = max(o.quantity, -sell_room)
                    if q < 0:
                        clipped.append(Order(o.symbol, o.price, q))
                        sell_room += q

            tr = trades_idx.loc[(ts, sym)] if (ts, sym) in has_key else None
            if tr is not None and isinstance(tr, pd.Series):
                tr = tr.to_frame().T
            mid_for_classify = last_mid.get(sym)
            fills = simulate_fills_v2(clipped, depth, tr, mid_for_classify)
            for p, q in fills:
                cash_per_sym[sym] -= p*q
                position[sym] += q
                fills_per_sym[sym] += 1
                vol_per_sym[sym] += abs(q)

        if i and i % 2000 == 0:
            print(f"  day{day} {i}/{len(timestamps)}  ({time.time()-t0:.0f}s)")

    # End-of-day MTM
    pnl_per_sym = {}
    total = 0.0
    for sym in ALL_PRODUCTS:
        mtm = position[sym] * last_mid.get(sym, 0)
        pnl = cash_per_sym[sym] + mtm
        pnl_per_sym[sym] = pnl
        total += pnl

    return {
        "day": day,
        "total": total,
        "pnl_per_sym": pnl_per_sym,
        "position": dict(position),
        "fills": dict(fills_per_sym),
        "vol": dict(vol_per_sym),
        "last_mid": dict(last_mid),
        "elapsed": time.time()-t0,
    }


def run_all_days(days=(2,3,4)):
    results = []
    for d in days:
        print(f"\n=== Running day {d} ===")
        r = run_one_day(d)
        print(f"  Day {d} PnL: {r['total']:,.0f}  ({r['elapsed']:.0f}s)")
        nz = [(s,p) for s,p in r['position'].items() if p!=0]
        print(f"  Non-zero positions: {len(nz)}")
        results.append(r)
    print("\n=== TOTAL ===")
    total = sum(r['total'] for r in results)
    for r in results:
        print(f"  Day {r['day']}: {r['total']:,.0f}")
    print(f"  TOTAL: {total:,.0f}")
    # Aggregate per-product PnL
    agg = defaultdict(float)
    agg_fills = defaultdict(int)
    agg_vol = defaultdict(int)
    for r in results:
        for s,p in r['pnl_per_sym'].items(): agg[s] += p
        for s,n in r['fills'].items(): agg_fills[s] += n
        for s,v in r['vol'].items(): agg_vol[s] += v
    print("\nTop 15 winning products:")
    for s,p in sorted(agg.items(), key=lambda x:-x[1])[:15]:
        print(f"  {s:<35} pnl={p:+,.0f}  fills={agg_fills[s]}  vol={agg_vol[s]}")
    print("\nTop 15 losing products:")
    for s,p in sorted(agg.items(), key=lambda x: x[1])[:15]:
        print(f"  {s:<35} pnl={p:+,.0f}  fills={agg_fills[s]}  vol={agg_vol[s]}")
    return total, agg, agg_fills


if __name__ == "__main__":
    run_all_days()