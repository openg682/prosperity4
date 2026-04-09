"""
Prosperity Backtester v2
========================
Supports both aggressive and passive fills.

Passive fill logic (the key addition):
  - Each tick, we check actual trades that occurred.
  - If a trade happened at price P for a product:
      * P <= our resting BUY price  → a seller accepted ≤ our price → we fill
      * P >= our resting SELL price → a buyer paid ≥ our price → we fill
  - Since we quote inside the spread, we have price priority over book orders.

Usage:
    python backtester.py

To test your own algorithm, subclass TradingAlgorithm and implement get_orders().
"""
#to run this file, use `python3 backtester.py wrapper.py:totest.py` from the root directory of the repo


import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class Order:
    product: str
    side: str        # 'BUY' or 'SELL'
    price: int
    quantity: int


@dataclass
class OrderBook:
    product: str
    bids: List[Tuple[int, int]]
    asks: List[Tuple[int, int]]
    mid_price: float
    timestamp: int
    day: int

    @property
    def best_bid(self):
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self):
        return self.asks[0][0] if self.asks else None

    @property
    def spread(self):
        if self.best_bid and self.best_ask:
            return self.best_ask - self.best_bid
        return None


# ── Base algorithm ────────────────────────────────────────────────────────────

class TradingAlgorithm:
    def __init__(self):
        self.state = {}

    def get_orders(self, order_books, positions):
        raise NotImplementedError


# ── Backtester ────────────────────────────────────────────────────────────────

class Backtester:
    POSITION_LIMITS = {'EMERALDS': 80, 'TOMATOES': 80}
    DEFAULT_LIMIT   = 80

    def __init__(self, price_files, trade_files, algorithm):
        self.algorithm = algorithm

        price_frames = [pd.read_csv(f, sep=';') for f in price_files]
        self.prices = pd.concat(price_frames, ignore_index=True)

        days = sorted(self.prices['day'].unique())
        trade_frames = []
        for day, f in zip(days, trade_files):
            df = pd.read_csv(f, sep=';')
            df['day'] = day
            trade_frames.append(df)
        self.trades = pd.concat(trade_frames, ignore_index=True)

        self._trade_idx = {}
        for (day, ts, sym), grp in self.trades.groupby(['day','timestamp','symbol']):
            self._trade_idx[(day, ts, sym)] = grp

        self.positions = defaultdict(int)
        self.cash      = defaultdict(float)
        self.pnl_history      = []
        self.position_history = defaultdict(list)

    def _limit(self, product):
        return self.POSITION_LIMITS.get(product, self.DEFAULT_LIMIT)

    def _parse_book(self, row, product):
        bids, asks = [], []
        for i in range(1, 4):
            bp = row.get(f'bid_price_{i}')
            bv = row.get(f'bid_volume_{i}')
            ap = row.get(f'ask_price_{i}')
            av = row.get(f'ask_volume_{i}')
            if pd.notna(bp) and pd.notna(bv) and bv > 0:
                bids.append((int(bp), int(bv)))
            if pd.notna(ap) and pd.notna(av) and av > 0:
                asks.append((int(ap), int(av)))
        return OrderBook(
            product=product,
            bids=sorted(bids, key=lambda x: -x[0]),
            asks=sorted(asks, key=lambda x:  x[0]),
            mid_price=float(row['mid_price']),
            timestamp=int(row['timestamp']),
            day=int(row['day']),
        )

    def _aggressive_fill(self, order, book):
        pos = self.positions[order.product]
        lim = self._limit(order.product)
        filled, cost = 0, 0.0
        if order.side == 'BUY':
            want = min(order.quantity, max(lim - pos, 0))
            for px, vol in book.asks:
                if want <= 0 or px > order.price: break
                qty = min(want, vol); filled += qty; want -= qty; cost += qty * px
        else:
            want = min(order.quantity, max(lim + pos, 0))
            for px, vol in book.bids:
                if want <= 0 or px < order.price: break
                qty = min(want, vol); filled += qty; want -= qty; cost -= qty * px
        return filled, cost

    def _passive_fill(self, order, day, ts, remaining):
        pos  = self.positions[order.product]
        lim  = self._limit(order.product)
        key  = (day, ts, order.product)
        tick_trades = self._trade_idx.get(key)
        if tick_trades is None or remaining <= 0:
            return 0, 0.0
        filled, cost = 0, 0.0
        if order.side == 'BUY':
            want = min(remaining, max(lim - pos, 0))
            for _, t in tick_trades.iterrows():
                if want <= 0: break
                if t['price'] <= order.price:
                    qty = min(want, int(t['quantity'])); filled += qty; want -= qty; cost += qty * t['price']
        else:
            want = min(remaining, max(lim + pos, 0))
            for _, t in tick_trades.iterrows():
                if want <= 0: break
                if t['price'] >= order.price:
                    qty = min(want, int(t['quantity'])); filled += qty; want -= qty; cost -= qty * t['price']
        return filled, cost

    def _execute(self, order, book, day, ts):
        agg_qty, agg_cost = self._aggressive_fill(order, book)
        sign = 1 if order.side == 'BUY' else -1
        self.positions[order.product] += sign * agg_qty
        self.cash[order.product]      -= agg_cost

        pas_qty, pas_cost = self._passive_fill(order, day, ts, order.quantity - agg_qty)
        self.positions[order.product] += sign * pas_qty
        self.cash[order.product]      -= pas_cost

    def _mark_to_mid(self, order_books):
        total = sum(self.cash.values())
        for product, pos in self.positions.items():
            if product in order_books:
                total += pos * order_books[product].mid_price
        return total

    def _pnl_per_product(self, order_books):
        result = {}
        for product, pos in self.positions.items():
            mtm = self.cash[product]
            if product in order_books:
                mtm += pos * order_books[product].mid_price
            result[product] = mtm
        return result

    def run(self):
        all_products = self.prices['product'].unique().tolist()
        grouped = self.prices.groupby(['day', 'timestamp'], sort=True)

        for (day, ts), group in grouped:
            order_books = {row['product']: self._parse_book(row, row['product'])
                           for _, row in group.iterrows()}
            orders = self.algorithm.get_orders(order_books, dict(self.positions))
            for order in orders:
                if order.product in order_books:
                    self._execute(order, order_books[order.product], day, ts)
            self.pnl_history.append(self._mark_to_mid(order_books))
            for p in all_products:
                self.position_history[p].append(self.positions[p])

        return self._report(all_products)

    def _report(self, products):
        report = {
            'final_pnl': self.pnl_history[-1] if self.pnl_history else 0,
            'max_pnl':   max(self.pnl_history, default=0),
            'min_pnl':   min(self.pnl_history, default=0),
            'n_ticks':   len(self.pnl_history),
            'position_utilisation': {},
        }
        for p in products:
            hist = np.array(self.position_history[p])
            lim  = self._limit(p)
            report['position_utilisation'][p] = {
                'mean_abs_pos':    float(np.mean(np.abs(hist))),
                'max_abs_pos':     int(np.max(np.abs(hist))),
                'utilisation_pct': float(np.mean(np.abs(hist)) / lim * 100),
            }
        self._plot(products)
        self._save_output(products)
        return report

    def _save_output(self, products):
        rows = []
        for i in range(len(self.pnl_history)):
            row = {'tick': i, 'pnl': self.pnl_history[i]}
            for p in products:
                row[f'position_{p}'] = self.position_history[p][i]
            rows.append(row)
        df = pd.DataFrame(rows)
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_output.csv')
        df.to_csv(out, index=False)
        print(f"[output → {out}]")

    def _plot(self, products):
        n = 1 + len(products)
        fig, axes = plt.subplots(n, 1, figsize=(16, 4*n), sharex=True)
        if n == 1: axes = [axes]
        ticks = np.arange(len(self.pnl_history))

        ax = axes[0]
        ax.plot(ticks, self.pnl_history, color='#2ecc71', linewidth=1.0)
        ax.axhline(0, color='black', linewidth=0.8, linestyle='--')
        ax.fill_between(ticks, self.pnl_history, 0,
                         where=[p >= 0 for p in self.pnl_history], alpha=0.15, color='#2ecc71')
        ax.fill_between(ticks, self.pnl_history, 0,
                         where=[p < 0 for p in self.pnl_history], alpha=0.15, color='#e74c3c')
        ax.set_title('PnL Over Time (mark-to-mid)', fontsize=13, fontweight='bold')
        ax.set_ylabel('PnL'); ax.grid(True, alpha=0.3)

        colors = ['#3498db', '#e67e22']
        for i, p in enumerate(products):
            ax  = axes[i+1]; lim = self._limit(p)
            color = colors[i % len(colors)]; hist = self.position_history[p]
            ax.plot(ticks, hist, color=color, linewidth=0.9)
            ax.axhline( lim, color='#e74c3c', linestyle='--', linewidth=1, label=f'+{lim} limit')
            ax.axhline(-lim, color='#e74c3c', linestyle='--', linewidth=1, label=f'-{lim} limit')
            ax.axhline(0, color='black', linewidth=0.5)
            ax.fill_between(ticks, hist, 0, alpha=0.12, color=color)
            ax.set_title(f'{p} Position', fontsize=12, fontweight='bold')
            ax.set_ylabel('Units'); ax.set_ylim(-lim*1.15, lim*1.15)
            ax.legend(loc='upper right', fontsize=8); ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Tick')
        plt.tight_layout()
        out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backtest_results.png')
        plt.savefig(out, dpi=150, bbox_inches='tight'); plt.close()
        print(f"[plot → {out}]")


# ── Algorithms ────────────────────────────────────────────────────────────────

class EmeraldsMarketMaker(TradingAlgorithm):
    FAIR_VALUE = 10_000
    EDGE = 6
    def get_orders(self, order_books, positions):
        book = order_books.get('EMERALDS')
        if not book: return []
        pos = positions.get('EMERALDS', 0); lim = 80
        orders = []
        if lim - pos > 0: orders.append(Order('EMERALDS','BUY', self.FAIR_VALUE-self.EDGE, lim-pos))
        if lim + pos > 0: orders.append(Order('EMERALDS','SELL',self.FAIR_VALUE+self.EDGE, lim+pos))
        return orders


class TomatoesEMAMarketMaker(TradingAlgorithm):
    ALPHA = 0.15; EDGE = 8
    def __init__(self):
        super().__init__(); self.ema = None
    def get_orders(self, order_books, positions):
        book = order_books.get('TOMATOES')
        if not book: return []
        mid = book.mid_price
        self.ema = mid if self.ema is None else self.ALPHA*mid + (1-self.ALPHA)*self.ema
        fv = round(self.ema); pos = positions.get('TOMATOES',0); lim = 80
        orders = []
        if lim - pos > 0: orders.append(Order('TOMATOES','BUY', fv-self.EDGE, lim-pos))
        if lim + pos > 0: orders.append(Order('TOMATOES','SELL',fv+self.EDGE, lim+pos))
        return orders


class CombinedAlgorithm(TradingAlgorithm):
    def __init__(self):
        super().__init__()
        self._em = EmeraldsMarketMaker()
        self._tm = TomatoesEMAMarketMaker()
    def get_orders(self, order_books, positions):
        return self._em.get_orders(order_books, positions) + self._tm.get_orders(order_books, positions)


# ── Entry point ───────────────────────────────────────────────────────────────

def _install_jsonpickle_shim():
    """Provide a stdlib-only jsonpickle shim so strategies don't need the package."""
    import sys, types, json
    if 'jsonpickle' not in sys.modules:
        shim = types.ModuleType('jsonpickle')
        shim.encode = json.dumps
        shim.decode = json.loads
        sys.modules['jsonpickle'] = shim


def _load_module(path: str, name: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_algorithm(name: str) -> TradingAlgorithm:
    BUILTIN = {
        'combined': CombinedAlgorithm,
        'emeralds': EmeraldsMarketMaker,
        'tomatoes': TomatoesEMAMarketMaker,
    }
    if name in BUILTIN:
        return BUILTIN[name]()

    _install_jsonpickle_shim()
    if ':' in name:
        wrapper_path, strategy_path = name.split(':', 1)
        wrapper_mod  = _load_module(wrapper_path, 'wrapper')
        strategy_mod = _load_module(strategy_path, 'custom_strategy')
        algo_class   = wrapper_mod.wrap(strategy_mod.Trader)
        return algo_class()

    mod = _load_module(name, 'custom_strategy')
    return mod.ALGORITHM()


if __name__ == '__main__':
    PRICE_FILES = [
        '../../Files/Prices/prices_round_0_day_-2.csv',
        '../../Files/Prices/prices_round_0_day_-1.csv',
    ]
    TRADE_FILES = [
        '../../Files/Trades/trades_round_0_day_-2.csv',
        '../../Files/Trades/trades_round_0_day_-1.csv',
    ]

    strategy_arg = sys.argv[1] if len(sys.argv) > 1 else 'combined'
    print(f"Loading strategy: {strategy_arg}")

    algo   = load_algorithm(strategy_arg)
    bt     = Backtester(PRICE_FILES, TRADE_FILES, algo)
    report = bt.run()

    print("\n======= BACKTEST RESULTS =======")
    print(f"Ticks simulated : {report['n_ticks']:,}")
    print(f"Final PnL       : {report['final_pnl']:>12,.2f}")
    print(f"Peak  PnL       : {report['max_pnl']:>12,.2f}")
    print(f"Trough PnL      : {report['min_pnl']:>12,.2f}")
    print("\n--- Position Utilisation ---")
    for product, s in report['position_utilisation'].items():
        print(f"  {product}:")
        print(f"    Mean |pos| : {s['mean_abs_pos']:>6.1f}  /  80")
        print(f"    Max  |pos| : {s['max_abs_pos']:>6}  /  80")
        print(f"    Utilisation: {s['utilisation_pct']:>5.1f}%")