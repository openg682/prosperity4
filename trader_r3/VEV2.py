import math
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict

class Trader:
    
    def __init__(self):
        # We target the liquid 'Belly' of the options chain
        self.target_options = [
            'VEV_5000', 'VEV_5100', 'VEV_5200', 
            'VEV_5300', 'VEV_5400', 'VEV_5500'
        ]
        self.underlying = 'VELVETFRUIT_EXTRACT'
        
        # Base volatility assumption derived from our historical data
        self.base_vol = 0.25 
        
        # Position limits
        self.LIMIT_OPT = 300
        self.LIMIT_EXTRACT = 200

    def norm_cdf(self, x: float) -> float:
        """Approximates the Normal CDF without needing scipy."""
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    def bs_price_delta(self, S: float, K: float, T: float, r: float, sigma: float):
        """Calculates Black-Scholes Fair Value and Delta."""
        if T <= 0 or sigma <= 0:
            return max(0.0, S - K), (1.0 if S > K else 0.0)
            
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        fv = S * self.norm_cdf(d1) - K * math.exp(-r * T) * self.norm_cdf(d2)
        delta = self.norm_cdf(d1)
        return fv, delta

    def get_mid_price(self, order_depth: OrderDepth) -> float:
        """Safely calculates the mid price from an order book."""
        best_bid = max(order_depth.buy_orders.keys()) if len(order_depth.buy_orders) > 0 else None
        best_ask = min(order_depth.sell_orders.keys()) if len(order_depth.sell_orders) > 0 else None
        
        if best_bid and best_ask:
            return (best_bid + best_ask) / 2.0
        elif best_bid:
            return best_bid
        elif best_ask:
            return best_ask
        return 0.0

    def run(self, state: TradingState) -> Dict[str, List[Order]]:
        result = {}
        
        # 1. Get the underlying asset price
        if self.underlying not in state.order_depths:
            return result
            
        S = self.get_mid_price(state.order_depths[self.underlying])
        if S == 0.0:
            return result

        # Intraday Time to Expiry (TTE) decay
        # Starts at 5 days, decays fractionally over the 1,000,000 tick round
        days_remaining = 5.0 - (state.timestamp / 1000000.0)
        T = days_remaining / 365.0
        r = 0.0 

        # 2. Track total Portfolio Delta for Hedging
        total_delta = 0.0
        
        # Add existing underlying position to total delta (1 Extract = 1 Delta)
        current_extract_pos = state.position.get(self.underlying, 0)
        total_delta += current_extract_pos

        # 3. Market Make the Options
        for opt in self.target_options:
            if opt not in state.order_depths:
                continue
                
            K = float(opt.split('_')[1])
            fv, delta = self.bs_price_delta(S, K, T, r, self.base_vol)
            
            # Add this option's exposure to our portfolio delta
            current_pos = state.position.get(opt, 0)
            total_delta += current_pos * delta
            
            orders: List[Order] = []
            order_depth = state.order_depths[opt]
            
            # Quote slightly wider to ensure profitability on execution
            my_bid = math.floor(fv - 1.5)
            my_ask = math.ceil(fv + 1.5)
            
            # Execute Buy Orders (if market is cheaper than our Fair Value)
            if len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                if best_ask <= my_bid:
                    # We buy aggressively, especially for the 5400 kink
                    trade_vol = self.LIMIT_OPT - current_pos
                    if trade_vol > 0:
                        orders.append(Order(opt, best_ask, trade_vol))
                        total_delta += trade_vol * delta # Update expected delta
                        
            # Execute Sell Orders (if market is paying more than our Fair Value)
            if len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                if best_bid >= my_ask:
                    trade_vol = -self.LIMIT_OPT - current_pos
                    if trade_vol < 0:
                        orders.append(Order(opt, best_bid, trade_vol))
                        total_delta += trade_vol * delta # Update expected delta
                        
            result[opt] = orders

        # 4. Execute the Delta Hedge
        # If total_delta is positive, we need to sell Extract. If negative, buy Extract.
        hedge_orders: List[Order] = []
        extract_depth = state.order_depths[self.underlying]
        
        # We only hedge if our exposure drifts beyond a safe threshold to save on transaction costs
        if abs(total_delta) > 5.0:
            hedge_qty = -int(round(total_delta))
            
            # Respect the 200 limit
            expected_pos = current_extract_pos + hedge_qty
            if expected_pos > self.LIMIT_EXTRACT:
                hedge_qty = self.LIMIT_EXTRACT - current_extract_pos
            elif expected_pos < -self.LIMIT_EXTRACT:
                hedge_qty = -self.LIMIT_EXTRACT - current_extract_pos
                
            if hedge_qty > 0 and len(extract_depth.sell_orders) > 0:
                best_ask = min(extract_depth.sell_orders.keys())
                hedge_orders.append(Order(self.underlying, best_ask, hedge_qty))
            elif hedge_qty < 0 and len(extract_depth.buy_orders) > 0:
                best_bid = max(extract_depth.buy_orders.keys())
                hedge_orders.append(Order(self.underlying, best_bid, hedge_qty))
                
        result[self.underlying] = hedge_orders
        
        return result



        