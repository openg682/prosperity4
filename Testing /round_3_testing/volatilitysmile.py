import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq

# 1. Black-Scholes Pricing Formula
def bs_call(S, K, T, r, sigma):
    if sigma <= 0.0:
        return max(0.0, S - K)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 2. Implied Volatility Solver
def calculate_iv(S, K, T, r, market_price):
    intrinsic = max(0.0, S - K)
    if market_price < intrinsic:
        return np.nan 
        
    objective = lambda sigma: bs_call(S, K, T, r, sigma) - market_price
    
    try:
        return brentq(objective, 1e-4, 5.0)
    except ValueError:
        return np.nan

# 3. Setup Multi-Day Parameters
days = [0, 1, 2]
ttes = {0: 8.0, 1: 7.0, 2: 6.0} 
colors = {0: '#1f77b4', 1: '#2ca02c', 2: '#9467bd'} 
r = 0.0 

print("Calculating Implied Volatility and generating separate charts...")

# 4. Loop through each day
for day in days:
    # --- CREATE A NEW CHART FOR THIS SPECIFIC DAY ---
    plt.figure(figsize=(10, 6))
    
    df = pd.read_csv(f"prices_round_3_day_{day}.csv", sep=";")
    snapshot = df[df['timestamp'] <= 100000].groupby('product')['mid_price'].mean().to_dict()
    
    S = snapshot.get('VELVETFRUIT_EXTRACT')
    T = ttes[day] / 365.0 
    
    strikes = []
    ivs = []
    
    for product, price in snapshot.items():
        if product.startswith('VEV_'):
            K = float(product.split('_')[1])
            iv = calculate_iv(S, K, T, r, price)
            
            strikes.append(K)
            ivs.append(iv)
            
    smile_data = pd.DataFrame({'Strike': strikes, 'IV': ivs}).dropna().sort_values('Strike')
    
    # Plot the data
    plt.plot(smile_data['Strike'], smile_data['IV'], marker='o', linestyle='-', 
             color=colors[day], linewidth=2, label=f'Day {day} IV')
    
    # Highlight the underlying price
    plt.axvline(x=S, color='gray', linestyle='--', alpha=0.7, label=f'Underlying Price ({S:.0f})')

    # --- FORMAT AND SAVE THIS SPECIFIC CHART ---
    plt.title(f'Volatility Smile: Velvetfruit Extract (Day {day}, TTE={ttes[day]}d)', fontsize=14, fontweight='bold')
    plt.xlabel('Strike Price', fontsize=12)
    plt.ylabel('Implied Volatility ($\sigma$)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Save with a dynamic filename based on the day
    filename = f"volatility_smile_day_{day}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Success! Saved {filename}")
    
    # Close the figure to free memory and prevent lines overlapping on the next loop
    plt.close()

print("\nAll charts generated successfully.")