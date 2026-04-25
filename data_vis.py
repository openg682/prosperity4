import pandas as pd
import matplotlib.pyplot as plt

# 1. Configuration
days = [0, 1, 2]
prices_list = []

print("Loading and processing CSV files...")

# 2. Load and concatenate the prices data
for day in days:
    # Read the semicolon-separated files
    file_path = f"prices_round_3_day_{day}.csv"
    df = pd.read_csv(file_path, sep=';')
    
    # Prosperity timestamps usually go up to 1,000,000 per day. 
    # We offset them so Day 1 starts where Day 0 ends.
    df['continuous_timestamp'] = (df['day'] * 1000000) + df['timestamp']
    prices_list.append(df)

# Combine all 3 days into one large DataFrame
all_prices = pd.concat(prices_list, ignore_index=True)

# 3. Extract the assets we want to compare
# Let's compare the underlying Extract against a middle-strike option
underlying = all_prices[all_prices['product'] == 'VELVETFRUIT_EXTRACT']
option_5000 = all_prices[all_prices['product'] == 'VEV_5000']

# 4. Create the dual-axis plot
fig, ax1 = plt.subplots(figsize=(14, 7))

# Plot the underlying asset on the primary (left) y-axis
color1 = '#1f77b4' # standard blue
ax1.set_xlabel('Continuous Timestamp (Days 0, 1, 2)')
ax1.set_ylabel('VELVETFRUIT_EXTRACT Mid Price', color=color1, fontweight='bold')
ax1.plot(underlying['continuous_timestamp'], underlying['mid_price'], color=color1, label='Extract (Underlying)')
ax1.tick_params(axis='y', labelcolor=color1)

# Create a secondary (right) y-axis for the option
ax2 = ax1.twinx()
color2 = '#ff7f0e' # standard orange
ax2.set_ylabel('VEV_5000 Call Option Mid Price', color=color2, fontweight='bold')
ax2.plot(option_5000['continuous_timestamp'], option_5000['mid_price'], color=color2, alpha=0.8, label='VEV_5000')
ax2.tick_params(axis='y', labelcolor=color2)

# 5. Formatting
plt.title('Price Correlation: Velvetfruit Extract vs. Strike 5000 Call Option', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Combine legends from both axes
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

fig.tight_layout()
plt.show()