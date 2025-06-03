import pandas as pd
import matplotlib.pyplot as plt

# Load the saved stats (or use the existing DataFrame)
monthly_stats = pd.read_csv("heatpump_monthly_broken_stats.csv")
monthly_stats["CommissionedMonth"] = pd.to_datetime(monthly_stats["CommissionedMonth"])

# Create plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Bar plot: Total commissioned
ax1.bar(monthly_stats["CommissionedMonth"], monthly_stats["TotalPumps"], color='skyblue', label='Total Commissioned')
ax1.set_xlabel("Commissioned Month")
ax1.set_ylabel("Total Pumps", color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Line plot: % Broken
ax2 = ax1.twinx()
ax2.plot(monthly_stats["CommissionedMonth"], monthly_stats["% Broken"], color='red', marker='o', label='% Broken')
ax2.set_ylabel("% Broken", color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and layout
plt.title("Monthly Commissioned Heat Pumps and % Broken")
fig.tight_layout()
plt.grid(True, axis='y', linestyle='--', alpha=0.4)

# Save or show
plt.savefig("plots/commissioning_vs_breakdown_rate.png", dpi=300)
plt.show()
