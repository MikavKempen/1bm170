import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("heatpump_augmented_mika.csv")

# Filter out BoilerType and DhwType == 0
df = df[(df["BoilerType"] != 0) & (df["DhwType"] != 0)]

# Group and compute failure rate in %
rate = (
    df
    .groupby(["DhwType", "BoilerType"])["Target_Broken"]
    .mean()
    .mul(100)
    .reset_index()
)

# Pivot for heatmap
pivot = rate.pivot(index="DhwType", columns="BoilerType", values="Target_Broken")

# Plot
plt.figure(figsize=(6, 4))
ax = sns.heatmap(
    pivot,
    annot=True,
    fmt=".1f",
    cmap="coolwarm",
    vmin=0, vmax=4,
    annot_kws={"fontsize": 14},
    cbar_kws={"label": "% Broken"}  # Removed 'fontsize'
)

# Manually set font size of colorbar label
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.label.set_size(12)

# Adjust other fonts
ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=12)
ax.set_title("All Models – % Broken per DHW Type (y) & Boiler Type (x)", fontsize=14)
ax.set_xlabel("Boiler Type", fontsize=13)
ax.set_ylabel("DHW Type", fontsize=13)

plt.tight_layout()
plt.show()
