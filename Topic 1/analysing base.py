import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load the cleaned heatpumps data ===
df = pd.read_csv("heatpumps_cleaned_v2.csv")

# === Create folder for plots ===
os.makedirs("plots_heatmap_broken_configs", exist_ok=True)

# === Group and calculate % broken (State == 5) per configuration per model ===
grouped = df.groupby(['Model', 'BoilerType', 'DhwType'])

# Count total and broken units
summary = grouped['State'].agg(
    total='count',
    broken=lambda x: (x == 5).sum()
).reset_index()

# Calculate % broken
summary['broken_pct'] = (summary['broken'] / summary['total']) * 100

# === Plot heatmaps per model ===
models = sorted(df['Model'].dropna().unique())

for model in models:
    sub = summary[summary['Model'] == model].copy()
    if sub.empty:
        continue

    pivot = sub.pivot(index='BoilerType', columns='DhwType', values='broken_pct')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': '% Broken'})
    plt.title(f"Model {int(model)} – % Broken per BoilerType and DhwType")
    plt.xlabel("DHW Type")
    plt.ylabel("Boiler Type")
    plt.tight_layout()
    path = f"plots_heatmap_broken_configs/heatmap_model_{int(model)}.png"
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved heatmap: {path}")
# === Combined heatmap across all models ===
combined = df.groupby(['BoilerType', 'DhwType'])['State'].agg(
    total='count',
    broken=lambda x: (x == 5).sum()
).reset_index()

combined['broken_pct'] = (combined['broken'] / combined['total']) * 100

pivot_combined = combined.pivot(index='BoilerType', columns='DhwType', values='broken_pct')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_combined, annot=True, fmt=".1f", cmap="coolwarm", cbar_kws={'label': '% Broken'})
plt.title("All Models – % Broken per BoilerType and DhwType")
plt.xlabel("DHW Type")
plt.ylabel("Boiler Type")
plt.tight_layout()
plt.savefig("plots_heatmap_broken_configs/heatmap_all_models_combined.png")
plt.close()
print("✅ Saved heatmap: plots_heatmap_broken_configs/heatmap_all_models_combined.png")
