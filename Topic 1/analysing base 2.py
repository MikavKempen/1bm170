import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# === Load the cleaned dataset ===
df = pd.read_csv("heatpumps_cleaned_v2.csv")

# === Filter out uncertain configurations ===
df = df[(df['BoilerType'] != 0) & (df['DhwType'] != 0)]

# === Create output folder ===
output_dir = "plots_heatmap_broken_configs_with_counts_filtered"
os.makedirs(output_dir, exist_ok=True)

# === Group and calculate % broken per configuration per model ===
grouped = df.groupby(['Model', 'BoilerType', 'DhwType'])

summary = grouped['State'].agg(
    total='count',
    broken=lambda x: (x == 5).sum()
).reset_index()

summary['broken_pct'] = (summary['broken'] / summary['total']) * 100
summary['label'] = summary.apply(lambda row: f"{row['broken_pct']:.1f}%\n(n={int(row['total'])})", axis=1)

# === Plot heatmaps per model ===
models = sorted(df['Model'].dropna().unique())

for model in models:
    sub = summary[summary['Model'] == model]
    if sub.empty:
        continue

    pivot_values = sub.pivot(index='BoilerType', columns='DhwType', values='broken_pct')
    pivot_labels = sub.pivot(index='BoilerType', columns='DhwType', values='label')

    plt.figure(figsize=(8, 6))
    sns.heatmap(pivot_values, annot=pivot_labels, fmt="", cmap="coolwarm", cbar_kws={'label': '% Broken'})
    plt.title(f"Model {int(model)} – % Broken per BoilerType and DhwType (Filtered)")
    plt.xlabel("DHW Type")
    plt.ylabel("Boiler Type")
    plt.tight_layout()
    path = os.path.join(output_dir, f"heatmap_model_{int(model)}.png")
    plt.savefig(path)
    plt.close()
    print(f"✅ Saved heatmap: {path}")

# === Combined heatmap for all models ===
combined = df.groupby(['BoilerType', 'DhwType'])['State'].agg(
    total='count',
    broken=lambda x: (x == 5).sum()
).reset_index()

combined['broken_pct'] = (combined['broken'] / combined['total']) * 100
combined['label'] = combined.apply(lambda row: f"{row['broken_pct']:.1f}%\n(n={int(row['total'])})", axis=1)

pivot_values_combined = combined.pivot(index='BoilerType', columns='DhwType', values='broken_pct')
pivot_labels_combined = combined.pivot(index='BoilerType', columns='DhwType', values='label')

plt.figure(figsize=(8, 6))
sns.heatmap(pivot_values_combined, annot=pivot_labels_combined, fmt="", cmap="coolwarm", cbar_kws={'label': '% Broken'})
plt.title("All Models – % Broken per BoilerType and DhwType (Filtered)")
plt.xlabel("DHW Type")
plt.ylabel("Boiler Type")
plt.tight_layout()
combined_path = os.path.join(output_dir, "heatmap_all_models_combined.png")
plt.savefig(combined_path)
plt.close()
print(f"✅ Saved heatmap: {combined_path}")
