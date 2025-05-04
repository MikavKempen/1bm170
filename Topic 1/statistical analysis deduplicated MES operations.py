import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

np.float = float  # Patch for seaborn/numpy compatibility

# === Load and merge ===
mes_df = pd.read_csv("mes_operations_cleaned_deduplicated.csv", parse_dates=['BeginDateTime', 'EndDateTime'])
heatpumps_df = pd.read_csv("heatpumps_cleaned_v2.csv")
mes_df = mes_df.merge(heatpumps_df[['SerialNumber', 'Model']], on='SerialNumber', how='left')

# Drop rows without model
missing_models = mes_df['Model'].isna().sum()
if missing_models > 0:
    print(f"⚠️ Missing Model info for {missing_models} rows")
mes_df = mes_df.dropna(subset=['Model'])

# Compute duration if needed
if 'DurationSeconds' not in mes_df.columns:
    mes_df['DurationSeconds'] = (mes_df['EndDateTime'] - mes_df['BeginDateTime']).dt.total_seconds()

# Drop invalid durations
mes_df = mes_df[mes_df['DurationSeconds'].notnull() & (mes_df['DurationSeconds'] >= 0)]

# === Fixed upper cutoffs per operation ===
upper_cutoff_dict = {
    'Assembly': 3000,
    'ST': 150,
    'FI': 200,
    'PROP': 3000,
    'MAI': 1000,
    'FT': 4000
}

# === Filtering logic ===
filtered_rows = []

for (op, model), group in mes_df.groupby(['MfgOrderOperationText', 'Model']):
    durations = group['DurationSeconds']
    mean = durations.mean()
    std = durations.std()
    q_low = durations.quantile(0.01)
    lower_cutoff = max(q_low, mean - 3 * std, mean / 10000)

    # Apply fixed upper cutoff if defined, else use default
    if op in upper_cutoff_dict:
        upper_cutoff = upper_cutoff_dict[op]
    else:
        q_high = durations.quantile(0.99)
        upper_cutoff = min(q_high, mean + 3 * std, 28800, mean * 50)

    filtered = group[(durations >= lower_cutoff) & (durations <= upper_cutoff)]
    print(f"{op} (Model {model}): {len(group)} → {len(filtered)} | Cutoff: {lower_cutoff:.1f} - {upper_cutoff}")
    filtered_rows.append(filtered)

df_filtered = pd.concat(filtered_rows)

# === Save plots ===
os.makedirs("plots_cleaned_boxplots", exist_ok=True)
top_ops = df_filtered['MfgOrderOperationText'].value_counts().head(6).index

for op in top_ops:
    subset = df_filtered[df_filtered['MfgOrderOperationText'] == op]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=subset, x='Model', y='DurationSeconds')
    plt.title(f"Duration Distribution by Model — Operation: {op}")
    plt.ylabel("Duration (seconds)")
    plt.xlabel("Model")
    path = f"plots_cleaned_boxplots/boxplot_{op.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()
