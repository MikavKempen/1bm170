# Patch for NumPy 1.24+ compatibility with seaborn
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Load dataset ===
df = pd.read_csv("mes_operations_cleaned_flagged_veryshort.csv")

# === Verify required columns ===
required_cols = {
    'SerialNumber', 'Model', 'MfgOrderOperationText', 'DurationSeconds',
    'VeryShortDuration', 'Exceeds8h', 'Outlier_IQR'
}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns: {missing_cols}")

# === Filter only non-flagged and positive durations for plotting ===
df_clean = df[
    (df['VeryShortDuration'] == 0) &
    (df['Exceeds8h'] == 0) &
    (df['Outlier_IQR'] == 0) &
    (df['DurationSeconds'] > 0)
].copy()

# === Create output folders ===
os.makedirs("plots/boxplots_combined", exist_ok=True)
os.makedirs("plots/pdf_combined", exist_ok=True)

# === Create combined plots per operation ===
operations = df_clean['MfgOrderOperationText'].dropna().unique()

for op in operations:
    subset = df_clean[df_clean['MfgOrderOperationText'] == op]
    if subset.empty:
        continue

    safe_name = op.replace('/', '_').replace(' ', '_')

    # --- Boxplot per Model for this Operation ---
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=subset, x='Model', y='DurationSeconds')
    plt.title(f"Boxplot - Operation: {op} (Models compared)")
    plt.xlabel("Model")
    plt.ylabel("Duration (seconds)")
    plt.tight_layout()
    plt.savefig(f"plots/boxplots_combined/boxplot_{safe_name}.png")
    plt.close()

    # --- PDF per Model for this Operation ---
    plt.figure(figsize=(10, 5))
    for model in sorted(subset['Model'].dropna().unique()):
        model_data = subset[subset['Model'] == model]['DurationSeconds']
        if not model_data.empty:
            sns.kdeplot(model_data, label=f"Model {int(model)}", fill=True, alpha=0.3)
    plt.title(f"PDF – {op} (All Models, Positive Durations Only)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/pdf_combined/pdf_{safe_name}.png")
    plt.close()

# === Summary Statistics ===
serials_total = df['SerialNumber'].nunique()

# Serial numbers flagged on *any* criteria
flagged = df[
    (df['VeryShortDuration'] == 1) |
    (df['Exceeds8h'] == 1) |
    (df['Outlier_IQR'] == 1)
]
serials_flagged = flagged['SerialNumber'].nunique()

# Serial numbers that went through the same operation more than once
duplicates = df.groupby(['SerialNumber', 'MfgOrderOperationText']).size()
serials_with_repeats = duplicates[duplicates > 1].reset_index()['SerialNumber'].nunique()

# === Print Summary ===
print("🔍 Dataset summary:")
print(f"- Total unique serial numbers: {serials_total}")
print(f"- Serial numbers with at least one flagged step: {serials_flagged}")
print(f"- Serial numbers with duplicate operations: {serials_with_repeats}")
print("\n✅ Combined plots saved to 'plots/boxplots_combined' and 'plots/pdf_combined'")
import pandas as pd

# Load your dataset
df = pd.read_csv("mes_operations_cleaned_flagged_veryshort.csv")

# Count total number of operations (each row is one operation instance)
total_operations = len(df)

print(f"🛠️ Total number of operations in the dataset: {total_operations}")
