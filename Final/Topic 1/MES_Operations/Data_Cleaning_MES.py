import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

np.float = float  # For seaborn compatibility
sns.set(style="whitegrid")

# === Load data ===
mes_df = pd.read_csv("mes_operations_cleaned.csv", parse_dates=['BeginDateTime', 'EndDateTime'])
heatpumps_df = pd.read_csv("heatpumps_cleaned_v2.csv")

# === Drop exact duplicates ===
print(f"Original rows: {len(mes_df)}")
mes_df = mes_df.drop_duplicates(subset=['SerialNumber', 'BeginDateTime', 'EndDateTime'])
print(f"After dropping duplicates: {len(mes_df)}")

# === Merge model info ===
mes_df = mes_df.merge(heatpumps_df[['SerialNumber', 'Model']], on='SerialNumber', how='left')
missing_models = mes_df['Model'].isna().sum()
if missing_models > 0:
    print(f"⚠️ Missing Model info for {missing_models} rows")
mes_df = mes_df.dropna(subset=['Model'])

# === Compute duration ===
mes_df['DurationSeconds'] = (mes_df['EndDateTime'] - mes_df['BeginDateTime']).dt.total_seconds()
mes_df = mes_df[mes_df['DurationSeconds'].notnull() & (mes_df['DurationSeconds'] >= 0)]

# === Flag specific outliers ===
mes_df['Exceeds8h'] = mes_df['DurationSeconds'] > 28800
mes_df['ShortDuration'] = mes_df['DurationSeconds'] < 3
mes_df['Outlier_IQR'] = False

# === IQR-based outliers per operation-model group ===
for (op, model), group in mes_df.groupby(['MfgOrderOperationText', 'Model']):
    q1 = group['DurationSeconds'].quantile(0.25)
    q3 = group['DurationSeconds'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    mask_iqr = (group['DurationSeconds'] < lower_bound) | (group['DurationSeconds'] > upper_bound)
    mes_df.loc[group[mask_iqr].index, 'Outlier_IQR'] = True
    print(f"{op} (Model {model}): Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, Outliers={mask_iqr.sum()}")

# === Consolidated OutlierReason column ===
def classify_reason(row):
    if row['DurationSeconds'] == 0:
        return 'ZeroDuration'
    elif row['ShortDuration']:
        return 'ShortDuration'
    elif row['Exceeds8h']:
        return 'TooLong'
    elif row['Outlier_IQR']:
        return 'IQR_Outlier'
    else:
        return 'None'

mes_df['OutlierReason'] = mes_df.apply(classify_reason, axis=1)

# === Save full dataset with outlier flags ===
mes_df.to_csv("mes_operations_tagged.csv", index=False)
print("✅ Saved: mes_operations_tagged.csv (all rows with flags)")

# === Save filtered clean-only dataset (optional) ===
clean_df = mes_df[mes_df['OutlierReason'] == 'None'].copy()
clean_df.to_csv("mes_operations_cleaned.csv", index=False)
print("✅ Saved: mes_operations_cleaned.csv (cleaned subset)")

# === Visualizations using clean_df ===
os.makedirs("plots_cleaned_iqr_boxplots", exist_ok=True)
top_ops = clean_df['MfgOrderOperationText'].value_counts().head(6).index

for op in top_ops:
    subset = clean_df[clean_df['MfgOrderOperationText'] == op]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=subset, x='Model', y='DurationSeconds')
    plt.title(f"Duration Distribution by Model — Operation: {op}")
    plt.ylabel("Duration (seconds)")
    plt.xlabel("Model")
    path = f"plots_cleaned_iqr_boxplots/boxplot_{op.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# === Summary statistics ===
print("\n" + "="*80)
print("DESCRIPTIVE ANALYSIS — Cleaned Only")
print("="*80)
operation_stats = clean_df.groupby('MfgOrderOperationText')['DurationSeconds'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print(operation_stats)

summary_df = operation_stats.copy()
summary_df.to_csv("mes_operation_summary_stats.csv")
print("📄 Summary exported to 'mes_operation_summary_stats.csv'")

# === Plot duration histograms (clean_df) ===
dist_folder = "plots_operation_distributions"
os.makedirs(dist_folder, exist_ok=True)

operations = clean_df['MfgOrderOperationText'].unique()
models = sorted(clean_df['Model'].dropna().unique())

for op in operations:
    op_df = clean_df[clean_df['MfgOrderOperationText'] == op]
    plt.figure(figsize=(10, 6))
    for model in models:
        model_df = op_df[op_df['Model'] == model]
        if not model_df.empty:
            sns.kdeplot(
                data=model_df,
                x='DurationSeconds',
                label=f"Model {int(model)}",
                bw_adjust=0.7,
                fill=True,
                alpha=0.4
            )
    plt.title(f"PDF – {op} (Cleaned Data)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    safe_name = op.replace("/", "_").replace(" ", "_")
    plt.savefig(f"{dist_folder}/pdf_{safe_name}_cleaned.png")
    plt.close()

# === Report short durations across full dataset ===
short_counts = mes_df[mes_df['ShortDuration']].groupby(['MfgOrderOperationText', 'Model']).size()
print("\n🔍 Count of durations < 3 seconds (full dataset):")
print(short_counts)
