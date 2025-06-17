import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

np.float = float  # For seaborn compatibility

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

# === Flag durations over 8 hours ===
mes_df['Exceeds8h'] = mes_df['DurationSeconds'] > 28800

# === IQR-based filtering ===
mes_df['Outlier_IQR'] = False
for (op, model), group in mes_df.groupby(['MfgOrderOperationText', 'Model']):
    q1 = group['DurationSeconds'].quantile(0.25)
    q3 = group['DurationSeconds'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    mask_iqr = (group['DurationSeconds'] < lower_bound) | (group['DurationSeconds'] > upper_bound)
    mes_df.loc[group[mask_iqr].index, 'Outlier_IQR'] = True

    print(f"{op} (Model {model}): Q1={q1:.2f}, Q3={q3:.2f}, IQR={iqr:.2f}, Lower={lower_bound:.2f}, Upper={upper_bound:.2f}, Outliers={mask_iqr.sum()}")

# === Save cleaned dataset ===
filtered_df = mes_df[~mes_df['Exceeds8h'] & ~mes_df['Outlier_IQR']].copy()
filtered_df.to_csv("mes_operations_cleaned_deduplicated_8h_IQR.csv", index=False)
print("Saved: mes_operations_cleaned_deduplicated_8h_IQR.csv")

# === Plotting main boxplots ===
os.makedirs("plots_cleaned_iqr_boxplots", exist_ok=True)
top_ops = filtered_df['MfgOrderOperationText'].value_counts().head(6).index

for op in top_ops:
    subset = filtered_df[filtered_df['MfgOrderOperationText'] == op]
    plt.figure(figsize=(10, 5))
    sns.boxplot(data=subset, x='Model', y='DurationSeconds')
    plt.title(f"Duration Distribution by Model — Operation: {op}")
    plt.ylabel("Duration (seconds)")
    plt.xlabel("Model")
    path = f"plots_cleaned_iqr_boxplots/boxplot_{op.replace('/', '_')}.png"
    plt.tight_layout()
    plt.savefig(path)
    print(f"Saved: {path}")
    plt.close()

# === Additional plots ===
os.makedirs("plots_additional_analysis", exist_ok=True)
df = pd.read_csv("mes_operations_cleaned_deduplicated_8h_IQR.csv")

# Assembly – Model 0 only
assembly_0 = df[(df['MfgOrderOperationText'] == 'Assembly') & (df['Model'] == 0.0)]
plt.figure(figsize=(8, 5))
sns.boxplot(data=assembly_0, x='Model', y='DurationSeconds')
plt.title("Boxplot – Assembly (Model 0 only)")
plt.ylabel("Duration (seconds)")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("plots_additional_analysis/boxplot_Assembly_Model0.png")
plt.close()

# Assembly – Models 1 to 3
assembly_1_3 = df[(df['MfgOrderOperationText'] == 'Assembly') & (df['Model'].isin([1.0, 2.0, 3.0]))]
plt.figure(figsize=(8, 5))
sns.boxplot(data=assembly_1_3, x='Model', y='DurationSeconds')
plt.title("Boxplot – Assembly (Models 1–3)")
plt.ylabel("Duration (seconds)")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("plots_additional_analysis/boxplot_Assembly_Models1to3.png")
plt.close()

# Packing – Histogram
packing = df[df['MfgOrderOperationText'] == 'Packing']
plt.figure(figsize=(8, 5))
sns.distplot(packing['DurationSeconds'], bins=50, kde=True, hist=True)
plt.title("Histogram – Packing Duration")
plt.xlabel("Duration (seconds)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("plots_additional_analysis/histogram_Packing.png")
plt.close()

# Packing – Boxplot
plt.figure(figsize=(8, 5))
sns.boxplot(data=packing, x='Model', y='DurationSeconds')
plt.title("Boxplot – Packing Operation")
plt.ylabel("Duration (seconds)")
plt.xlabel("Model")
plt.tight_layout()
plt.savefig("plots_additional_analysis/boxplot_Packing.png")
plt.close()

print("✅ All plots saved in 'plots_cleaned_iqr_boxplots' and 'plots_additional_analysis'")


# === Additional Summary Statistics ===
print("\n" + "=" * 80)
print("DESCRIPTIVE ANALYSIS OF MES OPERATIONS (CLEANED DATA)")
print("=" * 80)

# 1. General descriptive stats by operation
operation_stats = df.groupby('MfgOrderOperationText')['DurationSeconds'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print("\n1. Overall operation statistics (not model-specific):")
print(operation_stats)

# 2. Total assembly time per model
print("\n2. Total assembly duration per model:")
assembly_per_model = df[df['MfgOrderOperationText'] == 'Assembly'].groupby('Model')['DurationSeconds'].agg(['mean', 'std', 'count']).round(2)
print(assembly_per_model)

# 3. Mean duration of each operation per model (with std)
print("\n3. Mean and std duration of each operation per model:")
mean_std_per_op_model = df.groupby(['MfgOrderOperationText', 'Model'])['DurationSeconds'].agg(['mean', 'std', 'count']).round(2)
print(mean_std_per_op_model)


# 4. Flagged outliers – by operation
print("\n4. Number of flagged IQR outliers by operation:")
outliers_by_op = mes_df[mes_df['Outlier_IQR']].groupby('MfgOrderOperationText')['Outlier_IQR'].count()
print(outliers_by_op)

# 5. Flagged outliers – by operation and model
print("\n5. Number of flagged IQR outliers by operation and model:")
outliers_by_op_model = mes_df[mes_df['Outlier_IQR']].groupby(['MfgOrderOperationText', 'Model'])['Outlier_IQR'].count()
print(outliers_by_op_model)

# 6. Share of flagged outliers per operation
print("\n6. Outlier percentage per operation:")
counts_total = mes_df.groupby('MfgOrderOperationText')['DurationSeconds'].count()
counts_outliers = mes_df[mes_df['Outlier_IQR']].groupby('MfgOrderOperationText')['Outlier_IQR'].count()
outlier_share = ((counts_outliers / counts_total) * 100).fillna(0).round(2)
print(outlier_share)

# Optional: Export summary to CSV
summary_df = operation_stats.copy()
summary_df['% IQR Outliers'] = outlier_share
summary_df.to_csv("mes_operation_summary_stats.csv")
print("\nSummary exported to 'mes_operation_summary_stats.csv'")

# === Final Part: Distributions and Short Duration Flagging ===
print("\nGenerating distribution plots...")

# Add new flag for very short durations
df['ShortDuration'] = df['DurationSeconds'] < 3

# Create output folder
dist_folder = "plots_operation_distributions"
os.makedirs(dist_folder, exist_ok=True)

# Get all operations and all models
operations = df['MfgOrderOperationText'].unique()
all_models = [0.0, 1.0, 2.0, 3.0]

for op in operations:
    if op == 'Assembly':
        # Plot each model separately
        for model in all_models:
            subset = df[(df['MfgOrderOperationText'] == op) & (df['Model'] == model)]
            if len(subset) == 0:
                continue
            plt.figure(figsize=(8, 5))
            plt.hist(subset['DurationSeconds'], bins=50, alpha=0.6, edgecolor='black')
            plt.title(f"Duration Histogram – {op} (Model {int(model)})")
            plt.xlabel("Duration (seconds)")
            plt.ylabel("Count")
            plt.tight_layout()
            path = f"{dist_folder}/hist_{op.replace('/', '_')}_model{int(model)}.png"
            plt.savefig(path)
            plt.close()
    else:
        plt.figure(figsize=(8, 5))
        for model in all_models:
            subset = df[(df['MfgOrderOperationText'] == op) & (df['Model'] == model)]
            if len(subset) == 0:
                continue
            plt.hist(subset['DurationSeconds'], bins=50, alpha=0.5, label=f"Model {int(model)}", edgecolor='black', histtype='stepfilled')
        plt.title(f"Duration Histogram – {op} (All Models)")
        plt.xlabel("Duration (seconds)")
        plt.ylabel("Count")
        plt.legend()
        plt.tight_layout()
        path = f"{dist_folder}/hist_{op.replace('/', '_')}_all_models.png"
        plt.savefig(path)
        plt.close()

print(f"\n✅ Distributions saved in: {dist_folder}")

# === Report short durations
short_counts = df[df['ShortDuration']].groupby(['MfgOrderOperationText', 'Model']).size()
print("\nNumber of operations with duration < 3 seconds (flagged as ShortDuration):")
print(short_counts)
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Ensure seaborn styles
sns.set(style="whitegrid")

# Load the cleaned data
df = pd.read_csv("mes_operations_cleaned_deduplicated_8h_IQR.csv")

# Create output folder
os.makedirs("pdf_plots_clipped", exist_ok=True)

# Select operation
operation = "FI"  # change this for other operations
op_df = df[df['MfgOrderOperationText'] == operation]

# Filter out negative durations
op_df = op_df[op_df['DurationSeconds'] >= 0]

# Plot PDF for all models (combined)
plt.figure(figsize=(10, 6))
for model in sorted(op_df['Model'].unique()):
    model_df = op_df[op_df['Model'] == model]
    sns.kdeplot(
        model_df['DurationSeconds'],
        label=f"Model {int(model)}",
        clip=(0, None),  # This ensures values < 0 are excluded
        bw_adjust=0.7,
        fill=True,
        alpha=0.4
    )
plt.title(f"PDF – {operation} (All Models, No Negative Durations)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"pdf_plots_clipped/pdf_{operation}_all_models_clipped.png")
plt.close()

print(f"✅ PDF plot for '{operation}' saved without negative durations.")

# Plot PDF for all models (combined) – corrected version
plt.figure(figsize=(10, 6))
for model in sorted(op_df['Model'].unique()):
    model_df = op_df[op_df['Model'] == model]
    sns.kdeplot(
        model_df['DurationSeconds'],
        label=f"Model {int(model)}",
        clip=(0, float('inf')),  # Prevents negative values
        bw_adjust=0.7,
        fill=True,
        alpha=0.4
    )
plt.title(f"PDF – {operation} (All Models, No Negative Durations)")
plt.xlabel("Duration (seconds)")
plt.ylabel("Density")
plt.legend()
plt.tight_layout()
plt.savefig(f"pdf_plots_clipped/pdf_{operation}_all_models_clipped.png")
plt.close()

print(f"✅ PDF plot for '{operation}' saved without negative durations.")

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === Config ===
input_file = "mes_operations_cleaned_deduplicated_8h_IQR.csv"
output_folder = "pdf_plots_positive_range"
operation_filter = None  # Set to e.g. "FI" if you want only one operation

# === Load data ===
df = pd.read_csv(input_file)
sns.set(style="whitegrid")
os.makedirs(output_folder, exist_ok=True)

# === Filter durations ≥ 0 ===
df = df[df["DurationSeconds"] >= 0]

# === Get operations and models ===
operations = df['MfgOrderOperationText'].unique()
models = sorted(df['Model'].dropna().unique())

# === Generate plots ===
for op in operations:
    if operation_filter and op != operation_filter:
        continue

    op_df = df[df['MfgOrderOperationText'] == op]

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

    plt.title(f"PDF – {op} (All Models, Positive Durations Only)")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # === Save plot ===
    safe_name = op.replace("/", "_").replace(" ", "_")
    plt.savefig(f"{output_folder}/pdf_{safe_name}_all_models_positive.png")
    plt.close()

print(f"\n✅ All plots saved in folder: '{output_folder}'")

# === Count very short durations (0–3 seconds) per operation ===
short_durations = df[df['DurationSeconds'].isin([0, 1, 2, 3])]
short_counts = short_durations.groupby(['MfgOrderOperationText', 'DurationSeconds']).size().unstack(fill_value=0)

print("\nCount of durations exactly 0, 1, 2, or 3 seconds per operation:")
print(short_counts)

