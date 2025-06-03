import pandas as pd

# === Load cleaned dataset ===
file_path = "mes_operations_cleaned_deduplicated_8h_IQR.csv"
df = pd.read_csv(file_path)

# === Basic info ===
print(f"✅ Loaded: {file_path}")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}\n")

# === Missing values ===
missing = df.isnull().sum()
print("Missing values per column:\n", missing[missing > 0], "\n")

# === Summary statistics per operation ===
summary_stats = df.groupby('MfgOrderOperationText')['DurationSeconds'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
print("Descriptive stats per operation:\n", summary_stats)

# === Count of short durations (0–3 seconds) per operation ===
short_durations = df[df['DurationSeconds'].isin([0, 1, 2, 3])]
short_counts = short_durations.groupby(['MfgOrderOperationText', 'DurationSeconds']).size().unstack(fill_value=0)
print("\nCount of durations exactly 0, 1, 2, or 3 seconds per operation:\n", short_counts)

# === Export summaries to CSV ===
summary_stats.to_csv("summary_stats_per_operation.csv")
short_counts.to_csv("short_durations_0_1_2_3_counts.csv")

print("\n📁 Exported:\n - summary_stats_per_operation.csv\n - short_durations_0_1_2_3_counts.csv")

