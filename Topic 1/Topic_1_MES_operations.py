import pandas as pd
from scipy.stats import describe

# === 1. Load dataset ===
df = pd.read_csv("Dataset2-mes_operations.csv")

print("=== Initial checks ===")
print("Missing values per column:\n")
print(df.isnull().sum())
print("\nShape before any cleaning:", df.shape)

# === 2. Parse datetime columns ===
df['BeginDateTime'] = pd.to_datetime(df['BeginDateTime'], errors='coerce')
df['EndDateTime'] = pd.to_datetime(df['EndDateTime'], errors='coerce')

# === 3. Drop rows with missing key info ===
df = df.dropna(subset=['BeginDateTime', 'EndDateTime', 'SerialNumber'])

# === 4. Compute durations ===
df['DurationSeconds'] = (df['EndDateTime'] - df['BeginDateTime']).dt.total_seconds()

# === 5. Flag zero and negative durations ===
zero_durations = df[df['DurationSeconds'] == 0]
negative_durations = df[df['DurationSeconds'] < 0]
print(f"\nZero-duration rows: {len(zero_durations)}")
print(f"Negative-duration rows: {len(negative_durations)}")

# === 6. Drop exact full-row duplicates ===
full_duplicates = df.duplicated()
print(f"\nExact duplicate rows: {full_duplicates.sum()}")
df = df[~full_duplicates]

# === 7. Drop duplicates with same SerialNumber, BeginDateTime, EndDateTime ===
dup_cols = ['SerialNumber', 'BeginDateTime', 'EndDateTime']
duplicates_based_on_time = df.duplicated(subset=dup_cols, keep='first')
print(f"Duplicate rows based on SerialNumber + time: {duplicates_based_on_time.sum()}")

df_dedup = df[~duplicates_based_on_time].copy()
df_dedup.reset_index(drop=True, inplace=True)

# === 8. Save cleaned versions ===
df.to_csv("mes_operations_cleaned.csv", index=False)
df_dedup.to_csv("mes_operations_cleaned_deduplicated.csv", index=False)
print("\nSaved:\n- 'mes_operations_cleaned.csv'\n- 'mes_operations_cleaned_deduplicated.csv'")
print("Shape after deduplication:", df_dedup.shape)

# === 9. Load heat pump models ===
heatpumps_df = pd.read_csv("heatpumps_cleaned_v2.csv")
df_dedup = df_dedup.merge(heatpumps_df[['SerialNumber', 'Model']], on='SerialNumber', how='left')

if df_dedup['Model'].isnull().sum() > 0:
    print("⚠️ Warning: Some SerialNumbers could not be matched to a Model.")

# === 10. Generate stats per (Model, Operation) ===
model_op_stats = df_dedup.groupby(['Model', 'MfgOrderOperationText'])['DurationSeconds'].agg(
    count='count',
    mean='mean',
    std='std',
    min='min',
    q1=lambda x: x.quantile(0.25),
    median='median',
    q3=lambda x: x.quantile(0.75),
    max='max'
).reset_index()
model_op_stats.to_csv("duration_stats_model_operation.csv", index=False)

# === 11. Generate stats per (ManufacturingOrder, Operation) ===
order_op_stats = df_dedup.groupby(['ManufacturingOrder', 'MfgOrderOperationText'])['DurationSeconds'].agg(
    count='count',
    mean='mean',
    std='std',
    min='min',
    q1=lambda x: x.quantile(0.25),
    median='median',
    q3=lambda x: x.quantile(0.75),
    max='max'
).reset_index()
order_op_stats.to_csv("duration_stats_order_operation.csv", index=False)

# === 12. Print summaries ===
print("\n=== Sample (Model, Operation) Duration Stats ===")
print(model_op_stats[['Model', 'MfgOrderOperationText', 'count', 'mean', 'median', 'std']].sample(5))

print("\n=== Top 5 (Model, Operation) by Median Duration ===")
print(model_op_stats[['Model', 'MfgOrderOperationText', 'median', 'mean', 'count']].sort_values(by='median', ascending=False).head())

print("\n=== Sample (ManufacturingOrder, Operation) Duration Stats ===")
print(order_op_stats[['ManufacturingOrder', 'MfgOrderOperationText', 'count', 'mean', 'median', 'std']].sample(5))

print("\n=== Top 5 (ManufacturingOrder, Operation) by Median Duration ===")
print(order_op_stats[['ManufacturingOrder', 'MfgOrderOperationText', 'median', 'mean', 'count']].sort_values(by='median', ascending=False).head())
