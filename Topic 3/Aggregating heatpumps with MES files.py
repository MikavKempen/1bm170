import pandas as pd

# === Load CSVs (from current Topic 3 folder) ===
mes_df = pd.read_csv("mes_operations_tagged.csv", parse_dates=["BeginDateTime", "EndDateTime"])
heatpumps_df = pd.read_csv("heatpumps_cleaned_v2.csv")

# === Add target column: 1 if State == 5 ===
heatpumps_df['Target_Broken'] = (heatpumps_df['State'] == 5).astype(int)

# === Aggregate MES features per SerialNumber ===
agg = mes_df.groupby("SerialNumber").agg(
    total_duration=("DurationSeconds", "sum"),
    total_operations=("DurationSeconds", "count"),
    avg_operation_duration=("DurationSeconds", "mean"),
    std_operation_duration=("DurationSeconds", "std"),
    num_outliers_total=("OutlierReason", lambda x: (x != "None").sum()),
    num_outliers_iqr=("Outlier_IQR", "sum"),
    num_outliers_8h=("Exceeds8h", "sum"),
    num_short_duration=("ShortDuration", "sum"),
    unique_outlier_types=("OutlierReason", lambda x: x[x != "None"].nunique()),
    num_unique_operations=("MfgOrderOperationText", "nunique"),
    num_switches_workcenter=("WorkCenter", pd.Series.nunique)
).reset_index()

# === Merge and save ===
result = heatpumps_df.merge(agg, on="SerialNumber", how="left")
result.to_csv("heatpumps_with_general_features.csv", index=False)

print("✅ heatpumps_with_general_features.csv saved to Topic 3/")
