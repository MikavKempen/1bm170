import pandas as pd
from datetime import time

# === Load CSVs ===
mes_df = pd.read_csv("mes_operations_tagged.csv", parse_dates=["BeginDateTime", "EndDateTime"])
heatpumps_df = pd.read_csv("heatpumps_cleaned_v2.csv")

# === Add target column ===
heatpumps_df['Target_Broken'] = (heatpumps_df['State'] == 5).astype(int)

# === Feature: has MES data ===
mes_presence = mes_df[['SerialNumber']].drop_duplicates()
mes_presence['has_mes_data'] = 1
heatpumps_df = heatpumps_df.merge(mes_presence, on='SerialNumber', how='left')
heatpumps_df['has_mes_data'] = heatpumps_df['has_mes_data'].fillna(0).astype(int)

# === Base Aggregates (per SerialNumber) ===
agg = mes_df.groupby("SerialNumber").agg(
    total_duration=("DurationSeconds", "sum"),
    total_operations=("DurationSeconds", "count"),
    avg_operation_duration=("DurationSeconds", "mean"),
    num_outliers_total=("OutlierReason", lambda x: (x != "None").sum()),
    num_outliers_iqr=("Outlier_IQR", "sum"),
    num_outliers_8h=("Exceeds8h", "sum"),
    num_short_duration=("ShortDuration", "sum"),
    unique_outlier_types=("OutlierReason", lambda x: x[x != "None"].nunique()),
    num_unique_operations=("MfgOrderOperationText", "nunique"),
    num_switches_workcenter=("WorkCenter", pd.Series.nunique),
    pct_outliers_total=("OutlierReason", lambda x: (x != "None").sum() / len(x)),
    first_operation_type=("MfgOrderOperationText", "first"),
    last_operation_type=("MfgOrderOperationText", "last"),
    total_process_timespan_seconds=("EndDateTime", lambda x: (x.max() - x.min()).total_seconds()),
    dominant_operation_type=("MfgOrderOperationText", lambda x: x.mode().iloc[0] if not x.mode().empty else "Unknown"),
    has_assembly=("MfgOrderOperationText", lambda x: int("Assembly" in x.values)),
    has_test=("MfgOrderOperationText", lambda x: int(any(t in x.values for t in ["FI", "FT", "LT"])))
).reset_index()

# === Merge with heatpump base data ===
result = heatpumps_df.merge(agg, on="SerialNumber", how="left")

# === Compute timing-based features per ID ===
def compute_timing_features(group):
    group = group.sort_values("BeginDateTime").reset_index(drop=True)

    gaps = (group["BeginDateTime"].shift(-1) - group["EndDateTime"]).dt.total_seconds()
    avg_gap = gaps.mean()
    max_gap = gaps.max()
    total_proc_time = (group["EndDateTime"].max() - group["BeginDateTime"].min()).total_seconds()

    has_invalid = (
        (group["EndDateTime"] < group["BeginDateTime"]) |
        (group["BeginDateTime"] == group["EndDateTime"]) |
        (group["BeginDateTime"] < group["EndDateTime"].shift())
    ).any()

    def is_night_or_weekend(dt):
        return dt.weekday() >= 5 or dt.time() < time(7) or dt.time() > time(19)

    has_night = group["BeginDateTime"].apply(is_night_or_weekend).any()

    return pd.Series({
        "total_process_duration": total_proc_time,
        "has_impossible_timing": int(has_invalid),
        "was_produced_outside_shift_hours": int(has_night)
    })

# Apply per ID
timing_features = mes_df.groupby("ID").apply(compute_timing_features).reset_index()

# Map IDs back to SerialNumber
id_to_serial = mes_df[['ID', 'SerialNumber']].drop_duplicates()
timing_features = timing_features.merge(id_to_serial, on="ID", how="left")

# Aggregate timing features per SerialNumber
agg_timing = timing_features.groupby("SerialNumber").agg({
    "total_process_duration": "sum",
    "has_impossible_timing": "max",
    "was_produced_outside_shift_hours": "max"
}).reset_index()

# Merge into result
result = result.merge(agg_timing, on="SerialNumber", how="left")

# === Save ===
result.to_csv("heatpump_data_combined_MES_operation_3_With_IDs.csv", index=False)
print("✅ Saved: heatpump_data_combined_MES_operation_3_With_IDs.csv")
