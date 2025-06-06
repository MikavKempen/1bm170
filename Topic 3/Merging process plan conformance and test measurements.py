import pandas as pd

# --- Load all datasets ---
df_main = pd.read_csv("heatpump_data_combined_MES_operation_3_With_IDs.csv")
mes_ops = pd.read_csv("mes_operations_tagged.csv")
process_plans = pd.read_csv("process_plans_cleaned.csv")
tests_df = pd.read_csv("test_measurements_cleaned.csv")

# --- Ensure SerialNumber consistency ---
df_main["SerialNumber"] = df_main["SerialNumber"].astype(str).str.strip()
mes_ops["SerialNumber"] = mes_ops["SerialNumber"].astype(str).str.strip()
tests_df["dut_sn"] = tests_df["dut_sn"].astype(str).str.strip()

# === PART 1: MES OPERATIONS FEATURES ===

# --- Expected Operations Mapping ---
expected_ops_dict = (
    process_plans
    .groupby("ProcessPlan")["OperationID"]
    .apply(list)
    .to_dict()
)

# --- Get actual operations per SerialNumber & ProcessPlan ---
actual_ops = (
    mes_ops
    .groupby(["SerialNumber", "ProcessPlan"])["ManufacturingOrderOperation"]
    .apply(list)
    .reset_index()
)
actual_ops["ExpectedOperations"] = actual_ops["ProcessPlan"].map(expected_ops_dict)
actual_ops = actual_ops.dropna(subset=["ExpectedOperations"])

# --- Compute missing & repeat stats ---
def compute_mes_features(row):
    expected = set(row["ExpectedOperations"])
    actual = pd.Series(row["ManufacturingOrderOperation"])
    actual_set = set(actual)

    missing_ops = len(expected - actual_set)
    repeated_counts = actual.value_counts()
    repeated = repeated_counts[repeated_counts > 1]
    total_repeats = (repeated - 1).sum()
    distinct_repeats = len(repeated)

    return pd.Series({
        "MissingOperations": missing_ops,
        "TotalRepeatedExecutions": total_repeats,
        "DistinctOperationsRepeated": distinct_repeats
    })

mes_features = actual_ops.apply(compute_mes_features, axis=1)
mes_features["SerialNumber"] = actual_ops["SerialNumber"]

# If multiple process plans per serial: aggregate
mes_features_agg = mes_features.groupby("SerialNumber").sum().reset_index()

# --- One-hot encode MfgOrderOperationText ---
op_presence = (
    mes_ops
    .groupby("SerialNumber")["MfgOrderOperationText"]
    .unique()
    .reset_index()
)

op_ohe = op_presence["MfgOrderOperationText"].apply(lambda ops: {op: 1 for op in ops})
op_ohe_df = pd.json_normalize(op_ohe).fillna(0).astype(int)
op_ohe_df["SerialNumber"] = op_presence["SerialNumber"]

# === Merge MES features into main dataset ===
df_augmented = df_main.merge(mes_features_agg, on="SerialNumber", how="left")
df_augmented = df_augmented.merge(op_ohe_df, on="SerialNumber", how="left")

# === PART 2: TEST MEASUREMENT FEATURES ===

# Create a set of serials that have test measurements
serials_with_tests = set(tests_df["dut_sn"])

# Add binary column to indicate presence of test measurements
df_augmented["has_test_measurements"] = df_augmented["SerialNumber"].apply(
    lambda sn: 1 if sn in serials_with_tests else 0
)

# --- Define test IDs of interest from the image (manually extracted) ---
relevant_test_ids = [
    "CoolingCompressorPressureOutInDelta",
    "GroundContinuityPowerPanel",
    "HeatingWaterTempOutInDeltaTestSystemCalori",
    "CoolingAirTempInOutDelta",
    "VoltageWithstandAc",
    "GroundContinuityFanCover",
    "AirTempInOutDelta",
    "GroundContinuityFanTopCover",
    "GroundContinuityFan",
    "GroundContinuityEvaporator",
    "HeatingAmbientHumidity",
    "HeatingTransientTempIn",
    "CompressorTempInAbsolute",
    "Vacuum not reached to soon",
    "HeatingAmbientTemperature",
    "CoolingWaterTempOutInDelta",
    "InitAmbientTemperature",
    "DeInitAmbientTemperature",
    "TestSystemPowerConsumed",
    "TestSystemPowerCreated",
    "StartWaterTempInTestSystem",
    "StartWaterTempOutTestSystem",
    "DuctCOP",
    "CompressorPressureInAbsolute"
]

# --- Compute number of failed tests ---
failed_tests = (
    tests_df[tests_df["test_passed"] == 0]
    .groupby("dut_sn")
    .size()
    .rename("NumberOfFailedTests")
    .reset_index()
    .rename(columns={"dut_sn": "SerialNumber"})
)

# --- Create status matrix for test IDs ---
status_matrix = []

# Loop over serial numbers
for serial, group in tests_df[tests_df["operation_id"].isin(relevant_test_ids)].groupby("dut_sn"):
    row = {"SerialNumber": serial}
    for test_id in relevant_test_ids:
        test_subset = group[group["operation_id"] == test_id]
        if len(test_subset) == 0:
            row[test_id] = float("nan")  # Not tested
        elif (test_subset["test_passed"] == 0).any():
            row[test_id] = 1  # At least one failure
        else:
            row[test_id] = 0  # Only passed
    status_matrix.append(row)

test_status_df = pd.DataFrame(status_matrix)

# --- Merge test features into main dataset ---
df_augmented = df_augmented.merge(failed_tests, on="SerialNumber", how="left")
df_augmented = df_augmented.merge(test_status_df, on="SerialNumber", how="left")

# Ensure consistent serial number format
df_augmented["SerialNumber"] = df_augmented["SerialNumber"].astype(str).str.strip()
mes_ops["SerialNumber"] = mes_ops["SerialNumber"].astype(str).str.strip()
tests_df["dut_sn"] = tests_df["dut_sn"].astype(str).str.strip()

# --- Get all serial sets ---
all_serials = set(df_augmented["SerialNumber"])
mes_serials = set(mes_ops["SerialNumber"])
test_serials = set(tests_df["dut_sn"])

# --- Intersections ---
in_both_mes_and_test = all_serials & mes_serials & test_serials

print(f"All heatpumps: {len(all_serials)} total")
print(f"In MES only: {len(all_serials & mes_serials)}")
print(f"In Tests only: {len(all_serials & test_serials)}")
print(f"In both MES & Tests: {len(in_both_mes_and_test)}")
print(f"Coverage (All): {(len(in_both_mes_and_test) / len(all_serials)) * 100:.2f}%")

# --- Broken heatpumps only ---
broken_serials = set(df_augmented[df_augmented["State"] == 5]["SerialNumber"])
broken_in_both = broken_serials & mes_serials & test_serials

print(f"\nBroken heatpumps: {len(broken_serials)} total")
print(f"In both MES & Tests: {len(broken_in_both)}")
print(f"Coverage (Broken): {(len(broken_in_both) / len(broken_serials)) * 100:.2f}%")

# Ensure consistency
mes_ops["SerialNumber"] = mes_ops["SerialNumber"].astype(str).str.strip()

# Convert the relevant columns to numeric
mes_ops["OpPlannedTotalQuantity"] = pd.to_numeric(mes_ops["OpPlannedTotalQuantity"], errors="coerce")
mes_ops["OpTotalConfirmedYieldQty"] = pd.to_numeric(mes_ops["OpTotalConfirmedYieldQty"], errors="coerce")
mes_ops["OpTotalConfirmedScrapQty"] = pd.to_numeric(mes_ops["OpTotalConfirmedScrapQty"], errors="coerce")

# Aggregate (sum) these per SerialNumber
quantity_aggregates = (
    mes_ops
    .groupby("SerialNumber")[
        ["OpPlannedTotalQuantity", "OpTotalConfirmedYieldQty", "OpTotalConfirmedScrapQty"]
    ]
    .sum()
    .reset_index()
)

# Merge into your final dataset
df_augmented = df_augmented.merge(quantity_aggregates, on="SerialNumber", how="left")

# === Final Output ===
# Save to CSV if needed
df_augmented.to_csv("heatpump_data_augmented.csv", index=False)

df_augmented_golf = df_augmented.copy()

# --- Step 2: Fill missing commissioning info ---
df_augmented_golf["CommissionedAt"] = df_augmented_golf["CommissionedAt"].fillna("Unknown")
df_augmented_golf["CommissionedMonth"] = df_augmented_golf["CommissionedMonth"].fillna("Unknown")

# --- Step 3: Drop rows where both MES and Test data are missing ---
df_augmented_golf = df_augmented_golf[
    ~((df_augmented_golf["has_mes_data"] == 0) & (df_augmented_golf["has_test_measurements"] == 0))
].copy()

# === Step 4: Fill missing MES and Test fields ===

# 1. --- Define MES columns ---
start_col = 'total_operations'
end_col = 'Top Preparation'
cols = list(df_augmented_golf.columns)

start_idx = cols.index(start_col)
end_idx = cols.index(end_col)

mes_columns = cols[start_idx:end_idx + 1] + [
    "OpPlannedTotalQuantity",
    "OpTotalConfirmedYieldQty",
    "OpTotalConfirmedScrapQty"
]

# Fill missing MES data with 'Unknown'
df_augmented_golf[mes_columns] = df_augmented_golf[mes_columns].fillna("Unknown")

# 2. --- Fill relevant test columns with 'Not tested' if missing and should be present ---
for col in relevant_test_ids:
    condition = (df_augmented_golf["has_test_measurements"] == 1) & (df_augmented_golf[col].isna())
    df_augmented_golf.loc[condition, col] = 2

# 3. --- Fill missing NumberOfFailedTests with 0 if has_test_measurements == 1 ---
missing_failed_tests = (df_augmented_golf["has_test_measurements"] == 1) & (df_augmented_golf["NumberOfFailedTests"].isna())
df_augmented_golf.loc[missing_failed_tests, "NumberOfFailedTests"] = 0

# --- Save final dataset ---
df_augmented_golf.to_csv("heatpump_augmented_golf.csv", index=False)

