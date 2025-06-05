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

# --- Define test IDs of interest from the image (manually extracted) ---
relevant_test_ids = [
    "weight of R290 in heat pump",
    "GroundContinuityPowerPanel",
    "Weight of R290 in heat pump",
    "CoolingWaterTempOutAbsoluteCalori",
    "CoolingWaterTemperatureInOutDelta",
    "GroundContinuityFanCover",
    "HeatingWaterTempInAbsoluteCalori",
    "HeatingCompressorSpeedAbsolute",
    "CompressorTempInAbsolute",
    "GroundContinuityFan",
    "Total defrost time",
    "Time for state change from 7 > 0",
    "Time for state change from 0 > 7",
    "HeatingTransientTempIn",
    "PropaneCylinderWeight",
    "CompressorTempOutAbsolute",
    "StartCompressorPressureOut",
    "GroundContinuityEvaporator",
    "VacuumingTime",
    "StartAirTempOut",
    "StartWaterTempOut",
    "StartAirTempIn",
    "StartWaterTempIn"
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
for serial, group in tests_df[tests_df["id"].isin(relevant_test_ids)].groupby("dut_sn"):
    row = {"SerialNumber": serial}
    for test_id in relevant_test_ids:
        test_subset = group[group["id"] == test_id]
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

# === Final Output ===
# Save to CSV if needed
df_augmented.to_csv("heatpump_data_augmented.csv", index=False)
