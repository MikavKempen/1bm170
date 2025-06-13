import pandas as pd

# Load datasets
df_main = pd.read_csv("heatpump_data_combined_MES_operation_3_With_IDs.csv")
mes_ops = pd.read_csv("mes_operations_tagged.csv")
tests_df_1 = pd.read_csv("test_measurements_cleaned.csv")
tests_df_2 = pd.read_csv("test_measurements.csv")
genealogy_df = pd.read_csv("Dataset4-genealogy.csv")

# Ensure consistent serial formats
df_main["SerialNumber"] = df_main["SerialNumber"].astype(str).str.strip()
mes_ops["SerialNumber"] = mes_ops["SerialNumber"].astype(str).str.strip()
tests_df_1["dut_sn"] = tests_df_1["dut_sn"].astype(str).str.strip()
tests_df_2["dut_sn"] = tests_df_2["dut_sn"].astype(str).str.strip()

genealogy_df["child"] = genealogy_df["ChildSerialNumber"].astype(str).str.strip()
genealogy_df["parent"] = genealogy_df["ParentSerialNumber"].astype(str).str.strip()

# Define serial sets
serials_main = set(df_main["SerialNumber"])
serials_mes = set(mes_ops["SerialNumber"])
serials_test_1 = set(tests_df_1["dut_sn"])
serials_test_2 = set(tests_df_2["dut_sn"])
serials_genealogy = set(genealogy_df["child"]) | set(genealogy_df["parent"])

broken_serials = set(df_main[df_main["State"] == 5]["SerialNumber"])

# Define combinations
combinations = {
    "heatpumps": [serials_main],
    "heatpumps - mes operations": [serials_main, serials_mes],
    "heatpumps - test_measurement": [serials_main, serials_test_1],
    "heatpumps - test_measurement_2": [serials_main, serials_test_2],
    "heatpumps - genealogy": [serials_main, serials_genealogy],
    "heatpumps - mes operations - test_measurement": [serials_main, serials_mes, serials_test_1],
    "heatpumps - mes operations - test_measurement_2": [serials_main, serials_mes, serials_test_2],
    "heatpumps - mes operations - genealogy": [serials_main, serials_mes, serials_genealogy],
    "heatpumps - mes operations - test_measurement - genealogy": [serials_main, serials_mes, serials_test_1, serials_genealogy],
    "heatpumps - mes operations - test_measurement_2 - genealogy": [serials_main, serials_mes, serials_test_2, serials_genealogy]
}

# Compute intersections and matching broken
results = []

for label, sets in combinations.items():
    intersected = set.intersection(*sets)
    broken_matched = intersected & broken_serials
    results.append({
        "Dataset Combination": label,
        "Matching Serial Numbers": len(intersected),
        "Matching Serial Numbers Broken": len(broken_matched)
    })

# Create DataFrame and optionally save
results_df = pd.DataFrame(results)
results_df.to_csv("matching_serials_summary.csv", index=False)