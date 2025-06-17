import pandas as pd

# --- Load all datasets ---
df_main = pd.read_csv("heatpump_data_combined_MES_operation_3_With_IDs.csv")
mes_ops = pd.read_csv("mes_operations_tagged.csv")
tests_cleaned_df = pd.read_csv("test_measurements_cleaned.csv")
test_df = pd.read_csv("test_measurements.csv")

# --- Ensure SerialNumber consistency ---
df_main["SerialNumber"] = df_main["SerialNumber"].astype(str).str.strip()
mes_ops["SerialNumber"] = mes_ops["SerialNumber"].astype(str).str.strip()
tests_cleaned_df["dut_sn"] = tests_cleaned_df["dut_sn"].astype(str).str.strip()
test_df["dut_sn"] = test_df["dut_sn"].astype(str).str.strip()

# --- Get all serial sets ---
all_serials = set(df_main["SerialNumber"])
mes_serials = set(mes_ops["SerialNumber"])
test_serials_cleaned = set(tests_cleaned_df["dut_sn"])
test_serials_raw = set(test_df["dut_sn"])

# --- Intersections ---
in_mes_and_main = all_serials & mes_serials
in_test_and_main = all_serials & test_serials_raw
in_test_cleaned_and_main = all_serials & test_serials_cleaned
in_all_three = all_serials & mes_serials & test_serials_raw

# --- Output for ALL heatpumps ---
print("All heatpumps:")
print(f"Total in main dataset: {len(all_serials)}")
print(f"In MES + main: {len(in_mes_and_main)}")
print(f"In test measurements + main: {len(in_test_and_main)}")
print(f"In cleaned test measurements + main: {len(in_test_cleaned_and_main)}")
print(f"In MES + test + main: {len(in_all_three)}")
print(f"Coverage (All): {(len(in_all_three) / len(all_serials)) * 100:.2f}%")

# --- Broken heatpumps only ---
broken_serials = set(df_main[df_main["State"] == 5]["SerialNumber"])
broken_in_mes = broken_serials & mes_serials
broken_in_test = broken_serials & test_serials_raw
broken_in_cleaned_test = broken_serials & test_serials_cleaned
broken_in_all_three = broken_serials & mes_serials & test_serials_raw

# --- Output for BROKEN heatpumps ---
print("\nBroken heatpumps:")
print(f"Total broken in main: {len(broken_serials)}")
print(f"In MES + broken: {len(broken_in_mes)}")
print(f"In test measurements + broken: {len(broken_in_test)}")
print(f"In cleaned test measurements + broken: {len(broken_in_cleaned_test)}")
print(f"In MES + test + broken: {len(broken_in_all_three)}")
print(f"Coverage (Broken): {(len(broken_in_all_three) / len(broken_serials)) * 100:.2f}%")
