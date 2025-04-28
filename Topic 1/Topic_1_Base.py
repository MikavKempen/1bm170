import pandas as pd

# Load the CSV files
heatpumps = pd.read_csv('Dataset1-heatpumps.csv')
mes_operations = pd.read_csv('Dataset2-mes_operations.csv')
test_measurements = pd.read_csv('Dataset3-test_measurements.csv', low_memory=False)
genealogy = pd.read_csv('Dataset4-genealogy.csv')
monthly_energy_logs = pd.read_csv('Dataset5-monthly_energy_logs.csv')
process_plans = pd.read_csv('Dataset6-process_plans.csv')

# # Convert the 'CommissionedAt' column to datetime
# heatpumps['CommissionedAt'] = pd.to_datetime(heatpumps['CommissionedAt'], errors='coerce')
#
# # Select only devices that have been commissioned
# commissioned = heatpumps[heatpumps['CommissionedAt'].notnull()]
#
# # Group by model: count total commissioned devices and total broken devices
# model_stats = commissioned.groupby('Model').agg(
#     total_commissioned=('SerialNumber', 'count'),
#     total_broken=('Broken', 'sum')
# )
#
# # Calculate the percentage of broken devices per model
# model_stats['broken_percentage'] = (model_stats['total_broken'] / model_stats['total_commissioned']) * 100
#
# # Sort by broken percentage (highest first)
# model_stats = model_stats.sort_values(by='broken_percentage', ascending=False)
#
# # Print the model statistics
# print("Broken percentage by model:")
# print(model_stats)
#
# print("\n" + "="*80 + "\n")
#
# # ADDITION: Check consistency between 'Broken' and 'State'
# # Devices where State == 5 (officially broken) but Broken == 0 (not marked broken)
# state5_broken0 = heatpumps[(heatpumps['State'] == 5) & (heatpumps['Broken'] == 0)]
#
# # Devices where Broken == 1 (marked broken) but State != 5 (not officially broken)
# broken1_state_not5 = heatpumps[(heatpumps['Broken'] == 1) & (heatpumps['State'] != 5)]
#
# print(f"Number of devices with State = 5 but Broken = 0: {len(state5_broken0)}")
# print(f"Number of devices with Broken = 1 but State != 5: {len(broken1_state_not5)}")

# 1. Check missing values
print("Missing values per column:\n")
print(heatpumps.isnull().sum())
print("\n" + "="*50 + "\n")

# 2. Check unique values for important columns
important_columns = ['State', 'BoilerType', 'DhwType', 'Model', 'Broken']

for col in important_columns:
    print(f"Unique values in column '{col}':")
    print(heatpumps[col].unique())
    print("\n" + "="*50 + "\n")

# 3. Check data types
print("Data types of all columns:\n")
print(heatpumps.dtypes)

# Clean: force State = 5 where Broken = 1 but State != 5
heatpumps.loc[(heatpumps['Broken'] == 1) & (heatpumps['State'] != 5), 'State'] = 5

# Optionally, also fix the 3 devices where State == 5 but Broken == 0
# If you want a fully clean dataset:
heatpumps.loc[(heatpumps['State'] == 5) & (heatpumps['Broken'] == 0), 'Broken'] = 1


# Recheck inconsistencies after cleaning
state5_broken0 = heatpumps[(heatpumps['State'] == 5) & (heatpumps['Broken'] == 0)]
broken1_state_not5 = heatpumps[(heatpumps['Broken'] == 1) & (heatpumps['State'] != 5)]

print(f"After cleaning:")
print(f"Devices with State = 5 but Broken = 0: {len(state5_broken0)}")
print(f"Devices with Broken = 1 but State != 5: {len(broken1_state_not5)}")
