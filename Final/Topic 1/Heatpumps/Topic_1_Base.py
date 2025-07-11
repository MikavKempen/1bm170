import pandas as pd

# Load the CSV files
heatpumps = pd.read_csv('Dataset1-heatpumps.csv', delimiter=';')
mes_operations = pd.read_csv('Dataset2-mes_operations.csv')
test_measurements = pd.read_csv('Dataset3-test_measurements.csv', low_memory=False)
genealogy = pd.read_csv('Dataset4-genealogy.csv')
monthly_energy_logs = pd.read_csv('Dataset5-monthly_energy_logs.csv')
process_plans = pd.read_csv('Dataset6-process_plans.csv')

import pandas as pd

# Load the dataset
heatpumps = pd.read_csv('Dataset1-heatpumps.csv')

# 1. ===== Basic checks =====
print("=== Missing values ===")
print(heatpumps.isnull().sum())
print("\n=== Duplicate SerialNumbers ===")
print(heatpumps['SerialNumber'].duplicated().sum())

# Drop duplicates if they exist (based on SerialNumber)
heatpumps = heatpumps.drop_duplicates(subset='SerialNumber')

# 2. ===== CommissionedAt analysis =====
heatpumps['CommissionedAt'] = pd.to_datetime(heatpumps['CommissionedAt'], errors='coerce')
commissioned = heatpumps[heatpumps['CommissionedAt'].notnull()]
print(f"\nCommissioned devices: {len(commissioned)} out of {len(heatpumps)}")
print(f"Percentage commissioned: {len(commissioned) / len(heatpumps) * 100:.2f}%")
print(f"Earliest commission date: {commissioned['CommissionedAt'].min()}")
print(f"Latest commission date: {commissioned['CommissionedAt'].max()}")

# 3. ===== State column analysis =====
print("\n=== State distribution ===")
print(heatpumps['State'].value_counts(dropna=False))

# 4. ===== BoilerType and DhwType analysis =====
print("\n=== BoilerType distribution ===")
print(heatpumps['BoilerType'].value_counts(dropna=False))
print("\n=== DhwType distribution ===")
print(heatpumps['DhwType'].value_counts(dropna=False))

# 5. ===== Model distribution =====
print("\n=== Model distribution ===")
print(heatpumps['Model'].value_counts(dropna=False))

# 6. ===== Merge Broken and State columns =====
# Fix: If Broken == 1 but State != 5, set State = 5
heatpumps.loc[(heatpumps['Broken'] == 1) & (heatpumps['State'] != 5), 'State'] = 5
# Fix: If State == 5 but Broken == 0, set Broken = 1
heatpumps.loc[(heatpumps['State'] == 5) & (heatpumps['Broken'] == 0), 'Broken'] = 1

# Now drop the Broken column
heatpumps = heatpumps.drop(columns=['Broken'])

# 7. ===== Optional DHW percentage per model =====
dhw_by_model = heatpumps.groupby('Model')['DhwType'].value_counts(normalize=True).unstack().fillna(0) * 100
print("\n=== % DHW availability per model ===")
print(dhw_by_model)

# 8. ===== Save cleaned version (optional step) =====
heatpumps.to_csv('heatpumps_cleaned.csv', index=False)
print("\nCleaned dataset saved to 'heatpumps_cleaned.csv'")

import pandas as pd

# Load cleaned dataset
heatpumps = pd.read_csv('heatpumps_cleaned.csv', parse_dates=['CommissionedAt'])

print("=== Extra Consistency Checks ===")
from datetime import datetime
import pandas as pd

# Use timezone-aware timestamp
now = pd.Timestamp.now(tz='UTC')  # <--- fix here
future_commissions = heatpumps[heatpumps['CommissionedAt'] > now]


# 1. Check for future CommissionedAt dates

future_commissions = heatpumps[heatpumps['CommissionedAt'] > now]
print(f"\nNumber of future CommissionedAt dates: {len(future_commissions)}")
if len(future_commissions) > 0:
    print(future_commissions[['SerialNumber', 'CommissionedAt']].head())

# 2. State = Active (3), but no CommissionedAt → this is likely a data error
active_without_commissioned = heatpumps[(heatpumps['State'] == 3) & (heatpumps['CommissionedAt'].isna())]
print(f"\nNumber of 'Active' heat pumps without CommissionedAt: {len(active_without_commissioned)}")
if len(active_without_commissioned) > 0:
    print(active_without_commissioned[['SerialNumber', 'State', 'CommissionedAt']].head())

# 3. Check for values outside expected category ranges
expected_states = {1, 2, 3, 4, 5}
unexpected_states = heatpumps[~heatpumps['State'].isin(expected_states)]
print(f"\nNumber of invalid 'State' values: {len(unexpected_states)}")

expected_boiler = {0, 1, 2, 3}
unexpected_boiler = heatpumps[~heatpumps['BoilerType'].isin(expected_boiler)]
print(f"Number of invalid 'BoilerType' values: {len(unexpected_boiler)}")

expected_dhw = {0, 1, 2}
unexpected_dhw = heatpumps[~heatpumps['DhwType'].isin(expected_dhw)]
print(f"Number of invalid 'DhwType' values: {len(unexpected_dhw)}")

# 4. Add a CommissionedMonth column (optional, useful for plotting trends)
heatpumps['CommissionedMonth'] = heatpumps['CommissionedAt'].dt.to_period('M')
print("\nSample of CommissionedMonth:")
print(heatpumps[['SerialNumber', 'CommissionedAt', 'CommissionedMonth']].dropna().head())

# (Optional) Save this enriched version
heatpumps.to_csv('heatpumps_cleaned_v2.csv', index=False)
print("\nEnriched dataset saved to 'heatpumps_cleaned_v2.csv'")




#constructing clear overview of the data

print("\n" + "="*80)
print("SUMMARY OF CLEANED DATASET: HEATPUMPS.CSV")
print("="*80)

# 1. CommissionedAt
commissioned_count = heatpumps['CommissionedAt'].notnull().sum()
total_count = len(heatpumps)
percent_commissioned = commissioned_count / total_count * 100
print(f"\nCommissionedAt:")
print(f"- Total commissioned: {commissioned_count} out of {total_count} ({percent_commissioned:.2f}%)")
print(f"- Earliest commissioning date: {heatpumps['CommissionedAt'].min()}")
print(f"- Latest commissioning date: {heatpumps['CommissionedAt'].max()}")

# Monthly commissioning check
print("\n- Sample CommissionedMonth values:")
print(heatpumps['CommissionedMonth'].dropna().value_counts().sort_index().head())

# Check for active but uncommissioned
active_without_commissioned = heatpumps[(heatpumps['State'] == 3) & (heatpumps['CommissionedAt'].isna())]
print(f"- Active but not commissioned: {len(active_without_commissioned)}")

# 2. State distribution
print("\nState:")
print(heatpumps['State'].value_counts().sort_index())

# 3. BoilerType
print("\nBoilerType:")
print(heatpumps['BoilerType'].value_counts().sort_index())

# 4. DhwType
print("\nDhwType:")
print(heatpumps['DhwType'].value_counts().sort_index())

# 5. % DHW availability per model
print("\n% DHW Availability per Model:")
print(dhw_by_model.round(2))

# 6. Model distribution
print("\nModel:")
print(heatpumps['Model'].value_counts().sort_index())

# 7. Final note on Broken merge
print("\nBroken:")
print("- 'Broken' column successfully merged into 'State = 5'")
print("- Column 'Broken' was removed to avoid duplication")

print("\n" + "="*80 + "\n")


print(heatpumps.describe())



import pandas as pd

# Load the cleaned dataset
heatpumps = pd.read_csv('heatpumps_cleaned_v2.csv', parse_dates=['CommissionedAt'])

# Create CommissionedMonth column, using 'Uncommissioned' for missing dates
heatpumps['CommissionedMonth'] = heatpumps['CommissionedAt'].dt.to_period('M').astype(str)
heatpumps['CommissionedMonth'] = heatpumps['CommissionedMonth'].fillna('Uncommissioned')

# Filter for BoilerType = 0 and DhwType = 0
boiler0 = heatpumps[heatpumps['BoilerType'] == 0]
dhw0 = heatpumps[heatpumps['DhwType'] == 0]

# Group by month and count
boiler0_counts = boiler0['CommissionedMonth'].value_counts().sort_index()
dhw0_counts = dhw0['CommissionedMonth'].value_counts().sort_index()

# Print the results
print("Occurrences of BoilerType = 0 per month (including uncommissioned):")
print(boiler0_counts)

print("\nOccurrences of DhwType = 0 per month (including uncommissioned):")
print(dhw0_counts)
