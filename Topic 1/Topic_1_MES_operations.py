import pandas as pd
mes_operations = pd.read_csv('Dataset2-mes_operations.csv')

import pandas as pd

# Make sure BeginDateTime and EndDateTime are datetime
mes_operations['BeginDateTime'] = pd.to_datetime(mes_operations['BeginDateTime'], errors='coerce')
mes_operations['EndDateTime'] = pd.to_datetime(mes_operations['EndDateTime'], errors='coerce')

print("=== Missing Values ===\n")
print(mes_operations.isnull().sum())
print("\n" + "="*60 + "\n")

# Drop rows where BeginDateTime or EndDateTime are missing (can't calculate process time)
mes_operations_valid = mes_operations.dropna(subset=['BeginDateTime', 'EndDateTime'])

# Calculate operation duration in seconds
mes_operations_valid['DurationSeconds'] = (mes_operations_valid['EndDateTime'] - mes_operations_valid['BeginDateTime']).dt.total_seconds()

print("=== Operation Duration Statistics (in seconds) ===\n")
print(mes_operations_valid['DurationSeconds'].describe())

print("\n" + "="*60 + "\n")

# Optional: Check if there are any negative durations (data errors)
negative_durations = mes_operations_valid[mes_operations_valid['DurationSeconds'] < 0]

print(f"Number of operations with negative duration: {len(negative_durations)}")
if len(negative_durations) > 0:
    print("Sample of negative duration operations:")
    print(negative_durations[['SerialNumber', 'BeginDateTime', 'EndDateTime', 'DurationSeconds']].head())





# Make sure BeginDateTime and EndDateTime are proper datetimes
mes_operations['BeginDateTime'] = pd.to_datetime(mes_operations['BeginDateTime'], errors='coerce')
mes_operations['EndDateTime'] = pd.to_datetime(mes_operations['EndDateTime'], errors='coerce')

# Drop rows without a SerialNumber or EndDateTime (they are useless for production)
mes_operations_clean = mes_operations.dropna(subset=['SerialNumber', 'EndDateTime'])

# Find the first recorded EndDateTime per SerialNumber (first completion moment)
first_completion = mes_operations_clean.groupby('SerialNumber')['EndDateTime'].min().reset_index()

# Create a Month column
first_completion['ProductionMonth'] = first_completion['EndDateTime'].dt.to_period('M')

# Count number of units produced per month
production_per_month = first_completion.groupby('ProductionMonth').size().reset_index(name='UnitsProduced')

# Sort by date just to be sure
production_per_month = production_per_month.sort_values(by='ProductionMonth')

print(production_per_month)
