import pandas as pd
import matplotlib.pyplot as plt

# Load the test measurements CSV from the local project folder
df = pd.read_csv('Dataset3-test_measurements.csv', low_memory=False)

# 1. Drop 'uncertainty' and 'method' columns
print("1. Drop 'uncertainty' and 'method' columns \n")
df = df.drop(columns=['uncertainty', 'method'])

# 2. Missing values
print("2. Missing values per column:")
print(df.isnull().sum(), "\n")

# 2.1. Drop rows where 'value' is missing
print("2.1. Drop rows where 'value' is missing")
missing_value_rows = df[df['value'].isnull()]
print(missing_value_rows[['value', 'lower_specification_limit', 'upper_specification_limit']], "\n")
df = df.dropna(subset=['value'])

# 2.3. Fill in/drop missing limits
print("2.3. Fill in/drop missing limits")
# Get rows with missing lower or upper limits
missing_limits = df[df['lower_specification_limit'].isnull() | df['upper_specification_limit'].isnull()]

# Unique operation_id values where limits are missing
unique_ops_with_missing_limits = missing_limits['operation_id'].dropna().unique()
print("Unique operation_id values with missing limits:", unique_ops_with_missing_limits)

# Get indexes of rows where either lower or upper specification limit is missing
missing_limits_idx = df[df['lower_specification_limit'].isnull() | df['upper_specification_limit'].isnull()].index
print(missing_limits_idx.tolist(), "\n")

# Operation IDs with missing lower limits that should be set to 0
lower_limit_fix_ids = [
    "PressureAfterVacuuming", "PressureAfterVenting", "Total defrost time",
    "Time for state change from 0 > 7", "Time for state change from 7 > 0"
]

# Operation IDs with missing upper limits that should be set to 0
upper_limit_fix_ids = [
    "VacuumingTime", "PropaneCylinderWeight",
    "CompressorPressureOutInDelta", "AirTempInOutDelta",
    "WaterTempOutInDelta", "CompressorTempOutInDelta",
    "WaterTempOutInDeltaTestSystemCalori", "WaterTempOutInDeltaTestSystem"
]

# Set missing lower limits to 0
df.loc[df['operation_id'].isin(lower_limit_fix_ids) & df['lower_specification_limit'].isnull(), 'lower_specification_limit'] = 0

# Set missing upper limits to 0
df.loc[df['operation_id'].isin(upper_limit_fix_ids) & df['upper_specification_limit'].isnull(), 'upper_specification_limit'] = 0

# Operation IDs with missing lower limits that should be set to 0
lower_fix_ids = [
    'Vacuuming', 'Pressure while vacuuming', 'Pressure while releasing',
    'Pressure after releasing', 'Pressure while releasing gas when emptying system'
]

# Operation IDs with missing upper limits that should be set to 0
upper_fix_ids = [
    'Vacuum not reached to soon', 'Weight of propane cilinder at beginning of filling'
]

# Set missing lower limits to 0 for specified operation_ids
df.loc[df['operation_id'].isin(lower_fix_ids) & df['lower_specification_limit'].isnull(), 'lower_specification_limit'] = 0

# Set missing upper limits to 0 for specified operation_ids
df.loc[df['operation_id'].isin(upper_fix_ids) & df['upper_specification_limit'].isnull(), 'upper_specification_limit'] = 0

# Get rows with missing lower or upper limits
missing_limits = df[df['lower_specification_limit'].isnull() | df['upper_specification_limit'].isnull()]

# Unique operation_id values where limits are missing
unique_ops_with_missing_limits = missing_limits['operation_id'].dropna().unique()
print("Unique operation_id values with missing limits after setting to 0:", unique_ops_with_missing_limits, "\n")

# 2.4. Delete rows where 'operation_id' is missing
print("2.4. Delete rows where 'operation_id' is missing \n")
# Get rows where operation_id is missing
missing_op_rows = df[df['operation_id'].isnull()]

# Get unique (lower, upper) limit combinations from these rows
missing_limit_pairs = missing_op_rows[['lower_specification_limit', 'upper_specification_limit']].drop_duplicates()

# For each limit pair, check for a unique matching operation_id and assign if exactly one match
for _, row in missing_limit_pairs.iterrows():
    lower = row['lower_specification_limit']
    upper = row['upper_specification_limit']

    matching_ops = df[
        (df['operation_id'].notnull()) &
        (df['lower_specification_limit'] == lower) &
        (df['upper_specification_limit'] == upper)
        ]['operation_id'].unique()

    if len(matching_ops) == 1:
        df.loc[
            (df['operation_id'].isnull()) &
            (df['lower_specification_limit'] == lower) &
            (df['upper_specification_limit'] == upper),
            'operation_id'
        ] = matching_ops[0]

df = df[df['operation_id'].notnull()]

# 3. Check negative values
print("3. Check negative values for certain measurement units and set to positive")
# Normalize 'unit' column values
df['unit'] = df['unit'].replace({
    'A': 'Ampere',
    'degreesC': 'DegreesC'
})

# Filter relevant rows
bar_high_limit_ops = df[(df['unit'] == 'Bar') & (df['upper_specification_limit'] > 100)]

# Count occurrences per operation_id
operation_id_counts = bar_high_limit_ops['operation_id'].value_counts()

# Print the result
print("Count of occurrences where 'Bar' unit and upper_specification_limit > 100 by operation_id:")
print(operation_id_counts)

# Count occurrences per operation_id where lower_specification_limit < 0 for unit 'seconds'
negative_lower_counts = df[
    (df['unit'] == 'seconds') & (df['lower_specification_limit'] < 0)
].groupby('operation_id').size()

print(f"number of negative lower counts seconds: {negative_lower_counts}")


for unit in df['unit'].dropna().unique():
    unit_df = df[df['unit'] == unit]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    unit_df['value'].plot.box()
    plt.title(f'{unit} - Value')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    unit_df['lower_specification_limit'].plot.box()
    plt.title(f'{unit} - Lower Limit')
    plt.ylabel('Lower Spec Limit')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    unit_df['upper_specification_limit'].plot.box()
    plt.title(f'{unit} - Upper Limit')
    plt.ylabel('Upper Spec Limit')
    plt.grid(True)

    plt.suptitle(f'Boxplots for Unit: {unit}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

# Units to check
units = ['Ampere', 'Bar', 'COP', 'Gram', 'Pa', 'RPM', 'seconds']

# Count negative values per unit
negative_value_counts = {
    unit: df[(df['unit'] == unit) & (df['value'] < 0)].shape[0]
    for unit in units
}

# Convert to DataFrame for display
negative_counts_df = pd.DataFrame(list(negative_value_counts.items()), columns=['unit', 'negative_value_count'])
print(negative_counts_df, "\n")

# Units for which to convert negative 'value' entries to positive
units_to_convert = ['Bar', 'COP', 'Gram', 'Pa']

# Apply absolute value for those units
df.loc[df['unit'].isin(units_to_convert), 'value'] = df.loc[df['unit'].isin(units_to_convert), 'value'].abs()

# Set lower_specification_limit to 0 if it's negative for specific units
units_to_adjust_lower = ['Gram', 'COP']
df.loc[(df['unit'].isin(units_to_adjust_lower)) & (df['lower_specification_limit'] < 0), 'lower_specification_limit'] = 0

# 3. Create test_passed column
print("3. Create test_passed column")
# Swap lower and upper limits where lower > upper
condition = df['lower_specification_limit'] > df['upper_specification_limit']
df.loc[condition, ['lower_specification_limit', 'upper_specification_limit']] = df.loc[condition, ['upper_specification_limit', 'lower_specification_limit']].values

# Create 'test_passed' column: 1 if value is within limits, 0 otherwise
df['test_passed'] = ((df['value'] >= df['lower_specification_limit']) & (df['value'] <= df['upper_specification_limit'])).astype(int)
print(df[['timestamp', 'value', 'test_passed']].head(), "\n")

# 4. Drop 'timestamp'  columns
print("4. Drop 'timestamp' columns \n")
df = df.drop(columns=['timestamp'])

df.to_csv('test_measurements_cleaned.csv', index=False)


