import pandas as pd

# Load the test measurements CSV from the local project folder
df = pd.read_csv('Dataset3-test_measurements.csv', low_memory=False)

print("=== TEST MEASUREMENTS DATASET ANALYSIS ===\n")

# 1. Shape
print(f"1. Dataset shape: {df.shape[0]} rows, {df.shape[1]} columns\n")

# 2. Missing values
print("2. Missing values per column:")
print(df.isnull().sum(), "\n")

# 3. Duplicate rows
dupes = df.duplicated().sum()
print(f"3. Full duplicate rows: {dupes}\n")

# 4. Unique value counts
print("4. Unique counts per column:")
for col in df.columns:
    print(f"   - {col}: {df[col].nunique()} unique values")
print()

# 5. 'value' column summary
print("5. 'value' column statistics:")
print(df['value'].describe(), "\n")

# 6. Specification limits summary
print("6. Specification limits statistics:")
print(df[['lower_specification_limit', 'upper_specification_limit']].describe(), "\n")

# 7. Out-of-spec values
valid_limits = df.dropna(subset=['value', 'lower_specification_limit', 'upper_specification_limit'])
out_of_spec = valid_limits[
    (valid_limits['value'] < valid_limits['lower_specification_limit']) |
    (valid_limits['value'] > valid_limits['upper_specification_limit'])
]
print(f"7. Out-of-spec measurements: {len(out_of_spec)} rows out of {len(valid_limits)} with limits")
print("   Example out-of-spec rows:")
print(out_of_spec[['timestamp','operation_id','value','lower_specification_limit','upper_specification_limit']].head(), "\n")

# 8. Check 'uncertainty' field
print("8. 'uncertainty' field:")
print(f"   Non-null count: {df['uncertainty'].notnull().sum()} ({df['uncertainty'].notnull().mean()*100:.2f}%)")
print(f"   Unique values: {df['uncertainty'].dropna().unique()}\n")

# 9. Units distribution
print("9. Top 10 units:")
print(df['unit'].value_counts().head(10), "\n")

# 10. Operation type distribution
print("10. Operation types:")
print(df['operation_type'].value_counts(), "\n")

# 11. Method distribution
print("11. Method usage (top 10):")
print(df['method'].value_counts(dropna=False).head(10), "\n")

# 12. Instrument ID overview
print("12. Instrument IDs:")
print(f"   Unique instruments: {df['instrument_id'].nunique()}")
print("   Top 5 instruments:")
print(df['instrument_id'].value_counts().head(5), "\n")

# 13. Timestamp format examples
print("13. Timestamp samples:")
print(df['timestamp'].dropna().unique()[:5], "\n")

print("=== END OF ANALYSIS ===")

# 14. Drop 'uncertainty' and 'method' columns
print("14. Drop 'uncertainty' and 'method' columns \n")
df = df.drop(columns=['uncertainty', 'method'])

# 15. Drop rows where 'value' is missing
print("15. value column rows with missing values")
missing_value_rows = df[df['value'].isnull()]
print(missing_value_rows[['value', 'lower_specification_limit', 'upper_specification_limit']], "\n")
df = df.dropna(subset=['value'])

# 16. Check negative values
print("16. Check negative values")
# Units to check
units = ['Ampere', 'Bar', 'COP', 'Gram', 'Pa', 'RPM', 'seconds']

# Count negative values per unit
negative_value_counts = {
    unit: df[(df['unit'] == unit) & (df['value'] < 0)].shape[0]
    for unit in units
}

# Convert to DataFrame for display
negative_counts_df = pd.DataFrame(list(negative_value_counts.items()), columns=['unit', 'negative_value_count'])
print(negative_counts_df)

# Units for which to convert negative 'value' entries to positive
units_to_convert = ['Bar', 'COP', 'Gram', 'Pa']

# Apply absolute value for those units
df.loc[df['unit'].isin(units_to_convert), 'value'] = df.loc[df['unit'].isin(units_to_convert), 'value'].abs()

# 16. Fill in/drop missing limits
print("16. Fill in/drop missing limits")
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

# 17. Create test_passed column
print("17. Create test_passed column \n")
# Swap lower and upper limits where lower > upper
condition = df['lower_specification_limit'] > df['upper_specification_limit']
df.loc[condition, ['lower_specification_limit', 'upper_specification_limit']] = df.loc[condition, ['upper_specification_limit', 'lower_specification_limit']].values

# Create 'test_passed' column: 1 if value is within limits, 0 otherwise
df['test_passed'] = ((df['value'] >= df['lower_specification_limit']) & (df['value'] <= df['upper_specification_limit'])).astype(int)
print(df[['timestamp', 'value', 'test_passed']].head(), "\n")

# 18. Delete rows where 'operation_id' is missing
print("18. Delete rows where 'operation_id' is missing \n")
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

# 14. Drop 'timestamp' and 'operation_type' columns
print("14. Drop 'timestamp' and 'operation_type' columns \n")
df = df.drop(columns=['timestamp', 'operation_type'])

df.to_csv('test_measurements_cleaned.csv', index=False)

# 19. Group by 'operation_id' and count 'test_passed' values
print("19. Group by 'operation_id' and count 'test_passed' values")
operation_group = df.groupby('operation_id')['test_passed'].count()
print("Test count per operation_id:\n", operation_group, "\n")

# 20. Group by 'dut_sn' and count 'test_passed' values
print("20. Group by 'dut_sn' and count 'test_passed' values")
dut_group = df.groupby('dut_sn')['test_passed'].count()
print("Test count per dut_sn:\n", dut_group, "\n")
