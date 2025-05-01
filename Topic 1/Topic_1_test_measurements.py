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
