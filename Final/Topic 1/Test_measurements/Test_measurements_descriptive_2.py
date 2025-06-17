import pandas as pd

# Load datasets
heatpumps_df = pd.read_csv('heatpumps_cleaned.csv')
tests_df = pd.read_csv('test_measurements_cleaned.csv')

# Clean identifiers
tests_df['dut_sn'] = tests_df['dut_sn'].astype(str).str.strip()
heatpumps_df['SerialNumber'] = heatpumps_df['SerialNumber'].astype(str).str.strip()

# Filter broken heatpumps
broken_heatpumps = heatpumps_df[heatpumps_df['State'] == 5]
broken_serials = broken_heatpumps['SerialNumber'].unique()

# Filter tests for broken heatpumps
tests_broken = tests_df[tests_df['dut_sn'].isin(broken_serials)]

# Group and calculate failed/total tests for broken heatpumps
failed_tests_broken = tests_broken[tests_broken['test_passed'] == 0].groupby('operation_id').size().rename('failed_tests_broken')
total_tests_broken = tests_broken.groupby('operation_id').size().rename('total_tests_broken')
percentage_failed_broken = (failed_tests_broken / total_tests_broken * 100).rename('percentage_failed_broken')

# Group and calculate failed/total tests for all heatpumps
failed_tests_all = tests_df[tests_df['test_passed'] == 0].groupby('operation_id').size().rename('failed_tests_all')
total_tests_all = tests_df.groupby('operation_id').size().rename('total_tests_all')
percentage_failed_all = (failed_tests_all / total_tests_all * 100).rename('percentage_failed_all')

# Combine into one table
result_df = pd.concat([
    failed_tests_broken,
    total_tests_broken,
    percentage_failed_broken,
    failed_tests_all,
    total_tests_all,
    percentage_failed_all
], axis=1).fillna(0)

# Add difference column
result_df['percentage_difference'] = result_df['percentage_failed_broken'] - result_df['percentage_failed_all']

# Sort by difference descending
result_df = result_df.sort_values(by='percentage_difference', ascending=False)

# Display settings
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 200)
pd.set_option('display.colheader_justify', 'left')
pd.set_option('display.expand_frame_repr', False)

print(result_df)
