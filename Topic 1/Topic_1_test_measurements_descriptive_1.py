import pandas as pd

# Load the test measurements CSV from the local project folder
df = pd.read_csv('test_measurements_cleaned.csv', low_memory=False)

# Group by 'operation_id' and count total and passed tests
total_tests = df.groupby('operation_id').size()
passed_tests = df.groupby('operation_id')['test_passed'].sum()

# Combine into a DataFrame
operation_test_summary = pd.DataFrame({
    'total_tests': total_tests,
    'passed_tests': passed_tests
})

# Calculate percentage of passed tests
operation_test_summary['passed_percentage'] = (operation_test_summary['passed_tests'] / operation_test_summary['total_tests']) * 100

# Total number of operation_ids
total_operations = len(operation_test_summary)
total_tests_all = operation_test_summary['total_tests'].sum()

# Operation_ids with 0% passed
zero_percent = operation_test_summary[operation_test_summary['passed_percentage'] == 0]
num_zero = len(zero_percent)
tests_zero = zero_percent['total_tests'].sum()

# Operation_ids with 100% passed
hundred_percent = operation_test_summary[operation_test_summary['passed_percentage'] == 100]
num_hundred = len(hundred_percent)
tests_hundred = hundred_percent['total_tests'].sum()

# Print results
print(f"Total operation_ids: {total_operations} (total tests: {total_tests_all}, mean number of tests: {operation_test_summary['total_tests'].mean()})")
print(f"Operation_ids with 0% passed: {num_zero} (total tests: {tests_zero}, mean number of tests: {zero_percent['total_tests'].mean()})")
print(f"Operation_ids with 100% passed: {num_hundred} (total tests: {tests_hundred}, mean number of tests: {hundred_percent['total_tests'].mean()})")

# Calculate the average passing rate across all operation_ids
average_pass_rate = operation_test_summary['passed_percentage'].mean()

# Print the average
print(f"Average passing rate: {average_pass_rate:.2f}% \n")

# Group by 'dut_sn' to get total and passed test counts
total_tests_dut = df.groupby('dut_sn').size()
passed_tests_dut = df.groupby('dut_sn')['test_passed'].sum()

# Combine into a DataFrame
dut_test_summary = pd.DataFrame({
    'total_tests': total_tests_dut,
    'passed_tests': passed_tests_dut
})

# Calculate passed percentage
dut_test_summary['passed_percentage'] = (dut_test_summary['passed_tests'] / dut_test_summary['total_tests']) * 100

# Calculate totals
total_duts = len(dut_test_summary)
total_tests_all_dut = dut_test_summary['total_tests'].sum()

# DUTs with 0% passed
zero_percent_dut = dut_test_summary[dut_test_summary['passed_percentage'] == 0]
num_zero_dut = len(zero_percent_dut)
tests_zero_dut = zero_percent_dut['total_tests'].sum()

# DUTs with 100% passed
hundred_percent_dut = dut_test_summary[dut_test_summary['passed_percentage'] == 100]
num_hundred_dut = len(hundred_percent_dut)
tests_hundred_dut = hundred_percent_dut['total_tests'].sum()

# Average passing rate
average_pass_rate_dut = dut_test_summary['passed_percentage'].mean()

# Print results
print(f"Total dut_sn: {total_duts} (total tests: {total_tests_all_dut}, mean number of tests: {dut_test_summary['total_tests'].mean()})")
print(f"dut_sn with 0% passed: {num_zero_dut} (total tests: {tests_zero_dut}, mean number of tests: {zero_percent_dut['total_tests'].mean()})")
print(f"dut_sn with 100% passed: {num_hundred_dut} (total tests: {tests_hundred_dut}, mean number of tests: {hundred_percent_dut['total_tests'].mean()})")
print(f"Average passing rate: {average_pass_rate_dut:.2f}%")
# DUTs with 0-5% passed
low_pass_dut = dut_test_summary[(dut_test_summary['passed_percentage'] >= 0) & (dut_test_summary['passed_percentage'] <= 10)]
num_low_pass_dut = len(low_pass_dut)
tests_low_pass_dut = low_pass_dut['total_tests'].sum()

# DUTs with 95-100% passed
high_pass_dut = dut_test_summary[(dut_test_summary['passed_percentage'] >= 90) & (dut_test_summary['passed_percentage'] <= 100)]
num_high_pass_dut = len(high_pass_dut)
tests_high_pass_dut = high_pass_dut['total_tests'].sum()

# Print results
print(f"dut_sn with 0-10% passed: {num_low_pass_dut} (total tests: {tests_low_pass_dut}, mean number of tests: {low_pass_dut['total_tests'].mean()})")
print(f"dut_sn with 90-100% passed: {num_high_pass_dut} (total tests: {tests_high_pass_dut}, mean number of tests: {high_pass_dut['total_tests'].mean()})\n")



print("Filter out operation_ids with 0% or 100% passed tests")
# Filter out operation_ids with 0% or 100% passed tests
exclude_ops = operation_test_summary[
    (operation_test_summary['passed_percentage'] == 0) |
    (operation_test_summary['passed_percentage'] == 100)
].index

# Filter dataframe to only include tests with operation_ids not in exclude list
filtered_df = df[~df['operation_id'].isin(exclude_ops)]

# Group by 'dut_sn' on filtered data
total_tests_dut_filtered = filtered_df.groupby('dut_sn').size()
passed_tests_dut_filtered = filtered_df.groupby('dut_sn')['test_passed'].sum()

# Combine into summary DataFrame
dut_test_summary_filtered = pd.DataFrame({
    'total_tests': total_tests_dut_filtered,
    'passed_tests': passed_tests_dut_filtered
})

# Calculate passed percentage
dut_test_summary_filtered['passed_percentage'] = (
    dut_test_summary_filtered['passed_tests'] / dut_test_summary_filtered['total_tests'] * 100
)

# Calculate metrics
total_duts_f = len(dut_test_summary_filtered)
total_tests_all_dut_f = dut_test_summary_filtered['total_tests'].sum()

zero_percent_dut_f = dut_test_summary_filtered[dut_test_summary_filtered['passed_percentage'] == 0]
num_zero_dut_f = len(zero_percent_dut_f)
tests_zero_dut_f = zero_percent_dut_f['total_tests'].sum()

hundred_percent_dut_f = dut_test_summary_filtered[dut_test_summary_filtered['passed_percentage'] == 100]
num_hundred_dut_f = len(hundred_percent_dut_f)
tests_hundred_dut_f = hundred_percent_dut_f['total_tests'].sum()

average_pass_rate_dut_f = dut_test_summary_filtered['passed_percentage'].mean()

# Print results
print(f"Filtered - Total dut_sn: {total_duts_f} (total tests: {total_tests_all_dut_f}, mean number of tests: {dut_test_summary_filtered['total_tests'].mean()})")
print(f"Filtered - dut_sn with 0% passed: {num_zero_dut_f} (total tests: {tests_zero_dut_f}, mean number of tests: {zero_percent_dut_f['total_tests'].mean()})")
print(f"Filtered - dut_sn with 100% passed: {num_hundred_dut_f} (total tests: {tests_hundred_dut_f}, mean number of tests: {hundred_percent_dut_f['total_tests'].mean()})")
print(f"Filtered - Average passing rate: {average_pass_rate_dut_f:.2f}%")
# DUTs with 0-5% passed
low_pass_dut = dut_test_summary_filtered[(dut_test_summary_filtered['passed_percentage'] >= 0) & (dut_test_summary_filtered['passed_percentage'] <= 10)]
num_low_pass_dut = len(low_pass_dut)
tests_low_pass_dut = low_pass_dut['total_tests'].sum()

# DUTs with 95-100% passed
high_pass_dut = dut_test_summary_filtered[(dut_test_summary_filtered['passed_percentage'] >= 90) & (dut_test_summary_filtered['passed_percentage'] <= 100)]
num_high_pass_dut = len(high_pass_dut)
tests_high_pass_dut = high_pass_dut['total_tests'].sum()

# Print results
print(f"Filtered - dut_sn with 0-10% passed: {num_low_pass_dut} (total tests: {tests_low_pass_dut}, mean number of tests: {low_pass_dut['total_tests'].mean()})")
print(f"Filtered - dut_sn with 90-100% passed: {num_high_pass_dut} (total tests: {tests_high_pass_dut}, mean number of tests: {high_pass_dut['total_tests'].mean()})\n")

# Group by 'instrument_id' to get total and passed test counts
total_tests_instr = df.groupby('instrument_id').size()
passed_tests_instr = df.groupby('instrument_id')['test_passed'].sum()

# Combine into a DataFrame
instr_test_summary = pd.DataFrame({
    'total_tests': total_tests_instr,
    'passed_tests': passed_tests_instr
})

# Calculate passed percentage
instr_test_summary['passed_percentage'] = (
    instr_test_summary['passed_tests'] / instr_test_summary['total_tests'] * 100
)

# Total counts
total_instr = len(instr_test_summary)
total_tests_all_instr = instr_test_summary['total_tests'].sum()

# Instruments with 0% passed
zero_percent_instr = instr_test_summary[instr_test_summary['passed_percentage'] == 0]
num_zero_instr = len(zero_percent_instr)
tests_zero_instr = zero_percent_instr['total_tests'].sum()

# Instruments with 100% passed
hundred_percent_instr = instr_test_summary[instr_test_summary['passed_percentage'] == 100]
num_hundred_instr = len(hundred_percent_instr)
tests_hundred_instr = hundred_percent_instr['total_tests'].sum()

# Average pass rate
average_pass_rate_instr = instr_test_summary['passed_percentage'].mean()

# Instruments with 0-5% passed
low_pass_instr = instr_test_summary[(instr_test_summary['passed_percentage'] >= 0) & (instr_test_summary['passed_percentage'] <= 10)]
num_low_pass_instr = len(low_pass_instr)
tests_low_pass_instr = low_pass_instr['total_tests'].sum()

# Instruments with 95-100% passed
high_pass_instr = instr_test_summary[(instr_test_summary['passed_percentage'] >= 90) & (instr_test_summary['passed_percentage'] <= 100)]
num_high_pass_instr = len(high_pass_instr)
tests_high_pass_instr = high_pass_instr['total_tests'].sum()

# Filter out operation_ids with 0% or 100% passed tests
filtered_df_instr = df[~df['operation_id'].isin(exclude_ops)]

# Group filtered data by instrument_id
total_tests_instr_f = filtered_df_instr.groupby('instrument_id').size()
passed_tests_instr_f = filtered_df_instr.groupby('instrument_id')['test_passed'].sum()

# Summary DataFrame
instr_test_summary_f = pd.DataFrame({
    'total_tests': total_tests_instr_f,
    'passed_tests': passed_tests_instr_f
})
instr_test_summary_f['passed_percentage'] = (
    instr_test_summary_f['passed_tests'] / instr_test_summary_f['total_tests'] * 100
)

# Filtered stats
total_instr_f = len(instr_test_summary_f)
total_tests_all_instr_f = instr_test_summary_f['total_tests'].sum()

zero_percent_instr_f = instr_test_summary_f[instr_test_summary_f['passed_percentage'] == 0]
num_zero_instr_f = len(zero_percent_instr_f)
tests_zero_instr_f = zero_percent_instr_f['total_tests'].sum()

hundred_percent_instr_f = instr_test_summary_f[instr_test_summary_f['passed_percentage'] == 100]
num_hundred_instr_f = len(hundred_percent_instr_f)
tests_hundred_instr_f = hundred_percent_instr_f['total_tests'].sum()

average_pass_rate_instr_f = instr_test_summary_f['passed_percentage'].mean()

low_pass_instr_f = instr_test_summary_f[(instr_test_summary_f['passed_percentage'] >= 0) & (instr_test_summary_f['passed_percentage'] <= 10)]
num_low_pass_instr_f = len(low_pass_instr_f)
tests_low_pass_instr_f = low_pass_instr_f['total_tests'].sum()

high_pass_instr_f = instr_test_summary_f[(instr_test_summary_f['passed_percentage'] >= 90) & (instr_test_summary_f['passed_percentage'] <= 100)]
num_high_pass_instr_f = len(high_pass_instr_f)
tests_high_pass_instr_f = high_pass_instr_f['total_tests'].sum()

# Print instrument-level test summary results
print(f"Total instrument_id: {total_instr} (total tests: {total_tests_all_instr}, mean number of tests: {instr_test_summary['total_tests'].mean()})")
print(f"instrument_id with 0% passed: {num_zero_instr} (total tests: {tests_zero_instr}, mean number of tests: {zero_percent_instr['total_tests'].mean()})")
print(f"instrument_id with 100% passed: {num_hundred_instr} (total tests: {tests_hundred_instr}, mean number of tests: {hundred_percent_instr['total_tests'].mean()})")
print(f"Average passing rate: {average_pass_rate_instr:.2f}%")
print(f"instrument_id with 0-10% passed: {num_low_pass_instr} (total tests: {tests_low_pass_instr}, mean number of tests: {low_pass_instr['total_tests'].mean()})")
print(f"instrument_id with 90-100% passed: {num_high_pass_instr} (total tests: {tests_high_pass_instr}, mean number of tests: {high_pass_instr['total_tests'].mean()})\n")
# Print filtered instrument-level test summary results
print("Filtered (excluding operation_ids with 0% or 100% passed):")
print(f"Total instrument_id: {total_instr_f} (total tests: {total_tests_all_instr_f}, mean number of tests: {instr_test_summary_f['total_tests'].mean()})")
print(f"instrument_id with 0% passed: {num_zero_instr_f} (total tests: {tests_zero_instr_f}, mean number of tests: {zero_percent_instr_f['total_tests'].mean()})")
print(f"instrument_id with 100% passed: {num_hundred_instr_f} (total tests: {tests_hundred_instr_f}, mean number of tests: {hundred_percent_instr_f['total_tests'].mean()})")
print(f"Average passing rate: {average_pass_rate_instr_f:.2f}%")
print(f"instrument_id with 0-10% passed: {num_low_pass_instr_f} (total tests: {tests_low_pass_instr_f}, mean number of tests: {low_pass_instr_f['total_tests'].mean()})")
print(f"instrument_id with 90-100% passed: {num_high_pass_instr_f} (total tests: {tests_high_pass_instr_f}, mean number of tests: {high_pass_instr_f['total_tests'].mean()})\n")

# Group by dut_sn and operation_id, count how many times each combination appears
counts = df.groupby(['dut_sn', 'operation_id']).size().reset_index(name='count')

# Filter for combinations with more than one occurrence
repeated = counts[counts['count'] > 1]

# Merge to get the full rows for repeated operation_ids
merged = pd.merge(df, repeated[['dut_sn', 'operation_id']], on=['dut_sn', 'operation_id'])

# Group by dut_sn to count repeated tests and passed tests
summary = merged.groupby('dut_sn').agg(
    repeated_tests=('test_passed', 'count'),
    passed_tests=('test_passed', 'sum')
).reset_index()

# Calculate percentage passed
summary['passed_percentage'] = (summary['passed_tests'] / summary['repeated_tests']) * 100

# Get passing rates for each dut_sn (based on all tests, not just repeated ones)
overall_pass_rate = df.groupby('dut_sn')['test_passed'].mean().reset_index(name='pass_rate')

# Filter to only include dut_sns that had repeated operation_ids
repeated_dut_sns = summary['dut_sn'].unique()
filtered_pass_rate = overall_pass_rate[overall_pass_rate['dut_sn'].isin(repeated_dut_sns)]

# Calculate mean passing rate of these dut_sns
mean_pass_rate = filtered_pass_rate['pass_rate'].mean()

# Print the mean pass rate
print(f"Mean pass rate for dut_sns with repeated operation_ids: {mean_pass_rate:.2%}\n")

# Count how many times each (dut_sn, operation_id) pair appears
op_counts = df.groupby(['dut_sn', 'operation_id']).size().reset_index(name='count')

# Filter to only repeated operation_ids per dut_sn
repeated_ops = op_counts[op_counts['count'] > 1]

# Count how often each operation_id gets repeated across different dut_sns
repeated_op_counts = repeated_ops['operation_id'].value_counts()

# Total number of tests per operation_id
total_tests_per_op = df['operation_id'].value_counts()

# Merge with repeated operation_id counts
repeated_op_df = repeated_op_counts.reset_index()
repeated_op_df.columns = ['operation_id', 'repeated_dut_count']
repeated_op_df['total_tests'] = repeated_op_df['operation_id'].map(total_tests_per_op)
repeated_op_df['percentage_repeated'] = (repeated_op_df['repeated_dut_count'] / repeated_op_df['total_tests']) * 100

# Sort and show top 5
top5_repeated_ops = repeated_op_df.sort_values(by='percentage_repeated', ascending=False).head(5)
print(top5_repeated_ops)


# Count how many times each operation_id is performed per dut_sn
counts_per_dut_op = df.groupby(['dut_sn', 'operation_id']).size().reset_index(name='count')

# Filter only those with more than 1 occurrence (repeated)
repeated_tests = counts_per_dut_op[counts_per_dut_op['count'] > 1]

# Total number of repeated tests (each repeated pair counts as 1)
total_repeated_tests = repeated_tests.shape[0]
print("Total number of repeated tests:", total_repeated_tests)
# Total number of all tests
total_tests = df.shape[0]

# Percentage of repeated tests
percentage_repeated = (total_repeated_tests / total_tests) * 100

print(f"Total number of tests: {total_tests}")
print(f"Percentage of repeated tests: {percentage_repeated:.2f}%")

# Calculate tolerance
df['tolerance'] = df['upper_specification_limit'] - df['lower_specification_limit']
df['lower_margin'] = df['lower_specification_limit'] + 0.05 * df['tolerance']
df['upper_margin'] = df['upper_specification_limit'] - 0.05 * df['tolerance']

# Check if values are within 5% of limits
df['within_5_percent'] = (
    ((df['value'] >= df['lower_specification_limit']) & (df['value'] <= df['lower_margin'])) |
    ((df['value'] <= df['upper_specification_limit']) & (df['value'] >= df['upper_margin']))
) & (df['test_passed'] == 1)

# Calculate percentage for each operation_id (over ALL operation_ids, not just top 5)
within_range_percent_all = df.groupby('operation_id')['within_5_percent'].sum() / df.groupby('operation_id')['test_passed'].sum() * 100

# Drop NaNs
within_range_percent_all = within_range_percent_all.dropna()

# Show top 5 for display
print(within_range_percent_all.sort_values(ascending=False).head(5).reset_index(name='percent_within_5_percent'))

# Mean over all operation_ids
mean_within_range_percent_all = within_range_percent_all.mean()
print(f"Mean percentage within 5% limits over all operation_ids: {mean_within_range_percent_all:.2f}%")


# Total passed tests across all operation_ids
total_passed_tests = df[df['test_passed'] == 1].shape[0]

# Total number of passed tests within 5% margin across all operation_ids
within_margin_total = df[
    (df['test_passed'] == 1) &
    (
        ((df['value'] - df['lower_specification_limit']) <= 0.05 * (df['upper_specification_limit'] - df['lower_specification_limit'])) |
        ((df['upper_specification_limit'] - df['value']) <= 0.05 * (df['upper_specification_limit'] - df['lower_specification_limit']))
    )
].shape[0]

# Compute overall percentage
overall_percent_within_margin = (within_margin_total / total_passed_tests) * 100
print(f"Percentage of all passed tests within 5% of limits: {overall_percent_within_margin:.2f}%")

# Calculate percentage of tests within 5% tolerance per dut_sn
within_range_percent_dut = df.groupby('dut_sn')['within_5_percent'].sum() / df.groupby('dut_sn')['test_passed'].sum() * 100
within_range_percent_dut = within_range_percent_dut.dropna()

# Top 5 dut_sn with highest percentage
top_5_dut = within_range_percent_dut.sort_values(ascending=False).head(5).reset_index(name='percent_within_5_percent')
print(top_5_dut)

# Mean percentage over all dut_sn
mean_within_range_percent_dut = within_range_percent_dut.mean()
print(f"Mean percentage within 5% limits over all dut_sn: {mean_within_range_percent_dut:.2f}%")

# Overall percentage of all passed tests within 5% margin
total_passed_tests = df[df['test_passed'] == 1].shape[0]
within_margin_total_dut = df[(df['test_passed'] == 1) & df['within_5_percent']].shape[0]
overall_percent_within_margin_dut = (within_margin_total_dut / total_passed_tests) * 100
print(f"Percentage of all passed tests within 5% of limits: {overall_percent_within_margin_dut:.2f}%")

average_passing_rate_all = df['test_passed'].mean() * 100
print(f"Average passing rate over all tests: {average_passing_rate_all:.2f}%")
