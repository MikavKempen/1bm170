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

# Print result
print(operation_test_summary[['total_tests', 'passed_tests', 'passed_percentage']])

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
