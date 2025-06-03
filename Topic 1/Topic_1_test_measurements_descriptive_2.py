import pandas as pd

# Load the test measurements CSV from the local project folder
df = pd.read_csv('test_measurements_cleaned.csv', low_memory=False)

# Group by operation_id and calculate total tests and passed tests
operation_stats = df.groupby('operation_id').agg(
    total_tests=('test_passed', 'count'),
    passed_tests=('test_passed', 'sum')
).reset_index()

# Calculate passed percentage
operation_stats['passed_percent'] = (operation_stats['passed_tests'] / operation_stats['total_tests']) * 100

# Define bins and labels
bins = [-0.01, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99.99, 100]
labels = ['0%', '>0-10%', '10-20%', '20-30%', '30-40%', '40-50%', '50-60%',
          '60-70%', '70-80%', '80-90%', '90-<100%', '100%']

# Categorize the passed_percent into bins
operation_stats['passed_percent_range'] = pd.cut(operation_stats['passed_percent'], bins=bins, labels=labels, include_lowest=True)

# Count operation_ids in each range
range_counts = operation_stats['passed_percent_range'].value_counts().sort_index().reset_index()
range_counts.columns = ['passed_percent_range', 'operation_id_count']

print(range_counts)

# Step 1: Total tests and passed tests per operation_type
tests_summary = df.groupby('operation_type').agg(
    total_tests=('test_passed', 'count'),
    passed_tests=('test_passed', 'sum')
).reset_index()

# Step 2: Total operation_ids per operation_type
total_ops = df.groupby('operation_type')['operation_id'].nunique().reset_index(name='total_operation_ids')

# Step 3: Operation_ids with 0% passing rate
zero_pass_ops = operation_stats[operation_stats['passed_percent'] == 0]['operation_id']
df_zero_pass = df[df['operation_id'].isin(zero_pass_ops)]

# Step 4: Number of 0% operation_ids per operation_type
zero_ops = df_zero_pass.groupby('operation_type')['operation_id'].nunique().reset_index(name='zero_pass_operation_ids')

# Step 5: Count of tests from 0% passing operation_ids per operation_type
tests_in_zero_ops = df_zero_pass.groupby('operation_type').size().reset_index(name='tests_in_zero_pass_ops')

# Step 6: Measurement units for 0% operation_ids
units_zero_ops = df_zero_pass.groupby('operation_type')['unit'].unique().reset_index()
units_zero_ops['unit'] = units_zero_ops['unit'].apply(lambda x: ', '.join(sorted(set(x))))
units_zero_ops.rename(columns={'unit': 'measurement_units_0%'}, inplace=True)

# Step 7: Merge all summaries
summary = tests_summary.merge(total_ops, on='operation_type', how='left')
summary = summary.merge(zero_ops, on='operation_type', how='left').fillna({'zero_pass_operation_ids': 0})
summary = summary.merge(tests_in_zero_ops, on='operation_type', how='left').fillna({'tests_in_zero_pass_ops': 0})
summary = summary.merge(units_zero_ops, on='operation_type', how='left')

# Step 8: Calculate passing rates
summary['passing_rate'] = (summary['passed_tests'] / summary['total_tests']) * 100
summary['tests_zero_ops_passing_rate'] = (summary['tests_in_zero_pass_ops'] / summary['total_tests']) * 100
summary['operation_ids_zero_pass_percent'] = (summary['zero_pass_operation_ids'] / summary['total_operation_ids']) * 100

# Step 9: Reorder and rename columns
final_summary = summary[[
    'operation_type',
    'total_tests',
    'passed_tests',
    'total_operation_ids',
    'zero_pass_operation_ids',
    'passing_rate',
    'tests_zero_ops_passing_rate',
    'operation_ids_zero_pass_percent',
    'measurement_units_0%'
]]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(final_summary)


import matplotlib.pyplot as plt

# Filter operation_ids with 0% passing rate
zero_pass_ops = operation_stats[operation_stats['passed_percent'] == 0]['operation_id']
df_zero_pass = df[df['operation_id'].isin(zero_pass_ops)]

# Plot boxplots with average lower and upper specification limits
for op_id in df_zero_pass['operation_id'].unique():
    op_df = df_zero_pass[df_zero_pass['operation_id'] == op_id]

    avg_lower = op_df['lower_specification_limit'].mean()
    avg_upper = op_df['upper_specification_limit'].mean()

    plt.figure(figsize=(8, 6))
    plt.boxplot(op_df['value'].dropna(), vert=True)
    plt.axhline(avg_lower, color='red', linestyle='--', label='Avg Lower Spec Limit')
    plt.axhline(avg_upper, color='green', linestyle='--', label='Avg Upper Spec Limit')
    plt.title(f'Boxplot of Values for Operation ID: {op_id}')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    plt.show()
