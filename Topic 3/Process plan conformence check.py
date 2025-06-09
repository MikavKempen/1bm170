import pandas as pd
import matplotlib.pyplot as plt

# Define color map and category order (removed 'Incorrect')
category_color_map = {
    'Complete no duplicates': 'green',
    'Complete with duplicates': 'orange',
    'Incomplete': 'red'
}
ordered_columns = list(category_color_map.keys())

# Load the datasets
heatpumps = pd.read_csv('heatpumps_cleaned_v2.csv')
operations = pd.read_csv('mes_operations_tagged.csv')
process_plans = pd.read_csv('process_plans_cleaned.csv')

# -------------------- Broken Heat Pumps Section --------------------

# Filter for broken heat pumps only (state = 5)
broken_heatpumps = heatpumps[heatpumps['State'] == 5]
print(f"Number of broken heatpumps: {len(broken_heatpumps)}")
broken_serials = set(broken_heatpumps['SerialNumber'])

# Filter MES operations for only broken heat pumps
broken_operations = operations[operations['SerialNumber'].isin(broken_serials)]

# Get expected operations per process plan
expected_ops = (
    process_plans
    .groupby('ProcessPlan')['OperationID']
    .apply(list)
    .to_dict()
)

# Get actual operations per serial number
actual_ops = (
    broken_operations
    .groupby(['SerialNumber', 'ProcessPlan'])['ManufacturingOrderOperation']
    .apply(list)
    .reset_index()
)

# Classification function (removed 'Incorrect')
def classify_process(actual, expected):
    actual_set = set(actual)
    expected_set = set(expected)

    if actual_set == expected_set:
        if len(actual) == len(actual_set):
            return 'Complete no duplicates'
        else:
            return 'Complete with duplicates'
    else:
        return 'Incomplete'

# Map classification to serial numbers
actual_ops['ExpectedOperations'] = actual_ops['ProcessPlan'].map(expected_ops)
actual_ops = actual_ops.dropna(subset=['ExpectedOperations'])

actual_ops['Category'] = actual_ops.apply(
    lambda row: classify_process(row['ManufacturingOrderOperation'], row['ExpectedOperations']),
    axis=1
)

# Count occurrences per process plan and category
category_counts = (
    actual_ops
    .groupby(['ProcessPlan', 'Category'])['SerialNumber']
    .nunique()
    .unstack(fill_value=0)
)
category_counts = category_counts.reindex(columns=ordered_columns, fill_value=0)

# Plot 1: Counts
category_counts.plot(kind='bar', stacked=True, figsize=(14, 6),
                     color=[category_color_map[col] for col in category_counts.columns])
plt.title('Process Performance by Plan (Broken Heat Pumps)')
plt.xlabel('Process Plan')
plt.ylabel('Count')
plt.legend(title='Performance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 2: Percentages
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
category_percentages = category_percentages.reindex(columns=ordered_columns, fill_value=0)
category_percentages.plot(kind='bar', stacked=True, figsize=(14, 6),
                          color=[category_color_map[col] for col in category_percentages.columns])
plt.title('Process Performance Distribution by Plan (%) (Broken Heat Pumps)')
plt.xlabel('Process Plan')
plt.ylabel('Percentage (%)')
plt.legend(title='Performance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# All broken heatpumps from the heatpumps dataset
broken_serials_set = set(heatpumps[heatpumps['State'] == 5]['SerialNumber'])

# All serials that appear in the mes_operations dataset
mes_serials_set = set(operations['SerialNumber'])

# Intersection
broken_with_operations = broken_serials_set.intersection(mes_serials_set)

# Report
print("\nMES Coverage of Broken Heat Pumps:")
print(f"Total broken heat pumps in dataset: {len(broken_serials_set)}")
print(f"Broken heat pumps with MES operations: {len(broken_with_operations)}")
print(f"Coverage: {(len(broken_with_operations) / len(broken_serials_set)) * 100:.2f}%")


# -------------------- All Heat Pumps Section --------------------

# Merge all heat pumps with their operations
merged = pd.merge(
    heatpumps[['SerialNumber']],
    operations,
    on='SerialNumber',
    how='inner'
)

# Keep only valid process plans that are defined
valid_plans = process_plans['ProcessPlan'].unique()
merged = merged[merged['ProcessPlan'].isin(valid_plans)]

# Create a dictionary of expected operations per process plan
expected_ops = (
    process_plans
    .groupby('ProcessPlan')['OperationID']
    .apply(list)
    .to_dict()
)

# Get actual operations per SerialNumber-ProcessPlan combo
actual_ops = (
    merged
    .groupby(['SerialNumber', 'ProcessPlan'])['ManufacturingOrderOperation']
    .apply(list)
    .reset_index()
)

# Drop rows with undefined expected operations
actual_ops['ExpectedOperations'] = actual_ops['ProcessPlan'].map(expected_ops)
actual_ops = actual_ops.dropna(subset=['ExpectedOperations'])

# Classify each SerialNumber
actual_ops['Category'] = actual_ops.apply(
    lambda row: classify_process(row['ManufacturingOrderOperation'], row['ExpectedOperations']),
    axis=1
)

# Count number of serials in each category per process plan
category_counts = (
    actual_ops
    .groupby(['ProcessPlan', 'Category'])['SerialNumber']
    .nunique()
    .unstack(fill_value=0)
)
category_counts = category_counts.reindex(columns=ordered_columns, fill_value=0)

# Plot 1: Count of serials
category_counts.plot(kind='bar', stacked=True, figsize=(14, 6),
                     color=[category_color_map[col] for col in category_counts.columns])
plt.title('Process Performance by Plan (All Heat Pumps)')
plt.xlabel('Process Plan')
plt.ylabel('Count')
plt.legend(title='Performance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Plot 2: Percentage per category
category_percentages = category_counts.div(category_counts.sum(axis=1), axis=0) * 100
category_percentages = category_percentages.reindex(columns=ordered_columns, fill_value=0)
category_percentages.plot(kind='bar', stacked=True, figsize=(14, 6),
                          color=[category_color_map[col] for col in category_percentages.columns])
plt.title('Process Performance Distribution by Plan (%) (All Heat Pumps)')
plt.xlabel('Process Plan')
plt.ylabel('Percentage (%)')
plt.legend(title='Performance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

def prepare_actual_operations(df_serial_ops, expected_ops_dict):
    """Groups operations by SerialNumber and ProcessPlan and maps expected operations."""
    grouped = (
        df_serial_ops
        .groupby(['SerialNumber', 'ProcessPlan'])['ManufacturingOrderOperation']
        .apply(list)
        .reset_index()
    )
    grouped['ExpectedOperations'] = grouped['ProcessPlan'].map(expected_ops_dict)
    grouped = grouped.dropna(subset=['ExpectedOperations'])
    return grouped

def count_missing_operations(df_ops):
    """Counts how many expected operations are missing per serial/process plan."""
    def count_missing(row):
        expected = set(row['ExpectedOperations'])
        actual = set(row['ManufacturingOrderOperation'])
        return len(expected - actual)

    df_ops['MissingCount'] = df_ops.apply(count_missing, axis=1)
    return df_ops['MissingCount'].value_counts().sort_index()

def print_distribution_with_percentages(label, series):
    """Prints missing count distribution with percentages."""
    total = series.sum()
    print(f"\n{label}:")
    for missing_count, count in series.items():
        percentage = (count / total) * 100
        print(f"{missing_count}: {count} ({percentage:.2f}%)")

# --- Analysis for ALL Heat Pumps ---

merged_all = pd.merge(heatpumps[['SerialNumber']], operations, on='SerialNumber', how='inner')
merged_all = merged_all[merged_all['ProcessPlan'].isin(process_plans['ProcessPlan'].unique())]

actual_ops_all = prepare_actual_operations(merged_all, expected_ops)
missing_counts_all = count_missing_operations(actual_ops_all)

# --- Analysis for BROKEN Heat Pumps Only ---

broken_heatpumps = heatpumps[heatpumps['State'] == 5]
broken_serials = set(broken_heatpumps['SerialNumber'])

broken_ops = operations[operations['SerialNumber'].isin(broken_serials)]
broken_ops = broken_ops[broken_ops['ProcessPlan'].isin(process_plans['ProcessPlan'].unique())]

actual_ops_broken = prepare_actual_operations(broken_ops, expected_ops)
missing_counts_broken = count_missing_operations(actual_ops_broken)

# --- Print Results ---

print_distribution_with_percentages("Missing Operations per Serial (ALL HEATPUMPS)", missing_counts_all)
print_distribution_with_percentages("Missing Operations per Serial (BROKEN HEATPUMPS)", missing_counts_broken)

def count_total_and_distinct_repeats(df_ops):
    """Returns two series:
    - Total repeated operations per serial number (repeat count above 1)
    - Number of distinct operations that were repeated per serial number
    """
    def compute_repeats(row):
        counts = pd.Series(row['ManufacturingOrderOperation']).value_counts()
        repeated_ops = counts[counts > 1]
        total_repeats = (repeated_ops - 1).sum()
        num_distinct_repeats = len(repeated_ops)
        return pd.Series({'TotalRepeats': total_repeats, 'NumRepeatedOps': num_distinct_repeats})

    repeat_df = df_ops.apply(compute_repeats, axis=1)
    total_repeat_dist = repeat_df['TotalRepeats'].value_counts().sort_index()
    distinct_repeat_dist = repeat_df['NumRepeatedOps'].value_counts().sort_index()
    return total_repeat_dist, distinct_repeat_dist

def print_repetition_stats(label, total_series, distinct_series):
    print(f"\n{label} — Total Repeated Executions per Serial:")
    total = total_series.sum()
    for n, c in total_series.items():
        print(f"{n} repetitions: {c} ({(c / total * 100):.2f}%)")

    print(f"\n{label} — Number of Distinct Operations Repeated per Serial:")
    total = distinct_series.sum()
    for n, c in distinct_series.items():
        print(f"{n} repeated ops: {c} ({(c / total * 100):.2f}%)")

# --- Repeated Operations Analysis for ALL HEATPUMPS ---
total_repeats_all, distinct_repeats_all = count_total_and_distinct_repeats(actual_ops_all)
print_repetition_stats("Repeated Operations (ALL HEATPUMPS)", total_repeats_all, distinct_repeats_all)

# --- Repeated Operations Analysis for BROKEN HEATPUMPS ---
total_repeats_broken, distinct_repeats_broken = count_total_and_distinct_repeats(actual_ops_broken)
print_repetition_stats("Repeated Operations (BROKEN HEATPUMPS)", total_repeats_broken, distinct_repeats_broken)

# --- Top Operations in BROKEN HEATPUMPS ---

# Get unique broken and all serials
all_serials = heatpumps['SerialNumber'].unique()
broken_serials = heatpumps[heatpumps['State'] == 5]['SerialNumber'].unique()

# Filter operations for both groups
all_ops_filtered = operations[operations['SerialNumber'].isin(all_serials)]
broken_ops_filtered = operations[operations['SerialNumber'].isin(broken_serials)]

# Count in how many unique broken heatpump serials each operation appeared
broken_op_serials = (
    broken_ops_filtered
    .groupby('MfgOrderOperationText')['SerialNumber']
    .nunique()
    .rename('BrokenHeatpumpsWithOp')
)

# Count in how many unique all heatpump serials each operation appeared
all_op_serials = (
    all_ops_filtered
    .groupby('MfgOrderOperationText')['SerialNumber']
    .nunique()
    .rename('AllHeatpumpsWithOp')
)

# Total number of unique serial numbers
total_broken_serials = len(broken_serials)
print(f"Number of broken serial numbers: {total_broken_serials}")
total_all_serials = len(all_serials)

# Combine and calculate percentages
serial_stats = pd.concat([broken_op_serials, all_op_serials], axis=1)
serial_stats['%BrokenHeatpumps'] = (serial_stats['BrokenHeatpumpsWithOp'] / total_broken_serials) * 100
serial_stats['%AllHeatpumps'] = (serial_stats['AllHeatpumpsWithOp'] / total_all_serials) * 100

# Sort and show top 10
top_serial_stats = serial_stats.sort_values(by='BrokenHeatpumpsWithOp', ascending=False).head(10)

# Show all columns
pd.set_option('display.max_columns', None)

print("\n🔝 Top 10 Operations by Occurrence in Broken Heat Pumps (per Serial Number):")
print(top_serial_stats.round(2))











