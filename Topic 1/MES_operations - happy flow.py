import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

# Load the data
mes_df = pd.read_csv("mes_operations_cleaned.csv")
plans_df = pd.read_csv("Dataset6-process_plans.csv", delimiter=";")

# Prepare reference dicts
process_plan_dict = plans_df.sort_values(by='Operation').groupby('Process Plan')['Operation Name'].apply(list).to_dict()
serial_ops = mes_df.sort_values(by='BeginDateTime').groupby('SerialNumber')['MfgOrderOperationText'].apply(list)
serial_plan_map = mes_df.drop_duplicates('SerialNumber').set_index('SerialNumber')['ProcessPlan'].to_dict()

# Check if plan_ops is a subsequence of ops
def is_subsequence(plan_ops, ops):
    it = iter(ops)
    return all(op in it for op in plan_ops)

def compute_serial_metrics(sn, ops):
    plan_id = serial_plan_map.get(sn)
    if not plan_id or plan_id not in process_plan_dict:
        return None

    plan_ops = process_plan_dict[plan_id]

    # Check for perfect match
    is_exact_match = ops == plan_ops

    # Check all operations from plan exist in serial trace (including frequency)
    ops_counter = Counter(ops)
    plan_counter = Counter(plan_ops)
    is_all_present = all(ops_counter[op] >= count for op, count in plan_counter.items())

    # Get filtered execution sequence (preserving order)
    filtered_ops = [op for op in ops if op in plan_ops]
    is_orderly_correct = filtered_ops == plan_ops

    # Final classification
    if is_exact_match:
        perf = 'Perfect Process Plan'
    elif is_all_present and is_orderly_correct:
        perf = 'Orderly Complete Process'
    elif is_all_present:
        perf = 'Complete Process Orderly Incorrect'
    else:
        perf = 'Incomplete Process'

    return {
        'SerialNumber': sn,
        'ProcessPlan': plan_id,
        'Performance': perf
    }

# Run and build DataFrame
serial_results = pd.DataFrame([
    compute_serial_metrics(sn, ops)
    for sn, ops in serial_ops.items()
    if compute_serial_metrics(sn, ops)
])

# Aggregate
counts = serial_results.groupby(['ProcessPlan', 'Performance']).size().unstack(fill_value=0)

# Plot - Absolute
fig, ax = plt.subplots(figsize=(16, 6))
counts.plot(kind='bar', stacked=True, ax=ax, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
ax.set_title('Process Performance by Plan (Serial Number Based)')
ax.set_ylabel('Count')
plt.xticks(rotation=90)
ax.legend(title='Performance')
plt.tight_layout()
plt.show()

# Plot - Percentage
percentages = counts.div(counts.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(16, 6))
percentages.plot(kind='bar', stacked=True, ax=ax, color=['#66c2a5', '#fc8d62', '#8da0cb', '#e78ac3'])
ax.set_title('Process Performance Distribution by Plan (%)')
ax.set_ylabel('Percentage (%)')
plt.xticks(rotation=90)
ax.legend(title='Performance')
plt.tight_layout()
plt.show()

