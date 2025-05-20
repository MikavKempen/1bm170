import pandas as pd

# Load dataset
df = pd.read_csv("Dataset2-mes_operations.csv")

print("=== Initial checks ===")
print("Missing values per column:\n")
print(df.isnull().sum())
print("\nShape before any cleaning:", df.shape)

# Ensure datetime columns are parsed
df['BeginDateTime'] = pd.to_datetime(df['BeginDateTime'], errors='coerce')
df['EndDateTime'] = pd.to_datetime(df['EndDateTime'], errors='coerce')

# Drop rows with missing critical data (Begin/End time or SerialNumber)
df = df.dropna(subset=['BeginDateTime', 'EndDateTime', 'SerialNumber'])

# Calculate duration
df['DurationSeconds'] = (df['EndDateTime'] - df['BeginDateTime']).dt.total_seconds()

# Flag suspicious durations
zero_durations = df[df['DurationSeconds'] == 0]
negative_durations = df[df['DurationSeconds'] < 0]

print(f"\nZero-duration rows: {len(zero_durations)}")
print(f"Negative-duration rows: {len(negative_durations)}")

# Keep them for now — just log
# You could export these for QA if needed:
# zero_durations.to_csv("zero_durations.csv", index=False)
# negative_durations.to_csv("negative_durations.csv", index=False)

# Check for duplicates based on full row
full_duplicates = df.duplicated()
print(f"\nExact duplicate rows: {full_duplicates.sum()}")
df = df[~full_duplicates]

# Check for duplicate operations (same SerialNumber + OperationText)
duplicate_ops = df.duplicated(subset=['SerialNumber', 'MfgOrderOperationText', 'BeginDateTime', 'EndDateTime'], keep=False)
print(f"Duplicate operation entries (same SN + op + times): {duplicate_ops.sum()}")

# Just mark them — don’t drop!
df['IsDuplicatedOperation'] = duplicate_ops

# Output cleaned version
df.to_csv("mes_operations_cleaned.csv", index=False)
print("\nCleaned dataset saved to 'mes_operations_cleaned.csv'")
print("Shape after cleaning:", df.shape)

import pandas as pd

# Load the cleaned dataset
df = pd.read_csv("mes_operations_cleaned.csv", parse_dates=['BeginDateTime', 'EndDateTime'])

print("\n" + "="*80)
print("MES OPERATIONS OVERVIEW")
print("="*80 + "\n")

# === 1. Check for same Begin == End timestamps (zero duration)
same_start_end = df[df['BeginDateTime'] == df['EndDateTime']]
print(f"Rows with same Begin and End time (zero-duration): {len(same_start_end)}")
print("Example of affected SerialNumbers:")
print(same_start_end[['SerialNumber', 'MfgOrderOperationText']].head())

# === 2. ID column: Should be unique
print(f"\nUnique ID values: {df['ID'].nunique()} out of {len(df)} rows")
if df['ID'].nunique() != len(df):
    print("⚠️ Warning: IDs are not unique!")

# === 3. ManufacturingOrder: how many operations per order
ops_per_order = df.groupby('ManufacturingOrder')['MfgOrderOperationText'].count()
print(f"\nAverage number of operations per ManufacturingOrder: {ops_per_order.mean():.2f}")
print(f"Min operations: {ops_per_order.min()}, Max operations: {ops_per_order.max()}")

# === 4. Does same order use consistent operation names?
print("\nChecking if operations are consistent within orders...")
inconsistent_orders = (
    df.groupby(['ManufacturingOrder', 'MfgOrderOperationText'])['Material']
    .nunique().reset_index()
    .groupby('ManufacturingOrder')['Material'].nunique()
)
weird_orders = inconsistent_orders[inconsistent_orders > 1]
print(f"Orders with inconsistent materials for the same operation: {len(weird_orders)}")
if len(weird_orders) > 0:
    print(weird_orders.head())

# === 5. Material: Unique values and example link to model
print("\nUnique materials:", df['Material'].nunique())
print("Example materials:")
print(df['Material'].value_counts().head())

# === 6. ProcessPlan: how many plans, how many steps per plan
print("\nUnique ProcessPlans:", df['ProcessPlan'].nunique())
steps_per_plan = df.groupby('ProcessPlan')['MfgOrderOperationText'].nunique()
print("Steps per ProcessPlan (sample):")
print(steps_per_plan.head())

# === 7. SerialNumber: flow times through process
flow_times = df.groupby('SerialNumber').agg(
    start=('BeginDateTime', 'min'),
    end=('EndDateTime', 'max')
)
flow_times['FlowTimeMinutes'] = (flow_times['end'] - flow_times['start']).dt.total_seconds() / 60
print("\nSample flow times (in minutes):")
print(flow_times['FlowTimeMinutes'].describe())

# === 8. WorkCenter: importance & usage
print("\nTop 10 most used Workcenters:")
print(df['WorkCenter'].value_counts().head(10))

# === 9. TotalQuantity checks per order
quantity_summary = df.groupby('ManufacturingOrder').agg(
    total_qty=('TotalQuantity', 'first'),
    serials_count=('SerialNumber', 'nunique')
)
quantity_summary['delta'] = quantity_summary['total_qty'] - quantity_summary['serials_count']
print("\nOrders where TotalQuantity ≠ number of SerialNumbers (top 5 mismatches):")
print(quantity_summary[quantity_summary['delta'] != 0].sort_values(by='delta', ascending=False).head())

# === 10. Planned quantity vs confirmed yield + scrap
df['ConfirmedTotal'] = df['OpTotalConfirmedYieldQty'] + df['OpTotalConfirmedScrapQty']
qty_mismatch = df[df['ConfirmedTotal'] != df['TotalQuantity']]
print(f"\nRows where yield + scrap ≠ TotalQuantity: {len(qty_mismatch)}")

# === 11. ManufacturingOrderOperation vs OperationText (optional)
# Just show 5 examples where the same ID has multiple Texts
op_text_conflict = df.groupby(['ManufacturingOrderOperation'])['MfgOrderOperationText'].nunique()
conflicting_ops = op_text_conflict[op_text_conflict > 1]
print(f"\nManufacturingOrderOperation with multiple operation texts: {len(conflicting_ops)}")
if len(conflicting_ops) > 0:
    print(conflicting_ops.head())

print("\n" + "="*80)
print("Done analyzing mes_operations.csv")
print("="*80)
