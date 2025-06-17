import pandas as pd

# Load the file
df = pd.read_csv("Dataset6-process_plans.csv", delimiter=";")

print("=== Initial Structure ===")
print(df.info())
print("\nSample rows:")
print(df.head())

# 1. Check for missing values
print("\n=== Missing values per column ===")
print(df.isnull().sum())

# 2. Drop rows with missing info
df_clean = df.dropna(subset=["Process Plan", "Operation", "Operation Name"])

# 3. Rename columns for consistency
df_clean = df_clean.rename(columns={
    "Process Plan": "ProcessPlan",
    "Operation": "OperationID",
    "Operation Name": "OperationName"
})

# 4. Check for duplicate operations per plan
dup_ops = df_clean.duplicated(subset=["ProcessPlan", "OperationID"])
print(f"\nDuplicate step IDs per plan: {dup_ops.sum()}")
if dup_ops.sum() > 0:
    print("Example duplicates:")
    print(df_clean[dup_ops].head())

# 5. Check for duplicated operation names within a plan
dup_names = df_clean.duplicated(subset=["ProcessPlan", "OperationName"])
print(f"Duplicate operation names per plan: {dup_names.sum()}")

# 6. Final stats
print(f"\nNumber of unique ProcessPlans: {df_clean['ProcessPlan'].nunique()}")
ops_per_plan = df_clean.groupby("ProcessPlan")["OperationID"].count()
print("\nOperations per ProcessPlan (summary):")
print(ops_per_plan.describe())

# 7. Save cleaned version
df_clean.to_csv("process_plans_cleaned.csv", index=False)
print("\nCleaned process plan saved as 'process_plans_cleaned.csv'")
