import pandas as pd

# Load data
df = pd.read_csv("heatpump_augmented_mika.csv")

# Replace 'Unknown' with NaN across all columns
df.replace("Unknown", pd.NA, inplace=True)

# Convert CommissionedMonth to datetime and sort unique months
df["CommissionedMonth"] = pd.to_datetime(df["CommissionedMonth"], format="%Y-%m", errors="coerce")
unique_months_sorted = sorted(df["CommissionedMonth"].dropna().unique())

# Map CommissionedMonth to integer index
month_map = {month: idx for idx, month in enumerate(unique_months_sorted)}
df["CommissionedMonthIndex"] = df["CommissionedMonth"].map(month_map)

# Drop unused or redundant columns
df = df.drop(columns=["CommissionedAt", "CommissionedMonth", "State", "SerialNumber", "ID"], errors="ignore")

# One-hot encode simple categorical columns
df = pd.get_dummies(df, columns=["BoilerType", "DhwType"], dtype=int)

# Binary features from operation type columns
df["first_op_is_assembly"] = df["first_operation_type"].fillna("").str.contains("Assembly").astype(int)
df["last_op_is_packing"] = df["last_operation_type"].fillna("").str.contains("Packing").astype(int)
df["dominant_op_is_assembly"] = df["dominant_operation_type"].fillna("").str.contains("Assembly").astype(int)

# Drop original operation type columns
df = df.drop(columns=["first_operation_type", "last_operation_type", "dominant_operation_type"], errors="ignore")

# Drop rows where target is missing
df = df.dropna(subset=["Target_Broken"])

# Save the result
df.to_csv("heatpump_nn_ready.csv", index=False)
