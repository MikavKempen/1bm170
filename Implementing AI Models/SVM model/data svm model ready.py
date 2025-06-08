import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load data
df = pd.read_csv("heatpump_nn_ready.csv")

# Replace 'Unknown' with NaN just in case
df.replace("Unknown", pd.NA, inplace=True)

# Drop rows with missing target
df = df.dropna(subset=["Target_Broken"])

# Drop all-NaN columns
nan_cols = df.columns[df.isna().all()]
print("Dropped all-NaN columns:", list(nan_cols))
df = df.drop(columns=nan_cols)

# Columns for transformation
one_hot_cols = ["Model"]
std_scale_cols = [
    "total_duration",
    "avg_operation_duration",  # will cap first
    "total_process_timespan_seconds",
    "total_process_duration",
    "NumberOfFailedTests",
]
target_col = "Target_Broken"

# Cap extreme values
df["avg_operation_duration"] = df["avg_operation_duration"].clip(upper=14400)

# Identify binary yes/no columns
binary_cols = [
    col for col in df.select_dtypes(include="number").columns
    if set(df[col].dropna().unique()).issubset({0, 1}) and col != target_col
]

# Fill binary yes/no NaNs with 0.5
df[binary_cols] = df[binary_cols].fillna(0.5)

# One-hot encode categorical + binary columns
df = pd.get_dummies(df, columns=one_hot_cols + binary_cols, dtype=int)

# Separate target
y = df[target_col]
X = df.drop(columns=[target_col])

# Standard scale selected columns
scaler_std = StandardScaler()
X[std_scale_cols] = scaler_std.fit_transform(X[std_scale_cols])

# MinMax scale everything else
mm_cols = X.select_dtypes(include="number").columns.difference(std_scale_cols)
scaler_mm = MinMaxScaler()
X[mm_cols] = scaler_mm.fit_transform(X[mm_cols])

# Combine and save
final_df = pd.concat([X, y], axis=1)
final_df.to_csv("heatpump_svm_ready.csv", index=False)
