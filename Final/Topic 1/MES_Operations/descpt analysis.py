import pandas as pd

# Load datasets
mes_df = pd.read_csv("Dataset2-mes_operations.csv", parse_dates=["BeginDateTime", "EndDateTime"])
heatpump_df = pd.read_csv("Dataset1-heatpumps.csv")

# Merge to get Model info
merged_df = pd.merge(mes_df, heatpump_df[['SerialNumber', 'Model']], on='SerialNumber', how='left')

# Calculate operation duration in seconds
merged_df["Duration_sec"] = (merged_df["EndDateTime"] - merged_df["BeginDateTime"]).dt.total_seconds()

# Group by operation type and model
grouped_stats = merged_df.groupby(["MfgOrderOperationText", "Model"])["Duration_sec"].describe()

# Print full descriptive stats (optional)
print(grouped_stats)

# Show three rows with known large spread
print("\nThree examples with large spread:\n")
examples = grouped_stats.reset_index()
examples = examples.sort_values(by="std", ascending=False)  # sort by std to pick ones with large spread
print(examples.head(3))
import pandas as pd

# Load datasets
mes_df = pd.read_csv("Dataset2-mes_operations.csv", parse_dates=["BeginDateTime", "EndDateTime"])
heatpump_df = pd.read_csv("Dataset1-heatpumps.csv")

# Merge to get Model info
merged_df = pd.merge(mes_df, heatpump_df[['SerialNumber', 'Model']], on='SerialNumber', how='left')

# Calculate operation duration in seconds
merged_df["Duration_sec"] = (merged_df["EndDateTime"] - merged_df["BeginDateTime"]).dt.total_seconds()

# Filter to only include 'FT' operations
ft_df = merged_df[merged_df["MfgOrderOperationText"] == "FT"]

# Group by model and describe durations
ft_stats = ft_df.groupby("Model")["Duration_sec"].describe()

# Optional: round for readability
ft_stats = ft_stats.round(2)

# Print the table
print("Descriptive statistics for 'FT' operation by model:\n")
print(ft_stats)

# === Commissioning Month Analysis ===
print("\nDescriptive stats of commissioned pumps per month (including % broken):\n")

# Ensure datetime is parsed
heatpump_df['CommissionedAt'] = pd.to_datetime(heatpump_df['CommissionedAt'], errors='coerce')

# Drop NaT values in CommissionedAt
commissioned_df = heatpump_df[heatpump_df['CommissionedAt'].notnull()].copy()

# Create CommissionedMonth column
commissioned_df['CommissionedMonth'] = commissioned_df['CommissionedAt'].dt.to_period('M')

# Group by month
monthly_stats = commissioned_df.groupby('CommissionedMonth').agg(
    TotalPumps=('SerialNumber', 'count'),
    BrokenPumps=('State', lambda x: (x == 5).sum())
)

monthly_stats['% Broken'] = (monthly_stats['BrokenPumps'] / monthly_stats['TotalPumps'] * 100).round(2)

print(monthly_stats)

# Optional: Save to CSV
monthly_stats.to_csv("heatpump_monthly_broken_stats.csv")
print("\nMonthly stats saved to 'heatpump_monthly_broken_stats.csv'")
# === Commissioning Month Analysis ===
print("\nDescriptive stats of commissioned pumps per month (including % broken):\n")

# Ensure datetime is parsed
heatpump_df['CommissionedAt'] = pd.to_datetime(heatpump_df['CommissionedAt'], errors='coerce')

# Drop NaT values in CommissionedAt
commissioned_df = heatpump_df[heatpump_df['CommissionedAt'].notnull()].copy()

# Create CommissionedMonth column
commissioned_df['CommissionedMonth'] = commissioned_df['CommissionedAt'].dt.to_period('M')

# Group by month
monthly_stats = commissioned_df.groupby('CommissionedMonth').agg(
    TotalPumps=('SerialNumber', 'count'),
    BrokenPumps=('State', lambda x: (x == 5).sum())
)

monthly_stats['% Broken'] = (monthly_stats['BrokenPumps'] / monthly_stats['TotalPumps'] * 100).round(2)

print(monthly_stats)

# Optional: Save to CSV
monthly_stats.to_csv("heatpump_monthly_broken_stats.csv")
print("\nMonthly stats saved to 'heatpump_monthly_broken_stats.csv'")
