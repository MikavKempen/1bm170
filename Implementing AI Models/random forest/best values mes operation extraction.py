import pandas as pd

# Load dataset
df = pd.read_csv("heatpump_augmented_mika.csv")

# Columns H–Y = columns 7 to 24
feature_columns = df.columns[7:25].drop("Target_Broken", errors='ignore')
target = df["Target_Broken"]

# Prepare output
results = []

for col in feature_columns:
    present_broken = df[target == 1][col].notna() & (df[target == 1][col] != 0)
    present_nonbroken = df[target == 0][col].notna() & (df[target == 0][col] != 0)

    pct_broken = 100 * present_broken.sum() / (target == 1).sum()
    pct_nonbroken = 100 * present_nonbroken.sum() / (target == 0).sum()
    difference = abs(pct_broken - pct_nonbroken)

    results.append({
        "Feature": col,
        "Present in Broken (%)": round(pct_broken, 1),
        "Present in Non-Broken (%)": round(pct_nonbroken, 1),
        "Difference": round(difference, 1)
    })

# Create and print full sorted DataFrame
df_result = pd.DataFrame(results).sort_values("Difference", ascending=False).reset_index(drop=True)
print(df_result.to_string(index=False))
