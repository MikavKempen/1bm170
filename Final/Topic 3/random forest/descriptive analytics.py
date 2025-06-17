import pandas as pd

# 1. Load your dataset
df = pd.read_csv("heatpump_augmented_mika.csv")

# 2. If your model column is literally called 'Model', great.
#    Otherwise rename accordingly:
#    e.g. df = df.rename(columns={"WhichModelColumn":"Model"})

# 3. Group by each model
summary = df.groupby("Model").agg(
    total_instances  = ("Model",       "size"),
    failures         = ("Target_Broken","sum")
)

# 4. Compute percentage failed
summary["percent_failed"] = summary["failures"] / summary["total_instances"] * 100

# 5. Compute cumulative totals
total_runs      = summary["total_instances"].sum()
total_failures  = summary["failures"].sum()
total_pct_fail  = total_failures / total_runs * 100

# 6. Append a “Total” row
summary.loc["Total"] = [total_runs, total_failures, total_pct_fail]

# 7. (Optional) format for nicer printing
pd.options.display.float_format = "{:,.1f}".format

print(summary)
