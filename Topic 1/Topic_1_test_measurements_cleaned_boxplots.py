import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('test_measurements_cleaned.csv', low_memory=False)

# Normalize 'unit' column values
df['unit'] = df['unit'].replace({
    'A': 'Ampere',
    'degreesC': 'DegreesC'
})

# Filter relevant rows
bar_high_limit_ops = df[(df['unit'] == 'Bar') & (df['upper_specification_limit'] > 100)]

# Count occurrences per operation_id
operation_id_counts = bar_high_limit_ops['operation_id'].value_counts()

# Print the result
print("Count of occurrences where 'Bar' unit and upper_specification_limit > 100 by operation_id:")
print(operation_id_counts)

# Count occurrences per operation_id where lower_specification_limit < 0 for unit 'seconds'
negative_lower_counts = df[
    (df['unit'] == 'seconds') & (df['lower_specification_limit'] < 0)
].groupby('operation_id').size()

print(f"number of negative lower counts seconds: {negative_lower_counts}")


for unit in df['unit'].dropna().unique():
    unit_df = df[df['unit'] == unit]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    unit_df['value'].plot.box()
    plt.title(f'{unit} - Value')
    plt.ylabel('Value')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    unit_df['lower_specification_limit'].plot.box()
    plt.title(f'{unit} - Lower Limit')
    plt.ylabel('Lower Spec Limit')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    unit_df['upper_specification_limit'].plot.box()
    plt.title(f'{unit} - Upper Limit')
    plt.ylabel('Upper Spec Limit')
    plt.grid(True)

    plt.suptitle(f'Boxplots for Unit: {unit}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()