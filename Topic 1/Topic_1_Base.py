import pandas as pd

# Laad de CSV-bestanden
heatpumps = pd.read_csv('Dataset1-heatpumps.csv')
mes_operations = pd.read_csv('Dataset2-mes_operations.csv')
test_measurements = pd.read_csv('Dataset3-test_measurements.csv', low_memory=False)
genealogy = pd.read_csv('Dataset4-genealogy.csv')
monthly_energy_logs = pd.read_csv('Dataset5-monthly_energy_logs.csv')
process_plans = pd.read_csv('Dataset6-process_plans.csv')
