# Load up training set
# Look average out the each quarters data
# plot the data as line chart, Years on x-axis and average on y-axis


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv(
    '/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/imputed_forward_fill.csv')

# Drop non-quarter columns
quarterly_columns = [col for col in df.columns if 'Q' in col]
df_quarters = df[quarterly_columns]

# Convert to numeric (ignoring errors for missing or non-numeric values)
df_quarters = df_quarters.apply(pd.to_numeric, errors='coerce')

# Calculate the average for each quarter across all entries
average_per_quarter = df_quarters.mean()

# Convert index (e.g., "2005-Q1") to datetime for proper plotting
average_per_quarter.index = pd.PeriodIndex(
    average_per_quarter.index, freq='Q').to_timestamp()

# Plot the data
plt.figure(figsize=(12, 6))
plt.plot(average_per_quarter.index,
         average_per_quarter.values, marker='o', linestyle='-')
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.title('Average Quarterly Values Over Time')
plt.grid(True)
plt.show()


# Identify unique bank types
bank_types = df['CBS_BASIS'].dropna().unique()

# Create a plot for each bank type
plt.figure(figsize=(12, 6))

for bank_type in bank_types:
    # Filter data for the specific bank type
    df_filtered = df[df['CBS_BASIS'] == bank_type][quarterly_columns]

    # Convert to numeric (ignoring errors for missing or non-numeric values)
    df_filtered = df_filtered.apply(pd.to_numeric, errors='coerce')

    # Calculate the average for each quarter
    average_per_quarter = df_filtered.mean()

    # Convert index (e.g., "2005-Q1") to datetime for proper plotting
    average_per_quarter.index = pd.PeriodIndex(
        average_per_quarter.index, freq='Q').to_timestamp()

    # Plot the data
    plt.plot(average_per_quarter.index, average_per_quarter.values,
             marker='o', linestyle='-', label=f'Bank Type {bank_type}')

# Customize plot
plt.xlabel('Year')
plt.ylabel('Average Value')
plt.title('Average Quarterly Values Over Time by Bank Type')
plt.legend(title="Bank Type")
plt.grid(True)
plt.show()
