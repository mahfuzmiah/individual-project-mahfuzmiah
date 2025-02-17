import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import sys
import os


# Input and output file paths
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/WS_CBS_PUB_csv_col.csv"
output_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedCBSDataSet.csv"

# Load the dataset
data = pd.read_csv(input_file)

# Define unwanted entries, these are entries that are not actual countries
unwanted_entries = [
    "All reporting countries",
    "All countries (total)",
    "All countries excluding residents",
    "Euro area",
    "Developed countries",
    "Developing countries",
    "Developing Asia and Pacific",
    "Developing Europe",
    "Developing Africa and Middle East",
    "Residual developing Europe",
    "Residual developing Africa and Middle East",
    "Unallocated location",
    "Residual British Overseas Territories",
    "Residual West Indies UK",
    "Emerging market and developing economies",
    "European developed countries",
    "Developing Latin America and Caribbean",
    "International organisations",
    "Residents/Local"
]

# Convert unwanted entries to lowercase and strip spaces
unwanted_entries = [x.lower().strip() for x in unwanted_entries]

# Ensure column names are clean
data.columns = data.columns.str.strip()

# Identify numerical columns related to years
# Remove the rows with no data from 1982-2024 as this will slow down computation
time_columns = [col for col in data.columns if any(
    str(year) in col for year in range(1982, 2025))]

# Filter out rows where 'Reporting country' or 'Counterparty country' contains unwanted entries
filtered_data_country = data[
    ~data['Reporting country'].fillna('').str.lower().str.strip().isin(unwanted_entries) &
    ~data['Counterparty country'].fillna(
        '').str.lower().str.strip().isin(unwanted_entries)
]
name_removed_rows = len(data) - len(filtered_data_country)
print(f"Removed {name_removed_rows} rows that didn't include actual countries.")

# Remove rows where all time-related columns are empty (NaN)
filtered_data_years = filtered_data_country.dropna(
    subset=time_columns, how='all')
number_removed_rows = len(filtered_data_country) - len(filtered_data_years)

# Remove rows labeled "B" in L_MEASURE
measure_b_rows_removed = len(
    filtered_data_years[filtered_data_years['L_MEASURE'] == 'B'])
filtered_data_b_rows = filtered_data_years[filtered_data_years['L_MEASURE'] != 'B']

# Remove rows labled "Derivatives" in "Type of instruments"
derivatives_rows_removed = len(
    filtered_data_b_rows[filtered_data_b_rows['Type of instruments'] == 'Derivatives'])
filtered_data_Instruments = filtered_data_b_rows[
    filtered_data_b_rows['Type of instruments'] != 'Derivatives']

# Remove rows labels "Up to and including 1 year" in "Remaining maturity"
remaining_maturity_rows_removed = len(
    filtered_data_Instruments[filtered_data_Instruments['Remaining maturity'] == 'Up to and including 1 year'])
filtered_data_maturity_rows = filtered_data_Instruments[filtered_data_Instruments['Remaining maturity']
                                                        != 'Up to and including 1 year']

# Remove rows labelled "Local currency" in "Currency type of booking location"
local_currency_rows_removed = len(
    filtered_data_maturity_rows[filtered_data_maturity_rows['Currency type of booking location'] == 'Local currency'])
filtered_data_currency = filtered_data_maturity_rows[filtered_data_maturity_rows[
    'Currency type of booking location'] != 'Local currency']


# Filter out rows where 'Counterparty sector' is not labeled as 'All sectors'
counter_party_sector_rows_removed = len(
    filtered_data_currency[filtered_data_currency['Counterparty sector'].str.lower().str.strip() != 'all sectors'])
filtered_data_sector = filtered_data_currency[
    filtered_data_currency['Counterparty sector'].str.lower().str.strip() == 'all sectors']

# Filter out rows where "Balance sheet position" is not labeled as "Total claims"
balance_sheet_position_removed = len(
    filtered_data_sector[filtered_data_sector['Balance sheet position'].str.lower().str.strip() != 'total claims'])
filtered_data_positon = filtered_data_sector[filtered_data_sector['Balance sheet position'].str.lower(
).str.strip() == 'total claims']

# Clean column names
filtered_data_positon.columns = filtered_data_positon.columns.str.strip()

# Identify relevant categories
data_4B = filtered_data_positon[filtered_data_positon['CBS_BANK_TYPE'].str.upper(
).str.strip() == '4B']
data_4R = filtered_data_positon[filtered_data_positon['CBS_BANK_TYPE'].str.upper(
).str.strip() == '4R']
data_4O = filtered_data_positon[filtered_data_positon['CBS_BANK_TYPE'].str.upper(
).str.strip() == '4O']
data_4C = filtered_data_positon[filtered_data_positon['CBS_BANK_TYPE'].str.upper(
).str.strip() == '4C']


# Total rows removed
total_removed_rows = (
    name_removed_rows +
    number_removed_rows +
    measure_b_rows_removed +
    derivatives_rows_removed +
    remaining_maturity_rows_removed +
    local_currency_rows_removed +
    counter_party_sector_rows_removed +
    balance_sheet_position_removed
)

# Reporting
print(
    f"Removed {number_removed_rows} rows that had no numerical data from 1982-2024.")
print(f"Removed {measure_b_rows_removed} rows with 'B' in L_MEASURE.")
print(
    f"Removed {derivatives_rows_removed} rows with 'Derivatives' in Type of instruments.")
print(
    f"Removed {remaining_maturity_rows_removed} rows with 'Up to and including 1 year' in Remaining maturity.")
print(
    f"Removed {local_currency_rows_removed} rows with 'Local currency' in Currency type of booking location.")
print(f"Removed {counter_party_sector_rows_removed} rows without 'All sectors' in Counterparty sector.")
print(
    f"Removed {balance_sheet_position_removed} rows without 'Total claims' in Balance sheet position.")
# print(
#     f"Removed {Bank_type_removed} rows without '4B' in CBS_BANK_TYPE.")
print(f"Total number of rows removed: {total_removed_rows}")

print(
    f"Reduced the dataset from {len(data)} to {len(filtered_data_positon)} rows.")

# Save the filtered dataset
filtered_data_positon.to_csv(output_file, index=False)
print(f"Filtered dataset saved to {output_file}.")


# Now the goal is to subtract all 4R values from 4B values for each quarter and store the result in a new column called "Difference".
# The resulting DataFrame should have the following columns:
# I need


# 1. Create your DataFrame from the filtered data
df = pd.DataFrame(filtered_data_positon)

# 2. Filter for only the four bank types of interest
df_filtered = df[df['CBS_BANK_TYPE'].isin(['4B', '4R', '4C', '4O'])].copy()

# 3. Identify the quarter columns (assuming they contain '-Q' in their name)
quarter_cols = [col for col in df.columns if '-Q' in col]


# 4. Clean up the country codes
df_filtered['L_REP_CTY'] = df_filtered['L_REP_CTY'].str.strip().str.upper()
df_filtered['L_CP_COUNTRY'] = df_filtered['L_CP_COUNTRY'].str.strip().str.upper()

# 5. Create a pivot table with the grouping keys and bank types as columns
df_pivot = df_filtered.pivot_table(
    index=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS'],
    columns='CBS_BANK_TYPE',
    values=quarter_cols,
    aggfunc='sum'
)

# 6. Fill missing values with 0 (to ensure proper arithmetic later)
df_pivot = df_pivot.fillna(0)

# 7. Extract each bank type's values from the MultiIndex columns
df_4B = df_pivot.xs('4B', axis=1, level='CBS_BANK_TYPE')
df_4R = df_pivot.xs('4R', axis=1, level='CBS_BANK_TYPE')
df_4C = df_pivot.xs('4C', axis=1, level='CBS_BANK_TYPE')
df_4O = df_pivot.xs('4O', axis=1, level='CBS_BANK_TYPE')

# 8. Calculate the consolidated edgeweight for each quarter with conditional logic:
#    If the foreign component (4C + 4O) is 0, then use the domestic value (4B) directly.
#    Otherwise, use the formula: (4B - 4R) + (4C + 4O)
result_dict = {}
for col in quarter_cols:
    foreign_total = df_4C[col] + df_4O[col]
    domestic_diff = df_4B[col] - df_4R[col]
    result_dict[col] = np.where(
        foreign_total == 0, df_4B[col], domestic_diff + foreign_total)

# Create a DataFrame from the dictionary, then add the index back
consolidated = pd.DataFrame(result_dict, index=df_4B.index).reset_index()

# 9. Reset index (optional) and save the consolidated results
consolidated = consolidated.reset_index()
consolidated.to_csv(output_file, index=False)
print("Consolidated Edgeweight with conditional logic:")

missing_by_column = df.isnull().sum()
missing_percentage = df.isnull().mean() * 100

print(missing_by_column.sort_values(ascending=False))
print(missing_percentage.sort_values(ascending=False))
