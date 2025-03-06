import matplotlib.pyplot as plt
import json
import numpy as np
import pandas as pd
import sys
import os


# Input and output file paths
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/WS_LBS_D_PUB_csv_col-3.csv"
output_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedLBSdataset.csv"

# Load the dataset
data = pd.read_csv(input_file)


# Define unwanted entries, these are entries that are not actual countries
unwanted_entries = [
    "All reporting countries",
    "Euro area",
    "All countries (total)",
    "Residual developed countries",
    "Residual west Indies UK",
    "Residual developing Asia and pacific",
    "Residual developing Lating America and Caribbean",
    "Residual developing Europe",
    "Residual former Yugoslavia",
    "Residual offshore centres",
    "Residual former Soviet Union",
    "Residual former Czechoslovakia",
    "International organisations",
    "Residual British Overseas Territories"
]
# Convert unwanted entries to lowercase and strip spaces
unwanted_entries = [x.lower().strip() for x in unwanted_entries]

# Ensure column names are clean
data.columns = data.columns.str.strip()


# Identify numerical columns related to years
# Remove the rows with no data from 1977-2024 as this will slow down computation
time_columns = [col for col in data.columns if any(
    str(year) in col for year in range(1977, 2025))]

# Filter out rows where 'Reporting country' or 'Counterparty country' contains unwanted entries
filtered_data_country = data[
    ~data['Reporting country'].fillna('').str.lower().str.strip().isin(unwanted_entries) &
    ~data['Counterparty country'].fillna(
        '').str.lower().str.strip().isin(unwanted_entries)
]
name_removed_rows = len(data) - len(filtered_data_country)


# Remove rows where all time-related columns are empty (NaN)
filtered_data_years = filtered_data_country.dropna(
    subset=time_columns, how='all')
number_removed_rows = len(filtered_data_country) - len(filtered_data_years)


# Remove rows labeled "L" in L_position
L_position_claims_removed = len(
    filtered_data_years[filtered_data_years['L_POSITION'] == 'L'])
filtered_data_L_rows = filtered_data_years[filtered_data_years['L_POSITION'] != 'L']

# Remove rows not labled "Loans and deposits" in "Type of instruments"
Instruments_rows_removed = len(
    filtered_data_L_rows[filtered_data_L_rows['Type of instruments'] != 'Loans and deposits'])
filtered_data_Instruments = filtered_data_L_rows[
    filtered_data_L_rows['Type of instruments'] == 'Loans and deposits']

# Remove rows not labelled "TO1" in "L_denom"
Currency_rows_removed = len(
    filtered_data_Instruments[filtered_data_Instruments['Currency type of reporting country'] != 'All currencies (=D+F+U)'])
filtered_data_currency = filtered_data_Instruments[filtered_data_Instruments[
    'Currency type of reporting country'] == 'All currencies (=D+F+U)']

# Remove rows labeled "B" in L_MEASURE
measure_S_rows_removed = len(
    filtered_data_currency[filtered_data_currency['L_MEASURE'] != 'S'])
filtered_data_S_rows = filtered_data_currency[filtered_data_currency['L_MEASURE'] == 'S']

# Remove rows with counter_party sector "Non-banks total"
counter_party_sector_rows_removed = len(
    filtered_data_S_rows[filtered_data_S_rows['Counterparty sector'] != 'All sectors'])
filtered_data_sector = filtered_data_S_rows[
    filtered_data_S_rows['Counterparty sector'] == 'All sectors']
# Total rows removed
total_removed_rows = (
    name_removed_rows +
    number_removed_rows +
    L_position_claims_removed +
    Instruments_rows_removed +
    Currency_rows_removed +
    measure_S_rows_removed +
    counter_party_sector_rows_removed
)

# Reporting
print(
    f"Removed {name_removed_rows} rows that didnt include countries.")
print(
    f"Removed {number_removed_rows} rows that had no numerical data from 1977-2024.")

print(
    f"Removed {L_position_claims_removed} rows that were labeled 'L' in L_POSITION.")
print(
    f"Removed {Instruments_rows_removed} rows that were not labeled 'Loans and deposits' in 'Type of instruments'.")
print(
    f"Removed {Currency_rows_removed} rows that were not labeled 'All currencies (=D+F+U)' in 'Currency type of reporting country'.")
print(
    f"Removed {measure_S_rows_removed} rows that were labeled 'B' in 'L_MEASURE'.")
print(
    f"Removed {counter_party_sector_rows_removed} rows that were not labeled 'All sectors' in 'Counterparty sector'.")


print(
    f"Reduced the dataset from {len(data)} to {len(filtered_data_sector)} rows.")

# Save the filtered dataset
filtered_data_sector.to_csv(output_file, index=False)
print(f"Filtered dataset saved to {output_file}.")
