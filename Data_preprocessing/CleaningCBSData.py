import pandas as pd

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
    "International organisations"
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
print(f"Removed {name_removed_rows} rows that didnt include actual countries.")

# Remove rows where all time-related columns are empty (NaN)
filtered_data_years = filtered_data_country.dropna(
    subset=time_columns, how='all')

filtered_data_years.to_csv(output_file, index=False)


# Print the number of rows removed
number_removed_rows = len(filtered_data_country) - len(filtered_data_years)

print(
    f"Removed {number_removed_rows} rows that had no numerical data from 1982-2024.")
# print total number of removed rows
print(
    f"Total number of rows removed: {name_removed_rows + number_removed_rows}")
print(
    f"Reduced the dataset from {len(data)} to {len(filtered_data_years)} rows.")
# Save the filtered dataset
print(f"Filtered dataset saved to {output_file}.")
