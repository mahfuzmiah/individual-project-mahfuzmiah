import pandas as pd

# Define input file path
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedCBSDataSet.csv"

# Load dataset
data = pd.read_csv(input_file)

# Ensure column names are clean
data.columns = data.columns.str.strip()

# Dictionary to store unique values per column
unique_values_summary = {}

# Iterate through each column and collect unique values
for col in data.columns:
    # Check if the column is numeric
    if pd.api.types.is_numeric_dtype(data[col]):
        unique_values_summary[col] = f"{data[col].count()} instances"
    else:
        unique_values = data[col].dropna().unique()

        # Check if column is categorical (skip columns with too many unique values)
        if len(unique_values) < 200:  # Adjust threshold if needed
            unique_values_summary[col] = unique_values[:200]
        else:
            unique_values_summary[col] = f"{len(unique_values)} unique values (too many to display)"

# Convert to DataFrame for better readability
unique_values_df = pd.DataFrame(list(unique_values_summary.items()), columns=[
                                "Column", "Example Unique Values"])

# Save to a file (optional)
output_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/UniqueColumnSummary.csv"
unique_values_df.to_csv(output_file, index=False)

print(f"Unique values per column summary saved to {output_file}.")
