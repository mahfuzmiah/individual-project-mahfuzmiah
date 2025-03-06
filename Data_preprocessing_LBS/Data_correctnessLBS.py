import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Input file path for the LBS dataset (adjust if necessary)
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedLBSdataset.csv"
df = pd.read_csv(input_file)

# For LBS, we assume that the key filtering has already been done so that:
# L_MEASURE is 'S' (Amounts outstanding / Stocks)
# L_POSITION is 'C' (Total claims)
# L_INSTR is 'G' (Loans and deposits)
# L_DENOM is 'TO1' (All currencies)
# L_CURR_TYPE is 'A' (All currencies aggregate)
# L_PARENT_CTY is '5J' (but we'll ignore this, since we focus on Reporting country)
# And Reporting country contains individual countries

# Identify the time columns (assuming they contain "-Q" in their names)
time_columns = [col for col in df.columns if '-Q' in col]
if not time_columns:
    # Alternatively, if time columns contain a year string, you could adjust this pattern.
    time_columns = [col for col in df.columns if any(
        str(year) in col for year in range(1982, 2025))]

# Replace zeros with NaN in the time columns (treating zeros as missing)
df[time_columns] = df[time_columns].replace(0, np.nan)

# -------------------------------
# 1. Line Chart: Missing Data Percentage for Time Columns
# -------------------------------
# Compute the missing percentage for each time column
missing_percent = df[time_columns].isnull().mean() * 100

print("Missing percentages for time columns:")
print(missing_percent.sort_values(ascending=False))

# Plot a line chart of missing percentages across time
plt.figure(figsize=(20, 6))
plt.plot(missing_percent.index, missing_percent.values,
         marker='o', color='skyblue', linestyle='-')
plt.title("Missing Data Percentage for LBS Time Period Columns")
plt.xlabel("Quarter")
plt.ylabel("Percentage Missing")

# To reduce clutter, display every 4th label on the x-axis
step = 4
plt.xticks(
    ticks=range(0, len(missing_percent.index), step),
    labels=missing_percent.index[::step],
    rotation=45
)

plt.tight_layout()
plt.savefig(
    "/Users/mahfuz/Final_project/Final_repo/Diagrams/MissingPercentage_by_Year_LBS.png", dpi=300)
plt.show()

# -------------------------------
# 2. Bar Chart: Average Missing Percentage per Reporting Country
# -------------------------------
# Assuming the column "Reporting country" exists and lists individual countries,
# compute missingness percentages per quarter grouped by Reporting country.
missing_by_country_quarter = df.groupby("Reporting country")[
    time_columns].apply(lambda x: x.isnull().mean() * 100)

# Compute the average missing percentage across all time columns for each reporting country.
average_missing_by_country = missing_by_country_quarter.mean(
    axis=1).sort_values(ascending=False)

# Plot a bar chart of the average missing percentages by reporting country.
plt.figure(figsize=(12, 6))
average_missing_by_country.plot(kind='bar', color='skyblue')
plt.title("Average Missing Percentage per Reporting Country (LBS)")
plt.xlabel("Reporting Country")
plt.ylabel("Average Missing Percentage Across Quarters")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("/Users/mahfuz/Final_project/Final_repo/Diagrams/MissingPercentage_by_ReportingCountry_LBS.png", dpi=300)
plt.show()
