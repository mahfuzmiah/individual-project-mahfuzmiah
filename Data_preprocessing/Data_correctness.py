import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedCBSDataSet.csv"
# input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/WS_CBS_PUB_csv_col.csv"
data = pd.read_csv(input_file)

df = pd.DataFrame(data)

# # Identify the quarter columns
# time_columns = [col for col in df.columns if '-Q' in col]

# # Replace all zeros with NaN in those columns
# df[time_columns] = df[time_columns].replace(0, np.nan)

# # Compute the percentage of missing values for each quarter column
# missing_percent = df[time_columns].isnull().mean() * 100

# # Print the missing percentages
# print(missing_percent.sort_values(ascending=False))

# # Create a line chart of the missing data percentages
# plt.figure(figsize=(20, 6))
# plt.plot(missing_percent.index, missing_percent.values,
#          marker='o', color='skyblue', linestyle='-')
# plt.title("Missing Data Percentage for Time Period Columns")
# plt.xlabel("Quarter")
# plt.ylabel("Percentage Missing")

# # Show every 4th label
# step = 4
# plt.xticks(
#     ticks=range(0, len(missing_percent.index), step),
#     labels=missing_percent.index[::step],
#     rotation=45  # rotate for better readability
# )

# plt.tight_layout()
# plt.savefig(
#     "/Users/mahfuz/Final_project/Final_repo/Diagrams/MissingPercentage_by_Year.png", dpi=300)
# plt.show()
# # If you want to treat zeros in time columns as missing, first identify time columns:

# # Identify the quarter columns (those containing '-Q' in their names)
# time_columns = [col for col in df.columns if '-Q' in col]

# # If you want to treat 0 as missing, replace 0 with NaN in those columns
# df[time_columns] = df[time_columns].replace(0, np.nan)

# # Calculate missing percentages for the time columns grouped by CBS_BASIS.
# # (We assume CBS_BASIS is a column whose values are one of F, O, Q, U.)
# time_missing_by_basis = df.groupby('CBS_BASIS')[time_columns].apply(
#     lambda x: x.isnull().mean() * 100)
# time_missing_by_basis = time_missing_by_basis.reset_index()

# # Melt the DataFrame from wide to long format.
# melted = pd.melt(time_missing_by_basis,
#                  id_vars='CBS_BASIS',
#                  value_vars=time_columns,
#                  var_name='Quarter',
#                  value_name='MissingPercent')

# # Extract the year from the Quarter column (assuming format like "1983-Q4")
# melted['Year'] = melted['Quarter'].str[:4].astype(int)
# melted.sort_values(['CBS_BASIS', 'Year'], inplace=True)

# # Define a color mapping for each CBS_BASIS group
# colors = {'F': 'blue', 'O': 'red', 'Q': 'gray', 'U': 'orange'}
# markers = {'F': 'o', 'O': '*', 'Q': 'p', 'U': 's'}
# alphas = {'F': 0.7, 'O': 1, 'Q': 1, 'U': 0.6}
# size = {'F': 100, 'O': 110, 'Q': 100, 'U': 100}

# # Create a line plot instead of a scatter plot.
# plt.figure(figsize=(12, 6))
# for basis, group in melted.groupby('CBS_BASIS'):
#     group_sorted = group.sort_values('Year')
#     plt.scatter(group_sorted['Year'], group_sorted['MissingPercent'],
#                 marker=markers.get(basis, None), label=basis, color=colors.get(basis, None),
#                 alpha=alphas.get(basis, 1), s=size.get(basis, 50))

# plt.xlabel('Year')
# plt.ylabel('Missing Percentage')
# plt.title(
#     'Missing Percentage by Year for each CBS_BASIS Group')
# plt.legend(title='CBS_BASIS')
# plt.grid(True)
# plt.tight_layout()

# plt.savefig(
#     "/Users/mahfuz/Final_project/Final_repo/Diagrams/MissingPercentage_by_Reporting_basis.png", dpi=300)
# plt.show()


# Identify the quarter columns (those containing '-Q' in their names)
time_columns = [col for col in df.columns if '-Q' in col]

# If you want to treat 0 as missing, replace 0 with NaN in those columns
df[time_columns] = df[time_columns].replace(0, np.nan)

# 1. Compute the missingness for each time column grouped by reporting country (L_REP_CTY)
#    This gives a missingness percentage for each quarter, per reporting country.
missing_by_country_quarter = df.groupby(
    'L_REP_CTY')[time_columns].apply(lambda x: x.isnull().mean() * 100)

# 2. Now compute the average missingness across all quarters for each reporting country.
#    i.e., the mean of the quarter-level missing percentages.
average_missing_by_country = missing_by_country_quarter.mean(
    axis=1).sort_values(ascending=False)

# 3. Plot a bar chart of the average missingness by country
plt.figure(figsize=(12, 6))
average_missing_by_country.plot(kind='bar', color='skyblue')
plt.title("Average Missing Percentage per Reporting Country (0 Treated as Missing)")
plt.xlabel("Reporting Country")
plt.ylabel("Average Missing Percentage Across Quarters")
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Save and show the plot
plt.savefig(
    "/Users/mahfuz/Final_project/Final_repo/Diagrams/MissingPercentage_by_ReportingCountry.png", dpi=300)
plt.show()
