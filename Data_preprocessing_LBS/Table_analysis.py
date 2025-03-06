
import pandas as pd

# Define input file path
input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedLBSdataset.csv"

# Load dataset
data = pd.read_csv(input_file)
columns_of_interest = ['L_MEASURE', 'L_POSITION', 'L_INSTR', 'L_DENOM', 'L_CURR_TYPE',
                       'L_PARENT_CTY', 'L_REP_BANK_TYPE', 'Reporting country',
                       'L_CP_SECTOR', 'L_CP_COUNTRY', 'L_POS_TYPE']
df = pd.DataFrame(data)

# Define output file path for the counts
output_file = "/Users/mahfuz/Final_project/Final_repo/Datasets/Output_countsLBS.txt"

with open(output_file, "w") as f:
    for col in columns_of_interest:
        if col in df.columns:
            f.write(f"Counts for {col}:\n")
            counts_str = df[col].value_counts(dropna=False).to_string()
            f.write(counts_str)
            f.write("\n\n")
        else:
            f.write(f"Column {col} not found in the DataFrame.\n\n")

print(f"Output saved to {output_file}")
