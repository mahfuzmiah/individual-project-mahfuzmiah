import pandas as pd
import re


def filter_and_aggregate_cbs_data(input_csv, output_csv=None):

    df = pd.read_csv(input_csv)

    # Find the quarter columns to sum up later, will use regex, this is from GPT
    #    Adjust the regex or logic as needed for your actual column names.
    time_cols = [col for col in df.columns if re.match(r'^\d{4}-Q\d$', col)]
    print("Detected time columns:", time_cols)

    # Filter the DataFrame on core dimensions
    df_filtered = df[
        (df['FREQ'] == 'Q') &
        (df['CURR_TYPE_BOOK'] == 'TO1') &
        (df['L_MEASURE'] == 'S') &
        (df['CBS_BASIS'] == 'F') &
        (df['L_POSITION'] == 'C') &
        (df['L_INSTR'] == 'A') &
        (df['REM_MATURITY'] == 'A') &
        (df['L_CP_SECTOR'] == 'A')
    ]

    # Filter to valid bank types (Not 4M as theres only 19 of these)
    valid_bank_types = ['4B', '4C', '4O', '4R', '4D', '4E']
    df_filtered = df_filtered[df_filtered['CBS_BANK_TYPE'].isin(
        valid_bank_types)]

    #   sum across the bank types for each (reporter, counterparty) only want this to build laplacian matrix
    id_cols = [
        'Reporting country', 'Counterparty country']
    id_cols = [c for c in id_cols if c in df_filtered.columns]

    # Group by these ID columns and sum only the quarter columns.
    df_summed = (
        df_filtered
        .groupby(id_cols, as_index=False)[time_cols]
        .sum(numeric_only=True)
    )

    if output_csv is not None:
        df_summed.to_csv(output_csv, index=False)

    return df_summed


if __name__ == "__main__":
    # Example usage
    input_file = "/Users/mahfuz/Final_project/Final_repo/DataSets/CleanedCBSDataSet.csv"
    output_file = "cbs_data_filtered_wide.csv"

    aggregated_data = filter_and_aggregate_cbs_data(
        input_file, output_file)
    print("Aggregated data shape:", aggregated_data.shape)
