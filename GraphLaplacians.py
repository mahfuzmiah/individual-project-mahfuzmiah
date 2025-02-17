import pandas as pd
import re
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import pickle


def build_dual_laplacians(A_out):
    """
    Given a directed adjacency matrix A_out (shape [N, N]),
    return (L_out, L_in).

    A_in = A_out^T
    D_out = diag of row-sums of A_out
    D_in  = diag of row-sums of A_in
    L_out = D_out - A_out
    L_in  = D_in - A_in
    """
    A_in = A_out.T  # shape [N, N]

    # Row-sums => out-degrees
    out_degrees = A_out.sum(axis=1)
    D_out = np.diag(out_degrees)

    # Row-sums of A_in => in-degrees
    in_degrees = A_in.sum(axis=1)
    D_in = np.diag(in_degrees)

    L_out = D_out - A_out
    L_in = D_in - A_in

    return L_out, L_in


def build_adjacency_dict_from_wide(df, reporting_col='Reporting country', counterparty_col='Counterparty country'):
    # 1. Identify all unique countries (reporter + counterparty)
    reporters = df[reporting_col].unique()
    counterparties = df[counterparty_col].unique()
    all_countries = sorted(set(reporters).union(set(counterparties)))

    # Create a mapping country -> index
    country2idx = {c: i for i, c in enumerate(all_countries)}
    N = len(all_countries)

    # 2. Find all columns that match the pattern "YYYY-Qn"
    time_cols = [col for col in df.columns if re.match(r'^\d{4}-Q\d$', col)]

    # Prepare a dictionary for adjacency matrices
    A_out_dict = {}

    # 3. For each time column, create an NxN matrix
    for tcol in time_cols:
        # Initialize adjacency to zero
        A_out = np.zeros((N, N), dtype=float)

        # Fill adjacency for each row in the DataFrame
        for idx, row in df.iterrows():
            # i: index for reporting country
            i = country2idx[row[reporting_col]]
            # j: index for counterparty country
            j = country2idx[row[counterparty_col]]
            # exposure value in that quarter
            val = row[tcol]
            if pd.notna(val):
                A_out[i, j] = val

        A_out_dict[tcol] = A_out

    return A_out_dict, all_countries


if __name__ == "__main__":
    # 1. Load your wide dataset from CSV
    csv_path = "/Users/mahfuz/Final_project/Final_repo/DataSets/TestLaplacian.Csv"
    df = pd.read_csv(csv_path)

    # 2. Build adjacency dict
    A_out_dict, country_list = build_adjacency_dict_from_wide(df)

    print("Number of countries:", len(country_list))
    print("Time columns found:", list(A_out_dict.keys())[:10])  # show first 10

    # 3. Pick a single quarter, e.g. "2000-Q1"
    example_quarter = "2000-Q1"
    if example_quarter in A_out_dict:
        A_out_example = A_out_dict[example_quarter]
        print(f"\nAdjacency for {example_quarter}:\n", A_out_example)

        # 4. Compute the dual Laplacians for that quarter
        L_out_example, L_in_example = build_dual_laplacians(A_out_example)

        # 5. Store them in a dictionary with your chosen keys (e.g. "2000-Q1Out"/"2000-Q1In")
        laplacians_dict = {}
        laplacians_dict[f"{example_quarter}Out"] = L_out_example
        laplacians_dict[f"{example_quarter}In"] = L_in_example

        # Show them on screen
        print(f"\nL_out for {example_quarter}:\n", L_out_example)
        print(f"\nL_in for {example_quarter}:\n", L_in_example)

        # 6. (Optional) Save to file for later use
        #    Example: save as a pickle
        with open("laplacians_2000Q1.pkl", "wb") as f:
            pickle.dump(laplacians_dict, f)
        print("\nSaved 2000-Q1 Laplacians dictionary to laplacians_2000Q1.pkl.")

    else:
        print(f"Quarter '{example_quarter}' not found in the dictionary.")

    with open("laplacians_2000Q1.pkl", "rb") as f:
        loaded_laplacians = pickle.load(f)

    print("Keys in the loaded dictionary:", loaded_laplacians.keys())

    # For example, if you stored them as "2000-Q1Out" and "2000-Q1In":
    L_out_2000Q1 = loaded_laplacians["2000-Q1Out"]
    L_in_2000Q1 = loaded_laplacians["2000-Q1In"]
