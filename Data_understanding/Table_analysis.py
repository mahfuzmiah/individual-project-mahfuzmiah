

import pandas as pd
import os
import re
import matplotlib.pyplot as plt
import sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from config import (
    RAW_CSV,
    CLEANED_CSV,
    UNIQUE_COUNTS_CSV,
    UNIQUE_COUNTS_CLEAN_CSV,
    COUNTRY_COUNTS_CSV,
    COUNTRY_COUNTS_CLEAN_CSV,
    GRAPHS_DIR,
)  # nopep8
# step up from this file into your repo root


# now import the paths you defined


def get_unique_values_summary(input_file, output_file):
    # Define input file path

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
            if len(unique_values) < 1000:  # Adjust threshold if needed
                unique_values_summary[col] = unique_values[:250]
            else:
                unique_values_summary[col] = f"{len(unique_values)} unique values (too many to display)"

    # Convert to DataFrame for better readability
    unique_values_df = pd.DataFrame(list(unique_values_summary.items()), columns=[
                                    "Column", "Example Unique Values"])

    # Save to a file (optional)
    unique_values_df.to_csv(output_file, index=False)

    print(f"Unique values per column summary saved to {output_file}.")


def get_unique_values_summary_with_counts(input_file, output_file):
    # Load dataset
    df = pd.read_csv(input_file)
    df.columns = df.columns.str.strip()

    # Total distinct Reporter–Counterparty pairs (rows)
    total_pairs = df.shape[0]
    print(f"Total distinct Reporter–Counterparty pairs: {total_pairs:,}")

    # Prepare records for summary
    records = []
    # Identify quarter columns via regex
    quarter_pattern = re.compile(r"^(?P<year>\d{4})-Q[1-4]$")

    # Compute per-column stats
    for col in df.columns:
        non_null = df[col].count()
        unique = df[col].nunique(dropna=True)
        vc = df[col].value_counts(dropna=True)
        top_vals = vc.to_dict() if unique <= 20 else vc.head(10).to_dict()

        # Initialize pair count for quarters only
        num_pairs = ''
        m = quarter_pattern.match(col)
        if m:
            mask = df[col].notna()
            pairs = df.loc[mask, ['L_REP_CTY',
                                  'L_CP_COUNTRY']].drop_duplicates()
            num_pairs = len(pairs)

        records.append({
            'Column': col,
            'Non-Null Count': non_null,
            'Unique Count': unique,
            'Top Values & Counts': top_vals,
            'NumReporters': '',
            'NumCounterparties': '',
            'NumPairs': num_pairs
        })

    summary_df = pd.DataFrame(records)

    # Compute reporters & counterparties counts for quarter columns
    quarters = summary_df['Column'].str.match(quarter_pattern)
    quarter_cols = summary_df.loc[quarters, 'Column']
    rep_counts = {q: df.loc[df[q].notna(), 'L_REP_CTY'].nunique()
                  for q in quarter_cols}
    cp_counts = {q: df.loc[df[q].notna(), 'L_CP_COUNTRY'].nunique()
                 for q in quarter_cols}

    # Map counts into summary
    summary_df.loc[summary_df['Column'].isin(
        rep_counts), 'NumReporters'] = summary_df['Column'].map(rep_counts)
    summary_df.loc[summary_df['Column'].isin(
        cp_counts), 'NumCounterparties'] = summary_df['Column'].map(cp_counts)

    # Total filled datapoints across all quarters
    total_points = int(summary_df.loc[quarters, 'Non-Null Count'].sum())
    print(f"Total filled datapoints across all quarters: {total_points:,}")

    # Append total row
    total_row = {
        'Column': 'Total Filled Datapoints',
        'Non-Null Count': total_points,
        'Unique Count': '',
        'Top Values & Counts': '',
        'NumReporters': '',
        'NumCounterparties': '',
        'NumPairs': ''
    }
    summary_df = pd.concat(
        [summary_df, pd.DataFrame([total_row])], ignore_index=True)

    # Persist summary
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    summary_df.to_csv(output_file, index=False)
    print(f"Saved summary (including unique pairs) to: {output_file}")

    return summary_df


def draw_Unique_graphs(input_file, output_dir="graphs"):
    # detect version label from filename
    version = os.path.splitext(os.path.basename(input_file))[0]
    os.makedirs(output_dir, exist_ok=True)

    # Load & prep
    df = pd.read_csv(input_file)
    quarter_df = df[df['Column'].str.match(r'^\d{4}-Q[1-4]$')].copy()
    quarter_df['Year'] = quarter_df['Column'].str[:4].astype(int)
    quarter_df['Qnum'] = quarter_df['Column'].str[-1].astype(int)
    quarter_df = quarter_df.sort_values(
        ['Year', 'Qnum']).reset_index(drop=True)

    # pick Q4s every 3 years
    is_q4 = quarter_df['Column'].str.endswith('-Q4')
    mask = is_q4 & (quarter_df['Year'] % 2 == 0)
    ticks = quarter_df.index[mask].tolist()
    labels = quarter_df.loc[mask, 'Column'].tolist()
    # 4) key inflection annotations
    inflections = {
        '2000-Q4': 'Quarterly\nreporting begins',
        '2014-Q4': 'New economies\nadded'
    }

    def plot_series(y, ylabel, title, fname):
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(quarter_df.index, y, marker='o')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_xlabel('Quarter (Q4 every 3 years)')
        ax.set_ylabel(ylabel)
        for quarter_label, annotation in inflections.items():
            if quarter_label in quarter_df['Column'].values:
                idx = quarter_df.index[quarter_df['Column']
                                       == quarter_label][0]
                y_val = y.iloc[idx]
                ax.annotate(
                    annotation,
                    xy=(idx, y_val),
                    xytext=(idx+1, y.max()*0.6),
                    arrowprops=dict(arrowstyle='->', lw=1),
                    va='center',
                    ha='left',
                    fontsize=8
                )
                # bold the first/last ticks…
        for lbl in ax.get_xticklabels():
            if lbl.get_text() in (labels[0], labels[-1]):
                lbl.set_fontweight('bold')
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, fname), dpi=300)
        plt.close(fig)

    plot_series(
        quarter_df['NumReporters'],
        'Unique Reporting Countries',
        'Unique Reporting Countries per Quarter',
        f'reporting_countries_{version}.png'
    )

    plot_series(
        quarter_df['NumCounterparties'],
        'Unique Counterparty Countries',
        'Unique Counterparty Countries per Quarter',
        f'counterparty_countries_{version}.png'
    )

    plot_series(
        quarter_df['NumPairs'],
        'Unique Reporter–Counterparty Pairs',
        'Unique Reporter–Counterparty Pairs per Quarter',
        f'reporter_counterparty_pairs_{version}.png'
    )


if __name__ == "__main__":
    # Define input file paths from config
    input_unclean = RAW_CSV
    input_clean = CLEANED_CSV

    # Define output summary paths from config
    out_unclean_summary = UNIQUE_COUNTS_CSV
    out_unclean_pair_counts = COUNTRY_COUNTS_CSV
    out_clean_summary = UNIQUE_COUNTS_CLEAN_CSV
    out_clean_pair_counts = COUNTRY_COUNTS_CLEAN_CSV

    # Graph sub-folders
    unclean_graph_dir = GRAPHS_DIR / "unclean"
    clean_graph_dir = GRAPHS_DIR / "cleaned"

    # ---- RAW data ----
    # 1) Unique-values list
    get_unique_values_summary(
        input_unclean,
        out_unclean_summary
    )
    # 2) Unique-values WITH counts
    get_unique_values_summary_with_counts(
        input_unclean,
        out_unclean_pair_counts
    )
    # 3) Draw graphs from the counts file
    draw_Unique_graphs(
        out_unclean_pair_counts,
        output_dir=unclean_graph_dir
    )

    # ---- CLEANED data ----
    get_unique_values_summary(
        input_clean,
        out_clean_summary
    )
    get_unique_values_summary_with_counts(
        input_clean,
        out_clean_pair_counts
    )
    draw_Unique_graphs(
        out_clean_pair_counts,
        output_dir=clean_graph_dir
    )
