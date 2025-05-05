

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def make_ticks(date_labels, every_n_years=3):
    """Return tick positions/labels at Q2 of every_n_years."""
    years = [int(lbl[:4]) for lbl in date_labels]
    is_q2 = [lbl.endswith('-Q2') for lbl in date_labels]
    positions = [i for i, (yr, q2) in enumerate(zip(years, is_q2))
                 if q2 and (yr % every_n_years == 0)]
    labels = [date_labels[i] for i in positions]
    return positions, labels


def find_inflections(date_labels):
    """Return dict of inflection annotations only if present."""
    candidates = {
        '2000-Q4': 'Quarterly\nreporting begins',
        '2011-Q4': 'New economies\nadded'
    }
    return {q: txt for q, txt in candidates.items() if q in date_labels}


def investigation(df, title_suffix, output_dir):
    # Identify quarterly columns
    time_cols = [c for c in df.columns if re.match(r"^\d{4}-Q[1-4]$", c)]
    date_labels = time_cols.copy()

    # 1) Total exposure time-series (log scale)
    total_by_q = df[time_cols].sum(axis=0)
    ticks, tick_labels = make_ticks(date_labels)
    inflections = find_inflections(date_labels)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(total_by_q.values, marker='o', linewidth=1)
    ax.set_yscale('log')
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=45, ha='right')
    ax.set_title(f"Total BIS CBS Cross-Border Claims{title_suffix} Over Time")
    ax.set_xlabel("Quarter (Q2 every 3 years)")
    ax.set_ylabel("Total Exposure (USD bn, log scale)")

    for q, txt in inflections.items():
        idx = date_labels.index(q)
        y_val = total_by_q.iloc[idx]
        ax.axvline(idx, linestyle='--', color='red')
        ax.annotate(
            txt,
            xy=(idx, y_val),
            xytext=(idx + 1, total_by_q.max() * 0.8),
            textcoords='data',
            arrowprops=dict(arrowstyle='->', lw=1, color='red'),
            va='center',
            ha='left',
            fontsize=9,
            color='red'
        )

    ax.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    fig.savefig(os.path.join(
        output_dir, f"total_exposure{title_suffix.strip()}.png"), dpi=300)

    plt.show()

    # 2) Exposure distribution & summary stats
    all_vals = df[time_cols].to_numpy().ravel()
    all_vals = all_vals[~np.isnan(all_vals)]

    print(f"{title_suffix} Mean:   {all_vals.mean():.2f}")
    print(f"{title_suffix} Median: {np.median(all_vals):.2f}")
    print(f"{title_suffix} Std:    {all_vals.std():.2f}")

    plt.figure(figsize=(8, 5))
    plt.hist(all_vals, bins=50, edgecolor='k', alpha=0.7)
    plt.title(f"{title_suffix} Exposure Distribution (linear)")
    plt.xlabel("Exposure (USD bn)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(
        output_dir, f"hist_linear{title_suffix.strip()}.png"), dpi=300)

    plt.show()

    pos_vals = all_vals[all_vals > 0]
    plt.figure(figsize=(8, 5))
    plt.hist(np.log1p(pos_vals), bins=50, edgecolor='k', alpha=0.7)
    plt.title(f"{title_suffix} Exposure Distribution (log)")
    plt.xlabel("log(Exposure + 1)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    fig.savefig(os.path.join(
        output_dir, f"hist_log{title_suffix.strip()}.png"), dpi=300)

    plt.show()

    # 3) Missingness breakdown
    total_cells = df.shape[0] * len(time_cols)
    missing = df[time_cols].isna().sum().sum()
    zeros = (df[time_cols] == 0).sum().sum()
    print(f"{title_suffix} Total cells: {total_cells}")
    print(f"{title_suffix} Missing:     {missing} ({missing/total_cells:.1%})")
    print(f"{title_suffix} Zeros:       {zeros} ({zeros/total_cells:.1%})")


if __name__ == "__main__":
    # Update these paths to your files:
    path_clean = "/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/CleanedCBSDataSet.csv"
    path_raw = "/Users/mahfuz/Final_project/Final_repo/DataSetsCBS/WS_CBS_PUB_csv_col.csv"

    df_clean = pd.read_csv(path_clean)
    df_raw = pd.read_csv(path_raw)

    investigation(df_clean, " (Cleaned)", output_dir="graphs/cleaned")
    investigation(df_raw,   " (Raw)",     output_dir="graphs/raw")
