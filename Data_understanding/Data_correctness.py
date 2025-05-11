import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
REPO_ROOT_PATH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT_PATH))
from config import REPO_ROOT, DATASETS_DIR, DIAGRAMS_DIR   # nopep8


INPUT_FILE = DATASETS_DIR / "CleanedCBSDataSet.csv"  # Original dataset
input_stem = INPUT_FILE.stem
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
DIAGRAMS_DIR = DIAGRAMS_DIR / "Data_correctness"
DIAGRAMS_DIR.mkdir(parents=True, exist_ok=True)
df = pd.read_csv(INPUT_FILE)
time_cols = [c for c in df.columns if "-Q" in c]
df[time_cols] = df[time_cols].replace(0, np.nan)
missing = df[time_cols].isnull().mean() * 100

date_labels = missing.index.tolist()
x, y = np.arange(len(date_labels)), missing.values

sns.set_theme(style="whitegrid", context="talk")
plt.figure(figsize=(16, 6))

# Main line + Q4 markers
plt.plot(x, y, color="navy", linewidth=2, label="Missing %")
plt.scatter(x[::4], y[::4], color="navy", s=60)

# Sparse‑era shading
start, end = 0, date_labels.index("1998-Q4")
plt.axvspan(start, end, color="lightgrey", alpha=0.4)
# nudge “Sparse era” label left and up a bit
plt.text((start+end)/2 - 2, 96, "Sparse era",
         color="dimgrey", ha="center",
         fontsize=12, weight="bold")
# Regime lines
for ev in ["1998-Q4", "2004-Q4", "2013-Q2"]:
    idx = date_labels.index(ev)
    plt.axvline(idx, color="dimgrey", linestyle="--", lw=4)
    plt.text(idx+0.5, 70, ev, rotation=0, va="top",
             color="black", fontsize=17)

# Gain annotations in forestgreen
# Font size of txt should be 17
drops = [("1999-Q4", 90.8, "+11 ppt"), ("2004-Q4", 71.5, "+13 ppt")]
for ev, val, txt in drops:
    idx = date_labels.index(ev)
    # shift the arrow off the bar and move the text 2 points downwards
    dx, dy = (5, +5) if ev == "1999-Q4" else (3, +4)
    plt.annotate(txt, xy=(idx, val),
                 xytext=(idx+dx, val+dy),
                 arrowprops=dict(arrowstyle="->",
                                 color="forestgreen",
                                 lw=1.2,
                                 shrinkA=5, shrinkB=2),
                 color="forestgreen", fontsize=17)
# Biannual Q4 ticks
tick_pos = [i for i, lab in enumerate(date_labels)
            if lab.endswith("Q4") and (int(lab[:4]) % 2 == 0)]
tick_labs = [date_labels[i] for i in tick_pos]
plt.xticks(ticks=tick_pos, labels=tick_labs, rotation=45)

# Labels & limits
plt.ylim(30, 100)
plt.title("CBS Panel:% of Quarter-Level Observations Missing Over Time")
plt.xlabel("Quarter")
plt.ylabel("Missing (%)")
plt.tight_layout()
diagram_type = "MissingPercentage"
outfile = DIAGRAMS_DIR / f"{diagram_type}_{input_stem}.png"
plt.savefig(outfile, dpi=300)
plt.show()

# If you want to treat zeros in time columns as missing, first identify time columns:

tcols = [c for c in df.columns if "-Q" in c]
df[tcols] = df[tcols].replace(0, np.nan)

bm = (df.groupby("CBS_BASIS")[tcols]
        .apply(lambda x: x.isna().mean() * 100)
        .reset_index()
        .melt(id_vars="CBS_BASIS", value_vars=tcols,
              var_name="Quarter", value_name="Pct")
      )
bm["Year"] = bm["Quarter"].str[:4].astype(int)

sns.set_theme(style="whitegrid", context="talk")
fig, ax = plt.subplots(figsize=(12, 5))

colmap = {"F": "navy", "O": "firebrick", "Q": "dimgray", "U": "goldenrod"}
for b, g in bm.groupby("CBS_BASIS"):
    ax.plot(g["Year"], g["Pct"],
            label=b, color=colmap[b], linewidth=2, alpha=0.8)

# sparse era
ax.axvspan(1983, 1999, color="lightgray", alpha=0.15)
ax.text(1991, 95, "Sparse era", ha="center", va="top", color="black")


# axes/grid/legend
ax.set_xlim(1982, 2024)
ax.set_ylim(28, 100)
ax.set_xticks(range(1985, 2026, 5))
ax.set_xlabel("Year")
ax.set_ylabel("Missing %")
ax.set_title("CBS Panel: Quarterly Observations Missing by Basis")
ax.xaxis.grid(False)
ax.yaxis.grid(True, color="0.85")
ax.legend(title="CBS_BASIS", loc="upper left",
          bbox_to_anchor=(1.06, 1.0),   # <-- moved from 1.02 to 1.06
          borderaxespad=0)
# draw the 1998 Q4, 2005 Q4 and 2013 Q4 markers
for yr, lbl in [(1999, "1998 Q4"), (2005, "2005 Q4"), (2013, "2013 Q2")]:
    ax.axvline(yr, color="black", linestyle=":", linewidth=1.5)
    # put the text a bit above the 2013 line so it doesn’t overlap the curves
    ax.text(
        yr+1.4,                    # x at the year
        # y–position (tweak this if your curves go above 82)
        82,
        lbl,                   # the label to draw
        rotation=0,
        va="bottom",           # text bottom at y=82
        ha="center",
        fontsize=10,
        color="black"
    )

plt.tight_layout()
diagram_type = "MissingPercentage_by_Basis"
outfile = DIAGRAMS_DIR / f"{diagram_type}_{input_stem}.png"
plt.savefig(outfile, dpi=300)
plt.show()

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
plt.title("Average Missing Percentage per Reporting Country")
plt.xlabel("Reporting Country")
plt.ylabel("Average Missing Percentage Across Quarters")
plt.xticks(rotation=45)
plt.tight_layout()

# 4. Save and show the plot
diagram_type = "MissingPercentage_by_Country"
outfile = DIAGRAMS_DIR / f"{diagram_type}_{input_stem}.png"
plt.savefig(outfile, dpi=300)
plt.show()
