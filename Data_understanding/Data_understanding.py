

#!/usr/bin/env python
"""
Plot average quarterly values overall and by bank type, saving outputs to diagrams.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
# ─── Setup module path to import config.py ───────────────────────────────────
# this file assumed in Data_preparation/ or a sibling under repo root
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from config import IMPUTED_RESULTS_DIR_TRAIN, DIAGRAMS_DIR  # nopep8

# ─── Paths ──────────────────────────────────────────────────────────────────
INPUT_FILE = IMPUTED_RESULTS_DIR_TRAIN / "ffill.csv"
OUTPUT_DIR = DIAGRAMS_DIR / "AverageQuarterly"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Load Data ──────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_FILE)
# identify quarter columns by pattern
quarter_cols = [c for c in df.columns if re.match(r"^\d{4}-Q[1-4]$", c)]

# ─── 1) Overall average per quarter ─────────────────────────────────────────
avg_overall = df[quarter_cols].apply(pd.to_numeric, errors='coerce').mean()
# convert period index for plotting
idx = pd.PeriodIndex(avg_overall.index, freq='Q').to_timestamp()

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(idx, avg_overall.values, marker='o', linestyle='-')
ax.set_xlabel('Date')
ax.set_ylabel('Average Value')
ax.set_title('Average Quarterly Values Over Time (All)')
ax.grid(True)
outfile = OUTPUT_DIR / 'average_overall.png'
fig.savefig(outfile, dpi=300)
plt.close(fig)

# ─── 2) Average per quarter by bank type ─────────────────────────────────────
bank_types = df['CBS_BASIS'].dropna().unique()
fig, ax = plt.subplots(figsize=(12, 6))
for bt in bank_types:
    sub = df[df['CBS_BASIS'] == bt][quarter_cols]
    avg_bt = sub.apply(pd.to_numeric, errors='coerce').mean()
    idx_bt = pd.PeriodIndex(avg_bt.index, freq='Q').to_timestamp()
    ax.plot(idx_bt, avg_bt.values, marker='o',
            linestyle='-', label=f'Basis {bt}')
ax.set_xlabel('Date')
ax.set_ylabel('Average Value')
ax.set_title('Average Quarterly Values by Bank Basis')
ax.legend(title='Bank Basis')
ax.grid(True)
outfile2 = OUTPUT_DIR / 'average_by_bank_basis.png'
fig.savefig(outfile2, dpi=300)
plt.close(fig)

print(f"Saved overall average plot to: {outfile}")
print(f"Saved bank-basis average plot to: {outfile2}")
