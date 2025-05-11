import numpy as np
import pandas as pd
data = {
    'L_REP_CTY': ['AT', 'AT', 'AT', 'AT', 'AT', 'AT', 'AT', 'AT'],
    'L_CP_COUNTRY': ['AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD', 'AD'],
    # two groups: F and X
    'CBS_BASIS': ['F', 'F', 'F', 'F', 'X', 'X', 'X', 'X'],
    'CBS_BANK_TYPE': ['4B', '4R', '4C', '4O', '4B', '4R', '4C', '4O'],
    '2005-Q1': [100, 100, 25, 10, 120, 100, 30, 15],
    '2005-Q2': [200, 200, 0, 0, 240, 200, 60, 30]
}
df_sample = pd.DataFrame(data)
filtered_data_positon = df_sample.copy()
df = pd.DataFrame(filtered_data_positon)

df_filtered = df[df['CBS_BANK_TYPE'].isin(['4B', '4R', '4C', '4O'])].copy()
quarter_cols = [col for col in df.columns if '-Q' in col]

# Clean country codes
df_filtered['L_REP_CTY'] = df_filtered['L_REP_CTY'].str.strip().str.upper()
df_filtered['L_CP_COUNTRY'] = df_filtered['L_CP_COUNTRY'].str.strip().str.upper()

# Pivot the data
df_pivot = df_filtered.pivot_table(
    index=['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS'],
    columns='CBS_BANK_TYPE',
    values=quarter_cols,
    aggfunc='sum'
)
df_pivot = df_pivot.fillna(0)

# Extract each bank type's values
df_4B = df_pivot.xs('4B', axis=1, level='CBS_BANK_TYPE')
df_4R = df_pivot.xs('4R', axis=1, level='CBS_BANK_TYPE')
df_4C = df_pivot.xs('4C', axis=1, level='CBS_BANK_TYPE')
df_4O = df_pivot.xs('4O', axis=1, level='CBS_BANK_TYPE')

# Build the consolidated DataFrame using conditional logic:
consolidated = pd.DataFrame(index=df_4B.index)
for col in quarter_cols:
    foreign_total = df_4C[col] + df_4O[col]
    domestic_diff = df_4B[col] - df_4R[col]

    # If foreign_total is 0, use df_4B[col] directly; otherwise, use (4B-4R) + (4C+4O)
    consolidated[col] = np.where(
        foreign_total == 0, df_4B[col], domestic_diff + foreign_total)

consolidated = consolidated.reset_index()
print("Consolidated Edgeweight with conditional logic:")
print(consolidated)
# Assuming the two groups are ordered, retrieve each:
group_F = consolidated[consolidated['CBS_BASIS'] == 'F'].iloc[0]
group_X = consolidated[consolidated['CBS_BASIS'] == 'X'].iloc[0]

# Test Group F values
assert group_F['2005-Q1'] == 35, f"Group F, 2005-Q1 expected 35, got {group_F['2005-Q1']}"
assert group_F['2005-Q2'] == 200, f"Group F, 2005-Q2 expected 200, got {group_F['2005-Q2']}"

# Test Group X values
assert group_X['2005-Q1'] == 65, f"Group X, 2005-Q1 expected 65, got {group_X['2005-Q1']}"
assert group_X['2005-Q2'] == 130, f"Group X, 2005-Q2 expected 130, got {group_X['2005-Q2']}"

print("All tests passed!")
