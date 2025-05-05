

# #!/usr/bin/env python3
# import os
# import re
# import numpy as np
# import pandas as pd
# from sklearn.impute import KNNImputer
# from sklearn.metrics import mean_absolute_error, mean_squared_error
# from scipy.stats import ks_2samp, wasserstein_distance

# # ───── CONFIG ────────────────────────────────────────────────────────────────
# DATA_DIR = "DataSetsCBS"
# TRAIN_FILE = os.path.join(DATA_DIR, "TrainingData.csv")
# TEST_FILE = os.path.join(DATA_DIR, "TestingData.csv")
# IMPUTED_DIR = "imputed_results"
# MASK_FRACTION = 0.10
# DATE_PATTERN = r"^\d{4}-Q[1-4]$"
# ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
# # ───────────────────────────────────────────────────────────────────────────────


# def get_quarter_cols(df):
#     return [c for c in df.columns if re.match(DATE_PATTERN, c)]


# # — Imputers on DataFrame ——————————————————————————————————————————————
# def impute_with_zeros_df(df):
#     return df.fillna(0)


# def impute_forward_fill_df(df):
#     out = df.copy()
#     num = out.select_dtypes(include="number").columns
#     out[num] = out[num].ffill(axis=1).bfill(axis=1)
#     return out


# def impute_linear_interpolation_df(df):
#     out = df.copy()
#     num = out.select_dtypes(include="number").columns
#     out[num] = out[num].interpolate(
#         method="linear", axis=1, limit_direction="both")
#     return out.fillna(0)


# def impute_polynomial_interpolation_df(df, order=3):
#     out = df.copy()
#     quarters = get_quarter_cols(df)
#     for idx in out.index:
#         row = out.loc[idx, quarters].astype(float)
#         mask = row.isna()
#         if mask.sum() and mask.sum() < len(row):
#             x = np.arange(len(row), dtype=float)
#             vx = x[~mask]
#             vy = row[~mask].values.astype(float)
#             if len(vy) >= order+1:
#                 try:
#                     coeffs = np.polyfit(vx, vy, order)
#                     row[mask] = np.poly1d(coeffs)(x[mask])
#                 except np.linalg.LinAlgError:
#                     row = row.interpolate(
#                         method="linear", limit_direction="both")
#             else:
#                 row = row.interpolate(method="linear", limit_direction="both")
#         out.loc[idx, quarters] = row
#     return out.fillna(0)


# def impute_knn_df(df, n_neighbors=5, scale=True, weights='uniform'):
#     quarters = get_quarter_cols(df)
#     ids = df[ID_COLS]
#     X = df[quarters].astype(float).replace([np.inf, -np.inf], np.nan).values
#     means = np.nanmean(X, axis=1, keepdims=True)
#     stds = np.nanstd(X, axis=1, keepdims=True)
#     stds[stds == 0] = 1.0
#     Xs = (X-means)/stds if scale else X
#     Xi = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric="nan_euclidean")\
#         .fit_transform(Xs)
#     Xi = (Xi*stds+means) if scale else Xi
#     out = pd.DataFrame(Xi, columns=quarters, index=df.index)
#     return pd.concat([ids.reset_index(drop=True), out.reset_index(drop=True)], axis=1)


# # — Core eval pipeline ——————————————————————————————————————————————
# def evaluate_on(file_path, output_metrics):
#     # load & mask
#     orig = pd.read_csv(file_path)
#     quarters = get_quarter_cols(orig)
#     mask = orig[quarters].notna().values
#     pos = np.argwhere(mask)
#     n_mask = int(len(pos)*MASK_FRACTION)
#     sel_idx = np.random.choice(len(pos), n_mask, replace=False)
#     sel = pos[sel_idx]
#     masked = orig.copy()
#     for i, j in sel:
#         masked.at[i, quarters[j]] = np.nan

#     # run and save each imputer
#     os.makedirs(IMPUTED_DIR, exist_ok=True)
#     methods = {
#         "zeros":  impute_with_zeros_df,
#         "ffill":  impute_forward_fill_df,
#         "linear": impute_linear_interpolation_df,
#         "poly": lambda df: impute_polynomial_interpolation_df(df, order=3),
#         "knn": lambda df: impute_knn_df(df, n_neighbors=7, weights="distance"),
#     }
#     for name, func in methods.items():
#         out = func(masked)
#         out.to_csv(os.path.join(IMPUTED_DIR, f"{name}.csv"), index=False)

#     # collect metrics
#     true_vals = {(i, j): orig.at[i, quarters[j]] for i, j in sel}
#     records = []
#     for fname in os.listdir(IMPUTED_DIR):
#         if not fname.endswith(".csv"):
#             continue
#         method = fname[:-4]
#         imp = pd.read_csv(os.path.join(IMPUTED_DIR, fname))
#         imp[quarters] = imp[quarters].replace([np.inf, -np.inf], np.nan)

#         preds, trues = [], []
#         for i, j in sel:
#             p = imp.at[i, quarters[j]]
#             t = true_vals[(i, j)]
#             if pd.notna(p) and pd.notna(t):
#                 preds.append(p)
#                 trues.append(t)
#         if not preds:
#             continue

#         mae = mean_absolute_error(trues, preds)
#         rmse = np.sqrt(mean_squared_error(trues, preds))
#         ov = orig[quarters].values.ravel()
#         iv = imp[quarters].values.ravel()
#         ov = ov[np.isfinite(ov)]
#         iv = iv[np.isfinite(iv)]
#         ks = ks_2samp(ov, iv)[0]
#         wd = wasserstein_distance(ov, iv)

#         records.append({
#             "method": method,
#             "MAE":    mae,
#             "RMSE":   rmse,
#             "KS":     ks,
#             "Wasserstein": wd
#         })
#         # Print which method has been done
#         print(f"→ {method} done")
#     pd.DataFrame(records).to_csv(output_metrics, index=False)
#     print(f"→ wrote {output_metrics}")


# if __name__ == "__main__":
#     evaluate_on(TRAIN_FILE, "imputation_evaluation_metrics_train.csv")
#     evaluate_on(TEST_FILE,  "imputation_evaluation_metrics_test.csv")


#!/usr/bin/env python3
import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ks_2samp, wasserstein_distance

# ───── CONFIG ────────────────────────────────────────────────────────────────
DATA_DIR = "DataSetsCBS"
TRAIN_FILE = os.path.join(DATA_DIR, "TrainingData.csv")
TEST_FILE = os.path.join(DATA_DIR, "TestingData.csv")
IMPUTED_DIR = "imputed_results"
MASK_FRACTION = 0.10
DATE_PATTERN = r"^\d{4}-Q[1-4]$"
ID_COLS = ['L_REP_CTY', 'L_CP_COUNTRY', 'CBS_BASIS']
# ───────────────────────────────────────────────────────────────────────────────


def get_quarter_cols(df):
    return [c for c in df.columns if re.match(DATE_PATTERN, c)]


# — Imputers on DataFrame ——————————————————————————————————————————————
def impute_with_zeros_df(df):
    return df.fillna(0)


def impute_forward_fill_df(df):
    out = df.copy()
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].ffill(axis=1).bfill(axis=1)
    return out


def impute_linear_interpolation_df(df):
    out = df.copy()
    num = out.select_dtypes(include="number").columns
    out[num] = out[num].interpolate(
        method="linear", axis=1, limit_direction="both")
    return out.fillna(0)


def impute_polynomial_interpolation_df(df, order=3):
    out = df.copy()
    quarters = get_quarter_cols(df)
    for idx in out.index:
        row = out.loc[idx, quarters].astype(float)
        mask = row.isna()
        if mask.sum() and mask.sum() < len(row):
            x = np.arange(len(row), dtype=float)
            vx = x[~mask]
            vy = row[~mask].values.astype(float)
            if len(vy) >= order + 1:
                try:
                    coeffs = np.polyfit(vx, vy, order)
                    row[mask] = np.poly1d(coeffs)(x[mask])
                except np.linalg.LinAlgError:
                    row = row.interpolate(
                        method="linear", limit_direction="both")
            else:
                row = row.interpolate(method="linear", limit_direction="both")
        out.loc[idx, quarters] = row
    return out.fillna(0)


def impute_knn_df(df, n_neighbors=5, scale=True, weights='uniform'):
    quarters = get_quarter_cols(df)
    ids = df[ID_COLS]
    # build matrix
    X = df[quarters].astype(float).replace([np.inf, -np.inf], np.nan).values
    means = np.nanmean(X, axis=1, keepdims=True)
    stds = np.nanstd(X, axis=1, keepdims=True)
    stds[stds == 0] = 1.0
    Xs = (X - means) / stds if scale else X
    Xi = KNNImputer(n_neighbors=n_neighbors, weights=weights, metric="nan_euclidean")\
        .fit_transform(Xs)
    Xi = (Xi * stds + means) if scale else Xi
    out = pd.DataFrame(Xi, columns=quarters, index=df.index)
    return pd.concat([ids.reset_index(drop=True),
                      out.reset_index(drop=True)], axis=1)


# — Core eval pipeline ——————————————————————————————————————————————
def evaluate_on(file_path, output_metrics):
    # load & mask
    orig = pd.read_csv(file_path)
    quarters = get_quarter_cols(orig)
    mask_mat = orig[quarters].notna().values
    positions = np.argwhere(mask_mat)
    n_mask = int(len(positions) * MASK_FRACTION)
    sel_idx = np.random.choice(len(positions), n_mask, replace=False)
    sel = positions[sel_idx]
    masked = orig.copy()
    for i, j in sel:
        masked.at[i, quarters[j]] = np.nan

    # ensure output dirs
    os.makedirs(IMPUTED_DIR, exist_ok=True)

    # define methods
    methods = {
        "zeros":  impute_with_zeros_df,
        "ffill":  impute_forward_fill_df,
        "linear": impute_linear_interpolation_df,
        "poly": lambda df: impute_polynomial_interpolation_df(df, order=3),
        "knn": lambda df: impute_knn_df(df, n_neighbors=7, weights="distance"),
    }

    # run imputers, time each, save CSVs
    timings = {}
    for name, func in methods.items():
        t0 = time.time()
        out = func(masked)
        elapsed = time.time() - t0
        timings[name] = elapsed
        out.to_csv(os.path.join(IMPUTED_DIR, f"{name}.csv"), index=False)
        print(f"→ {name}: {elapsed:.2f}s")

    # collect metrics
    true_vals = {(i, j): orig.at[i, quarters[j]] for i, j in sel}
    records = []
    for fname in os.listdir(IMPUTED_DIR):
        if not fname.endswith(".csv"):
            continue
        method = fname[:-4]
        imp = pd.read_csv(os.path.join(IMPUTED_DIR, fname))
        imp[quarters] = imp[quarters].replace([np.inf, -np.inf], np.nan)

        preds, trues = [], []
        for i, j in sel:
            p = imp.at[i, quarters[j]]
            t = true_vals[(i, j)]
            if pd.notna(p) and pd.notna(t):
                preds.append(p)
                trues.append(t)
        if not preds:
            continue

        mae = mean_absolute_error(trues, preds)
        rmse = np.sqrt(mean_squared_error(trues, preds))
        ov = orig[quarters].values.ravel()
        iv = imp[quarters].values.ravel()
        ov = ov[np.isfinite(ov)]
        iv = iv[np.isfinite(iv)]
        ks = ks_2samp(ov, iv)[0]
        wd = wasserstein_distance(ov, iv)
        time_s = timings.get(method, np.nan)

        records.append({
            "method":      method,
            "MAE":         mae,
            "RMSE":        rmse,
            "KS":          ks,
            "Wasserstein": wd,
            "time_sec":    time_s
        })
        # Print which method has been done
        print(f"→ {method} done")

    pd.DataFrame(records).to_csv(output_metrics, index=False)
    print(f"→ wrote {output_metrics}")


if __name__ == "__main__":
    evaluate_on(TRAIN_FILE, "imputation_evaluation_metrics_train.csv")
    evaluate_on(TEST_FILE,  "imputation_evaluation_metrics_test.csv")
