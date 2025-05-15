import os
import re
import time
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import ks_2samp, wasserstein_distance
import sys
from typing import Sequence
from pathlib import Path
# so repo root is three levels up:
REPO_ROOT_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT_PATH))
from config import REPO_ROOT, DATASETS_DIR  # nopep8
np.random.seed(42)
# now we can import our config

# ───── CONFIG ────────────────────────────────────────────────────────────────

TRAIN_FILE = DATASETS_DIR / "TrainingData.csv"
TEST_FILE = DATASETS_DIR / "TestingData.csv"
IMPUTED_DIR = REPO_ROOT / "imputed_results"
IMPUTED_DIR = REPO_ROOT / "imputed_results"
METRICS_TRAIN = IMPUTED_DIR / "metrics_train.csv"
METRICS_TEST = IMPUTED_DIR / "metrics_test.csv"
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
    """
    Linear interpolate each row across quarters, then
    forward‐fill/backward‐fill to handle any edge NaNs.
    """
    out = df.copy()
    quarters = get_quarter_cols(out)

    # 1) linear interpolate between known points
    out[quarters] = out[quarters].interpolate(
        method="linear",
        axis=1,
        limit_direction="both"
    )

    # 2) for any remaining NaNs at the very start or end, carry the nearest value
    out[quarters] = out[quarters].ffill(axis=1).bfill(axis=1)

    return out


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
def evaluate_on(file_path: Path, output_metrics: Path, out_dir: Path, knn_values: Sequence[int] = (3, 5, 7, 10)):
    # load & mask
    os.makedirs(out_dir, exist_ok=True)

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
    out_dir.mkdir(parents=True, exist_ok=True)
    # define methods
    methods = {
        "zeros":  impute_with_zeros_df,
        "ffill":  impute_forward_fill_df,
        "linear": impute_linear_interpolation_df,
        "poly": lambda df: impute_polynomial_interpolation_df(df, order=3),
    }
    for k in knn_values:
        methods[f"knn_{k}"] = lambda df, k=k: impute_knn_df(
            df, n_neighbors=k, weights="distance"
        )

    # run imputers, time each, save CSVs
    timings = {}
    for name, func in methods.items():
        t0 = time.time()
        out = func(masked)
        elapsed = time.time() - t0
        timings[name] = elapsed
        out.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"→ {name}: {elapsed:.2f}s")

    # collect metrics
    true_vals = {(i, j): orig.at[i, quarters[j]] for i, j in sel}
    records = []
    for fname in os.listdir(out_dir):
        if not fname.endswith(".csv"):
            continue
        method = fname[:-4]
        imp = pd.read_csv(os.path.join(out_dir, fname))
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
        k = int(method.split("_")[1]) if method.startswith("knn_") else None

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

    output_metrics.parent.mkdir(exist_ok=True)
    pd.DataFrame(records).to_csv(output_metrics, index=False)
    print(f"→ wrote {output_metrics}")


if __name__ == "__main__":
    evaluate_on(
        TRAIN_FILE,
        METRICS_TRAIN,
        IMPUTED_DIR / "train",
        knn_values=[1, 3, 5, 7, 9])
    evaluate_on(
        TEST_FILE,
        METRICS_TEST,
        IMPUTED_DIR / "test",
        knn_values=[1, 3, 5, 7, 9])
