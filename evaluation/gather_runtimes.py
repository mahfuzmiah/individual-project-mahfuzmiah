# evaluation/gather_runtimes.py

import pandas as pd
from pathlib import Path


def _drop_header_rows(df):
    mask = ~df.apply(lambda row: (row == df.columns).all(), axis=1)
    return df[mask].reset_index(drop=True)


def gather_runtimes(root: Path = Path("predictions")) -> pd.DataFrame:
    rows = []
    for csv_path in root.rglob("*.csv"):
        stem = csv_path.stem.lower()
        parts = [p.lower() for p in csv_path.parts]

        # only timing/runtimes/summary files
        if not any(tag in stem for tag in ("runtime", "timing", "summary")):
            continue

        # --- map to your metric‐model keys ---
        if "lightgbm_iterative" in parts:
            model_name = "lightgbm_iterative"
        elif "lightgbm" in parts:
            model_name = "lightgbm"
        elif "arima" in parts:
            model_name = "arima"
        elif "naive_runtime" in stem:
            model_name = "naive"
        elif "seasonalnaive" in stem:
            model_name = "seasonal_naive"
        elif "seasonalwa" in stem:
            model_name = "seasonal_window_average"
        elif "lstm_blocks" in stem:
            # or match exactly your key, e.g. with epoch suffix later
            model_name = "lstm_blocks_4feat"
        elif "vector_lstm" in stem:
            model_name = "vector_lstm"
        elif stem.startswith(("lstm1_", "lstm2_")):
            # pull out just e.g. "lstm1" or "lstm2"
            model_name = stem.split("_")[0]
        else:
            # otherwise skip
            continue

        df = pd.read_csv(csv_path)
        df = _drop_header_rows(df)

        # LightGBM‐style timing CSVs
        if {"model", "stage", "total_time_s"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append({
                    "model":             r["model"].lower(),
                    "stage":             r["stage"],
                    "total_time_s":      float(r["total_time_s"]),
                    "avg_time_per_row_s": float(r.get("avg_time_per_row_s", pd.NA)),
                    "n_candidates":       r.get("n_candidates", pd.NA),
                })

        # LSTM with blocks
        elif {"block_size", "epochs", "total_time_s"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append({
                    "model":             model_name,
                    "stage":             f"blocks={int(r.block_size)},epochs={int(r.epochs)}",
                    "total_time_s":      float(r["total_time_s"]),
                    "avg_time_per_row_s": float(r.get("avg_time_per_row_s", pd.NA)),
                    "n_candidates":       pd.NA,
                })

        # Plain LSTM epochs
        elif {"epochs", "total_time_s"}.issubset(df.columns):
            for _, r in df.iterrows():
                rows.append({
                    "model":             model_name,
                    "stage":             f"epochs={int(r.epochs)}",
                    "total_time_s":      float(r["total_time_s"]),
                    "avg_time_per_row_s": float(r.get("avg_time_per_row_s", pd.NA)),
                    "n_candidates":       pd.NA,
                })

        # Fallback one‐row CSVs
        elif "total_time_s" in df.columns:
            stage = stem.replace("_timing_summary", "").replace("_runtime", "")
            total = float(df["total_time_s"].iat[0])
            avg = float(df["avg_time_per_row_s"].iat[0]
                        ) if "avg_time_per_row_s" in df.columns else pd.NA
            ncand = df["n_candidates"].iat[0] if "n_candidates" in df.columns else pd.NA

            rows.append({
                "model":             model_name,
                "stage":             stage,
                "total_time_s":      total,
                "avg_time_per_row_s": avg,
                "n_candidates":       ncand,
            })

    return pd.DataFrame(rows)
