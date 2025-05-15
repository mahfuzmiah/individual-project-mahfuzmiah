import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
# ─── Paths & Config ───────────────────────────────────────────────────────────
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from config import DIAGRAMS_DIR  # nopep8


def plot_metric_vs_horizon(metrics_df, metric: str, log_scale: bool = False):
    """
    Plot a given error metric vs. forecast horizon for all models.

    Expects metrics_df to have columns:
      ['model','horizon',<metric>]
    """
    # Pivot on 'horizon', not 'step'
    df = metrics_df.pivot_table(
        index='step',
        columns='model',
        values=metric,
        aggfunc='mean'
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for model in df.columns:
        ax.plot(df.index, df[model], label=model, linewidth=1)

    ax.set_xlabel('Forecast Horizon (quarters ahead)')
    ax.set_ylabel(metric)
    ax.set_title(f'{metric} vs. Horizon')
    if log_scale:
        ax.set_yscale('log')

    ax.legend(
        title='Model',
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=8,
        title_fontsize=9
    )
    ax.grid(which='both', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / f"{metric}_vs_horizon.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_accuracy_speed(runtime_df, metrics_df, metric="RMSE", max_horizon=4):
    """
    Scatter of Total Training Time vs. Mean {metric} (h=1..max_horizon) for all model families.
    """
    # 1) restrict metrics to first few horizons
    df = metrics_df[metrics_df["horizon"] <= max_horizon]

    # 2) compute mean metric for each model
    met = (
        df
        .groupby("model", as_index=False)[metric]
        .mean()
        .rename(columns={metric: f"mean_{metric}"})
    )

    # 3) sum up total runtime for each model
    rt = (
        runtime_df
        .groupby("model", as_index=False)["total_time_s"]
        .sum()
    )

    # 4) merge on model
    combo = pd.merge(met, rt, on="model", how="inner")

    # 5) plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(combo["total_time_s"], combo[f"mean_{metric}"], s=100)

    for _, row in combo.iterrows():
        ax.text(
            row["total_time_s"] * 1.01,
            row[f"mean_{metric}"] * 1.01,
            row["model"],
            fontsize=9
        )

    ax.set_xlabel("Total training time (s)")
    ax.set_ylabel(f"Mean {metric} (h=1…{max_horizon})")
    ax.set_title(f"Speed vs. {metric} trade–off")
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / f"accuracy_speed_{metric}.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lstm_training_times(runtime_df):
    """
    Plot LSTM training time vs. epochs for each block size.

    Parameters:
    - runtime_df: DataFrame with ['model', 'stage', 'total_time_s']
                  where 'stage' strings contain 'blocks=X,epochs=Y'
    """
    # Filter LSTM entries with block & epoch info
    lstm_df = runtime_df[runtime_df['stage'].str.contains('blocks=')].copy()

    # Extract block size and epoch count
    lstm_df[['block', 'epoch']] = lstm_df['stage'] \
        .str.extract(r'blocks=(\d+),epochs=(\d+)') \
        .astype(int)

    # Pivot: rows=epoch, cols=block
    pivot = lstm_df.pivot_table(
        index='epoch', columns='block', values='total_time_s', aggfunc='sum'
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    pivot.plot(marker='o', ax=ax)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total training time (s)')
    ax.set_title('LSTM Training Time vs. Epochs')
    ax.legend(title='Block size', fontsize=8, title_fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / "lstm_training_times.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lstm_epoch_runtimes(runtime_df):
    """
    Bar chart of total training time vs. epochs for each LSTM variant.
    Expects runtime_df with columns ['model','stage','total_time_s'].
    """
    print("▶ runtime_df.sample(5):\n", runtime_df.sample(5))
    print("▶ unique stages:", runtime_df['stage'].unique())
    df = runtime_df.copy()
    # debug: what stage values do we actually have?
    print(">> Found stages:", df['stage'].unique())

    # only keep rows with 'epochs=' in the stage
    mask = df['stage'].str.contains(r'epochs=', na=False)
    df = df[mask].copy()
    if df.empty:
        print("⚠️  No 'epochs=' stages found in runtime_df, check your gather_runtimes output.")
        return

    # coerce to numeric
    df['total_time_s'] = pd.to_numeric(df['total_time_s'], errors='coerce')
    df = df.dropna(subset=['total_time_s'])

    # extract epoch count
    df['epochs'] = df['stage'].str.extract(r'epochs=(\d+)').astype(int)

    # sum total_time by epochs & model
    agg = (
        df
        .groupby(['epochs', 'model'], as_index=False)['total_time_s']
        .sum()
        .pivot(index='epochs', columns='model', values='total_time_s')
    )

    # plot grouped bar
    ax = agg.plot(kind='bar', figsize=(8, 5))
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Total training time (s)')
    ax.set_title('LSTM Training Time by Epochs & Variant')
    ax.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / "lstm_epoch_runtimes.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lstm_epoch_accuracy(metrics_df, metric='WMAPE'):
    """
    Line+marker chart of {metric} vs. forecast horizon,
    for each LSTM epoch‐variant.
    Expects metrics_df with ['model','horizon',metric].
    Models must be named like 'lstm1_50epochs', etc.
    """
    # 1) Filter only the *_<N>epochs models
    mask = metrics_df['model'].str.contains(r'_\d+epochs$', na=False)
    df = metrics_df[mask].copy()
    if df.empty:
        print("⚠️  No epoch‐variant models found. Check your model names.")
        return

    # 2) Coerce metric to numeric & drop bad rows
    df[metric] = pd.to_numeric(df[metric], errors='coerce')
    df = df.dropna(subset=[metric])

    # 3) Pivot: index=horizon, columns=model, values=metric
    pivot = df.pivot(index='horizon', columns='model', values=metric)
    if pivot.empty:
        print("⚠️  Pivot resulted in empty DataFrame.")
        return

    # 4) Plot with markers so single‐point segments still show
    ax = pivot.plot(
        figsize=(8, 5),
        marker='o',       # show each point
        linewidth=1.5
    )
    ax.set_xlabel('Forecast Horizon (quarters ahead)')
    ax.set_ylabel(f'Mean {metric}')
    ax.set_title(f'LSTM {metric} vs. Horizon by Epoch Variant')
    ax.legend(title='Variant', bbox_to_anchor=(1.02, 1), loc='upper left')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / f"lstm_epoch_accuracy_{metric}.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lightgbm_accuracy_runtime(metrics_df, runtime_df, metrics=('RMSE', 'SMAPE', 'WMAPE')):
    """
    Compare lightgbm vs. lightgbm_iterative on both training time and accuracy.
    Will produce one small scatter for each metric in `metrics`.
    """
    # summarize mean metrics
    met = (
        metrics_df
        .groupby('model', as_index=False)[list(metrics)]
        .mean()
    )

    # summarize total runtime
    rt = (
        runtime_df.assign(total_time_s=pd.to_numeric(
            runtime_df['total_time_s'], errors='coerce'))
        .groupby('model', as_index=False)['total_time_s']
        .sum()
    )

    cmp = pd.merge(met, rt, on='model')
    cmp = cmp[cmp['model'].isin(['lightgbm', 'lightgbm_iterative'])]
    if cmp.empty:
        print("⚠️  No lightgbm models found to compare.")
        return

    # one subplot per metric
    fig, axes = plt.subplots(1, len(metrics), figsize=(
        5*len(metrics), 5), squeeze=False)
    for ax, metric in zip(axes[0], metrics):
        for _, row in cmp.iterrows():
            ax.scatter(row['total_time_s'], row[metric], s=100)
            ax.text(row['total_time_s']*1.02, row[metric]*1.02,
                    row['model'], fontsize=9)
        ax.set_xlabel('Total training time (s)')
        ax.set_ylabel(f'Mean {metric}')
        ax.set_title(f'LightGBM: Time vs. {metric}')
        ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / "lightgbm_accuracy_runtime.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lstm_variant_tradeoff(metrics_df, runtime_df, metric='WMAPE'):
    """
    Scatter four 500-epoch LSTMs on time vs. accuracy.
    Abbreviations:
      L1, L2, LV, L4  for the four variants.
    """
    mapping = {
        'lstm1_500epochs':         'L1',
        'lstm2_500epochs':         'L2',
        'vector_lstm_500epochs':   'LV',
        'lstm_blocks_4feat_500ep': 'L4',
    }

    # mean metric
    met = (
        metrics_df
        .groupby('model', as_index=False)[metric]
        .mean()
        .query("model in @mapping.keys()")
    )
    # total runtime
    rt = (
        runtime_df.assign(total_time_s=pd.to_numeric(
            runtime_df['total_time_s'], errors='coerce'))
        .groupby('model', as_index=False)['total_time_s']
        .sum()
        .query("model in @mapping.keys()")
    )
    df = pd.merge(met, rt, on='model')
    if df.empty:
        print("⚠️  No LSTM-500epoch variants found in inputs.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for _, r in df.iterrows():
        lbl = mapping[r['model']]
        ax.scatter(r['total_time_s'], r[metric], s=120)
        ax.text(r['total_time_s']*1.01, r[metric]*1.01, lbl,
                fontsize=12, weight='bold')
    ax.set_xlabel('Total training time (s)')
    ax.set_ylabel(f'Mean {metric}')
    ax.set_title('LSTM Variant Trade‐off (500 epochs)')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / f"lstm_variant_tradeoff_{metric}.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_lstm_hybrid_by_block(
    metrics_df,
    metric: str = "RMSE",
    block_sizes=(4, 8, 20, 40),
    epochs=(100, 200, 500)
):
    """
    Line plot of mean {metric} vs. epochs, one line per block_size
    for your lstm_blocks_{block}feat_{epoch}ep variants.
    """
    # 1) Filter only your hybrid‐block models
    df = metrics_df[metrics_df["model"].str.startswith("lstm_blocks_")].copy()

    # 2) Extract block size and epoch
    df[["block", "epoch"]] = (
        df["model"]
        .str.extract(r"lstm_blocks_(\d+)feat_(\d+)ep")
        .astype(int)
    )

    # 3) Compute mean metric per (block, epoch)
    pivot = (
        df
        .groupby(["block", "epoch"], as_index=False)[metric]
        .mean()
        .pivot(index="epoch", columns="block", values=metric)
        .reindex(index=epochs, columns=block_sizes)
    )

    # 4) Plot
    ax = pivot.plot(marker="o", linewidth=2, figsize=(8, 5))
    ax.set_xlabel("Epochs")
    ax.set_ylabel(f"Mean {metric}")
    ax.set_title(f"LSTM-Hybrid Mean {metric} by Epochs & Block Size")
    ax.legend(title="Block size", loc="center left",
              bbox_to_anchor=(1.02, 0.5))
    ax.grid(alpha=0.3, linestyle="--")
    plt.tight_layout()
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / f"lstm_hybrid_by_block_{metric}.png",
        dpi=300,
        bbox_inches='tight'
    )


def plot_all_hybrid_metrics(metrics_df):
    metrics = ["RMSE", "SMAPE", "WMAPE", "Log-RMSE"]
    block_sizes = [4, 8, 20, 40]
    epochs = [100, 200, 500]

    # extract only your hybrid (blocked) LSTM rows
    df = metrics_df[metrics_df["model"].str.startswith("lstm_blocks_")].copy()
    df[["block", "epoch"]] = (
        df["model"]
        .str.extract(r"lstm_blocks_(\d+)feat_(\d+)ep")
        .astype(int)
    )

    fig, axs = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    for ax, metric in zip(axs.ravel(), metrics):
        # pivot to epochs×blocks
        pivot = (
            df
            .groupby(["block", "epoch"], as_index=False)[metric]
            .mean()
            .pivot(index="epoch", columns="block", values=metric)
            .reindex(index=epochs, columns=block_sizes)
        )

        pivot.plot(
            marker="o", linewidth=2, ax=ax, title=f"Mean {metric}"
        )
        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric)
        ax.grid(alpha=0.3, linestyle="--")

        # remove this subplot’s legend
        ax.get_legend().remove()

    # add a single legend to the right of the figure
    handles, labels = pivot.plot().get_legend_handles_labels()
    fig.legend(
        handles, labels,
        title="Block size",
        loc="upper right",
        bbox_to_anchor=(0.95, 0.75)
    )

    fig.suptitle("LSTM-Hybrid Accuracy vs. Epochs by Block-Size", fontsize=16)
    plt.show()
    plt.savefig(
        DIAGRAMS_DIR / "lstm_hybrid_metrics.png",
        dpi=300,
        bbox_inches='tight'
    )
