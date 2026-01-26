#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# Load all runs containing "per_ticker" anywhere in filename
# ============================================================
def load_runs_from_dir(input_dir: Path):
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    # Match ANY file containing "per_ticker" anywhere in the name
    files = sorted(input_dir.glob("*per_ticker*.csv"))
    if not files:
        raise SystemExit(f"No files containing 'per_ticker' found in: {input_dir}")

    dfs = []
    for f in files:
        df = pd.read_csv(f)

        stem = f.stem

        # Robust run_id extraction:
        # Take everything AFTER the last "per_ticker"
        if "per_ticker" in stem:
            parts = stem.split("per_ticker")
            head = parts[0]
            tail = parts[-1].lstrip("_-")
            run_id = head + tail
            if not run_id:
                run_id = stem
        else:
            run_id = stem  # fallback

        df["run_id"] = run_id
        dfs.append(df)

    return dfs, files


# ============================================================
# Summaries
# ============================================================
def summarize_run(df):
    return {
        "run_id": df["run_id"].iloc[0],
        "avg_total_return": df["total_return"].mean(),
        "median_total_return": df["total_return"].median(),
        "avg_pf_eff": df["profit_factor_eff"].mean() if "profit_factor_eff" in df else np.nan,
        "median_pf_eff": df["profit_factor_eff"].median() if "profit_factor_eff" in df else np.nan,
        "avg_maxdd": df["maxdd"].mean(),
        "eligible_rate": df["eligible"].mean() if "eligible" in df else np.nan,
        "avg_trades": df["num_trades"].mean(),
        "num_tickers": len(df),
    }


def aggregate_summary(dfs):
    rows = [summarize_run(df) for df in dfs]
    return (
        pd.DataFrame(rows)
        .sort_values("avg_total_return", ascending=False)
        .reset_index(drop=True)
    )


# ============================================================
# Best run per ticker
# ============================================================
def best_per_ticker(dfs):
    combined = pd.concat(dfs, ignore_index=True)
    idx = combined.groupby("ticker")["total_return"].idxmax()
    winners = combined.loc[idx]
    return (
        winners["run_id"]
        .value_counts()
        .rename("num_best_tickers")
        .reset_index()
        .rename(columns={"index": "run_id"})
    )


# ============================================================
# Per-ticker matrix
# ============================================================
def per_ticker_matrix(dfs):
    combined = pd.concat(dfs, ignore_index=True)
    return combined.pivot_table(
        index="ticker",
        columns="run_id",
        values="total_return",
        aggfunc="first",
    )


# ============================================================
# Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Compare per-ticker optimization result CSV files"
    )
    ap.add_argument(
        "--input_dir",
        type=str,
        default="input",
        help="Directory containing per_ticker*.csv files (default: input)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="comparison_output",
        help="Directory to write comparison outputs",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    dfs, files = load_runs_from_dir(input_dir)

    print(f"Loaded {len(files)} optimization runs:")
    for f in files:
        print(f"  - {f.name}")

    # =========================
    # Aggregate summary
    # =========================
    summary_df = aggregate_summary(dfs)
    print("\n=== AGGREGATE RUN COMPARISON ===")
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    summary_df.to_csv(out_dir / "run_summary.csv", index=False)

    # =========================
    # Best run per ticker
    # =========================
    winners_df = best_per_ticker(dfs)
    print("\n=== BEST RUN PER TICKER (by total_return) ===")
    print(winners_df.to_string(index=False))

    winners_df.to_csv(out_dir / "best_run_counts.csv", index=False)

    # =========================
    # Per-ticker matrix
    # =========================
    pivot_df = per_ticker_matrix(dfs)
    pivot_df.to_csv(out_dir / "per_ticker_return_matrix.csv")

    print(f"\nSaved comparison outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
