#!/usr/bin/env python3
import pandas as pd
import numpy as np
import argparse
from pathlib import Path
import glob


# ------------------------------------------------------------
# Flexible column detection
# ------------------------------------------------------------

PF_CANDIDATES = [
    "profit_factor",          # half_RSI
    "profit_factor_diag",     # Vidya_RSI
    "profit_factor_eff",
    "profit_factor_raw",
]

RET_CANDIDATES = [
    "total_return",           # half_RSI
    "tot_ret",                # Vidya_RSI
]

TRADES_CANDIDATES = [
    "num_trades",             # half_RSI
    "trades",                 # Vidya_RSI
]


def pick_column(df, candidates, name, file_label):
    for c in candidates:
        if c in df.columns:
            return c
    raise KeyError(f"{file_label}: no valid {name} column found. "
                   f"Tried: {candidates}")


# ------------------------------------------------------------
# Summary function
# ------------------------------------------------------------

def summarize(df: pd.DataFrame, name: str):
    pf_col = pick_column(df, PF_CANDIDATES, "profit_factor", name)
    ret_col = pick_column(df, RET_CANDIDATES, "total_return", name)
    tr_col = pick_column(df, TRADES_CANDIDATES, "num_trades", name)

    pf = df[pf_col].replace([np.inf, -np.inf], np.nan)
    ret = df[ret_col]
    tr = df[tr_col]

    print(f"\n=== {name} ===")
    print(f"Using PF column     : {pf_col}")
    print(f"Using return column : {ret_col}")
    print(f"Using trades column : {tr_col}")
    print(f"Median PF           : {pf.median():.4f}")
    print(f"Median PF_eff(c10)  : {pf.clip(0, 10).median():.4f}")
    print(f"Median return       : {ret.median():.4f}")
    print(f"Median trades       : {tr.median():.1f}")
    print(f"Neg-return tickers  : {(ret <= 0).mean() * 100:.1f}%")
    print(f"Stability (MAD)     : {np.median(np.abs(ret - ret.median())):.4f}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def load_csvs_from_dir(directory: str):
    pattern = str(Path(directory) / "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {directory}")
    return [(Path(f).name, pd.read_csv(f)) for f in files]


def main():
    ap = argparse.ArgumentParser(description="Flexible per-ticker summary tool")
    ap.add_argument(
        "--data-dir",
        type=str,
        default="input_summary",
        help='Directory containing per-ticker CSV files (default: "input_summary")'
    )
    args = ap.parse_args()

    csvs = load_csvs_from_dir(args.data_dir)

    print(f"\nFound {len(csvs)} CSV files in {args.data_dir}")

    # Per-file summaries
    for fname, df in csvs:
        summarize(df, fname)

    # Combined summary
    print("\n=== Combined Summary Across All Files ===")
    combined = pd.concat([df for _, df in csvs], ignore_index=True)
    summarize(combined, "ALL_FILES")


if __name__ == "__main__":
    main()
