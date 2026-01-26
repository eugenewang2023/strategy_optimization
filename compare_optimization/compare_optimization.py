#!/usr/bin/env python3

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import io


# ============================================================
# Setup Windows console encoding
# ============================================================
def setup_windows_encoding():
    """Fix encoding issues on Windows console"""
    if sys.platform == "win32":
        try:
            # Try to set UTF-8 encoding for Windows console
            import codecs
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except:
            pass


setup_windows_encoding()


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
    loaded_files = []
    for f in files:
        try:
            df = pd.read_csv(f)
            
            # Skip if dataframe is empty
            if df.empty:
                print(f"Warning: {f.name} is empty, skipping")
                continue
                
            stem = f.stem

            # Robust run_id extraction:
            # Take everything AFTER the last "per_ticker"
            if "per_ticker" in stem:
                parts = stem.split("per_ticker")
                head = parts[0]
                tail = parts[-1].lstrip("_-")
                run_id = head + tail
            else:
                run_id = stem  # fallback

            df["run_id"] = run_id
            dfs.append(df)
            loaded_files.append(f)
        except Exception as e:
            print(f"Warning: Could not read {f.name}: {e}")
            continue

    return dfs, loaded_files


# ============================================================
# Safe column access helper
# ============================================================
def safe_column_op(df, column_name, operation="mean", default=np.nan):
    """Safely access a column and perform an operation"""
    if column_name not in df.columns:
        return default
    
    try:
        if operation == "mean":
            return df[column_name].mean()
        elif operation == "median":
            return df[column_name].median()
        elif operation == "sum":
            return df[column_name].sum()
        else:
            return default
    except Exception:
        return default


# ============================================================
# Summaries
# ============================================================
def summarize_run(df):
    """Create a summary for a single run dataframe"""
    if df.empty:
        return {
            "run_id": "unknown",
            "avg_total_return": np.nan,
            "median_total_return": np.nan,
            "avg_pf_eff": np.nan,
            "median_pf_eff": np.nan,
            "avg_maxdd": np.nan,
            "eligible_rate": np.nan,
            "avg_trades": np.nan,
            "num_tickers": 0,
        }
    
    # Get run_id safely
    run_id = df["run_id"].iloc[0] if "run_id" in df.columns and len(df) > 0 else "unknown"
    
    return {
        "run_id": run_id,
        "avg_total_return": safe_column_op(df, "total_return", "mean"),
        "median_total_return": safe_column_op(df, "total_return", "median"),
        "avg_pf_eff": safe_column_op(df, "profit_factor_eff", "mean"),
        "median_pf_eff": safe_column_op(df, "profit_factor_eff", "median"),
        "avg_maxdd": safe_column_op(df, "maxdd", "mean"),
        "eligible_rate": safe_column_op(df, "eligible", "mean"),
        "avg_trades": safe_column_op(df, "num_trades", "mean"),
        "num_tickers": len(df),
    }


def aggregate_summary(dfs):
    """Aggregate summaries from multiple dataframes"""
    if not dfs:
        print("Warning: No dataframes to summarize")
        return pd.DataFrame()
        
    rows = []
    for i, df in enumerate(dfs):
        try:
            summary = summarize_run(df)
            rows.append(summary)
        except Exception as e:
            print(f"Warning: Could not summarize dataframe {i}: {e}")
            continue
    
    if not rows:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(rows)
    
    # Only sort by avg_total_return if it exists and has valid values
    if "avg_total_return" in summary_df.columns:
        # Check if we have any non-NaN values to sort by
        if not summary_df["avg_total_return"].isna().all():
            summary_df = summary_df.sort_values("avg_total_return", ascending=False)
    
    return summary_df.reset_index(drop=True)


# ============================================================
# Best run per ticker (only if total_return exists)
# ============================================================
def best_per_ticker(dfs):
    """Find which run performed best for each ticker"""
    # Filter out dataframes without required columns
    valid_dfs = []
    for df in dfs:
        if "total_return" in df.columns and "ticker" in df.columns and "run_id" in df.columns:
            if not df.empty:
                valid_dfs.append(df)
    
    if not valid_dfs:
        print("Warning: No dataframes with 'total_return', 'ticker', and 'run_id' columns found")
        return pd.DataFrame(columns=["run_id", "num_best_tickers"])
    
    try:
        combined = pd.concat(valid_dfs, ignore_index=True)
        idx = combined.groupby("ticker")["total_return"].idxmax()
        winners = combined.loc[idx]
        
        result = (
            winners["run_id"]
            .value_counts()
            .rename("num_best_tickers")
            .reset_index()
            .rename(columns={"index": "run_id"})
        )
        return result
    except Exception as e:
        print(f"Warning: Error in best_per_ticker: {e}")
        return pd.DataFrame(columns=["run_id", "num_best_tickers"])


# ============================================================
# Per-ticker matrix (only if total_return exists)
# ============================================================
def per_ticker_matrix(dfs):
    """Create a matrix of returns by ticker and run"""
    # Filter out dataframes without required columns
    valid_dfs = []
    for df in dfs:
        if all(col in df.columns for col in ["ticker", "total_return", "run_id"]):
            if not df.empty:
                valid_dfs.append(df)
    
    if not valid_dfs:
        print("Warning: No dataframes with required columns for matrix")
        return pd.DataFrame()
    
    try:
        combined = pd.concat(valid_dfs, ignore_index=True)
        pivot = combined.pivot_table(
            index="ticker",
            columns="run_id",
            values="total_return",
            aggfunc="first",
        )
        return pivot
    except Exception as e:
        print(f"Warning: Error creating pivot table: {e}")
        return pd.DataFrame()


# ============================================================
# Show column info for debugging
# ============================================================
def show_column_info(dfs, files):
    """Display column information for debugging"""
    print("\n=== COLUMN INFORMATION ===")
    for i, (df, f) in enumerate(zip(dfs, files)):
        print(f"\n{i+1}. {f.name}:")
        print(f"   Shape: {df.shape}")
        print(f"   Columns ({len(df.columns)}):")
        
        # Group columns for better readability
        common_metrics = ["total_return", "profit_factor", "profit_factor_eff", 
                         "maxdd", "eligible", "num_trades", "ticker", "run_id"]
        
        print("   Common metrics:")
        for col in common_metrics:
            present = "[YES]" if col in df.columns else "[NO]"
            print(f"     {col}: {present}")
        
        # Show other columns
        other_cols = [col for col in df.columns if col not in common_metrics]
        if other_cols:
            print(f"   Other columns ({len(other_cols)}): {', '.join(other_cols[:10])}")
            if len(other_cols) > 10:
                print(f"     ... and {len(other_cols) - 10} more")
        
        # Show sample data if available
        if not df.empty and "total_return" in df.columns:
            print(f"   total_return range: [{df['total_return'].min():.4f}, {df['total_return'].max():.4f}]")


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
    ap.add_argument(
        "--debug",
        action="store_true",
        help="Show detailed column information for debugging",
    )

    args = ap.parse_args()

    input_dir = Path(args.input_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load data
    dfs, files = load_runs_from_dir(input_dir)
    
    if not dfs:
        raise SystemExit("No valid data files could be loaded")
    
    print(f"Loaded {len(dfs)} valid optimization runs:")
    for f in files:
        print(f"  - {f.name}")

    # Debug mode: show column information
    if args.debug:
        show_column_info(dfs, files)

    # =========================
    # Aggregate summary
    # =========================
    print("\n" + "="*50)
    print("AGGREGATE RUN COMPARISON")
    print("="*50)
    
    summary_df = aggregate_summary(dfs)
    if not summary_df.empty:
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}" if pd.notnull(x) else "N/A"))
        summary_df.to_csv(out_dir / "run_summary.csv", index=False)
    else:
        print("No summary data available")

    # =========================
    # Best run per ticker
    # =========================
    print("\n" + "="*50)
    print("BEST RUN PER TICKER")
    print("="*50)
    
    winners_df = best_per_ticker(dfs)
    if not winners_df.empty:
        print(winners_df.to_string(index=False))
        winners_df.to_csv(out_dir / "best_run_counts.csv", index=False)
    else:
        print("No best run data available")

    # =========================
    # Per-ticker matrix
    # =========================
    print("\n" + "="*50)
    print("PER-TICKER MATRIX")
    print("="*50)
    
    pivot_df = per_ticker_matrix(dfs)
    if not pivot_df.empty:
        pivot_df.to_csv(out_dir / "per_ticker_return_matrix.csv")
        print(f"Matrix shape: {pivot_df.shape}")
        print("Saved to per_ticker_return_matrix.csv")
    else:
        print("No matrix data available")

    # =========================
    # Additional diagnostics
    # =========================
    print("\n" + "="*50)
    print("DIAGNOSTICS")
    print("="*50)
    
    # Check which files have total_return column
    has_total_return = []
    no_total_return = []
    
    for df, f in zip(dfs, files):
        if "total_return" in df.columns:
            has_total_return.append(f.name)
        else:
            no_total_return.append(f.name)
    
    if has_total_return:
        print(f"Files WITH 'total_return' column ({len(has_total_return)}):")
        for name in has_total_return:
            print(f"  [YES] {name}")
    
    if no_total_return:
        print(f"\nFiles WITHOUT 'total_return' column ({len(no_total_return)}):")
        for name in no_total_return:
            print(f"  [NO]  {name}")
    
    print(f"\nSaved comparison outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()