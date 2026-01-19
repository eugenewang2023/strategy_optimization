#!/usr/bin/env python3
"""
split_data.py

Automatically split data files into LOW / HIGH volatility groups
using clustering, and output two TEXT FILE LISTS.

Outputs:
  - low_vol.txt
  - high_vol.txt
  - split_summary.csv

Default split method: KMeans
Optional: Gaussian Mixture Model (GMM)

Volatility metric:
  atr_pct_med = median( ATR(atr_len) / close )

NOTE (per your request):
- low/high list files contain ONLY filenames (e.g., AAPL.parquet),
  not directory paths.
- split_summary.csv keeps BOTH:
    * file_path (full path)
    * file_name (basename only)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# -------------------------------------------------
# Robust OHLC normalization
# -------------------------------------------------
def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {str(c).strip().lower(): c for c in df.columns}

    alias = {
        "open": ["open", "o"],
        "high": ["high", "h"],
        "low": ["low", "l"],
        "close": ["close", "c", "adj close", "adj_close", "adjclose"],
    }

    resolved = {}
    for k, names in alias.items():
        for n in names:
            if n in cols:
                resolved[k] = cols[n]
                break

    missing = [k for k in alias if k not in resolved]
    if missing:
        raise KeyError(
            f"Missing OHLC columns {missing}. "
            f"Available: {list(df.columns)}"
        )

    out = df[[resolved["open"], resolved["high"],
              resolved["low"], resolved["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out


def atr_pct_median(df: pd.DataFrame, atr_len: int) -> float:
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)

    if len(c) < max(atr_len + 5, 50):
        return float("nan")

    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]

    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(atr_len).mean().to_numpy()

    atr_pct = atr / (c + 1e-12)
    atr_pct = atr_pct[np.isfinite(atr_pct)]
    return float(np.median(atr_pct)) if atr_pct.size else float("nan")


# -------------------------------------------------
# Clustering methods
# -------------------------------------------------
def split_kmeans(vol: np.ndarray, seed: int):
    from sklearn.cluster import KMeans

    x = np.log(vol + 1e-12).reshape(-1, 1)
    km = KMeans(n_clusters=2, n_init=20, random_state=seed)
    labels = km.fit_predict(x)

    centers = km.cluster_centers_.reshape(-1)
    hi = int(np.argmax(centers))
    return (labels == hi).astype(int), float(np.exp(np.mean(centers))), "kmeans"


def split_gmm(vol: np.ndarray, seed: int):
    from sklearn.mixture import GaussianMixture

    x = np.log(vol + 1e-12).reshape(-1, 1)
    gm = GaussianMixture(n_components=2, random_state=seed)
    gm.fit(x)

    means = gm.means_.reshape(-1)
    hi = int(np.argmax(means))
    probs = gm.predict_proba(x)

    return (probs[:, hi] >= 0.5).astype(int), float(np.exp(np.mean(means))), "gmm"


# -------------------------------------------------
# Main
# -------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--atr-len", type=int, default=25)
    ap.add_argument("--split-method", choices=["kmeans", "gmm"], default="kmeans")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--low-list", default="low_vol.txt")
    ap.add_argument("--high-list", default="high_vol.txt")
    ap.add_argument("--summary-csv", default="split_summary.csv")
    args = ap.parse_args()

    try:
        import sklearn  # noqa: F401
    except Exception:
        print("ERROR: scikit-learn required (pip install scikit-learn)", file=sys.stderr)
        sys.exit(2)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        print("No parquet files found.", file=sys.stderr)
        sys.exit(2)

    rows = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df = ensure_ohlc(df)
            vol = atr_pct_median(df, args.atr_len)
            rows.append(
                {
                    "ticker": f.stem,
                    "file_path": str(f),
                    "file_name": f.name,  # <-- basename only
                    "atr_pct_med": vol,
                }
            )
        except Exception as e:
            rows.append(
                {
                    "ticker": f.stem,
                    "file_path": str(f),
                    "file_name": f.name,  # <-- basename only
                    "atr_pct_med": float("nan"),
                    "error": str(e),
                }
            )

    summary = pd.DataFrame(rows)
    good = summary[np.isfinite(summary["atr_pct_med"])].copy()

    if good.empty:
        print("All files failed volatility computation.", file=sys.stderr)
        summary.to_csv(args.summary_csv, index=False)
        sys.exit(2)

    vol = good["atr_pct_med"].to_numpy()

    if args.split_method == "kmeans":
        labels, thr, used = split_kmeans(vol, args.seed)
    else:
        labels, thr, used = split_gmm(vol, args.seed)

    good["split"] = np.where(labels == 1, "high", "low")

    # Merge split label back into full summary
    summary = summary.merge(good[["ticker", "split"]], on="ticker", how="left")

    # Write lists (FILENAMES ONLY)
    low_files = good.loc[good["split"] == "low", "file_name"]
    high_files = good.loc[good["split"] == "high", "file_name"]

    Path(args.low_list).write_text("\n".join(low_files.astype(str)) + "\n")
    Path(args.high_list).write_text("\n".join(high_files.astype(str)) + "\n")
    summary.to_csv(args.summary_csv, index=False)

    print("==============================================")
    print("Volatility split completed")
    print(f"Method:              {used}")
    print(f"ATR length:          {args.atr_len}")
    print(f"Approx threshold:    {thr:.6f}")
    print(f"Low-vol count:       {len(low_files)}  -> {args.low_list}")
    print(f"High-vol count:      {len(high_files)} -> {args.high_list}")
    print(f"Summary CSV:         {args.summary_csv}")
    print("==============================================")


if __name__ == "__main__":
    main()
