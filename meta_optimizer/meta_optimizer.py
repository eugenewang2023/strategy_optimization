#!/usr/bin/env python3
"""
meta_optimizer.py — Per‑ticker meta‑optimizer with regime inference and meta best report.

Features:
- Scans a directory for per‑ticker CSVs and best.txt files
- Picks the two most recent per‑ticker CSVs (or uses explicit paths)
- Infers which file is swing vs trend from per‑ticker stats
- Selects the better phase per ticker based on ticker_score
- Outputs a combined per‑ticker CSV with identical columns + final 'regime' column
- Reads the two corresponding best.txt files, extracts params
- Writes a meta_best.txt with:
    - degeneracy diagnostics
    - coverage & stability
    - PF & returns summary
    - BEST PARAMS (SWING)
    - BEST PARAMS (TREND)
    - BEST PARAMS (BLENDED) — weighted by ticker counts
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd


# ---------- File selection ----------

def pick_two_latest_per_ticker(csv_dir: Path) -> Tuple[Path, Path]:
    files = sorted(
        csv_dir.glob("*per_ticker*.csv"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if len(files) < 2:
        raise SystemExit(f"Need at least two per-ticker CSV files in {csv_dir}")
    return files[0], files[1]


def pick_two_latest_best(csv_dir: Path) -> Tuple[Path, Path]:
    files = sorted(
        csv_dir.glob("*best*.txt"),
        key=lambda f: f.stat().st_mtime,
        reverse=True,
    )
    if len(files) < 2:
        raise SystemExit(f"Need at least two best*.txt files in {csv_dir}")
    return files[0], files[1]


# ---------- Regime inference ----------

def summarize_per_ticker(df: pd.DataFrame) -> Dict[str, float]:
    return {
        "median_trades": df["num_trades"].median(),
        "median_pf_diag": df["profit_factor_diag"].median(),
        "avg_pf_eff": df["profit_factor_eff"].mean(),
        "cap_pct": df["pf_capped"].mean(),
        "zero_loss_pct": df["zero_loss"].mean(),
        "stability_score": df["run_stability_score"].median(),
    }


def infer_regimes(sumA: Dict[str, float], sumB: Dict[str, float]) -> Tuple[str, str]:
    """
    Decide which file is swing vs trend.
    Returns (trend_file_label, swing_file_label) where labels are "A" or "B".
    """

    trend_score_A = (
        (sumA["median_trades"] > sumB["median_trades"])
        + (sumA["avg_pf_eff"] > sumB["avg_pf_eff"])
        + (sumA["cap_pct"] > sumB["cap_pct"])
        + (sumA["stability_score"] > sumB["stability_score"])
    )

    trend_score_B = (
        (sumB["median_trades"] > sumA["median_trades"])
        + (sumB["avg_pf_eff"] > sumA["avg_pf_eff"])
        + (sumB["cap_pct"] > sumA["cap_pct"])
        + (sumB["stability_score"] > sumA["stability_score"])
    )

    if trend_score_A > trend_score_B:
        return "A", "B"  # A = trend, B = swing
    else:
        return "B", "A"  # B = trend, A = swing


# ---------- best.txt parsing ----------

PARAM_KEYS = ["fp", "reg_ratio", "sp", "ts", "vl", "vs"]


def parse_best_params(best_path: Path) -> Dict[str, float]:
    params: Dict[str, float] = {}
    in_best_block = False

    with best_path.open("r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("=== BEST PARAMS"):
                in_best_block = True
                continue
            if in_best_block:
                if stripped.startswith("===") and "BEST PARAMS" not in stripped:
                    # End of params block
                    break
                if ":" in stripped:
                    key, val = [x.strip() for x in stripped.split(":", 1)]
                    if key in PARAM_KEYS:
                        try:
                            params[key] = float(val)
                        except ValueError:
                            pass

    return params


def blend_params(
    swing_params: Dict[str, float],
    trend_params: Dict[str, float],
    n_swing: int,
    n_trend: int,
) -> Dict[str, float]:
    blended: Dict[str, float] = {}
    total = n_swing + n_trend if (n_swing + n_trend) > 0 else 1

    for key in PARAM_KEYS:
        s_val = swing_params.get(key)
        t_val = trend_params.get(key)
        if s_val is not None and t_val is not None:
            blended[key] = (n_swing * s_val + n_trend * t_val) / total
        elif s_val is not None:
            blended[key] = s_val
        elif t_val is not None:
            blended[key] = t_val

    return blended


# ---------- Meta stats & report ----------

def compute_meta_stats(df: pd.DataFrame) -> Dict[str, float]:
    stats = {}

    # Coverage & stability
    eligible_count = int(df["eligible"].sum())
    total_count = len(df)
    coverage = eligible_count / total_count if total_count > 0 else 0.0
    stats["coverage"] = coverage
    stats["eligible_count"] = eligible_count
    stats["total_count"] = total_count

    stats["zero_loss_pct"] = float(df["zero_loss"].mean())
    stats["cap_pct"] = float(df["pf_capped"].mean())
    stats["stability_score"] = float(df["run_stability_score"].median())

    # PF & returns summary
    stats["median_pf_diag"] = float(df["profit_factor_diag"].median())
    stats["avg_pf_diag"] = float(df["profit_factor_diag"].mean())
    stats["median_pf_eff"] = float(df["profit_factor_eff"].median())
    stats["avg_pf_eff"] = float(df["profit_factor_eff"].mean())
    stats["median_trades"] = float(df["num_trades"].median())
    stats["median_total_return"] = float(df["total_return"].median())

    # Objective score: use median ticker_score
    if "ticker_score" in df.columns:
        stats["objective_score"] = float(df["ticker_score"].median())
    else:
        stats["objective_score"] = float(stats["median_pf_diag"] * stats["stability_score"])

    return stats


def format_float(x: float, decimals: int) -> str:
    return f"{x:.{decimals}f}"


def write_meta_best(
    out_path: Path,
    stats: Dict[str, float],
    swing_params: Dict[str, float],
    trend_params: Dict[str, float],
    blended_params: Dict[str, float],
):
    with out_path.open("w") as f:
        f.write("=== OBJECTIVE DEGENERACY DIAGNOSTICS ===\n")
        f.write("penalty_mult: 1.000000\n")
        f.write("zero_loss_mult: 1.000000\n")
        f.write("cap_mult: 1.000000\n")
        f.write("glpt_mult: 1.000000\n")
        f.write("ret_mult: 1.000000\n")
        f.write("pf_floor_mult: 1.000000\n")
        f.write("trades_mult: 1.000000\n\n")

        f.write("=== COVERAGE & STABILITY ===\n")
        f.write(
            f"coverage: {format_float(stats['coverage'], 4)}  "
            f"(eligible {stats['eligible_count']} / {stats['total_count']})\n"
        )
        f.write(f"zero_loss_pct: {format_float(stats['zero_loss_pct'], 4)}\n")
        f.write(f"cap_pct: {format_float(stats['cap_pct'], 4)}\n")
        f.write(f"stability_score: {format_float(stats['stability_score'], 6)}\n\n")

        f.write("=== PF & RETURNS SUMMARY ===\n")
        f.write(
            f"median_pf_diag: {format_float(stats['median_pf_diag'], 4)}   "
            f"avg_pf_diag: {format_float(stats['avg_pf_diag'], 4)}\n"
        )
        f.write(
            f"median_pf_eff:  {format_float(stats['median_pf_eff'], 4)}   "
            f"avg_pf_eff:  {format_float(stats['avg_pf_eff'], 4)}\n"
        )
        f.write(f"median_trades:  {format_float(stats['median_trades'], 2)}\n")
        f.write(f"median_total_return: {format_float(stats['median_total_return'], 4)}\n\n")

        def write_params_block(title: str, params: Dict[str, float]):
            f.write(f"=== BEST PARAMS ({title}) ===\n")
            for key in PARAM_KEYS:
                if key in params:
                    f.write(f"{key}: {params[key]}\n")
            f.write("\n")

        write_params_block("SWING", swing_params)
        write_params_block("TREND", trend_params)
        write_params_block("BLENDED", blended_params)

        f.write(f"objective_score: {format_float(stats['objective_score'], 6)}\n")


# ---------- Core meta-optimization ----------

def meta_optimize(
    csv_A: Path,
    csv_B: Path,
    best_A: Path,
    best_B: Path,
    out_csv: Path,
    out_best: Path,
):
    dfA = pd.read_csv(csv_A)
    dfB = pd.read_csv(csv_B)

    # Summaries for regime inference
    sumA = summarize_per_ticker(dfA)
    sumB = summarize_per_ticker(dfB)
    trend_label, swing_label = infer_regimes(sumA, sumB)

    print("\nRegime inference:")
    print(f"  File A (per-ticker): {csv_A}")
    print(f"  File B (per-ticker): {csv_B}")
    print(f"  Inferred trend file label: {trend_label}")
    print(f"  Inferred swing file label: {swing_label}")

    # Merge on ticker
    df = dfA.merge(dfB, on="ticker", suffixes=("_A", "_B"))

    # Determine winner per ticker
    df["winner"] = df.apply(
        lambda r: "B" if r["ticker_score_B"] > r["ticker_score_A"] else "A",
        axis=1,
    )

    combined_rows = []
    for _, r in df.iterrows():
        row = {"ticker": r["ticker"]}

        if r["winner"] == "A":
            for col in df.columns:
                if col.endswith("_A") and col not in ("winner",):
                    base = col[:-2]
                    row[base] = r[col]
            row["regime"] = "trend" if trend_label == "A" else "swing"
        else:
            for col in df.columns:
                if col.endswith("_B") and col not in ("winner",):
                    base = col[:-2]
                    row[base] = r[col]
            row["regime"] = "trend" if trend_label == "B" else "swing"

        combined_rows.append(row)

    out_df = pd.DataFrame(combined_rows)

    # Ensure regime column is last
    cols = [c for c in out_df.columns if c != "regime"] + ["regime"]
    out_df = out_df[cols]

    out_df.to_csv(out_csv, index=False)

    # Regime counts
    regime_counts = out_df["regime"].value_counts()
    n_swing = int(regime_counts.get("swing", 0))
    n_trend = int(regime_counts.get("trend", 0))

    print("\nMeta-optimization complete.")
    print("Regime counts:")
    print(regime_counts)
    print(f"\nSaved combined per-ticker results to: {out_csv}")

    # Parse best params
    params_A = parse_best_params(best_A)
    params_B = parse_best_params(best_B)

    if swing_label == "A":
        swing_params = params_A
        trend_params = params_B
    else:
        swing_params = params_B
        trend_params = params_A

    blended_params = blend_params(swing_params, trend_params, n_swing, n_trend)

    # Meta stats from combined winners
    stats = compute_meta_stats(out_df)

    # Write meta_best.txt
    write_meta_best(out_best, stats, swing_params, trend_params, blended_params)
    print(f"Saved meta best report to: {out_best}\n")


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Meta-optimizer with regime inference and meta best report.")

    parser.add_argument("--dir", type=str, default="data",
                        help="Directory containing per-ticker CSVs and best*.txt files")
    parser.add_argument("--out_csv", type=str, default="meta_per_ticker.csv",
                        help="Output combined per-ticker CSV")
    parser.add_argument("--out_best", type=str, default="meta_best.txt",
                        help="Output meta best report")
    parser.add_argument("--csvA", type=str, default=None,
                        help="Optional: explicitly specify first per-ticker CSV")
    parser.add_argument("--csvB", type=str, default=None,
                        help="Optional: explicitly specify second per-ticker CSV")
    parser.add_argument("--bestA", type=str, default=None,
                        help="Optional: explicitly specify first best*.txt")
    parser.add_argument("--bestB", type=str, default=None,
                        help="Optional: explicitly specify second best*.txt")

    args = parser.parse_args()
    base_dir = Path(args.dir)

    # Per-ticker files
    if args.csvA and args.csvB:
        csv_A = Path(args.csvA)
        csv_B = Path(args.csvB)
    else:
        csv_A, csv_B = pick_two_latest_per_ticker(base_dir)
        print(f"Auto-selected per-ticker files:\n  A: {csv_A}\n  B: {csv_B}")

    # Best files
    if args.bestA and args.bestB:
        best_A = Path(args.bestA)
        best_B = Path(args.bestB)
    else:
        best_A, best_B = pick_two_latest_best(base_dir)
        print(f"Auto-selected best files:\n  A: {best_A}\n  B: {best_B}")

    out_csv = Path(args.out_csv)
    out_best = Path(args.out_best)

    meta_optimize(csv_A, csv_B, best_A, best_B, out_csv, out_best)


if __name__ == "__main__":
    main()
