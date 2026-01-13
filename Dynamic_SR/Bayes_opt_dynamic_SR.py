#!/usr/bin/env python3
"""
Optuna_opt_dynamic_SR.py  (PARQUET-ONLY)

Features:
- Parquet only (*.parquet)
- Hard constraints:
    1) tpMultiplier >= 1.0
    2) slMultiplier > tpMultiplier
- Robustness check in objective:
    target = average score across fills: ["same_close", "next_open"]
- --loss_floor (default 0.001) used in PF scoring:
    effective_gl = max(gross_loss, loss_floor * gross_profit)  (if gross_profit>0)

Penalty (NON-BINARY / soft):
- Keep --penalty, but don't zero-out negative-return tickers.
- Instead multiply ticker score by:
      ret_mult = sigmoid(penalty_ret_k * (total_return - penalty_ret_center))
  Defaults:
      penalty_ret_center = 0.0
      penalty_ret_k      = 10.0
  so slightly negative returns get down-weighted, not killed.

Dependencies:
  pip install optuna pandas numpy pyarrow
"""

import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
import pandas as pd

import optuna


# =============================
# Helpers
# =============================
def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"Parquet missing required columns: {missing}. Found: {list(df.columns)}")
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out


def heikin_ashi_from_real(
    o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    prev_close = close[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - prev_close), abs(low[i] - prev_close))
        prev_close = close[i]

    atr = np.full(n, np.nan, dtype=float)
    if period <= 1:
        atr[:] = tr
        return atr
    if n < period:
        return atr

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def crossover(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return (a_prev <= b_prev) and (a_now > b_now)


def crossunder(a_prev: float, a_now: float, b_prev: float, b_now: float) -> bool:
    return (a_prev >= b_prev) and (a_now < b_now)


# =============================
# Backtest core (simplified Dynamic SR)
# =============================
@dataclass
class TradeStats:
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    num_trades: int = 0
    total_return: float = 0.0
    maxdd: float = 0.0
    profit_factor: float = 0.0


def backtest_dynamic_sr(
    df: pd.DataFrame,
    atrPeriod: int,
    slMult: float,
    tpMult: float,
    srBandMult: float,
    commission_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool = True,
    trail_mode: str = "trail_only",
    close_on_sellSignal: bool = True,
) -> TradeStats:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)
    if n < max(atrPeriod + 2, 60):
        return TradeStats()

    _, ha_h, ha_l, ha_c = heikin_ashi_from_real(o, h, l, c)
    real_atr = atr_wilder(h, l, c, atrPeriod)

    valid_idx = np.where(~np.isnan(real_atr))[0]
    if len(valid_idx) == 0:
        return TradeStats()
    start = int(valid_idx[0])

    # Dynamic SR line
    sr = np.full(n, np.nan, dtype=float)
    sr[start] = ha_c[start]
    for i in range(start + 1, n):
        band = srBandMult * real_atr[i]
        prev = sr[i - 1]
        if ha_c[i] > prev:
            sr[i] = max(prev, ha_l[i] + band)
        elif ha_c[i] < prev:
            sr[i] = min(prev, ha_h[i] - band)
        else:
            sr[i] = prev

    def get_fill(i: int) -> Optional[float]:
        if fill_mode == "same_close":
            return float(c[i])
        if fill_mode == "next_open":
            if i + 1 < n:
                return float(o[i + 1])
            return None
        return float(c[i])

    in_pos = False
    entry = 0.0
    stop = 0.0
    trail_stop = np.nan

    equity = 1.0
    peak = 1.0
    maxdd = 0.0

    gp = 0.0
    gl = 0.0
    trades = 0

    for i in range(start + 1, n):
        if np.isnan(sr[i - 1]) or np.isnan(sr[i]):
            continue

        buy_sig = crossover(ha_c[i - 1], ha_c[i], sr[i - 1], sr[i])
        sell_sig = crossunder(ha_c[i - 1], ha_c[i], sr[i - 1], sr[i])

        # update trailing stop
        if in_pos and use_trailing_exit and not np.isnan(real_atr[i]):
            candidate = c[i] - real_atr[i] * tpMult
            trail_stop = candidate if np.isnan(trail_stop) else max(trail_stop, candidate)

        # entry
        if (not in_pos) and buy_sig:
            fill = get_fill(i)
            if fill is None:
                continue
            in_pos = True
            entry = fill
            trades += 1

            equity *= (1.0 - commission_per_side)

            stop = entry - real_atr[i] * slMult if not np.isnan(real_atr[i]) else entry * 0.9
            trail_stop = np.nan

        # exits
        if in_pos:
            exit_now = False
            exit_price = None

            # stop hit (conservative)
            if l[i] <= stop:
                exit_now = True
                exit_price = stop

            # trailing stop
            if (not exit_now) and use_trailing_exit and (not np.isnan(trail_stop)):
                if l[i] <= trail_stop:
                    exit_now = True
                    exit_price = trail_stop

            # close on sell signal
            if (not exit_now) and close_on_sellSignal and sell_sig:
                fill = get_fill(i)
                if fill is not None:
                    exit_now = True
                    exit_price = fill

            if exit_now and exit_price is not None:
                equity *= (1.0 - commission_per_side)
                pnl = (exit_price - entry) / entry
                equity *= (1.0 + pnl)

                if pnl >= 0:
                    gp += pnl
                else:
                    gl += abs(pnl)

                in_pos = False
                entry = 0.0
                stop = 0.0
                trail_stop = np.nan

        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0
        if dd < maxdd:
            maxdd = dd

    if in_pos:
        equity *= (1.0 - commission_per_side)
        pnl = (float(c[-1]) - entry) / entry
        equity *= (1.0 + pnl)
        if pnl >= 0:
            gp += pnl
        else:
            gl += abs(pnl)

    pf = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    return TradeStats(
        gross_profit=gp,
        gross_loss=gl,
        num_trades=trades,
        total_return=equity - 1.0,
        maxdd=maxdd,
        profit_factor=pf,
    )


# =============================
# Scoring + evaluation
# =============================
def score_from_stats(
    st: TradeStats,
    *,
    min_trades: int,
    pf_baseline: float,
    pf_k: float,
    trades_baseline: float,
    trades_k: float,
    weight_pf: float,
    score_power: float,
    pf_cap_score_only: float,
    penalty_enabled: bool,
    loss_floor: float,
    penalty_ret_center: float,
    penalty_ret_k: float,
) -> float:
    if st.num_trades < min_trades:
        return 0.0

    gp = max(st.gross_profit, 0.0)
    gl = max(st.gross_loss, 0.0)

    # loss-floor PF for scoring
    if gp > 0:
        effective_gl = max(gl, loss_floor * gp)
    else:
        effective_gl = max(gl, loss_floor)

    pf_eff = (gp / effective_gl) if effective_gl > 0 else 0.0
    pf_eff = min(pf_eff, pf_cap_score_only)

    pf_w = sigmoid(pf_k * (pf_eff - pf_baseline))
    tr_w = sigmoid(trades_k * (st.num_trades - trades_baseline))

    s = weight_pf * pf_w + (1.0 - weight_pf) * tr_w
    if score_power != 1.0:
        s = s ** score_power

    # ✅ soft (non-binary) penalty on negative returns
    if penalty_enabled:
        ret_mult = sigmoid(penalty_ret_k * (st.total_return - penalty_ret_center))
        s *= ret_mult

    return float(s)


def evaluate_params_on_files(
    file_paths: List[Path],
    *,
    atrPeriod: int,
    slMultiplier: float,
    tpMultiplier: float,
    srBandMultiplier: float,
    commission_rate_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool,
    trail_mode: str,
    close_on_sellSignal: bool,
    # scoring opts
    min_trades: int,
    pf_baseline: float,
    pf_k: float,
    trades_baseline: float,
    trades_k: float,
    weight_pf: float,
    score_power: float,
    pf_cap_score_only: float,
    penalty_enabled: bool,
    loss_floor: float,
    penalty_ret_center: float,
    penalty_ret_k: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], float, float]:
    per: List[Dict[str, Any]] = []
    scores: List[float] = []
    num_neg = 0

    for p in file_paths:
        df = pd.read_parquet(p)  # PARQUET ONLY
        df = ensure_ohlc(df)

        st = backtest_dynamic_sr(
            df,
            atrPeriod=atrPeriod,
            slMult=slMultiplier,
            tpMult=tpMultiplier,
            srBandMult=srBandMultiplier,
            commission_per_side=commission_rate_per_side,
            fill_mode=fill_mode,
            use_trailing_exit=use_trailing_exit,
            trail_mode=trail_mode,
            close_on_sellSignal=close_on_sellSignal,
        )

        sc = score_from_stats(
            st,
            min_trades=min_trades,
            pf_baseline=pf_baseline,
            pf_k=pf_k,
            trades_baseline=trades_baseline,
            trades_k=trades_k,
            weight_pf=weight_pf,
            score_power=score_power,
            pf_cap_score_only=pf_cap_score_only,
            penalty_enabled=penalty_enabled,
            loss_floor=loss_floor,
            penalty_ret_center=penalty_ret_center,
            penalty_ret_k=penalty_ret_k,
        )

        if st.total_return <= 0:
            num_neg += 1

        per.append({
            "ticker": p.stem,
            "profit_factor": st.profit_factor,
            "num_trades": st.num_trades,
            "ticker_score": sc,
            "total_return": st.total_return,
            "gross_profit": st.gross_profit,
            "gross_loss": st.gross_loss,
        })
        scores.append(sc)

    mean_score = float(np.mean(scores)) if scores else 0.0
    pf_vals = [x["profit_factor"] for x in per if np.isfinite(x["profit_factor"])]
    pf_avg = float(np.mean(pf_vals)) if pf_vals else float("inf")
    trades_avg = float(np.mean([x["num_trades"] for x in per])) if per else 0.0

    overall = {
        "mean_ticker_score": mean_score,
        "avg_pf_raw": pf_avg,
        "avg_trades": trades_avg,
        "num_neg": f"{num_neg}/{len(per)}" if per else "0/0",
    }
    return mean_score, overall, per, pf_avg, trades_avg


# =============================
# CLI + Main
# =============================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="Folder containing per-ticker .parquet files.")
    ap.add_argument("--output_dir", type=str, default="output")

    ap.add_argument("--files", type=int, default=200, help="Max number of parquet files to use (random subset if more).")
    ap.add_argument("--trials", type=int, default=2000, help="Optuna trials.")
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)

    # scoring knobs
    ap.add_argument("--pf-cap", type=float, default=30.0, dest="pf_cap_score_only")
    ap.add_argument("--pf-baseline", type=float, default=1.8)
    ap.add_argument("--pf-k", type=float, default=1.5)
    ap.add_argument("--trades-baseline", type=float, default=8.0)
    ap.add_argument("--trades-k", type=float, default=5.0)
    ap.add_argument("--weight-pf", type=float, default=0.9)
    ap.add_argument("--score-power", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=8)
    ap.add_argument("--penalty", action="store_true")
    ap.add_argument("--loss_floor", type=float, default=0.001)

    # ✅ soft penalty knobs
    ap.add_argument("--penalty-ret-center", type=float, default=0.0,
                    help="Soft penalty center for total_return. 0.0 means break-even.")
    ap.add_argument("--penalty-ret-k", type=float, default=10.0,
                    help="Soft penalty steepness. Higher = harsher penalty against negative returns.")

    ap.add_argument("--fill", type=str, default="same_close", choices=["same_close", "next_open"],
                    help="Fill mode used for final reporting; objective uses robustness avg across both.")

    # exits
    ap.add_argument("--use_trailing_exit", type=bool, default=True)
    ap.add_argument("--trail_mode", type=str, default="trail_only", choices=["trail_only"])
    ap.add_argument("--close_on_sellSignal", type=bool, default=True)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.parquet"))  # PARQUET ONLY
    if not all_files:
        raise SystemExit(f"No .parquet files found in {data_dir.resolve()}")

    if len(all_files) > args.files:
        file_paths = random.sample(all_files, args.files)
    else:
        file_paths = all_files

    ROBUST_FILLS = ["same_close", "next_open"]

    def objective(trial: optuna.Trial) -> float:
        atrPeriod = trial.suggest_int("atrPeriod", 10, 100)
        slMultiplier = trial.suggest_float("slMultiplier", 1.5, 12.0)
        tpMultiplier = trial.suggest_float("tpMultiplier", 1.0, 8.0)  # constraint #1
        srBandMultiplier = trial.suggest_float("srBandMultiplier", 0.4, 3.0)

        # constraint #2: enforce SL > TP
        if slMultiplier <= tpMultiplier:
            raise optuna.TrialPruned()
    
        scores = []
        for fm in ROBUST_FILLS:
            s, _, _, _, _ = evaluate_params_on_files(
                file_paths,
                atrPeriod=atrPeriod,
                slMultiplier=slMultiplier,
                tpMultiplier=tpMultiplier,
                srBandMultiplier=srBandMultiplier,
                commission_rate_per_side=args.commission_rate_per_side,
                fill_mode=fm,
                use_trailing_exit=args.use_trailing_exit,
                trail_mode=args.trail_mode,
                close_on_sellSignal=args.close_on_sellSignal,
                min_trades=args.min_trades,
                pf_baseline=args.pf_baseline,
                pf_k=args.pf_k,
                trades_baseline=args.trades_baseline,
                trades_k=args.trades_k,
                weight_pf=args.weight_pf,
                score_power=args.score_power,
                pf_cap_score_only=args.pf_cap_score_only,
                penalty_enabled=args.penalty,
                loss_floor=args.loss_floor,
                penalty_ret_center=args.penalty_ret_center,
                penalty_ret_k=args.penalty_ret_k,
            )
            scores.append(s)

        return float(np.mean(scores)) if scores else -1e9

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = best.params
    best_params["atrPeriod"] = int(best_params["atrPeriod"])
    best_params["slMultiplier"] = float(best_params["slMultiplier"])
    best_params["tpMultiplier"] = float(best_params["tpMultiplier"])
    best_params["srBandMultiplier"] = float(best_params["srBandMultiplier"])

    # Final reporting on chosen fill
    best_score_single, overall, per, pf_avg, trades_avg = evaluate_params_on_files(
        file_paths,
        atrPeriod=best_params["atrPeriod"],
        slMultiplier=best_params["slMultiplier"],
        tpMultiplier=best_params["tpMultiplier"],
        srBandMultiplier=best_params["srBandMultiplier"],
        commission_rate_per_side=args.commission_rate_per_side,
        fill_mode=args.fill,
        use_trailing_exit=args.use_trailing_exit,
        trail_mode=args.trail_mode,
        close_on_sellSignal=args.close_on_sellSignal,
        min_trades=args.min_trades,
        pf_baseline=args.pf_baseline,
        pf_k=args.pf_k,
        trades_baseline=args.trades_baseline,
        trades_k=args.trades_k,
        weight_pf=args.weight_pf,
        score_power=args.score_power,
        pf_cap_score_only=args.pf_cap_score_only,
        penalty_enabled=args.penalty,
        loss_floor=args.loss_floor,
        penalty_ret_center=args.penalty_ret_center,
        penalty_ret_k=args.penalty_ret_k,
    )

    per_df = pd.DataFrame(per).sort_values(
        ["profit_factor", "total_return", "ticker_score", "num_trades"],
        ascending=False
    )
    per_csv = out_dir / "per_ticker_summary.csv"
    per_df.to_csv(per_csv, index=False)

    print("\n=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===")
    print(f"Avg Profit Factor (raw):      {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else float('inf')}")
    print(f"Avg Trades / Ticker:          {overall['avg_trades']:.3f}")
    print(f"Mean Ticker Score:            {overall['mean_ticker_score']:.6f}")
    print(f"Penalty enabled:              {args.penalty} (soft)")
    print(f"Soft penalty center/k:        {args.penalty_ret_center} / {args.penalty_ret_k}")
    print(f"num_neg (return<=0):          {overall['num_neg']}")
    print(f"commission_rate_per_side:     {args.commission_rate_per_side:.6f}")
    print(f"PF ROC baseline/k:            {args.pf_baseline:.3f} / {args.pf_k:.3f}")
    print(f"Trades ROC baseline/k:        {args.trades_baseline:.3f} / {args.trades_k:.3f}")
    print(f"weight_pf:                    {args.weight_pf:.3f}")
    print(f"score_power:                  {args.score_power:.3f}")
    print(f"min_trades gate:              {args.min_trades}")
    print(f"loss_floor (scoring):         {args.loss_floor}")
    print(f"Score (single fill '{args.fill}'): {best_score_single:.6f}")

    print("\n=== BEST PARAMS ===")
    print(f"atrPeriod: {best_params['atrPeriod']}")
    print(f"slMultiplier: {best_params['slMultiplier']}")
    print(f"tpMultiplier: {best_params['tpMultiplier']}")
    print(f"srBandMultiplier: {best_params['srBandMultiplier']}")
    print(f"best_score (ROBUST avg fills): {best.value}")
    print(f"fill_mode (report): {args.fill}")
    print(f"pf_cap_score_only: {args.pf_cap_score_only}")
    print(f"Saved per-ticker CSV to: {per_csv}")

    print("\n=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===")
    print(per_df[["ticker", "profit_factor", "num_trades", "ticker_score", "total_return", "gross_profit", "gross_loss"]].to_string(index=False))


if __name__ == "__main__":
    main()
