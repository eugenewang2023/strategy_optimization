#!/usr/bin/env python3
"""
Vidya_RSI.py - High Density Aggression Edition
Unified Engine: Kaufman VIDYA + ZLEMA + Regime Slope.
Unified Reporting: All Overall Metrics, Penalty Diagnostics, and Sorted Ticker Tables.
"""

import math
import random
import argparse
import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import optuna

try:
    from optuna.integration import TQDMProgressBarCallback
except Exception:
    TQDMProgressBarCallback = None

# =================================================================================
# INDICATORS & ENGINE
# =================================================================================

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out

def vidya_ema(price: np.ndarray, length: int, smoothing: int) -> np.ndarray:
    n = len(price)
    vidya = np.full(n, np.nan)
    L, S = max(2, int(length)), max(1, int(smoothing))
    if n < L: return vidya
    alpha_base = 2.0 / (S + 1.0)
    signal = np.abs(pd.Series(price).diff(L).to_numpy())
    noise = pd.Series(np.abs(np.diff(price, prepend=price[0]))).rolling(L).sum().to_numpy()
    vi = signal / (noise + 1e-12)
    vidya[L-1] = price[L-1]
    for i in range(L, n):
        k = alpha_base * vi[i]
        prev = vidya[i-1] if np.isfinite(vidya[i-1]) else price[i-1]
        vidya[i] = price[i] * k + prev * (1.0 - k)
    return vidya

def zlema(series: np.ndarray, period: int) -> np.ndarray:
    pd_s = pd.Series(series)
    lag = (period - 1) // 2
    de_lagged = pd_s + (pd_s - pd_s.shift(lag))
    return de_lagged.ewm(span=period, adjust=False).mean().to_numpy()

@dataclass
class TradeStats:
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    profit_factor_raw: float = 0.0
    profit_factor_diag: float = 0.0
    profit_factor_diag_uncapped: float = 0.0

def backtest_vidya_engine(df: pd.DataFrame, **p) -> TradeStats:
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(c)
    reg_len = int(p["slowPeriod"] * p.get("regime_ratio", 3.0))
    if n < max(200, reg_len + 50): return TradeStats()

    ha_c = (o + h + l + c) / 4.0
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    atr = pd.Series(tr).rolling(25).mean().to_numpy()
    
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])
    fast = zlema(v_main, int(p["fastPeriod"]))
    slow = zlema(v_main, int(p["slowPeriod"]))
    regime_ema = pd.Series(c).ewm(span=reg_len, adjust=False).mean().to_numpy()
    
    hot_sm = pd.Series((fast - slow) / (atr + 1e-12)).rolling(5).mean().to_numpy()

    equity, gp, gl, trades = 1.0, 0.0, 0.0, 0
    in_pos, entry, bars_in_trade, cooldown_left = False, 0.0, 0, 0
    equity_curve = [1.0]

    for i in range(reg_len + 10, n - 2):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue
            
        regime_ok = (c[i] > regime_ema[i]) and (regime_ema[i] > regime_ema[i-5]) if p.get("use_regime") else True
        if (not in_pos) and (hot_sm[i-1] <= p["threshold"]) and (hot_sm[i] > p["threshold"]) and regime_ok:
            entry = float(o[i+1]) if p["fill_mode"] == "next_open" else float(c[i])
            if entry > 0:
                in_pos, trades = True, trades + 1
                bars_in_trade = 0

        elif in_pos:
            bars_in_trade += 1
            stop, tgt = entry - atr[i] * 3.0, entry + atr[i] * 3.6
            exit_p = None
            
            if l[i] <= stop: exit_p = float(stop)
            elif h[i] >= tgt: exit_p = float(tgt)
            elif p["time_stop_bars"] > 0 and bars_in_trade >= p["time_stop_bars"] and c[i] <= entry:
                exit_p = float(o[i+1] if i+1 < n else c[i])
            elif hot_sm[i] < -p["threshold"]:
                exit_p = float(o[i+1] if i+1 < n else c[i])
            elif i == n - 3: exit_p = float(c[i])

            if exit_p:
                pnl = ((exit_p - entry) / entry) - (p["commission"] * 2)
                equity *= (1.0 + pnl)
                equity_curve.append(equity)
                if pnl >= 0: gp += pnl
                else: 
                    gl += abs(pnl)
                    cooldown_left = p.get("cooldown_bars", 1)
                in_pos = False

    eff_gl = max(gl, (p['loss_floor'] * gp) if gp > 0 else p['loss_floor'])

    pf_diag_uncapped = (gp / eff_gl) if eff_gl > 0 else 0.0
    pf_diag = min(pf_diag_uncapped, float(p['pf_cap_score_only']))

    # raw PF (no floors, just real gp/gl)
    if gl > 0:
        pf_raw = gp / gl
    else:
        pf_raw = float("inf") if gp > 0 else 0.0

    return TradeStats(
        gp=float(gp),
        gl=float(gl),
        trades=int(trades),
        tot_ret=float(equity - 1.0),
        profit_factor_raw=float(pf_raw if np.isfinite(pf_raw) else 1e9),  # avoid inf in CSV
        profit_factor_diag=float(pf_diag),
        profit_factor_diag_uncapped=float(pf_diag_uncapped),
    )

# =================================================================================
# SCORING & UTILS
# =================================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def score_trial(st: TradeStats, args) -> float:
    if st.trades < args.min_trades: return 0.0
    pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (st.profit_factor_diag - args.pf_baseline)))
    tr_w = sigmoid(args.trades_k * (st.trades - args.trades_baseline))
    return float((args.weight_pf * pf_w + (1.0 - args.weight_pf) * tr_w) ** args.score_power)

# =================================================================================
# MAIN
# =================================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--files", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fill", type=str, default="next_open")
    ap.add_argument("--use-regime", action="store_true")
    
    # Penalties & Baseline flags
    ap.add_argument("--penalty", type=str, default="enabled")
    ap.add_argument("--penalty_ret_center", type=float, default=0.01)
    ap.add_argument("--penalty_ret_k", type=float, default=10.0)
    ap.add_argument("--ret_floor", type=float, default=-0.15)
    ap.add_argument("--ret_floor_k", type=float, default=2.0)
    ap.add_argument("--max_trades", type=int, default=100)
    ap.add_argument("--max_trades_k", type=float, default=0.1)
    ap.add_argument("--pf_floor", type=float, default=1.0)
    ap.add_argument("--pf_floor_k", type=float, default=5.0)
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)
    ap.add_argument("--pf-baseline", type=float, default=1.02)
    ap.add_argument("--pf-k", type=float, default=1.2)
    ap.add_argument("--trades-baseline", type=float, default=10.0)
    ap.add_argument("--trades-k", type=float, default=0.4)
    ap.add_argument("--weight-pf", type=float, default=0.4)
    ap.add_argument("--score-power", type=float, default=1.1)
    ap.add_argument("--min-trades", type=int, default=5) # Added explicitly to fix error
    ap.add_argument("--loss_floor", type=float, default=0.001)
    ap.add_argument("--threshold-fixed", type=float, default=0.04)
    ap.add_argument("--vol-floor-mult-fixed", type=float, default=0.55)
    ap.add_argument("--threshold_mode", type=str, default="fixed")
    ap.add_argument("--pf-cap", type=float, default=5.0)
    ap.add_argument("--coverage-target", type=float, default=0.85)
    ap.add_argument("--coverage-k", type=float, default=8.0)
    ap.add_argument("--opt-time-stop", action="store_true")
    ap.add_argument("--min-tp2sl", type=float, default=0.8)
    ap.add_argument("--opt-vidya", action="store_true")
    ap.add_argument("--opt-fastslow", action="store_true")

    # Map aliases for hyphenated args
    args_raw = ap.parse_args()
    # Normalize naming internally
    args = args_raw
    args.min_trades = args_raw.min_trades 

    random.seed(args.seed); np.random.seed(args.seed)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    data_list = [(f.stem, ensure_ohlc(pd.read_parquet(f))) for f in random.sample(files, min(len(files), args.files))]

    def objective(trial):
        p = {
            "vidya_len": trial.suggest_int("vl", 5, 25) if args.opt_vidya else 14,
            "vidya_smooth": trial.suggest_int("vs", 5, 40) if args.opt_vidya else 14,
            "fastPeriod": trial.suggest_int("fp", 5, 20) if args.opt_fastslow else 10,
            "slowPeriod": trial.suggest_int("sp", 21, 60) if args.opt_fastslow else 40,
            "time_stop_bars": trial.suggest_int("ts", 5, 40) if args.opt_time_stop else 15,
            "regime_ratio": trial.suggest_float("reg_ratio", 2.0, 5.0),
            "threshold": args.threshold_fixed, "vol_floor_mult": args.vol_floor_mult_fixed,
            "commission": args.commission_rate_per_side, "fill_mode": args.fill, 
            "use_regime": args.use_regime, "pf_cap": args.pf_cap, "cooldown_bars": 1
        }
        res = [backtest_vidya_engine(df, **p) for _, df in data_list]
        scores = [score_trial(st, args) for st in res]
        eligible = sum(1 for st in res if st.trades >= args.min_trades)
        cov_mult = sigmoid(args.coverage_k * ((eligible / len(data_list)) - args.coverage_target))
        return float(np.mean(scores) * cov_mult)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True, 
                   callbacks=[TQDMProgressBarCallback()] if TQDMProgressBarCallback else None)

    best_p = {**study.best_params, "threshold": args.threshold_fixed, "vol_floor_mult": args.vol_floor_mult_fixed, 
              "commission": args.commission_rate_per_side, "fill_mode": args.fill, "use_regime": args.use_regime, 
              "pf_cap": args.pf_cap, "time_stop_bars": study.best_params.get("ts", 15), 
              "fastPeriod": study.best_params.get("fp", 10), "slowPeriod": study.best_params.get("sp", 40), 
              "vidya_len": study.best_params.get("vl", 14), "vidya_smooth": study.best_params.get("vs", 14),
              "cooldown_bars": 1}

    per_ticker = []
    for name, df in data_list:
        st = backtest_vidya_engine(df, **best_p)
        per_ticker.append({
            "ticker": name, "profit_factor": st.pf_raw, "profit_factor_diag": st.pf_diag,
            "num_trades": st.trades, "ticker_score": score_trial(st, args),
            "total_return": st.tot_ret, "gross_profit": st.gp, "gross_loss": st.gl,
            "maxdd": st.maxdd, "eligible": 1 if st.trades >= args.min_trades else 0, "is_neg": st.num_neg
        })

    per_df = pd.DataFrame(per_ticker).sort_values(["profit_factor", "profit_factor_diag", "total_return", "ticker_score"], ascending=False)
    eligible_count = per_df["eligible"].sum()
    coverage = eligible_count / len(per_ticker)

    print("\n=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===")
    print(f"Avg Profit Factor (raw):      {per_df['profit_factor'].mean() if np.isfinite(per_df['profit_factor'].mean()) else float('inf')}")
    print(f"Avg Profit Factor (diag):     {per_df['profit_factor_diag'].mean():.6f}")
    print(f"Avg Trades / Ticker:          {per_df['num_trades'].mean():.3f}")
    print(f"Mean Ticker Score (eligible): {per_df[per_df['eligible']==1]['ticker_score'].mean():.6f}")
    print(f"Coverage (eligible/total):    {eligible_count}/{len(per_ticker)} = {coverage:.3f}")
    print(f"Penalty enabled:              {args.penalty} (soft)")
    print(f"Soft penalty center/k:        {args.penalty_ret_center} / {args.penalty_ret_k}")
    print(f"Tail-protection ret floor/k:  {args.ret_floor} / {args.ret_floor_k}")
    print(f"Max-trades penalty:           max_trades={args.max_trades}, k={args.max_trades_k}")
    print(f"PF-floor penalty:             pf_floor={args.pf_floor}, k={args.pf_floor_k}")
    print(f"cooldown_bars:                1 (opt=False) [loss-only]")
    print(f"time_stop_bars:               {best_p['time_stop_bars']} (0=disabled) (opt={args.opt_time_stop})")
    print(f"num_neg (return<=0):          {per_df['is_neg'].sum()}")
    print(f"commission_rate_per_side:     {args.commission_rate_per_side:.6f}")
    print(f"PF ROC baseline/k:            {args.pf_baseline:.3f} / {args.pf_k:.3f}")
    print(f"Trades ROC baseline/k:        {args.trades_baseline:.3f} / {args.trades_k:.3f}")
    print(f"weight_pf:                    {args.weight_pf:.3f}")
    print(f"score_power:                  {args.score_power:.3f}")
    print(f"min_trades gate:              {args.min_trades}")
    print(f"loss_floor (scoring):         {args.loss_floor}")
    print(f"threshold_mode:               {args.threshold_mode}")
    print(f"Score (single fill '{args.fill}'): {study.best_value:.6f}")

    print("\n=== BEST PARAMS ===")
    for k in sorted(study.best_params.keys()): print(f"{k}: {study.best_params[k]}")
    print(f"best_score (OPTUNA objective): {study.best_value}")
    print(f"fill_mode (final report): {args.fill}")

    print("\n=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===")
    print(per_df[["ticker", "profit_factor", "profit_factor_diag", "num_trades", "ticker_score", "total_return", "gross_profit", "gross_loss", "maxdd", "eligible"]].to_string(index=False))

    per_csv = Path("output") / f"per_ticker_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    per_csv.parent.mkdir(exist_ok=True); per_df.to_csv(per_csv, index=False)
    print(f"\nSaved per-ticker CSV to: {per_csv}")

if __name__ == "__main__":
    main()