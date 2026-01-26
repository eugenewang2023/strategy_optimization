#!/usr/bin/env python3
"""
adapt_half_RSI.py  —  with clean, timestamped result saving + module prefix
Saves:
- adapt_half_RSI_per_ticker_YYYYMMDD_HHMMSS.csv
- adapt_half_RSI_best_YYYYMMDD_HHMMSS.txt
"""

import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import optuna
import datetime


# =============================
# Module-specific prefix for output files
# =============================
MODULE_PREFIX = "adapt_half_RSI_"


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
        raise ValueError(f"Parquet missing required columns: {missing}")
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out


def heikin_ashi_from_real(o, h, l, c):
    ha_close = (o + h + l + c) / 4.0
    ha_open = np.empty_like(ha_close)
    ha_open[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_close)):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
    ha_high = np.maximum.reduce([h, ha_open, ha_close])
    ha_low = np.minimum.reduce([l, ha_open, ha_close])
    return ha_open, ha_high, ha_low, ha_close


def atr_wilder(high, low, close, period):
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


def crossover(a_prev, a_now, b_prev, b_now):
    return (a_prev <= b_prev) and (a_now > b_now)


def crossunder(a_prev, a_now, b_prev, b_now):
    return (a_prev >= b_prev) and (a_now < b_now)


# =============================
# half_RSI helpers
# =============================
def sma(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < length:
        return out
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    isn = np.isnan(x).astype(np.int32)
    nan_csum = np.cumsum(isn)
    for i in range(length - 1, n):
        nan_count = nan_csum[i] - (nan_csum[i - length] if i >= length else 0)
        if nan_count > 0:
            out[i] = np.nan
        else:
            s = csum[i] - (csum[i - length] if i >= length else 0.0)
            out[i] = s / length
    return out


def rsi_tv(price: np.ndarray, length: int) -> np.ndarray:
    n = len(price)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < 2:
        return out
    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    def rma(x, L):
        y = np.full(n, np.nan, dtype=float)
        if n < L:
            return y
        if np.isnan(x[:L]).any():
            start = None
            for i in range(L - 1, n):
                w = x[i - L + 1:i + 1]
                if not np.isnan(w).any():
                    y[i] = float(np.mean(w))
                    start = i + 1
                    break
            if start is None:
                return y
        else:
            y[L - 1] = float(np.mean(x[:L]))
            start = L
        alpha = 1.0 / float(L)
        for i in range(start, n):
            if np.isnan(x[i]) or np.isnan(y[i - 1]):
                y[i] = np.nan
            else:
                y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        return y

    ag = rma(gain, length)
    al = rma(loss, length)
    for i in range(n):
        if np.isnan(ag[i]) or np.isnan(al[i]):
            out[i] = np.nan
        else:
            if al[i] == 0.0:
                out[i] = 100.0 if ag[i] > 0 else 0.0
            else:
                rs = ag[i] / al[i]
                out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def shift_series(x: np.ndarray, shift: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if shift <= 0:
        return x.astype(float, copy=True)
    out[shift:] = x[:-shift]
    return out


# =============================
# Backtest core
# =============================
@dataclass
class TradeStats:
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    num_trades: int = 0
    total_return: float = 0.0
    maxdd: float = 0.0
    profit_factor: float = 0.0


def backtest_adapt_half_rsi_dynamic_engine(
    df: pd.DataFrame,
    *,
    atrPeriod: int,
    slMult: float,
    tpMult: float,
    commission_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool = True,
    trail_mode: str = "trail_only",
    close_on_sellSignal: bool = True,
    cooldown_bars: int = 0,
    time_stop_bars: int = 0,
    base_slow_window: int = 32,
    adapt_k: float = 0.1,
    min_window: int = 14,
    max_window: int = 64,
    shift: int = 0,
    smooth_len: int = 12,
) -> TradeStats:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    if n < max(atrPeriod + 2, 60):
        return TradeStats()

    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)

    real_atr = atr_wilder(h, l, c, atrPeriod)
    valid_idx = np.where(~np.isnan(real_atr))[0]
    if len(valid_idx) == 0:
        return TradeStats()
    start = int(valid_idx[0])

    def get_fill(i: int) -> Optional[float]:
        if fill_mode == "same_close":
            return float(c[i])
        if fill_mode == "next_open":
            return float(o[i + 1]) if (i + 1 < n) else None
        return float(c[i])

    sigClose = ha_c

    # Example adaptive logic: window size varies based on some factor (e.g. volatility)
    # Replace with your real adaptive mechanism
    base_slow = max(2, int(base_slow_window))

    # Compute ATR stats only on valid values
    atr_valid = real_atr[~np.isnan(real_atr)]
    atr_mean = np.nanmean(atr_valid)
    atr_std = np.nanstd(atr_valid)

    # Avoid division by zero or NaN
    if atr_std == 0 or np.isnan(atr_std):
        atr_std = 1e-8

    # Compute adapt_factor only where ATR is valid
    adapt_factor = np.full_like(real_atr, 1.0, dtype=float)
    valid = ~np.isnan(real_atr)
    adapt_factor[valid] = 1.0 + adapt_k * (real_atr[valid] - atr_mean) / atr_std

    # Clip and replace NaNs before casting
    adapt_slow = np.clip(base_slow * adapt_factor, min_window, max_window)
    adapt_slow = np.nan_to_num(adapt_slow, nan=base_slow).astype(int)

    # For simplicity we use fixed slow_window here — replace with per-bar adaptive
    slow_window = base_slow  # ← your adaptive logic goes here

    slow_len = max(2, int(slow_window))
    fast_len = max(1, int(round(slow_len / 2.0)))

    fast_rsi_raw = rsi_tv(sigClose, fast_len)
    slow_rsi_raw = rsi_tv(sigClose, slow_len)

    sh = max(0, int(shift))
    fast_rsi = shift_series(fast_rsi_raw, sh)
    slow_rsi = 100.0 - shift_series(slow_rsi_raw, sh)

    hot = fast_rsi - slow_rsi
    sm = max(1, int(smooth_len))
    hot_sm = sma(hot, sm)

    in_pos = False
    entry = 0.0
    trades = 0
    equity = 1.0
    peak = 1.0
    maxdd = 0.0
    gp = 0.0
    gl = 0.0
    cooldown_left = 0
    bars_in_trade = 0
    ts_bars = max(0, int(time_stop_bars))
    trail_active = False
    trail_stop = np.nan
    trail_high_since = np.nan
    trail_dist = np.nan

    def apply_commission(eq: float) -> float:
        return eq * (1.0 - commission_per_side)

    for i in range(start + 1, n):
        if np.isnan(real_atr[i]) or np.isnan(hot_sm[i - 1]) or np.isnan(hot_sm[i]):
            continue
        if cooldown_left > 0:
            cooldown_left -= 1

        cross_up = crossover(hot_sm[i - 1], hot_sm[i], 0.0, 0.0)
        cross_dn = crossunder(hot_sm[i - 1], hot_sm[i], 0.0, 0.0)

        buy_sig = bool(cross_up)
        sell_sig = bool(cross_dn)

        if (not in_pos) and buy_sig and (cooldown_left == 0):
            fill = get_fill(i)
            if fill is None:
                continue
            in_pos = True
            entry = float(fill)
            trades += 1
            equity = apply_commission(equity)
            bars_in_trade = 0
            trail_active = False
            trail_stop = np.nan
            trail_high_since = np.nan
            trail_dist = np.nan

        if in_pos:
            bars_in_trade += 1
            stop_level = l[i] - real_atr[i] * slMult
            tgt_level = h[i] + real_atr[i] * tpMult
            trail_dist = real_atr[i] * tpMult
            exit_now = False
            exit_price = None

            if use_trailing_exit:
                if (not trail_active) and (h[i] >= entry + trail_dist):
                    trail_active = True
                    trail_high_since = float(h[i])
                    trail_stop = trail_high_since - trail_dist
                elif trail_active:
                    trail_high_since = float(h[i]) if np.isnan(trail_high_since) else max(trail_high_since, float(h[i]))
                    trail_stop = trail_high_since - trail_dist

            hard_stop_enabled = (not use_trailing_exit) or (trail_mode == "trail_plus_hard_sl")
            if hard_stop_enabled and (l[i] <= stop_level):
                exit_now = True
                exit_price = float(stop_level)

            if (not exit_now) and use_trailing_exit and trail_active and (not np.isnan(trail_stop)):
                if l[i] <= trail_stop:
                    exit_now = True
                    exit_price = float(trail_stop)

            if (not exit_now) and (h[i] >= tgt_level):
                exit_now = True
                exit_price = float(tgt_level)

            if (not exit_now) and (ts_bars > 0) and (bars_in_trade >= ts_bars):
                fill = get_fill(i)
                if fill is not None:
                    unreal = (float(fill) - entry) / entry
                    if unreal <= 0.0:
                        exit_now = True
                        exit_price = float(fill)

            if (not exit_now) and close_on_sellSignal and sell_sig:
                fill = get_fill(i)
                if fill is not None:
                    exit_now = True
                    exit_price = float(fill)

            if exit_now and (exit_price is not None):
                equity = apply_commission(equity)
                pnl = (exit_price - entry) / entry
                equity *= (1.0 + pnl)
                if pnl >= 0:
                    gp += pnl
                else:
                    gl += abs(pnl)
                    cooldown_left = max(0, int(cooldown_bars))
                in_pos = False
                entry = 0.0
                bars_in_trade = 0
                trail_active = False
                trail_stop = np.nan
                trail_high_since = np.nan
                trail_dist = np.nan

        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0
        if dd < maxdd:
            maxdd = dd

    if in_pos:
        equity = apply_commission(equity)
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
    max_trades: int,
    max_trades_k: float,
    pf_floor: float,
    pf_floor_k: float,
    ret_floor: float,
    ret_floor_k: float,
) -> float:
    if st.num_trades < min_trades:
        return 0.0

    gp = max(st.gross_profit, 0.0)
    gl = max(st.gross_loss, 0.0)

    if gp > 0:
        effective_gl = max(gl, loss_floor * gp)
    else:
        effective_gl = max(gl, loss_floor)

    pf_eff = (gp / effective_gl) if effective_gl > 0 else 0.0
    pf_eff = min(pf_eff, pf_cap_score_only)

    pf_floor_mult = sigmoid(pf_floor_k * (pf_eff - pf_floor))
    pf_w = sigmoid(pf_k * (pf_eff - pf_baseline))
    tr_w = sigmoid(trades_k * (st.num_trades - trades_baseline))

    s = weight_pf * pf_w + (1.0 - weight_pf) * tr_w
    if score_power != 1.0:
        s = s ** score_power

    if penalty_enabled:
        ret_mult = sigmoid(penalty_ret_k * (st.total_return - penalty_ret_center))
        s *= ret_mult

    ret_floor_mult = sigmoid(ret_floor_k * (st.total_return - ret_floor))
    s *= ret_floor_mult

    over_mult = sigmoid(max_trades_k * (max_trades - st.num_trades))
    s *= over_mult
    s *= pf_floor_mult
    return float(s)


def evaluate_params_on_files(
    file_paths: List[Path],
    **kwargs
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], float, float]:
    per = []
    scores = []
    num_neg = 0

    for p in file_paths:
        try:
            df = pd.read_parquet(p)
            df = ensure_ohlc(df)
        except Exception:
            continue

        st = backtest_adapt_half_rsi_dynamic_engine(
            df,
            atrPeriod=kwargs["atrPeriod"],
            slMult=kwargs["slMultiplier"],
            tpMult=kwargs["tpMultiplier"],
            commission_per_side=kwargs["commission_rate_per_side"],
            fill_mode=kwargs["fill_mode"],
            use_trailing_exit=kwargs["use_trailing_exit"],
            trail_mode=kwargs["trail_mode"],
            close_on_sellSignal=kwargs["close_on_sellSignal"],
            cooldown_bars=kwargs["cooldown_bars"],
            time_stop_bars=kwargs["time_stop_bars"],
            base_slow_window=kwargs.get("base_slow_window", 32),
            adapt_k=kwargs.get("adapt_k", 0.1),
            shift=kwargs["shift"],
            smooth_len=kwargs["smooth_len"],
        )

        sc = score_from_stats(
            st,
            min_trades=kwargs["min_trades"],
            pf_baseline=kwargs["pf_baseline"],
            pf_k=kwargs["pf_k"],
            trades_baseline=kwargs["trades_baseline"],
            trades_k=kwargs["trades_k"],
            weight_pf=kwargs["weight_pf"],
            score_power=kwargs["score_power"],
            pf_cap_score_only=kwargs["pf_cap_score_only"],
            penalty_enabled=kwargs["penalty_enabled"],
            loss_floor=kwargs["loss_floor"],
            penalty_ret_center=kwargs["penalty_ret_center"],
            penalty_ret_k=kwargs["penalty_ret_k"],
            max_trades=kwargs["max_trades"],
            max_trades_k=kwargs["max_trades_k"],
            pf_floor=kwargs["pf_floor"],
            pf_floor_k=kwargs["pf_floor_k"],
            ret_floor=kwargs["ret_floor"],
            ret_floor_k=kwargs["ret_floor_k"],
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
            "maxdd": st.maxdd,
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
# CLI
# =============================
def parse_args():
    ap = argparse.ArgumentParser(description="Adaptive Half-RSI strategy optimization")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--files", type=int, default=200)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)

    # Scoring & penalty parameters — ALL required flags are here
    ap.add_argument("--pf-cap", type=float, default=10.0, dest="pf_cap_score_only")
    ap.add_argument("--pf-baseline", type=float, default=1.8)
    ap.add_argument("--pf-k", type=float, default=1.5)
    ap.add_argument("--trades-baseline", type=float, default=20.0)
    ap.add_argument("--trades-k", type=float, default=0.5)
    ap.add_argument("--weight-pf", type=float, default=0.9)
    ap.add_argument("--score-power", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=8)
    ap.add_argument("--penalty", action="store_true")
    ap.add_argument("--loss-floor", type=float, default=0.001, dest="loss_floor")
    ap.add_argument("--penalty-ret-center", type=float, default=-0.02)
    ap.add_argument("--penalty-ret-k", type=float, default=8.0)
    ap.add_argument("--ret-floor", type=float, default=0.0)
    ap.add_argument("--ret-floor-k", type=float, default=8.0)
    ap.add_argument("--max-trades", type=int, default=60)
    ap.add_argument("--max-trades-k", type=float, default=0.15)
    ap.add_argument("--pf-floor", type=float, default=1.0)
    ap.add_argument("--pf-floor-k", type=float, default=6.0)

    ap.add_argument("--fill", type=str, default="same_close", choices=["same_close", "next_open"])

    # Exits - FIXED: Changed from type=bool to action='store_true' for proper boolean handling
    ap.add_argument("--use_trailing_exit", action="store_true", default=True)
    ap.add_argument("--trail_mode", type=str, default="trail_only", choices=["trail_only", "trail_plus_hard_sl"])
    ap.add_argument("--close_on_sellSignal", action="store_true", default=True)

    # Cooldown / time stop
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--opt-cooldown", action="store_true")
    ap.add_argument("--time-stop", type=int, default=0)
    ap.add_argument("--opt-time-stop", action="store_true")

    # TP/SL constraint - ADDED FROM half_RSI.py
    ap.add_argument("--min-tp2sl", type=float, default=1.30)
    ap.add_argument("--tp2sl-auto", action="store_true")
    ap.add_argument("--tp2sl-base", type=float, default=1.25)
    ap.add_argument("--tp2sl-sr0", type=float, default=30.0)
    ap.add_argument("--tp2sl-k", type=float, default=0.01)
    ap.add_argument("--tp2sl-min", type=float, default=1.10)
    ap.add_argument("--tp2sl-max", type=float, default=1.80)

    # Coverage - ADDED FROM adapt_RSI.py
    ap.add_argument("--coverage-target", type=float, default=0.70)
    ap.add_argument("--coverage-k", type=float, default=12.0)

    # Modes & reporting
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report-both-fills", action="store_true")
    ap.add_argument("--atrPeriod-fixed", type=int, default=None)
    ap.add_argument("--slMultiplier-fixed", type=float, default=None)
    ap.add_argument("--tpMultiplier-fixed", type=float, default=None)

    # Adaptive half_RSI params
    ap.add_argument("--opt-adaptive", action="store_true")
    ap.add_argument("--base_slow_window-fixed", type=int, default=32)
    ap.add_argument("--adapt_k-fixed", type=float, default=0.1)
    ap.add_argument("--min_window-fixed", type=int, default=14)
    ap.add_argument("--max_window-fixed", type=int, default=64)
    ap.add_argument("--shift-fixed", type=int, default=0)
    ap.add_argument("--smooth_len-fixed", type=int, default=12)

    # ADDED: Boolean flags to disable trailing exit features if needed
    ap.add_argument("--no-use_trailing_exit", action="store_false", dest="use_trailing_exit")
    ap.add_argument("--no-close_on_sellSignal", action="store_false", dest="close_on_sellSignal")

    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # TP/SL constraint validation - ADDED FROM half_RSI.py
    if args.tp2sl_auto:
        if args.tp2sl_min <= 0 or args.tp2sl_max <= 0:
            raise SystemExit("--tp2sl-min and --tp2sl-max must be > 0")
        if args.tp2sl_min > args.tp2sl_max:
            raise SystemExit("--tp2sl-min must be <= --tp2sl-max")
    else:
        if args.min_tp2sl <= 0:
            raise SystemExit("--min-tp2sl must be > 0")

    # Coverage validation - ADDED FROM adapt_RSI.py
    if not (0.0 <= args.coverage_target <= 1.0):
        raise SystemExit("--coverage-target must be in [0, 1]")
    if args.coverage_k <= 0:
        raise SystemExit("--coverage-k must be > 0")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        raise SystemExit(f"No .parquet files found in {data_dir.resolve()}")

    file_paths = random.sample(all_files, args.files) if len(all_files) > args.files else all_files

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ROBUST_FILLS = ["same_close", "next_open"]

    # ADDED FROM half_RSI.py: TP/SL constraint function
    def min_tp2sl_eff_for(atr_val: int) -> float:
        if args.tp2sl_auto:
            v = args.tp2sl_base + args.tp2sl_k * (float(atr_val) - float(args.tp2sl_sr0))
            return max(args.tp2sl_min, min(args.tp2sl_max, v))
        return float(args.min_tp2sl)

    # ── REPORT ONLY ─────────────────────────────────────────────
    if args.report_only:
        required = [
            ("--atrPeriod-fixed", args.atrPeriod_fixed),
            ("--slMultiplier-fixed", args.slMultiplier_fixed),
            ("--tpMultiplier-fixed", args.tpMultiplier_fixed),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            raise SystemExit("Missing required flags for report-only")

        atrP = int(args.atrPeriod_fixed)
        slM = float(args.slMultiplier_fixed)
        tpM = float(args.tpMultiplier_fixed)

        min_eff = min_tp2sl_eff_for(atrP)
        if slM <= min_eff * tpM:
            raise SystemExit("Constraint violated: slMultiplier <= min_tp2sl_eff * tpMultiplier")

        fills = ROBUST_FILLS if args.report_both_fills else [args.fill]

        for fm in fills:
            _, overall, per, _, _ = evaluate_params_on_files(
                file_paths,
                atrPeriod=atrP,
                slMultiplier=slM,
                tpMultiplier=tpM,
                commission_rate_per_side=args.commission_rate_per_side,
                fill_mode=fm,
                use_trailing_exit=args.use_trailing_exit,
                trail_mode=args.trail_mode,
                close_on_sellSignal=args.close_on_sellSignal,
                cooldown_bars=int(args.cooldown),
                time_stop_bars=int(args.time_stop),
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
                max_trades=args.max_trades,
                max_trades_k=args.max_trades_k,
                pf_floor=args.pf_floor,
                pf_floor_k=args.pf_floor_k,
                ret_floor=args.ret_floor,
                ret_floor_k=args.ret_floor_k,
                base_slow_window=int(args.base_slow_window_fixed),
                adapt_k=float(args.adapt_k_fixed),
                shift=int(args.shift_fixed),
                smooth_len=int(args.smooth_len_fixed),
            )

            per_df = pd.DataFrame(per).sort_values(
                ["ticker_score", "total_return", "profit_factor", "num_trades"],
                ascending=False
            )

            csv_path = out_dir / f"{MODULE_PREFIX}per_ticker_report-only_{fm}_{now_str}.csv"
            per_df.to_csv(csv_path, index=False)

            txt_path = out_dir / f"{MODULE_PREFIX}report_summary_{fm}_{now_str}.txt"
            with txt_path.open("w", encoding="utf-8") as f:
                f.write(f"REPORT ONLY - fill = {fm}\n")
                f.write(f"Run: {now_str}\n\n")
                f.write(f"Mean ticker score   : {overall['mean_ticker_score']:.6f}\n")
                f.write(f"Avg PF (raw)        : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}\n")
                f.write(f"Avg trades/ticker   : {overall['avg_trades']:.2f}\n")
                f.write(f"Negative returns    : {overall['num_neg']}\n")

            print(f"Saved: {csv_path}")
            print(f"Saved: {txt_path}")

        print("\nFixed parameters:")
        print(f"  atrPeriod     : {atrP}")
        print(f"  slMultiplier  : {slM:.4f}")
        print(f"  tpMultiplier  : {tpM:.4f}")
        print(f"  min_tp2sl_eff : {min_eff:.4f}")
        print(f"  adapt_half_RSI: base_slow_window={args.base_slow_window_fixed}, adapt_k={args.adapt_k_fixed}, shift={args.shift_fixed}, smooth_len={args.smooth_len_fixed}")
        return

    # ── OPTIMIZATION ────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        atrPeriod = trial.suggest_int("atrPeriod", 5, 80)
        slMultiplier = trial.suggest_float("slMultiplier", 0.5, 12.0)
        tpMultiplier = trial.suggest_float("tpMultiplier", 0.5, 12.0)

        min_eff = min_tp2sl_eff_for(atrPeriod)
        if slMultiplier <= min_eff * tpMultiplier:
            raise optuna.TrialPruned()

        # Adaptive params (optimized if --opt-adaptive)
        base_slow_window = trial.suggest_int("base_slow_window", 20, 50) if args.opt_adaptive else int(args.base_slow_window_fixed)
        adapt_k = trial.suggest_float("adapt_k", 0.05, 0.3) if args.opt_adaptive else float(args.adapt_k_fixed)
        shift = trial.suggest_int("shift", 0, 3)
        smooth_len = trial.suggest_int("smooth_len", 3, 20)

        cooldown_bars = trial.suggest_int("cooldown", 0, 7) if args.opt_cooldown else int(args.cooldown)
        time_stop_bars = trial.suggest_int("time_stop", 0, 12) if args.opt_time_stop else int(args.time_stop)

        scores = []
        for fm in ROBUST_FILLS:
            s, _, _, _, _ = evaluate_params_on_files(
                file_paths,
                atrPeriod=atrPeriod,
                slMultiplier=slMultiplier,
                tpMultiplier=tpMultiplier,
                commission_rate_per_side=args.commission_rate_per_side,
                fill_mode=fm,
                use_trailing_exit=args.use_trailing_exit,
                trail_mode=args.trail_mode,
                close_on_sellSignal=args.close_on_sellSignal,
                cooldown_bars=cooldown_bars,
                time_stop_bars=time_stop_bars,
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
                max_trades=args.max_trades,
                max_trades_k=args.max_trades_k,
                pf_floor=args.pf_floor,
                pf_floor_k=args.pf_floor_k,
                ret_floor=args.ret_floor,
                ret_floor_k=args.ret_floor_k,
                base_slow_window=base_slow_window,
                adapt_k=adapt_k,
                shift=shift,
                smooth_len=smooth_len,
            )
            scores.append(s)

        return float(np.mean(scores)) if scores else -1e9

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)
    best_params["atrPeriod"] = int(best_params["atrPeriod"])
    best_params["slMultiplier"] = float(best_params["slMultiplier"])
    best_params["tpMultiplier"] = float(best_params["tpMultiplier"])
    best_params["base_slow_window"] = int(best_params.get("base_slow_window", args.base_slow_window_fixed))
    best_params["adapt_k"] = float(best_params.get("adapt_k", args.adapt_k_fixed))
    best_params["shift"] = int(best_params["shift"])
    best_params["smooth_len"] = int(best_params["smooth_len"])

    best_cooldown = int(best_params.get("cooldown", args.cooldown)) if not args.opt_cooldown else int(best_params.get("cooldown", 1))
    best_time_stop = int(best_params.get("time_stop", args.time_stop)) if not args.opt_time_stop else int(best_params.get("time_stop", 0))

    best_score_single, overall, per, _, _ = evaluate_params_on_files(
        file_paths,
        atrPeriod=best_params["atrPeriod"],
        slMultiplier=best_params["slMultiplier"],
        tpMultiplier=best_params["tpMultiplier"],
        commission_rate_per_side=args.commission_rate_per_side,
        fill_mode=args.fill,
        use_trailing_exit=args.use_trailing_exit,
        trail_mode=args.trail_mode,
        close_on_sellSignal=args.close_on_sellSignal,
        cooldown_bars=best_cooldown,
        time_stop_bars=best_time_stop,
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
        max_trades=args.max_trades,
        max_trades_k=args.max_trades_k,
        pf_floor=args.pf_floor,
        pf_floor_k=args.pf_floor_k,
        ret_floor=args.ret_floor,
        ret_floor_k=args.ret_floor_k,
        base_slow_window=best_params["base_slow_window"],
        adapt_k=best_params["adapt_k"],
        shift=best_params["shift"],
        smooth_len=best_params["smooth_len"],
    )

    per_df = pd.DataFrame(per).sort_values(
        ["ticker_score", "total_return", "profit_factor", "num_trades"],
        ascending=False
    )

    # ── SAVE RESULTS with prefix ────────────────────────────────────────────
    csv_path = out_dir / f"{MODULE_PREFIX}per_ticker_{now_str}.csv"
    per_df.to_csv(csv_path, index=False)

    summary_lines = [
        "=== BEST RESULT - Adaptive Half-RSI ===",
        f"Run timestamp         : {now_str}",
        f"Objective value       : {best.value:.6f}",
        "",
        "Best parameters:",
    ]
    for k, v in sorted(best_params.items()):
        summary_lines.append(f"  {k:18} : {v}")
    summary_lines.extend([
        "",
        "Performance:",
        f"  Mean ticker score   : {overall['mean_ticker_score']:.6f}",
        f"  Avg PF (raw)        : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}",
        f"  Avg trades/ticker   : {overall['avg_trades']:.2f}",
        f"  Negative returns    : {overall['num_neg']}",
        "",
        f"  Commission/side     : {args.commission_rate_per_side:.6f}",
        f"  Penalty enabled     : {args.penalty}",
        f"  min_tp2sl_eff       : {min_tp2sl_eff_for(best_params['atrPeriod']):.4f}",
    ])

    txt_path = out_dir / f"{MODULE_PREFIX}best_{now_str}.txt"
    txt_path.write_text("\n".join(summary_lines), encoding="utf-8")

    # ── Clean console summary - ASCII only ───────────────────────────────
    print("\n" + "="*60)
    print("          BEST RESULT - Adaptive Half-RSI")
    print("="*60)
    print(f"Objective value       : {best.value:.6f}")
    print(f"Mean ticker score     : {overall['mean_ticker_score']:.6f}")
    print(f"Avg PF (raw)          : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}")
    print(f"Avg trades/ticker     : {overall['avg_trades']:.2f}")
    print(f"\nSaved:")
    print(f"  CSV -> {csv_path}")
    print(f"  TXT -> {txt_path}")
    print("\n".join(summary_lines))


if __name__ == "__main__":
    main()