#!/usr/bin/env python3
"""
adapt_RSI.py  —  with clean, timestamped result saving + module prefix
Saves:
- adapt_RSI_per_ticker_YYYYMMDD_HHMMSS.csv
- adapt_RSI_best_YYYYMMDD_HHMMSS.txt
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

import sys
import io

# Force UTF-8 output on Windows to avoid charmaps errors
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# =============================
# Module-specific prefix for output files
# =============================
MODULE_PREFIX = "adapt_RSI_"


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


def shift_series(x: np.ndarray, shift: int) -> np.ndarray:
    """Shift series forward by 'shift' positions (like TradingView's offset)"""
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    if shift <= 0:
        return x.astype(float, copy=True)
    out[shift:] = x[:-shift]
    return out


# =============================
# Signal helpers
# =============================
def sma(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    L = int(length)
    if L <= 0 or n < L:
        return out
    csum = np.cumsum(np.nan_to_num(x, nan=0.0))
    isn = np.isnan(x).astype(np.int32)
    nan_csum = np.cumsum(isn)
    for i in range(L - 1, n):
        nan_count = nan_csum[i] - (nan_csum[i - L] if i >= L else 0)
        if nan_count > 0:
            out[i] = np.nan
        else:
            s = csum[i] - (csum[i - L] if i >= L else 0.0)
            out[i] = s / L
    return out


def rsi_tv(price: np.ndarray, length: int) -> np.ndarray:
    n = len(price)
    out = np.full(n, np.nan, dtype=float)
    L = int(length)
    if L <= 0 or n < 2:
        return out
    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    def rma(x, period):
        y = np.full(n, np.nan, dtype=float)
        if n < period:
            return y
        if np.isnan(x[:period]).any():
            start = None
            for i in range(period - 1, n):
                w = x[i - period + 1:i + 1]
                if not np.isnan(w).any():
                    y[i] = float(np.mean(w))
                    start = i + 1
                    break
            if start is None:
                return y
        else:
            y[period - 1] = float(np.mean(x[:period]))
            start = period
        alpha = 1.0 / float(period)
        for i in range(start, n):
            if np.isnan(x[i]) or np.isnan(y[i - 1]):
                y[i] = np.nan
            else:
                y[i] = y[i - 1] + alpha * (x[i] - y[i - 1])
        return y

    ag = rma(gain, L)
    al = rma(loss, L)
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


def ema_tv(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=float)
    L = int(length)
    if L <= 0 or n < L:
        return out
    alpha = 2.0 / float(L + 1)
    start = None
    for i in range(L - 1, n):
        w = x[i - L + 1:i + 1]
        if not np.isnan(w).any():
            out[i] = float(np.mean(w))
            start = i + 1
            break
    if start is None:
        return out
    for i in range(start, n):
        if np.isnan(x[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
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
    profit_factor_diag: float = 0.0
    trend_metric: float = 0.0


def backtest_adapt_rsi_dynamic_engine(
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
    cooldown_bars: int,
    time_stop_bars: int,
    basePeriod: int,
    minPeriod: int,
    maxPeriod: int,
    fastPeriod: int,
    slowPeriod: int,
    smooth_len: int,
    threshold: float,
    threshold_mode: str,
    threshold_floor: float,
    threshold_std_mult: float,
    vol_floor_mult: float,
    vol_floor_len: int,
    loss_floor: float,
    pf_cap_score_only: float,
    shift: int = 0,
) -> TradeStats:
    df = ensure_ohlc(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    if n < max(int(atrPeriod) + 2, 200):
        return TradeStats()

    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)
    sigClose = ha_c

    atr = atr_wilder(h, l, c, int(atrPeriod))
    valid_idx = np.where(~np.isnan(atr))[0]
    if len(valid_idx) == 0:
        return TradeStats()
    start = int(valid_idx[0])

    # Trend metric (EMA separation normalized by ATR)
    ema_fast_tr = ema_tv(sigClose, 50)
    ema_slow_tr = ema_tv(sigClose, 200)
    sep = np.abs(ema_fast_tr - ema_slow_tr) / (atr + 1e-12)
    trend_metric = float(np.nanmean(sep)) if np.isfinite(sep).any() else 0.0

    def get_fill(i: int) -> Optional[float]:
        if fill_mode == "same_close":
            return float(c[i])
        if fill_mode == "next_open":
            return float(o[i + 1]) if (i + 1 < n) else None
        return float(c[i])

    # Adaptive RSI pipeline
    baseP = max(2, int(basePeriod))
    minP = max(2, int(minPeriod))
    maxP = max(minP, int(maxPeriod))
    fp = max(1, int(fastPeriod))
    sp = max(fp + 1, int(slowPeriod))
    sm = max(1, int(smooth_len))
    sh = max(0, int(shift))

    base_rsi = rsi_tv(sigClose, baseP)
    k_sigmoid = 1.0 / (1.0 + np.exp(-0.1 * (base_rsi - 50.0)))
    adapt_p = (k_sigmoid * float(maxP - minP) + float(minP))
    adapt_p = np.clip(adapt_p, float(minP), float(maxP))
    adapt_p_int = np.where(np.isfinite(adapt_p), np.rint(adapt_p), float(minP)).astype(int)

    rsi_map: Dict[int, np.ndarray] = {}
    for L in range(minP, maxP + 1):
        rsi_map[L] = rsi_tv(sigClose, L)

    adaptive_rsi = np.full(n, np.nan, dtype=float)
    for i in range(n):
        p = int(adapt_p_int[i])
        if p in rsi_map:
            adaptive_rsi[i] = rsi_map[p][i]

    fast_rsi_raw = ema_tv(adaptive_rsi, fp)
    slow_rsi_raw = ema_tv(adaptive_rsi, sp)
    
    # Apply shift if specified
    if sh > 0:
        fast_rsi = shift_series(fast_rsi_raw, sh)
        slow_rsi = shift_series(slow_rsi_raw, sh)
    else:
        fast_rsi = fast_rsi_raw
        slow_rsi = slow_rsi_raw
    
    hot = fast_rsi - slow_rsi
    hot_sm = sma(hot, sm)

    # Dynamic threshold
    if threshold_mode == "dynamic":
        s_std = pd.Series(hot_sm).rolling(int(vol_floor_len), min_periods=int(vol_floor_len)).std().to_numpy()
        dyn_thresh = np.maximum(float(threshold_floor), np.nan_to_num(s_std) * float(threshold_std_mult))
    else:
        dyn_thresh = np.full(n, float(threshold), dtype=float)

    # Volatility floor
    atr_ma = sma(atr, int(vol_floor_len))
    vol_ok = atr > (atr_ma * float(vol_floor_mult))

    # Trade engine
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
    has_hit_be = False
    
    # Trailing exit variables
    trail_active = False
    trail_stop = np.nan
    trail_high_since = np.nan
    trail_dist = np.nan

    def apply_commission(eq: float) -> float:
        return eq * (1.0 - commission_per_side)

    loop_start = max(start + 2, int(vol_floor_len) + 2, 2)

    for i in range(loop_start, n - 1):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if np.isnan(hot_sm[i - 1]) or np.isnan(hot_sm[i]) or np.isnan(dyn_thresh[i]) or np.isnan(atr[i]):
            continue

        th = float(dyn_thresh[i])

        # Signal detection - use crossover/crossunder logic for consistency
        cross_up = crossover(hot_sm[i - 1], hot_sm[i], -th, -th)
        cross_dn = crossunder(hot_sm[i - 1], hot_sm[i], th, th)
        
        buy_sig = bool(cross_up)
        sell_sig = bool(cross_dn)

        if (not in_pos) and buy_sig and bool(vol_ok[i]):
            fill = get_fill(i)
            if fill is None:
                continue
            entry = float(fill)
            in_pos = True
            trades += 1
            equity = apply_commission(equity)
            bars_in_trade = 0
            has_hit_be = False
            trail_active = False
            trail_stop = np.nan
            trail_high_since = np.nan
            trail_dist = np.nan
            continue

        if in_pos:
            bars_in_trade += 1
            pnl_atr = (c[i] - entry) / (atr[i] + 1e-12)
            if pnl_atr > 1.5:
                has_hit_be = True

            # Basic stop and target levels
            stop_level = entry if has_hit_be else (entry - atr[i] * slMult)
            tgt_level = entry + atr[i] * tpMult
            
            # Trailing exit setup
            trail_dist = atr[i] * tpMult
            
            if use_trailing_exit:
                if (not trail_active) and (h[i] >= entry + trail_dist):
                    trail_active = True
                    trail_high_since = float(h[i])
                    trail_stop = trail_high_since - trail_dist
                elif trail_active:
                    trail_high_since = float(h[i]) if np.isnan(trail_high_since) else max(trail_high_since, float(h[i]))
                    trail_stop = trail_high_since - trail_dist

            exit_now = False
            exit_price: Optional[float] = None

            # Check exits in priority order
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

            if (not exit_now) and i == n - 2:
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
                has_hit_be = False
                trail_active = False
                trail_stop = np.nan
                trail_high_since = np.nan
                trail_dist = np.nan

        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0
        if dd < maxdd:
            maxdd = dd

    pf_raw = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    if gp > 0:
        effective_gl = max(gl, loss_floor * gp)
    else:
        effective_gl = max(gl, loss_floor)
    pf_diag = (gp / effective_gl) if effective_gl > 0 else 0.0
    pf_diag = min(pf_diag, float(pf_cap_score_only))

    return TradeStats(
        gross_profit=float(gp),
        gross_loss=float(gl),
        num_trades=int(trades),
        total_return=float(equity - 1.0),
        maxdd=float(maxdd),
        profit_factor=float(pf_raw),
        profit_factor_diag=float(pf_diag),
        trend_metric=float(trend_metric),
    )


# =============================
# Scoring + evaluation
# =============================
def score_from_stats(st: TradeStats, **kwargs) -> float:
    if st.num_trades < kwargs["min_trades"]:
        return float("nan")

    gp = max(st.gross_profit, 0.0)
    gl = max(st.gross_loss, 0.0)

    if gp > 0:
        effective_gl = max(gl, kwargs["loss_floor"] * gp)
    else:
        effective_gl = max(gl, kwargs["loss_floor"])

    pf_eff = (gp / effective_gl) if effective_gl > 0 else 0.0
    pf_eff = min(pf_eff, kwargs["pf_cap_score_only"])

    pf_floor_mult = sigmoid(kwargs["pf_floor_k"] * (pf_eff - kwargs["pf_floor"]))
    pf_w = sigmoid(kwargs["pf_k"] * (pf_eff - kwargs["pf_baseline"]))
    tr_w = sigmoid(kwargs["trades_k"] * (st.num_trades - kwargs["trades_baseline"]))

    s = kwargs["weight_pf"] * pf_w + (1.0 - kwargs["weight_pf"]) * tr_w
    if kwargs["score_power"] != 1.0:
        s = s ** kwargs["score_power"]

    if kwargs["penalty_enabled"]:
        ret_mult = sigmoid(kwargs["penalty_ret_k"] * (st.total_return - kwargs["penalty_ret_center"]))
        s *= ret_mult

    ret_floor_mult = sigmoid(kwargs["ret_floor_k"] * (st.total_return - kwargs["ret_floor"]))
    s *= ret_floor_mult

    over_mult = sigmoid(kwargs["max_trades_k"] * (kwargs["max_trades"] - st.num_trades))
    s *= over_mult
    s *= pf_floor_mult

    trend_mult = 0.5 + 0.5 * sigmoid(kwargs["trend_k"] * (st.trend_metric - kwargs["trend_center"]))
    s *= trend_mult
    return float(s)


def evaluate_params_on_files(file_paths: List[Path], **kwargs) -> Tuple[float, Dict, List[Dict], float, float, float, int]:
    per = []
    eligible_scores = []
    num_neg = 0
    eligible_count = 0

    for p in file_paths:
        try:
            df = pd.read_parquet(p)
            df = ensure_ohlc(df)
        except Exception:
            continue

        st = backtest_adapt_rsi_dynamic_engine(
            df,
            atrPeriod=kwargs["atrPeriod"],
            slMult=kwargs["slMultiplier"],
            tpMult=kwargs["tpMultiplier"],
            commission_per_side=kwargs["commission_rate_per_side"],
            fill_mode=kwargs["fill_mode"],
            use_trailing_exit=kwargs.get("use_trailing_exit", True),
            trail_mode=kwargs.get("trail_mode", "trail_only"),
            close_on_sellSignal=kwargs.get("close_on_sellSignal", True),
            cooldown_bars=kwargs["cooldown_bars"],
            time_stop_bars=kwargs["time_stop_bars"],
            basePeriod=kwargs["basePeriod"],
            minPeriod=kwargs["minPeriod"],
            maxPeriod=kwargs["maxPeriod"],
            fastPeriod=kwargs["fastPeriod"],
            slowPeriod=kwargs["slowPeriod"],
            smooth_len=kwargs["smooth_len"],
            threshold=kwargs["threshold"],
            threshold_mode=kwargs["threshold_mode"],
            threshold_floor=kwargs["threshold_floor"],
            threshold_std_mult=kwargs["threshold_std_mult"],
            vol_floor_mult=kwargs["vol_floor_mult"],
            vol_floor_len=kwargs["vol_floor_len"],
            loss_floor=kwargs["loss_floor"],
            pf_cap_score_only=kwargs["pf_cap_score_only"],
            shift=kwargs.get("shift", 0),
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
            trend_center=kwargs.get("trend_center", 0.80),
            trend_k=kwargs.get("trend_k", 3.0),
        )

        is_eligible = (st.num_trades >= kwargs["min_trades"])
        if is_eligible:
            eligible_count += 1
            if np.isfinite(sc):
                eligible_scores.append(float(sc))

        if st.total_return <= 0:
            num_neg += 1

        per.append({
            "ticker": p.stem,
            "profit_factor": st.profit_factor,
            "profit_factor_diag": st.profit_factor_diag,
            "trend_metric": st.trend_metric,
            "num_trades": st.num_trades,
            "ticker_score": sc,
            "total_return": st.total_return,
            "gross_profit": st.gross_profit,
            "gross_loss": st.gross_loss,
            "maxdd": st.maxdd,
            "eligible": bool(is_eligible),
        })

    mean_score = float(np.mean(eligible_scores)) if eligible_scores else 0.0
    pf_raw_vals = [x["profit_factor"] for x in per if np.isfinite(x["profit_factor"])]
    pf_raw_avg = float(np.mean(pf_raw_vals)) if pf_raw_vals else float("inf")
    pf_diag_vals = [x["profit_factor_diag"] for x in per if np.isfinite(x["profit_factor_diag"])]
    pf_diag_avg = float(np.mean(pf_diag_vals)) if pf_diag_vals else 0.0
    trades_avg = float(np.mean([x["num_trades"] for x in per])) if per else 0.0
    total = len(per)
    coverage = (eligible_count / total) if total > 0 else 0.0

    overall = {
        "mean_ticker_score": mean_score,
        "avg_pf_raw": pf_raw_avg,
        "avg_pf_diag": pf_diag_avg,
        "avg_trades": trades_avg,
        "num_neg": f"{num_neg}/{len(per)}" if per else "0/0",
        "eligible_count": eligible_count,
        "coverage": coverage,
    }
    return mean_score, overall, per, pf_raw_avg, trades_avg, coverage, eligible_count


# =============================
# CLI - UPDATED TO ACCEPT RANGE PARAMETERS
# =============================
def parse_args():
    ap = argparse.ArgumentParser(description="Adaptive RSI strategy optimization")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--files", type=int, default=200)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)

    # Scoring
    ap.add_argument("--pf-cap", type=float, default=12.0, dest="pf_cap_score_only")
    ap.add_argument("--pf-baseline", type=float, default=1.8)
    ap.add_argument("--pf-k", type=float, default=1.5)
    ap.add_argument("--trades-baseline", type=float, default=20.0)
    ap.add_argument("--trades-k", type=float, default=0.5)
    ap.add_argument("--weight-pf", type=float, default=0.9)
    ap.add_argument("--score-power", type=float, default=1.0)
    ap.add_argument("--min-trades", type=int, default=8)
    ap.add_argument("--penalty", action="store_true")
    ap.add_argument("--loss_floor", type=float, default=0.001)
    ap.add_argument("--penalty-ret-center", type=float, default=-0.02)
    ap.add_argument("--penalty-ret-k", type=float, default=8.0)
    ap.add_argument("--ret-floor", type=float, default=0.0)
    ap.add_argument("--ret-floor-k", type=float, default=8.0)
    ap.add_argument("--max-trades", type=int, default=60)
    ap.add_argument("--max-trades-k", type=float, default=0.15)
    ap.add_argument("--pf-floor", type=float, default=1.0)
    ap.add_argument("--pf-floor-k", type=float, default=6.0)

    ap.add_argument("--fill", type=str, default="same_close", choices=["same_close", "next_open"])

    # Exits
    ap.add_argument("--use_trailing_exit", type=bool, default=True)
    ap.add_argument("--trail_mode", type=str, default="trail_only", choices=["trail_only", "trail_plus_hard_sl"])
    ap.add_argument("--close_on_sellSignal", type=bool, default=True)

    # Cooldown / time stop
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--opt-cooldown", action="store_true")
    ap.add_argument("--time-stop", type=int, default=0)
    ap.add_argument("--opt-time-stop", action="store_true")

    # TP/SL constraint
    ap.add_argument("--min-tp2sl", type=float, default=1.30)
    ap.add_argument("--tp2sl-auto", action="store_true")
    ap.add_argument("--tp2sl-base", type=float, default=1.25)
    ap.add_argument("--tp2sl-sr0", type=float, default=30.0)
    ap.add_argument("--tp2sl-k", type=float, default=0.01)
    ap.add_argument("--tp2sl-min", type=float, default=1.10)
    ap.add_argument("--tp2sl-max", type=float, default=1.80)

    # Coverage
    ap.add_argument("--coverage-target", type=float, default=0.70)
    ap.add_argument("--coverage-k", type=float, default=12.0)

    # Modes
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--report-both-fills", action="store_true")
    ap.add_argument("--report-only", action="store_true")

    # ADDED: Range parameters for optimization
    ap.add_argument("--basePeriod-min", type=int)
    ap.add_argument("--basePeriod-max", type=int)
    ap.add_argument("--minPeriod-min", type=int)
    ap.add_argument("--minPeriod-max", type=int)
    ap.add_argument("--maxPeriod-min", type=int)
    ap.add_argument("--maxPeriod-max", type=int)
    ap.add_argument("--fastPeriod-min", type=int)
    ap.add_argument("--fastPeriod-max", type=int)
    ap.add_argument("--slowPeriod-min", type=int)
    ap.add_argument("--slowPeriod-max", type=int)
    ap.add_argument("--smooth_len-min", type=int)
    ap.add_argument("--smooth_len-max", type=int)
    ap.add_argument("--shift-min", type=int)
    ap.add_argument("--shift-max", type=int)
    ap.add_argument("--threshold_floor-min", type=float)
    ap.add_argument("--threshold_floor-max", type=float)
    ap.add_argument("--threshold_std_mult-min", type=float)
    ap.add_argument("--threshold_std_mult-max", type=float)
    ap.add_argument("--atrPeriod-min", type=int)
    ap.add_argument("--atrPeriod-max", type=int)
    ap.add_argument("--slMultiplier-min", type=float)
    ap.add_argument("--slMultiplier-max", type=float)
    ap.add_argument("--tpMultiplier-min", type=float)
    ap.add_argument("--tpMultiplier-max", type=float)
    ap.add_argument("--cooldown-min", type=int)
    ap.add_argument("--cooldown-max", type=int)
    ap.add_argument("--time_stop-min", type=int)
    ap.add_argument("--time_stop-max", type=int)

    # Fixed params (for backwards compatibility)
    ap.add_argument("--atrPeriod-fixed", type=int, default=25)
    ap.add_argument("--slMultiplier-fixed", type=float, default=3.0)
    ap.add_argument("--tpMultiplier-fixed", type=float, default=3.0)

    ap.add_argument("--basePeriod-fixed", type=int, default=20)
    ap.add_argument("--minPeriod-fixed", type=int, default=5)
    ap.add_argument("--maxPeriod-fixed", type=int, default=35)

    ap.add_argument("--opt-adaptive", action="store_true",
                    help="Optimize base/min/max period instead of using fixed values")
    ap.add_argument("--opt-fastslow", action="store_true",
                    help="Optimize fast/slow EMA periods instead of fixed values")

    ap.add_argument("--fastPeriod-fixed", type=int, default=4)
    ap.add_argument("--slowPeriod-fixed", type=int, default=50)
    ap.add_argument("--smooth_len-fixed", type=int, default=5)
    
    ap.add_argument("--shift-fixed", type=int, default=0)

    ap.add_argument("--threshold-fixed", type=float, default=0.5)
    ap.add_argument("--threshold-mode", type=str, default="dynamic", choices=["fixed", "dynamic"])
    ap.add_argument("--threshold-floor", type=float, default=0.1)
    ap.add_argument("--threshold-std-mult", type=float, default=0.5)

    ap.add_argument("--vol-floor-mult-fixed", type=float, default=1.0)
    ap.add_argument("--vol-floor-len", type=int, default=100)

    ap.add_argument("--trend-center", type=float, default=0.80)
    ap.add_argument("--trend-k", type=float, default=3.0)

    return ap.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.tp2sl_auto:
        if args.tp2sl_min <= 0 or args.tp2sl_max <= 0:
            raise SystemExit("--tp2sl-min and --tp2sl-max must be > 0")
        if args.tp2sl_min > args.tp2sl_max:
            raise SystemExit("--tp2sl-min must be <= --tp2sl-max")
    else:
        if args.min_tp2sl <= 0:
            raise SystemExit("--min-tp2sl must be > 0")

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

    def min_tp2sl_eff_for(atr_val: int) -> float:
        if args.tp2sl_auto:
            v = args.tp2sl_base + args.tp2sl_k * (float(atr_val) - float(args.tp2sl_sr0))
            return max(args.tp2sl_min, min(args.tp2sl_max, v))
        return float(args.min_tp2sl)

    def validate_adaptive_params(baseP, minP, maxP):
        if baseP < 5 or minP < 2 or maxP <= minP or maxP > 120 or (maxP - minP) < 5:
            return False
        return True

    def validate_fastslow(fp, sp):
        if fp < 2 or sp < 10 or sp < (fp * 2) or sp > 200:
            return False
        return True

    # ── REPORT-ONLY MODE ─────────────────────────────
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

        baseP = int(args.basePeriod_fixed)
        minP = int(args.minPeriod_fixed)
        maxP = int(args.maxPeriod_fixed)
        if not validate_adaptive_params(baseP, minP, maxP):
            raise SystemExit("Invalid adaptive RSI parameters (fixed)")

        fp = int(args.fastPeriod_fixed)
        sp = int(args.slowPeriod_fixed)
        if not validate_fastslow(fp, sp):
            raise SystemExit("Invalid fast/slow EMA parameters (fixed)")

        sm = int(args.smooth_len_fixed)
        if sm < 1:
            raise SystemExit("--smooth_len-fixed must be >= 1")
            
        shift = int(args.shift_fixed)

        threshold = float(args.threshold_fixed)
        vol_floor_mult = float(args.vol_floor_mult_fixed)

        fills = ROBUST_FILLS if args.report_both_fills else [args.fill]

        for fm in fills:
            _, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
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
                basePeriod=baseP,
                minPeriod=minP,
                maxPeriod=maxP,
                fastPeriod=fp,
                slowPeriod=sp,
                smooth_len=sm,
                threshold=threshold,
                threshold_mode=args.threshold_mode,
                threshold_floor=float(args.threshold_floor),
                threshold_std_mult=float(args.threshold_std_mult),
                vol_floor_mult=vol_floor_mult,
                vol_floor_len=int(args.vol_floor_len),
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
                trend_center=float(args.trend_center),
                trend_k=float(args.trend_k),
                shift=shift,
            )

            per_df = pd.DataFrame(per)
            per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
            per_df = per_df.sort_values(
                ["ticker_score", "total_return", "profit_factor_diag", "num_trades"],
                ascending=False
            )

            csv_path = out_dir / f"{MODULE_PREFIX}per_ticker_report-only_{fm}_{now_str}.csv"
            per_df.to_csv(csv_path, index=False)

            txt_path = out_dir / f"{MODULE_PREFIX}report_summary_{fm}_{now_str}.txt"
            with txt_path.open("w", encoding='utf-8') as f:  # explicit UTF-8
                f.write(f"REPORT ONLY - fill = {fm}\n")
                f.write(f"Run: {now_str}\n\n")
                f.write(f"Mean ticker score (eligible): {overall['mean_ticker_score']:.6f}\n")
                f.write(f"Avg PF (raw)                  : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}\n")
                f.write(f"Avg PF (diag)                 : {overall['avg_pf_diag']:.6f}\n")
                f.write(f"Avg trades/ticker             : {overall['avg_trades']:.2f}\n")
                f.write(f"Coverage                      : {coverage:.3f} ({eligible_count}/{len(per)})\n")
                f.write(f"Negative returns              : {overall['num_neg']}\n")

            print(f"Saved: {csv_path}")
            print(f"Saved: {txt_path}")

        mode = "auto" if args.tp2sl_auto else "fixed"
        print("\nFixed parameters used:")
        print(f"  atrPeriod     : {atrP}")
        print(f"  slMultiplier  : {slM:.4f}")
        print(f"  tpMultiplier  : {tpM:.4f}")
        print(f"  min_tp2sl_eff : {min_eff:.4f} ({mode})")
        print(f"  adaptive RSI  : base={baseP}, min={minP}, max={maxP}")
        print(f"  fast/slow EMA : fast={fp}, slow={sp}")
        print(f"  smooth_len    : {sm}")
        print(f"  shift         : {shift}")
        print(f"  threshold_mode: {args.threshold_mode}")
        print(f"  vol_floor_mult: {vol_floor_mult}")
        return

    # ── REPORT MODE (non-optimize) ─────────────────────────────────────────────
    if not args.optimize:
        atrP = int(args.atrPeriod_fixed)
        slM = float(args.slMultiplier_fixed)
        tpM = float(args.tpMultiplier_fixed)

        min_eff = min_tp2sl_eff_for(atrP)
        if slM <= min_eff * tpM:
            raise SystemExit("Constraint violated: slMultiplier <= min_tp2sl_eff * tpMultiplier")

        baseP = int(args.basePeriod_fixed)
        minP = int(args.minPeriod_fixed)
        maxP = int(args.maxPeriod_fixed)
        if not validate_adaptive_params(baseP, minP, maxP):
            raise SystemExit("Invalid adaptive RSI parameters (fixed)")

        fp = int(args.fastPeriod_fixed)
        sp = int(args.slowPeriod_fixed)
        if not validate_fastslow(fp, sp):
            raise SystemExit("Invalid fast/slow EMA parameters (fixed)")

        sm = int(args.smooth_len_fixed)
        if sm < 1:
            raise SystemExit("--smooth_len-fixed must be >= 1")
            
        shift = int(args.shift_fixed)

        threshold = float(args.threshold_fixed)
        vol_floor_mult = float(args.vol_floor_mult_fixed)

        fills = ROBUST_FILLS if args.report_both_fills else [args.fill]

        for fm in fills:
            _, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
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
                basePeriod=baseP,
                minPeriod=minP,
                maxPeriod=maxP,
                fastPeriod=fp,
                slowPeriod=sp,
                smooth_len=sm,
                threshold=threshold,
                threshold_mode=args.threshold_mode,
                threshold_floor=float(args.threshold_floor),
                threshold_std_mult=float(args.threshold_std_mult),
                vol_floor_mult=vol_floor_mult,
                vol_floor_len=int(args.vol_floor_len),
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
                trend_center=float(args.trend_center),
                trend_k=float(args.trend_k),
                shift=shift,
            )

            per_df = pd.DataFrame(per)
            per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
            per_df = per_df.sort_values(
                ["ticker_score", "total_return", "profit_factor_diag", "num_trades"],
                ascending=False
            )

            csv_path = out_dir / f"{MODULE_PREFIX}per_ticker_report-only_{fm}_{now_str}.csv"
            per_df.to_csv(csv_path, index=False)

            txt_path = out_dir / f"{MODULE_PREFIX}report_summary_{fm}_{now_str}.txt"
            with txt_path.open("w", encoding='utf-8') as f:  # explicit UTF-8
                f.write(f"REPORT ONLY - fill = {fm}\n")
                f.write(f"Run: {now_str}\n\n")
                f.write(f"Mean ticker score (eligible): {overall['mean_ticker_score']:.6f}\n")
                f.write(f"Avg PF (raw)                  : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}\n")
                f.write(f"Avg PF (diag)                 : {overall['avg_pf_diag']:.6f}\n")
                f.write(f"Avg trades/ticker             : {overall['avg_trades']:.2f}\n")
                f.write(f"Coverage                      : {coverage:.3f} ({eligible_count}/{len(per)})\n")
                f.write(f"Negative returns              : {overall['num_neg']}\n")

            print(f"Saved: {csv_path}")
            print(f"Saved: {txt_path}")

        mode = "auto" if args.tp2sl_auto else "fixed"
        print("\nFixed parameters used:")
        print(f"  atrPeriod     : {atrP}")
        print(f"  slMultiplier  : {slM:.4f}")
        print(f"  tpMultiplier  : {tpM:.4f}")
        print(f"  min_tp2sl_eff : {min_eff:.4f} ({mode})")
        print(f"  adaptive RSI  : base={baseP}, min={minP}, max={maxP}")
        print(f"  fast/slow EMA : fast={fp}, slow={sp}")
        print(f"  smooth_len    : {sm}")
        print(f"  shift         : {shift}")
        print(f"  threshold_mode: {args.threshold_mode}")
        print(f"  vol_floor_mult: {vol_floor_mult}")
        return

    # ── OPTIMIZATION (UPDATED TO USE RANGE PARAMETERS) ────────────────────────────────────────────
    def objective(trial: optuna.Trial) -> float:
        # Use range parameters if provided, otherwise use defaults
        if args.atrPeriod_min is not None and args.atrPeriod_max is not None:
            atrPeriod = trial.suggest_int("atrPeriod", args.atrPeriod_min, args.atrPeriod_max)
        else:
            atrPeriod = trial.suggest_int("atrPeriod", 10, 25)
            
        if args.slMultiplier_min is not None and args.slMultiplier_max is not None:
            slMultiplier = trial.suggest_float("slMultiplier", args.slMultiplier_min, args.slMultiplier_max)
        else:
            slMultiplier = trial.suggest_float("slMultiplier", 1.2, 2.5)
            
        if args.tpMultiplier_min is not None and args.tpMultiplier_max is not None:
            tpMultiplier = trial.suggest_float("tpMultiplier", args.tpMultiplier_min, args.tpMultiplier_max)
        else:
            tpMultiplier = trial.suggest_float("tpMultiplier", 2.0, 5.0)

        # Adaptive RSI parameters
        if args.opt_adaptive:
            if args.basePeriod_min is not None and args.basePeriod_max is not None:
                basePeriod = trial.suggest_int("basePeriod", args.basePeriod_min, args.basePeriod_max)
            else:
                basePeriod = trial.suggest_int("basePeriod", 10, 25)
                
            if args.minPeriod_min is not None and args.minPeriod_max is not None:
                minPeriod = trial.suggest_int("minPeriod", args.minPeriod_min, args.minPeriod_max)
            else:
                minPeriod = trial.suggest_int("minPeriod", 2, 5)
                
            if args.maxPeriod_min is not None and args.maxPeriod_max is not None:
                maxPeriod = trial.suggest_int("maxPeriod", args.maxPeriod_min, args.maxPeriod_max)
            else:
                maxPeriod = trial.suggest_int("maxPeriod", 12, 25)
        else:
            basePeriod = int(args.basePeriod_fixed)
            minPeriod = int(args.minPeriod_fixed)
            maxPeriod = int(args.maxPeriod_fixed)

        # Fast/slow EMA parameters
        if args.opt_fastslow:
            if args.fastPeriod_min is not None and args.fastPeriod_max is not None:
                fastPeriod = trial.suggest_int("fastPeriod", args.fastPeriod_min, args.fastPeriod_max)
            else:
                fastPeriod = trial.suggest_int("fastPeriod", 2, 8)
                
            if args.slowPeriod_min is not None and args.slowPeriod_max is not None:
                slowPeriod = trial.suggest_int("slowPeriod", args.slowPeriod_min, args.slowPeriod_max)
            else:
                slowPeriod = trial.suggest_int("slowPeriod", 15, 50)
        else:
            fastPeriod = int(args.fastPeriod_fixed)
            slowPeriod = int(args.slowPeriod_fixed)

        # Smooth length
        if args.smooth_len_min is not None and args.smooth_len_max is not None:
            smooth_len = trial.suggest_int("smooth_len", args.smooth_len_min, args.smooth_len_max)
        else:
            smooth_len = trial.suggest_int("smooth_len", 2, 6)
            
        # Shift
        if args.shift_min is not None and args.shift_max is not None:
            shift = trial.suggest_int("shift", args.shift_min, args.shift_max)
        else:
            shift = trial.suggest_int("shift", 0, 3)
            
        vol_floor_mult = float(args.vol_floor_mult_fixed)

        # Cooldown and time stop
        if args.opt_cooldown:
            if args.cooldown_min is not None and args.cooldown_max is not None:
                cooldown_bars = trial.suggest_int("cooldown", args.cooldown_min, args.cooldown_max)
            else:
                cooldown_bars = trial.suggest_int("cooldown", 0, 7)
        else:
            cooldown_bars = int(args.cooldown)
            
        if args.opt_time_stop:
            if args.time_stop_min is not None and args.time_stop_max is not None:
                time_stop_bars = trial.suggest_int("time_stop", args.time_stop_min, args.time_stop_max)
            else:
                time_stop_bars = trial.suggest_int("time_stop", 5, 15)
        else:
            time_stop_bars = int(args.time_stop)

        # Constraints
        if tpMultiplier < 1.01 * slMultiplier:
            raise optuna.TrialPruned()
        if fastPeriod >= slowPeriod:
            raise optuna.TrialPruned()
        if maxPeriod <= minPeriod:
            raise optuna.TrialPruned()

        threshold_mode = str(args.threshold_mode)
        if threshold_mode == "fixed":
            threshold = float(args.threshold_fixed)
            threshold_floor = 0.0
            threshold_std_mult = 0.0
        else:
            threshold = 0.0
            if args.threshold_floor_min is not None and args.threshold_floor_max is not None:
                threshold_floor = trial.suggest_float("threshold_floor", args.threshold_floor_min, args.threshold_floor_max)
            else:
                threshold_floor = trial.suggest_float("threshold_floor", 0.005, 0.08)
                
            if args.threshold_std_mult_min is not None and args.threshold_std_mult_max is not None:
                threshold_std_mult = trial.suggest_float("threshold_std_mult", args.threshold_std_mult_min, args.threshold_std_mult_max)
            else:
                threshold_std_mult = trial.suggest_float("threshold_std_mult", 0.05, 0.40)

        mean_score, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
            file_paths,
            atrPeriod=int(atrPeriod),
            slMultiplier=float(slMultiplier),
            tpMultiplier=float(tpMultiplier),
            commission_rate_per_side=float(args.commission_rate_per_side),
            fill_mode=str(args.fill),
            use_trailing_exit=bool(args.use_trailing_exit),
            trail_mode=str(args.trail_mode),
            close_on_sellSignal=bool(args.close_on_sellSignal),
            cooldown_bars=int(cooldown_bars),
            time_stop_bars=int(time_stop_bars),
            basePeriod=int(basePeriod),
            minPeriod=int(minPeriod),
            maxPeriod=int(maxPeriod),
            fastPeriod=int(fastPeriod),
            slowPeriod=int(slowPeriod),
            smooth_len=int(smooth_len),
            threshold=float(threshold),
            threshold_mode=threshold_mode,
            threshold_floor=float(threshold_floor),
            threshold_std_mult=float(threshold_std_mult),
            vol_floor_mult=float(vol_floor_mult),
            vol_floor_len=int(args.vol_floor_len),
            min_trades=int(args.min_trades),
            pf_baseline=float(args.pf_baseline),
            pf_k=float(args.pf_k),
            trades_baseline=float(args.trades_baseline),
            trades_k=float(args.trades_k),
            weight_pf=float(args.weight_pf),
            score_power=float(args.score_power),
            pf_cap_score_only=float(args.pf_cap_score_only),
            penalty_enabled=bool(args.penalty),
            loss_floor=float(args.loss_floor),
            penalty_ret_center=float(args.penalty_ret_center),
            penalty_ret_k=float(args.penalty_ret_k),
            max_trades=int(args.max_trades),
            max_trades_k=float(args.max_trades_k),
            pf_floor=float(args.pf_floor),
            pf_floor_k=float(args.pf_floor_k),
            ret_floor=float(args.ret_floor),
            ret_floor_k=float(args.ret_floor_k),
            trend_center=float(args.trend_center),
            trend_k=float(args.trend_k),
            shift=int(shift),
        )

        if not per or coverage <= 0.0:
            return -1.0

        returns = np.array([x["total_return"] for x in per], dtype=float)
        avg_dd = np.mean([abs(x.get("maxdd", 0.0)) for x in per])
        portfolio_ret = np.mean(returns)

        std_dev = float(np.std(returns)) if returns.size > 1 else 1.0
        stability_penalty = 1.0 / (1.0 + std_dev)

        trendiness = portfolio_ret / (avg_dd + 0.001)
        regime_weight = sigmoid(float(args.trend_k) * (trendiness - float(args.trend_center)))

        score_trend = float(mean_score) * (1.0 + portfolio_ret)
        score_chop = float(mean_score) * stability_penalty

        blended_score = (regime_weight * score_trend) + ((1.0 - regime_weight) * score_chop)

        target_trades = 10.0
        trade_density_mult = (min(1.0, float(overall["avg_trades"]) / target_trades)) ** 2

        max_ticker_dd = max((abs(x.get("maxdd", 0.0)) for x in per), default=0.0)
        dd_gate = sigmoid(6.0 * (0.2 - max_ticker_dd))

        cov_p = sigmoid(float(args.coverage_k) * (float(coverage) - float(args.coverage_target)))

        final_score = blended_score * trade_density_mult * cov_p * dd_gate
        return float(final_score)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    if not args.opt_adaptive:
        best_params["basePeriod"] = int(args.basePeriod_fixed)
        best_params["minPeriod"] = int(args.minPeriod_fixed)
        best_params["maxPeriod"] = int(args.maxPeriod_fixed)
    if not args.opt_fastslow:
        best_params["fastPeriod"] = int(args.fastPeriod_fixed)
        best_params["slowPeriod"] = int(args.slowPeriod_fixed)
    if "smooth_len" not in best_params:
        best_params["smooth_len"] = int(args.smooth_len_fixed)
    if "shift" not in best_params:
        best_params["shift"] = int(args.shift_fixed)
    if "cooldown" not in best_params:
        best_params["cooldown"] = int(args.cooldown)
    if "time_stop" not in best_params:
        best_params["time_stop"] = int(args.time_stop)

    best_cooldown = int(best_params.get("cooldown", args.cooldown))
    best_time_stop = int(best_params.get("time_stop", args.time_stop))

    best_score_single, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
        file_paths,
        atrPeriod=int(best_params["atrPeriod"]),
        slMultiplier=float(best_params["slMultiplier"]),
        tpMultiplier=float(best_params["tpMultiplier"]),
        commission_rate_per_side=args.commission_rate_per_side,
        fill_mode=args.fill,
        use_trailing_exit=args.use_trailing_exit,
        trail_mode=args.trail_mode,
        close_on_sellSignal=args.close_on_sellSignal,
        cooldown_bars=best_cooldown,
        time_stop_bars=best_time_stop,
        basePeriod=int(best_params["basePeriod"]),
        minPeriod=int(best_params["minPeriod"]),
        maxPeriod=int(best_params["maxPeriod"]),
        fastPeriod=int(best_params["fastPeriod"]),
        slowPeriod=int(best_params["slowPeriod"]),
        smooth_len=int(best_params["smooth_len"]),
        threshold=args.threshold_fixed,
        threshold_mode=args.threshold_mode,
        threshold_floor=float(args.threshold_floor),
        threshold_std_mult=float(args.threshold_std_mult),
        vol_floor_mult=float(args.vol_floor_mult_fixed),
        vol_floor_len=int(args.vol_floor_len),
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
        trend_center=float(args.trend_center),
        trend_k=float(args.trend_k),
        shift=int(best_params.get("shift", args.shift_fixed)),
    )

    per_df = pd.DataFrame(per)
    per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
    per_df = per_df.sort_values(
        ["ticker_score", "total_return", "profit_factor_diag", "num_trades"],
        ascending=False
    )

    # ── SAVE RESULTS ────────────────────────────────────────────
    csv_path = out_dir / f"{MODULE_PREFIX}per_ticker_{now_str}.csv"
    per_df.to_csv(csv_path, index=False)

    summary_lines = [
        "=== BEST RESULT - Adaptive RSI ===",
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
        f"  Mean ticker score (eligible) : {overall['mean_ticker_score']:.6f}",
        f"  Avg PF (raw)                 : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}",
        f"  Avg PF (diag)                : {overall['avg_pf_diag']:.6f}",
        f"  Avg trades/ticker            : {overall['avg_trades']:.2f}",
        f"  Coverage                     : {coverage:.3f} ({eligible_count}/{len(per_df)})",
        f"  Negative returns             : {overall['num_neg']}",
        "",
        f"  Commission/side              : {args.commission_rate_per_side:.6f}",
        f"  Penalty enabled              : {args.penalty}",
        f"  min_tp2sl_eff                : {min_tp2sl_eff_for(int(best_params['atrPeriod'])):.4f}",
        f"  use_trailing_exit            : {args.use_trailing_exit}",
        f"  trail_mode                   : {args.trail_mode}",
        f"  close_on_sellSignal          : {args.close_on_sellSignal}",
    ])

    txt_path = out_dir / f"{MODULE_PREFIX}best_{now_str}.txt"
    txt_path.write_text("\n".join(summary_lines), encoding='utf-8')

    # ── Clean console summary (ASCII only) ───────────────────────────────
    print("\n" + "="*60)
    print("               BEST RESULT - Adaptive RSI")
    print("="*60)
    print(f"Objective value              : {best.value:.6f}")
    print(f"Mean ticker score (eligible) : {overall['mean_ticker_score']:.6f}")
    print(f"Avg PF (raw)                 : {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else 'inf'}")
    print(f"Avg trades/ticker            : {overall['avg_trades']:.2f}")
    print(f"Coverage                     : {coverage:.3f}")
    print(f"\nSaved:")
    print(f"  CSV -> {csv_path}")
    print(f"  TXT -> {txt_path}")
    print("\n".join(summary_lines))

if __name__ == "__main__":
    main()