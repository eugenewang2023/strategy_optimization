#!/usr/bin/env python3
"""
Bayes_opt_adapt_RSI.py

Goal:
✅ Keep the SAME scoring/coverage/robust-fills framework as your Bayes_opt_adapt_half_RSI.py
✅ Only signals differ (adaptive RSI / hot_sm), plus optional hysteresis + volatility floor

Key fixes in this version (your "Report" & "Final Report" issues):
- REPORT mode and FINAL REPORT now run through the SAME evaluation pipeline.
- evaluate_params_on_files() returns the same tuple everywhere:
    (mean_score, overall, per, pf_raw_avg, trades_avg, coverage, eligible_count)
- Backtest uses PARQUET + ensure_ohlc (no accidental pd.read_csv)
- Per-ticker dict always includes: profit_factor, profit_factor_diag, num_trades, ticker_score, etc.
- profit_factor_diag is computed with the SAME loss_floor logic as scoring AND capped to pf_cap_score_only.
- Report mode correctly passes threshold/vol_floor_mult and prints avg_pf_diag without KeyErrors.
- Final report uses the same per-ticker sorting keys and saves CSV consistently.

Signals (only difference from half_RSI):
- Compute base RSI on HA close (sigClose), map to adaptive RSI length per bar (sigmoid mapping)
- adaptive_rsi -> EMA(fast/slow) -> hot -> SMA -> hysteresis cross using +/- threshold
- Optional: dynamic threshold based on rolling std of hot_sm (if threshold_mode="dynamic")
- Volatility filter: ATR must exceed SMA(ATR)*vol_floor_mult
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
    """
    TradingView-ish RSI: Wilder RMA of gains/losses.
    Supports length >= 1 (including RSI(1)).
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=float)
    L = int(length)
    if L <= 0 or n < 2:
        return out

    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    def rma(x: np.ndarray, period: int) -> np.ndarray:
        y = np.full(n, np.nan, dtype=float)
        if n < period:
            return y

        # seed
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
    """
    TradingView-ish EMA:
    - seed with SMA(length) at first full non-nan window
    - then EMA with alpha=2/(length+1)
    """
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
    profit_factor_diag: float = 0.0  # stable + capped like scoring
    trend_metric: float = 0.0


def backtest_adapt_rsi_dynamic_engine(
    df: pd.DataFrame,
    *,
    atrPeriod: int,
    slMult: float,
    tpMult: float,
    commission_per_side: float,
    fill_mode: str,
    cooldown_bars: int,
    time_stop_bars: int,
    # adaptive RSI knobs
    basePeriod: int,
    minPeriod: int,
    maxPeriod: int,
    fastPeriod: int,
    slowPeriod: int,
    smooth_len: int,
    # hysteresis / vol filter knobs
    threshold: float,
    threshold_mode: str,          # "fixed" or "dynamic"
    threshold_floor: float,       # minimum threshold used in dynamic mode
    threshold_std_mult: float,    # multiplier * rolling std in dynamic mode
    vol_floor_mult: float,
    vol_floor_len: int,
    # scoring alignment
    loss_floor: float,
    pf_cap_score_only: float,
) -> TradeStats:
    df = ensure_ohlc(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    if n < max(int(atrPeriod) + 2, 200):
        return TradeStats()

    # HA signals from REAL OHLC (chart-type stable)
    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)
    sigClose = ha_c

    # REAL ATR for risk & vol-floor
    atr = atr_wilder(h, l, c, int(atrPeriod))
    valid_idx = np.where(~np.isnan(atr))[0]
    if len(valid_idx) == 0:
        return TradeStats()
    start = int(valid_idx[0])

    # -----------------------------
    # Trend metric (soft weighting)
    # -----------------------------
    # Normalize EMA separation by ATR so it's scale-invariant across tickers.
    ema_fast_tr = ema_tv(sigClose, 50)
    ema_slow_tr = ema_tv(sigClose, 200)

    sep = np.abs(ema_fast_tr - ema_slow_tr) / (atr + 1e-12)
    # Use mean over valid region
    if np.isfinite(sep).any():
        trend_metric = float(np.nanmean(sep))
    else:
        trend_metric = 0.0

    def get_fill(i: int) -> Optional[float]:
        if fill_mode == "same_close":
            return float(c[i])
        if fill_mode == "next_open":
            return float(o[i + 1]) if (i + 1 < n) else None
        return float(c[i])

    # -----------------------------
    # Signal: sigmoid-adaptive RSI lengths
    # -----------------------------
    baseP = max(2, int(basePeriod))
    minP = max(2, int(minPeriod))
    maxP = max(minP, int(maxPeriod))
    fp = max(1, int(fastPeriod))
    sp = max(fp + 1, int(slowPeriod))
    sm = max(1, int(smooth_len))

    base_rsi = rsi_tv(sigClose, baseP)  # 0..100

    # sigmoid mapping around 50
    # k_sigmoid in (0,1), pushes period toward maxP when RSI high (or vice versa depending on sign)
    k_sigmoid = 1.0 / (1.0 + np.exp(-0.1 * (base_rsi - 50.0)))
    adapt_p = (k_sigmoid * float(maxP - minP) + float(minP))
    adapt_p = np.clip(adapt_p, float(minP), float(maxP))
    adapt_p_int = np.where(np.isfinite(adapt_p), np.rint(adapt_p), float(minP)).astype(int)

    # precompute RSI for lengths in [minP..maxP]
    rsi_map: Dict[int, np.ndarray] = {}
    for L in range(minP, maxP + 1):
        rsi_map[L] = rsi_tv(sigClose, L)

    adaptive_rsi = np.full(n, np.nan, dtype=float)
    for i in range(n):
        p = int(adapt_p_int[i])
        if p in rsi_map:
            adaptive_rsi[i] = rsi_map[p][i]

    fast_rsi = ema_tv(adaptive_rsi, fp)
    slow_rsi = ema_tv(adaptive_rsi, sp)
    hot_sm = sma(fast_rsi - slow_rsi, sm)

    # Dynamic threshold
    if threshold_mode == "dynamic":
        # stddev of hot_sm over vol_floor_len (reuse len for simplicity)
        # using pandas rolling to handle NaNs cleanly
        s_std = pd.Series(hot_sm).rolling(int(vol_floor_len), min_periods=int(vol_floor_len)).std().to_numpy()
        dyn_thresh = np.maximum(float(threshold_floor), np.nan_to_num(s_std) * float(threshold_std_mult))
    else:
        dyn_thresh = np.full(n, float(threshold), dtype=float)

    # Volatility floor: ATR > SMA(ATR, vol_floor_len) * vol_floor_mult
    atr_ma = sma(atr, int(vol_floor_len))
    vol_ok = atr > (atr_ma * float(vol_floor_mult))

    # -----------------------------
    # Trade engine (long-only)
    # -----------------------------
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

    def apply_commission(eq: float) -> float:
        return eq * (1.0 - commission_per_side)

    # require enough history for vol floor + dyn thresh
    loop_start = max(start + 2, int(vol_floor_len) + 2, 2)

    for i in range(loop_start, n - 1):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if np.isnan(hot_sm[i - 1]) or np.isnan(hot_sm[i]) or np.isnan(dyn_thresh[i]) or np.isnan(atr[i]):
            continue

        th = float(dyn_thresh[i])

        buy_sig = (hot_sm[i - 1] <= th) and (hot_sm[i] > th)
        sell_sig = (hot_sm[i - 1] >= -th) and (hot_sm[i] < -th)

        # Entry
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
            continue

        # Manage open position
        if in_pos:
            bars_in_trade += 1

            # Breakeven logic in ATR units
            pnl_atr = (c[i] - entry) / (atr[i] + 1e-12)
            if pnl_atr > 1.5:
                has_hit_be = True

            # SL/TP levels anchored to entry (fixed bug)
            stop_level = entry if has_hit_be else (entry - atr[i] * slMult)
            tgt_level = entry + atr[i] * tpMult

            exit_now = False
            exit_price: Optional[float] = None

            # Intrabar conservative: stop uses low, tp uses high
            if l[i] <= stop_level:
                exit_now = True
                exit_price = float(stop_level)
            elif h[i] >= tgt_level:
                exit_now = True
                exit_price = float(tgt_level)
            elif (ts_bars > 0) and (bars_in_trade >= ts_bars):
                # loss-only time stop
                fill = get_fill(i)
                if fill is not None:
                    unreal = (float(fill) - entry) / entry
                    if unreal <= 0.0:
                        exit_now = True
                        exit_price = float(fill)
            elif sell_sig:
                fill = get_fill(i)
                if fill is not None:
                    exit_now = True
                    exit_price = float(fill)
            elif i == n - 2:
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

        if equity > peak:
            peak = equity
        dd = (equity / peak) - 1.0
        if dd < maxdd:
            maxdd = dd

    # Raw PF
    pf_raw = (gp / gl) if gl > 0 else (float("inf") if gp > 0 else 0.0)

    # Diagnostic PF: same effective_gl logic as scoring + cap
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
    trend_center: float,
    trend_k: float,
) -> float:
    if st.num_trades < min_trades:
        return float("nan")

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
    # Soft trend weighting (no regime split, no discontinuities)
    # Multiplier in [0.5, 1.0]
    trend_mult = 0.5 + 0.5 * sigmoid(trend_k * (st.trend_metric - trend_center))
    s *= trend_mult
    return float(s)


def evaluate_params_on_files(
    file_paths: List[Path],
    *,
    # engine params
    atrPeriod: int,
    slMultiplier: float,
    tpMultiplier: float,
    commission_rate_per_side: float,
    fill_mode: str,
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
    # scoring params
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
    trend_center: float,
    trend_k: float,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], float, float, float, int]:
    per: List[Dict[str, Any]] = []
    eligible_scores: List[float] = []
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
            atrPeriod=atrPeriod,
            slMult=slMultiplier,
            tpMult=tpMultiplier,
            commission_per_side=commission_rate_per_side,
            fill_mode=fill_mode,
            cooldown_bars=cooldown_bars,
            time_stop_bars=time_stop_bars,
            basePeriod=basePeriod,
            minPeriod=minPeriod,
            maxPeriod=maxPeriod,
            fastPeriod=fastPeriod,
            slowPeriod=slowPeriod,
            smooth_len=smooth_len,
            threshold=threshold,
            threshold_mode=threshold_mode,
            threshold_floor=threshold_floor,
            threshold_std_mult=threshold_std_mult,
            vol_floor_mult=vol_floor_mult,
            vol_floor_len=vol_floor_len,
            loss_floor=loss_floor,
            pf_cap_score_only=pf_cap_score_only,
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
            max_trades=max_trades,
            max_trades_k=max_trades_k,
            pf_floor=pf_floor,
            pf_floor_k=pf_floor_k,
            ret_floor=ret_floor,
            ret_floor_k=ret_floor_k,
            trend_center=trend_center,
            trend_k=trend_k,
        )

        is_eligible = (st.num_trades >= min_trades)
        if is_eligible:
            eligible_count += 1
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

        if np.isfinite(sc):
            eligible_scores.append(float(sc))

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
# CLI + Main
# =============================
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--output_dir", type=str, default="output")

    ap.add_argument("--files", type=int, default=200)
    ap.add_argument("--trials", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)

    # scoring knobs
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

    # cooldown/time-stop
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--opt-cooldown", action="store_true")

    ap.add_argument("--time-stop", type=int, default=0)
    ap.add_argument("--opt-time-stop", action="store_true")

    # TP/SL asymmetry constraint
    ap.add_argument("--min-tp2sl", type=float, default=1.30)
    ap.add_argument("--tp2sl-auto", action="store_true")
    ap.add_argument("--tp2sl-base", type=float, default=1.25)
    ap.add_argument("--tp2sl-sr0", type=float, default=30.0)
    ap.add_argument("--tp2sl-k", type=float, default=0.01)
    ap.add_argument("--tp2sl-min", type=float, default=1.10)
    ap.add_argument("--tp2sl-max", type=float, default=1.80)

    # Coverage penalty knobs
    ap.add_argument("--coverage-target", type=float, default=0.70)
    ap.add_argument("--coverage-k", type=float, default=12.0)

    # Modes
    ap.add_argument("--optimize", action="store_true", help="Run Optuna optimization (otherwise report mode).")
    ap.add_argument("--report-both-fills", action="store_true")

    # Fixed params (defaults)
    ap.add_argument("--atrPeriod-fixed", type=int, default=25)
    ap.add_argument("--slMultiplier-fixed", type=float, default=3.0)
    ap.add_argument("--tpMultiplier-fixed", type=float, default=3.0)

    ap.add_argument("--basePeriod-fixed", type=int, default=20)
    ap.add_argument("--minPeriod-fixed", type=int, default=5)
    ap.add_argument("--maxPeriod-fixed", type=int, default=35)
    ap.add_argument("--opt-adaptive", action="store_true")

    ap.add_argument("--fastPeriod-fixed", type=int, default=4)
    ap.add_argument("--slowPeriod-fixed", type=int, default=50)
    ap.add_argument("--opt-fastslow", action="store_true")

    ap.add_argument("--smooth_len-fixed", type=int, default=5)

    # hysteresis / vol-floor (fixed defaults, optional optimization inside objective)
    ap.add_argument("--threshold-fixed", type=float, default=0.5)
    ap.add_argument("--threshold-mode", type=str, default="dynamic", choices=["fixed", "dynamic"])
    ap.add_argument("--threshold-floor", type=float, default=0.1)
    ap.add_argument("--threshold-std-mult", type=float, default=0.5)

    ap.add_argument("--vol-floor-mult-fixed", type=float, default=1.0)
    ap.add_argument("--vol-floor-len", type=int, default=100)

    ap.add_argument("--trend-center", type=float, default=0.80)
    ap.add_argument("--trend-k", type=float, default=3.0)

    return ap.parse_args()


def main() -> None:
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

    file_paths = random.sample(all_files, args.files) if (len(all_files) > args.files) else all_files
    ROBUST_FILLS = ["same_close", "next_open"]

    def min_tp2sl_eff_for(atrPeriod_val: int) -> float:
        if args.tp2sl_auto:
            v = args.tp2sl_base + args.tp2sl_k * (float(atrPeriod_val) - float(args.tp2sl_sr0))
            return max(args.tp2sl_min, min(args.tp2sl_max, v))
        return float(args.min_tp2sl)

    def validate_adaptive_params(baseP: int, minP: int, maxP: int) -> bool:
        # relaxed but still sane
        if baseP < 5:
            return False
        if minP < 2:
            return False
        if maxP <= minP:
            return False
        if maxP > 120:
            return False
        if (maxP - minP) < 5:
            return False
        return True

    def validate_fastslow(fp: int, sp: int) -> bool:
        if fp < 2:
            return False
        if sp < 10:
            return False
        if sp < (fp * 2):
            return False
        if sp > 200:
            return False
        return True

    # =========================
    # REPORT MODE
    # =========================
    if not args.optimize:
        atrP = int(args.atrPeriod_fixed)
        slM = float(args.slMultiplier_fixed)
        tpM = float(args.tpMultiplier_fixed)

        min_eff = min_tp2sl_eff_for(atrP)
        if slM <= min_eff * tpM:
            raise SystemExit(
                f"Constraint violated: slMultiplier ({slM}) <= min_tp2sl_eff ({min_eff:.4f}) * tpMultiplier ({tpM})"
            )

        baseP = int(args.basePeriod_fixed)
        minP = int(args.minPeriod_fixed)
        maxP = int(args.maxPeriod_fixed)
        if not validate_adaptive_params(baseP, minP, maxP):
            raise SystemExit("Invalid adaptive params (fixed defaults/CLI).")

        fp = int(args.fastPeriod_fixed)
        sp = int(args.slowPeriod_fixed)
        if not validate_fastslow(fp, sp):
            raise SystemExit("Invalid fast/slow EMA params (fixed defaults/CLI).")

        sm = int(args.smooth_len_fixed)
        if sm < 1:
            raise SystemExit("--smooth_len-fixed must be >= 1")

        threshold = float(args.threshold_fixed)
        vol_floor_mult = float(args.vol_floor_mult_fixed)

        fills = ROBUST_FILLS if args.report_both_fills else [args.fill]

        for fm in fills:
            best_score_single, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
                file_paths,
                atrPeriod=atrP,
                slMultiplier=slM,
                tpMultiplier=tpM,
                commission_rate_per_side=args.commission_rate_per_side,
                fill_mode=fm,
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
            )

            per_df = pd.DataFrame(per)
            per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
            per_df = per_df.sort_values(
                ["ticker_score", "total_return", "profit_factor_diag", "num_trades", "trend_metric"],
                ascending=False
            )

            per_csv = out_dir / f"per_ticker_summary_{fm}.csv"
            per_df.to_csv(per_csv, index=False)

            print(f"\n=== REPORT MODE ({fm}) ===")
            print(f"Saved: {per_csv}")
            print(f"Avg PF (raw):                {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else float('inf')}")
            print(f"Avg PF (diag):               {overall['avg_pf_diag']:.6f}")
            print(f"Avg Trades / Ticker:         {overall['avg_trades']:.3f}")
            print(f"Mean Ticker Score (eligible):{overall['mean_ticker_score']:.6f}")
            print(f"Coverage (eligible/total):   {eligible_count}/{len(per)} = {coverage:.3f}")
            print(f"num_neg (return<=0):         {overall['num_neg']}")
            print(f"Score (single fill):         {best_score_single:.6f}")

        mode = "auto" if args.tp2sl_auto else "fixed"
        print("\n=== FIXED PARAMS USED ===")
        print(f"atrPeriod: {atrP}")
        print(f"slMultiplier: {slM}")
        print(f"tpMultiplier: {tpM}")
        print(f"min_tp2sl_eff (constraint): {min_eff:.4f} ({mode})")
        print(f"adaptive: basePeriod={baseP}, minPeriod={minP}, maxPeriod={maxP}")
        print(f"fast/slow EMA: fastPeriod={fp}, slowPeriod={sp}")
        print(f"smoothing: smooth_len={sm}")
        print(f"threshold_mode: {args.threshold_mode}, threshold_fixed={threshold}")
        print(f"vol_floor_mult: {vol_floor_mult}, vol_floor_len={args.vol_floor_len}")
        return

    # =========================
    # OPTUNA
    # =========================
    def objective(trial: optuna.Trial) -> float:
        # =========================================================================
        # 1) Suggest / set parameters (High-Density Aggressive)
        # =========================================================================
        atrPeriod    = trial.suggest_int("atrPeriod", 10, 25)
        slMultiplier = trial.suggest_float("slMultiplier", 1.2, 2.5) 
        tpMultiplier = trial.suggest_float("tpMultiplier", 2.0, 5.0)

        if getattr(args, "opt_adaptive", False):
            basePeriod = trial.suggest_int("basePeriod", 10, 25) 
            minPeriod  = trial.suggest_int("minPeriod", 2, 5)   
            maxPeriod  = trial.suggest_int("maxPeriod", 12, 25) 
        else:
            basePeriod = int(args.basePeriod_fixed)
            minPeriod  = int(args.minPeriod_fixed)
            maxPeriod  = int(args.maxPeriod_fixed)

        if getattr(args, "opt_fastslow", False):
            fastPeriod = trial.suggest_int("fastPeriod", 2, 8)  
            slowPeriod = trial.suggest_int("slowPeriod", 15, 50) 
        else:
            fastPeriod = int(args.fastPeriod_fixed)
            slowPeriod = int(args.slowPeriod_fixed)

        smooth_len = trial.suggest_int("smooth_len", 2, 6) 
        vol_floor_mult = float(args.vol_floor_mult_fixed)

        # FORCED: 1-bar cooldown to maximize frequency
        cooldown_bars = 1 
        time_stop_bars = trial.suggest_int("time_stop", 5, 15) if getattr(args, "opt_time_stop", False) else int(args.time_stop)

        # =========================================================================
        # 2) Logical constraints
        # =========================================================================
        if tpMultiplier < 1.01 * slMultiplier: return -1.0
        if fastPeriod >= slowPeriod: return -1.0
        if maxPeriod <= minPeriod: return -1.0

        # =========================================================================
        # 3) Threshold knobs
        # =========================================================================
        threshold_mode = str(args.threshold_mode)
        if threshold_mode == "fixed":
            threshold = float(args.threshold_fixed)
            threshold_floor, threshold_std_mult = 0.0, 0.0
        else:
            threshold = 0.0 
            threshold_floor = trial.suggest_float("threshold_floor", 0.005, 0.08)
            threshold_std_mult = trial.suggest_float("threshold_std_mult", 0.05, 0.40)

        # =========================================================================
        # 4) Evaluate
        # =========================================================================
        mean_score, overall, per, pf_raw_avg, trades_avg, coverage, eligible_count = evaluate_params_on_files(
            file_paths,
            atrPeriod=int(atrPeriod),
            slMultiplier=float(slMultiplier),
            tpMultiplier=float(tpMultiplier),
            commission_rate_per_side=float(args.commission_rate_per_side),
            fill_mode=str(args.fill),
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
        )

        if not per or coverage <= 0.0:
            return -1.0

        # =========================================================================
        # 5) Two-Regime Scoring Logic
        # =========================================================================
        returns = np.array([x["total_return"] for x in per], dtype=float)
        avg_dd = np.mean([abs(x.get("maxdd", 0.0)) for x in per])
        portfolio_ret = np.mean(returns)
        
        # Stability: Standard Deviation Penalty
        std_dev = float(np.std(returns)) if returns.size > 1 else 1.0
        stability_penalty = 1.0 / (1.0 + std_dev)

        # Regime Determination (Return-to-Drawdown Ratio)
        # High Ratio = Trending | Low Ratio = Choppy
        trendiness = portfolio_ret / (avg_dd + 0.001)
        regime_weight = sigmoid(float(args.trend_k) * (trendiness - float(args.trend_center)))

        # Regime A: Trend-Focused Score (Rewards high Profit and absolute returns)
        score_trend = float(mean_score) * (1.0 + portfolio_ret)
        
        # Regime B: Chop-Focused Score (Rewards low variance and drawdown protection)
        score_chop = float(mean_score) * stability_penalty

        # Blended Base Score
        blended_score = (regime_weight * score_trend) + ((1.0 - regime_weight) * score_chop)

        # =========================================================================
        # 6) Multipliers (Trade Density, Coverage, Drawdown)
        # =========================================================================
        # Target 10 trades; aggressive power penalty for under-trading
        target_trades = 10.0
        trade_density_mult = (min(1.0, float(trades_avg) / target_trades)) ** 2

        # Drawdown gate (Punishes any configuration exceeding 20% DD on any ticker)
        max_ticker_dd = max((abs(x.get("maxdd", 0.0)) for x in per), default=0.0)
        dd_gate = sigmoid(6.0 * (0.2 - max_ticker_dd))

        # Coverage Multiplier
        cov_p = sigmoid(float(args.coverage_k) * (float(coverage) - float(args.coverage_target)))

        # Final Composite Score
        final_score = blended_score * trade_density_mult * cov_p * dd_gate
        return float(final_score)


    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    # Fill in non-optimized fixed toggles
    if not args.opt_adaptive:
        best_params["basePeriod"] = int(args.basePeriod_fixed)
        best_params["minPeriod"] = int(args.minPeriod_fixed)
        best_params["maxPeriod"] = int(args.maxPeriod_fixed)
    if not args.opt_fastslow:
        best_params["fastPeriod"] = int(args.fastPeriod_fixed)
        best_params["slowPeriod"] = int(args.slowPeriod_fixed)
    if "smooth_len" not in best_params:
        best_params["smooth_len"] = int(args.smooth_len_fixed)

    best_cooldown = int(best_params.get("cooldown", args.cooldown)) if args.opt_cooldown else int(args.cooldown)
    best_time_stop = int(best_params.get("time_stop", args.time_stop)) if args.opt_time_stop else int(args.time_stop)

    best_min_eff = min_tp2sl_eff_for(int(best_params["atrPeriod"]))
    constraint_mode = "auto" if args.tp2sl_auto else "fixed"

    # FINAL REPORT (single fill args.fill)
    best_score_single, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
        file_paths,
        atrPeriod=int(best_params["atrPeriod"]),
        slMultiplier=float(best_params["slMultiplier"]),
        tpMultiplier=float(best_params["tpMultiplier"]),
        commission_rate_per_side=args.commission_rate_per_side,
        fill_mode=args.fill,
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
    )

    per_df = pd.DataFrame(per)
    per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
    per_df = per_df.sort_values(
        ["ticker_score", "total_return", "profit_factor_diag", "num_trades"],
        ascending=False
    )

    per_csv = out_dir / "per_ticker_summary.csv"
    per_df.to_csv(per_csv, index=False)

    print("\n=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===")
    print(f"Avg Profit Factor (raw):      {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else float('inf')}")
    print(f"Avg Profit Factor (diag):     {overall['avg_pf_diag']:.6f}")
    print(f"Avg Trades / Ticker:          {overall['avg_trades']:.3f}")
    print(f"Mean Ticker Score (eligible): {overall['mean_ticker_score']:.6f}")
    print(f"Coverage (eligible/total):    {eligible_count}/{len(per)} = {coverage:.3f}")
    print(f"Penalty enabled:              {args.penalty} (soft)")
    print(f"Soft penalty center/k:        {args.penalty_ret_center} / {args.penalty_ret_k}")
    print(f"Tail-protection ret floor/k:  {args.ret_floor} / {args.ret_floor_k}")
    print(f"Max-trades penalty:           max_trades={args.max_trades}, k={args.max_trades_k}")
    print(f"PF-floor penalty:             pf_floor={args.pf_floor}, k={args.pf_floor_k}")
    print(f"cooldown_bars:                {best_cooldown} (opt={args.opt_cooldown})  [loss-only]")
    print(f"time_stop_bars:               {best_time_stop} (0=disabled)  (opt={args.opt_time_stop})")
    print(f"num_neg (return<=0):          {overall['num_neg']}")
    print(f"commission_rate_per_side:     {args.commission_rate_per_side:.6f}")
    print(f"PF ROC baseline/k:            {args.pf_baseline:.3f} / {args.pf_k:.3f}")
    print(f"Trades ROC baseline/k:        {args.trades_baseline:.3f} / {args.trades_k:.3f}")
    print(f"weight_pf:                    {args.weight_pf:.3f}")
    print(f"score_power:                  {args.score_power:.3f}")
    print(f"min_trades gate:              {args.min_trades}")
    print(f"loss_floor (scoring):         {args.loss_floor}")
    print(f"min_tp2sl_eff (constraint):   {best_min_eff:.4f} ({constraint_mode})")
    print(f"threshold_mode:               {args.threshold_mode}")
    print(f"Score (single fill '{args.fill}'): {best_score_single:.6f}")

    print("\n=== BEST PARAMS ===")
    for k in sorted(best_params.keys()):
        print(f"{k}: {best_params[k]}")
    print(f"best_score (OPTUNA objective): {best.value}")
    print(f"fill_mode (final report): {args.fill}")
    print(f"Saved per-ticker CSV to: {per_csv}")

    print("\n=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===")
    print(per_df[
        ["ticker", "profit_factor", "profit_factor_diag", "num_trades", "ticker_score",
         "total_return", "gross_profit", "gross_loss", "maxdd", "eligible"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
