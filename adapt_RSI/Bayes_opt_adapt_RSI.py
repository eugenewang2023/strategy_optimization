#!/usr/bin/env python3
"""
Bayes_opt_adapt_RSI.py

✅ Matches Bayes_opt_adapt_half_RSI.py in EVERYTHING EXCEPT signal generation.

Everything kept identical to Bayes_opt_adapt_half_RSI.py:
- PARQUET-only loading + ensure_ohlc
- Fill modes: same_close / next_open
- Multiplicative equity model (start 1.0)
- Commission: commission_rate_per_side applied on entry and exit
- Exits: same trailing engine / optional hard SL / time-stop / close_on_sellSignal
- Cooldown: loss-only
- Scoring: identical sigmoid ROC weights and penalties (eligible-only mean score)
- Coverage metric + coverage penalty
- Robust objective: mean score across BOTH fills, gap penalty
- Optuna structure + tp2sl asymmetry constraint
- Report-only & report-both-fills
- Per-ticker CSV sorting

ONLY CHANGE (signals):
- Use adapt_RSI-style fast/slow RSI from adaptive_rsi via EMA:
    baseRSI -> adaptivePeriod (round + clamp)
    adaptive_rsi selected per-bar from RSI(1..maxPeriod)
    fast_rsi = EMA(adaptive_rsi, fastPeriod)
    slow_rsi = EMA(adaptive_rsi, slowPeriod)
    hot = fast_rsi - slow_rsi
    hot_sm = SMA(hot, smooth_len)   <-- keep SAME smoothing type as half_RSI
    buy/sell = crossover/crossunder(hot_sm, 0)

Per your request:
- smooth_len stays as an input/opt knob (same as half_RSI optimizer).
- NO trend filter / future filter.

Notes on CLI compatibility:
- All flags from Bayes_opt_adapt_half_RSI.py are preserved.
- Added optional fast/slow EMA knobs:
    --fastPeriod-fixed, --slowPeriod-fixed, --opt-fastslow
  If you don't use them, defaults are fast=4, slow=50 (from your Bayes_opt_adapt_RSI defaults).
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
    """
    TradingView-ish RSI: Wilder RMA of gains/losses.
    Supports length >= 1 (including RSI(1)).
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < 2:
        return out

    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    def rma(x: np.ndarray, L: int) -> np.ndarray:
        y = np.full(n, np.nan, dtype=float)
        if n < L:
            return y

        # seed at first non-nan window of length L
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

    # seed at first index where window has no NaNs
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
# Backtest core (dynamic_SR engine; ONLY signals differ)
# =============================
@dataclass
class TradeStats:
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    num_trades: int = 0
    total_return: float = 0.0
    maxdd: float = 0.0
    profit_factor: float = 0.0


def backtest_adapt_rsi_dynamic_engine(
    df: pd.DataFrame,
    *,
    atrPeriod: int,
    slMult: float,
    tpMult: float,
    commission_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool = True,
    trail_mode: str = "trail_only",          # "trail_only" or "trail_plus_hard_sl"
    close_on_sellSignal: bool = True,
    cooldown_bars: int = 0,
    time_stop_bars: int = 0,
    # --- adaptive period knobs (period selection) ---
    basePeriod: int = 15,
    minPeriod: int = 5,
    maxPeriod: int = 20,
    # --- EMA knobs for fast/slow lines ---
    fastPeriod: int = 4,
    slowPeriod: int = 50,
    # --- keep SAME smoothing layer as half_RSI ---
    smooth_len: int = 2,
) -> TradeStats:
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    if n < max(int(atrPeriod) + 2, 60):
        return TradeStats()

    # HA from REAL; signals use HA close
    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)
    sigClose = ha_c

    # REAL ATR + REAL high/low for risk engine
    real_atr = atr_wilder(h, l, c, int(atrPeriod))
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

    # =============================
    # Signal pipeline (ONLY part that differs vs adapt_half_RSI)
    #
    # adaptive_rsi -> EMA fast/slow -> hot -> SMA(hot, smooth_len) -> zero cross
    # =============================
    baseP = int(max(1, basePeriod))
    minP = int(max(2, minPeriod))
    maxP = int(max(minP, maxPeriod))

    fp = int(max(1, fastPeriod))
    sp = int(max(fp + 1, slowPeriod))  # keep sane ordering

    base_rsi = rsi_tv(sigClose, baseP)
    scaled = (100.0 - base_rsi) / 100.0
    adapt_raw = np.rint(scaled * float(maxP - minP) + float(minP))  # Pine: round()
    adapt_clipped = np.clip(adapt_raw, float(minP), float(maxP))
    adaptive = np.where(np.isfinite(adapt_clipped), adapt_clipped, float(minP)).astype(int)

    # Precompute RSI(1..maxP) and select per-bar adaptive_rsi
    rsi_by_len: Dict[int, np.ndarray] = {}
    for L in range(1, maxP + 1):
        rsi_by_len[L] = rsi_tv(sigClose, L)

    adaptive_rsi = np.full(n, np.nan, dtype=float)
    for i in range(n):
        p = int(adaptive[i])
        if p in rsi_by_len:
            adaptive_rsi[i] = float(rsi_by_len[p][i])

    fast_rsi = ema_tv(adaptive_rsi, fp)
    slow_rsi = ema_tv(adaptive_rsi, sp)

    hot = fast_rsi - slow_rsi
    sm = max(1, int(smooth_len))
    hot_sm = sma(hot, sm)

    # =============================
    # dynamic_SR-like trade engine
    # (gp/gl accounting: pnl percent, not equity-units)
    # =============================
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

        buy_sig = crossover(hot_sm[i - 1], hot_sm[i], 0.0, 0.0)
        sell_sig = crossunder(hot_sm[i - 1], hot_sm[i], 0.0, 0.0)

        # Entry
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

        # Manage open position
        if in_pos:
            bars_in_trade += 1

            stop_level = l[i] - real_atr[i] * slMult
            tgt_level = h[i] + real_atr[i] * tpMult
            trail_dist = real_atr[i] * tpMult

            exit_now = False
            exit_price: Optional[float] = None

            # Trailing activation/update
            if use_trailing_exit:
                if (not trail_active) and (h[i] >= entry + trail_dist):
                    trail_active = True
                    trail_high_since = float(h[i])
                    trail_stop = trail_high_since - trail_dist
                elif trail_active:
                    trail_high_since = float(h[i]) if np.isnan(trail_high_since) else max(trail_high_since, float(h[i]))
                    trail_stop = trail_high_since - trail_dist

            # Intrabar triggers (conservative long)
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

    # Force-close at end
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
# Scoring + evaluation (UNCHANGED from adapt_half_RSI optimizer)
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
    return float(s)


def evaluate_params_on_files(
    file_paths: List[Path],
    *,
    atrPeriod: int,
    slMultiplier: float,
    tpMultiplier: float,
    commission_rate_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool,
    trail_mode: str,
    close_on_sellSignal: bool,
    cooldown_bars: int,
    time_stop_bars: int,
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
    # signal knobs
    basePeriod: int,
    minPeriod: int,
    maxPeriod: int,
    fastPeriod: int,
    slowPeriod: int,
    smooth_len: int,
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], float, float, float, int]:
    per: List[Dict[str, Any]] = []
    eligible_scores: List[float] = []
    num_neg = 0
    eligible_count = 0

    for p in file_paths:
        df = pd.read_parquet(p)
        df = ensure_ohlc(df)

        st = backtest_adapt_rsi_dynamic_engine(
            df,
            atrPeriod=atrPeriod,
            slMult=slMultiplier,
            tpMult=tpMultiplier,
            commission_per_side=commission_rate_per_side,
            fill_mode=fill_mode,
            use_trailing_exit=use_trailing_exit,
            trail_mode=trail_mode,
            close_on_sellSignal=close_on_sellSignal,
            cooldown_bars=cooldown_bars,
            time_stop_bars=time_stop_bars,
            basePeriod=basePeriod,
            minPeriod=minPeriod,
            maxPeriod=maxPeriod,
            fastPeriod=fastPeriod,
            slowPeriod=slowPeriod,
            smooth_len=smooth_len,
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
        )

        is_eligible = (st.num_trades >= min_trades)
        if is_eligible:
            eligible_count += 1
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
            "eligible": bool(is_eligible),
        })

        if np.isfinite(sc):
            eligible_scores.append(float(sc))

    mean_score = float(np.mean(eligible_scores)) if eligible_scores else 0.0

    pf_vals = [x["profit_factor"] for x in per if np.isfinite(x["profit_factor"])]
    pf_avg = float(np.mean(pf_vals)) if pf_vals else float("inf")
    trades_avg = float(np.mean([x["num_trades"] for x in per])) if per else 0.0

    total = len(per)
    coverage = (eligible_count / total) if total > 0 else 0.0

    overall = {
        "mean_ticker_score": mean_score,
        "avg_pf_raw": pf_avg,
        "avg_trades": trades_avg,
        "num_neg": f"{num_neg}/{len(per)}" if per else "0/0",
        "eligible_count": eligible_count,
        "coverage": coverage,
    }
    return mean_score, overall, per, pf_avg, trades_avg, coverage, eligible_count


# =============================
# CLI + Main (kept matching adapt_half_RSI optimizer)
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
    ap.add_argument("--pf-cap", type=float, default=10.0, dest="pf_cap_score_only")
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

    # exits
    ap.add_argument("--use_trailing_exit", type=bool, default=True)
    ap.add_argument("--trail_mode", type=str, default="trail_only", choices=["trail_only", "trail_plus_hard_sl"])
    ap.add_argument("--close_on_sellSignal", type=bool, default=True)

    # cooldown
    ap.add_argument("--cooldown", type=int, default=1)
    ap.add_argument("--opt-cooldown", action="store_true")

    # time stop
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

    # Reporting helpers
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--report-both-fills", action="store_true")
    ap.add_argument("--atrPeriod-fixed", type=int, default=None)
    ap.add_argument("--slMultiplier-fixed", type=float, default=None)
    ap.add_argument("--tpMultiplier-fixed", type=float, default=None)

    # adaptive period knobs
    ap.add_argument("--basePeriod-fixed", type=int, default=15)
    ap.add_argument("--minPeriod-fixed", type=int, default=5)
    ap.add_argument("--maxPeriod-fixed", type=int, default=20)
    ap.add_argument("--opt-adaptive", action="store_true")

    # NEW (optional): fast/slow EMA knobs for adapt_RSI-style fast_rsi/slow_rsi
    ap.add_argument("--fastPeriod-fixed", type=int, default=4)
    ap.add_argument("--slowPeriod-fixed", type=int, default=50)
    ap.add_argument("--opt-fastslow", action="store_true")

    # keep smooth_len as input option (same as half_RSI)
    ap.add_argument("--smooth_len-fixed", type=int, default=2)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Validate constraint knobs
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
        if baseP < 5: # Increased from 1: Base RSI needs at least a week of data
            return False
        if minP < 5:  # Increased from 2: This kills the "2-period noise" fitting
            return False
        if maxP <= minP: 
            return False
        if maxP > 120: # Increased from 60: Allow the RSI to get very slow if needed
            return False
        # Ensure there is a meaningful "Adaptive Delta"
        if (maxP - minP) < 10:
            return False
        return True

    def validate_fastslow(fp: int, sp: int) -> bool:
        if fp < 3: # Floor of 3 to avoid 1-period "jitter"
            return False
        if sp < 10: # Slow period should be long enough to establish a trend
            return False
        # Crucial: Ensure the Slow period is at least 2x the Fast period
        # This prevents the signal line from being too "noisy"
        if sp < (fp * 2):
            return False
        if sp > 200:
            return False
        return True

    # =========================
    # REPORT ONLY MODE
    # =========================
    if args.report_only:
        required = [
            ("--atrPeriod-fixed", args.atrPeriod_fixed),
            ("--slMultiplier-fixed", args.slMultiplier_fixed),
            ("--tpMultiplier-fixed", args.tpMultiplier_fixed),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            raise SystemExit("Missing required flags: " + ", ".join(missing))

        atrP_fixed = int(args.atrPeriod_fixed)
        sl_fixed = float(args.slMultiplier_fixed)
        tp_fixed = float(args.tpMultiplier_fixed)

        min_eff = min_tp2sl_eff_for(atrP_fixed)
        if sl_fixed <= min_eff * tp_fixed:
            raise SystemExit(
                f"Constraint violated: slMultiplier ({sl_fixed}) <= "
                f"min_tp2sl_eff ({min_eff:.4f}) * tpMultiplier ({tp_fixed})"
            )

        baseP = int(args.basePeriod_fixed)
        minP = int(args.minPeriod_fixed)
        maxP = int(args.maxPeriod_fixed)
        if not validate_adaptive_params(baseP, minP, maxP):
            raise SystemExit("Invalid adaptive params: require base>=1, min>=2, max>=3, and min<=max.")

        fp = int(args.fastPeriod_fixed)
        sp = int(args.slowPeriod_fixed)
        if not validate_fastslow(fp, sp):
            raise SystemExit("Invalid fast/slow EMA params: require fast>=1, slow>=2, fast<slow, slow<=200.")

        sm = int(args.smooth_len_fixed)
        if sm < 1:
            raise SystemExit("--smooth_len-fixed must be >= 1")

        fills = ROBUST_FILLS if args.report_both_fills else [args.fill]

        for fm in fills:
            best_score_single, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
                file_paths,
                atrPeriod=atrP_fixed,
                slMultiplier=sl_fixed,
                tpMultiplier=tp_fixed,
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
                basePeriod=baseP,
                minPeriod=minP,
                maxPeriod=maxP,
                fastPeriod=fp,
                slowPeriod=sp,
                smooth_len=sm,
            )

            per_df = pd.DataFrame(per)
            per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
            per_df = per_df.sort_values(
                ["profit_factor", "ticker_score", "total_return", "num_trades"],
                ascending=False
            )

            per_csv = out_dir / f"per_ticker_summary_{fm}.csv"
            per_df.to_csv(per_csv, index=False)

            print(f"\n=== REPORT ONLY ({fm}) ===")
            print(f"Saved: {per_csv}")
            print(f"Avg PF: {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else float('inf')}")
            print(f"Avg Trades: {overall['avg_trades']}")
            print(f"Mean Score (eligible-only): {overall['mean_ticker_score']}")
            print(f"Coverage (eligible/total):  {eligible_count}/{len(per)} = {coverage:.3f}")
            print(f"num_neg: {overall['num_neg']}")
            print(f"Score: {best_score_single}")

        mode = "auto" if args.tp2sl_auto else "fixed"
        print("\n=== FIXED PARAMS USED ===")
        print(f"atrPeriod: {atrP_fixed}")
        print(f"slMultiplier: {sl_fixed}")
        print(f"tpMultiplier: {tp_fixed}")
        print(f"min_tp2sl_eff (constraint): {min_eff:.4f} ({mode})")
        print(f"adaptive: basePeriod={baseP}, minPeriod={minP}, maxPeriod={maxP}")
        print(f"fast/slow EMA: fastPeriod={fp}, slowPeriod={sp}")
        print(f"smoothing: smooth_len={sm}")
        return

    # =========================
    # OPTUNA
    # =========================
    def objective(trial: optuna.Trial) -> float:
            # 1. ATR & Risk (Phase 4 Polished)
            atrPeriod = trial.suggest_int("atrPeriod", 10, 25)
            slMultiplier = trial.suggest_float("slMultiplier", 2.5, 4.0)
            tpMultiplier = trial.suggest_float("tpMultiplier", 2.0, 4.5)

            # TP/SL Efficiency Check
            min_eff = min_tp2sl_eff_for(atrPeriod)
            if slMultiplier <= (min_eff * tpMultiplier):
                raise optuna.TrialPruned()

            # 2. Adaptive RSI - Consolidated to avoid Optuna Distribution Warnings
            if args.opt_adaptive:
                # We define these once and ONLY here
                baseP = trial.suggest_int("basePeriod", 30, 60) # Phase 4 Range
                minP = trial.suggest_int("minPeriod", 5, 15)
                maxP = trial.suggest_int("maxPeriod", 40, 90)
                
                if not validate_adaptive_params(baseP, minP, maxP):
                    raise optuna.TrialPruned()
            else:
                # Use fixed values from CLI, avoiding trial.suggest calls entirely
                baseP = int(args.basePeriod_fixed)
                minP = int(args.minPeriod_fixed)
                maxP = int(args.maxPeriod_fixed)
                # Still validate to ensure CLI inputs are sane
                if not validate_adaptive_params(baseP, minP, maxP):
                    raise optuna.TrialPruned()

            # 3. Fast/Slow & Smoothing
            if args.opt_fastslow:
                fastP = trial.suggest_int("fastPeriod", 4, 12)
                slowP = trial.suggest_int("slowPeriod", 40, 100)
                if not validate_fastslow(fastP, slowP):
                    raise optuna.TrialPruned()
            else:
                fastP = int(args.fastPeriod_fixed)
                slowP = int(args.slowPeriod_fixed)

            smooth_len = trial.suggest_int("smooth_len", 5, 15) # Phase 4 Stability Floor

            cooldown_bars = trial.suggest_int("cooldown", 1, 5) if args.opt_cooldown else int(args.cooldown)
            time_stop_bars = trial.suggest_int("time_stop", 4, 12) if args.opt_time_stop else int(args.time_stop)

            # -------------------------
            # 4) Evaluation Loop
            # -------------------------
            scores = []
            coverages = []
            for fm in ROBUST_FILLS:
                s, _, _, _, _, cov, _ = evaluate_params_on_files(
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
                    basePeriod=baseP,
                    minPeriod=minP,
                    maxPeriod=maxP,
                    fastPeriod=fastP,
                    slowPeriod=slowP,
                    smooth_len=smooth_len,
                )
                scores.append(float(s))
                coverages.append(float(cov))

            if not scores:
                return -1e9

            # -------------------------
            # 5) Robust Multi-Fill Scoring
            # -------------------------
            mean_s = float(np.mean(scores))
            # Gap penalty ensures the strategy isn't "fill-dependent"
            gap = float(np.max(scores) - np.min(scores))

            coverage = float(np.min(coverages)) if coverages else 0.0
            coverage_mult = sigmoid(float(args.coverage_k) * (coverage - float(args.coverage_target)))

            # Slightly higher gap penalty (0.40) to force consistency across fills
            lam = 0.15
            return (mean_s * coverage_mult) - (lam * gap)


    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    best_params["atrPeriod"] = int(best_params["atrPeriod"])
    best_params["slMultiplier"] = float(best_params["slMultiplier"])
    best_params["tpMultiplier"] = float(best_params["tpMultiplier"])
    best_params["smooth_len"] = int(best_params.get("smooth_len", int(args.smooth_len_fixed)))

    if args.opt_adaptive:
        best_params["basePeriod"] = int(best_params["basePeriod"])
        best_params["minPeriod"] = int(best_params["minPeriod"])
        best_params["maxPeriod"] = int(best_params["maxPeriod"])
    else:
        best_params["basePeriod"] = int(args.basePeriod_fixed)
        best_params["minPeriod"] = int(args.minPeriod_fixed)
        best_params["maxPeriod"] = int(args.maxPeriod_fixed)

    if args.opt_fastslow:
        best_params["fastPeriod"] = int(best_params["fastPeriod"])
        best_params["slowPeriod"] = int(best_params["slowPeriod"])
    else:
        best_params["fastPeriod"] = int(args.fastPeriod_fixed)
        best_params["slowPeriod"] = int(args.slowPeriod_fixed)

    best_cooldown = int(best_params.get("cooldown", 0)) if args.opt_cooldown else int(args.cooldown)
    best_time_stop = int(best_params.get("time_stop", 0)) if args.opt_time_stop else int(args.time_stop)

    best_min_eff = min_tp2sl_eff_for(best_params["atrPeriod"])
    constraint_mode = "auto" if args.tp2sl_auto else "fixed"

    best_score_single, overall, per, _, _, coverage, eligible_count = evaluate_params_on_files(
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
        basePeriod=best_params["basePeriod"],
        minPeriod=best_params["minPeriod"],
        maxPeriod=best_params["maxPeriod"],
        fastPeriod=best_params["fastPeriod"],
        slowPeriod=best_params["slowPeriod"],
        smooth_len=best_params["smooth_len"],
    )

    per_df = pd.DataFrame(per)
    per_df["ticker_score"] = per_df["ticker_score"].fillna(0.0)
    per_df = per_df.sort_values(
        ["profit_factor", "ticker_score", "total_return", "num_trades"],
        ascending=False
    )
    per_csv = out_dir / "per_ticker_summary.csv"
    per_df.to_csv(per_csv, index=False)

    print("\n=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===")
    print(f"Avg Profit Factor (raw):      {overall['avg_pf_raw'] if np.isfinite(overall['avg_pf_raw']) else float('inf')}")
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
    print(f"Score (single fill '{args.fill}'): {best_score_single:.6f}")

    print("\n=== BEST PARAMS ===")
    print(f"atrPeriod: {best_params['atrPeriod']}")
    print(f"slMultiplier: {best_params['slMultiplier']}")
    print(f"tpMultiplier: {best_params['tpMultiplier']}")
    print(f"basePeriod: {best_params['basePeriod']}")
    print(f"minPeriod: {best_params['minPeriod']}")
    print(f"maxPeriod: {best_params['maxPeriod']}")
    print(f"fastPeriod: {best_params['fastPeriod']}")
    print(f"slowPeriod: {best_params['slowPeriod']}")
    print(f"smooth_len: {best_params['smooth_len']}")
    print(f"best_score (OPTUNA objective): {best.value}")
    print(f"pf_cap_score_only: {args.pf_cap_score_only}")
    if args.opt_cooldown:
        print(f"cooldown (best): {best_cooldown}")
    if args.opt_time_stop:
        print(f"time_stop (best): {best_time_stop}")
    print(f"fill_mode (report): {args.fill}")
    print(f"Saved per-ticker CSV to: {per_csv}")

    print("\n=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===")
    print(per_df[["ticker", "profit_factor", "num_trades", "ticker_score", "total_return",
                 "gross_profit", "gross_loss", "maxdd", "eligible"]].to_string(index=False))


if __name__ == "__main__":
    main()
