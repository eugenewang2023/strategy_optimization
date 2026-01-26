#!/usr/bin/env python3
"""
Vidya_RSI.py — Hybrid Adaptive Edition

This file has been reorganized for clarity and extended with:
- Hybrid adaptive thresholding
- Adaptive ATR multipliers
- Volatility regime classification
- Trend regime classification
- Clean modular structure
- Full compatibility with your shell script + comparison tools
"""

# ============================================================
# Imports
# ============================================================

import math
import random
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from pathlib import Path
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import optuna

try:
    from optuna.integration import TQDMProgressBarCallback
except Exception:
    TQDMProgressBarCallback = None


def penalty_sigmoid(x, floor, k):
    """
    Smooth penalty: 1.0 when x >= floor, decays smoothly below.
    """
    if x >= floor:
        return 1.0
    return 1.0 / (1.0 + math.exp(k * (floor - x)))


def smooth_penalty(value, center, steepness, low=0.1, high=1.0):
    """
    Generic smooth penalty curve using a sigmoid.
    
    Parameters:
        value      : metric value (e.g., trades_med, coverage)
        center     : midpoint where penalty ≈ (low+high)/2
        steepness  : how sharp the transition is
        low        : minimum penalty multiplier
        high       : maximum penalty multiplier
        
    Returns:
        penalty factor in [low, high]
    """
    curve = 1 / (1 + math.exp(-steepness * (value - center)))
    return low + (high - low) * curve


# ============================================================
# handle interrupt gracefully
# ============================================================

import signal
import sys

def signal_handler(sig, frame):
    print("\nOptimization interrupted by user (Ctrl+C). Saving best so far...")
    # Optional: save study.best_params if needed
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


# ============================================================
# Module prefix for output files
# ============================================================

MODULE_PREFIX = "vidya_RSI_"

# Define the variable at the global level
loaded_data = {}

# ============================================================
# Adaptive Helper Functions
# ============================================================

def rolling_slope(series, window):
    # rolling_slope MUST return np.ndarray of len(series)
    series = np.asarray(series, dtype=np.float64)
    n = len(series)

    slopes = np.full(n, np.nan)

    if n < window + 1:
        return slopes

    x = np.arange(window)

    for i in range(window, n):
        y = series[i - window:i]

        if np.all(np.isnan(y)):
            continue

        x_mean = np.nanmean(x)
        y_mean = np.nanmean(y)

        num = np.nansum((x - x_mean) * (y - y_mean))
        den = np.nansum((x - x_mean) ** 2)

        if den == 0 or np.isnan(den):
            continue

        slopes[i] = num / den

    return slopes

def classify_volatility_regime(atr_pct: np.ndarray) -> float:
    """
    Returns a volatility regime score in [0, 1].
    0 = low volatility
    1 = high volatility
    """
    finite_vals = atr_pct[np.isfinite(atr_pct)]
    if finite_vals.size == 0:
        return 0.5

    med = np.median(finite_vals)
    # Normalize using a soft logistic transform
    return float(1.0 / (1.0 + math.exp(-20 * (med - 0.02))))


def classify_trend_regime(regime_ema: np.ndarray, slope_window: int = 5) -> float:
    """
    Computes a trend regime score in [0, 1].
    0 = no trend / chop
    1 = strong trend
    """
    slope = rolling_slope(regime_ema, slope_window)
    finite_slopes = slope[np.isfinite(slope)]
    if finite_slopes.size == 0:
        return 0.5

    med_slope = np.median(finite_slopes)
    # Normalize slope into [0,1]
    return float(1.0 / (1.0 + math.exp(-50 * med_slope)))


def adaptive_threshold(base_threshold: float,
                       vol_regime: float,
                       trend_regime: float) -> float:
    """
    Computes an adaptive entry threshold.

    - In high volatility: threshold increases (avoid noise)
    - In strong trends: threshold decreases (enter earlier)
    - In mixed regimes: threshold stays near base
    """
    # Volatility pushes threshold UP
    vol_adj = 1.0 + 0.8 * vol_regime

    # Trend pushes threshold DOWN
    trend_adj = 1.0 - 0.6 * trend_regime

    # Combine multiplicatively
    thr = base_threshold * vol_adj * trend_adj

    # Clamp to reasonable bounds
    return float(max(0.001, min(thr, 0.20)))


def adaptive_atr_multipliers(vol_regime: float,
                             trend_regime: float,
                             base_stop_mult: float = 1.5,
                             base_tgt_mult: float = 2.0):
    """
    Computes adaptive ATR multipliers for stop and target.

    - In high volatility: widen stops/targets
    - In strong trends: widen targets, tighten stops slightly
    - In chop: tighten both
    """
    # Volatility widens both
    vol_factor = 1.0 + 0.5 * vol_regime

    # Trend widens target but tightens stop
    stop_factor = 1.0 - 0.3 * trend_regime
    tgt_factor = 1.0 + 0.6 * trend_regime

    stop_mult = base_stop_mult * vol_factor * stop_factor
    tgt_mult = base_tgt_mult * vol_factor * tgt_factor

    # Clamp to avoid extreme values
    stop_mult = float(max(1.0, min(stop_mult, 3.0)))
    tgt_mult = float(max(1.0, min(tgt_mult, 4.0)))

    return stop_mult, tgt_mult


# ============================================================
# CHUNK 2 / 6 — Indicators (VIDYA, ZLEMA, Regime EMA, Momentum Slope)
# ============================================================

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures the DataFrame has open/high/low/close columns
    in lowercase and in the correct order.
    """
    cols = {c.lower(): c for c in df.columns}
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out


# ------------------------------------------------------------
# VIDYA (Variable Index Dynamic Average)
# ------------------------------------------------------------

import numpy as np

def vidya_ema(price, length=14, smooth=9):
    """
    Variable Index Dynamic Average (VIDYA) EMA implementation.
    
    Parameters:
        price: array-like of prices (close or HA close)
        length: VIDYA length (also used for CMO-like signal period)
        smooth: smoothing period for final alpha (usually smaller than length)
    
    Returns:
        NumPy array of VIDYA values (same length as price)
    """
    price = np.asarray(price, dtype=np.float64)
    n = len(price)
    
    if n < length + 1:
        return np.full(n, np.nan)

    # ── 1. Compute absolute momentum signal ────────────────────────────────
    # Absolute change over 'length' periods
    diff_L = np.diff(price, n=length)
    # Pad front with zeros/NaN so signal has same length as price
    signal = np.concatenate([np.zeros(length), np.abs(diff_L)])

    # ── 2. Normalize signal (Chande Momentum Oscillator style) ──────────────
    # Avoid div-by-zero: use safe denominator
    denom = signal + np.roll(signal, 1)
    denom[denom == 0] = 1e-10  # very small value instead of zero
    
    signal_norm = signal / denom
    
    # Clean NaN/inf (should be rare now, but still safe)
    signal_norm = np.nan_to_num(
        signal_norm,
        nan=0.0,
        posinf=1.0,   # if denom very small and signal positive
        neginf=-1.0
    )
    
    # Optional: smooth the normalized signal (common in some VIDYA variants)
    if smooth > 1:
        signal_norm = np.convolve(
            signal_norm,
            np.ones(smooth)/smooth,
            mode='valid'
        )
        # Pad back to original length
        signal_norm = np.concatenate([
            np.full(n - len(signal_norm), signal_norm[0]),
            signal_norm
        ])

    # ── 3. Compute dynamic alpha ────────────────────────────────────────────
    alpha_base = 2.0 / (length + 1)
    alpha = alpha_base * np.abs(signal_norm)  # abs ensures alpha in [0, alpha_base*2]

    # Clamp alpha to reasonable range (prevents instability)
    alpha = np.clip(alpha, 0.0, 1.0)

    # ── 4. Variable alpha EMA ───────────────────────────────────────────────
    vidya = np.zeros(n, dtype=np.float64)
    vidya[0] = price[0]

    for i in range(1, n):
        vidya[i] = alpha[i] * price[i] + (1 - alpha[i]) * vidya[i-1]

    return vidya
# ------------------------------------------------------------
# ZLEMA (Zero-Lag Exponential Moving Average)
# ------------------------------------------------------------

def zlema(series: np.ndarray, period: int) -> np.ndarray:
    """
    Zero-lag EMA using de-lagged price series.
    """
    pd_s = pd.Series(series)
    lag = (period - 1) // 2
    de_lagged = pd_s + (pd_s - pd_s.shift(lag))
    return de_lagged.ewm(span=period, adjust=False).mean().to_numpy()


# ------------------------------------------------------------
# Regime EMA (long-term trend filter)
# ------------------------------------------------------------

def regime_ema_series(close: np.ndarray, span: int) -> np.ndarray:
    """
    Computes a long-term EMA used for trend regime classification.
    """
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()


# ------------------------------------------------------------
# Momentum Slope (for fade exits and trend persistence)
# ------------------------------------------------------------

def momentum_slope(series: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Computes a rolling slope of a series to detect momentum fade.
    """
    return rolling_slope(series, window)


# ============================================================
# CHUNK 3 / 6 — PF Unification + Utility Functions
# ============================================================

# ------------------------------------------------------------
# Profit Factor Unification (single source of truth)
# ------------------------------------------------------------

def compute_pf_metrics(gp: float,
                       gl: float,
                       loss_floor: float,
                       pf_diag_cap: float):
    """
    Canonical PF pipeline.

    PF_raw:
        - gp / gl  (if gl > 0)
        - +inf     (if gl == 0 and gp > 0)
        - 1.0      (if gp <= 0 and gl <= 0)

    PF_eff:
        - gp / max(gl, loss_floor * gp)
        - 0 if gp <= 0

    PF_diag:
        - min(PF_eff, pf_diag_cap)
        - used for scoring and reporting
    """
    gp = float(gp)
    gl = float(gl)
    loss_floor = float(loss_floor)
    pf_diag_cap = float(pf_diag_cap)

    # PF_raw (debug only)
    if gl > 0.0:
        pf_raw = gp / gl
        pf_raw_is_inf = False
    else:
        if gp > 0.0:
            pf_raw = float("inf")
            pf_raw_is_inf = True
        else:
            pf_raw = 1.0
            pf_raw_is_inf = False

    # PF_eff
    if gp <= 0.0:
        pf_eff = 0.0
    else:
        denom = max(gl, loss_floor * gp)
        pf_eff = gp / denom if denom > 0.0 else 0.0

    # PF_diag
    pf_diag = min(pf_eff, pf_diag_cap)

    zero_loss = (gl == 0.0 and gp > 0.0)
    capped = (pf_eff > pf_diag_cap)

    return {
        "profit_factor_raw": pf_raw,
        "profit_factor_raw_is_inf": bool(pf_raw_is_inf),
        "profit_factor_eff": float(pf_eff),
        "profit_factor_diag": float(pf_diag),
        "zero_loss": int(zero_loss),
        "pf_capped": int(capped),
    }


def safe_pf_raw_for_csv(pf_raw: float, inf_placeholder: float = 1e9) -> float:
    """
    Only for CSV export. Never use this for averaging/scoring.
    """
    if not np.isfinite(pf_raw):
        return float(inf_placeholder)
    return float(pf_raw)


# ------------------------------------------------------------
# Robust Series Utilities
# ------------------------------------------------------------

def safe_series(x: pd.Series) -> pd.Series:
    """
    Coerces to numeric, removes infs and NaNs.
    """
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()


def safe_mean(x: pd.Series) -> float:
    s = safe_series(x)
    return float(s.mean()) if len(s) else 0.0


def safe_median(x: pd.Series) -> float:
    s = safe_series(x)
    return float(s.median()) if len(s) else 0.0


def trimmed_mean(x, trim_frac: float = 0.05) -> float:
    """
    Computes a trimmed mean, dropping trim_frac from each tail.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    x = np.sort(x)
    k = int(len(x) * trim_frac)
    if 2 * k >= len(x):
        return float(np.nanmean(x))
    return float(np.nanmean(x[k:-k]))


# ------------------------------------------------------------
# Sigmoid Helper
# ------------------------------------------------------------

def sigmoid(x: float) -> float:
    """
    Numerically stable sigmoid.
    """
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    else:
        ex = math.exp(x)
        return ex / (1.0 + ex)

def _sigmoid(x: float, k: float = 1.0) -> float:
    return 1.0 / (1.0 + math.exp(-k * x))


# ============================================================
# CHUNK 4 / 6 — Adaptive Backtest Engine + TradeStats
# ============================================================

def rsi_tv(close, length=14):
    close = np.asarray(close, dtype=float)
    if len(close) < length + 1:
        return np.full(len(close), np.nan)

    delta = np.diff(close)
    up = np.maximum(delta, 0)
    down = np.abs(np.minimum(delta, 0))

    # Pad to match original length (diff loses 1 element)
    up = np.concatenate(([0], up))
    down = np.concatenate(([0], down))

    ag = np.zeros(len(close))
    al = np.zeros(len(close))

    # First average (simple mean over first 'length' periods)
    if len(up) > length:
        ag[length-1] = np.mean(up[1:length+1])
        al[length-1] = np.mean(down[1:length+1])

    # Smoothed averages (Wilder method)
    for i in range(length, len(close)):
        ag[i] = (ag[i-1] * (length - 1) + up[i]) / length
        al[i] = (al[i-1] * (length - 1) + down[i]) / length

    # RS and RSI – only compute where al != 0 and ag/al defined
    rs = np.zeros(len(close))
    rsi = np.full(len(close), np.nan)

    for i in range(length-1, len(close)):
        if al[i] != 0:
            rs[i] = ag[i] / al[i]
            rsi[i] = 100 - (100 / (1 + rs[i]))
        elif ag[i] > 0:
            rsi[i] = 100.0
        else:
            rsi[i] = 0.0

    # Optional: check for NaN in smoothing
    # if np.isnan(ag[i]) or np.isnan(al[i]):
    #     rsi[i] = np.nan   # already handled by init

    return rsi


@dataclass
class ObjectiveConfig:
    # core weights
    weight_pf: float = 0.80
    score_power: float = 2.0

    # trades
    min_trades: float = 3.0
    trades_baseline: float = 3.0
    trades_k: float = 1.2

    # profit factor
    pf_baseline: float = 1.15
    pf_k: float = 2.5
    pf_cap: float = 4.0

    # return / GLPT
    ret_floor: float = -0.11
    ret_floor_k: float = 2.4
    min_glpt: float = 0.003
    min_glpt_k: float = 13.0

    # loss floor
    loss_floor: float = 0.001

    # coverage / stability
    coverage_target: float = 0.40
    coverage_k: float = 4.0
    stability_power: float = 1.0  # keep neutral; you can bump later

    # zero‑loss & cap softening
    zero_loss_soft_cap: float = 0.30   # above this, start penalizing
    cap_soft_cap: float = 0.40        # above this, start penalizing

    # global penalty cap (so scores don't collapse)
    max_penalty_mult: float = 0.40


@dataclass
class TradeStats:
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    maxdd: float = 0.0

    # Diagnostics
    num_neg_trades: int = 0
    profit_factor_raw: float = 0.0
    profit_factor_eff: float = 0.0
    profit_factor_diag: float = 0.0
    zero_loss: int = 0
    pf_capped: int = 0

    gl_per_trade: float = 0.0
    atr_pct_med: float = 0.0
    coverage: float = 0.0 
    stability: float = 0.0

@dataclass
class EnhancedTradeStats:
    """Enhanced stats with returns and trades data."""
    # Original TradeStats fields
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    maxdd: float = 0.0
    num_neg_trades: int = 0
    profit_factor_raw: float = 0.0
    profit_factor_eff: float = 0.0
    profit_factor_diag: float = 0.0
    zero_loss: int = 0
    pf_capped: int = 0
    gl_per_trade: float = 0.0
    atr_pct_med: float = 0.0
    coverage: float = 0.0
    stability: float = 0.0
    
    # New fields for enhanced analysis
    daily_returns: np.ndarray = None
    equity_curve: np.ndarray = None
    individual_trades: list = None  # List of dicts with trade details


def backtest_vidya_engine_enhanced(df: pd.DataFrame, **p) -> EnhancedTradeStats:
    """
    Enhanced hybrid-adaptive backtest engine.
    Returns complete data for objective functions.
    """
    # --------------------------------------------------------
    # Extract OHLC
    # --------------------------------------------------------
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    # --------------------------------------------------------
    # Regime length
    # --------------------------------------------------------
    regime_ratio = float(p.get("regime_ratio", 3.0))
    reg_len = int(p["slowPeriod"] * regime_ratio)

    if n < max(200, reg_len + 50):
        return EnhancedTradeStats()

    # --------------------------------------------------------
    # Heikin-Ashi close proxy
    # --------------------------------------------------------
    ha_c = (o + h + l + c) / 4.0

    # --------------------------------------------------------
    # ATR and ATR%
    # --------------------------------------------------------
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(25).mean().to_numpy()

    atr_pct = atr / (c + 1e-12)
    finite_atr_pct = atr_pct[np.isfinite(atr_pct)]
    atr_pct_med = float(np.nanmedian(finite_atr_pct)) if finite_atr_pct.size else 0.0

    # --------------------------------------------------------
    # Indicators: VIDYA (for regime only) + half_RSI fast/slow
    # --------------------------------------------------------
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])

    # --- half_RSI-style fast/slow RSI lengths ---
    slow_len = int(p["slowPeriod"])
    fast_len = max(1, int(round(slow_len / 2)))

    # --- Compute RSI on Heikin-Ashi close ---
    fast_rsi_raw = rsi_tv(ha_c, fast_len)
    slow_rsi_raw = rsi_tv(ha_c, slow_len)

    # --- Apply shift if provided ---
    shift = int(p.get("shift", 0))
    if shift > 0:
        fast_rsi = np.roll(fast_rsi_raw, shift)
        slow_rsi = np.roll(slow_rsi_raw, shift)
        fast_rsi[:shift] = np.nan
        slow_rsi[:shift] = np.nan
    else:
        fast_rsi = fast_rsi_raw
        slow_rsi = slow_rsi_raw

    # --- Invert slow RSI (half_RSI logic) ---
    slow_rsi_inv = 100.0 - slow_rsi

    # --- Hot spread identical to half_RSI ---
    hot = fast_rsi - slow_rsi_inv

    # --- Smooth hot spread ---
    def rolling_mean(arr, window, min_periods=1):
        arr = np.asarray(arr, dtype=np.float64).ravel()
        n = len(arr)
        
        if n == 0:
            return np.array([], dtype=float)
        
        window = int(window)
        if window <= 1:
            return arr.copy()
        
        if window > n:
            cumsum = np.cumsum(arr)
            result = cumsum / (np.arange(n) + 1)
            result[:min_periods-1] = np.nan
            return result
        
        kernel = np.ones(window, dtype=float) / window
        pad_width = window - 1
        padded = np.concatenate([
            np.full(pad_width, arr[0] if min_periods > 1 else np.nan),
            arr
        ])
        
        conv = np.convolve(padded, kernel, mode='valid')
        result = conv.copy()
        
        if min_periods > 1:
            result[:min_periods-1] = np.nan
        return result

    smooth_len = int(p.get("smooth_len", 5))
    hot_sm = rolling_mean(hot, smooth_len, min_periods=1)

    # --------------------------------------------------------
    # Regime EMA + slope
    # --------------------------------------------------------
    reg_ema = regime_ema_series(c, reg_len)
    non_nan_count = np.sum(~np.isnan(reg_ema))
    if non_nan_count < 10:
        return EnhancedTradeStats()
    
    reg_slope = momentum_slope(reg_ema, window=5)
    if (np.isscalar(reg_slope) or not isinstance(reg_slope, np.ndarray) 
        or reg_slope.ndim != 1 or len(reg_slope) != n):
        return EnhancedTradeStats()

    # --------------------------------------------------------
    # Adaptive regime classification
    # --------------------------------------------------------
    vol_regime = classify_volatility_regime(atr_pct)
    trend_regime = classify_trend_regime(reg_ema, slope_window=5)

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------
    base_thr = float(p.get("threshold", 0.04))
    thr = adaptive_threshold(base_thr, vol_regime, trend_regime)

    # --------------------------------------------------------
    # Adaptive ATR multipliers
    # --------------------------------------------------------
    stop_mult, tgt_mult = adaptive_atr_multipliers(
        vol_regime,
        trend_regime,
        base_stop_mult=1.5,
        base_tgt_mult=2.0
    )

    # --------------------------------------------------------
    # Backtest state with ENHANCED TRACKING
    # --------------------------------------------------------
    equity = 1.0
    gp = 0.0
    gl = 0.0
    trades = 0
    in_pos = False
    entry = 0.0
    entry_bar = 0
    bars_in_trade = 0
    cooldown_left = 0
    
    # ENHANCED: Track daily returns and equity curve
    daily_returns = np.zeros(n)
    equity_curve = np.ones(n)
    
    # ENHANCED: Track individual trades
    individual_trades = []
    
    fill_mode = p.get("fill_mode", "next_open")
    commission = float(p.get("commission", 0.0))
    cooldown_bars = int(p.get("cooldown_bars", 1))
    time_stop_bars = int(p.get("time_stop_bars", 15))
    use_regime = bool(p.get("use_regime", False))

    # --------------------------------------------------------
    # Main loop with ENHANCED TRACKING
    # --------------------------------------------------------
    for i in range(reg_len + 10, n - 2):
        # Track equity and returns for each day
        if i > 0:
            equity_curve[i] = equity
        
        # Cooldown after losing trades
        if cooldown_left > 0:
            cooldown_left -= 1
            daily_returns[i] = 0.0
            continue

        # Need valid hot_sm and ATR
        if not (np.isfinite(hot_sm[i]) and np.isfinite(hot_sm[i - 1]) and np.isfinite(atr[i])):
            daily_returns[i] = 0.0
            continue

        # ----------------------------------------------------
        # Regime filter (adaptive)
        # ----------------------------------------------------
        if use_regime and i - 5 >= 0:
            regime_ok = (
                c[i] > reg_ema[i] and
                reg_ema[i] > reg_ema[i - 5] and
                reg_slope[i] > 0
            )
        else:
            regime_ok = True

        # ----------------------------------------------------
        # ENTRY LOGIC — hot crosses above threshold
        # ----------------------------------------------------
        if (not in_pos) and (hot_sm[i - 1] <= thr) and (hot_sm[i] > thr) and regime_ok:
            entry = float(o[i + 1]) if fill_mode == "next_open" else float(c[i])
            if entry > 0 and np.isfinite(entry):
                in_pos = True
                entry_bar = i
                trades += 1
                bars_in_trade = 0
            daily_returns[i] = 0.0
            continue

        # ----------------------------------------------------
        # EXIT LOGIC
        # ----------------------------------------------------
        if in_pos:
            bars_in_trade += 1

            # Adaptive stop/target
            stop = entry - atr[i] * stop_mult
            tgt = entry + atr[i] * tgt_mult

            exit_p = None
            exit_reason = ""

            # 1) Hard stop
            if np.isfinite(stop) and l[i] <= stop:
                exit_p = float(stop)
                exit_reason = "stop_loss"

            # 2) Target
            elif np.isfinite(tgt) and h[i] >= tgt:
                exit_p = float(tgt)
                exit_reason = "target"

            # 3) Time-stop (adaptive)
            elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])
                exit_reason = "time_stop"

            # 4) Momentum fade exit
            elif hot_sm[i] < hot_sm[i - 3] if i - 3 >= 0 else False:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])
                exit_reason = "momentum_fade"

            # 5) Regime flip exit
            elif use_regime and reg_ema[i] < reg_ema[i - 3]:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])
                exit_reason = "regime_flip"

            # 6) Final bar
            elif i == n - 3:
                exit_p = float(c[i])
                exit_reason = "final_bar"

            # ------------------------------------------------
            # EXECUTE EXIT
            # ------------------------------------------------
            if exit_p is not None and np.isfinite(exit_p) and entry > 0:
                # Calculate P&L
                pnl = ((exit_p - entry) / entry) - (commission * 2.0)
                equity *= (1.0 + pnl)
                
                # ENHANCED: Record daily return
                daily_returns[i] = pnl
                equity_curve[i] = equity
                
                if pnl >= 0:
                    gp += pnl
                else:
                    gl += abs(pnl)
                    cooldown_left = cooldown_bars
                
                # ENHANCED: Record individual trade
                individual_trades.append({
                    'entry_price': entry,
                    'exit_price': exit_p,
                    'entry_bar': entry_bar,
                    'exit_bar': i,
                    'pnl': pnl,
                    'bars_held': bars_in_trade,
                    'exit_reason': exit_reason,
                    'is_win': pnl >= 0
                })
                
                in_pos = False
                continue
        
        # If no trade action, record 0 return
        daily_returns[i] = 0.0

    # Fill in the rest of equity curve
    for i in range(n - 2, n):
        equity_curve[i] = equity
    
    # Trim arrays to actual data length
    daily_returns = daily_returns[reg_len + 10:]
    equity_curve = equity_curve[reg_len + 10:]
    
    # --------------------------------------------------------
    # Max Drawdown
    # --------------------------------------------------------
    if len(equity_curve) >= 2:
        peak = np.maximum.accumulate(equity_curve)
        dd = (equity_curve / (peak + 1e-12)) - 1.0
        maxdd = float(dd.min())
    else:
        maxdd = 0.0

    # --------------------------------------------------------
    # PF metrics
    # --------------------------------------------------------
    pfm = compute_pf_metrics(
        gp, gl,
        loss_floor=float(p.get("loss_floor", 0.001)),
        pf_diag_cap=float(p.get("pf_cap_score_only", 5.0)),
    )

    pf_raw_csv = safe_pf_raw_for_csv(pfm["profit_factor_raw"], inf_placeholder=1e9)
    gl_per_trade = float(gl / max(trades, 1))
    
    # ENHANCED: Calculate num_neg_trades from individual trades
    num_neg_trades = sum(1 for trade in individual_trades if not trade['is_win'])

    # --------------------------------------------------------
    # Coverage & Stability
    # --------------------------------------------------------
    coverage = trades / max(n, 1)

    # ENHANCED: Calculate stability from daily returns
    if len(daily_returns) > 2:
        returns_std = np.std(daily_returns)
        stability = 1.0 / (1.0 + returns_std)
    else:
        stability = 0.0

    return EnhancedTradeStats(
        # Original fields
        gp=float(gp),
        gl=float(gl),
        trades=int(trades),
        tot_ret=float(equity - 1.0),
        maxdd=float(maxdd),
        num_neg_trades=int(num_neg_trades),
        profit_factor_raw=float(pf_raw_csv),
        profit_factor_eff=float(pfm["profit_factor_eff"]),
        profit_factor_diag=float(pfm["profit_factor_diag"]),
        zero_loss=int(pfm["zero_loss"]),
        pf_capped=int(pfm["pf_capped"]),
        gl_per_trade=float(gl_per_trade),
        atr_pct_med=float(atr_pct_med),
        coverage=float(coverage),
        stability=float(stability),
        
        # New enhanced fields
        daily_returns=daily_returns,
        equity_curve=equity_curve,
        individual_trades=individual_trades
    )


def stats_to_dataframes(stats_list, ticker_names=None):
    """
    Convert list of EnhancedTradeStats to returns_df and trades_df.
    
    Returns:
        returns_df: DataFrame with columns ['ticker', 'returns', 'equity']
        trades_df: DataFrame with columns ['ticker', 'profit_loss']
    """
    returns_data = []
    trades_data = []
    
    for idx, st in enumerate(stats_list):
        if not isinstance(st, EnhancedTradeStats):
            continue
            
        ticker = ticker_names[idx] if ticker_names and idx < len(ticker_names) else f"ticker_{idx}"
        
        # Build returns DataFrame
        if st.daily_returns is not None and len(st.daily_returns) > 0:
            for j, (ret, equity) in enumerate(zip(st.daily_returns, st.equity_curve)):
                returns_data.append({
                    'ticker': ticker,
                    'returns': ret,
                    'equity': equity
                })
        else:
            # Fallback: single return
            returns_data.append({
                'ticker': ticker,
                'returns': st.tot_ret,
                'equity': 1.0 + st.tot_ret
            })
        
        # Build trades DataFrame
        if st.individual_trades:
            for trade in st.individual_trades:
                trades_data.append({
                    'ticker': ticker,
                    'profit_loss': trade['pnl']
                })
        elif st.trades > 0:
            # Fallback: aggregate trade
            trades_data.append({
                'ticker': ticker,
                'profit_loss': st.gp - st.gl
            })
    
    returns_df = pd.DataFrame(returns_data) if returns_data else pd.DataFrame()
    trades_df = pd.DataFrame(trades_data) if trades_data else pd.DataFrame()
    
    return returns_df, trades_df


def objective_balanced_enhanced(trial, all_data, args):
    """
    Enhanced objective using the balanced objective functions.
    """
    # 1. Sample parameters
    p = sample_params(trial, args)
    
    # 2. Run enhanced backtest on all tickers
    stats = []
    for ticker, df in all_data.items():
        st = backtest_vidya_engine_enhanced(df, **p)
        stats.append(st)
    
    # 3. Convert to dataframes for objective functions
    ticker_names = list(all_data.keys())
    returns_df, trades_df = stats_to_dataframes(stats, ticker_names)
    
    # 4. Calculate aggregate metrics
    eligible_stats = [s for s in stats if s.trades >= args.min_trades]
    coverage = len(eligible_stats) / len(stats) if stats else 0.0
    
    # Calculate stability from daily returns
    all_returns = []
    for st in stats:
        if hasattr(st, 'daily_returns') and st.daily_returns is not None:
            all_returns.extend(st.daily_returns)
    
    if len(all_returns) > 1:
        returns_std = np.std(all_returns)
        stability = 1.0 / (1.0 + returns_std)
    else:
        stability = 0.5  # Default
    
    # 5. Prepare data for objective
    data_for_obj = {
        'coverage': coverage,
        'stability': stability
    }
    
    # 6. Use the appropriate objective function
    if len(returns_df) > 0 and len(trades_df) > 0:
        score = objective_balanced_simple(p, data_for_obj, returns_df, trades_df)
    else:
        # Fallback to minimal objective
        score = objective_balanced_minimal(p, data_for_obj, returns_df, trades_df)
    return score


def backtest_vidya_engine(df: pd.DataFrame, **p) -> TradeStats:
    """
    Hybrid-adaptive backtest engine.
    Integrates:
    - Adaptive thresholds
    - Adaptive ATR multipliers
    - Adaptive regime filter
    - Momentum fade exits
    - Time-stop logic
    - PF unification
    """

    # --------------------------------------------------------
    # Extract OHLC
    # --------------------------------------------------------
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    # --------------------------------------------------------
    # Regime length
    # --------------------------------------------------------
    regime_ratio = float(p.get("regime_ratio", 3.0))
    reg_len = int(p["slowPeriod"] * regime_ratio)

    if n < max(200, reg_len + 50):
        return TradeStats()

    # --------------------------------------------------------
    # Heikin-Ashi close proxy
    # --------------------------------------------------------
    ha_c = (o + h + l + c) / 4.0

    # --------------------------------------------------------
    # ATR and ATR%
    # --------------------------------------------------------
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(25).mean().to_numpy()

    atr_pct = atr / (c + 1e-12)
    finite_atr_pct = atr_pct[np.isfinite(atr_pct)]
    atr_pct_med = float(np.nanmedian(finite_atr_pct)) if finite_atr_pct.size else 0.0

    # --------------------------------------------------------
    # Indicators: VIDYA (for regime only) + half_RSI fast/slow
    # --------------------------------------------------------

    # VIDYA still used for regime classification
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])
    _ = v_main  # kept for potential future use / debugging

    # --- half_RSI-style fast/slow RSI lengths ---
    slow_len = int(p["slowPeriod"])
    fast_len = max(1, int(round(slow_len / 2)))

    # --- Compute RSI on Heikin-Ashi close (same as half_RSI) ---
    fast_rsi_raw = rsi_tv(ha_c, fast_len)
    slow_rsi_raw = rsi_tv(ha_c, slow_len)

    # --- Apply shift if provided ---
    shift = int(p.get("shift", 0))
    if shift > 0:
        fast_rsi = np.roll(fast_rsi_raw, shift)
        slow_rsi = np.roll(slow_rsi_raw, shift)
        fast_rsi[:shift] = np.nan
        slow_rsi[:shift] = np.nan
    else:
        fast_rsi = fast_rsi_raw
        slow_rsi = slow_rsi_raw

    # --- Invert slow RSI (half_RSI logic) ---
    slow_rsi_inv = 100.0 - slow_rsi

    # --- Hot spread identical to half_RSI ---
    hot = fast_rsi - slow_rsi_inv

    # --- Smooth hot spread (half_RSI SMA smoothing) ---
    smooth_len = int(p.get("smooth_len", 5))

    # Ensure hot is always a 1D numeric array
    def rolling_mean(arr, window, min_periods=1):
        """
        Safe rolling mean using NumPy convolution.
        Handles short arrays, large windows, and min_periods correctly.
        """
        arr = np.asarray(arr, dtype=np.float64).ravel()
        n = len(arr)
        
        if n == 0:
            return np.array([], dtype=float)
        
        window = int(window)
        if window <= 1:
            return arr.copy()
        
        if window > n:
            # Window larger than data → use cumulative mean
            cumsum = np.cumsum(arr)
            result = cumsum / (np.arange(n) + 1)
            result[:min_periods-1] = np.nan
            return result
        
        # Standard convolution case
        kernel = np.ones(window, dtype=float) / window
        
        # Pad front with first valid value (or zero/NaN)
        pad_width = window - 1
        padded = np.concatenate([
            np.full(pad_width, arr[0] if min_periods > 1 else np.nan),
            arr
        ])
        
        # Convolve in 'valid' mode
        conv = np.convolve(padded, kernel, mode='valid')
        
        # conv should now have exactly length n
        assert len(conv) == n, f"Length mismatch: conv={len(conv)}, expected={n}"
        
        result = conv.copy()
        
        # Apply min_periods: set early values to NaN if too few points
        if min_periods > 1:
            result[:min_periods-1] = np.nan
        return result

    # Then use it instead of pandas rolling:
    hot_sm = rolling_mean(hot, smooth_len, min_periods=1)
            
    # --------------------------------------------------------
    # Regime EMA + slope (must come BEFORE regime classification)
    # --------------------------------------------------------
    reg_ema = regime_ema_series(c, reg_len)
    non_nan_count = np.sum(~np.isnan(reg_ema))
    if non_nan_count < 10:  # arbitrary small threshold
        print(f"Warning: reg_ema too short/NaN-heav: np.sum(~np.isnan(reg_ema)) < 10")
        # return early with invalid stats or skip   
    reg_slope = momentum_slope(reg_ema, window=5)
    if (
        np.isscalar(reg_slope)
        or not isinstance(reg_slope, np.ndarray)
        or reg_slope.ndim != 1
        or len(reg_slope) != n
    ):
        return TradeStats()


    # --------------------------------------------------------
    # Adaptive regime classification
    # --------------------------------------------------------
    vol_regime = classify_volatility_regime(atr_pct)
    trend_regime = classify_trend_regime(reg_ema, slope_window=5)

    # --------------------------------------------------------
    # Adaptive threshold
    # --------------------------------------------------------
    base_thr = float(p.get("threshold", 0.04))
    thr = adaptive_threshold(base_thr, vol_regime, trend_regime)

    # --------------------------------------------------------
    # Adaptive ATR multipliers
    # --------------------------------------------------------
    stop_mult, tgt_mult = adaptive_atr_multipliers(
        vol_regime,
        trend_regime,
        base_stop_mult=1.5,
        base_tgt_mult=2.0
    )

    # --------------------------------------------------------
    # Backtest state
    # --------------------------------------------------------
    equity = 1.0
    gp = 0.0
    gl = 0.0
    trades = 0
    in_pos = False
    entry = 0.0
    bars_in_trade = 0
    cooldown_left = 0
    equity_curve = [1.0]
    num_neg_trades = 0

    fill_mode = p.get("fill_mode", "next_open")
    commission = float(p.get("commission", 0.0))
    cooldown_bars = int(p.get("cooldown_bars", 1))
    time_stop_bars = int(p.get("time_stop_bars", 15))
    use_regime = bool(p.get("use_regime", False))

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------
    for i in range(reg_len + 10, n - 2):

        # Cooldown after losing trades
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # Need valid hot_sm and ATR
        if not (np.isfinite(hot_sm[i]) and np.isfinite(hot_sm[i - 1]) and np.isfinite(atr[i])):
            continue

        # ----------------------------------------------------
        # Regime filter (adaptive)
        # ----------------------------------------------------
        if use_regime and i - 5 >= 0:
            regime_ok = (
                c[i] > reg_ema[i] and
                reg_ema[i] > reg_ema[i - 5] and
                reg_slope[i] > 0
            )
        else:
            regime_ok = True

        # ----------------------------------------------------
        # ENTRY LOGIC — hot crosses above threshold
        # ----------------------------------------------------
        if (not in_pos) and (hot_sm[i - 1] <= thr) and (hot_sm[i] > thr) and regime_ok:
            entry = float(o[i + 1]) if fill_mode == "next_open" else float(c[i])
            if entry > 0 and np.isfinite(entry):
                in_pos = True
                trades += 1
                bars_in_trade = 0
            continue

        # ----------------------------------------------------
        # EXIT LOGIC
        # ----------------------------------------------------
        if in_pos:
            bars_in_trade += 1

            # Adaptive stop/target
            stop = entry - atr[i] * stop_mult
            tgt = entry + atr[i] * tgt_mult

            exit_p = None

            # 1) Hard stop
            if np.isfinite(stop) and l[i] <= stop:
                exit_p = float(stop)

            # 2) Target
            elif np.isfinite(tgt) and h[i] >= tgt:
                exit_p = float(tgt)

            # 3) Time-stop (adaptive)
            elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])

            # 4) Momentum fade exit
            elif hot_sm[i] < hot_sm[i - 3] if i - 3 >= 0 else False:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])

            # 5) Regime flip exit
            elif use_regime and reg_ema[i] < reg_ema[i - 3]:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])

            # 6) Final bar
            elif i == n - 3:
                exit_p = float(c[i])

            # ------------------------------------------------
            # EXECUTE EXIT
            # ------------------------------------------------
            if exit_p is not None and np.isfinite(exit_p) and entry > 0:
                pnl = ((exit_p - entry) / entry) - (commission * 2.0)
                equity *= (1.0 + pnl)
                equity_curve.append(equity)

                if pnl >= 0:
                    gp += pnl
                else:
                    gl += abs(pnl)
                    cooldown_left = cooldown_bars
                    num_neg_trades += 1

                in_pos = False

    # --------------------------------------------------------
    # Max Drawdown
    # --------------------------------------------------------
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) >= 2:
        peak = np.maximum.accumulate(eq)
        dd = (eq / (peak + 1e-12)) - 1.0
        maxdd = float(dd.min())
    else:
        maxdd = 0.0

    # --------------------------------------------------------
    # PF metrics
    # --------------------------------------------------------
    pfm = compute_pf_metrics(
        gp, gl,
        loss_floor=float(p.get("loss_floor", 0.001)),
        pf_diag_cap=float(p.get("pf_cap_score_only", 5.0)),
    )

    pf_raw_csv = safe_pf_raw_for_csv(pfm["profit_factor_raw"], inf_placeholder=1e9)
    gl_per_trade = float(gl / max(trades, 1))

    # --------------------------------------------------------
    # Coverage & Stability
    # --------------------------------------------------------
    coverage = trades / max(n, 1)

    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) > 2:
        diffs = np.diff(eq)
        stability = 1.0 / (1.0 + np.std(diffs))
    else:
        stability = 0.0

    return TradeStats(
        gp=float(gp),
        gl=float(gl),
        trades=int(trades),
        tot_ret=float(equity - 1.0),
        maxdd=float(maxdd),
        num_neg_trades=int(num_neg_trades),
        profit_factor_raw=float(pf_raw_csv),
        profit_factor_eff=float(pfm["profit_factor_eff"]),
        profit_factor_diag=float(pfm["profit_factor_diag"]),
        zero_loss=int(pfm["zero_loss"]),
        pf_capped=int(pfm["pf_capped"]),
        gl_per_trade=float(gl_per_trade),
        atr_pct_med=float(atr_pct_med),
        coverage=float(coverage), 
        stability=float(stability),        
    )

def compute_per_ticker_diagnostics(per_ticker_stats: dict):
    """
    per_ticker_stats: dict[ticker -> TradeStats]

    Returns a dict of aggregate diagnostics used by the optimizer
    and for reporting.
    """
    if not per_ticker_stats:
        return {
            "coverage_med": 0.0,
            "stability_med": 0.0,
            "trades_med": 0.0,
        }

    coverages = []
    stabilities = []
    trades_list = []

    for tkr, st in per_ticker_stats.items():
        coverages.append(st.coverage)
        stabilities.append(st.stability)
        trades_list.append(st.trades)

    coverages = np.asarray(coverages, dtype=float)
    stabilities = np.asarray(stabilities, dtype=float)
    trades_arr = np.asarray(trades_list, dtype=float)

    return {
        "coverage_med": float(np.median(coverages)) if coverages.size else 0.0,
        "stability_med": float(np.median(stabilities)) if stabilities.size else 0.0,
        "trades_med": float(np.median(trades_arr)) if trades_arr.size else 0.0,
    }

# ============================================================
# CHUNK 5 / 6 — Scoring, Penalties, Stability, Objective Aggregation
# ============================================================

# ------------------------------------------------------------
# Scoring Function (PF_diag + Trades)
# ------------------------------------------------------------

def score_trial(st: TradeStats, args) -> float:
    """
    Unified multi-metric scoring function.
    All metrics are normalized to [0,1] using logistic or clamped transforms.
    Weighted sum → exponentiation → final score.
    """

    # ------------------------------------------------------------
    # 0. Hard filters
    # ------------------------------------------------------------
    if st.trades < args.min_trades:
        return 0.0

    # ------------------------------------------------------------
    # 1. Extract raw metrics
    # ------------------------------------------------------------
    pf   = st.profit_factor_diag
    glpt = st.gl_per_trade
    cov  = st.coverage
    stab = st.stability
    zl   = st.zero_loss

    # ------------------------------------------------------------
    # 2. Normalization helpers
    # ------------------------------------------------------------
    def logistic(x, k, x0):
        # Stable logistic transform
        return 1.0 / (1.0 + math.exp(-k * (x - x0)))

    def clamp01(x):
        return max(0.0, min(1.0, x))

    # ------------------------------------------------------------
    # 3. Normalized metrics
    # ------------------------------------------------------------
    pf_n   = logistic(pf,   args.pf_k,   args.pf_baseline)
    glpt_n = logistic(glpt, args.glpt_k, args.glpt_baseline)
    cov_n  = logistic(cov,  args.cov_k,  args.cov_baseline)
    stab_n = logistic(stab, args.stab_k, args.stab_baseline)
    zl_n   = logistic(zl,   args.zl_k,   args.zl_baseline)

    # Safety clamp (logistic is already bounded, but this prevents NaNs)
    pf_n   = clamp01(pf_n)
    glpt_n = clamp01(glpt_n)
    cov_n  = clamp01(cov_n)
    stab_n = clamp01(stab_n)
    zl_n   = clamp01(zl_n)

    # ------------------------------------------------------------
    # 4. Weighted combination
    # ------------------------------------------------------------
    base_score = (
        args.w_pf   * pf_n   +
        args.w_glpt * glpt_n +
        args.w_cov  * cov_n  +
        args.w_stab * stab_n +
        args.w_zl   * zl_n
    )

    # ------------------------------------------------------------
    # 5. Safety clamp before exponentiation
    # ------------------------------------------------------------
    safe_base = max(base_score, 0.0)

    # ------------------------------------------------------------
    # 6. Final score
    # ------------------------------------------------------------
    return float(safe_base ** args.score_power)


# ------------------------------------------------------------
# Stability Score (MAD-based)
# ------------------------------------------------------------

def stability_score(stats, args) -> float:
    """
    Computes a robustness score based on the dispersion of per-ticker scores.
    Uses median absolute deviation (MAD).
    """
    eligible = [st for st in stats if st.trades >= args.min_trades]
    if not eligible:
        return 0.0

    scores = np.asarray([score_trial(st, args) for st in eligible], dtype=float)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        return 0.0

    med = float(np.median(scores))
    mad = float(np.median(np.abs(scores - med)))

    if mad <= 0:
        return 1.0

    return float(1.0 / (1.0 + mad))


# === GLOBAL TUNED CONSTANTS ===

MAX_PENALTY_MULT = 0.40        # allows good configs to breathe
SCORE_POWER = 2.45            # sharpens separation

# GLPT floors tuned to your best region
MIN_GLPT = 0.0075
MIN_GLPT_K = 17

# Return floors tuned to your best region
RET_FLOOR = -0.092
RET_FLOOR_K = 3.0

# Zero-loss & PF-cap soft caps
ZERO_LOSS_SOFT = 0.12
ZERO_LOSS_HARD = 0.22

CAP_SOFT = 0.30
CAP_HARD = 0.50

def penalty_trades(trades: float, cfg: ObjectiveConfig) -> float:
    """
    Soft penalty if trades are below baseline; no extra reward above.
    Returns value in [0, 1], where 0 = no penalty, 1 = max.
    """
    if trades <= 0:
        return 1.0
    if trades >= cfg.trades_baseline:
        return 0.0
    gap = cfg.trades_baseline - trades
    # gentle logistic; gap=1 gives modest penalty
    return _sigmoid(cfg.trades_k * gap) - 0.5  # ~0.0–0.5 range

def penalty_pf(pf: float, cfg: ObjectiveConfig) -> float:
    """
    Penalize PF below baseline; no penalty above.
    """
    if pf <= 0:
        return 1.0
    if pf >= cfg.pf_baseline:
        return 0.0
    gap = cfg.pf_baseline - pf
    return _sigmoid(cfg.pf_k * gap) - 0.5

def penalty_ret_floor(ret_med: float, cfg: ObjectiveConfig) -> float:
    """
    Soft penalty if median return drops below floor.
    Floor is negative; we care about going *more* negative.
    """
    if ret_med >= cfg.ret_floor:
        return 0.0
    gap = cfg.ret_floor - ret_med  # positive if worse than floor
    # quadratic but scaled; keep it tame
    return min(1.0, cfg.ret_floor_k * (gap ** 2))

def penalty_glpt(glpt_med: float, cfg: ObjectiveConfig) -> float:
    """
    Penalize if GL per trade is below target.
    """
    if glpt_med >= cfg.min_glpt:
        return 0.0
    gap = cfg.min_glpt - glpt_med
    return min(1.0, cfg.min_glpt_k * (gap ** 2))

def penalty_zero_loss(zero_loss_pct):
    """
    Penalize excessive zero-loss tickers.
    """
    if zero_loss_pct <= ZERO_LOSS_SOFT:
        return 1.0
    if zero_loss_pct >= ZERO_LOSS_HARD:
        return 0.20
    return max(0.20, 1.0 - 4.0 * (zero_loss_pct - ZERO_LOSS_SOFT))

def penalty_cap(cap_pct: float, cfg: ObjectiveConfig) -> float:
    """
    Soft penalty for too many PF‑capped symbols.
    """
    if cap_pct <= cfg.cap_soft_cap:
        return 0.0
    gap = cap_pct - cfg.cap_soft_cap
    return min(1.0, 2.0 * gap)

def penalty_cap_pct(cap_pct):
    """
    Penalize excessive PF-capped tickers.
    Smooth decay from 1.0 → 0.20 between CAP_SOFT and CAP_HARD.
    """
    # Fully acceptable region
    if cap_pct <= CAP_SOFT:
        return 1.0

    # Fully penalized region
    if cap_pct >= CAP_HARD:
        return 0.20

    # Linear decay between soft and hard caps
    span = CAP_HARD - CAP_SOFT
    frac = (cap_pct - CAP_SOFT) / span

    # Interpolate from 1.0 → 0.20
    penalty = 1.0 - frac * (1.0 - 0.20)
    return max(0.20, float(penalty))


def penalty_coverage(cov, target, k):
    """
    Asymmetric coverage penalty.
    """
    if cov >= target:
        return 1.0
    return 1.0 / (1.0 + math.exp(k * (target - cov)))

def penalty_stability(stab):
    """
    Stability penalty: encourage MAD stability around 0.75–0.85.
    """
    if stab >= 0.80:
        return 1.0
    if stab <= 0.60:
        return 0.25
    return 0.25 + 0.75 * ((stab - 0.60) / 0.20)

def combine_penalties(penalties: dict[str, float], cfg: ObjectiveConfig) -> float:
    """
    Combine individual penalties into a single multiplicative factor in [0, max_penalty_mult].
    We treat each penalty as a weight in [0,1] and average them, then cap.
    """
    if not penalties:
        return 0.0
    vals = list(penalties.values())
    avg = float(np.nanmean(vals))
    return min(cfg.max_penalty_mult, max(0.0, avg))


# ------------------------------------------------------------
# Objective Aggregation (mean/median/hybrid)
# ------------------------------------------------------------

def robust_objective_aggregate(
    pf_med,
    glpt_med,
    ret_med,
    trades_med,
    weight_pf=0.80,
    score_power=SCORE_POWER
):
    """
    Core aggregate score before penalties.
    """

    # Normalize components
    pf_term = min(pf_med / 2.0, 1.0)          # PF 2.0+ is excellent
    glpt_term = min(glpt_med / 0.012, 1.0)    # 0.012 is typical for best region
    ret_term = min((ret_med + 0.02) / 0.05, 1.0)
    trades_term = min(trades_med / 6.0, 1.0)

    # Weighted hybrid edge
    base = (
        weight_pf * pf_term +
        0.15 * glpt_term +
        0.10 * ret_term +
        0.05 * trades_term
    )

    # Sharpen separation
    return max(0.0, base) ** score_power

# ------------------------------------------------------------
# Degeneracy Penalties (Zero-loss, PF-cap, GLPT, Return Floor, PF Floor, Max Trades)
# ------------------------------------------------------------

def objective_penalty_multiplier(stats, args):
    """
    Modern, smooth, non-destructive penalty engine.
    All penalties are soft, sigmoid-based, and multiplicative.
    """

    # Extract arrays
    zero_loss_pct = np.nanmean([st.zero_loss for st in stats])
    cap_pct       = np.nanmean([st.pf_capped for st in stats])
    glpt_med      = np.median([st.gl_per_trade for st in stats])
    ret_med       = np.median([st.tot_ret for st in stats])
    pf_med        = np.median([st.profit_factor_diag for st in stats])
    trades_med    = np.median([st.trades for st in stats])

    # ─────────────────────────────────────────────────────────────
    # ZERO‑LOSS PENALTY (encourages real losses)
    # target = args.zero_loss_target (usually 0.0)
    # k = args.zero_loss_k
    # ─────────────────────────────────────────────────────────────
    p_zero = sigmoid(-args.zero_loss_k * (zero_loss_pct - args.zero_loss_target))

    # ─────────────────────────────────────────────────────────────
    # PF CAP PENALTY (discourages capped PF)
    # target = args.cap_target (usually 0.10–0.20)
    # ─────────────────────────────────────────────────────────────
    p_cap = sigmoid(-args.cap_k * (cap_pct - args.cap_target))

    # ─────────────────────────────────────────────────────────────
    # GLPT MEDIAN PENALTY (encourages larger per‑trade gains)
    # target = args.min_glpt
    # ─────────────────────────────────────────────────────────────
    p_glpt = sigmoid(args.min_glpt_k * (glpt_med - args.min_glpt))

    # ─────────────────────────────────────────────────────────────
    # RETURN FLOOR PENALTY (protects against negative drift)
    # floor = args.ret_floor
    # ─────────────────────────────────────────────────────────────
    p_ret = sigmoid(args.ret_floor_k * (ret_med - args.ret_floor))

    # ─────────────────────────────────────────────────────────────
    # PF FLOOR PENALTY (ensures PF > 1.0)
    # floor = args.pf_floor (default 1.0)
    # ─────────────────────────────────────────────────────────────
    pf_floor = getattr(args, "pf_floor", 1.0)
    p_pf = sigmoid(args.pf_floor_k * (pf_med - pf_floor))

    # ─────────────────────────────────────────────────────────────
    # TRADES PENALTY (encourages enough trades)
    # baseline = args.trades_baseline
    # ─────────────────────────────────────────────────────────────
    p_trades = sigmoid(args.trades_k * (trades_med - args.trades_baseline))

    # ─────────────────────────────────────────────────────────────
    # FINAL PENALTY MULTIPLIER
    # ─────────────────────────────────────────────────────────────
    penalty_mult = (
        p_zero *
        p_cap *
        p_glpt *
        p_ret *
        p_pf *
        p_trades
    )

    return {
            "penalty_mult": float(penalty_mult),
            "zero_loss_pct": zero_loss_pct,
            "cap_pct": cap_pct,
            "glpt_med": glpt_med,
            "ret_med": ret_med,
            "pf_med": pf_med,
            "trades_med": trades_med,
            "glpt_target": args.min_glpt,
            "ret_floor": args.ret_floor,
            "pf_floor": getattr(args, "pf_floor", 1.0),
            # add others if needed
        }

def run_engine(params, loaded_data, args):
    """
    Runs the backtest engine across all tickers in loaded_data.
    Returns a list of TradeStats objects (one per ticker).
    """
    stats = []

    for ticker, df in loaded_data.items():
        try:
            st = backtest_vidya_engine(
                df,
                vidya_len=params["vidya_len"],
                vidya_smooth=params["vidya_smooth"],
                fastPeriod=params["fastPeriod"],
                slowPeriod=params["slowPeriod"],
                time_stop_bars=params["time_stop_bars"],
                regime_ratio=params["regime_ratio"],
                threshold=params["threshold"],
                vol_floor_mult=params["vol_floor_mult"],
                loss_floor=params["loss_floor"],
                cooldown_bars=params["cooldown_bars"],
                fill_mode=args.fill,
                commission=args.commission_rate_per_side,
                use_regime=args.use_regime,
                pf_cap_score_only=args.pf_cap,
            )
        except Exception as e:
            # Fail-safe: return empty stats for this ticker
            st = TradeStats()

        stats.append(st)

    return stats

# ============================================================
# CHUNK 6 / 6 — Optuna Objective + Reporting + CLI + main()
# ============================================================

def sample_params(trial, args):
    """
    Sample parameters for a trial.
    """
    # If user supplied --regime-ratio, use it.
    # Otherwise, search between min/max.
    if getattr(args, "regime_ratio", None) is not None:
        regime_ratio_val = float(args.regime_ratio)
    else:
        regime_ratio_val = trial.suggest_float(
            "regime_ratio",
            args.regime_ratio_min,
            args.regime_ratio_max
        )

    p = {
        "vidya_len": trial.suggest_int("vidya_len", 8, 24),
        "vidya_smooth": trial.suggest_int("vidya_smooth", 20, 60),
        "fastPeriod": trial.suggest_int("fastPeriod", 6, 18),
        "slowPeriod": trial.suggest_int("slowPeriod", 14, 28),
        "time_stop_bars": trial.suggest_int("time_stop_bars", 6, 18),

        "regime_ratio": regime_ratio_val,

        "threshold": trial.suggest_float("threshold", 0.0015, 0.008),
        "vol_floor_mult": trial.suggest_float("vol_floor_mult", 0.02, 0.20),
        "commission": args.commission_rate_per_side,
        "fill_mode": "next_open",
        "use_regime": args.use_regime,
        "loss_floor": 0.001,  # Use a realistic value
        "pf_cap_score_only": 5.0,  # Cap PF at 5x
        "cooldown_bars": trial.suggest_int("cooldown_bars", 1, 3),
        "shift": 0,
        "smooth_len": 5,
    }
    
    return p

def objective_for_params(params, args):
    """
    Run backtest with given parameters and compute score using enhanced engine.
    """
    # Run enhanced backtest on all tickers
    stats = []
    for ticker, df in loaded_data.items():
        try:
            st = backtest_vidya_engine_enhanced(df, **params)
        except Exception as e:
            st = EnhancedTradeStats()
        stats.append(st)
    
    # Use enhanced objective function
    ticker_names = list(loaded_data.keys())
    returns_df, trades_df = stats_to_dataframes(stats, ticker_names)
    
    # Calculate aggregate metrics
    eligible_stats = [s for s in stats if s.trades >= args.min_trades]
    coverage = len(eligible_stats) / len(stats) if stats else 0.0
    
    # Calculate stability from daily returns
    all_returns = []
    for st in stats:
        if hasattr(st, 'daily_returns') and st.daily_returns is not None:
            all_returns.extend(st.daily_returns)
    
    if len(all_returns) > 1:
        returns_std = np.std(all_returns)
        stability = 1.0 / (1.0 + returns_std)
    else:
        stability = 0.5  # Default
    
    # Prepare data for objective
    data_for_obj = {
        'coverage': coverage,
        'stability': stability
    }
    
    # Use the appropriate objective function
    if len(returns_df) > 0 and len(trades_df) > 0:
        score = objective_balanced_simple(params, data_for_obj, returns_df, trades_df)
    else:
        # Fallback to minimal objective
        score = objective_balanced_minimal(params, data_for_obj, returns_df, trades_df)
    
    # Diagnostics
    diagnostics = {
        "stats": stats,
        "params": params,
        "score": score,
    }
    
    return score, diagnostics

def objective(trial, all_data, args):
    """
    Full Phase‑F objective function with unified smooth penalties.
    """
    # ============================================================
    # 1. SAMPLE PARAMETERS (with regime_ratio override support)
    # ============================================================

    # If user supplied --regime-ratio, use it.
    # Otherwise, search between min/max.
    if getattr(args, "regime_ratio", None) is not None:
        regime_ratio_val = float(args.regime_ratio)
    else:
        if trial is None:
            # For objective_for_params, use default
            regime_ratio_val = args.regime_ratio_min
        else:
            regime_ratio_val = trial.suggest_float(
                "regime_ratio",
                args.regime_ratio_min,
                args.regime_ratio_max
            )

    p = {
        "vidya_len": trial.suggest_int("vidya_len", 8, 24) if trial else 16,
        "vidya_smooth": trial.suggest_int("vidya_smooth", 20, 60) if trial else 40,
        "fastPeriod": trial.suggest_int("fastPeriod", 6, 18) if trial else 12,
        "slowPeriod": trial.suggest_int("slowPeriod", 14, 28) if trial else 24,
        "time_stop_bars": trial.suggest_int("time_stop_bars", 6, 18) if trial else 12,

        "regime_ratio": regime_ratio_val,

        "threshold": trial.suggest_float("threshold", 0.0015, 0.008) if trial else 0.005,
        "vol_floor_mult": trial.suggest_float("vol_floor_mult", 0.02, 0.20) if trial else 0.10,
        "commission": args.commission_rate_per_side,
        "fill_mode": "next_open",
        "use_regime": args.use_regime,
        "loss_floor": trial.suggest_float("loss_floor", 0.0008, 0.0025) if trial else 0.0015,
        "pf_cap_score_only": args.pf_cap,
        "cooldown_bars": trial.suggest_int("cooldown_bars", 1, 3) if trial else 2,
        "shift": 0,
        "smooth_len": 5,
    }

    # ============================================================
    # 2. RUN BACKTEST ON ALL TICKERS
    # ============================================================
    stats = []
    for ticker, df in all_data.items():
        st = backtest_vidya_engine(df, **p)
        stats.append(st)

    df_stats = pd.DataFrame([s.__dict__ for s in stats])
    total_tickers = len(df_stats)

    # ============================================================
    # 3. COVERAGE + ELIGIBLE FILTER
    # ============================================================
    eligible_df = df_stats[df_stats["trades"] >= args.min_trades]
    eligible_count = len(eligible_df)
    coverage = eligible_count / total_tickers if total_tickers > 0 else 0.0

    # ============================================================
    # 4. METRIC CALCULATION (Phase F unified)
    # ============================================================
    trades_med = eligible_df["trades"].median() if eligible_count > 0 else 0.0
    pf_med = eligible_df["profit_factor_diag"].median() if eligible_count > 0 else 0.0
    glpt_med = eligible_df["gl_per_trade"].median() if eligible_count > 0 else 0.0
    maxdd_med = eligible_df["maxdd"].median() if eligible_count > 0 else 0.0
    zero_loss_pct = eligible_df["zero_loss"].mean() if eligible_count > 0 else 1.0

    # Spacing metric (fallback)
    if "spacing_score" in eligible_df.columns:
        spacing_score = eligible_df["spacing_score"].median()
    else:
        spacing_score = trades_med / (eligible_df["trades"].max() + 1e-9)

    # ============================================================
    # 5. STABILITY (MAD‑based)
    # ============================================================
    if eligible_count > 0:
        indiv_scores = np.asarray([score_trial(st, args) for st in stats])
        med = np.median(indiv_scores)
        mad = np.median(np.abs(indiv_scores - med))
        stability = float(1.0 / (1.0 + mad))
    else:
        stability = 0.0

    # ============================================================
    # 6. TUNED SMOOTH PENALTIES — PHASE F
    # ============================================================

    penalty_mult = 1.0

    # --- Coverage ---
    # Your universe naturally sits around 0.55–0.60.
    penalty_mult *= smooth_penalty(
        value=coverage,
        center=0.55,
        steepness=3.0,
        low=0.35,
        high=1.00
    )

    # --- Trades (median) ---
    # Your median trades are 4–7. Keep this gentle.
    penalty_mult *= smooth_penalty(
        value=trades_med,
        center=3.0,
        steepness=1.0,
        low=0.50,
        high=1.00
    )

    # --- GLPT (median) ---
    # Your GLPT median is ~0.0016–0.0020.
    penalty_mult *= smooth_penalty(
        value=glpt_med,
        center=args.min_glpt,      # typically ~0.0017–0.0018
        steepness=5.0,
        low=0.40,
        high=1.00
    )

    # --- Profit Factor (diagnostic PF) ---
    # Your PF_diag median is ~2.0–3.0.
    penalty_mult *= smooth_penalty(
        value=pf_med,
        center=2.0,
        steepness=4.0,
        low=0.40,
        high=1.00
    )

    # --- Spacing (median spacing_score) ---
    # Spacing is noisy; keep this soft.
    penalty_mult *= smooth_penalty(
        value=spacing_score,
        center=0.45,
        steepness=3.0,
        low=0.50,
        high=1.00
    )

    # --- Stability (MAD-based) ---
    # Your stability is ~0.75–0.80.
    penalty_mult *= smooth_penalty(
        value=stability,
        center=0.70,
        steepness=5.0,
        low=0.40,
        high=1.00
    )

    # --- Max Drawdown (invert sign so higher = better) ---
    # Your median DD is ~0.04–0.08.
    penalty_mult *= smooth_penalty(
        value=-maxdd_med,
        center=0.08,
        steepness=5.0,
        low=0.50,
        high=1.00
    )

    # --- Zero-loss realism ---
    # Your zero_loss_pct is usually 0.0–0.2.
    penalty_mult *= smooth_penalty(
        value=1 - zero_loss_pct,
        center=0.40,
        steepness=4.0,
        low=0.40,
        high=1.00
    )

    # --- Cap the multiplier ---
    penalty_mult = min(args.max_penalty_mult, penalty_mult)

    # ============================================================
    # 7. FINAL SCORE
    # ============================================================
    base_score = float(np.mean([score_trial(st, args) for st in stats]))
    final_score = base_score * penalty_mult
    return final_score

# ========== REQUIRED FUNCTIONS ==========

def calculate_sharpe(returns, risk_free_rate=0.02):
    """Annualized Sharpe Ratio"""
    if len(returns) < 2 or returns.std() == 0:
        return 0
    excess_returns = returns - risk_free_rate/252  # Daily risk-free
    return excess_returns.mean() / returns.std() * np.sqrt(252)

def calculate_max_drawdown(equity_curve):
    """Maximum drawdown (positive value)"""
    if len(equity_curve) == 0:
        return 0.5  # Conservative penalty
    
    peak = equity_curve.expanding().max()
    drawdown = (equity_curve - peak) / peak
    return abs(drawdown.min())  # Returns positive value (e.g., 0.15 for 15% DD)

# ========== OPTIONAL (but helpful) ==========

def calculate_sortino(returns, risk_free_rate=0.02, target_return=0):
    """Only needed if you want Sortino Ratio"""
    excess_returns = returns - risk_free_rate/252
    downside_returns = excess_returns[excess_returns < target_return]
    if len(downside_returns) < 2 or downside_returns.std() == 0:
        return 0
    return excess_returns.mean() / downside_returns.std() * np.sqrt(252)

def objective_balanced_minimal(params, data, returns_df, trades_df):
    """
    Minimal balanced objective - requires only pandas/numpy.
    """
    # 1. Basic metrics (handle empty cases)
    if len(returns_df) == 0 or len(trades_df) == 0:
        return -100  # Very bad score
    
    # Total Return
    total_return = returns_df['equity'].iloc[-1] / returns_df['equity'].iloc[0] - 1
    
    # Simple Risk-Adjusted Return (Sharpe-like)
    returns = returns_df['returns']
    if returns.std() > 0:
        risk_adj_return = returns.mean() / returns.std()
    else:
        risk_adj_return = 0
    
    # Win Rate
    win_rate = (trades_df['profit_loss'] > 0).mean()
    
    # Profit Factor
    wins = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
    losses = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
    profit_factor = wins / losses if losses > 0 else 1.0
    
    # Composite Score
    score = (
        total_return * 0.3 +
        risk_adj_return * 0.3 +
        min(profit_factor, 3.0) * 0.2 +  # Cap at 3
        win_rate * 0.2
    )
    
    # Apply data quality penalties
    score *= data.get('coverage', 0.5)
    score *= data.get('stability', 0.5)
    return score

def objective_balanced_simple(params, data, returns_df, trades_df):
    """
    Balanced objective - NO external dependencies needed.
    """
    # ========== 1. Calculate metrics from returns_df ==========
    # Total Return
    if len(returns_df) > 0:
        total_return = returns_df['equity'].iloc[-1] / returns_df['equity'].iloc[0] - 1
    else:
        total_return = -0.5  # Heavy penalty for no returns
    
    # Sharpe Ratio (built-in)
    if len(returns_df) > 1 and returns_df['returns'].std() > 0:
        risk_free_daily = 0.02 / 252
        excess_returns = returns_df['returns'] - risk_free_daily
        sharpe_ratio = excess_returns.mean() / returns_df['returns'].std() * np.sqrt(252)
    else:
        sharpe_ratio = 0
    
    # Max Drawdown (built-in)
    if len(returns_df) > 0:
        equity = returns_df['equity']
        peak = equity.expanding().max()
        drawdown = (equity - peak) / peak
        max_dd = abs(drawdown.min())
    else:
        max_dd = 0.5  # Conservative penalty
    
    # ========== 2. Calculate metrics from trades_df ==========
    if len(trades_df) > 0:
        # Win Rate
        win_rate = (trades_df['profit_loss'] > 0).mean()
        
        # Profit Factor
        gross_profit = trades_df[trades_df['profit_loss'] > 0]['profit_loss'].sum()
        gross_loss = abs(trades_df[trades_df['profit_loss'] < 0]['profit_loss'].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 1.0
        profit_factor = min(profit_factor, 5.0)  # Cap at 5.0
    else:
        win_rate = 0
        profit_factor = 0
    
    # ========== 3. Penalties ==========
    # Coverage penalty
    coverage = data.get('coverage', 0.5)
    coverage_penalty = max(0.5, coverage)
    
    # Stability penalty
    stability = data.get('stability', 0.5)
    stability_penalty = stability ** 2  # Quadratic
    
    # Diversity penalty
    if 'ticker' in trades_df.columns:
        unique_tickers = trades_df['ticker'].nunique()
        diversity_penalty = min(1.0, unique_tickers / 20)
    else:
        diversity_penalty = 0.5
    
    # ========== 4. Composite Score ==========
    weights = {
        'total_return': 0.25,
        'sharpe_ratio': 0.25,
        'profit_factor': 0.20,
        'max_dd': 0.15,  # Note: max_dd is positive, so we invert
        'win_rate': 0.15
    }
    
    # Invert max_dd (lower is better)
    max_dd_score = 1 - min(max_dd, 0.5)  # Cap at 50% DD
    
    # Normalize profit factor (0-5 becomes 0-1)
    pf_score = profit_factor / 5.0
    
    score = (
        total_return * weights['total_return'] +
        sharpe_ratio * weights['sharpe_ratio'] +
        pf_score * weights['profit_factor'] +
        max_dd_score * weights['max_dd'] +
        win_rate * weights['win_rate']
    )
    
    # Apply penalties
    score *= coverage_penalty
    score *= stability_penalty
    score *= diversity_penalty
    
    # Penalty for too few trades
    min_trades = 20
    trade_count = len(trades_df)
    if trade_count < min_trades:
        penalty = (trade_count / min_trades) ** 2  # Quadratic penalty
        score *= penalty
    return score


def update_optimizer_diagnostics(diag, trial, score):
    """
    Update optimizer diagnostics with trial results.
    """
    if score is None:
        diag["degenerate"] += 1
    elif score == 0.0:
        diag["degenerate"] += 1
    
    diag["best_curve"].append(score)
    if trial:
        diag["params"].append(trial.params)
    diag["values"].append(score)

def print_optimizer_health_report(diagnostics):
    diag = diagnostics

    values = [v for v in diag.get("values", []) if v is not None]
    best_values = [v for v in diag.get("best_curve", []) if v is not None]

    trials_completed = len(values)

    if trials_completed == 0:
        print("=== OPTIMIZER HEALTH REPORT ===")
        print("No completed trials.")
        print("=== END REPORT ===")
        return

    vals_arr = np.asarray(values, dtype=float)
    print("=== OPTIMIZER HEALTH REPORT ===")
    print(f"Trials completed: {trials_completed}")
    print(f"Score mean:       {vals_arr.mean():.6f}")
    print(f"Score median:     {np.median(vals_arr):.6f}")
    print(f"Score std:        {vals_arr.std(ddof=0):.6f}")
    print(f"Degenerate trials (0 or None): {diag.get('degenerate', 0)}")

    # Monotonic best-so-far curve
    if len(best_values) >= 2:
        best_arr = np.asarray(best_values, dtype=float)
        monotonic = np.all(best_arr[1:] >= best_arr[:-1])
    else:
        monotonic = True

    print(f"Best-so-far monotonic: {monotonic}")
    
    # Parameter diversity
    flat = []
    for p in diag.get("params", []):
        for k, v in p.items():
            flat.append((k, v))

    counts = Counter([k for k, v in flat])
    print("\nParameter usage counts:")
    for k, c in counts.most_common():
        print(f"  {k}: {c}")

    # Parameter entropy (search space health)
    print("\nParameter entropy:")
    if diag.get("params"):
        for key in diag["params"][0].keys():
            vals = np.array([p.get(key, np.nan) for p in diag["params"]])
            # Filter out NaN values
            vals = vals[~np.isnan(vals)]
            if len(vals) > 0:
                entropy = np.std(vals)
                print(f"  {key}: std={entropy:.4f}")

    print("\n=== END REPORT ===\n")


def main():
    global loaded_data

    # --------------------------------------------------------
    # CLI Arguments
    # --------------------------------------------------------
    ap = argparse.ArgumentParser()

    # Core
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--n-startup-trials", type=int, default=10)
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--files", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--fill", type=str, default="next_open",
                    choices=["next_open", "same_close"])
    ap.add_argument("--use-regime", action="store_true")

    # Objective modes
    ap.add_argument("--objective-mode", type=str, default="median_score",
                    choices=["mean_score", "median_score", "median_pf_diag",
                            "mean_pf_eff_excl_zero", "hybrid"])

    # Degeneracy penalties
    ap.add_argument("--obj-penalty-mode", type=str, default="both",
                    choices=["none", "zero_loss", "cap", "both"])
    ap.add_argument("--zero-loss-target", type=float, default=0.05)
    ap.add_argument("--zero-loss-k", type=float, default=12.0)
    ap.add_argument("--cap-target", type=float, default=0.30)
    ap.add_argument("--cap-k", type=float, default=6.0)

    ap.add_argument("--min-glpt", type=float, default=0.002)
    ap.add_argument("--min-glpt-k", type=float, default=12.0)

    ap.add_argument("--ret-floor", type=float, default=-0.15)
    ap.add_argument("--ret-floor-k", type=float, default=2.0)

    ap.add_argument("--pf-floor", type=float, default=1.0)
    ap.add_argument("--pf-floor-k", type=float, default=5.0)

    ap.add_argument("--max-trades", type=int, default=100)
    ap.add_argument("--max-trades-k", type=float, default=0.1)

    # Scoring knobs (legacy)
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)
    ap.add_argument("--pf-baseline", type=float, default=1.02)
    ap.add_argument("--pf-k", type=float, default=1.2)
    ap.add_argument("--trades-baseline", type=float, default=10.0)
    ap.add_argument("--trades-k", type=float, default=0.4)
    ap.add_argument("--weight-pf", type=float, default=0.4)
    ap.add_argument("--score-power", type=float, default=1.1)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--loss_floor", type=float, default=0.001)

    # === Unified Scoring Weights (canonical) ===
    ap.add_argument("--w_pf",   type=float, default=0.25)
    ap.add_argument("--w_glpt", type=float, default=0.25)
    ap.add_argument("--w_cov",  type=float, default=0.20)
    ap.add_argument("--w_stab", type=float, default=0.20)
    ap.add_argument("--w_zl",   type=float, default=0.10)
    ap.add_argument("--w_tr",   type=float, default=0.25)

    # === Normalization parameters (canonical) ===
    ap.add_argument("--glpt_k",        type=float, default=8.0)
    ap.add_argument("--glpt_baseline", type=float, default=0.0)

    ap.add_argument("--cov_k",         type=float, default=10.0)
    ap.add_argument("--cov_baseline",  type=float, default=0.90)

    ap.add_argument("--stab_k",        type=float, default=10.0)
    ap.add_argument("--stab_baseline", type=float, default=0.90)

    ap.add_argument("--zl_k",          type=float, default=10.0)
    ap.add_argument("--zl_baseline",   type=float, default=0.0)

    # Smooth penalty parameters
    ap.add_argument("--glpt_target", type=float, default=0.003)
    ap.add_argument("--glpt-k", type=float, default=1.0)
    ap.add_argument("--glpt-baseline", type=float, default=0.0)
    ap.add_argument("--spacing_center", type=float, default=0.50)
    ap.add_argument("--spacing_steepness", type=float, default=5.0)
    ap.add_argument("--dd_center", type=float, default=0.10)
    ap.add_argument("--dd_steepness", type=float, default=8.0)
    ap.add_argument("--zero_loss_center", type=float, default=0.70)
    ap.add_argument("--zero_loss_steepness", type=float, default=6.0)

    ap.add_argument("--threshold-fixed", type=float, default=0.04)
    ap.add_argument("--vol-floor-mult-fixed", type=float, default=0.55)
    ap.add_argument("--pf-cap", type=float, default=5.0)

    ap.add_argument("--coverage-target", type=float, default=0.85)
    ap.add_argument("--coverage-k", type=float, default=8.0)

    ap.add_argument("--max-penalty-mult", type=float, default=1.0)

    # Optimization toggles
    ap.add_argument("--opt-time-stop", action="store_true")
    ap.add_argument("--opt-vidya", action="store_true")
    ap.add_argument("--opt-fastslow", action="store_true")
    ap.add_argument("--fast-min", type=int, default=5)
    ap.add_argument("--fast-max", type=int, default=18)
    ap.add_argument("--slow-min", type=int, default=20)
    ap.add_argument("--slow-max", type=int, default=90)
    ap.add_argument("--vidya-len-fixed", type=int, default=None)
    ap.add_argument("--vidya-smooth-fixed", type=int, default=None)
    ap.add_argument("--time-stop-fixed", type=int, default=None)

    # Regime filter
    ap.add_argument("--regime-min", type=float, default=0.0)
    ap.add_argument("--regime-max", type=float, default=5.0)
    ap.add_argument("--regime-slope-min", type=float, default=0.0)
    ap.add_argument("--regime-persist", type=int, default=3)

    # Regime ratio search
    ap.add_argument("--regime-ratio-min", type=float, default=1.5)
    ap.add_argument("--regime-ratio-max", type=float, default=5.0)
    ap.add_argument("--regime-ratio", type=float, default=None)

    # Output
    ap.add_argument("--output_dir", type=str, default="output")
    ap.add_argument("--report-both-fills", action="store_true")

    # Add option to use enhanced objective
    ap.add_argument("--use-enhanced-objective", action="store_true", 
                   help="Use enhanced objective with returns and trades data")

    args = ap.parse_args()

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in: {args.data_dir}")

    sample_files = random.sample(files, min(len(files), args.files))
    data_list = [(f.stem, ensure_ohlc(pd.read_parquet(f))) for f in sample_files]
    loaded_data = dict(data_list)

    # --------------------------------------------------------
    # Wrap objective correctly
    # --------------------------------------------------------
    
    diagnostics = {
        "best_curve": [],
        "params": [],
        "values": [],
        "degenerate": 0,
    }

    def wrapped_objective(trial):
        params = sample_params(trial, args)
        if args.use_enhanced_objective:
            # Use enhanced objective
            score = objective_balanced_enhanced(trial, loaded_data, args)
        else:
            # Use original objective
            score, diagnostics_payload = objective_for_params(params, args)
        
        # Update Optuna diagnostics with the *score* we just computed
        update_optimizer_diagnostics(diagnostics, trial, score)
        return score

    # --------------------------------------------------------
    # Run Optuna
    # --------------------------------------------------------
    if args.optimize:
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=args.n_startup_trials,
            seed=args.seed
        )

        study = optuna.create_study(direction="maximize", sampler=sampler)

        if TQDMProgressBarCallback:
            callbacks = [TQDMProgressBarCallback()]
        else:
            callbacks = None

        study.optimize(
            wrapped_objective,
            n_trials=args.trials,
            n_jobs=args.n_jobs,
            callbacks=callbacks,
            show_progress_bar=True
        )

        print_optimizer_health_report(diagnostics)

        best_params = study.best_trial.params

        # Merge fixed overrides
        overrides = {
            "vidya_len": args.vidya_len_fixed,
            "vidya_smooth": args.vidya_smooth_fixed,
            "time_stop_bars": args.time_stop_fixed,
            "threshold": args.threshold_fixed,
            "loss_floor": args.loss_floor,
            "regime_ratio": args.regime_ratio_min,
        }
        best_params.update({k: v for k, v in overrides.items() if v is not None})

        best = dict(best_params)

    else:
        best = {
            "vidya_len": args.vidya_len_fixed if args.vidya_len_fixed else 16,
            "vidya_smooth": args.vidya_smooth_fixed if args.vidya_smooth_fixed else 40,
            "fastPeriod": args.fast_min,
            "slowPeriod": args.slow_min,
            "time_stop_bars": args.time_stop_fixed if args.time_stop_fixed else 12,
            "regime_ratio": args.regime_ratio_min,
        }

    # --------------------------------------------------------
    # Build best parameter set for evaluation
    # --------------------------------------------------------
    def get_param(best_dict, short_key, full_key, default):
        if full_key in best_dict:
            return best_dict[full_key]
        if short_key in best_dict:
            return best_dict[short_key]
        return default

    best_p = {
        "vidya_len": int(get_param(best, "vl", "vidya_len", 20)),
        "vidya_smooth": int(get_param(best, "vs", "vidya_smooth", 50)),
        "fastPeriod": int(get_param(best, "fp", "fastPeriod", 15)),
        "slowPeriod": int(get_param(best, "sp", "slowPeriod", 50)),
        "time_stop_bars": int(get_param(best, "ts", "time_stop_bars", 5)),
        "regime_ratio": float(get_param(best, "reg_ratio", "regime_ratio", 3.0)),
        "threshold": args.threshold_fixed,
        "vol_floor_mult": args.vol_floor_mult_fixed,
        "commission": args.commission_rate_per_side,
        "fill_mode": "next_open",
        "use_regime": bool(best.get("use_regime", args.use_regime)),
        "loss_floor": 0.001,  # FIXED: Use realistic loss floor
        "pf_cap_score_only": 5.0,  # FIXED: Cap PF at 5x, not 0.0056x
        "cooldown_bars": 1,
        "regime_slope_min": args.regime_slope_min,
        "regime_persist": args.regime_persist,
    }

    # --------------------------------------------------------
    # Evaluation + Reporting
    # --------------------------------------------------------
    def run_eval(fill_mode: str, suffix: str):
        p = dict(best_p)
        p["fill_mode"] = fill_mode

        stats = []
        rows = []

        for name, df in data_list:
            # Use enhanced engine for better reporting
            st = backtest_vidya_engine_enhanced(df, **p)
            stats.append(st)
            rows.append({
                "ticker": name,
                "gp": st.gp,
                "gl": st.gl,
                "trades": st.trades,
                "tot_ret": st.tot_ret,
                "maxdd": st.maxdd,
                "num_neg_trades": st.num_neg_trades,
                "profit_factor_raw": st.profit_factor_raw,
                "profit_factor_eff": st.profit_factor_eff,
                "profit_factor_diag": st.profit_factor_diag,
                "zero_loss": st.zero_loss,
                "pf_capped": st.pf_capped,
                "gl_per_trade": st.gl_per_trade,
                "atr_pct_med": st.atr_pct_med,
                "score": score_trial(st, args),
                "stability": st.stability,
                "coverage": st.coverage,
            })

        now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
        csv_path = Path(args.output_dir) / f"{MODULE_PREFIX}per_ticker_{suffix}_{now}.csv"
        df_out = pd.DataFrame(rows)
        df_out.to_csv(csv_path, index=False)

        eligible_stats = [s for s in stats if s.trades >= args.min_trades]
        coverage = len(eligible_stats) / len(stats) if stats else 0.0
        stab = stability_score(stats, args)

        pf_med = np.nanmedian([s.profit_factor_diag for s in eligible_stats]) if eligible_stats else 0.0
        glpt_med = np.nanmedian([s.gl_per_trade for s in eligible_stats]) if eligible_stats else 0.0
        ret_med = np.nanmedian([s.tot_ret for s in eligible_stats]) if eligible_stats else 0.0
        trades_med = np.nanmedian([s.trades for s in eligible_stats]) if eligible_stats else 0.0

        agg = robust_objective_aggregate(
            pf_med=pf_med,
            glpt_med=glpt_med,
            ret_med=ret_med,
            trades_med=trades_med,
            weight_pf=args.weight_pf,
            score_power=args.score_power
        )

        pen = objective_penalty_multiplier(stats, args)

        final_score = agg * max(pen.get("penalty_mult", 0.1), 0.1)

        return {
            "stats": stats,
            "agg": agg,
            "final_score": final_score,
            "pen": pen,
            "stab": stab,
            "coverage": coverage,
            "csv_path": csv_path,
        }

    # Ensure output directory exists
    Path(args.output_dir).mkdir(exist_ok=True)
    
    res_next = run_eval("next_open", "next_open")
    res_same = run_eval("same_close", "same_close") if args.report_both_fills else None

    # --------------------------------------------------------
    # Summary TXT
    # --------------------------------------------------------
    now = datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y%m%d_%H%M%S")
    summary_path = Path(args.output_dir) / f"{MODULE_PREFIX}best_{now}.txt"

    with summary_path.open("w") as f:
        f.write("Vidya_RSI Hybrid Adaptive — Summary\n")
        f.write(f"Generated: {now}\n")
        f.write(f"Data dir: {args.data_dir}\n")
        f.write(f"Files used: {len(data_list)} (requested {args.files})\n")
        f.write(f"Optimize: {args.optimize} (trials={args.trials})\n")
        f.write(f"Enhanced objective: {args.use_enhanced_objective}\n\n")

        f.write("Best parameters:\n")
        for k, v in best_p.items():
            f.write(f"  {k}: {v}\n")

        def write_block(label, res):
            f.write(f"\n=== {label} ===\n")
            f.write(f"Objective aggregate (raw): {res['agg']:.6f}\n")
            f.write(f"Final score (penalized): {res['final_score']:.6f}\n")
            f.write(f"Coverage: {res['coverage']:.3f}\n")
            f.write(f"Stability: {res['stab']:.6f}\n")

        write_block("next_open baseline", res_next)
        if res_same:
            write_block("same_close evaluation", res_same)

    print("Vidya_RSI Hybrid Adaptive — done.")
    print(f"next_open CSV: {res_next['csv_path']}")
    if res_same:
        print(f"same_close CSV: {res_same['csv_path']}")
    print(f"Summary TXT: {summary_path}")


if __name__ == "__main__":
    main()