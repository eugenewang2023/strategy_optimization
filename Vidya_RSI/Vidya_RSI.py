#!/usr/bin/env python3
"""
Vidya_RSI.py — Hybrid Adaptive Edition
(Chunk 1 / 6 — Imports, Constants, Adaptive Helpers)

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
import datetime
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import optuna

try:
    from optuna.integration import TQDMProgressBarCallback
except Exception:
    TQDMProgressBarCallback = None


# ============================================================
# Module prefix for output files
# ============================================================

MODULE_PREFIX = "vidya_RSI_"


# ============================================================
# Adaptive Helper Functions
# ============================================================

def rolling_slope(series: np.ndarray, window: int = 5) -> np.ndarray:
    """
    Computes a simple rolling slope using linear regression over a window.
    Used for trend persistence and momentum fade detection.
    """
    n = len(series)
    out = np.full(n, np.nan)
    if window < 2:
        return out

    x = np.arange(window)
    denom = float(np.sum((x - x.mean()) ** 2))

    for i in range(window, n):
        y = series[i - window:i]
        if np.any(~np.isfinite(y)):
            continue
        num = np.sum((x - x.mean()) * (y - y.mean()))
        out[i] = num / denom

    return out


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
                             base_stop_mult: float = 3.0,
                             base_tgt_mult: float = 3.6):
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
    stop_mult = float(max(1.0, min(stop_mult, 8.0)))
    tgt_mult = float(max(1.0, min(tgt_mult, 12.0)))

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

def vidya_ema(price: np.ndarray, length: int, smoothing: int) -> np.ndarray:
    """
    Kaufman-style VIDYA with adaptive smoothing.
    Uses volatility ratio (signal/noise) to modulate alpha.
    """
    n = len(price)
    vid = np.full(n, np.nan)

    L = max(2, int(length))
    S = max(1, int(smoothing))

    if n < L:
        return vid

    alpha_base = 2.0 / (S + 1.0)

    # Signal = |price - price[L bars ago]|
    signal = np.abs(pd.Series(price).diff(L).to_numpy())

    # Noise = rolling sum of absolute differences
    noise = pd.Series(np.abs(np.diff(price, prepend=price[0]))).rolling(L).sum().to_numpy()

    vi = signal / (noise + 1e-12)

    # Initialize
    vid[L - 1] = price[L - 1]

    for i in range(L, n):
        k = alpha_base * vi[i]
        prev = vid[i - 1] if np.isfinite(vid[i - 1]) else price[i - 1]
        vid[i] = price[i] * k + prev * (1.0 - k)

    return vid


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
        return float(np.mean(x))
    return float(np.mean(x[k:-k]))


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

# ============================================================
# CHUNK 4 / 6 — Adaptive Backtest Engine + TradeStats
# ============================================================

def rsi_tv(price: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView-style RSI using RMA (Wilder's smoothing).
    Fully NaN-safe and identical to half_RSI implementation.
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=float)
    if length <= 0 or n < 2:
        return out

    # Price change
    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    def rma(x, L):
        y = np.full(n, np.nan, dtype=float)
        if n < L:
            return y

        # Find first non-NaN window
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
    reg_len = int(p["slowPeriod"] * p.get("regime_ratio", 3.0))
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

    # # --------------------------------------------------------
    # # Indicators: VIDYA + ZLEMA
    # # --------------------------------------------------------
    # v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])
    # fast = zlema(v_main, int(p["fastPeriod"]))
    # slow = zlema(v_main, int(p["slowPeriod"]))

    # # --------------------------------------------------------
    # # Regime EMA + slope
    # # --------------------------------------------------------
    # reg_ema = regime_ema_series(c, reg_len)
    # reg_slope = momentum_slope(reg_ema, window=5)

    # # --------------------------------------------------------
    # # Hot spread (normalized)
    # # --------------------------------------------------------
    # hot = (fast - slow) / (atr + 1e-12)
    # hot_sm = pd.Series(hot).rolling(5).mean().to_numpy()

    # --------------------------------------------------------
    # Indicators: VIDYA (for regime only) + half_RSI fast/slow
    # --------------------------------------------------------

    # VIDYA still used for regime classification
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])

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
    hot_sm = pd.Series(hot).rolling(smooth_len).mean().to_numpy()

    # --------------------------------------------------------
    # Regime EMA + slope (must come BEFORE regime classification)
    # --------------------------------------------------------
    reg_ema = regime_ema_series(c, reg_len)
    reg_slope = momentum_slope(reg_ema, window=5)

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
        base_stop_mult=3.0,
        base_tgt_mult=3.6
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
    )

# ============================================================
# CHUNK 5 / 6 — Scoring, Penalties, Stability, Objective Aggregation
# ============================================================

# ------------------------------------------------------------
# Scoring Function (PF_diag + Trades)
# ------------------------------------------------------------

def score_trial(st: TradeStats, args) -> float:
    """
    Core scoring function used inside Optuna.
    Uses PF_diag (capped PF) and trade count.
    """
    if st.trades < args.min_trades:
        return 0.0

    # PF weight
    pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (st.profit_factor_diag - args.pf_baseline)))

    # Trades weight
    tr_w = 1.0 / (1.0 + math.exp(-args.trades_k * (st.trades - args.trades_baseline)))

    # Combined score
    base_score = args.weight_pf * pf_w + (1.0 - args.weight_pf) * tr_w

    return float(base_score ** args.score_power)


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


# ------------------------------------------------------------
# Objective Aggregation (mean/median/hybrid)
# ------------------------------------------------------------

def robust_objective_aggregate(stats, args, objective_mode: str) -> float:
    """
    Aggregates per-ticker scores into a single objective value.
    Supports:
        - mean_score
        - median_score
        - median_pf_diag
        - mean_pf_eff_excl_zero
        - hybrid
    """
    eligible = [st for st in stats if st.trades >= args.min_trades]
    if not eligible:
        return 0.0

    scores = np.asarray([score_trial(st, args) for st in eligible], dtype=float)
    pf_diag = np.asarray([st.profit_factor_diag for st in eligible], dtype=float)
    pf_eff = np.asarray([st.profit_factor_eff for st in eligible], dtype=float)
    zero_loss = np.asarray([st.zero_loss for st in eligible], dtype=int)

    med_score = float(np.median(scores)) if scores.size else 0.0
    mean_score = float(np.mean(scores)) if scores.size else 0.0
    med_pf_diag = float(np.median(pf_diag)) if pf_diag.size else 0.0

    mask_non_zero = (zero_loss == 0)
    mean_pf_eff_excl_zero = float(np.mean(pf_eff[mask_non_zero])) if np.any(mask_non_zero) else 0.0

    if objective_mode == "mean_score":
        return mean_score
    if objective_mode == "median_score":
        return med_score
    if objective_mode == "median_pf_diag":
        pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (med_pf_diag - args.pf_baseline)))
        return float(pf_w)
    if objective_mode == "mean_pf_eff_excl_zero":
        pf_eff_clip = min(max(mean_pf_eff_excl_zero, 0.0), 1000.0)
        return float(pf_eff_clip / (pf_eff_clip + 1.0))
    if objective_mode == "hybrid":
        pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (med_pf_diag - args.pf_baseline)))
        return float(med_score * pf_w)

    return med_score


# ------------------------------------------------------------
# Degeneracy Penalties (Zero-loss, PF-cap, GLPT, Return Floor, PF Floor, Max Trades)
# ------------------------------------------------------------

def objective_penalty_multiplier(stats, args) -> dict:
    """
    Computes all degeneracy penalties and returns a dict containing:
        - penalty_mult
        - zero_loss_pct, cap_pct
        - zero_loss_mult, cap_mult
        - glpt_med, glpt_target, glpt_mult
        - ret_med, ret_floor, ret_mult
        - pf_med, pf_floor, pf_floor_mult
        - trades_med, max_trades, trades_mult
        - vol_med
        - eligible_count, total_count
    """
    eligible = [st for st in stats if st.trades >= args.min_trades]
    total_count = len(stats)

    if not eligible:
        return {
            "penalty_mult": 0.0,
            "zero_loss_pct": 1.0,
            "cap_pct": 1.0,
            "zero_loss_mult": 0.0,
            "cap_mult": 0.0,
            "glpt_med": 0.0,
            "glpt_target": float(args.min_glpt),
            "glpt_mult": 0.0,
            "vol_med": 0.0,
            "ret_med": 0.0,
            "ret_floor": float(args.ret_floor),
            "ret_mult": 0.0,
            "pf_med": 0.0,
            "pf_floor": float(args.pf_floor),
            "pf_floor_mult": 0.0,
            "trades_med": 0.0,
            "max_trades": float(args.max_trades),
            "trades_mult": 0.0,
            "eligible_count": 0,
            "total_count": total_count,
        }

    # ------------------------------
    # Zero-loss fraction
    # ------------------------------
    z = np.asarray(
        [1.0 if (st.gl <= 0.0 and st.gp > 0.0) else 0.0 for st in eligible],
        dtype=float,
    )
    zero_loss_pct = float(np.mean(z)) if z.size else 0.0

    # ------------------------------
    # PF-cap fraction
    # ------------------------------
    cap_val = float(args.pf_cap)
    if cap_val > 0.0:
        c = np.asarray(
            [1.0 if st.profit_factor_eff > cap_val else 0.0 for st in eligible],
            dtype=float,
        )
    else:
        c = np.zeros(len(eligible), dtype=float)
    cap_pct = float(np.mean(c)) if c.size else 0.0

    # ------------------------------
    # GL per trade (GLPT)
    # ------------------------------
    glpt = np.asarray(
        [st.gl_per_trade for st in eligible],
        dtype=float,
    )
    glpt = glpt[np.isfinite(glpt)]
    glpt_med = float(np.median(glpt)) if glpt.size else 0.0

    glpt_target = float(args.min_glpt)
    min_glpt_k = float(args.min_glpt_k)

    if glpt_target > 0.0 and min_glpt_k > 0.0:
        x = min_glpt_k * (glpt_med - glpt_target)
        glpt_mult = 1.0 if x >= 0.0 else sigmoid(x)
    else:
        glpt_mult = 1.0

    # ------------------------------
    # Return floor penalty
    # ------------------------------
    rets = np.asarray([st.tot_ret for st in eligible], dtype=float)
    rets = rets[np.isfinite(rets)]
    ret_med = float(np.median(rets)) if rets.size else 0.0

    ret_floor = float(args.ret_floor)
    ret_floor_k = float(args.ret_floor_k)

    if ret_floor_k > 0.0:
        if ret_med >= ret_floor:
            ret_mult = 1.0
        else:
            xr = ret_floor_k * (ret_med - ret_floor)
            ret_mult = sigmoid(xr)
    else:
        ret_mult = 1.0

    # ------------------------------
    # PF floor penalty
    # ------------------------------
    pf_vals = np.asarray([st.profit_factor_diag for st in eligible], dtype=float)
    pf_vals = pf_vals[np.isfinite(pf_vals)]
    pf_med = float(np.median(pf_vals)) if pf_vals.size else 0.0

    pf_floor = float(args.pf_floor)
    pf_floor_k = float(args.pf_floor_k)

    if pf_floor_k > 0.0:
        if pf_med >= pf_floor:
            pf_floor_mult = 1.0
        else:
            xp = pf_floor_k * (pf_med - pf_floor)
            pf_floor_mult = sigmoid(xp)
    else:
        pf_floor_mult = 1.0

    # ------------------------------
    # Max-trades penalty
    # ------------------------------
    trades_arr = np.asarray([st.trades for st in eligible], dtype=float)
    trades_arr = trades_arr[np.isfinite(trades_arr)]
    trades_med = float(np.median(trades_arr)) if trades_arr.size else 0.0

    max_trades = float(args.max_trades)
    max_trades_k = float(args.max_trades_k)

    if max_trades_k > 0.0:
        if trades_med <= max_trades:
            trades_mult = 1.0
        else:
            xt = max_trades_k * (max_trades - trades_med)
            trades_mult = sigmoid(xt)
    else:
        trades_mult = 1.0

    # ------------------------------
    # Volatility median
    # ------------------------------
    vol_vals = np.asarray([st.atr_pct_med for st in eligible], dtype=float)
    vol_vals = vol_vals[np.isfinite(vol_vals)]
    vol_med = float(np.median(vol_vals)) if vol_vals.size else 0.0

    # ------------------------------
    # Combine penalties
    # ------------------------------
    mode = str(args.obj_penalty_mode).lower()
    if mode == "none":
        penalty_mult = 1.0
    elif mode == "zero_loss":
        penalty_mult = zero_loss_mult
    elif mode == "cap":
        penalty_mult = cap_mult
    else:  # "both"
        zero_loss_mult = 1.0 if zero_loss_pct <= args.zero_loss_target else sigmoid(
            args.zero_loss_k * (args.zero_loss_target - zero_loss_pct)
        )
        cap_mult = 1.0 if cap_pct <= args.cap_target else sigmoid(
            args.cap_k * (args.cap_target - cap_pct)
        )
        penalty_mult = zero_loss_mult * cap_mult

    # Add GLPT, return floor, PF floor, max trades
    penalty_mult *= glpt_mult
    penalty_mult *= ret_mult
    penalty_mult *= pf_floor_mult
    penalty_mult *= trades_mult

    return {
        "penalty_mult": float(penalty_mult),
        "zero_loss_pct": float(zero_loss_pct),
        "cap_pct": float(cap_pct),
        "zero_loss_mult": float(zero_loss_mult),
        "cap_mult": float(cap_mult),
        "glpt_med": float(glpt_med),
        "glpt_target": float(glpt_target),
        "glpt_mult": float(glpt_mult),
        "vol_med": float(vol_med),
        "ret_med": float(ret_med),
        "ret_floor": float(ret_floor),
        "ret_mult": float(ret_mult),
        "pf_med": float(pf_med),
        "pf_floor": float(pf_floor),
        "pf_floor_mult": float(pf_floor_mult),
        "trades_med": float(trades_med),
        "max_trades": float(max_trades),
        "trades_mult": float(trades_mult),
        "eligible_count": len(eligible),
        "total_count": total_count,
    }

# ============================================================
# CHUNK 6 / 6 — Optuna Objective + Reporting + CLI + main()
# ============================================================

def main():
    # --------------------------------------------------------
    # CLI Arguments
    # --------------------------------------------------------
    ap = argparse.ArgumentParser()

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

    # Return floor penalty
    ap.add_argument("--ret-floor", type=float, default=-0.15)
    ap.add_argument("--ret-floor-k", type=float, default=2.0)

    # PF floor penalty
    ap.add_argument("--pf-floor", type=float, default=1.0)
    ap.add_argument("--pf-floor-k", type=float, default=5.0)

    # Max trades penalty
    ap.add_argument("--max-trades", type=int, default=100)
    ap.add_argument("--max-trades-k", type=float, default=0.1)

    # Scoring knobs
    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)
    ap.add_argument("--pf-baseline", type=float, default=1.02)
    ap.add_argument("--pf-k", type=float, default=1.2)
    ap.add_argument("--trades-baseline", type=float, default=10.0)
    ap.add_argument("--trades-k", type=float, default=0.4)
    ap.add_argument("--weight-pf", type=float, default=0.4)
    ap.add_argument("--score-power", type=float, default=1.1)
    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--loss_floor", type=float, default=0.001)

    # Threshold + volatility floor
    ap.add_argument("--threshold-fixed", type=float, default=0.04)
    ap.add_argument("--vol-floor-mult-fixed", type=float, default=0.55)

    # PF cap
    ap.add_argument("--pf-cap", type=float, default=5.0)

    # Coverage penalty
    ap.add_argument("--coverage-target", type=float, default=0.85)
    ap.add_argument("--coverage-k", type=float, default=8.0)

    # Optimization toggles
    ap.add_argument("--opt-time-stop", action="store_true")
    ap.add_argument("--opt-vidya", action="store_true")
    ap.add_argument("--opt-fastslow", action="store_true")

    # Regime filter tuning
    ap.add_argument("--regime-slope-min", type=float, default=0.0)
    ap.add_argument("--regime-persist", type=int, default=3)

    ap.add_argument("--output_dir", type=str, default="output",
                    help="Directory to save per-ticker CSV and summary TXT")
    ap.add_argument("--report-both-fills", action="store_true",
                    help="Run evaluation for both fill modes: same_close and next_open")

    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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

    # --------------------------------------------------------
    # Optuna Objective
    # --------------------------------------------------------
    def objective(trial):
        p = {
            "vidya_len": trial.suggest_int("vl", 4, 40) if args.opt_vidya else 14,
            "vidya_smooth": trial.suggest_int("vs", 3, 60) if args.opt_vidya else 14,
            "fastPeriod": trial.suggest_int("fp", 3, 30) if args.opt_fastslow else 10,
            "slowPeriod": trial.suggest_int("sp", 15, 90) if args.opt_fastslow else 40,
            "time_stop_bars": trial.suggest_int("ts", 5, 60) if args.opt_time_stop else 15,
            "regime_ratio": trial.suggest_float("reg_ratio", 2.0, 5.0),
            "threshold": args.threshold_fixed,
            "vol_floor_mult": args.vol_floor_mult_fixed,
            "commission": args.commission_rate_per_side,
            "fill_mode": args.fill,
            "use_regime": args.use_regime,
            "loss_floor": args.loss_floor,
            "pf_cap_score_only": args.pf_cap,
            "cooldown_bars": 1,
            "regime_slope_min": args.regime_slope_min,
            "regime_persist": args.regime_persist,
        }

        stats = [backtest_vidya_engine(df, **p) for _, df in data_list]

        # Coverage
        eligible = sum(st.trades >= args.min_trades for st in stats)
        cov = eligible / len(stats)

        if cov >= args.coverage_target:
            cov_mult = 1.0
        else:
            cov_mult = sigmoid(args.coverage_k * (cov - args.coverage_target))

        # Aggregate objective
        agg = robust_objective_aggregate(stats, args, args.objective_mode)
        pen = objective_penalty_multiplier(stats, args)
        stab = stability_score(stats, args)

        return float(agg * cov_mult * pen["penalty_mult"] * stab)

    # --------------------------------------------------------
    # Run Optuna
    # --------------------------------------------------------
    study_kwargs = {"direction": "maximize"}

    if TQDMProgressBarCallback is not None:
        study = optuna.create_study(**study_kwargs)
        study.optimize(objective, n_trials=args.trials,
                       callbacks=[TQDMProgressBarCallback()])
    else:
        study = optuna.create_study(**study_kwargs)
        study.optimize(objective, n_trials=args.trials)

    best = dict(study.best_params)

    # --------------------------------------------------------
    # Build best parameter set
    # --------------------------------------------------------
    best_p = {
        "threshold": args.threshold_fixed,
        "vol_floor_mult": args.vol_floor_mult_fixed,
        "commission": args.commission_rate_per_side,
        "fill_mode": args.fill,
        "use_regime": args.use_regime,
        "loss_floor": args.loss_floor,
        "pf_cap_score_only": args.pf_cap,
        "cooldown_bars": 1,
        "time_stop_bars": best.get("ts", 15),
        "fastPeriod": best.get("fp", 10),
        "slowPeriod": best.get("sp", 40),
        "vidya_len": best.get("vl", 14),
        "vidya_smooth": best.get("vs", 14),
        "regime_ratio": best.get("reg_ratio", 3.0),
        "regime_slope_min": args.regime_slope_min,
        "regime_persist": args.regime_persist,
    }

    # --------------------------------------------------------
    # Per-ticker evaluation
    # --------------------------------------------------------
    rows = []
    best_stats = []

    for name, df in data_list:
        st = backtest_vidya_engine(df, **best_p)
        best_stats.append(st)
        rows.append({
            "ticker": name,
            "profit_factor_raw": st.profit_factor_raw,
            "profit_factor_eff": st.profit_factor_eff,
            "profit_factor_diag": st.profit_factor_diag,
            "pf_capped": st.pf_capped,
            "zero_loss": st.zero_loss,
            "num_trades": st.trades,
            "ticker_score": score_trial(st, args),
            "total_return": st.tot_ret,
            "gross_profit": st.gp,
            "gross_loss": st.gl,
            "gl_per_trade": st.gl_per_trade,
            "atr_pct_med": st.atr_pct_med,
            "maxdd": st.maxdd,
            "eligible": int(st.trades >= args.min_trades),
            "is_neg": int(st.tot_ret <= 0.0),
            "num_neg_trades": st.num_neg_trades,
        })

    per_df = pd.DataFrame(rows)

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------
    pen_best = objective_penalty_multiplier(best_stats, args)
    stab_best = stability_score(best_stats, args)

    eligible_count = int(per_df["eligible"].sum())
    coverage = eligible_count / len(per_df)

    med_pf_diag = safe_median(per_df["profit_factor_diag"])
    avg_pf_diag = safe_mean(per_df["profit_factor_diag"])
    med_pf_eff = safe_median(per_df["profit_factor_eff"])
    avg_pf_eff = safe_mean(per_df["profit_factor_eff"])
    med_trades = safe_median(per_df["num_trades"])
    med_tot_ret = safe_median(per_df["total_return"])

    # --------------------------------------------------------
    # Output files
    # --------------------------------------------------------
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add run-level diagnostics to CSV
    per_df["run_coverage"] = coverage
    per_df["run_zero_loss_pct"] = pen_best["zero_loss_pct"]
    per_df["run_cap_pct"] = pen_best["cap_pct"]
    per_df["run_glpt_med"] = pen_best["glpt_med"]
    per_df["run_vol_med"] = pen_best["vol_med"]
    per_df["run_stability_score"] = stab_best

    per_csv = out_dir / f"{MODULE_PREFIX}per_ticker_{run_ts}.csv"
    per_df.to_csv(per_csv, index=False)

    # Summary TXT
    txt_lines = []
    txt_lines.append("=== OBJECTIVE DEGENERACY DIAGNOSTICS ===")
    txt_lines.append(f"penalty_mult: {pen_best['penalty_mult']:.6f}")
    txt_lines.append(f"zero_loss_mult: {pen_best['zero_loss_mult']:.6f}")
    txt_lines.append(f"cap_mult: {pen_best['cap_mult']:.6f}")
    txt_lines.append(f"glpt_mult: {pen_best['glpt_mult']:.6f}")
    txt_lines.append(f"ret_mult: {pen_best['ret_mult']:.6f}")
    txt_lines.append(f"pf_floor_mult: {pen_best['pf_floor_mult']:.6f}")
    txt_lines.append(f"trades_mult: {pen_best['trades_mult']:.6f}")
    txt_lines.append("")
    txt_lines.append("=== COVERAGE & STABILITY ===")
    txt_lines.append(f"coverage: {coverage:.4f}  (eligible {eligible_count} / {len(per_df)})")
    txt_lines.append(f"zero_loss_pct: {pen_best['zero_loss_pct']:.4f}")
    txt_lines.append(f"cap_pct: {pen_best['cap_pct']:.4f}")
    txt_lines.append(f"stability_score: {stab_best:.6f}")
    txt_lines.append("")
    txt_lines.append("=== PF & RETURNS SUMMARY ===")
    txt_lines.append(f"median_pf_diag: {med_pf_diag:.4f}   avg_pf_diag: {avg_pf_diag:.4f}")
    txt_lines.append(f"median_pf_eff:  {med_pf_eff:.4f}   avg_pf_eff:  {avg_pf_eff:.4f}")
    txt_lines.append(f"median_trades:  {med_trades:.2f}")
    txt_lines.append(f"median_total_return: {med_tot_ret:.4f}")
    txt_lines.append("")
    txt_lines.append("=== BEST PARAMS ===")
    for k in sorted(study.best_params):
        txt_lines.append(f"{k}: {study.best_params[k]}")
    txt_lines.append("")
    txt_lines.append(f"objective_score: {study.best_value:.6f}")

    best_txt = out_dir / f"{MODULE_PREFIX}best_{run_ts}.txt"
    best_txt.write_text("\n".join(txt_lines), encoding="utf-8")

    # --------------------------------------------------------
    # Console summary
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print("          BEST RESULT — Vidya RSI (Hybrid Adaptive)")
    print("=" * 60)
    print(f"objective_score: {study.best_value:.6f}\n")
    print("Best parameters:")
    for k in sorted(study.best_params):
        print(f"  {k:18} : {study.best_params[k]}")
    print("\nCoverage / Stability:")
    print(f"  coverage         : {coverage:.4f}")
    print(f"  stability_score  : {stab_best:.6f}\n")
    print("Saved:")
    print(f"  CSV -> {per_csv}")
    print(f"  TXT -> {best_txt}\n")
    print("\n".join(txt_lines))


# ============================================================
# End of CHUNK 6 / 6 — Full Vidya_RSI.py Complete
# ============================================================

if __name__ == "__main__":
    main()
