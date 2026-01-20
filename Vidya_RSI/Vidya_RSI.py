#!/usr/bin/env python3
"""
Vidya_RSI.py - High Density Aggression Edition (PF-Unified Reporting)

Unified Engine: Kaufman VIDYA + ZLEMA + Regime Slope.
Unified Reporting: PF_raw vs PF_eff (loss_floor) vs PF_diag_cap (cap on PF_eff),
plus overall metrics, penalty diagnostics, and sorted ticker tables.

Reporting Fix:
- Print robust PF summaries so a few gl==0 tickers don't blow up averages.
- PF is computed exactly once via compute_pf_metrics().

Objective Fix:
- Robust Optuna objective aggregation via --objective-mode.
- Optional explicit objective penalties for:
  * too many zero-loss tickers
  * too many PF-capped tickers

This version fixes the "always-on shrinkage" smoking gun:
- zero_loss_mult and cap_mult are now ONE-SIDED (Option A style):
  -> 1.0 when compliant (<= target), penalize only when violating the threshold.
- GLPT is already ONE-SIDED hard-floor (as requested).
"""

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
    if n < L:
        return vidya
    alpha_base = 2.0 / (S + 1.0)
    signal = np.abs(pd.Series(price).diff(L).to_numpy())
    noise = pd.Series(np.abs(np.diff(price, prepend=price[0]))).rolling(L).sum().to_numpy()
    vi = signal / (noise + 1e-12)
    vidya[L - 1] = price[L - 1]
    for i in range(L, n):
        k = alpha_base * vi[i]
        prev = vidya[i - 1] if np.isfinite(vidya[i - 1]) else price[i - 1]
        vidya[i] = price[i] * k + prev * (1.0 - k)
    return vidya

def zlema(series: np.ndarray, period: int) -> np.ndarray:
    pd_s = pd.Series(series)
    lag = (period - 1) // 2
    de_lagged = pd_s + (pd_s - pd_s.shift(lag))
    return de_lagged.ewm(span=period, adjust=False).mean().to_numpy()

# =================================================================================
# PF UNIFICATION (single source of truth)
# =================================================================================

def compute_pf_metrics(gp: float, gl: float, loss_floor: float, pf_diag_cap: float):
    """
    Canonical PF pipeline.

    PF_raw: gp/gl (inf if gl==0,gp>0) [debug only]
    PF_eff: gp / max(gl, loss_floor*gp)
    PF_diag: min(PF_eff, pf_diag_cap) [headline + scoring]
    """
    gp = float(gp)
    gl = float(gl)
    loss_floor = float(loss_floor)
    pf_diag_cap = float(pf_diag_cap)

    # PF_raw (debug only)
    if gl > 0:
        pf_raw = gp / gl
        pf_raw_is_inf = False
    else:
        if gp > 0:
            pf_raw = float("inf")
            pf_raw_is_inf = True
        else:
            pf_raw = 1.0
            pf_raw_is_inf = False

    # PF_eff
    if gp <= 0:
        pf_eff = 0.0
    else:
        denom = max(gl, loss_floor * gp)
        pf_eff = gp / denom if denom > 0 else 0.0

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
    """Only for CSV export. Never use this for averaging/scoring."""
    if not np.isfinite(pf_raw):
        return float(inf_placeholder)
    return float(pf_raw)

# =================================================================================
# BACKTEST
# =================================================================================

@dataclass
class TradeStats:
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    maxdd: float = 0.0

    # trade-level diagnostics
    num_neg_trades: int = 0

    # unified PFs
    profit_factor_raw: float = 0.0
    profit_factor_eff: float = 0.0
    profit_factor_diag: float = 0.0

    # objective / penalty diagnostics
    zero_loss: int = 0
    pf_capped: int = 0

    # NEW: requested
    gl_per_trade: float = 0.0

    # optional vol proxy (even if you won’t use it now)
    atr_pct_med: float = 0.0

def backtest_vidya_engine(df: pd.DataFrame, **p) -> TradeStats:
    o, h, l, c = df["open"].values, df["high"].values, df["low"].values, df["close"].values
    n = len(c)

    reg_len = int(p["slowPeriod"] * p.get("regime_ratio", 3.0))
    if n < max(200, reg_len + 50):
        return TradeStats()

    ha_c = (o + h + l + c) / 4.0

    # True range + ATR
    prev_c = np.roll(c, 1)
    prev_c[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(25).mean().to_numpy()

    # ATR% proxy for volatility scaling (median over valid points)
    atr_pct = atr / (c + 1e-12)
    finite_atr_pct = atr_pct[np.isfinite(atr_pct)]
    atr_pct_med = float(np.nanmedian(finite_atr_pct)) if finite_atr_pct.size else 0.0

    # VIDYA + ZLEMA fast/slow
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])
    fast = zlema(v_main, int(p["fastPeriod"]))
    slow = zlema(v_main, int(p["slowPeriod"]))

    # Regime EMA on close
    regime_ema = pd.Series(c).ewm(span=reg_len, adjust=False).mean().to_numpy()

    # Hot (normalized spread) smoothed
    hot = (fast - slow) / (atr + 1e-12)
    hot_sm = pd.Series(hot).rolling(5).mean().to_numpy()

    # Backtest state
    equity = 1.0
    gp, gl, trades = 0.0, 0.0, 0
    in_pos, entry, bars_in_trade, cooldown_left = False, 0.0, 0, 0
    equity_curve = [1.0]
    num_neg_trades = 0

    fill_mode = p.get("fill_mode", "next_open")
    commission = float(p.get("commission", 0.0))
    cooldown_bars = int(p.get("cooldown_bars", 1))
    time_stop_bars = int(p.get("time_stop_bars", 15))
    threshold = float(p.get("threshold", 0.04))
    use_regime = bool(p.get("use_regime", False))

    for i in range(reg_len + 10, n - 2):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        # Need finite hot_sm values for cross logic
        if not (np.isfinite(hot_sm[i]) and np.isfinite(hot_sm[i - 1]) and np.isfinite(atr[i])):
            continue

        # Optional regime filter (guard i-5)
        if use_regime and i - 5 >= 0:
            regime_ok = (c[i] > regime_ema[i]) and (regime_ema[i] > regime_ema[i - 5])
        else:
            regime_ok = True

        # Entry: hot crosses above +threshold
        if (not in_pos) and (hot_sm[i - 1] <= threshold) and (hot_sm[i] > threshold) and regime_ok:
            entry = float(o[i + 1]) if fill_mode == "next_open" else float(c[i])
            if entry > 0 and np.isfinite(entry):
                in_pos = True
                trades += 1
                bars_in_trade = 0

        elif in_pos:
            bars_in_trade += 1

            stop = entry - atr[i] * 3.0
            tgt  = entry + atr[i] * 3.6

            exit_p = None
            if np.isfinite(stop) and l[i] <= stop:
                exit_p = float(stop)
            elif np.isfinite(tgt) and h[i] >= tgt:
                exit_p = float(tgt)
            elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars and c[i] <= entry:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])
            elif np.isfinite(hot_sm[i]) and hot_sm[i] < -threshold:
                exit_p = float(o[i + 1] if i + 1 < n else c[i])
            elif i == n - 3:
                exit_p = float(c[i])

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

    # MaxDD from equity curve
    eq = np.asarray(equity_curve, dtype=float)
    if len(eq) >= 2:
        peak = np.maximum.accumulate(eq)
        dd = (eq / (peak + 1e-12)) - 1.0
        maxdd = float(dd.min())
    else:
        maxdd = 0.0

    pfm = compute_pf_metrics(
        gp, gl,
        loss_floor=float(p.get("loss_floor", 0.001)),
        pf_diag_cap=float(p.get("pf_cap_score_only", 5.0)),
    )

    # For CSV only (never for scoring)
    pf_raw_csv = safe_pf_raw_for_csv(pfm["profit_factor_raw"], inf_placeholder=1e9)

    # GL per trade (return units)
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

# =================================================================================
# SCORING & UTILS
# =================================================================================

def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))

def score_trial(st: TradeStats, args) -> float:
    if st.trades < args.min_trades:
        return 0.0
    pf_w = 1.0 / (1.0 + math.exp(-args.pf_k * (st.profit_factor_diag - args.pf_baseline)))
    tr_w = sigmoid(args.trades_k * (st.trades - args.trades_baseline))
    return float((args.weight_pf * pf_w + (1.0 - args.weight_pf) * tr_w) ** args.score_power)

# =================================================================================
# Robust reporting helpers
# =================================================================================

def safe_series(x: pd.Series) -> pd.Series:
    return pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

def safe_mean(x: pd.Series) -> float:
    s = safe_series(x)
    return float(s.mean()) if len(s) else 0.0

def safe_median(x: pd.Series) -> float:
    s = safe_series(x)
    return float(s.median()) if len(s) else 0.0

def trimmed_mean(x, trim_frac=0.05):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return float("nan")
    x = np.sort(x)
    k = int(len(x) * trim_frac)
    if 2 * k >= len(x):
        return float(np.mean(x))
    return float(np.mean(x[k:-k]))

def robust_objective_aggregate(stats, args, objective_mode: str) -> float:
    eligible = [st for st in stats if st.trades >= args.min_trades]
    if not eligible:
        return 0.0

    scores = np.asarray([score_trial(st, args) for st in eligible], dtype=float)
    pf_diag = np.asarray([st.profit_factor_diag for st in eligible], dtype=float)
    pf_eff = np.asarray([st.profit_factor_eff for st in eligible], dtype=float)
    zero_loss = np.asarray([st.zero_loss for st in eligible], dtype=int)

    med_score = float(np.median(scores)) if len(scores) else 0.0
    mean_score = float(np.mean(scores)) if len(scores) else 0.0
    med_pf_diag = float(np.median(pf_diag)) if len(pf_diag) else 0.0

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

def objective_penalty_multiplier(stats, args) -> dict:
    """
    Returns a dict with:
      penalty_mult (overall),
      zero_loss_pct, cap_pct,
      zero_loss_mult, cap_mult,
      glpt_med, glpt_target, glpt_mult,
      vol_med,
      eligible_count, total_count

    Semantics (ONE-SIDED penalties):
      - zero_loss_mult: 1.0 when zero_loss_pct <= target, drops when above
      - cap_mult:       1.0 when cap_pct <= target, drops when above
      - glpt_mult:      1.0 when glpt_med >= target, drops when below (hard floor option)
    """
    import numpy as np

    eligible = [st for st in stats if getattr(st, "trades", 0) >= getattr(args, "min_trades", 0)]
    total_count = len(stats)

    if not eligible:
        return {
            "penalty_mult": 0.0,
            "zero_loss_pct": 1.0,
            "cap_pct": 1.0,
            "zero_loss_mult": 0.0,
            "cap_mult": 0.0,
            "glpt_med": 0.0,
            "glpt_target": float(getattr(args, "min_glpt", 0.0)),
            "glpt_mult": 0.0,
            "vol_med": 0.0,
            "eligible_count": 0,
            "total_count": int(total_count),
        }

    # ------------------------------------------------------------
    # Zero-loss fraction (computed directly from gp/gl)
    # zero-loss means gl==0 and gp>0 (PF_raw would blow up)
    # ------------------------------------------------------------
    z = np.asarray(
        [1.0 if (float(getattr(st, "gl", 0.0)) <= 0.0 and float(getattr(st, "gp", 0.0)) > 0.0) else 0.0 for st in eligible],
        dtype=float,
    )
    zero_loss_pct = float(np.mean(z)) if z.size else 0.0

    # ------------------------------------------------------------
    # Cap fraction (computed from PF_eff > pf_cap)
    # NOTE: args.pf_cap == 0/<=0 means "no cap" => cap_pct = 0
    # ------------------------------------------------------------
    cap_val = float(getattr(args, "pf_cap", 0.0))
    if cap_val > 0.0:
        c = np.asarray(
            [1.0 if float(getattr(st, "profit_factor_eff", 0.0)) > cap_val else 0.0 for st in eligible],
            dtype=float,
        )
    else:
        c = np.zeros(len(eligible), dtype=float)
    cap_pct = float(np.mean(c)) if c.size else 0.0

    # ------------------------------------------------------------
    # GL per trade (GLPT): fixed absolute target (return units)
    # ------------------------------------------------------------
    glpt = np.asarray(
        [float(getattr(st, "gl_per_trade", float(getattr(st, "gl", 0.0)) / max(int(getattr(st, "trades", 0)), 1)))
         for st in eligible],
        dtype=float,
    )
    glpt = glpt[np.isfinite(glpt)]
    glpt_med = float(np.median(glpt)) if glpt.size else 0.0

    # (Optional) volatility proxy for reporting only
    vol = np.asarray([float(getattr(st, "atr_pct_med", 0.0)) for st in eligible], dtype=float)
    vol = vol[np.isfinite(vol)]
    vol_med = float(np.median(vol)) if vol.size else 0.0

    glpt_target = float(getattr(args, "min_glpt", 0.0))
    min_glpt_k = float(getattr(args, "min_glpt_k", 0.0))

    # GLPT multiplier: HARD FLOOR (Option A)
    # - If glpt_med >= target => multiplier = 1
    # - If glpt_med <  target => multiplier drops via sigmoid
    if glpt_target > 0.0 and min_glpt_k > 0.0:
        x = min_glpt_k * (glpt_med - glpt_target)
        glpt_mult = 1.0 if x >= 0.0 else sigmoid(float(x))
    else:
        glpt_mult = 1.0  # disabled

    # ------------------------------------------------------------
    # ONE-SIDED multipliers for zero-loss and cap (FIXED SIGN)
    #   - No penalty if actual <= target
    #   - Penalty only if actual > target
    # ------------------------------------------------------------
    zero_loss_target = float(getattr(args, "zero_loss_target", 0.0))
    zero_loss_k = float(getattr(args, "zero_loss_k", 0.0))
    if zero_loss_k > 0.0:
        if zero_loss_pct <= zero_loss_target:
            zero_loss_mult = 1.0
        else:
            xzl = zero_loss_k * (zero_loss_target - zero_loss_pct)  # negative when violating
            zero_loss_mult = sigmoid(float(xzl))
    else:
        zero_loss_mult = 1.0

    cap_target = float(getattr(args, "cap_target", 0.0))
    cap_k = float(getattr(args, "cap_k", 0.0))
    if cap_k > 0.0:
        if cap_pct <= cap_target:
            cap_mult = 1.0
        else:
            xcap = cap_k * (cap_target - cap_pct)  # negative when violating
            cap_mult = sigmoid(float(xcap))
    else:
        cap_mult = 1.0

    # ------------------------------------------------------------
    # Combine based on obj_penalty_mode
    # ------------------------------------------------------------
    mode = str(getattr(args, "obj_penalty_mode", "both")).lower()
    if mode == "none":
        penalty_mult = 1.0
    elif mode == "zero_loss":
        penalty_mult = float(zero_loss_mult)
    elif mode == "cap":
        penalty_mult = float(cap_mult)
    else:  # "both"
        penalty_mult = float(zero_loss_mult * cap_mult)

    # Include GLPT guardrail only when enabled
    if glpt_target > 0.0 and min_glpt_k > 0.0:
        penalty_mult *= float(glpt_mult)

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
        "eligible_count": int(len(eligible)),
        "total_count": int(total_count),
    }


# =================================================================================
# MAIN
# =================================================================================

def main():
    import argparse, random, datetime
    from pathlib import Path
    import numpy as np
    import pandas as pd
    import optuna

    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--files", type=int, default=50)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fill", type=str, default="next_open", choices=["next_open", "same_close"])
    ap.add_argument("--use-regime", action="store_true")

    # Objective robustness
    ap.add_argument(
        "--objective-mode",
        type=str,
        default="median_score",
        choices=["mean_score", "median_score", "median_pf_diag",
                 "mean_pf_eff_excl_zero", "hybrid"],
    )

    # Objective degeneracy penalties
    ap.add_argument("--obj-penalty-mode", type=str, default="both",
                    choices=["none", "zero_loss", "cap", "both"])
    ap.add_argument("--zero-loss-target", type=float, default=0.05)
    ap.add_argument("--zero-loss-k", type=float, default=12.0)
    ap.add_argument("--cap-target", type=float, default=0.30)
    ap.add_argument("--cap-k", type=float, default=6.0)

    ap.add_argument("--min-glpt", type=float, default=0.002)
    ap.add_argument("--min-glpt-k", type=float, default=12.0)

    # Penalty knobs
    ap.add_argument("--penalty", type=str, default="enabled")
    ap.add_argument("--penalty_ret_center", type=float, default=0.01)
    ap.add_argument("--penalty_ret_k", type=float, default=10.0)
    ap.add_argument("--ret_floor", type=float, default=-0.15)
    ap.add_argument("--ret_floor_k", type=float, default=2.0)
    ap.add_argument("--max_trades", type=int, default=100)
    ap.add_argument("--max_trades_k", type=float, default=0.1)
    ap.add_argument("--pf_floor", type=float, default=1.0)
    ap.add_argument("--pf_floor_k", type=float, default=5.0)

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

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No parquet files found in: {args.data_dir}")

    sample_files = random.sample(files, min(len(files), args.files))
    data_list = [(f.stem, ensure_ohlc(pd.read_parquet(f))) for f in sample_files]

    # ======================
    # OPTUNA OBJECTIVE
    # ======================
    def objective(trial):
        p = {
            "vidya_len": trial.suggest_int("vl", 5, 25) if args.opt_vidya else 14,
            "vidya_smooth": trial.suggest_int("vs", 5, 40) if args.opt_vidya else 14,
            "fastPeriod": trial.suggest_int("fp", 5, 20) if args.opt_fastslow else 10,
            "slowPeriod": trial.suggest_int("sp", 21, 60) if args.opt_fastslow else 40,
            "time_stop_bars": trial.suggest_int("ts", 5, 40) if args.opt_time_stop else 15,
            "regime_ratio": trial.suggest_float("reg_ratio", 2.0, 5.0),
            "threshold": args.threshold_fixed,
            "vol_floor_mult": args.vol_floor_mult_fixed,
            "commission": args.commission_rate_per_side,
            "fill_mode": args.fill,
            "use_regime": args.use_regime,
            "loss_floor": args.loss_floor,
            "pf_cap_score_only": args.pf_cap,
            "cooldown_bars": 1,
        }

        stats = [backtest_vidya_engine(df, **p) for _, df in data_list]
        eligible = sum(st.trades >= args.min_trades for st in stats)
        cov = eligible / len(stats)
        cov_mult = sigmoid(args.coverage_k * (cov - args.coverage_target))

        agg = robust_objective_aggregate(stats, args, args.objective_mode)
        pen = objective_penalty_multiplier(stats, args)

        return float(agg * cov_mult * pen["penalty_mult"])

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=args.trials)

    best = dict(study.best_params)
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
    }

    # ======================
    # PER-TICKER REPORT
    # ======================
    rows = []
    for name, df in data_list:
        st = backtest_vidya_engine(df, **best_p)
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

    # ======================
    # DIAGNOSTICS
    # ======================
    best_stats = [backtest_vidya_engine(df, **best_p) for _, df in data_list]
    pen_best = objective_penalty_multiplier(best_stats, args)

    eligible_count = int(per_df["eligible"].sum())
    coverage = eligible_count / len(per_df)
    zero_loss_count = int(per_df["zero_loss"].sum())
    capped_count = int(per_df["pf_capped"].sum())

    med_pf_diag = safe_median(per_df["profit_factor_diag"])
    avg_pf_diag = safe_mean(per_df["profit_factor_diag"])
    med_pf_eff = safe_median(per_df["profit_factor_eff"])
    avg_pf_eff = safe_mean(per_df["profit_factor_eff"])

    # ======================
    # OUTPUT FILES
    # ======================
    out_dir = Path("output")
    out_dir.mkdir(exist_ok=True)
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    per_csv = out_dir / f"per_ticker_{run_ts}.csv"
    per_df.to_csv(per_csv, index=False)

    txt = []
    txt.append("=== OBJECTIVE DEGENERACY DIAGNOSTICS ===")
    txt.append(f"penalty_mult: {pen_best['penalty_mult']:.6f}")
    txt.append(f"zero_loss_mult: {pen_best['zero_loss_mult']:.6f}")
    txt.append(f"cap_mult: {pen_best['cap_mult']:.6f}")
    txt.append(f"glpt_mult: {pen_best['glpt_mult']:.6f}")
    txt.append("")
    txt.append("=== BEST PARAMS ===")
    for k in sorted(study.best_params):
        txt.append(f"{k}: {study.best_params[k]}")
    txt.append("")
    txt.append(f"objective_score: {study.best_value:.6f}")

    best_txt = out_dir / f"best_params_{run_ts}.txt"
    best_txt.write_text("\n".join(txt))

    print(f"Saved per-ticker CSV: {per_csv}")
    print(f"Saved best params TXT: {best_txt}")
    print("\n".join(txt))

if __name__ == "__main__":
    main()
