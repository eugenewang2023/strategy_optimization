#!/usr/bin/env python3
"""
Vidya_RSI.py - High Density Aggression Edition
With Ticker-by-Ticker Reporting and Optuna Progress Bar

Fixes:
- Proper PF diag uncapped + capped (stored and reportable)
- Fixed SL/TP constraint (was always failing with defaults)
- Safer ATR computation for short series
- Robust OHLC validation
- Removed manual stdout status spam (conflicted with tqdm)
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

# tqdm progress bar callback (gives "Best trial: ... Best value: ..." line)
try:
    from optuna.integration import TQDMProgressBarCallback
except Exception:
    TQDMProgressBarCallback = None


# =================================================================================
# INDICATORS & ENGINE
# =================================================================================

def ensure_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    cols = {c.lower(): c for c in df.columns}
    required = ["open", "high", "low", "close"]
    missing = [r for r in required if r not in cols]
    if missing:
        raise ValueError(f"Missing OHLC columns: {missing}. Found: {list(df.columns)}")
    out = df[[cols["open"], cols["high"], cols["low"], cols["close"]]].copy()
    out.columns = ["open", "high", "low", "close"]
    return out


def heikin_ashi_from_real(o: np.ndarray, h: np.ndarray, l: np.ndarray, c: np.ndarray):
    ha_c = (o + h + l + c) / 4.0
    ha_o = np.empty_like(ha_c)
    ha_o[0] = (o[0] + c[0]) / 2.0
    for i in range(1, len(ha_c)):
        ha_o[i] = (ha_o[i - 1] + ha_c[i - 1]) / 2.0
    ha_h = np.maximum.reduce([h, ha_o, ha_c])
    ha_l = np.minimum.reduce([l, ha_o, ha_c])
    return ha_o, ha_h, ha_l, ha_c


def atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, period: int) -> np.ndarray:
    n = len(c)
    atr = np.full(n, np.nan, dtype=float)
    period = int(period)
    if n < 2 or period <= 0 or n < period:
        return atr

    tr = np.empty(n, dtype=float)
    tr[0] = h[0] - l[0]
    prev_close = c[0]
    for i in range(1, n):
        tr[i] = max(h[i] - l[i], abs(h[i] - prev_close), abs(l[i] - prev_close))
        prev_close = c[i]

    atr[period - 1] = np.mean(tr[:period])
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


def vidya_ema(price: np.ndarray, length: int, smoothing: int) -> np.ndarray:
    """
    VIDYA using Kaufman's Efficiency Ratio (ER) as the Volatility Index.
    - length: The lookback for calculating the efficiency (Signal/Noise).
    - smoothing: The base EMA smoothing (e.g., 14).
    """
    n = len(price)
    vidya = np.full(n, np.nan, dtype=float)
    L = max(2, int(length))
    S = max(1, int(smoothing))
    
    if n < L:
        return vidya

    # Base Alpha: 2 / (smoothing + 1)
    alpha_base = 2.0 / (S + 1.0)

    # 1. Calculate Signal: Net change over the period
    # |Price_now - Price_L_ago|
    signal = np.abs(pd.Series(price).diff(L).to_numpy())

    # 2. Calculate Noise: Sum of absolute bar-to-bar changes
    abs_diff = np.abs(np.diff(price, prepend=price[0]))
    noise = pd.Series(abs_diff).rolling(L).sum().to_numpy()

    # 3. Efficiency Ratio (ER)
    # Avoid division by zero with 1e-12
    vi = signal / (noise + 1e-12)

    # 4. Compute VIDYA
    # Seed first available value
    start_idx = L
    vidya[start_idx - 1] = float(price[start_idx - 1])
    
    for i in range(start_idx, n):
        if not np.isfinite(vi[i]) or not np.isfinite(vidya[i - 1]):
            vidya[i] = vidya[i-1] # Carry forward or use price
            continue
            
        # alpha_eff = alpha_base * Efficiency_Ratio
        k = alpha_base * vi[i]
        vidya[i] = price[i] * k + vidya[i - 1] * (1.0 - k)
    return vidya

@dataclass
class TradeStats:
    gp: float = 0.0
    gl: float = 0.0
    trades: int = 0
    tot_ret: float = 0.0
    pf_diag: float = 0.0  # capped
    profit_factor_diag_uncapped: float = 0.0  # uncapped


def backtest_vidya_engine(df: pd.DataFrame, **p) -> TradeStats:
    df = ensure_ohlc(df)
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(c)

    if n < 200:
        return TradeStats()

    # HA signals
    _, _, _, ha_c = heikin_ashi_from_real(o, h, l, c)

    # ATR (risk + vol floor)
    atrPeriod = int(p.get("atrPeriod", 25))
    atr = atr_wilder(h, l, c, atrPeriod)

    # VIDYA on HA close
    v_main = vidya_ema(ha_c, p["vidya_len"], p["vidya_smooth"])

    # EMA(fast/slow) of VIDYA
    fast = pd.Series(v_main).ewm(span=int(p["fastPeriod"]), adjust=False).mean().to_numpy()
    slow = pd.Series(v_main).ewm(span=int(p["slowPeriod"]), adjust=False).mean().to_numpy()

    # Hot normalized by ATR, then smoothed
    raw_hot = (fast - slow) / (atr + 1e-12)
    hot_sm = pd.Series(raw_hot).rolling(int(p.get("smooth_len", 5)), min_periods=int(p.get("smooth_len", 5))).mean().to_numpy()

    # Volatility floor
    vlen = int(p["vol_floor_len"])
    atr_ma = pd.Series(atr).rolling(vlen, min_periods=vlen).mean().to_numpy()
    vol_ok = atr > (atr_ma * float(p["vol_floor_mult"]))

    commission = float(p["commission_per_side"])
    slMult = float(p["slMult"])
    tpMult = float(p["tpMult"])
    cooldown_bars = int(p["cooldown_bars"])
    time_stop_bars = int(p["time_stop_bars"])
    threshold = float(p["threshold"])
    loss_floor = float(p["loss_floor"])
    pf_cap = float(p["pf_cap_score_only"])
    fill_mode = str(p["fill_mode"])

    equity = 1.0
    gp = 0.0
    gl = 0.0
    trades = 0

    in_pos = False
    entry = 0.0
    bars_in_trade = 0
    cooldown_left = 0
    has_hit_be = False

    def get_fill(i: int) -> Optional[float]:
        if fill_mode == "next_open":
            return float(o[i + 1]) if (i + 1 < n) else None
        return float(c[i])

    loop_start = max(vlen + 5, atrPeriod + 2, 10)

    for i in range(loop_start, n - 2):
        if cooldown_left > 0:
            cooldown_left -= 1
            continue

        if not (np.isfinite(hot_sm[i - 1]) and np.isfinite(hot_sm[i]) and np.isfinite(atr[i]) and np.isfinite(vol_ok[i])):
            continue

        buy_sig = (hot_sm[i - 1] <= threshold) and (hot_sm[i] > threshold)
        sell_sig = (hot_sm[i - 1] >= -threshold) and (hot_sm[i] < -threshold)

        # Entry
        if (not in_pos) and buy_sig and bool(vol_ok[i]):
            fill = get_fill(i)
            if fill is None or fill <= 0:
                continue
            entry = float(fill)
            in_pos = True
            trades += 1
            equity *= (1.0 - commission)
            bars_in_trade = 0
            has_hit_be = False
            continue

        # Manage
        if in_pos:
            bars_in_trade += 1

            pnl_atr = (c[i] - entry) / (atr[i] + 1e-12)
            if pnl_atr > 1.5:
                has_hit_be = True

            stop = entry if has_hit_be else (entry - atr[i] * slMult)
            tgt = entry + atr[i] * tpMult

            exit_price: Optional[float] = None

            # intrabar conservative
            if l[i] <= stop:
                exit_price = float(stop)
            elif h[i] >= tgt:
                exit_price = float(tgt)
            elif time_stop_bars > 0 and bars_in_trade >= time_stop_bars:
                # loss-only time stop
                fill = get_fill(i)
                if fill is not None and fill <= entry:
                    exit_price = float(fill)
            elif sell_sig:
                fill = get_fill(i)
                if fill is not None:
                    exit_price = float(fill)

            # End-of-series
            if exit_price is None and i == n - 3:
                fill = get_fill(i)
                if fill is not None:
                    exit_price = float(fill)

            if exit_price is not None and exit_price > 0:
                equity *= (1.0 - commission)
                pnl = (exit_price - entry) / entry
                equity *= (1.0 + pnl)

                if pnl >= 0:
                    gp += pnl
                else:
                    gl += abs(pnl)
                    cooldown_left = cooldown_bars

                in_pos = False
                entry = 0.0
                bars_in_trade = 0
                has_hit_be = False

    # PF diag uncapped + capped
    eff_gl = max(gl, (loss_floor * gp) if gp > 0 else loss_floor)
    pf_uncapped = (gp / eff_gl) if eff_gl > 0 else 0.0
    pf_capped = min(pf_uncapped, pf_cap)

    return TradeStats(
        gp=float(gp),
        gl=float(gl),
        trades=int(trades),
        tot_ret=float(equity - 1.0),
        pf_diag=float(pf_capped),
        profit_factor_diag_uncapped=float(pf_uncapped),
    )


# =================================================================================
# SCORING
# =================================================================================

def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def score_trial(st: TradeStats, args) -> float:
    if st.trades < args.min_trades:
        return 0.0
    pf_w = sigmoid(args.pf_k * (st.pf_diag - args.pf_baseline))
    tr_w = sigmoid(args.trades_k * (st.trades - args.trades_baseline))
    s = (args.weight_pf * pf_w + (1.0 - args.weight_pf) * tr_w)
    if args.score_power != 1.0:
        s = s ** args.score_power
    return float(s)


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
    ap.add_argument("--fill", type=str, default="next_open", choices=["next_open", "same_close"])

    ap.add_argument("--min-trades", type=int, default=5)
    ap.add_argument("--min-tp2sl", type=float, default=1.1)

    ap.add_argument("--threshold-fixed", type=float, default=0.04)
    ap.add_argument("--vol-floor-len", type=int, default=50)
    ap.add_argument("--vol-floor-mult-fixed", type=float, default=0.55)

    ap.add_argument("--trades-baseline", type=float, default=10.0)
    ap.add_argument("--trades-k", type=float, default=0.4)
    ap.add_argument("--pf-baseline", type=float, default=1.02)
    ap.add_argument("--pf-k", type=float, default=1.2)
    ap.add_argument("--weight-pf", type=float, default=0.4)
    ap.add_argument("--score-power", type=float, default=1.1)

    ap.add_argument("--commission_rate_per_side", type=float, default=0.0006)
    ap.add_argument("--loss_floor", type=float, default=0.001)
    ap.add_argument("--pf-cap", type=float, default=5.0, dest="pf_cap_score_only")

    ap.add_argument("--coverage-target", type=float, default=0.85)
    ap.add_argument("--coverage-k", type=float, default=8.0)

    ap.add_argument("--opt-vidya", action="store_true")
    ap.add_argument("--opt-fastslow", action="store_true")
    ap.add_argument("--opt-time-stop", action="store_true")

    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    files = sorted(Path(args.data_dir).glob("*.parquet"))
    if not files:
        raise SystemExit(f"No .parquet files found in {Path(args.data_dir).resolve()}")

    selected_files = random.sample(files, min(len(files), args.files))

    # Pre-load data
    data_list = []
    for f in selected_files:
        try:
            df = pd.read_parquet(f)
            df = ensure_ohlc(df)
            data_list.append((f.stem, df))
        except Exception:
            continue

    if not data_list:
        raise SystemExit("No readable parquet files with OHLC columns.")

    # Fixed risk parameters (can later optimize if you want)
    ATR_PERIOD = 25
    SL_MULT = 3.0
    TP_MULT = 3.6  # IMPORTANT: satisfies default min_tp2sl=1.1 -> 3.6 >= 1.1*3.0
    COOLDOWN = 1
    SMOOTH_LEN = 5

    def objective(trial: optuna.Trial) -> float:
        p = {
            "atrPeriod": ATR_PERIOD,
            "slMult": SL_MULT,
            "tpMult": TP_MULT,
            "cooldown_bars": COOLDOWN,

            "vidya_len": trial.suggest_int("vl", 2, 20) if args.opt_vidya else 14,
            "vidya_smooth": trial.suggest_int("vs", 5, 40) if args.opt_vidya else 14,

            "fastPeriod": trial.suggest_int("fp", 2, 20) if args.opt_fastslow else 10,
            "slowPeriod": trial.suggest_int("sp", 21, 100) if args.opt_fastslow else 40,

            "time_stop_bars": trial.suggest_int("ts", 5, 50) if args.opt_time_stop else 15,

            "smooth_len": SMOOTH_LEN,

            "commission_per_side": float(args.commission_rate_per_side),
            "fill_mode": str(args.fill),

            "threshold": float(args.threshold_fixed),
            "vol_floor_len": int(args.vol_floor_len),
            "vol_floor_mult": float(args.vol_floor_mult_fixed),

            "loss_floor": float(args.loss_floor),
            "pf_cap_score_only": float(args.pf_cap_score_only),
        }

        # Constraint: TP must be sufficiently larger than SL
        if p["tpMult"] < (args.min_tp2sl * p["slMult"]):
            return 0.0

        scores = []
        eligible = 0
        for _, df in data_list:
            st = backtest_vidya_engine(df, **p)
            sc = score_trial(st, args)
            scores.append(sc)
            if st.trades >= args.min_trades:
                eligible += 1

        coverage_ratio = eligible / max(1, len(data_list))
        cov_mult = sigmoid(float(args.coverage_k) * (coverage_ratio - float(args.coverage_target)))
        final = float(np.mean(scores)) * float(cov_mult)
        return final

    study = optuna.create_study(direction="maximize")

    callbacks = []
    if TQDMProgressBarCallback is not None:
        callbacks.append(TQDMProgressBarCallback())

    study.optimize(
        objective,
        n_trials=args.trials,
        show_progress_bar=True,
        callbacks=callbacks if callbacks else None,
    )

    # Output directory + timestamped log
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    log_file = output_dir / f"opt_results_{timestamp}.csv"

    print("\n" + "=" * 60)
    print(f"BEST SCORE: {study.best_value}")
    print(f"BEST PARAMS: {study.best_params}")
    print("=" * 60)

    # Final report using best params
    best_p = {
        "atrPeriod": ATR_PERIOD,
        "slMult": SL_MULT,
        "tpMult": TP_MULT,
        "cooldown_bars": COOLDOWN,

        "vidya_len": int(study.best_params.get("vl", 14)),
        "vidya_smooth": int(study.best_params.get("vs", 14)),
        "fastPeriod": int(study.best_params.get("fp", 10)),
        "slowPeriod": int(study.best_params.get("sp", 40)),
        "time_stop_bars": int(study.best_params.get("ts", 15)),
        "smooth_len": SMOOTH_LEN,

        "commission_per_side": float(args.commission_rate_per_side),
        "fill_mode": str(args.fill),

        "threshold": float(args.threshold_fixed),
        "vol_floor_len": int(args.vol_floor_len),
        "vol_floor_mult": float(args.vol_floor_mult_fixed),

        "loss_floor": float(args.loss_floor),
        "pf_cap_score_only": float(args.pf_cap_score_only),
    }

    print(f"{'Ticker':<15} | {'PF uncap':<7} | {'PF diag':<8} | {'Return':<9} |  {'Score'}  | {'Trades':<6}")
    print("-" * 78)

    # build report rows (no printing yet)
    report_data = []
    for name, df in data_list:
        st = backtest_vidya_engine(df, **best_p)
        sc = score_trial(st, args)
        report_data.append({
            "ticker": name,
            "pf_unc": float(st.profit_factor_diag_uncapped),
            "pf_diag": float(st.pf_diag),
            "return_pct": float(st.tot_ret * 100.0),
            "trades": int(st.trades),
            "score": float(sc),
        })

    # sort by: PF unc, PF diag, Return, Trades, Score (all descending)
    report_data_sorted = sorted(
        report_data,
        key=lambda r: (
            r["pf_unc"],
            r["pf_diag"],
            r["return_pct"],
            r["score"],
            r["trades"],
        ),
        reverse=True,
    )

    # print sorted
    for r in report_data_sorted:
        print(f"{r['ticker']:<15} | {r['pf_unc']:>8.2f} | {r['pf_diag']:>8.2f} | {r['return_pct']:>8.2f}% | {r['score']:.6f} | {r['trades']:<6d}")

    df_report = pd.DataFrame(report_data).sort_values(
        ["pf_unc", "pf_diag", "return_pct", "score", "trades"],
       ascending=[False, False, False, False, False],
    )
    df_report.to_csv(log_file, index=False)

    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n--- OPTIMIZATION METADATA ---\n")
        f.write(f"BEST_SCORE,{study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k},{v}\n")

    print(f"\n[DONE] Detailed results archived in: {log_file}")


if __name__ == "__main__":
    main()
