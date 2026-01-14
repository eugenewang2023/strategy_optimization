#!/usr/bin/env python3
"""
Bayes_opt_dynamic_SR.py  (PARQUET-ONLY)  ✅ Pine-match for Dynamic_SR_hrr (HA-envelope SR + REAL exits)

Change in this version (per your request):
- Add CLI input option --min-tp2sl to FIX the asymmetry constraint:
      slMultiplier > min_tp2sl * tpMultiplier
- Remove optimization of min_tp2sl (no --opt-min-tp2sl, no trial.suggest for it)

Bonus (kept, but fixed):
- Optional adaptive constraint mode --tp2sl-auto that scales min_tp2sl with srBandMultiplier
  (if enabled, --min-tp2sl is ignored)

Everything else stays the same.
"""

import math
import random
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional, Dict

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


def backtest_dynamic_sr(
    df: pd.DataFrame,
    atrPeriod: int,
    slMult: float,
    tpMult: float,
    srBandMult: float,
    commission_per_side: float,
    fill_mode: str,
    use_trailing_exit: bool = True,
    trail_mode: str = "trail_only",          # "trail_only" or "trail_plus_hard_sl"
    close_on_sellSignal: bool = True,
    cooldown_bars: int = 0,
    time_stop_bars: int = 0,
) -> TradeStats:
    """
    Pine-matching core:
    - SR: HA-anchored envelope with REAL ATR band
    - Entries: buySignal cross of HA close vs SR
    - Exits: REAL dynamic stop/limit + optional trailing (TV trail_offset semantics)
    """
    o = df["open"].to_numpy(dtype=float)
    h = df["high"].to_numpy(dtype=float)
    l = df["low"].to_numpy(dtype=float)
    c = df["close"].to_numpy(dtype=float)
    n = len(df)

    if n < max(atrPeriod + 2, 60):
        return TradeStats()

    # HA from REAL
    _, ha_h, ha_l, ha_c = heikin_ashi_from_real(o, h, l, c)

    # REAL ATR for SR band + exits (matches Pine riskATR = realATR(atrPeriod))
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

    # SR: HA-anchored envelope with REAL ATR band
    sr = np.full(n, np.nan, dtype=float)
    sr[start] = ha_c[start]

    for i in range(start + 1, n):
        if np.isnan(real_atr[i]):
            sr[i] = sr[i - 1]
            continue

        band = real_atr[i] * srBandMult
        prev = sr[i - 1]

        if ha_c[i] > prev:
            sr[i] = max(prev, ha_l[i] + band)
        elif ha_c[i] < prev:
            sr[i] = min(prev, ha_h[i] - band)
        else:
            sr[i] = prev

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

    # trailing state (TV-like)
    trail_active = False
    trail_stop = np.nan
    trail_high_since = np.nan
    trail_dist = np.nan

    def apply_commission(eq: float) -> float:
        return eq * (1.0 - commission_per_side)

    for i in range(start + 1, n):
        if np.isnan(sr[i - 1]) or np.isnan(sr[i]) or np.isnan(real_atr[i]):
            continue

        if cooldown_left > 0:
            cooldown_left -= 1

        buy_sig = crossover(ha_c[i - 1], ha_c[i], sr[i - 1], sr[i])
        sell_sig = crossunder(ha_c[i - 1], ha_c[i], sr[i - 1], sr[i])

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

            # Trailing activation/update (TV semantics)
            if use_trailing_exit:
                if (not trail_active) and (h[i] >= entry + trail_dist):
                    trail_active = True
                    trail_high_since = float(h[i])
                    trail_stop = trail_high_since - trail_dist
                elif trail_active:
                    trail_high_since = float(h[i]) if np.isnan(trail_high_since) else max(trail_high_since, float(h[i]))
                    trail_stop = trail_high_since - trail_dist

            # Intrabar triggers (conservative for long):
            # 1) hard stop (if enabled)
            # 2) trailing stop (if active)
            # 3) limit target
            # 4) time stop (fill-based)
            # 5) sell signal close (fill-based)

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
            cooldown_left = max(0, int(cooldown_bars))

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
) -> Tuple[float, Dict[str, Any], List[Dict[str, Any]], float, float]:
    per: List[Dict[str, Any]] = []
    scores: List[float] = []
    num_neg = 0

    for p in file_paths:
        df = pd.read_parquet(p)
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
            cooldown_bars=cooldown_bars,
            time_stop_bars=time_stop_bars,
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

    # --- TP/SL asymmetry constraint ---
    # Fixed constraint (manual)
    ap.add_argument("--min-tp2sl", type=float, default=1.30,
                    help="Require slMultiplier > min_tp2sl * tpMultiplier.")

    # Optional auto scaling mode (if enabled, ignores --min-tp2sl)
    ap.add_argument("--tp2sl-auto", action="store_true",
                    help="If set, enforce an adaptive TP/SL asymmetry constraint based on srBandMultiplier.")
    ap.add_argument("--tp2sl-base", type=float, default=1.25,
                    help="Base min_tp2sl at srBandMultiplier = tp2sl-sr0.")
    ap.add_argument("--tp2sl-sr0", type=float, default=0.65,
                    help="Reference srBandMultiplier for tp2sl-base.")
    ap.add_argument("--tp2sl-k", type=float, default=0.60,
                    help="Slope: how much min_tp2sl increases per +1.0 srBandMultiplier.")
    ap.add_argument("--tp2sl-min", type=float, default=1.10,
                    help="Clamp lower bound for adaptive min_tp2sl.")
    ap.add_argument("--tp2sl-max", type=float, default=2.20,
                    help="Clamp upper bound for adaptive min_tp2sl.")

    # ===== Reporting helpers =====
    ap.add_argument("--report-only", action="store_true",
                    help="Skip Optuna and only report using fixed params.")
    ap.add_argument("--report-both-fills", action="store_true",
                    help="When used with --report-only, run both same_close and next_open.")
    ap.add_argument("--atrPeriod-fixed", type=int, default=None)
    ap.add_argument("--slMultiplier-fixed", type=float, default=None)
    ap.add_argument("--tpMultiplier-fixed", type=float, default=None)
    ap.add_argument("--srBandMultiplier-fixed", type=float, default=None)

    return ap.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    # =========================
    # REPORT ONLY MODE
    # =========================
    if args.report_only:
        required = [
            ("--atrPeriod-fixed", args.atrPeriod_fixed),
            ("--slMultiplier-fixed", args.slMultiplier_fixed),
            ("--tpMultiplier-fixed", args.tpMultiplier_fixed),
            ("--srBandMultiplier-fixed", args.srBandMultiplier_fixed),
        ]
        missing = [name for name, val in required if val is None]
        if missing:
            raise SystemExit("Missing required flags: " + ", ".join(missing))

        fills = ["same_close", "next_open"] if args.report_both_fills else [args.fill]

        data_dir = Path(args.data_dir)
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        all_files = sorted(data_dir.glob("*.parquet"))
        if not all_files:
            raise SystemExit(f"No .parquet files found in {data_dir.resolve()}")

        file_paths = random.sample(all_files, args.files) if (len(all_files) > args.files) else all_files

        for fm in fills:
            best_score_single, overall, per, pf_avg, trades_avg = evaluate_params_on_files(
                file_paths,
                atrPeriod=int(args.atrPeriod_fixed),
                slMultiplier=float(args.slMultiplier_fixed),
                tpMultiplier=float(args.tpMultiplier_fixed),
                srBandMultiplier=float(args.srBandMultiplier_fixed),
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
            )

            per_df = pd.DataFrame(per).sort_values(
                ["ticker_score", "total_return", "profit_factor", "num_trades"],
                ascending=False
            )

            per_csv = out_dir / f"per_ticker_summary_{fm}.csv"
            per_df.to_csv(per_csv, index=False)

            print(f"\n=== REPORT ONLY ({fm}) ===")
            print(f"Saved: {per_csv}")
            print(f"Avg PF: {overall['avg_pf_raw']}")
            print(f"Avg Trades: {overall['avg_trades']}")
            print(f"Mean Score: {overall['mean_ticker_score']}")
            print(f"num_neg: {overall['num_neg']}")
            print(f"Score: {best_score_single}")

        print("\n=== FIXED PARAMS USED ===")
        print(f"atrPeriod: {args.atrPeriod_fixed}")
        print(f"slMultiplier: {args.slMultiplier_fixed}")
        print(f"tpMultiplier: {args.tpMultiplier_fixed}")
        print(f"srBandMultiplier: {args.srBandMultiplier_fixed}")
        return

    # Validate constraint knobs
    if args.tp2sl_auto:
        if args.tp2sl_min <= 0 or args.tp2sl_max <= 0:
            raise SystemExit("--tp2sl-min and --tp2sl-max must be > 0")
        if args.tp2sl_min > args.tp2sl_max:
            raise SystemExit("--tp2sl-min must be <= --tp2sl-max")
    else:
        if args.min_tp2sl <= 0:
            raise SystemExit("--min-tp2sl must be > 0")

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_files = sorted(data_dir.glob("*.parquet"))
    if not all_files:
        raise SystemExit(f"No .parquet files found in {data_dir.resolve()}")

    file_paths = random.sample(all_files, args.files) if (len(all_files) > args.files) else all_files

    ROBUST_FILLS = ["same_close", "next_open"]

    def objective(trial: optuna.Trial) -> float:
        atrPeriod = trial.suggest_int("atrPeriod", 50, 75)
        slMultiplier = trial.suggest_float("slMultiplier", 6.0, 12.0)
        tpMultiplier = trial.suggest_float("tpMultiplier", 4.5, 7.0)
        srBandMultiplier = trial.suggest_float("srBandMultiplier", 0.44, 0.66)

        # --- TP/SL asymmetry constraint (FIXED or AUTO) ---
        if args.tp2sl_auto:
            min_tp2sl_eff = args.tp2sl_base + args.tp2sl_k * (srBandMultiplier - args.tp2sl_sr0)
            min_tp2sl_eff = max(args.tp2sl_min, min(args.tp2sl_max, min_tp2sl_eff))
        else:
            min_tp2sl_eff = args.min_tp2sl

        if slMultiplier <= (min_tp2sl_eff * tpMultiplier):
            raise optuna.TrialPruned()

        # optional prune for extreme combos
        if tpMultiplier < 4.5 and srBandMultiplier > 0.95:
            raise optuna.TrialPruned()

        cooldown_bars = trial.suggest_int("cooldown", 0, 7) if args.opt_cooldown else int(args.cooldown)
        time_stop_bars = trial.suggest_int("time_stop", 0, 12) if args.opt_time_stop else int(args.time_stop)

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
            )
            scores.append(s)

        return float(np.mean(scores)) if scores else -1e9

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials, show_progress_bar=True)

    best = study.best_trial
    best_params = dict(best.params)

    # normalize types
    best_params["atrPeriod"] = int(best_params["atrPeriod"])
    best_params["slMultiplier"] = float(best_params["slMultiplier"])
    best_params["tpMultiplier"] = float(best_params["tpMultiplier"])
    best_params["srBandMultiplier"] = float(best_params["srBandMultiplier"])

    best_cooldown = int(best_params.get("cooldown", 0)) if args.opt_cooldown else int(args.cooldown)
    best_time_stop = int(best_params.get("time_stop", 0)) if args.opt_time_stop else int(args.time_stop)

    # compute effective constraint at best params
    if args.tp2sl_auto:
        best_min_tp2sl_eff = args.tp2sl_base + args.tp2sl_k * (best_params["srBandMultiplier"] - args.tp2sl_sr0)
        best_min_tp2sl_eff = max(args.tp2sl_min, min(args.tp2sl_max, best_min_tp2sl_eff))
        constraint_mode = "auto"
    else:
        best_min_tp2sl_eff = args.min_tp2sl
        constraint_mode = "fixed"

    best_score_single, overall, per, _, _ = evaluate_params_on_files(
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
    )

    per_df = pd.DataFrame(per).sort_values(
        ["ticker_score", "total_return", "profit_factor", "num_trades"],
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
    print(f"Tail-protection ret floor/k:  {args.ret_floor} / {args.ret_floor_k}")
    print(f"Max-trades penalty:           max_trades={args.max_trades}, k={args.max_trades_k}")
    print(f"PF-floor penalty:             pf_floor={args.pf_floor}, k={args.pf_floor_k}")
    print(f"cooldown_bars:                {best_cooldown} (opt={args.opt_cooldown})  [loss-only]")
    print(f"time_stop_bars:               {best_time_stop} (0=disabled)  (opt={args.opt_time_stop})  [not-positive-now]")
    print(f"num_neg (return<=0):          {overall['num_neg']}")
    print(f"commission_rate_per_side:     {args.commission_rate_per_side:.6f}")
    print(f"PF ROC baseline/k:            {args.pf_baseline:.3f} / {args.pf_k:.3f}")
    print(f"Trades ROC baseline/k:        {args.trades_baseline:.3f} / {args.trades_k:.3f}")
    print(f"weight_pf:                    {args.weight_pf:.3f}")
    print(f"score_power:                  {args.score_power:.3f}")
    print(f"min_trades gate:              {args.min_trades}")
    print(f"loss_floor (scoring):         {args.loss_floor}")
    print(f"min_tp2sl_eff (constraint):   {best_min_tp2sl_eff:.4f} ({constraint_mode})")
    print(f"Score (single fill '{args.fill}'): {best_score_single:.6f}")

    print("\n=== BEST PARAMS ===")
    print(f"atrPeriod: {best_params['atrPeriod']}")
    print(f"slMultiplier: {best_params['slMultiplier']}")
    print(f"tpMultiplier: {best_params['tpMultiplier']}")
    print(f"srBandMultiplier: {best_params['srBandMultiplier']}")
    print(f"best_score (ROBUST avg fills): {best.value}")
    print(f"pf_cap_score_only: {args.pf_cap_score_only}")
    if args.opt_cooldown:
        print(f"cooldown (best): {best_cooldown}")
    if args.opt_time_stop:
        print(f"time_stop (best): {best_time_stop}")
    print(f"fill_mode (report): {args.fill}")
    print(f"Saved per-ticker CSV to: {per_csv}")

    print("\n=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===")
    print(per_df[["ticker", "profit_factor", "num_trades", "ticker_score", "total_return",
                 "gross_profit", "gross_loss", "maxdd"]].to_string(index=False))

    print(
        f"Done: Bayes_opt_dynamic_SR.py "
        f"--trials {args.trials} "
        f"--files {args.files} "
        f"{'--penalty ' if args.penalty else ''}"
        f"--penalty-ret-center {args.penalty_ret_center} "
        f"--penalty-ret-k {args.penalty_ret_k} "
        f"--min-trades {args.min_trades} "
        f"--trades-baseline {args.trades_baseline} "
        f"--trades-k {args.trades_k} "
        f"--max-trades {args.max_trades} "
        f"--max-trades-k {args.max_trades_k} "
        f"--ret-floor {args.ret_floor} "
        f"--ret-floor-k {args.ret_floor_k} "
        f"--pf-cap {args.pf_cap_score_only} "
        f"--pf-baseline {args.pf_baseline} "
        f"--pf-k {args.pf_k} "
        f"--pf-floor {args.pf_floor} "
        f"--pf-floor-k {args.pf_floor_k} "
        f"--weight-pf {args.weight_pf} "
        f"--score-power {args.score_power} "
        f"--commission_rate_per_side {args.commission_rate_per_side} "
        f"--loss_floor {args.loss_floor} "
        f"--fill {args.fill} "
        f"--cooldown {best_cooldown} "
        f"--time-stop {best_time_stop} "
        f"{'--tp2sl-auto ' if args.tp2sl_auto else ''}"
        f"{'--opt-cooldown ' if args.opt_cooldown else ''}"
        f"{'--opt-time-stop' if args.opt_time_stop else ''}"
    )


if __name__ == "__main__":
    main()
