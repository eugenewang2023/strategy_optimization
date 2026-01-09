#!/usr/bin/env python3
"""
bayes_opt_half_RSI.py

Goal: Match TradingView Pine strategy:
"half_RSI_hh (HA signals + HA risk)"  (your pasted Pine)

UPDATE:
- Optimizes SIGNAL params: slow_window, shift, smooth_len
  (fast_window is always derived as round(slow_window/2) like Pine)
- Optimizes RISK params: atrPeriod, slMultiplier, tpMultiplier, trailMultiplier
- Objective (Option A): maximize OVERALL Profit Factor across selected files

CHANGE (per your request):
- Profit Factor is now computed using AVERAGE win / AVERAGE loss
  (avg_profit = mean PnL of winners, avg_loss = mean abs(PnL) of losers)

Everything else stays aligned to the Pine logic described in earlier versions.
"""

import os, json, math, time, random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional plotting for equity curves
try:
    import matplotlib.pyplot as plt
    HAS_PLT = True
except Exception:
    HAS_PLT = False

# ----------------- USER CONFIG -----------------
DATA_DIR = Path("data")
OUT_DIR  = Path("output")

INITIAL_CAPITAL = 100000.0
POSITION_SIZE_PCT = 0.10     # TradingView default_qty_value=10
COMMISSION_PCT = 0.0006      # 0.06% (Pine: commission_value=0.06 with commission_type=strategy.commission.percent)
# ------------------------------------------------


# =========================
# Helpers: SMA, Wilder RMA, ATR, RSI (TradingView-style)
# =========================

def sma(x: np.ndarray, length: int) -> np.ndarray:
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0:
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


def rma_wilder(x: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView ta.rma(): Wilder smoothing.
    Seed = SMA(length) at index length-1 (first full non-na window).
    alpha = 1/length.
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < length:
        return out

    if np.isnan(x[:length]).any():
        for i in range(length - 1, n):
            w = x[i - length + 1:i + 1]
            if not np.isnan(w).any():
                out[i] = float(np.mean(w))
                start = i + 1
                break
        else:
            return out
    else:
        out[length - 1] = float(np.mean(x[:length]))
        start = length

    alpha = 1.0 / float(length)
    for i in range(start, n):
        if np.isnan(x[i]) or np.isnan(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = out[i - 1] + alpha * (x[i] - out[i - 1])
    return out


def true_range(high: np.ndarray, low: np.ndarray, close: np.ndarray) -> np.ndarray:
    n = len(close)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        if np.isnan(high[i]) or np.isnan(low[i]) or np.isnan(close[i - 1]):
            tr[i] = np.nan
            continue
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


def atr_wilder(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """
    Match Pine ta.atr(length): RMA of true range.
    """
    tr = true_range(high, low, close)
    return rma_wilder(tr, length)


def rsi_tv(price: np.ndarray, length: int) -> np.ndarray:
    """
    Match TradingView ta.rsi():
    rsi = 100 - 100 / (1 + rma(gain, len) / rma(loss, len))
    """
    n = len(price)
    out = np.full(n, np.nan, dtype=np.float64)
    if length <= 0 or n < 2:
        return out

    ch = np.diff(price, prepend=np.nan)
    gain = np.where(ch > 0, ch, 0.0)
    loss = np.where(ch < 0, -ch, 0.0)

    avg_gain = rma_wilder(gain, length)
    avg_loss = rma_wilder(loss, length)

    for i in range(n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if np.isnan(ag) or np.isnan(al):
            out[i] = np.nan
        else:
            if al == 0.0:
                out[i] = 100.0 if ag > 0 else 0.0
            else:
                rs = ag / al
                out[i] = 100.0 - (100.0 / (1.0 + rs))
    return out


def shift_series(x: np.ndarray, shift: int) -> np.ndarray:
    """
    Pine: x[shift] means past value.
    shift=0 -> x
    shift>0 -> x shifted right, out[i] = x[i-shift]
    """
    n = len(x)
    out = np.full(n, np.nan, dtype=np.float64)
    if shift <= 0:
        return x.astype(np.float64, copy=True)
    out[shift:] = x[:-shift]
    return out


# =========================
# Heikin Ashi from REAL OHLC
# =========================

def heikin_ashi_from_real(
    open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(close)
    ha_close = (open_ + high + low + close) / 4.0

    ha_open = np.full(n, np.nan, dtype=np.float64)
    ha_high = np.full(n, np.nan, dtype=np.float64)
    ha_low  = np.full(n, np.nan, dtype=np.float64)

    if n == 0:
        return ha_open, ha_high, ha_low, ha_close

    ha_open[0] = (open_[0] + close[0]) / 2.0
    ha_high[0] = max(high[0], ha_open[0], ha_close[0])
    ha_low[0]  = min(low[0],  ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(high[i], ha_open[i], ha_close[i])
        ha_low[i]  = min(low[i],  ha_open[i], ha_close[i])

    return ha_open, ha_high, ha_low, ha_close


# =========================
# Pine-like cross helpers
# =========================

def crossover(cur_x: float, prev_x: float, cur_y: float, prev_y: float) -> bool:
    return (cur_x > cur_y) and (prev_x <= prev_y)

def crossunder(cur_x: float, prev_x: float, cur_y: float, prev_y: float) -> bool:
    return (cur_x < cur_y) and (prev_x >= prev_y)


# =========================
# Backtest: half_RSI_hh
# =========================

@dataclass
class HalfRSIParams:
    # Pine defaults (kept fixed here; you can optimize later if wanted)
    channelLength: int = 107
    channelMulti: float = 4.85  # (present for parity; not used directly in this python port)

    # RISK (optimized)
    atrPeriod: int = 21
    slMultiplier: float = 0.5214
    tpMultiplier: float = 1.7452
    trailMultiplier: float = 0.6656

    # SIGNAL (optimized)
    slow_window: int = 80
    shift: int = 0
    smooth_len: int = 30

    initial_capital: float = INITIAL_CAPITAL
    position_size_pct: float = POSITION_SIZE_PCT
    commission_pct: float = COMMISSION_PCT

    fill_mode: str = "next_open"  # "next_open" or "same_close"


def backtest_half_rsi_hh(df: pd.DataFrame, p: HalfRSIParams) -> Dict[str, Any]:
    df = df.copy().reset_index(drop=True)
    df.columns = [c.lower() for c in df.columns]

    for col in ["open", "high", "low", "close"]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    real_o = df["open"].astype(float).to_numpy()
    real_h = df["high"].astype(float).to_numpy()
    real_l = df["low"].astype(float).to_numpy()
    real_c = df["close"].astype(float).to_numpy()
    n = len(df)

    if n < 5:
        return {
            "equity_curve": [p.initial_capital],
            "trades": [],
            "sharpe": 0.0,
            "maxdd": 0.0,
            "num_trades": 0,
            "final_equity": p.initial_capital,
            "sum_win": 0.0,
            "sum_loss": 0.0,
            "n_win": 0,
            "n_loss": 0,
            "avg_profit": 0.0,
            "avg_loss": 0.0,
            "profit_factor": 0.0,
        }

    # HA from REAL
    ha_o, ha_h, ha_l, ha_c = heikin_ashi_from_real(real_o, real_h, real_l, real_c)
    sigClose = ha_c
    sigHL2 = (ha_h + ha_l) / 2.0

    # Risk series are HA (per your Pine assignments)
    riskClose = sigClose
    riskHigh  = ha_h
    riskLow   = ha_l

    # ATRs from REAL OHLC (chart-independent)
    atr_exit = atr_wilder(real_h, real_l, real_c, int(p.atrPeriod))
    _atr_chan = atr_wilder(real_h, real_l, real_c, int(p.channelLength))  # kept for parity; not used in this port

    # RSI signals on HA close, with shift
    slow_len = max(2, int(p.slow_window))
    fast_window = max(1, int(round(float(slow_len) / 2.0)))

    fast_rsi_raw = rsi_tv(sigClose, fast_window)
    slow_rsi_raw = rsi_tv(sigClose, slow_len)

    sh = max(0, int(p.shift))
    fast_rsi = shift_series(fast_rsi_raw, sh)
    slow_rsi = 100.0 - shift_series(slow_rsi_raw, sh)

    hot = fast_rsi - slow_rsi
    sm = max(1, int(p.smooth_len))
    hot_sm = sma(hot, sm)

    # Trend filters
    midChannel = sma(sigHL2, int(p.channelLength))
    uptrend = sigClose > midChannel

    lowerTF_Length = max(1, int(p.channelLength // 2))
    midChannel_lowerTF = sma(sigHL2, lowerTF_Length)
    above_lowerTF = sigClose > midChannel_lowerTF

    # State
    in_pos = False
    pending_entry = False
    entry_fill_idx = None
    qty = 0.0

    cash = float(p.initial_capital)
    equity_curve = np.zeros(n, dtype=np.float64)
    trades: List[Dict[str, Any]] = []

    # Pine-like vars
    tp_touched = False
    tp_crossdown = False
    trail_offset = np.nan
    trail_stop_level = np.nan
    take_profit_high = np.nan
    take_profit_low = np.nan

    prev_hot_sm = hot_sm[0] if n > 0 else np.nan

    for i in range(n):
        # Handle pending entry fill (next_open)
        if p.fill_mode == "next_open":
            if pending_entry and entry_fill_idx == i:
                fill_price = real_o[i]
                pos_value = cash * float(p.position_size_pct)
                qty = pos_value / fill_price if fill_price > 0 else 0.0
                commission = pos_value * float(p.commission_pct)
                cash -= (pos_value + commission)
                in_pos = True
                pending_entry = False
                trades[-1].update({
                    "fill_idx": int(i),
                    "fill_price": float(fill_price),
                    "commission_entry": float(commission),
                    "qty": float(qty),
                })

        # Signal: crossover(hot_sm, 0)
        if i > 0 and not np.isnan(hot_sm[i]) and not np.isnan(prev_hot_sm):
            cross_up = crossover(hot_sm[i], prev_hot_sm, 0.0, 0.0)
        else:
            cross_up = False

        if cross_up and (not np.isnan(uptrend[i])) and (not np.isnan(above_lowerTF[i])):
            longCondition = bool(uptrend[i] or above_lowerTF[i])
        else:
            longCondition = False

        # ENTRY
        if longCondition and (not in_pos) and (not pending_entry):
            tp_touched = False
            tp_crossdown = False

            atr_i = atr_exit[i]
            if not np.isnan(atr_i) and not np.isnan(riskClose[i]):
                trail_stop_level = riskClose[i] - (atr_i * float(p.slMultiplier))
                min_trail_offset = atr_i * 0.5
                trail_offset = max(atr_i * float(p.trailMultiplier), min_trail_offset)
                take_profit_low = riskClose[i] + (atr_i * float(p.tpMultiplier))
                take_profit_high = take_profit_low + trail_offset

                trades.append({
                    "signal_idx": int(i),
                    "side": "LONG",
                    "signal_price": float(riskClose[i]),
                    "slow_window": int(slow_len),
                    "fast_window": int(fast_window),
                    "shift": int(sh),
                    "smooth_len": int(sm),
                })

                if p.fill_mode == "same_close":
                    fill_price = riskClose[i]
                    pos_value = cash * float(p.position_size_pct)
                    qty = pos_value / fill_price if fill_price > 0 else 0.0
                    commission = pos_value * float(p.commission_pct)
                    cash -= (pos_value + commission)
                    in_pos = True
                    trades[-1].update({
                        "fill_idx": int(i),
                        "fill_price": float(fill_price),
                        "commission_entry": float(commission),
                        "qty": float(qty),
                    })
                else:
                    if i + 1 < n:
                        pending_entry = True
                        entry_fill_idx = i + 1
                    else:
                        trades[-1].update({"ignored": True})

        # POSITION MANAGEMENT / EXITS
        if in_pos:
            if (not tp_touched) and (not np.isnan(take_profit_high)) and (not np.isnan(riskHigh[i])) and (riskHigh[i] > take_profit_high):
                tp_touched = True

            if tp_touched and (not np.isnan(take_profit_low)) and (not np.isnan(riskClose[i])):
                tp_crossdown = bool(riskClose[i] < take_profit_low)
                if not np.isnan(trail_offset):
                    new_low = riskClose[i] - trail_offset
                    take_profit_low = max(take_profit_low, new_low)

            if (not np.isnan(riskLow[i])) and (not np.isnan(trail_stop_level)) and (riskLow[i] < trail_stop_level):
                longExit = True
                exit_reason = "SL_breach(HA_low < trail_stop_level)"
            elif tp_crossdown:
                longExit = True
                exit_reason = "TP_crossdown(riskClose < take_profit_low)"
            else:
                longExit = False
                exit_reason = None

            if longExit:
                if p.fill_mode == "next_open":
                    if i + 1 < n:
                        fill_price = real_o[i + 1]
                        fill_idx = i + 1
                    else:
                        fill_price = riskClose[i]
                        fill_idx = i
                else:
                    fill_price = riskClose[i]
                    fill_idx = i

                exit_value = qty * fill_price
                commission = exit_value * float(p.commission_pct)
                cash += (exit_value - commission)

                trades[-1].update({
                    "exit_idx": int(fill_idx),
                    "exit_price": float(fill_price),
                    "commission_exit": float(commission),
                    "reason": exit_reason
                })

                qty = 0.0
                in_pos = False
                pending_entry = False
                entry_fill_idx = None

                tp_touched = False
                tp_crossdown = False
                trail_offset = np.nan
                trail_stop_level = np.nan
                take_profit_high = np.nan
                take_profit_low = np.nan
            else:
                atr_i = atr_exit[i]
                if (not np.isnan(atr_i)) and (not np.isnan(riskHigh[i])) and (not np.isnan(trail_stop_level)):
                    new_stop = riskHigh[i] - (atr_i * float(p.slMultiplier))
                    trail_stop_level = max(trail_stop_level, new_stop)
        else:
            tp_touched = False
            tp_crossdown = False
            trail_offset = np.nan
            trail_stop_level = np.nan
            take_profit_high = np.nan
            take_profit_low = np.nan

        # Mark-to-market equity (REAL close)
        mtm = qty * real_c[i] if in_pos else 0.0
        equity_curve[i] = cash + mtm

        prev_hot_sm = hot_sm[i]

    # Close any open position at end (REAL close)
    if in_pos and qty > 0:
        fill_price = real_c[-1]
        exit_value = qty * fill_price
        commission = exit_value * float(p.commission_pct)
        cash += (exit_value - commission)
        trades[-1].update({
            "exit_idx": int(n - 1),
            "exit_price": float(fill_price),
            "commission_exit": float(commission),
            "reason": "endClose"
        })
        qty = 0.0
        in_pos = False
        equity_curve[-1] = cash

    eq = equity_curve.copy()
    if len(eq) < 2:
        sharpe = 0.0
        maxdd = 0.0
    else:
        rets = np.diff(eq) / np.maximum(eq[:-1], 1e-12)
        sharpe = float((np.mean(rets) / (np.std(rets) + 1e-12)) * math.sqrt(252.0)) if len(rets) > 1 else 0.0
        peak = np.maximum.accumulate(eq)
        dd = (eq - peak) / np.maximum(peak, 1e-12)
        maxdd = float(np.min(dd))

    completed = [t for t in trades if ("exit_price" in t and "fill_price" in t and not t.get("ignored", False))]
    num_trades = len(completed)

    sum_win = 0.0
    sum_loss = 0.0
    n_win = 0
    n_loss = 0

    for t in completed:
        if "qty" not in t:
            continue
        q = float(t["qty"])
        entry = float(t["fill_price"])
        exitp = float(t["exit_price"])
        entry_comm = float(t.get("commission_entry", 0.0))
        exit_comm  = float(t.get("commission_exit", 0.0))
        pnl = (exitp - entry) * q - entry_comm - exit_comm

        if pnl > 0:
            sum_win += pnl
            n_win += 1
        elif pnl < 0:
            sum_loss += -pnl
            n_loss += 1

    avg_profit = (sum_win / n_win) if n_win > 0 else 0.0
    avg_loss   = (sum_loss / n_loss) if n_loss > 0 else 0.0

    if avg_loss > 0:
        profit_factor = avg_profit / avg_loss
    else:
        profit_factor = 10.0 if avg_profit > 0 else 0.0

    return {
        "final_equity": float(eq[-1]),
        "total_return": float(eq[-1] / p.initial_capital - 1.0),
        "sharpe": float(sharpe),
        "maxdd": float(maxdd),
        "num_trades": int(num_trades),
        "trades": trades,
        "equity_curve": eq.tolist(),

        # Average-based PF bookkeeping
        "sum_win": float(sum_win),
        "sum_loss": float(sum_loss),
        "n_win": int(n_win),
        "n_loss": int(n_loss),
        "avg_profit": float(avg_profit),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
    }


# =========================
# Optimization (Optuna)
# =========================

def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

def load_files() -> Dict[str, pd.DataFrame]:
    files = list(DATA_DIR.glob("*.parquet")) + list(DATA_DIR.glob("*.csv"))
    files.sort()
    data: Dict[str, pd.DataFrame] = {}
    for fp in files:
        try:
            df = pd.read_parquet(fp) if fp.suffix.lower() == ".parquet" else pd.read_csv(fp)
            data[fp.stem.upper()] = df
        except Exception as e:
            print(f"Failed to load {fp.name}: {e}")
    return data


def run_optuna_optimization(
    n_trials: int = 200,
    n_files: int = 200,
    fill_mode: str = "next_open",
    study_name: Optional[str] = None,
    seed: int = 42,
    use_penalties: bool = True,
    pf_cap: float = 10.0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    ensure_dirs()
    data = load_files()
    if not data:
        raise SystemExit(f"No .csv/.parquet found in {DATA_DIR}. Put files there and re-run.")

    tickers_all = sorted(list(data.keys()))

    try:
        import optuna
        from optuna.samplers import TPESampler
        from optuna.pruners import MedianPruner
    except Exception as e:
        raise SystemExit("Optuna not installed. pip install optuna") from e

    if study_name is None:
        study_name = f"half_rsi_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    storage_path = OUT_DIR / f"{study_name}.db"
    storage_url = f"sqlite:///{storage_path}"

    sampler = TPESampler(seed=seed, n_startup_trials=30, multivariate=True)
    pruner = MedianPruner(n_startup_trials=20, n_warmup_steps=10)

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=True,
    )

    rng = random.Random(seed)

    def choose_tickers_for_trial() -> List[str]:
        if n_files is None or n_files <= 0 or n_files >= len(tickers_all):
            return tickers_all
        return rng.sample(tickers_all, k=n_files)

    def opt_fn(trial):
        # --------- OPTIMIZED PARAMETERS ----------
        slowW = trial.suggest_int("slow_window", 14, 80)
        shft  = trial.suggest_int("shift", 0, 6)
        smth  = trial.suggest_int("smooth_len", 3, 30)

        atrP  = trial.suggest_int("atrPeriod", 5, 60)
        slM   = trial.suggest_float("slMultiplier", 0.5, 6.0)
        tpM   = trial.suggest_float("tpMultiplier", 0.5, 8.0)
        trM   = trial.suggest_float("trailMultiplier", 0.5, 8.0)

        tickers = choose_tickers_for_trial()

        total_sum_win = 0.0
        total_sum_loss = 0.0
        total_n_win = 0
        total_n_loss = 0
        agg_trades = 0
        agg_dd = 0.0

        for tk in tickers:
            params = HalfRSIParams(
                channelLength=107,
                channelMulti=4.85,

                slow_window=int(slowW),
                shift=int(shft),
                smooth_len=int(smth),

                atrPeriod=int(atrP),
                slMultiplier=float(slM),
                tpMultiplier=float(tpM),
                trailMultiplier=float(trM),

                initial_capital=INITIAL_CAPITAL,
                position_size_pct=POSITION_SIZE_PCT,
                commission_pct=COMMISSION_PCT,
                fill_mode=fill_mode,
            )
            m = backtest_half_rsi_hh(data[tk], params)

            total_sum_win  += float(m.get("sum_win", 0.0))
            total_sum_loss += float(m.get("sum_loss", 0.0))
            total_n_win    += int(m.get("n_win", 0))
            total_n_loss   += int(m.get("n_loss", 0))

            agg_trades += int(m.get("num_trades", 0))
            agg_dd += abs(float(m.get("maxdd", 0.0)))

        avg_profit_overall = (total_sum_win / total_n_win) if total_n_win > 0 else 0.0
        avg_loss_overall   = (total_sum_loss / total_n_loss) if total_n_loss > 0 else 0.0

        if avg_loss_overall > 0:
            pf_overall = avg_profit_overall / avg_loss_overall
        else:
            pf_overall = 10.0 if avg_profit_overall > 0 else 0.0

        pf_overall = float(min(pf_overall, float(pf_cap)))

        score = pf_overall
        penalty = 0.0
        if use_penalties:
            if agg_trades < 3:
                penalty += 3.0
            elif agg_trades < 10:
                penalty += 1.0
            score = pf_overall - (agg_dd * 0.25) - penalty

        trial.set_user_attr("pf_overall", float(pf_overall))
        trial.set_user_attr("avg_profit_overall", float(avg_profit_overall))
        trial.set_user_attr("avg_loss_overall", float(avg_loss_overall))
        trial.set_user_attr("total_sum_win", float(total_sum_win))
        trial.set_user_attr("total_sum_loss", float(total_sum_loss))
        trial.set_user_attr("total_n_win", int(total_n_win))
        trial.set_user_attr("total_n_loss", int(total_n_loss))
        trial.set_user_attr("total_trades", int(agg_trades))
        trial.set_user_attr("sum_abs_maxdd", float(agg_dd))
        trial.set_user_attr("penalty", float(penalty))
        trial.set_user_attr("n_files_used", int(len(tickers)))

        return float(score)

    study.optimize(opt_fn, n_trials=n_trials, n_jobs=1, show_progress_bar=True, gc_after_trial=True)

    best_params = dict(study.best_params)
    best_params["best_score"] = float(study.best_value)
    best_params["fill_mode"] = fill_mode
    best_params["timestamp"] = time.time()
    best_params["score_is_avg_pf_overall_optionA"] = True
    best_params["pf_cap"] = float(pf_cap)
    best_params["use_penalties"] = bool(use_penalties)

    with open(OUT_DIR / "best_params_half_rsi.json", "w") as f:
        json.dump(best_params, f, indent=2)

    rows = []
    for tk in tickers_all:
        params = HalfRSIParams(
            channelLength=107,
            channelMulti=4.85,

            slow_window=int(best_params.get("slow_window", 32)),
            shift=int(best_params.get("shift", 2)),
            smooth_len=int(best_params.get("smooth_len", 12)),

            atrPeriod=int(best_params["atrPeriod"]),
            slMultiplier=float(best_params["slMultiplier"]),
            tpMultiplier=float(best_params["tpMultiplier"]),
            trailMultiplier=float(best_params["trailMultiplier"]),

            initial_capital=INITIAL_CAPITAL,
            position_size_pct=POSITION_SIZE_PCT,
            commission_pct=COMMISSION_PCT,
            fill_mode=fill_mode,
        )
        m = backtest_half_rsi_hh(data[tk], params)
        rows.append({
            "ticker": tk,
            "final_equity": m["final_equity"],
            "total_return": m["total_return"],
            "sharpe": m["sharpe"],
            "maxdd": m["maxdd"],
            "num_trades": m["num_trades"],
            "avg_profit": m["avg_profit"],
            "avg_loss": m["avg_loss"],
            "n_win": m["n_win"],
            "n_loss": m["n_loss"],
            "profit_factor": m["profit_factor"],
        })

        pd.DataFrame(m["trades"]).to_csv(OUT_DIR / f"{tk}_trades_best.csv", index=False)

        if HAS_PLT:
            plt.figure(figsize=(10, 4))
            plt.plot(m["equity_curve"])
            plt.title(
                f"{tk} Equity (Best) slowW={params.slow_window} shift={params.shift} sm={params.smooth_len} "
                f"atr={params.atrPeriod} sl={params.slMultiplier:.3f} tp={params.tpMultiplier:.3f} trail={params.trailMultiplier:.3f}"
            )
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(OUT_DIR / f"equity_{tk}_best.png")
            plt.close()

    summary_df = pd.DataFrame(rows).sort_values("profit_factor", ascending=False)
    summary_df.to_csv(OUT_DIR / "optimization_summary_half_rsi.csv", index=False)

    trials_df = study.trials_dataframe()
    trials_df.to_csv(OUT_DIR / "optuna_trials_half_rsi.csv", index=False)

    # Recompute overall (ALL FILES) average-based PF using totals
    total_sum_win = 0.0
    total_sum_loss = 0.0
    total_n_win = 0
    total_n_loss = 0
    total_trades = 0

    for tk in tickers_all:
        params = HalfRSIParams(
            channelLength=107,
            channelMulti=4.85,

            slow_window=int(best_params.get("slow_window", 32)),
            shift=int(best_params.get("shift", 2)),
            smooth_len=int(best_params.get("smooth_len", 12)),

            atrPeriod=int(best_params["atrPeriod"]),
            slMultiplier=float(best_params["slMultiplier"]),
            tpMultiplier=float(best_params["tpMultiplier"]),
            trailMultiplier=float(best_params["trailMultiplier"]),

            initial_capital=INITIAL_CAPITAL,
            position_size_pct=POSITION_SIZE_PCT,
            commission_pct=COMMISSION_PCT,
            fill_mode=fill_mode,
        )
        m = backtest_half_rsi_hh(data[tk], params)

        total_sum_win  += float(m.get("sum_win", 0.0))
        total_sum_loss += float(m.get("sum_loss", 0.0))
        total_n_win    += int(m.get("n_win", 0))
        total_n_loss   += int(m.get("n_loss", 0))
        total_trades   += int(m.get("num_trades", 0))

    avg_profit_all = (total_sum_win / total_n_win) if total_n_win > 0 else 0.0
    avg_loss_all   = (total_sum_loss / total_n_loss) if total_n_loss > 0 else 0.0

    if avg_loss_all > 0:
        pf_all = avg_profit_all / avg_loss_all
    else:
        pf_all = 10.0 if avg_profit_all > 0 else 0.0

    pf_all = float(min(pf_all, float(pf_cap)))

    print("\n=== BEST PARAMS ===")
    for k, v in best_params.items():
        if k not in ("timestamp",):
            print(f"{k}: {v}")

    print("\n=== OVERALL (ALL FILES) PROFIT FACTOR (AVERAGE-BASED) ===")
    print(f"Avg Profit (wins): {avg_profit_all:,.2f}  (n_win={total_n_win})")
    print(f"Avg Loss (losses): {avg_loss_all:,.2f}  (n_loss={total_n_loss})")
    print(f"PF Overall:        {pf_all:.4f}")
    print(f"Total Trades:      {total_trades}")

    print("\nSaved outputs to:", str(OUT_DIR))
    print(summary_df.to_string(index=False))
    return best_params, summary_df


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200, help="Optuna trials")
    ap.add_argument("--files", type=int, default=200, help="Number of files to use per trial (subsample); if >= total -> use all")
    ap.add_argument("--fill", choices=["next_open", "same_close"], default="same_close", help="Execution model")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for file subsampling")
    ap.add_argument("--no-penalties", action="store_true", help="Disable trade/DD penalties (score = PF_overall only)")
    ap.add_argument("--pf-cap", type=float, default=10.0, help="Cap PF_overall at this value (default 10.0)")
    args = ap.parse_args()

    run_optuna_optimization(
        n_trials=args.trials,
        n_files=args.files,
        fill_mode=args.fill,
        seed=args.seed,
        use_penalties=(not args.no_penalties),
        pf_cap=args.pf_cap,
    )
