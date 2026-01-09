#!/usr/bin/env python3
"""
Bayes_opt_adapt_half_RSI.py  (Pine-match + 3 fill modes)  **HYBRID OBJECTIVE (like half_RSI)**

UPDATED PER REQUEST: **Use realRisk**
- SIGNALS use Heikin Ashi (HA) computed from REAL OHLC (chart-type independent)
- RISK / EXITS use REAL prices (riskClose/riskHigh/riskLow = REAL close/high/low)
- ATR is REAL ATR(atrPeriod) (Wilder/RMA of REAL TR)

SIGNALS (HA):
- baseRSI = RSI(haClose, basePeriod)
- adaptivePeriod = round(((100-baseRSI)/100) * (maxPeriod-minPeriod) + minPeriod), clipped
- slow_period = adaptivePeriod
- fast_period = round(adaptivePeriod/2), min 2
- fast_rsi = RSI(haClose, fast_period) (period selected from precomputed RSI 2..34)
- slow_rsi = RSI(haClose, slow_period) (period selected from precomputed RSI 2..34)
- longCondition = crossover(fast_rsi, slow_rsi)
- NO trend filter, NO shorts

EXITS / RISK (REAL price logic):
- riskClose = REAL close
- riskHigh  = REAL high
- riskLow   = REAL low
- atr = REAL ATR(atrPeriod) (Wilder/RMA of REAL TR)
Entry initializes:
    trail_stop_level = riskClose - atr*slMultiplier
    trail_offset     = max(atr*trailMultiplier, atr*0.5)
    take_profit_low  = riskClose + atr*tpMultiplier
    take_profit_high = take_profit_low + trail_offset
In position:
    if not tp_touched and riskHigh > take_profit_high: tp_touched = True
    if tp_touched:
        tp_crossdown = riskClose < take_profit_low
        take_profit_low = max(take_profit_low, riskClose - trail_offset)
    longExit = (riskLow < trail_stop_level) or tp_crossdown
    trail_stop_level = max(trail_stop_level, riskHigh - atr*slMultiplier)
Flat resets vars.

EXECUTION / FILL MODEL (--fill):
- same_close : enter/exit at REAL close[i]
- next_open  : enter/exit at REAL open[i+1]
- intrabar   : conservative intrabar exits:
    * STOP breach: fill at stop_level, but if open[i] <= stop_level (gap-through), fill at open[i].
    * TP-crossdown: fill at REAL close[i] (conservative market-at-close).
    * If both same bar, STOP has priority (worst-case).

OBJECTIVE (NO STOCK DOMINATES; equal weight per file):
- Optimize BOTH:
    1) avg_return_overall: mean of per-file total_return (decimal, not %)
    2) pct_positive_return: fraction of files with total_return > 0
- score = (1 + avg_return_overall) * (pct_positive_return ** alpha)
  alpha default 0.5 (configurable via --alpha)

Optional penalties (same spirit as half_RSI):
- trade penalty (aggregated across files):
    * total_trades < 3  => -3
    * total_trades < 10 => -1
- drawdown penalty: score -= 0.25 * sum(abs(maxdd))  where maxdd is per-file drawdown fraction (e.g. 0.18)

Outputs:
- best_params_score_*.json
- trials_score_*.csv
- per_file_results_score_*.csv (best params on ALL files)
- study_stats_score_*.json
"""

import os
import json
import hashlib
import logging
import warnings
import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import numpy as np
import pandas as pd
from numba import jit, prange

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =========================
# Defaults (Pine-ish)
# =========================

DEFAULT_PARAMS = {
    "atr_period": 58,
    "sl_multiplier": 0.9193,
    "tp_multiplier": 9.3842,
    "trail_multiplier": 6.705,

    "base_period": 19,
    "min_period": 20,
    "max_period": 32,

    # backtest settings
    "initial_capital": 100000.0,
    "position_size_pct": 0.10,   # 10% equity
    "commission_pct": 0.0006,    # 0.06%
}

# =========================
# Numba indicators
# =========================

@jit(nopython=True, cache=True)
def calculate_heikin_ashi_numba(open_arr, high_arr, low_arr, close_arr):
    n = len(close_arr)
    ha_open = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    ha_close = np.zeros(n)

    ha_close[0] = (open_arr[0] + high_arr[0] + low_arr[0] + close_arr[0]) / 4.0
    ha_open[0] = (open_arr[0] + close_arr[0]) / 2.0
    ha_high[0] = max(high_arr[0], ha_open[0], ha_close[0])
    ha_low[0] = min(low_arr[0], ha_open[0], ha_close[0])

    for i in range(1, n):
        ha_close[i] = (open_arr[i] + high_arr[i] + low_arr[i] + close_arr[i]) / 4.0
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) / 2.0
        ha_high[i] = max(high_arr[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_arr[i], ha_open[i], ha_close[i])

    return ha_open, ha_high, ha_low, ha_close


@jit(nopython=True, cache=True)
def rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    rsi = np.full(n, np.nan)
    if period <= 0 or n < period:
        return rsi

    deltas = np.zeros(n)
    deltas[1:] = close[1:] - close[:-1]

    gain = np.zeros(n)
    loss = np.zeros(n)

    for i in range(1, period):
        if deltas[i] > 0:
            gain[i] = deltas[i]
        else:
            loss[i] = -deltas[i]

    avg_gain = np.sum(gain[1:period]) / period
    avg_loss = np.sum(loss[1:period]) / period

    if avg_loss == 0.0:
        rsi[period - 1] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period - 1] = 100.0 - (100.0 / (1.0 + rs))

    for i in range(period, n):
        if deltas[i] > 0:
            g = deltas[i]
            l = 0.0
        else:
            g = 0.0
            l = -deltas[i]

        avg_gain = ((avg_gain * (period - 1)) + g) / period
        avg_loss = ((avg_loss * (period - 1)) + l) / period

        if avg_loss == 0.0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True, cache=True)
def calculate_all_rsi_numba(close: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    n = len(close)
    num_periods = max_period - min_period + 1
    out = np.full((n, num_periods), np.nan)
    for p_idx in prange(num_periods):
        p = min_period + p_idx
        out[:, p_idx] = rsi_numba(close, p)
    return out


@jit(nopython=True, cache=True)
def rma_numba(src: np.ndarray, length: int) -> np.ndarray:
    n = len(src)
    out = np.full(n, np.nan)
    if length <= 0 or n < length:
        return out

    s = 0.0
    for i in range(length):
        s += src[i]
    out[length - 1] = s / length

    alpha = 1.0 / length
    for i in range(length, n):
        out[i] = out[i - 1] + alpha * (src[i] - out[i - 1])
    return out


@jit(nopython=True, cache=True)
def true_range_real_numba(high_arr: np.ndarray, low_arr: np.ndarray, close_arr: np.ndarray) -> np.ndarray:
    n = len(close_arr)
    tr = np.zeros(n)
    tr[0] = high_arr[0] - low_arr[0]
    for i in range(1, n):
        hl = high_arr[i] - low_arr[i]
        hc = abs(high_arr[i] - close_arr[i - 1])
        lc = abs(low_arr[i] - close_arr[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr

# =========================
# Signals + exits with reason/stop_level for intrabar fills
# =========================

@jit(nopython=True, cache=True)
def build_signals_and_exits_with_reason_numba(
    fast_rsi: np.ndarray,
    slow_rsi: np.ndarray,
    risk_close: np.ndarray,   # REAL close  (UPDATED: realRisk)
    risk_high: np.ndarray,    # REAL high
    risk_low: np.ndarray,     # REAL low
    atr_real: np.ndarray,     # REAL ATR (Wilder)
    sl_mult: float,
    tp_mult: float,
    trail_mult: float,
    min_bars: int
):
    n = len(risk_close)
    buy = np.zeros(n, dtype=np.bool_)
    sell = np.zeros(n, dtype=np.bool_)
    exit_reason = np.zeros(n, dtype=np.int8)     # 0 none, 1 stop, 2 tp_crossdown
    stop_level_out = np.full(n, np.nan)

    in_pos = False

    tp_touched = False
    tp_crossdown = False
    trail_offset = np.nan
    trail_stop_level = np.nan
    take_profit_high = np.nan
    take_profit_low = np.nan

    prev_fast = np.nan
    prev_slow = np.nan

    for i in range(n):
        if i < min_bars:
            prev_fast = fast_rsi[i]
            prev_slow = slow_rsi[i]
            continue

        fr = fast_rsi[i]
        sr = slow_rsi[i]
        rc = risk_close[i]
        rh = risk_high[i]
        rl = risk_low[i]
        atr = atr_real[i]

        if np.isnan(fr) or np.isnan(sr) or np.isnan(rc) or np.isnan(rh) or np.isnan(rl) or np.isnan(atr):
            prev_fast = fr
            prev_slow = sr
            continue

        # crossover(fast, slow)
        long_condition = False
        if not np.isnan(prev_fast) and not np.isnan(prev_slow):
            if (fr > sr) and (prev_fast <= prev_slow):
                long_condition = True

        # ENTRY
        if (not in_pos) and long_condition:
            tp_touched = False
            tp_crossdown = False

            trail_stop_level = rc - (atr * sl_mult)

            min_trail_offset = atr * 0.5
            toff = atr * trail_mult
            trail_offset = toff if toff >= min_trail_offset else min_trail_offset

            take_profit_low = rc + (atr * tp_mult)
            take_profit_high = take_profit_low + trail_offset

            buy[i] = True
            in_pos = True

        # POSITION MGMT
        elif in_pos:
            # TP touched: bar-high > tp_high
            if (not tp_touched) and (not np.isnan(take_profit_high)):
                if rh > take_profit_high:
                    tp_touched = True

            # TP ratchet + crossdown
            if tp_touched and (not np.isnan(take_profit_low)) and (not np.isnan(trail_offset)):
                tp_crossdown = rc < take_profit_low
                new_low = rc - trail_offset
                if new_low > take_profit_low:
                    take_profit_low = new_low
            else:
                tp_crossdown = False

            # stop breach: bar-low < trail_stop_level
            stop_breach = False
            if not np.isnan(trail_stop_level):
                stop_breach = rl < trail_stop_level

            # STOP priority
            if stop_breach or tp_crossdown:
                sell[i] = True
                stop_level_out[i] = trail_stop_level
                exit_reason[i] = 1 if stop_breach else 2

                # flat reset
                in_pos = False
                tp_touched = False
                tp_crossdown = False
                trail_offset = np.nan
                trail_stop_level = np.nan
                take_profit_high = np.nan
                take_profit_low = np.nan
            else:
                # trail stop only up
                new_stop = rh - (atr * sl_mult)
                if np.isnan(trail_stop_level) or new_stop > trail_stop_level:
                    trail_stop_level = new_stop

        prev_fast = fr
        prev_slow = sr

    return buy, sell, exit_reason, stop_level_out

# =========================
# Unified simulator with fill modes (tracks GP/GL)
# =========================

@jit(nopython=True, cache=True)
def run_simulation_with_fill_modes_pf_numba(
    open_arr,
    close_arr,
    buy_signal,
    sell_signal,
    exit_reason,     # 0 none, 1 stop, 2 tp_crossdown
    stop_level,      # trailing stop level on bar i when sell fires
    initial_capital,
    pos_size_pct,
    comm_pct,
    fill_mode_code   # 0 same_close, 1 next_open, 2 intrabar
):
    n = len(close_arr)
    equity_curve = np.zeros(n)

    cash = initial_capital
    qty = 0.0
    entry_price = 0.0
    entry_comm = 0.0

    trade_count = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(n):
        # mark to market at REAL close
        mark = close_arr[i]
        if np.isnan(mark) or mark <= 0:
            equity_curve[i] = cash + (qty * mark if qty > 0 else 0.0)
            continue

        # ENTRY
        if buy_signal[i] and qty == 0.0:
            if fill_mode_code == 1:
                fill = open_arr[i + 1] if (i + 1) < n else np.nan
            else:
                fill = close_arr[i]  # same_close or intrabar enter at close[i]

            if (not np.isnan(fill)) and fill > 0:
                pos_value = cash * pos_size_pct
                if pos_value > 0:
                    qty = pos_value / fill
                    entry_price = fill
                    entry_comm = pos_value * comm_pct
                    cash -= (pos_value + entry_comm)

        # EXIT
        if sell_signal[i] and qty > 0.0:
            if fill_mode_code == 0:
                fill = close_arr[i]
            elif fill_mode_code == 1:
                fill = open_arr[i + 1] if (i + 1) < n else close_arr[i]
            else:
                # intrabar conservative
                if exit_reason[i] == 1:
                    stp = stop_level[i]
                    o = open_arr[i]
                    if np.isnan(stp):
                        fill = close_arr[i]
                    else:
                        if (not np.isnan(o)) and o > 0 and o <= stp:
                            fill = o
                        else:
                            fill = stp
                else:
                    fill = close_arr[i]  # tp_crossdown at close

            if (not np.isnan(fill)) and fill > 0:
                exit_value = qty * fill
                exit_comm = exit_value * comm_pct

                pnl = (fill - entry_price) * qty - entry_comm - exit_comm
                if pnl > 0:
                    gross_profit += pnl
                elif pnl < 0:
                    gross_loss += -pnl

                cash += (exit_value - exit_comm)
                trade_count += 1

                qty = 0.0
                entry_price = 0.0
                entry_comm = 0.0

        equity_curve[i] = cash + (qty * mark if qty > 0 else 0.0)

    # Close any open position at last close
    if qty > 0.0:
        fill = close_arr[-1]
        if not np.isnan(fill) and fill > 0:
            exit_value = qty * fill
            exit_comm = exit_value * comm_pct
            pnl = (fill - entry_price) * qty - entry_comm - exit_comm
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += -pnl
            cash += (exit_value - exit_comm)
            trade_count += 1
        qty = 0.0
        equity_curve[-1] = cash

    return equity_curve, trade_count, gross_profit, gross_loss

# =========================
# Strategy params
# =========================

@dataclass
class StrategyParams:
    atr_period: int = 47
    sl_multiplier: float = 5.805
    tp_multiplier: float = 4.082
    trail_multiplier: float = 7.76

    base_period: int = 14
    min_period: int = 3
    max_period: int = 34

    initial_capital: float = 100000.0
    position_size_pct: float = 0.1
    commission_pct: float = 0.0006

    def to_hash(self) -> str:
        params_str = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()

# =========================
# Optimizer
# =========================

class BayesianAdaptHalfRSIOptimizer:
    """
    Bayesian optimization for adapt_half_RSI_hh.

    Objective (like half_RSI; equal weight per file):
      score = (1 + avg_return_overall) * (pct_positive_return ** alpha)

    Optional penalties:
      score -= 0.25 * sum(abs(maxdd_fraction))  and trade-count penalty as described in docstring.

    UPDATED: realRisk (exits use REAL close/high/low).
    """

    def __init__(self, data_dir: str = "./data", results_dir: str = "./results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True, parents=True)
        self.study = None

    def load_data_chunk(self, file_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            if file_path.suffix.lower() == ".parquet":
                df = pd.read_parquet(file_path)
            elif file_path.suffix.lower() == ".csv":
                df = pd.read_csv(file_path)
            else:
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

        df.columns = [c.lower() for c in df.columns]
        for col in ["open", "high", "low", "close"]:
            if col not in df.columns:
                df[col] = np.nan
        if "volume" not in df.columns:
            df["volume"] = np.nan

        if sample_size and sample_size > 0 and sample_size < len(df):
            df = df.tail(sample_size)

        return df.reset_index(drop=True)

    def find_data_files(self) -> List[Path]:
        files = list(self.data_dir.rglob("*.parquet")) + list(self.data_dir.rglob("*.csv"))
        files.sort()
        logger.info(f"Found {len(files)} data files in {self.data_dir}")
        return files

    def calculate_indicators_pine(self, df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        out = df.copy()

        o = out["open"].values.astype(np.float64)
        h = out["high"].values.astype(np.float64)
        l = out["low"].values.astype(np.float64)
        c = out["close"].values.astype(np.float64)

        # HA for signals
        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi_numba(o, h, l, c)
        out["ha_open"] = ha_o
        out["ha_high"] = ha_h
        out["ha_low"] = ha_l
        out["ha_close"] = ha_c

        base_rsi = rsi_numba(ha_c, int(params.base_period))
        out["base_rsi"] = base_rsi

        scaled = (100.0 - base_rsi) / 100.0
        adapt = np.round(scaled * (params.max_period - params.min_period) + params.min_period).astype(np.float64)

        for i in range(len(adapt)):
            if np.isnan(adapt[i]):
                continue
            if adapt[i] < params.min_period:
                adapt[i] = params.min_period
            elif adapt[i] > params.max_period:
                adapt[i] = params.max_period

        out["adaptive_period"] = adapt

        fast_p = np.round(adapt / 2.0).astype(np.float64)
        for i in range(len(fast_p)):
            if np.isnan(fast_p[i]):
                continue
            if fast_p[i] < 2:
                fast_p[i] = 2
        out["fast_period"] = fast_p
        out["slow_period"] = adapt

        MIN_RSI = 2
        MAX_RSI = 34
        all_rsi = calculate_all_rsi_numba(ha_c, MIN_RSI, MAX_RSI)

        fast_rsi = np.full(len(out), np.nan)
        slow_rsi = np.full(len(out), np.nan)

        for i in range(len(out)):
            fp = fast_p[i]
            sp = adapt[i]
            if np.isnan(fp) or np.isnan(sp):
                continue
            fpi = int(fp)
            spi = int(sp)
            if MIN_RSI <= fpi <= MAX_RSI:
                fast_rsi[i] = all_rsi[i, fpi - MIN_RSI]
            if MIN_RSI <= spi <= MAX_RSI:
                slow_rsi[i] = all_rsi[i, spi - MIN_RSI]

        out["fast_rsi"] = fast_rsi
        out["slow_rsi"] = slow_rsi

        # REAL ATR (Wilder/RMA of REAL TR)
        tr_real = true_range_real_numba(h, l, c)
        atr_real = rma_numba(tr_real, int(params.atr_period))
        out["atr_real"] = atr_real

        return out

    def backtest_single(self, df: pd.DataFrame, params: StrategyParams, fill_mode: str = "same_close") -> Dict[str, Any]:
        try:
            dfi = self.calculate_indicators_pine(df, params)
            min_bars = int(max(params.max_period, params.atr_period, params.base_period) + 5)

            # UPDATED: realRisk -> pass REAL close/high/low for risk series
            buy_signal, sell_signal, exit_reason, stop_level = build_signals_and_exits_with_reason_numba(
                dfi["fast_rsi"].values.astype(np.float64),
                dfi["slow_rsi"].values.astype(np.float64),
                dfi["close"].values.astype(np.float64),   # risk_close (REAL)
                dfi["high"].values.astype(np.float64),    # risk_high  (REAL)
                dfi["low"].values.astype(np.float64),     # risk_low   (REAL)
                dfi["atr_real"].values.astype(np.float64),
                float(params.sl_multiplier),
                float(params.tp_multiplier),
                float(params.trail_multiplier),
                int(min_bars),
            )

            if fill_mode == "same_close":
                mode_code = 0
            elif fill_mode == "next_open":
                mode_code = 1
            else:
                mode_code = 2

            equity_curve, n_trades, gp, gl = run_simulation_with_fill_modes_pf_numba(
                dfi["open"].values.astype(np.float64),
                dfi["close"].values.astype(np.float64),
                buy_signal,
                sell_signal,
                exit_reason,
                stop_level,
                float(params.initial_capital),
                float(params.position_size_pct),
                float(params.commission_pct),
                int(mode_code),
            )

            final_equity = float(equity_curve[-1])
            total_return = (final_equity / float(params.initial_capital)) - 1.0
            total_ret_pct = total_return * 100.0

            # Sharpe (reporting only)
            if len(equity_curve) > 2:
                rets = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
                std = np.std(rets)
                sharpe = float((np.mean(rets) / std) * np.sqrt(252.0)) if std > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown (negative %)
            cum_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - cum_max) / np.maximum(cum_max, 1e-12)
            max_dd_pct = float(np.min(dd) * 100.0)

            gp = float(gp)
            gl = float(gl)
            pf = gp / gl if gl > 0 else (10.0 if gp > 0 else 0.0)

            return {
                "total_return": float(total_return),             # decimal
                "total_return_pct": float(total_ret_pct),        # percent
                "sharpe_ratio": float(sharpe),
                "max_drawdown_pct": float(max_dd_pct),
                "n_trades": int(n_trades),
                "final_equity": float(final_equity),
                "gross_profit": float(gp),
                "gross_loss": float(gl),
                "profit_factor": float(pf),                      # reference only
            }
        except Exception as e:
            logger.error(f"backtest_single error: {e}")
            return {
                "total_return": -1.0,
                "total_return_pct": -100.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": -100.0,
                "n_trades": 0,
                "final_equity": float(params.initial_capital),
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
            }

    def run_backtest_hybrid_score(
        self,
        files: List[Path],
        sample_size: int,
        fill_mode: str,
        alpha: float,
        use_penalties: bool,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Equal-weight per-file aggregation (no domination):
          avg_return_overall = mean(total_return per file)
          pct_positive_return = fraction(total_return > 0)
          score = (1 + avg_return_overall) * (pct_positive_return ** alpha)
        """
        params = StrategyParams(**kwargs)

        used = 0
        sum_ret = 0.0
        pos_cnt = 0

        total_trades = 0
        sum_abs_maxdd = 0.0  # in fraction units, e.g. 0.18
        per_file: List[Dict[str, Any]] = []

        for fp in files:
            df = self.load_data_chunk(fp, sample_size)
            if len(df) <= 100:
                continue

            r = self.backtest_single(df, params, fill_mode=fill_mode)
            r["file"] = fp.name
            per_file.append(r)

            tr = float(r.get("total_return", 0.0))
            sum_ret += tr
            if tr > 0.0:
                pos_cnt += 1
            used += 1

            total_trades += int(r.get("n_trades", 0))
            mdd_pct = float(r.get("max_drawdown_pct", 0.0))
            sum_abs_maxdd += abs(mdd_pct) / 100.0

        if used <= 0:
            return {
                "score": 0.0,
                "avg_return_overall": 0.0,
                "pct_positive_return": 0.0,
                "positive_files": 0,
                "n_files_used": 0,
                "total_trades": 0,
                "sum_abs_maxdd": 0.0,
                "penalty": 0.0,
                "per_file": per_file,
            }

        avg_return_overall = sum_ret / float(used)
        pct_positive_return = pos_cnt / float(used)

        base = 1.0 + avg_return_overall
        base = max(base, 1e-6)
        score = base * (pct_positive_return ** float(alpha))

        penalty = 0.0
        if use_penalties:
            if total_trades < 3:
                penalty += 3.0
            elif total_trades < 10:
                penalty += 1.0
            score = score - (sum_abs_maxdd * 0.25) - penalty

        return {
            "score": float(score),
            "avg_return_overall": float(avg_return_overall),
            "pct_positive_return": float(pct_positive_return),
            "positive_files": int(pos_cnt),
            "n_files_used": int(used),
            "total_trades": int(total_trades),
            "sum_abs_maxdd": float(sum_abs_maxdd),
            "penalty": float(penalty),
            "per_file": per_file,
        }

    def objective(
        self,
        trial: optuna.Trial,
        all_files: List[Path],
        sample_size: int,
        n_files: int,
        fill_mode: str,
        alpha: float,
        use_penalties: bool,
        seed: int,
    ) -> float:
        params = DEFAULT_PARAMS.copy()

        # Optimize Pine inputs (reasonable bounds)
        params["atr_period"] = trial.suggest_int("atr_period", 10, 80)
        params["sl_multiplier"] = trial.suggest_float("sl_multiplier", 0.5, 12.0)
        params["tp_multiplier"] = trial.suggest_float("tp_multiplier", 0.5, 12.0)
        params["trail_multiplier"] = trial.suggest_float("trail_multiplier", 0.5, 20.0)

        params["base_period"] = trial.suggest_int("base_period", 2, 34)
        params["min_period"] = trial.suggest_int("min_period", 2, 20)
        params["max_period"] = trial.suggest_int("max_period", 10, 34)

        if params["min_period"] >= params["max_period"]:
            return 0.0

        if len(all_files) <= 0:
            return 0.0

        # random subsample per trial (like your half_RSI)
        if n_files is None or n_files <= 0 or n_files >= len(all_files):
            trial_files = all_files
        else:
            rng = random.Random(int(seed) + int(trial.number) * 1009)
            trial_files = rng.sample(all_files, k=int(n_files))

        metrics = self.run_backtest_hybrid_score(
            files=trial_files,
            sample_size=sample_size,
            fill_mode=fill_mode,
            alpha=alpha,
            use_penalties=use_penalties,
            **params,
        )

        if int(metrics.get("total_trades", 0)) == 0 or int(metrics.get("n_files_used", 0)) == 0:
            return 0.0

        score = float(metrics.get("score", 0.0))

        trial.set_user_attr("avg_return_overall", float(metrics.get("avg_return_overall", 0.0)))
        trial.set_user_attr("pct_positive_return", float(metrics.get("pct_positive_return", 0.0)))
        trial.set_user_attr("positive_files", int(metrics.get("positive_files", 0)))
        trial.set_user_attr("n_files_used", int(metrics.get("n_files_used", 0)))
        trial.set_user_attr("alpha", float(alpha))
        trial.set_user_attr("fill_mode", str(fill_mode))

        trial.set_user_attr("total_trades", int(metrics.get("total_trades", 0)))
        trial.set_user_attr("sum_abs_maxdd", float(metrics.get("sum_abs_maxdd", 0.0)))
        trial.set_user_attr("penalty", float(metrics.get("penalty", 0.0)))

        return float(score)

    def optimize_bayesian(
        self,
        n_trials: int = 100,
        n_files: int = 5,
        sample_size: int = 5000,
        n_jobs: int = -1,
        fill_mode: str = "same_close",
        alpha: float = 0.5,
        seed: int = 42,
        use_penalties: bool = True,
    ) -> Dict[str, Any]:
        files = self.find_data_files()
        if len(files) == 0:
            logger.error("No data files found in data directory")
            return {}

        study_name = f"adapt_half_rsi_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = f"sqlite:///{self.results_dir}/{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=int(seed), n_startup_trials=20, multivariate=True),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            direction="maximize",
            load_if_exists=True,
        )
        self.study = study

        logger.info(
            f"Starting optimization (HYBRID score) with {n_trials} trials | fill={fill_mode} | alpha={alpha} | "
            f"files_per_trial={n_files} | penalties={use_penalties}"
        )

        def objective_with_args(tr):
            return self.objective(
                trial=tr,
                all_files=files,
                sample_size=sample_size,
                n_files=n_files,
                fill_mode=fill_mode,
                alpha=alpha,
                use_penalties=use_penalties,
                seed=seed,
            )

        study.optimize(
            objective_with_args,
            n_trials=int(n_trials),
            n_jobs=(n_jobs if n_jobs > 0 else os.cpu_count()),
            show_progress_bar=True,
            gc_after_trial=True,
        )

        if not study.trials or all(t.state != TrialState.COMPLETE for t in study.trials):
            logger.error("No successful trials completed.")
            return {}

        return self.save_optimization_results(
            study=study,
            all_files=files,
            sample_size=sample_size,
            fill_mode=fill_mode,
            alpha=alpha,
            use_penalties=use_penalties,
        )

    def save_optimization_results(
        self,
        study: optuna.Study,
        all_files: List[Path],
        sample_size: int,
        fill_mode: str,
        alpha: float,
        use_penalties: bool,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_params = study.best_params.copy()
        best_params["best_score"] = float(study.best_value)
        best_params["fill_mode"] = str(fill_mode)
        best_params["alpha"] = float(alpha)
        best_params["use_penalties"] = bool(use_penalties)

        best_file = self.results_dir / f"best_params_score_{timestamp}.json"
        with open(best_file, "w") as f:
            json.dump(best_params, f, indent=2)

        trials_df = study.trials_dataframe().sort_values("value", ascending=False)
        csv_file = self.results_dir / f"trials_score_{timestamp}.csv"
        trials_df.to_csv(csv_file, index=False)

        stats = {
            "best_value_score": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "failed_trials": len([t for t in study.trials if t.state == TrialState.FAIL]),
        }
        stats_file = self.results_dir / f"study_stats_score_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # Evaluate best on ALL files (and save per-file CSV)
        eval_params = DEFAULT_PARAMS.copy()
        eval_params.update(study.best_params)
        params_obj = StrategyParams(**eval_params)

        rows = []
        used = 0
        sum_ret = 0.0
        pos_cnt = 0
        total_trades = 0
        sum_abs_maxdd = 0.0

        # reference-only pooled PF (NOT optimized, just printed as info)
        total_gp = 0.0
        total_gl = 0.0

        for fp in all_files:
            df = self.load_data_chunk(fp, sample_size)
            if len(df) <= 100:
                continue
            r = self.backtest_single(df, params_obj, fill_mode=fill_mode)
            r["file"] = fp.name
            rows.append(r)

            tr = float(r.get("total_return", 0.0))
            sum_ret += tr
            if tr > 0.0:
                pos_cnt += 1
            used += 1

            total_trades += int(r.get("n_trades", 0))
            mdd_pct = float(r.get("max_drawdown_pct", 0.0))
            sum_abs_maxdd += abs(mdd_pct) / 100.0

            total_gp += float(r.get("gross_profit", 0.0))
            total_gl += float(r.get("gross_loss", 0.0))

        per_file_df = pd.DataFrame(rows)
        if not per_file_df.empty:
            per_file_df["positive_return"] = (per_file_df["total_return"] > 0.0).astype(int)
            per_file_df = per_file_df.sort_values(
                ["total_return", "positive_return", "n_trades"],
                ascending=[False, False, False],
            )

        per_file_csv = self.results_dir / f"per_file_results_score_{timestamp}.csv"
        per_file_df.to_csv(per_file_csv, index=False)

        if used > 0:
            avg_return_all = sum_ret / float(used)
            pct_pos_all = pos_cnt / float(used)
            base = max(1.0 + avg_return_all, 1e-6)
            score_all = base * (pct_pos_all ** float(alpha))

            penalty = 0.0
            if use_penalties:
                if total_trades < 3:
                    penalty += 3.0
                elif total_trades < 10:
                    penalty += 1.0
                score_all = score_all - (sum_abs_maxdd * 0.25) - penalty
        else:
            avg_return_all = 0.0
            pct_pos_all = 0.0
            score_all = 0.0
            penalty = 0.0

        pooled_pf_ref = (total_gp / total_gl) if total_gl > 0 else (10.0 if total_gp > 0 else 0.0)

        print("\n" + "=" * 90)
        print("BAYESIAN OPTIMIZATION RESULTS (Objective = (1+avg_return) * pct_pos^alpha)")
        print("REAL-RISK VERSION: exits use REAL close/high/low (signals still HA)")
        print("=" * 90)
        print(f"Best SCORE (study.best_value): {study.best_value:.6f}")
        print(f"Fill mode: {fill_mode}")
        print(f"Alpha: {alpha}")
        print(f"Use penalties: {use_penalties}")
        print(f"Completed Trials: {stats['completed_trials']} | Pruned: {stats['pruned_trials']} | Failed: {stats['failed_trials']}")
        print("\nBest Parameters Found:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        print("\n" + "-" * 90)
        print("OVERALL (ALL FILES) METRICS (BEST PARAMS)")
        print("-" * 90)
        print(f"Files used:       {used}")
        print(f"Avg Return:       {avg_return_all:.6f}  ({avg_return_all*100:.2f}%)")
        print(f"Pct Positive Ret: {pct_pos_all:.4f}  ({pos_cnt}/{used})")
        print(f"Alpha:            {alpha:.3f}")
        print(f"Sum abs MaxDD:    {sum_abs_maxdd:.4f}")
        print(f"Total Trades:     {total_trades}")
        if use_penalties:
            print(f"Penalty term:     {penalty:.3f}")
        print(f"SCORE (overall):  {score_all:.6f}")

        print("\nReference only (NOT optimized; shown for context):")
        print(f"Pooled PF:        {pooled_pf_ref:.6f}")
        print(f"Gross Profit:     {total_gp:,.2f}")
        print(f"Gross Loss:       {total_gl:,.2f}")

        print("\n" + "-" * 90)
        print("PER-FILE RESULTS (BEST PARAMS)")
        print("-" * 90)
        if per_file_df.empty:
            print("(No per-file results: not enough data / all files too short.)")
        else:
            show_cols = [
                "file",
                "total_return_pct",
                "positive_return",
                "n_trades",
                "max_drawdown_pct",
                "profit_factor",
                "final_equity",
            ]
            show_cols = [c for c in show_cols if c in per_file_df.columns]
            print(per_file_df[show_cols].to_string(index=False))

        print("\nSaved:")
        print(f"  Best params: {best_file}")
        print(f"  Trials CSV:  {csv_file}")
        print(f"  Per-file:    {per_file_csv}")
        print(f"  Stats:       {stats_file}")
        print("=" * 90)

        return {
            "best_params": best_params,
            "trials_df": trials_df,
            "per_file_df": per_file_df,
            "stats": stats,
            "study": study,
            "paths": {
                "best_params": str(best_file),
                "trials_csv": str(csv_file),
                "per_file_csv": str(per_file_csv),
                "stats_json": str(stats_file),
            },
        }

    def validate_best_params(
        self,
        best_params: Dict[str, Any],
        validation_files: Optional[List[Path]] = None,
        sample_size: int = 10000,
        fill_mode: str = "same_close",
        alpha: float = 0.5,
        use_penalties: bool = True,
    ) -> Dict[str, Any]:
        if validation_files is None:
            files = self.find_data_files()
            validation_files = files[-5:] if len(files) > 5 else files

        bp = dict(best_params)
        for k in ["best_score", "fill_mode", "alpha", "use_penalties"]:
            bp.pop(k, None)

        params = StrategyParams(**bp)

        metrics = self.run_backtest_hybrid_score(
            files=validation_files,
            sample_size=sample_size,
            fill_mode=fill_mode,
            alpha=alpha,
            use_penalties=use_penalties,
            **params.__dict__,
        )

        print("\nValidation (hybrid objective):")
        print(f"  Score:               {metrics['score']:.6f}")
        print(f"  Avg return:          {metrics['avg_return_overall']:.6f} ({metrics['avg_return_overall']*100:.2f}%)")
        print(f"  Pct positive return: {metrics['pct_positive_return']:.4f} ({metrics['positive_files']}/{metrics['n_files_used']})")
        print(f"  Total trades:        {metrics['total_trades']}")
        print(f"  Sum abs maxdd:       {metrics['sum_abs_maxdd']:.4f}")
        if use_penalties:
            print(f"  Penalty:             {metrics['penalty']:.3f}")

        df_out = pd.DataFrame(metrics.get("per_file", []))
        if not df_out.empty:
            df_out["positive_return"] = (df_out["total_return"] > 0.0).astype(int)
            df_out = df_out.sort_values(["total_return", "n_trades"], ascending=[False, False])
            show_cols = ["file", "total_return_pct", "positive_return", "n_trades", "max_drawdown_pct"]
            show_cols = [c for c in show_cols if c in df_out.columns]
            print("\nPer-file:")
            print(df_out[show_cols].to_string(index=False))

        return metrics

# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    def main():
        parser = argparse.ArgumentParser(
            description="Bayesian Optimization for adapt_half_RSI_hh (HYBRID score like half_RSI; 3 fill modes) [REAL-RISK exits]"
        )
        parser.add_argument("--mode", choices=["optimize", "validate", "test"], default="optimize",
                            help="Mode: optimize, validate, test")
        parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
        parser.add_argument("--sample", type=int, default=5000, help="Sample size per file")
        parser.add_argument("--files", type=int, default=3,
                            help="Number of files used per trial (random subsample); if >= total -> use all")
        parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers (-1 for auto)")
        parser.add_argument("--seed", type=int, default=42, help="RNG seed for file subsampling")
        parser.add_argument("--alpha", type=float, default=0.5, help="Exponent for pct_positive_return term")
        parser.add_argument("--no-penalties", action="store_true", help="Disable trade/DD penalties")
        parser.add_argument("--params-file", type=str, help="JSON file with parameters to validate/test")
        parser.add_argument("--test-file", type=str, help="Specific file to test")
        parser.add_argument("--fill", choices=["same_close", "next_open", "intrabar"], default="same_close",
                            help="Execution fill model (intrabar uses conservative stop fills)")
        parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
        parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")

        args = parser.parse_args()

        optimizer = BayesianAdaptHalfRSIOptimizer(data_dir=args.data_dir, results_dir=args.results_dir)

        use_penalties = (not args.no_penalties)

        if args.mode == "optimize":
            print("=" * 80)
            print("BAYESIAN OPTIMIZATION MODE (Objective = (1+avg_return) * pct_pos^alpha)")
            print("REAL-RISK VERSION: exits use REAL close/high/low (signals still HA)")
            print("=" * 80)
            print(f"Trials: {args.trials}")
            print(f"Sample size: {args.sample}")
            print(f"Files per trial: {args.files}")
            print(f"Fill mode: {args.fill}")
            print(f"Alpha: {args.alpha}")
            print(f"Penalties: {use_penalties}")
            print(f"Workers: {'Auto' if args.workers == -1 else args.workers}")
            print("=" * 80)

            optimizer.optimize_bayesian(
                n_trials=args.trials,
                n_files=args.files,
                sample_size=args.sample,
                n_jobs=args.workers,
                fill_mode=args.fill,
                alpha=args.alpha,
                seed=args.seed,
                use_penalties=use_penalties,
            )

        elif args.mode == "validate":
            if not args.params_file:
                print("Error: --params-file required for validate mode")
                return

            with open(args.params_file, "r") as f:
                params_data = json.load(f)

            # accept either {"best_params": {...}} or raw params dict
            if "best_params" in params_data:
                params = params_data["best_params"]
            else:
                params = params_data

            optimizer.validate_best_params(
                best_params=params,
                sample_size=args.sample,
                fill_mode=args.fill,
                alpha=args.alpha,
                use_penalties=use_penalties,
            )

        elif args.mode == "test":
            # pick file
            if args.test_file:
                test_fp = Path(args.test_file)
            else:
                files = optimizer.find_data_files()
                if not files:
                    print("No data files found.")
                    return
                test_fp = files[0]

            # pick params
            if args.params_file:
                with open(args.params_file, "r") as f:
                    params_data = json.load(f)
                if "best_params" in params_data:
                    params_data = params_data["best_params"]
                for k in ["best_score", "fill_mode", "alpha", "use_penalties"]:
                    params_data.pop(k, None)
                params = StrategyParams(**params_data)
            else:
                params = StrategyParams(**DEFAULT_PARAMS)

            df = optimizer.load_data_chunk(test_fp, sample_size=args.sample)
            if len(df) <= 100:
                print(f"Not enough data in {test_fp} ({len(df)} rows).")
                return

            r = optimizer.backtest_single(df, params, fill_mode=args.fill)

            print("\n" + "=" * 80)
            print("SINGLE FILE TEST")
            print("REAL-RISK VERSION: exits use REAL close/high/low (signals still HA)")
            print("=" * 80)
            print(f"File: {test_fp.name}")
            print(f"Fill mode: {args.fill}")
            print("-" * 80)
            print(f"Return:        {r['total_return_pct']:.2f}%")
            print(f"Trades:        {r['n_trades']}")
            print(f"Max DD:        {r['max_drawdown_pct']:.2f}%")
            print(f"Sharpe:        {r['sharpe_ratio']:.3f}")
            print(f"Final Equity:  {r['final_equity']:,.2f}")
            print(f"(Reference) PF:{r['profit_factor']:.6f}")
            print("=" * 80)

    main()
