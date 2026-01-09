#!/usr/bin/env python3
"""
Bayes_opt_adapt_RSI.py  (modified)

Changes requested:
1) Optimize OVERALL profit_factor (pooled GrossProfit / pooled GrossLoss) instead of Sharpe.
2) Print results for EACH file (and save a per-file summary CSV) after optimization.

Notes (matches the Dynamic_SR "Option A" idea):
- Overall PF is computed by pooling realized trade PnL across the selected files:
    PF_overall = (sum GrossProfit) / (sum GrossLoss)
  where GrossProfit/GrossLoss are based on realized trade PnL NET of commissions.
- If GrossLoss == 0:
    PF = pf_cap if GrossProfit > 0 else 0
- You can cap PF to avoid infinite PF dominating.

Execution model:
- Signals are generated on bar i (based on HA logic inside your indicator pipeline)
- Fills can be:
    --fill same_close : enter/exit at REAL close[i]
    --fill next_open  : enter/exit at REAL open[i+1]
- Equity is marked to REAL close.

Files:
- Reads .parquet and .csv from ./data (recursively).
- Outputs to ./results

CLI examples:
  python Bayes_opt_adapt_RSI.py --mode optimize --trials 200 --files 5 --sample 5000 --fill same_close
  python Bayes_opt_adapt_RSI.py --mode validate --params-file results/best_params_....json --sample 10000 --fill next_open
"""

import os
import json
import hashlib
import logging
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

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
# Defaults (your existing)
# =========================

DEFAULT_PARAMS = {
    "sl_multiplier": 3.1,
    "tp_multiplier": 3.97,
    "trail_multiplier": 1.9,
    "channel_multi": 2.63,
    "channel_length": 46,
    "atr_period": 20,
    "base_period": 20,
    "min_period": 15,
    "max_period": 23,
    "fast_period": 10,
    "slow_period": 48,
    "steepness": 2,
    "fast_steepness": 10,
    "slow_steepness": 3,
    "sigmoid_margin": 0.0,
    "median_length": 13,
    "trim_percent": 0.2,
    "use_replacement": True,
}

# =========================
# Numba indicators (yours)
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
    if n < period:
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

    if avg_loss == 0:
        rsi[period - 1] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period - 1] = 100 - (100 / (1 + rs))

    for i in range(period, n):
        if deltas[i] > 0:
            gain[i] = deltas[i]
            loss[i] = 0.0
        else:
            gain[i] = 0.0
            loss[i] = -deltas[i]

        avg_gain = ((avg_gain * (period - 1)) + gain[i]) / period
        avg_loss = ((avg_loss * (period - 1)) + loss[i]) / period

        if avg_loss == 0:
            rsi[i] = 100
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100 - (100 / (1 + rs))

    return rsi


@jit(nopython=True, cache=True)
def sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.full(n, np.nan)
    if n < period:
        return out

    out[period - 1] = np.sum(data[:period]) / period
    for i in range(period, n):
        out[i] = out[i - 1] + (data[i] - data[i - period]) / period
    return out


@jit(nopython=True, cache=True)
def ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    out = np.full(n, np.nan)

    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break
    if start_idx == -1 or (n - start_idx) < period:
        return out

    alpha = 2.0 / (period + 1.0)
    s = 0.0
    for i in range(start_idx, start_idx + period):
        s += data[i]
    out[start_idx + period - 1] = s / period

    for i in range(start_idx + period, n):
        out[i] = (data[i] - out[i - 1]) * alpha + out[i - 1]
    return out


@jit(nopython=True, cache=True)
def calculate_all_rsi_numba(close: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    n = len(close)
    num_periods = max_period - min_period + 1
    all_rsi = np.full((n, num_periods), np.nan)
    for p_idx in prange(num_periods):
        period = min_period + p_idx
        all_rsi[:, p_idx] = rsi_numba(close, period)
    return all_rsi


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
def ha_true_range_numba(ha_high: np.ndarray, ha_low: np.ndarray, ha_close: np.ndarray) -> np.ndarray:
    n = len(ha_close)
    tr = np.zeros(n)
    tr[0] = ha_high[0] - ha_low[0]
    for i in range(1, n):
        hl = ha_high[i] - ha_low[i]
        hc = abs(ha_high[i] - ha_close[i - 1])
        lc = abs(ha_low[i] - ha_close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


# =========================
# NEW: simulations that also return gross profit/loss
# =========================

@jit(nopython=True, cache=True)
def run_simulation_real_same_close_pf_numba(
    close_arr,
    buy_signal,
    sell_signal,
    initial_capital,
    pos_size_pct,
    comm_pct
):
    """
    Fill at REAL close[i].
    Tracks realized trade PnL net of commissions to compute GrossProfit/GrossLoss.
    """
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
        price = close_arr[i]
        if np.isnan(price) or price <= 0:
            equity_curve[i] = cash + (qty * price if qty > 0 else 0.0)
            continue

        # ENTRY
        if buy_signal[i] and qty == 0.0:
            pos_value = cash * pos_size_pct
            if pos_value > 0:
                qty = pos_value / price
                entry_price = price
                entry_comm = pos_value * comm_pct
                cash -= (pos_value + entry_comm)

        # EXIT
        elif sell_signal[i] and qty > 0.0:
            exit_value = qty * price
            exit_comm = exit_value * comm_pct

            pnl = (price - entry_price) * qty - entry_comm - exit_comm
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += -pnl

            cash += (exit_value - exit_comm)
            trade_count += 1

            qty = 0.0
            entry_price = 0.0
            entry_comm = 0.0

        equity_curve[i] = cash + (qty * price if qty > 0 else 0.0)

    # If still open, close at last close (common backtest convention)
    if qty > 0.0:
        price = close_arr[-1]
        if not np.isnan(price) and price > 0:
            exit_value = qty * price
            exit_comm = exit_value * comm_pct
            pnl = (price - entry_price) * qty - entry_comm - exit_comm
            if pnl > 0:
                gross_profit += pnl
            elif pnl < 0:
                gross_loss += -pnl
            cash += (exit_value - exit_comm)
            trade_count += 1
        qty = 0.0
        equity_curve[-1] = cash

    return equity_curve, trade_count, gross_profit, gross_loss


@jit(nopython=True, cache=True)
def run_simulation_real_next_open_pf_numba(
    open_arr,
    close_arr,
    buy_signal,
    sell_signal,
    initial_capital,
    pos_size_pct,
    comm_pct
):
    """
    Fill at REAL open[i+1], signals on i. Mark to REAL close.
    Tracks realized trade PnL net of commissions to compute GrossProfit/GrossLoss.
    """
    n = len(close_arr)
    equity_curve = np.zeros(n)

    cash = initial_capital
    qty = 0.0
    entry_price = 0.0
    entry_comm = 0.0

    trade_count = 0
    gross_profit = 0.0
    gross_loss = 0.0

    for i in range(n - 1):
        mark_price = close_arr[i]
        if np.isnan(mark_price) or mark_price <= 0:
            equity_curve[i] = cash + (qty * mark_price if qty > 0 else 0.0)
            continue

        # ENTRY at next open
        if buy_signal[i] and qty == 0.0:
            fill = open_arr[i + 1]
            if not np.isnan(fill) and fill > 0:
                pos_value = cash * pos_size_pct
                if pos_value > 0:
                    qty = pos_value / fill
                    entry_price = fill
                    entry_comm = pos_value * comm_pct
                    cash -= (pos_value + entry_comm)

        # EXIT at next open
        elif sell_signal[i] and qty > 0.0:
            fill = open_arr[i + 1]
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
                entry_price = 0.0
                entry_comm = 0.0

        equity_curve[i] = cash + (qty * mark_price if qty > 0 else 0.0)

    # last mark
    last_mark = close_arr[-1]
    equity_curve[-1] = cash + (qty * last_mark if qty > 0 and not np.isnan(last_mark) else 0.0)

    # If still open, close at last close
    if qty > 0.0 and not np.isnan(last_mark) and last_mark > 0:
        exit_value = qty * last_mark
        exit_comm = exit_value * comm_pct
        pnl = (last_mark - entry_price) * qty - entry_comm - exit_comm
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
    sl_multiplier: float = 3.3
    tp_multiplier: float = 3.755
    trail_multiplier: float = 2.69
    channel_multi: float = 1.906
    channel_length: int = 40
    atr_period: int = 20
    base_period: int = 15

    min_period: int = 10
    max_period: int = 16
    fast_period: int = 10
    slow_period: int = 36
    steepness: float = 5
    fast_steepness: float = 8
    slow_steepness: float = 5

    sigmoid_margin: float = 0.0
    median_length: int = 13
    trim_percent: float = 0.2
    use_replacement: bool = True

    initial_capital: float = 100000.0
    position_size_pct: float = 0.1
    commission_pct: float = 0.0006

    def to_hash(self) -> str:
        params_str = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()


class BayesianRSIATROptimizer:
    """
    Bayesian optimization for RSI ATR strategy.

    MODIFIED:
    - Objective is PF_overall (pooled GP/GL) over the selected files.
    - Prints per-file results after optimization.
    - Supports .parquet and .csv.
    - Supports --fill same_close|next_open.
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

        required_cols = ["open", "high", "low", "close"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = np.nan

        # volume optional
        if "volume" not in df.columns:
            df["volume"] = np.nan

        if sample_size and sample_size > 0 and sample_size < len(df):
            df = df.tail(sample_size)

        df = df.reset_index(drop=True)
        return df

    def find_data_files(self) -> List[Path]:
        files = list(self.data_dir.rglob("*.parquet")) + list(self.data_dir.rglob("*.csv"))
        files.sort()
        logger.info(f"Found {len(files)} data files in {self.data_dir}")
        return files

    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_olympian_mean(data: np.ndarray, length: int, trim_percent: float, replace: bool) -> np.ndarray:
        n = len(data)
        out = np.full(n, np.nan)

        if length <= 0:
            return out

        for i in range(length - 1, n):
            window = np.empty(length, dtype=np.float64)
            for k in range(length):
                v = data[i - k]
                window[k] = 0.0 if np.isnan(v) else v

            window.sort()

            trim = int(np.round(length * trim_percent))
            if trim < 1:
                trim = 1

            start_idx = trim
            end_idx = length - trim - 1
            count = end_idx - start_idx + 1

            if count > 0:
                if replace:
                    low_rep = window[start_idx]
                    high_rep = window[end_idx]
                    s = 0.0
                    for _ in range(trim):
                        s += low_rep
                    for j in range(start_idx, end_idx + 1):
                        s += window[j]
                    for _ in range(trim):
                        s += high_rep
                    out[i] = s / length
                else:
                    s = 0.0
                    for j in range(start_idx, end_idx + 1):
                        s += window[j]
                    out[i] = s / count
            else:
                mid = length // 2
                if length % 2 == 0:
                    out[i] = 0.5 * (window[mid - 1] + window[mid])
                else:
                    out[i] = window[mid]

        return out

    def calculate_indicators_fast(self, df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        df = df.copy()

        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi_numba(
            df["open"].values.astype(np.float64),
            df["high"].values.astype(np.float64),
            df["low"].values.astype(np.float64),
            df["close"].values.astype(np.float64),
        )
        df["ha_open"], df["ha_high"], df["ha_low"], df["ha_close"] = ha_o, ha_h, ha_l, ha_c
        df["ha_hl2"] = (df["ha_high"].values + df["ha_low"].values) / 2.0

        ha_tr = ha_true_range_numba(df["ha_high"].values, df["ha_low"].values, df["ha_close"].values)

        # Channel ATR: haATR(channelLength)
        df["atr_chan"] = rma_numba(ha_tr, params.channel_length)

        # Channel mid/upper/lower
        df["mid_channel"] = sma_numba(df["ha_hl2"].values, params.channel_length)
        df["lower_channel"] = df["mid_channel"].values - df["atr_chan"].values * params.channel_multi
        df["upper_channel"] = df["mid_channel"].values + df["atr_chan"].values * params.channel_multi

        lower_tf_len = max(1, params.channel_length // 2)
        df["mid_channel_lower_tf"] = sma_numba(df["ha_hl2"].values, lower_tf_len)

        # Adaptive RSI on ha_close
        df["base_rsi"] = rsi_numba(df["ha_close"].values, params.base_period)
        scaled = (100.0 - df["base_rsi"].values) / 100.0

        adaptive_period_raw = np.round(scaled * (params.max_period - params.min_period) + params.min_period)
        adaptive_period = adaptive_period_raw.copy()
        adaptive_period[np.isnan(adaptive_period)] = params.min_period
        adaptive_period = np.clip(adaptive_period, params.min_period, params.max_period).astype(np.int32)
        df["adaptive_period"] = adaptive_period

        min_switch = 3
        max_switch = 34
        all_rsi = calculate_all_rsi_numba(df["ha_close"].values, min_switch, max_switch)

        adaptive_rsi = np.full(len(df), np.nan)
        for i in range(len(df)):
            p = adaptive_period[i]
            if min_switch <= p <= max_switch:
                adaptive_rsi[i] = all_rsi[i, p - min_switch]
            else:
                adaptive_rsi[i] = np.nan
        df["adaptive_rsi"] = adaptive_rsi

        df["fast_rsi"] = ema_numba(df["adaptive_rsi"].values, params.fast_period)
        df["slow_rsi"] = ema_numba(df["adaptive_rsi"].values, params.slow_period)

        # sigmoid plots (not used for entries)
        def sigmoid_np(x01, steep, margin=0.0):
            raw = 1.0 / (1.0 + np.exp(-steep * (x01 - 0.5)))
            return (margin + (1.0 - 2.0 * margin) * raw)

        ar = df["adaptive_rsi"].values
        ar01 = ar / 100.0
        fast_sigmoidRSI = sigmoid_np(ar01, params.fast_steepness, 0.0) * 100.0
        slow_sigmoidRSI = sigmoid_np(ar01, params.slow_steepness, params.sigmoid_margin) * 100.0

        df["fast_sigmoid"] = ema_numba(fast_sigmoidRSI, params.fast_period)
        df["slow_sigmoid"] = ema_numba(slow_sigmoidRSI, params.slow_period)

        df["fast_sigmoid_mid"] = self.calculate_olympian_mean(
            df["fast_sigmoid"].values, params.median_length, params.trim_percent, params.use_replacement
        )
        df["slow_sigmoid_mid"] = self.calculate_olympian_mean(
            df["slow_sigmoid"].values, params.median_length, params.trim_percent, params.use_replacement
        )

        # Risk ATR on HA TR via RMA (your current requirement)
        atr_ha = rma_numba(ha_tr, params.atr_period)
        df["trail_atr"] = atr_ha
        df["trail_offset"] = df["trail_atr"].values * params.trail_multiplier

        return df

    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_signals_pine_like_numba(
        sig_close,            # ha_close
        sig_low,              # ha_low
        mid_channel,
        mid_channel_lower_tf,
        lower_channel,
        adaptive_rsi,
        slow_rsi,
        trail_offset,
        min_bars=10
    ):
        n = len(sig_close)
        buy = np.zeros(n, dtype=np.bool_)
        sell = np.zeros(n, dtype=np.bool_)

        in_pos = False
        support = 0.0

        prev_ad = np.nan
        prev_sl = np.nan
        prev_sig_close = np.nan
        prev_lower_channel = np.nan
        prev_risk_close = np.nan
        prev_support = np.nan

        for i in range(n):
            if i < min_bars:
                prev_ad = adaptive_rsi[i]
                prev_sl = slow_rsi[i]
                prev_sig_close = sig_close[i]
                prev_lower_channel = lower_channel[i]
                prev_risk_close = sig_close[i]
                prev_support = support
                continue

            if (np.isnan(adaptive_rsi[i]) or np.isnan(slow_rsi[i]) or
                np.isnan(mid_channel[i]) or np.isnan(mid_channel_lower_tf[i]) or
                np.isnan(lower_channel[i]) or np.isnan(trail_offset[i])):
                prev_ad = adaptive_rsi[i]
                prev_sl = slow_rsi[i]
                prev_sig_close = sig_close[i]
                prev_lower_channel = lower_channel[i]
                prev_risk_close = sig_close[i]
                prev_support = support
                continue

            uptrend = sig_close[i] > mid_channel[i]
            above_lowerTF = sig_close[i] > mid_channel_lower_tf[i]
            trend_ok = uptrend or above_lowerTF

            crossover = False
            crossunder_rsi = False
            if not np.isnan(prev_ad) and not np.isnan(prev_sl):
                crossover = (adaptive_rsi[i] > slow_rsi[i]) and (prev_ad <= prev_sl)
                crossunder_rsi = (adaptive_rsi[i] < slow_rsi[i]) and (prev_ad >= prev_sl)

            crossunder_channel = False
            if not np.isnan(prev_sig_close) and not np.isnan(prev_lower_channel):
                crossunder_channel = (sig_close[i] < lower_channel[i]) and (prev_sig_close >= prev_lower_channel)

            long_condition = crossover and trend_ok
            short_condition = crossunder_rsi or crossunder_channel

            risk_close = sig_close[i]
            risk_low = sig_low[i]

            if not in_pos:
                if long_condition:
                    buy[i] = True
                    in_pos = True
                    support = risk_low - trail_offset[i]
            else:
                cur_stop = risk_low - trail_offset[i]
                if cur_stop > support:
                    support = cur_stop

                crossunder_support = False
                if not np.isnan(prev_risk_close):
                    crossunder_support = (risk_close < support) and (prev_risk_close >= prev_support)

                if crossunder_support or short_condition:
                    sell[i] = True
                    in_pos = False

            prev_ad = adaptive_rsi[i]
            prev_sl = slow_rsi[i]
            prev_sig_close = sig_close[i]
            prev_lower_channel = lower_channel[i]
            prev_risk_close = risk_close
            prev_support = support

        return buy, sell

    def backtest_single(self, df: pd.DataFrame, params: StrategyParams, fill_mode: str = "same_close") -> Dict[str, Any]:
        """
        Returns per-file metrics including gross_profit/gross_loss/profit_factor (NEW).
        """
        try:
            df = self.calculate_indicators_fast(df, params)

            min_bars = max(
                params.max_period,
                params.slow_period,
                params.channel_length,
                params.atr_period
            ) + 2

            buy_signal, sell_signal = self.calculate_signals_pine_like_numba(
                df["ha_close"].values,
                df["ha_low"].values,
                df["mid_channel"].values,
                df["mid_channel_lower_tf"].values,
                df["lower_channel"].values,
                df["adaptive_rsi"].values,
                df["slow_rsi"].values,
                df["trail_offset"].values,
                min_bars=min_bars,
            )

            if fill_mode == "next_open":
                equity_curve, n_trades, gp, gl = run_simulation_real_next_open_pf_numba(
                    df["open"].values.astype(np.float64),
                    df["close"].values.astype(np.float64),
                    buy_signal,
                    sell_signal,
                    params.initial_capital,
                    params.position_size_pct,
                    params.commission_pct,
                )
            else:
                equity_curve, n_trades, gp, gl = run_simulation_real_same_close_pf_numba(
                    df["close"].values.astype(np.float64),
                    buy_signal,
                    sell_signal,
                    params.initial_capital,
                    params.position_size_pct,
                    params.commission_pct,
                )

            final_equity = float(equity_curve[-1])
            total_ret_pct = ((final_equity - params.initial_capital) / params.initial_capital) * 100.0

            # Sharpe (still computed for reporting)
            if len(equity_curve) > 2:
                returns = np.diff(equity_curve) / np.maximum(equity_curve[:-1], 1e-12)
                std = np.std(returns)
                sharpe = float((np.mean(returns) / std) * np.sqrt(252.0)) if std > 0 else 0.0
            else:
                sharpe = 0.0

            # Max drawdown (%)
            cum_max = np.maximum.accumulate(equity_curve)
            dd = (equity_curve - cum_max) / np.maximum(cum_max, 1e-12)
            max_dd_pct = float(np.min(dd) * 100.0)

            gp = float(gp)
            gl = float(gl)
            if gl > 0:
                pf = gp / gl
            else:
                pf = 10.0 if gp > 0 else 0.0

            return {
                "total_return_pct": float(total_ret_pct),
                "sharpe_ratio": float(sharpe),
                "max_drawdown_pct": float(max_dd_pct),
                "n_trades": int(n_trades),
                "final_equity": float(final_equity),
                "gross_profit": float(gp),
                "gross_loss": float(gl),
                "profit_factor": float(pf),
            }

        except Exception as e:
            logger.error(f"backtest_single error: {e}")
            return {
                "total_return_pct": -100.0,
                "sharpe_ratio": 0.0,
                "max_drawdown_pct": -100.0,
                "n_trades": 0,
                "final_equity": float(params.initial_capital),
                "gross_profit": 0.0,
                "gross_loss": 0.0,
                "profit_factor": 0.0,
            }

    def run_backtest(
        self,
        files: List[Path],
        sample_size: int,
        fill_mode: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Runs backtest across multiple files.
        Aggregation is now "pooled PF" (sum GP / sum GL) (NEW).
        """
        params = StrategyParams(**kwargs)

        total_gp = 0.0
        total_gl = 0.0
        total_trades = 0
        per_file: List[Dict[str, Any]] = []

        for fp in files:
            df = self.load_data_chunk(fp, sample_size)
            if len(df) <= 100:
                continue
            res = self.backtest_single(df, params, fill_mode=fill_mode)
            res["file"] = fp.name
            per_file.append(res)

            total_gp += float(res.get("gross_profit", 0.0))
            total_gl += float(res.get("gross_loss", 0.0))
            total_trades += int(res.get("n_trades", 0))

        if total_gl > 0:
            pf_overall = total_gp / total_gl
        else:
            pf_overall = 10.0 if total_gp > 0 else 0.0

        return {
            "pf_overall": float(pf_overall),
            "total_gross_profit": float(total_gp),
            "total_gross_loss": float(total_gl),
            "total_trades": int(total_trades),
            "n_files_used": int(len(per_file)),
            "per_file": per_file,
        }

    def objective(
        self,
        trial: optuna.Trial,
        files: List[Path],
        sample_size: int,
        n_files: int,
        fill_mode: str,
        pf_cap: float,
    ) -> float:
        # Start from defaults
        params = DEFAULT_PARAMS.copy()

        # Parameter ranges (your existing approach)
        param_ranges = {
            "sl_multiplier": (2.0, 4.5),
            "tp_multiplier": (3.0, 6.0),
            "trail_multiplier": (1.5, 4.0),
            "channel_multi": (2.0, 4.0),
            "channel_length": (30, 80),
            "atr_period": (10, 50),
            "base_period": (5, 20),
            "min_period": (1, 10),
            "max_period": (11, 25),
            "fast_period": (3, 10),
            "slow_period": (25, 50),
            "steepness": (2, 10),
            "fast_steepness": (5, 15),
            "slow_steepness": (1, 5),
            "median_length": (7, 20),
        }

        for name, (low, high) in param_ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)

        # hard constraints
        if params["fast_period"] >= params["slow_period"] or params["fast_steepness"] <= params["slow_steepness"]:
            return 0.0  # bad PF

        if params["min_period"] >= params["base_period"] or params["max_period"] <= params["base_period"]:
            return 0.0

        # Force a known-ish working init for trial 0
        if trial.number == 0:
            params.update({
                "trail_multiplier": 2.0,
                "channel_multi": 2.5,
                "min_period": 5,
                "max_period": 15,
                "fast_period": 10,
                "slow_period": 30,
            })

        # pick files for this trial (first n_files)
        n_files = min(max(1, int(n_files)), len(files))
        trial_files = files[:n_files]

        metrics = self.run_backtest(
            files=trial_files,
            sample_size=sample_size,
            fill_mode=fill_mode,
            **params
        )

        total_trades = int(metrics.get("total_trades", 0))
        if total_trades == 0:
            return 0.0

        pf = float(metrics.get("pf_overall", 0.0))
        pf = float(min(pf, float(pf_cap)))

        # attrs for inspection
        trial.set_user_attr("pf_overall", float(pf))
        trial.set_user_attr("total_gross_profit", float(metrics.get("total_gross_profit", 0.0)))
        trial.set_user_attr("total_gross_loss", float(metrics.get("total_gross_loss", 0.0)))
        trial.set_user_attr("total_trades", int(total_trades))
        trial.set_user_attr("n_files_used", int(metrics.get("n_files_used", 0)))
        trial.set_user_attr("fill_mode", str(fill_mode))

        return float(pf)

    def optimize_bayesian(
        self,
        n_trials: int = 100,
        n_files: int = 5,
        sample_size: int = 5000,
        n_jobs: int = -1,
        fill_mode: str = "same_close",
        pf_cap: float = 10.0,
    ) -> Dict[str, Any]:
        files = self.find_data_files()
        if len(files) == 0:
            logger.error("No data files found in data directory")
            return {}

        study_name = f"rsi_atr_pf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = f"sqlite:///{self.results_dir}/{study_name}.db"

        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            direction="maximize",
            load_if_exists=True,
        )
        self.study = study

        logger.info(f"Starting optimization (PF objective) with {n_trials} trials")

        def objective_with_args(trial):
            return self.objective(
                trial=trial,
                files=files,
                sample_size=sample_size,
                n_files=n_files,
                fill_mode=fill_mode,
                pf_cap=pf_cap,
            )

        study.optimize(
            objective_with_args,
            n_trials=n_trials,
            n_jobs=n_jobs if n_jobs > 0 else os.cpu_count(),
            show_progress_bar=True,
            gc_after_trial=True,
        )

        if not study.trials or all(t.state != TrialState.COMPLETE for t in study.trials):
            logger.error("No successful trials completed.")
            return {}

        results = self.save_optimization_results(
            study=study,
            all_files=files,
            sample_size=sample_size,
            fill_mode=fill_mode,
            pf_cap=pf_cap,
        )
        return results

    def save_optimization_results(
        self,
        study: optuna.Study,
        all_files: List[Path],
        sample_size: int,
        fill_mode: str,
        pf_cap: float,
    ) -> Dict[str, Any]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        best_params = study.best_params.copy()
        best_params["score_pf"] = float(study.best_value)
        best_params["fill_mode"] = fill_mode
        best_params["pf_cap"] = float(pf_cap)

        best_file = self.results_dir / f"best_params_pf_{timestamp}.json"
        with open(best_file, "w") as f:
            json.dump(best_params, f, indent=2)

        trials_df = study.trials_dataframe()
        trials_df = trials_df.sort_values("value", ascending=False)
        csv_file = self.results_dir / f"trials_pf_{timestamp}.csv"
        trials_df.to_csv(csv_file, index=False)

        stats = {
            "best_value_pf": float(study.best_value),
            "best_params": study.best_params,
            "n_trials": len(study.trials),
            "completed_trials": len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            "pruned_trials": len([t for t in study.trials if t.state == TrialState.PRUNED]),
            "failed_trials": len([t for t in study.trials if t.state == TrialState.FAIL]),
        }
        stats_file = self.results_dir / f"study_stats_pf_{timestamp}.json"
        with open(stats_file, "w") as f:
            json.dump(stats, f, indent=2)

        # ---- Evaluate BEST on EACH FILE and print results (REQUEST #2) ----
        eval_params = DEFAULT_PARAMS.copy()
        eval_params.update(study.best_params)

        params_obj = StrategyParams(**eval_params)

        rows = []
        total_gp = 0.0
        total_gl = 0.0
        total_trades = 0

        for fp in all_files:
            df = self.load_data_chunk(fp, sample_size)
            if len(df) <= 100:
                continue
            r = self.backtest_single(df, params_obj, fill_mode=fill_mode)
            r["file"] = fp.name
            rows.append(r)

            total_gp += float(r.get("gross_profit", 0.0))
            total_gl += float(r.get("gross_loss", 0.0))
            total_trades += int(r.get("n_trades", 0))

        if total_gl > 0:
            pf_all = total_gp / total_gl
        else:
            pf_all = pf_cap if total_gp > 0 else 0.0
        pf_all = float(min(pf_all, float(pf_cap)))

        per_file_df = pd.DataFrame(rows)
        if not per_file_df.empty:
            per_file_df = per_file_df.sort_values(["profit_factor", "n_trades"], ascending=[False, False])

        per_file_csv = self.results_dir / f"per_file_results_pf_{timestamp}.csv"
        per_file_df.to_csv(per_file_csv, index=False)

        print("\n" + "=" * 90)
        print("BAYESIAN OPTIMIZATION RESULTS (Objective = OVERALL PROFIT FACTOR)")
        print("=" * 90)
        print(f"Best PF Score (capped): {study.best_value:.6f}")
        print(f"Fill mode: {fill_mode}")
        print(f"PF cap: {pf_cap}")
        print(f"Completed Trials: {stats['completed_trials']} | Pruned: {stats['pruned_trials']} | Failed: {stats['failed_trials']}")
        print("\nBest Parameters Found:")
        for k, v in study.best_params.items():
            print(f"  {k}: {v}")

        print("\n" + "-" * 90)
        print("OVERALL (ALL FILES) POOLED PROFIT FACTOR")
        print("-" * 90)
        print(f"Gross Profit: {total_gp:,.2f}")
        print(f"Gross Loss:   {total_gl:,.2f}")
        print(f"PF Overall:   {pf_all:.6f}")
        print(f"Total Trades: {total_trades}")

        print("\n" + "-" * 90)
        print("PER-FILE RESULTS (BEST PARAMS)")
        print("-" * 90)
        if per_file_df.empty:
            print("(No per-file results: not enough data / all files too short.)")
        else:
            # print a readable subset of columns
            show_cols = [
                "file",
                "profit_factor",
                "gross_profit",
                "gross_loss",
                "n_trades",
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe_ratio",
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
        pf_cap: float = 10.0,
    ) -> Dict[str, Any]:
        if validation_files is None:
            files = self.find_data_files()
            validation_files = files[-5:] if len(files) > 5 else files

        params = StrategyParams(**best_params)

        rows = []
        total_gp = 0.0
        total_gl = 0.0
        total_trades = 0

        print(f"\nValidating on {len(validation_files)} files (fill={fill_mode}) ...")
        for fp in validation_files:
            df = self.load_data_chunk(fp, sample_size)
            if len(df) <= 100:
                continue
            r = self.backtest_single(df, params, fill_mode=fill_mode)
            r["file"] = fp.name
            rows.append(r)

            total_gp += float(r.get("gross_profit", 0.0))
            total_gl += float(r.get("gross_loss", 0.0))
            total_trades += int(r.get("n_trades", 0))

        if total_gl > 0:
            pf = total_gp / total_gl
        else:
            pf = pf_cap if total_gp > 0 else 0.0
        pf = float(min(pf, float(pf_cap)))

        df_out = pd.DataFrame(rows).sort_values(["profit_factor", "n_trades"], ascending=[False, False]) if rows else pd.DataFrame()

        print("\nValidation pooled PF:")
        print(f"  Gross Profit: {total_gp:,.2f}")
        print(f"  Gross Loss:   {total_gl:,.2f}")
        print(f"  PF Overall:   {pf:.6f}")
        print(f"  Total Trades: {total_trades}")

        if not df_out.empty:
            show_cols = ["file", "profit_factor", "gross_profit", "gross_loss", "n_trades", "total_return_pct", "max_drawdown_pct"]
            show_cols = [c for c in show_cols if c in df_out.columns]
            print("\nPer-file:")
            print(df_out[show_cols].to_string(index=False))

        return {
            "pooled_pf": pf,
            "total_gross_profit": total_gp,
            "total_gross_loss": total_gl,
            "total_trades": total_trades,
            "per_file": df_out,
        }


# =========================
# CLI
# =========================

if __name__ == "__main__":
    import argparse

    def main():
        parser = argparse.ArgumentParser(description="Bayesian Optimization for RSI ATR Strategy (PF objective)")
        parser.add_argument("--mode", choices=["optimize", "validate", "test"], default="optimize",
                            help="Mode: optimize, validate, test")
        parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
        parser.add_argument("--sample", type=int, default=5000, help="Sample size per file")
        parser.add_argument("--files", type=int, default=3, help="Number of files to use per trial (first N files)")
        parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers (-1 for auto)")
        parser.add_argument("--params-file", type=str, help="JSON file with parameters to validate/test")
        parser.add_argument("--test-file", type=str, help="Specific file to test")
        parser.add_argument("--fill", choices=["same_close", "next_open"], default="same_close",
                            help="Execution fill model")
        parser.add_argument("--pf-cap", type=float, default=10.0,
                            help="Cap PF to avoid infinite PF dominating (default 10.0)")
        parser.add_argument("--data-dir", type=str, default="./data", help="Data directory")
        parser.add_argument("--results-dir", type=str, default="./results", help="Results directory")

        args = parser.parse_args()

        optimizer = BayesianRSIATROptimizer(data_dir=args.data_dir, results_dir=args.results_dir)

        if args.mode == "optimize":
            print("=" * 80)
            print("BAYESIAN OPTIMIZATION MODE (Objective = OVERALL PROFIT FACTOR)")
            print("=" * 80)
            print(f"Trials: {args.trials}")
            print(f"Sample size: {args.sample}")
            print(f"Files per trial: {args.files}")
            print(f"Fill mode: {args.fill}")
            print(f"PF cap: {args.pf_cap}")
            print(f"Workers: {'Auto' if args.workers == -1 else args.workers}")
            print("=" * 80)

            optimizer.optimize_bayesian(
                n_trials=args.trials,
                n_files=args.files,
                sample_size=args.sample,
                n_jobs=args.workers,
                fill_mode=args.fill,
                pf_cap=args.pf_cap,
            )

        elif args.mode == "validate":
            if not args.params_file:
                print("Error: --params-file required for validate mode")
                return

            with open(args.params_file, "r") as f:
                params_data = json.load(f)

            # allow either full dict or wrapper dict
            if "best_params" in params_data:
                params = params_data["best_params"]
            else:
                params = params_data

            # remove non-StrategyParams keys if present
            for k in ["score_pf", "fill_mode", "pf_cap"]:
                if k in params:
                    del params[k]

            optimizer.validate_best_params(
                best_params=params,
                sample_size=args.sample,
                fill_mode=args.fill,
                pf_cap=args.pf_cap,
            )

        elif args.mode == "test":
            # pick test file
            if args.test_file:
                test_fp = Path(args.test_file)
            else:
                files = optimizer.find_data_files()
                if not files:
                    print("No data files found.")
                    return
                test_fp = files[0]

            # load params
            if args.params_file:
                with open(args.params_file, "r") as f:
                    params_data = json.load(f)
                if "best_params" in params_data:
                    params_data = params_data["best_params"]
                for k in ["score_pf", "fill_mode", "pf_cap"]:
                    params_data.pop(k, None)
                params = StrategyParams(**params_data)
            else:
                # use defaults
                base = DEFAULT_PARAMS.copy()
                params = StrategyParams(**base)

            df = optimizer.load_data_chunk(test_fp, sample_size=args.sample)
            if len(df) <= 100:
                print(f"Not enough data in {test_fp} ({len(df)} rows).")
                return

            r = optimizer.backtest_single(df, params, fill_mode=args.fill)

            print("\n" + "=" * 80)
            print("SINGLE FILE TEST")
            print("=" * 80)
            print(f"File: {test_fp.name}")
            print(f"Fill mode: {args.fill}")
            print("-" * 80)
            print(f"Profit Factor: {r['profit_factor']:.6f}")
            print(f"Gross Profit:  {r['gross_profit']:,.2f}")
            print(f"Gross Loss:    {r['gross_loss']:,.2f}")
            print(f"Trades:        {r['n_trades']}")
            print(f"Return:        {r['total_return_pct']:.2f}%")
            print(f"Max DD:        {r['max_drawdown_pct']:.2f}%")
            print(f"Sharpe:        {r['sharpe_ratio']:.3f}")
            print(f"Final Equity:  {r['final_equity']:,.2f}")
            print("=" * 80)

    main()
