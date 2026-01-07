## Bayes_opt_adapt_RSI.py

import numpy as np
import pandas as pd
from numba import jit, prange
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple, Any
import warnings
import os
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import logging
from datetime import datetime
import json
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.trial import TrialState
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


## best for adapt_RSI+adapt_RSI_h
# DEFAULT_PARAMS = {
#     "sl_multiplier": 2.7993,
#     "tp_multiplier": 4.4847,
#     "trail_multiplier": 1.6374,
#     "channel_multi": 2.1361,
#     "channel_length": 44,
#     "atr_period": 23,
#     "base_period": 10,
#     "min_period": 3,
#     "max_period": 16,
#     "fast_period": 11,
#     "slow_period": 34,
#     "steepness": 3,
#     "fast_steepness": 9,
#     "slow_steepness": 5,
# }

## best for adapt_RSI
# DEFAULT_PARAMS = {
#     "sl_multiplier": 4.0,
#     "tp_multiplier": 6.0,
#     "trail_multiplier":4.0,
#     "channel_multi": 3.2,
#     "channel_length": 100,
#     "atr_period": 14,
#     "base_period": 9,
#     "min_period": 3,
#     "max_period": 40,
#     "fast_period": 10,
#     "slow_period": 30,
#     "steepness": 9,
#     "fast_steepness": 11,
#     "slow_steepness": 7,
# }

## best for adapt_RSI + adapt_RSI_h
# DEFAULT_PARAMS = {
#     "sl_multiplier": 3.8,
#     "tp_multiplier": 5.0,
#     "trail_multiplier": 2.5,
#     "channel_multi": 3.2,
#     "channel_length": 100,
#     "atr_period": 14,
#     "base_period": 9,
#     "min_period": 6,
#     "max_period": 28,
#     "fast_period": 10,
#     "slow_period": 30,
#     "steepness": 9,
#     "fast_steepness": 11,
#     "slow_steepness": 7,
# }

## best for adapt_RSI_a
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
    "trim_percent":0.2,
    "use_replacement": True,
}

## best for adapt_RSI_b
# DEFAULT_PARAMS = {
#     "sl_multiplier": 3.7094731056601264,
#     "tp_multiplier": 5.137908398520907,
#     "trail_multiplier": 1.9952734571554853,
#     "channel_multi": 3.968825309105837,
#     "channel_length": 36,
#     "atr_period": 16,
#     "base_period": 8,
#     "min_period": 3,
#     "max_period": 11,
#     "fast_period": 9,
#     "slow_period": 49,
#     "steepness": 9,
#     "fast_steepness": 11,
#     "slow_steepness": 3,
#     "median_length": 8,
#     "sigmoid_margin": 0.07205219579381622,
#     "trim_percent": 0.1484549082816412,
#     "use_replacement": True,
# }

@jit(nopython=True, cache=True)
def calculate_heikin_ashi_numba(open_arr, high_arr, low_arr, close_arr):
    n = len(close_arr)
    ha_open = np.zeros(n)
    ha_high = np.zeros(n)
    ha_low = np.zeros(n)
    ha_close = np.zeros(n)

    # Initial candle initialization
    ha_close[0] = (open_arr[0] + high_arr[0] + low_arr[0] + close_arr[0]) / 4
    ha_open[0] = (open_arr[0] + close_arr[0]) / 2
    ha_high[0] = high_arr[0]
    ha_low[0] = low_arr[0]

    for i in range(1, n):
        ha_close[i] = (open_arr[i] + high_arr[i] + low_arr[i] + close_arr[i]) / 4
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
        ha_high[i] = max(high_arr[i], ha_open[i], ha_close[i])
        ha_low[i] = min(low_arr[i], ha_open[i], ha_close[i])
        
    return ha_open, ha_high, ha_low, ha_close

@jit(nopython=True, cache=True)
def rsi_numba(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI - numba optimized standalone function"""
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
        rsi[period-1] = 100
    else:
        rs = avg_gain / avg_loss
        rsi[period-1] = 100 - (100 / (1 + rs))
    
    for i in range(period, n):
        if deltas[i] > 0:
            gain[i] = deltas[i]
            loss[i] = 0
        else:
            gain[i] = 0
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
def atr_numba(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate Average True Range - numba optimized"""
    n = len(high)
    atr = np.full(n, np.nan)
    
    if n < period:
        return atr
    
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hc, lc)
    
    atr[period-1] = np.mean(tr[:period])
    
    for i in range(period, n):
        atr[i] = (atr[i-1] * (period - 1) + tr[i]) / period
    
    return atr

@jit(nopython=True, cache=True)
def sma_numba(data: np.ndarray, period: int) -> np.ndarray:
    """Simple Moving Average - numba optimized"""
    n = len(data)
    sma_result = np.full(n, np.nan)
    
    if n < period:
        return sma_result
    
    sma_result[period-1] = np.sum(data[:period]) / period
    
    for i in range(period, n):
        sma_result[i] = sma_result[i-1] + (data[i] - data[i-period]) / period
    
    return sma_result

@jit(nopython=True, cache=True)
def ema_numba(data: np.ndarray, period: int) -> np.ndarray:
    n = len(data)
    ema_result = np.full(n, np.nan)
    
    # Find the first non-NaN index
    start_idx = -1
    for i in range(n):
        if not np.isnan(data[i]):
            start_idx = i
            break
            
    if start_idx == -1 or (n - start_idx) < period:
        return ema_result
    
    # Initialize the first EMA value with the SMA of the first 'period' valid bars
    alpha = 2.0 / (period + 1.0)
    
    first_sum = 0.0
    for i in range(start_idx, start_idx + period):
        first_sum += data[i]
    
    ema_result[start_idx + period - 1] = first_sum / period
    
    # Calculate the rest
    for i in range(start_idx + period, n):
        ema_result[i] = (data[i] - ema_result[i-1]) * alpha + ema_result[i-1]
    return ema_result

@jit(nopython=True, cache=True)
def calculate_all_rsi_numba(close: np.ndarray, min_period: int, max_period: int) -> np.ndarray:
    """Calculate RSI for all periods in parallel"""
    n = len(close)
    num_periods = max_period - min_period + 1
    all_rsi = np.full((n, num_periods), np.nan)
    
    for p_idx in prange(num_periods):
        period = min_period + p_idx
        all_rsi[:, p_idx] = rsi_numba(close, period)
    return all_rsi

@jit(nopython=True, cache=True)
def rma_numba(src: np.ndarray, length: int) -> np.ndarray:
    """
    TradingView ta.rma(): Wilder's smoothing.
    Equivalent to EMA with alpha = 1/length, seeded by SMA(length).
    """
    n = len(src)
    out = np.full(n, np.nan)
    if length <= 0 or n < length:
        return out

    # seed with SMA of first length values
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
    """
    Pine:
    ha_tr = max(ha_high - ha_low,
                max(abs(ha_high - ha_close[1]), abs(ha_low - ha_close[1])))
    """
    n = len(ha_close)
    tr = np.zeros(n)
    tr[0] = ha_high[0] - ha_low[0]
    for i in range(1, n):
        hl = ha_high[i] - ha_low[i]
        hc = abs(ha_high[i] - ha_close[i - 1])
        lc = abs(ha_low[i] - ha_close[i - 1])
        tr[i] = max(hl, hc, lc)
    return tr


@jit(nopython=True, cache=True)
def run_simulation_real_same_close_numba(
    close_arr,
    buy_signal,
    sell_signal,
    initial_capital,
    pos_size_pct,
    comm_pct
):
    """
    Alternative execution model:
    - signals generated on bar i
    - orders filled immediately at REAL close[i]
    - equity marked to REAL close (same series)

    This is closer to a "bar-close" execution assumption.
    """
    n = len(close_arr)
    equity_curve = np.zeros(n)

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trade_count = 0

    for i in range(n):
        price = close_arr[i]

        # ---- ENTRY at same close ----
        if buy_signal[i] and position == 0.0 and not np.isnan(price):
            pos_value = capital * pos_size_pct
            qty = pos_value / price

            commission = pos_value * comm_pct
            capital -= (pos_value + commission)

            position = qty
            entry_price = price

        # ---- EXIT at same close ----
        elif sell_signal[i] and position > 0.0 and not np.isnan(price):
            exit_value = position * price
            commission = exit_value * comm_pct

            capital += (exit_value - commission)
            trade_count += 1

            position = 0.0
            entry_price = 0.0

        # ---- EQUITY marked at close ----
        equity_curve[i] = capital + (position * price if position > 0.0 else 0.0)

    return equity_curve, trade_count

@jit(nopython=True, cache=True)
def run_simulation_real_next_open_numba(
    open_arr,
    close_arr,
    buy_signal,
    sell_signal,
    initial_capital,
    pos_size_pct,
    comm_pct
):
    """
    Match TradingView backtest convention more closely:
    - signals generated on bar i
    - orders filled at next bar open (i+1) on REAL symbol
    - equity marked to REAL close
    """
    n = len(close_arr)
    equity_curve = np.zeros(n)

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trade_count = 0

    for i in range(n - 1):  # need i+1 open
        # Entry (next bar open)
        if buy_signal[i] and position == 0.0:
            fill_price = open_arr[i + 1]
            pos_value = capital * pos_size_pct
            qty = pos_value / fill_price

            commission = pos_value * comm_pct
            capital -= (pos_value + commission)

            position = qty
            entry_price = fill_price

        # Exit (next bar open)
        elif sell_signal[i] and position > 0.0:
            fill_price = open_arr[i + 1]
            exit_value = position * fill_price
            commission = exit_value * comm_pct

            capital += (exit_value - commission)
            trade_count += 1

            position = 0.0
            entry_price = 0.0

        # Mark-to-market at REAL close
        equity_curve[i] = capital + (position * close_arr[i] if position > 0.0 else 0.0)

    equity_curve[-1] = capital + (position * close_arr[-1] if position > 0.0 else 0.0)
    return equity_curve, trade_count

@jit(nopython=True, cache=True)
def run_simulation_ha_numba(
    ha_open,
    ha_high,
    ha_low,
    ha_close,
    buy_signal,
    sell_signal,
    initial_capital,
    pos_size_pct,
    comm_pct
):
    n = len(ha_close)
    equity_curve = np.zeros(n)

    capital = initial_capital
    position = 0.0
    entry_price = 0.0
    trade_count = 0
    for i in range(n - 1):  # need i+1 for next open

        # -------- ENTRY (next HA open) --------
        if buy_signal[i] and position == 0:
            entry_price = ha_open[i + 1]
            pos_value = capital * pos_size_pct
            position = pos_value / entry_price

            commission = pos_value * comm_pct
            capital -= (pos_value + commission)

        # -------- EXIT --------
        elif sell_signal[i] and position > 0:
            exit_price = ha_open[i + 1]
            exit_value = position * exit_price
            commission = exit_value * comm_pct

            pnl = exit_value - commission - (position * entry_price)
            capital += exit_value - commission
            trade_count += 1

            position = 0.0
            entry_price = 0.0

        # -------- EQUITY --------
        equity_curve[i] = capital + (position * ha_close[i] if position > 0 else 0)

    equity_curve[-1] = capital
    return equity_curve, trade_count

# @jit(nopython=True, cache=True)
# @jit(nopython=True, cache=True)
# def run_simulation_numba(close, buy_signal, sell_signal, initial_capital, pos_size_pct, comm_pct):
#     n = len(close)
#     equity_curve = np.zeros(n)
#     capital = initial_capital
#     position = 0.0
#     entry_price = 0.0
    
#     trade_pnls = [] # Numba handles lists of floats well
    
#     for i in range(n):
#         # Entry Logic
#         if buy_signal[i] and position == 0:
#             pos_value = capital * pos_size_pct
#             position = pos_value / close[i]
#             entry_price = close[i]
#             # Deduct position + commission
#             capital -= (pos_value + (pos_value * comm_pct))
        
#         # Exit Logic
#         elif sell_signal[i] and position > 0:
#             exit_val = position * close[i]
#             exit_comm = exit_val * comm_pct
#             capital += (exit_val - exit_comm)
            
#             # Record PnL for metrics later
#             trade_pnl = (exit_val - exit_comm) - (position * entry_price)
#             trade_pnls.append(trade_pnl)
            
#             position = 0.0
#             entry_price = 0.0
            
#         # Update Equity
#         cur_pos_val = position * close[i] if position > 0 else 0.0
#         equity_curve[i] = capital + cur_pos_val
#     return equity_curve, trade_pnls

@dataclass
class StrategyParams:
    """All strategy parameters"""
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
    
    @classmethod
    def from_trial(cls, trial: optuna.Trial) -> 'StrategyParams':
        """Create parameters from Optuna trial"""
        return cls(
            base_period=trial.suggest_int('base_period', 10, 20),
            min_period=trial.suggest_int('min_period', 3, 10),
            max_period=trial.suggest_int('max_period', 25, 50),
            fast_period=trial.suggest_int('fast_period', 5, 10),
            slow_period=trial.suggest_int('slow_period', 40, 60),
            channel_length=trial.suggest_int('channel_length', 30, 50),
            channel_multi=trial.suggest_float('channel_multi', 1.5, 3.5),
            atr_period=trial.suggest_int('atr_period', 12, 40),
            sl_multiplier=trial.suggest_float('sl_multiplier', 1.0, 3.0),
            tp_multiplier=trial.suggest_float('tp_multiplier', 2.0, 8.0),
            trail_multiplier=trial.suggest_float('trail_multiplier', 1.0, 5.0),
            steepness=trial.suggest_float('steepness', 5.0, 15.0),
            fast_steepness=trial.suggest_float('fast_steepness', 7.0, 15.0),
            slow_steepness=trial.suggest_float('slow_steepness', 5.0, 10.0),
            sigmoid_margin=trial.suggest_float('sigmoid_margin', 0.0, 0.3),
            median_length=trial.suggest_int('median_length', 5, 30),
            trim_percent=trial.suggest_float('trim_percent', 0.1, 0.3),
            use_replacement=trial.suggest_categorical('use_replacement', [True, False]),
            position_size_pct=trial.suggest_float('position_size_pct', 0.05, 0.2),
        )
    
    def to_hash(self) -> str:
        """Create unique hash for parameters"""
        params_str = json.dumps(self.__dict__, sort_keys=True)
        return hashlib.md5(params_str.encode()).hexdigest()


class BayesianRSIATROptimizer:
    """Bayesian optimization for RSI ATR strategy without TA-Lib"""
    
    def __init__(self, data_dir: str = "./data", results_dir: str = "./results"):
        self.data_dir = Path(data_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.study = None
        
    def load_data_chunk(self, file_path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        if isinstance(file_path, str):
            file_path = Path(file_path)

        try:
            df = pd.read_parquet(file_path)
        except Exception as e:
            import logging
            logging.getLogger(__name__).error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

        # Normalize column names to lowercase
        df.columns = [c.lower() for c in df.columns]

        # Ensure required columns exist
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                import numpy as np
                df[col] = np.nan  # safer than 0

        # Keep the most recent 'sample_size' rows
        if sample_size and sample_size > 0 and sample_size < len(df):
            df = df.tail(sample_size)
        return df

    def find_parquet_files(self, pattern: str = "*.parquet") -> List[Path]:
        """Find all parquet files in data directory"""
        files = list(self.data_dir.rglob(pattern))
        files.sort()
        logger.info(f"Found {len(files)} parquet files")
        return files
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_olympian_mean(data: np.ndarray, length: int, trim_percent: float, replace: bool) -> np.ndarray:
        """
        Matches Pine olympian_mean(_src, _length, _trim_percent, replace)
        - Sort window
        - Trim trim bars each side (trim = max(1, round(length*trim_percent)))
        - If replace=True: average length bars where trimmed sides replaced by edge values
        (low_replacement repeated trim times + untrimmed + high_replacement repeated trim times)
        - Else: average only untrimmed
        - Else (count<=0): median of window
        """
        n = len(data)
        out = np.full(n, np.nan)

        if length <= 0:
            return out

        for i in range(length - 1, n):
            # build window (Pine uses nz(); we'll treat NaN as 0 to mimic nz)
            window = np.empty(length, dtype=np.float64)
            for k in range(length):
                v = data[i - k]
                window[k] = 0.0 if np.isnan(v) else v

            # sort
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

                    # mean of: low_rep*trim + window[start_idx:end_idx+1] + high_rep*trim over "length" items
                    s = 0.0
                    # left pad
                    for _ in range(trim):
                        s += low_rep
                    # middle
                    for j in range(start_idx, end_idx + 1):
                        s += window[j]
                    # right pad
                    for _ in range(trim):
                        s += high_rep

                    out[i] = s / length
                else:
                    s = 0.0
                    for j in range(start_idx, end_idx + 1):
                        s += window[j]
                    out[i] = s / count
            else:
                # median fallback (like Pine median_filter)
                mid = length // 2
                if length % 2 == 0:
                    out[i] = 0.5 * (window[mid - 1] + window[mid])
                else:
                    out[i] = window[mid]

        return out

    def calculate_indicators_fast(self, df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        df = df.copy()

        # --- 1) HA series computed from REAL OHLC (matches Pine) ---
        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi_numba(
            df['open'].values, df['high'].values, df['low'].values, df['close'].values
        )
        df['ha_open'], df['ha_high'], df['ha_low'], df['ha_close'] = ha_o, ha_h, ha_l, ha_c
        df['ha_hl2'] = (df['ha_high'].values + df['ha_low'].values) / 2.0

        # --- 2) HA True Range + HA ATR via RMA (matches haATR()) ---
        ha_tr = ha_true_range_numba(df['ha_high'].values, df['ha_low'].values, df['ha_close'].values)

        # Channel ATR: haATR(channelLength)
        df['atr_chan'] = rma_numba(ha_tr, params.channel_length)

        # Trend channel: SMA(sigHL2, channelLength), where sigHL2 = haHL2
        df['mid_channel'] = sma_numba(df['ha_hl2'].values, params.channel_length)
        df['lower_channel'] = df['mid_channel'].values - df['atr_chan'].values * params.channel_multi
        df['upper_channel'] = df['mid_channel'].values + df['atr_chan'].values * params.channel_multi

        # lower TF
        lower_tf_len = max(1, params.channel_length // 2)
        df['mid_channel_lower_tf'] = sma_numba(df['ha_hl2'].values, lower_tf_len)

        # --- 3) Adaptive RSI on sigClose = haClose ---
        df['base_rsi'] = rsi_numba(df['ha_close'].values, params.base_period)
        scaled = (100.0 - df['base_rsi'].values) / 100.0

        adaptive_period_raw = np.round(scaled * (params.max_period - params.min_period) + params.min_period)
        adaptive_period = adaptive_period_raw.copy()
        adaptive_period[np.isnan(adaptive_period)] = params.min_period
        adaptive_period = np.clip(adaptive_period, params.min_period, params.max_period).astype(np.int32)
        df['adaptive_period'] = adaptive_period

        # Pine snippet defines switch only for RSI 3..34 (so outside => na)
        min_switch = 3
        max_switch = 34
        all_rsi = calculate_all_rsi_numba(df['ha_close'].values, min_switch, max_switch)

        adaptive_rsi = np.full(len(df), np.nan)
        for i in range(len(df)):
            p = adaptive_period[i]
            if min_switch <= p <= max_switch:
                adaptive_rsi[i] = all_rsi[i, p - min_switch]
            else:
                adaptive_rsi[i] = np.nan
        df['adaptive_rsi'] = adaptive_rsi

        # --- 4) fastRSI / slowRSI (EMA of adaptiveRSI) ---
        df['fast_rsi'] = ema_numba(df['adaptive_rsi'].values, params.fast_period)
        df['slow_rsi'] = ema_numba(df['adaptive_rsi'].values, params.slow_period)

        # --- 5) Sigmoid + EMA + Olympian Mean (plots in Pine, not used for entry/exit) ---
        # Pine:
        # fast_sigmoidRSI = sigmoid(adaptiveRSI/100, fast_steepness)*100
        # slow_sigmoidRSI = sigmoid(adaptiveRSI/100, slow_steepness, sigmoid_margin)*100
        def sigmoid_np(x01, steep, margin=0.0):
            raw = 1.0 / (1.0 + np.exp(-steep * (x01 - 0.5)))
            return (margin + (1.0 - 2.0 * margin) * raw)

        ar = df['adaptive_rsi'].values
        ar01 = ar / 100.0
        fast_sigmoidRSI = sigmoid_np(ar01, params.fast_steepness, 0.0) * 100.0
        slow_sigmoidRSI = sigmoid_np(ar01, params.slow_steepness, params.sigmoid_margin) * 100.0

        df['fast_sigmoid'] = ema_numba(fast_sigmoidRSI, params.fast_period)
        df['slow_sigmoid'] = ema_numba(slow_sigmoidRSI, params.slow_period)

        df['fast_sigmoid_mid'] = self.calculate_olympian_mean(
            df['fast_sigmoid'].values, params.median_length, params.trim_percent, params.use_replacement
        )
        df['slow_sigmoid_mid'] = self.calculate_olympian_mean(
            df['slow_sigmoid'].values, params.median_length, params.trim_percent, params.use_replacement
        )

        # --- 6) Risk management ATR + trail offset (matches Pine) ---
        atr_ha = rma_numba(ha_tr, params.atr_period)
        trail_atr = np.full(len(df), np.nan)
        for i in range(len(df)):
            a = atr_ha[i]
            trail_atr[i] = a if not np.isnan(a) else np.nan        

        # atr_real = atr_numba(df['high'].values, df['low'].values, df['close'].values, params.atr_period)
        # trail_atr = np.full(len(df), np.nan)
        # for i in range(len(df)):
        #     a1 = atr_ha[i]
        #     a2 = atr_real[i]
        #     if np.isnan(a1) and np.isnan(a2):
        #         trail_atr[i] = np.nan
        #     elif np.isnan(a1):
        #         trail_atr[i] = a2
        #     elif np.isnan(a2):
        #         trail_atr[i] = a1
        #     else:
        #         trail_atr[i] = a1 if a1 >= a2 else a2

        df['trail_atr'] = trail_atr
        df['trail_offset'] = df['trail_atr'].values * params.trail_multiplier
        return df

    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_signals_pine_like_numba(
        sig_close,            # ha_close
        sig_low,              # ha_low  (used for trailing support update)
        mid_channel,
        mid_channel_lower_tf,
        lower_channel,
        adaptive_rsi,
        slow_rsi,
        trail_offset,
        min_bars=10
    ):
        """
        Matches Pine:

        uptrend = sigClose > midChannel
        above_lowerTF = sigClose > midChannel_lowerTF
        longCondition = crossover(adaptiveRSI, slowRSI) and (uptrend or above_lowerTF)
        shortCondition = crossunder(adaptiveRSI, slowRSI) or crossunder(sigClose, lowerChannel)

        Risk:
        support_resistance := riskLow - trail_offset at entry
        if crossunder(riskClose, support_resistance) or shortCondition => exit
        else support_resistance := max(support_resistance, riskLow - trail_offset)

        Here riskClose=riskLow=sigClose/sigLow because your Pine sets them to HA.
        """
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

            # crossover(adaptiveRSI, slowRSI)
            crossover = False
            crossunder_rsi = False
            if not np.isnan(prev_ad) and not np.isnan(prev_sl):
                crossover = (adaptive_rsi[i] > slow_rsi[i]) and (prev_ad <= prev_sl)
                crossunder_rsi = (adaptive_rsi[i] < slow_rsi[i]) and (prev_ad >= prev_sl)

            # crossunder(sigClose, lowerChannel)
            crossunder_channel = False
            if not np.isnan(prev_sig_close) and not np.isnan(prev_lower_channel):
                crossunder_channel = (sig_close[i] < lower_channel[i]) and (prev_sig_close >= prev_lower_channel)

            long_condition = crossover and trend_ok
            short_condition = crossunder_rsi or crossunder_channel

            # riskClose/riskLow are HA (per your Pine)
            risk_close = sig_close[i]
            risk_low = sig_low[i]

            if not in_pos:
                if long_condition:
                    buy[i] = True
                    in_pos = True
                    support = risk_low - trail_offset[i]
            else:
                # update trailing support
                cur_stop = risk_low - trail_offset[i]
                if cur_stop > support:
                    support = cur_stop

                # crossunder(riskClose, support_resistance)
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

    def backtest_single(self, df: pd.DataFrame, params: StrategyParams) -> Dict[str, Any]:
        try:
            df = self.calculate_indicators_fast(df, params)           
            # 1. Generate Signals (Heikin Ashi logic)
            min_bars = max(
                params.max_period,      # adaptive RSI needs this
                params.slow_period,     # EMA needs this
                params.channel_length,  # SMA channel
                params.atr_period       # ATR
            ) + 2            
            buy_signal, sell_signal = self.calculate_signals_pine_like_numba(
                df['ha_close'].values,              # sig_close
                df['ha_low'].values,                # sig_low (riskLow in Pine)
                df['mid_channel'].values,
                df['mid_channel_lower_tf'].values,
                df['lower_channel'].values,
                df['adaptive_rsi'].values,
                df['slow_rsi'].values,
                df['trail_offset'].values,
                min_bars=min_bars
            )

            # 2. Run High-Speed Simulation
            equity_curve, n_trades = run_simulation_real_same_close_numba(
                df['close'].values,  # your chosen execution model: fill at REAL close
                buy_signal,
                sell_signal,
                params.initial_capital,
                params.position_size_pct,
                params.commission_pct
            )

            # equity_curve, n_trades = run_simulation_real_next_open_numba(
            #     df['open'].values,       # REAL fills
            #     df['close'].values,      # REAL mark-to-market
            #     buy_signal,
            #     sell_signal,
            #     params.initial_capital,
            #     params.position_size_pct,
            #     params.commission_pct
            # )
            
            # 3. Calculate Metrics from the results
            total_ret_pct = ((equity_curve[-1] - params.initial_capital) / params.initial_capital) * 100
            
            # Sharpe Ratio
            returns = np.diff(equity_curve) / equity_curve[:-1]
            sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(252)) if (len(returns) > 0 and np.std(returns) > 0) else 0.0
            
            # Drawdown
            cum_max = np.maximum.accumulate(equity_curve)
            drawdown = (equity_curve - cum_max) / cum_max
            max_dd = np.min(drawdown) * 100

            return {
                'total_return_pct': total_ret_pct,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'n_trades': n_trades,
                'final_equity': equity_curve[-1]
            }

        except Exception as e:
            logger.error(f"backtest_single error: {e}")
            return {'total_return_pct': -100.0, 'n_trades': 0}
        

    def run_backtest(self, files: list, sample_size: int, **kwargs) -> dict:
        """
        Run backtest across multiple files using provided strategy parameters.
        kwargs contains strategy parameters like base_period, fast_period, etc.
        Returns aggregated metrics.
        """
        params = StrategyParams(**kwargs)
        results = []
        for file_path in files:
            df = self.load_data_chunk(file_path, sample_size)
            if len(df) > 100:
                res = self.backtest_single(df, params)
                results.append(res)
        
        # Aggregate results across files
        if results:
            agg_metrics = self.aggregate_metrics(results)
        else:
            agg_metrics = {
                'avg_sharpe': 0.0,
                'avg_return_pct': 0.0,
                'avg_max_dd': 0.0,
                'avg_win_rate': 0.0,
                'total_trades': 0,
                'signal_count': 0,
                'n_files': 0
            }
        return agg_metrics

    def objective(self, trial, files, sample_size, n_files):
            # --- Start from defaults ---
            params = DEFAULT_PARAMS.copy()

            # --- Parameters to optimize ---
            # param_ranges = {
            #     "sl_multiplier": (1.5, 4.0), # Wider stops for HA smoothing
            #     "tp_multiplier": (1.5, 6.0),
            #     "trail_multiplier": (1.0, 3.0),
            #     "channel_multi": (1.5, 3.5),
            #     "channel_length": (40, 60),
            #     "atr_period": (10, 25),
            #     "base_period": (10, 20),
            #     "min_period": (3, 10),
            #     "max_period": (20, 40),
            #     "fast_period": (2, 10),
            #     "slow_period": (30, 45), # Increased range to force a gap
            #     "steepness": (1, 10),      # Lowered: avoid binary 0/100 RSI
            #     "fast_steepness": (5, 10), 
            #     "slow_steepness": (1, 5)
            # }
            param_ranges = {
                "sl_multiplier": (2.0, 4.5), # Wider stops for HA smoothing
                "tp_multiplier": (3.0, 6.0),
                "trail_multiplier": (1.5, 4.0),
                "channel_multi": (2.0, 4.0),
                "channel_length": (30, 80),
                "atr_period": (10, 50),
                "base_period": (5, 20),
                "min_period": (1, 10),
                "max_period": (11, 25),
                "fast_period": (3, 10),
                "slow_period": (25, 50), # Increased range to force a gap
                "steepness": (2, 10),      # Lowered: avoid binary 0/100 RSI
                "fast_steepness": (5, 15), 
                "slow_steepness": (1, 5),
                "median_length": (7, 20),
                # "sigmoid_margin":(0.0, 0.3),
                # "trim_percent": (0.0, 0.3),
            }

            # --- Sample ONLY the ranged parameters ---
            for name, (low, high) in param_ranges.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[name] = trial.suggest_int(name, low, high)
                else:
                    params[name] = trial.suggest_float(name, low, high)

            # --- HARD CONSTRAINTS (Calculated after sampling) ---
            # 1. Force Fast < Slow gap (Critical for HA)
            if params["fast_period"] >= params["slow_period"] or params["fast_steepness"] <= params["slow_steepness"]:
                # Instead of returning None, suggest a valid range or penalize
                return -20.0  

            # 2. Adaptive RSI logic constraints
            if params["min_period"] >= params["base_period"] or params["max_period"] <= params["base_period"]:
                return -20.0

            # --- Force known-working parameters for Trial 0 to kickstart Optuna ---
            if trial.number == 0:
                params.update({
                    "trail_multiplier":2.0,
                    "channel_multi": 2.5,     
                    "min_period": 5,
                    "max_period": 15,
                    "fast_period": 10,
                    "slow_period": 30,
                })

            try:
                # --- Run backtest ---
                if n_files > len(files):
                    n_files = len(files)
                result = self.run_backtest(
                    files=files[:n_files],
                    sample_size=sample_size,
                    **params
                )

                trades = result.get("total_trades", 0)
                
                # --- Handle Zero Trades ---
                if trades == 0:
                    # Assign a worse score than any losing strategy (-10)
                    # This ensures Optuna treats 0 trades as a total failure
                    return -10.0

                # --- Objective Logic ---
                # We want to maximize Sharpe Ratio, but we also want to reward 
                # strategies that actually trade. 
                sharpe = result.get("avg_sharpe", 0.0)
                # Add a small bonus for trade frequency to avoid "lucky" 1-trade trials
                # score = sharpe + (min(trades, 50) / 500) 
                return sharpe

            except Exception as e:
                print(f"Backtest error in Trial {trial.number}: {e}")
                return -20.0
            
    def aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics across files safely"""

        # --- weights based on executed trades ---
        weights = np.array([r.get('n_trades', 0) for r in results], dtype=float)
        total_weight = weights.sum()

        # --- NEW: aggregate signal count safely ---
        signal_counts = np.array([r.get('signal_count', 0) for r in results], dtype=int)
        total_signals = int(signal_counts.sum())

        # ---- SAFE FALLBACK (no trades anywhere) ----
        if total_weight == 0:
            return {
                'avg_sharpe': float(np.mean([r.get('sharpe_ratio', 0.0) for r in results])),
                'avg_return_pct': float(np.mean([r.get('total_return_pct', 0.0) for r in results])),
                'avg_max_dd': float(np.mean([r.get('max_drawdown', 0.0) for r in results])),
                'avg_win_rate': 0.0,
                'total_trades': 0,
                'signal_count': total_signals,   # ✅ FIX
                'n_files': len(results)
            }

        # ---- NORMAL PATH ----
        return {
            'avg_sharpe': float(np.average(
                [r.get('sharpe_ratio', 0.0) for r in results], weights=weights)),
            'avg_return_pct': float(np.average(
                [r.get('total_return_pct', 0.0) for r in results], weights=weights)),
            'avg_max_dd': float(np.average(
                [r.get('max_drawdown', 0.0) for r in results], weights=weights)),
            'avg_win_rate': float(np.average(
                [r.get('win_rate', 0.0) for r in results], weights=weights)),
            'total_trades': int(total_weight),
            'signal_count': total_signals,     # ✅ FIX
            'n_files': len(results)
        }

    def calculate_score(self, metrics: Dict[str, Any]) -> float:
        # Strong penalty for no trades
        if metrics['total_trades'] == 0:
            return -10.0

        sharpe_weight = 2.0
        return_weight = 1.0
        dd_weight = -0.5
        win_rate_weight = 0.3

        score = (
            metrics['avg_sharpe'] * sharpe_weight +
            metrics['avg_return_pct'] / 100 * return_weight +
            metrics['avg_max_dd'] / 100 * dd_weight +
            metrics['avg_win_rate'] / 100 * win_rate_weight
        )

        if metrics['total_trades'] < 5:
            score *= 0.5
        return score

    def optimize_bayesian(self, n_trials: int = 100, n_files: int = 5, 
                         sample_size: int = 5000, n_jobs: int = -1) -> Dict[str, Any]:
        """Run Bayesian optimization using Optuna"""
        
        files = self.find_parquet_files()
        if len(files) == 0:
            logger.error("No parquet files found in data directory")
            return {}
        
        # Create Optuna study
        study_name = f"rsi_atr_bayesian_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        storage_url = f"sqlite:///{self.results_dir}/{study_name}.db"
        
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            sampler=TPESampler(seed=42, n_startup_trials=20, multivariate=True),
            pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=5),
            direction="maximize",
            load_if_exists=True
        )
        
        self.study = study
        
        # Run optimization
        logger.info(f"Starting Bayesian optimization with {n_trials} trials")
        
        objective_with_args = lambda trial: self.objective(
            trial, files, sample_size, n_files
        )
        
        study.optimize(
            objective_with_args,
            n_trials=n_trials,
            n_jobs=n_jobs if n_jobs > 0 else os.cpu_count(),
            show_progress_bar=True,
            gc_after_trial=True
        )
        
        # Get best trial
        if not study.trials or all(t.state != TrialState.COMPLETE for t in study.trials):
            logger.error("No successful trials completed.")
            return {}

        best_trial = study.best_trial
                
        # Save results
        results = self.save_optimization_results(study)
        
        # Visualize results
        self.visualize_results(study)
        
        return results
    
    def save_optimization_results(self, study: optuna.Study) -> Dict[str, Any]:
        """Save optimization results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save best parameters
        best_params = study.best_params
        best_params['score'] = float(study.best_value)
        
        best_file = self.results_dir / f"best_params_bayesian_{timestamp}.json"
        with open(best_file, 'w') as f:
            json.dump(best_params, f, indent=2)
        
        # Save all trials to CSV
        trials_df = study.trials_dataframe()
        trials_df = trials_df.sort_values('value', ascending=False)
        
        csv_file = self.results_dir / f"trials_bayesian_{timestamp}.csv"
        trials_df.to_csv(csv_file, index=False)
        
        # Save study statistics
        stats_file = self.results_dir / f"study_stats_{timestamp}.json"
        stats = {
            'best_value': float(study.best_value),
            'best_params': study.best_params,
            'n_trials': len(study.trials),
            'completed_trials': len([t for t in study.trials if t.state == TrialState.COMPLETE]),
            'pruned_trials': len([t for t in study.trials if t.state == TrialState.PRUNED]),
            'failed_trials': len([t for t in study.trials if t.state == TrialState.FAIL]),
        }
        
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to {best_file}, {csv_file}, {stats_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("BAYESIAN OPTIMIZATION RESULTS")
        print("="*80)
        print(f"\nBest Score: {study.best_value:.4f}")
        print(f"Completed Trials: {stats['completed_trials']}")
        print(f"Pruned Trials: {stats['pruned_trials']}")
        
        print("\nBest Parameters Found:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value}")
        
        best_trial = study.best_trial
        print(f"\nPerformance Metrics:")
        print(f"  Avg Sharpe: {best_trial.user_attrs.get('avg_sharpe', 0):.3f}")
        print(f"  Avg Return: {best_trial.user_attrs.get('avg_return_pct', 0):.2f}%")
        print(f"  Avg Max DD: {best_trial.user_attrs.get('avg_max_dd', 0):.2f}%")
        print(f"  Avg Win Rate: {best_trial.user_attrs.get('avg_win_rate', 0):.1f}%")
        print(f"  Total Trades: {best_trial.user_attrs.get('total_trades', 0)}")
        
        return {
            'best_params': best_params,
            'trials_df': trials_df,
            'stats': stats,
            'study': study
        }
    
    def visualize_results(self, study: optuna.Study):
        """Create visualization plots"""
        try:
            import plotly
            from plotly.subplots import make_subplots
            
            # Optimization history
            fig1 = optuna.visualization.plot_optimization_history(study)
            fig1.write_html(str(self.results_dir / "optimization_history.html"))
            
            # Parameter importances
            fig2 = optuna.visualization.plot_param_importances(study)
            fig2.write_html(str(self.results_dir / "param_importances.html"))
            
            # Parallel coordinate plot
            fig3 = optuna.visualization.plot_parallel_coordinate(study)
            fig3.write_html(str(self.results_dir / "parallel_coordinate.html"))
            
            # Slice plot
            fig4 = optuna.visualization.plot_slice(study)
            fig4.write_html(str(self.results_dir / "slice_plot.html"))
            
            # Contour plot for important parameters
            important_params = self.get_important_params(study, n_params=4)
            if len(important_params) >= 2:
                fig5 = optuna.visualization.plot_contour(
                    study, params=important_params[:2]
                )
                fig5.write_html(str(self.results_dir / "contour_plot.html"))
            
            logger.info(f"Visualizations saved to {self.results_dir}")
            
        except ImportError:
            logger.warning("Plotly not installed. Skipping visualizations.")
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    def get_important_params(self, study: optuna.Study, n_params: int = 4) -> List[str]:
        """Get most important parameters"""
        try:
            importance_df = optuna.importance.get_param_importances(study)
            return importance_df.index[:n_params].tolist()
        except:
            # Fallback to trial parameters
            if study.trials:
                params = list(study.trials[0].params.keys())
                return params[:min(n_params, len(params))]
            return []
    
    def validate_best_params(self, best_params: Dict, validation_files: List[Path] = None,
                            sample_size: int = 10000) -> Dict[str, Any]:
        """Validate best parameters on additional files"""
        if validation_files is None:
            files = self.find_parquet_files()
            # Use different files for validation (last n files)
            validation_files = files[-5:] if len(files) > 5 else files
        
        params = StrategyParams(**best_params)
        validation_results = []
        
        print(f"\nValidating best parameters on {len(validation_files)} files...")
        
        for file_path in tqdm(validation_files, desc="Validation"):
            df = self.load_data_chunk(file_path, sample_size)
            if len(df) > 100:
                result = self.backtest_single(df, params)
                result['file'] = file_path.name
                validation_results.append(result)
        
        if validation_results:
            agg = self.aggregate_metrics(validation_results)
            
            print(f"\nValidation Results:")
            print(f"Average Sharpe Ratio: {agg['avg_sharpe']:.3f}")
            print(f"Average Return: {agg['avg_return_pct']:.2f}%")
            print(f"Average Max Drawdown: {agg['avg_max_dd']:.2f}%")
            print(f"Average Win Rate: {agg['avg_win_rate']:.1f}%")
            print(f"Total Trades: {agg['total_trades']}")
            
            return {
                'validation_results': validation_results,
                'aggregate_metrics': agg,
                'params': best_params
            }
        
        return {}


# Alternative: Hyperopt implementation (if you prefer Hyperopt over Optuna)
class HyperoptOptimizer:
    """Hyperopt implementation for comparison"""
    
    def __init__(self, data_dir: str = "./data"):
        self.data_dir = Path(data_dir)
        from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
        
    def optimize_hyperopt(self, n_trials: int = 100):
        """Optimize using Hyperopt"""
        from hyperopt import hp, fmin, tpe, Trials, STATUS_OK
        
        space = {
            'base_period': hp.quniform('base_period', 3, 20, 1),
            'min_period': hp.quniform('min_period', 2, 10, 1),
            'max_period': hp.quniform('max_period', 20, 60, 1),
            'fast_period': hp.quniform('fast_period', 5, 30, 1),
            'slow_period': hp.quniform('slow_period', 15, 60, 1),
            'channel_length': hp.quniform('channel_length', 50, 200, 1),
            'channel_multi': hp.uniform('channel_multi', 1.0, 5.0),
            'sl_multiplier': hp.uniform('sl_multiplier', 1.0, 8.0),
            'tp_multiplier': hp.uniform('tp_multiplier', 2.0, 12.0),
            'position_size_pct': hp.uniform('position_size_pct', 0.05, 0.2),
        }
        
        trials = Trials()
        
        best = fmin(
            fn=self.hyperopt_objective,
            space=space,
            algo=tpe.suggest,
            max_evals=n_trials,
            trials=trials,
            show_progressbar=True
        )
        
        return best, trials


# Usage example continued
if __name__ == "__main__":
    import argparse
    
    def main():
        parser = argparse.ArgumentParser(description="Bayesian Optimization for RSI ATR Strategy")
        parser.add_argument("--mode", choices=["optimize", "validate", "test"], default="optimize",
                          help="Mode: optimize (run bayesian opt), validate (validate params), test (single file test)")
        parser.add_argument("--trials", type=int, default=100, help="Number of optimization trials")
        parser.add_argument("--sample", type=int, default=5000, help="Sample size per file")
        parser.add_argument("--files", type=int, default=3, help="Number of files to use per trial")
        parser.add_argument("--workers", type=int, default=-1, help="Number of parallel workers (-1 for auto)")
        parser.add_argument("--params-file", type=str, help="JSON file with parameters to validate")
        parser.add_argument("--test-file", type=str, help="Specific file to test")
        
        args = parser.parse_args()
        
        # Initialize optimizer
        optimizer = BayesianRSIATROptimizer(data_dir="./data", results_dir="./results")
        
        if args.mode == "optimize":
            print("="*80)
            print("BAYESIAN OPTIMIZATION MODE")
            print("="*80)
            print(f"Trials: {args.trials}")
            print(f"Sample size: {args.sample}")
            print(f"Files per trial: {args.files}")
            print(f"Workers: {'Auto' if args.workers == -1 else args.workers}")
            print("="*80)
            
            results = optimizer.optimize_bayesian(
                n_trials=args.trials,
                n_files=args.files,
                sample_size=args.sample,
                n_jobs=args.workers
            )
            
            if results:
                # Validate best parameters
                best_params = results['best_params'].copy()
                # Remove the score key for validation
                if 'score' in best_params:
                    del best_params['score']
                
                print("\n" + "="*80)
                print("VALIDATING BEST PARAMETERS")
                print("="*80)
                
                validation_results = optimizer.validate_best_params(
                    best_params,
                    sample_size=args.sample
                )
            
        elif args.mode == "validate":
            if not args.params_file:
                print("Error: --params-file required for validation mode")
                return
            
            print("="*80)
            print("VALIDATION MODE")
            print("="*80)
            
            import json
            with open(args.params_file, 'r') as f:
                params_data = json.load(f)
            
            if 'best_params' in params_data:
                params = params_data['best_params']
            else:
                params = params_data
            
            validation_results = optimizer.validate_best_params(
                params,
                sample_size=args.sample
            )
            
        elif args.mode == "test":
            print("="*80)
            print("SINGLE FILE TEST MODE")
            print("="*80)
            
            files = optimizer.find_parquet_files()
            
            if args.test_file:
                test_files = [Path(args.test_file)]
            elif files:
                test_files = [files[0]]
            else:
                print("No parquet files found in ./data directory")
                return
            
            # Use default parameters or load from file
            if args.params_file:
                import json
                with open(args.params_file, 'r') as f:
                    params_data = json.load(f)
                if 'best_params' in params_data:
                    params = StrategyParams(**params_data['best_params'])
                else:
                    params = StrategyParams(**params_data)
            else:
                params = StrategyParams()
            
            print(f"Testing file: {test_files[0].name}")
            print(f"Parameters:")
            for key, value in params.__dict__.items():
                if not key.startswith('_'):
                    print(f"  {key}: {value}")
            print("="*80)
            
            df = optimizer.load_data_chunk(test_files[0], sample_size=args.sample)
            
            if len(df) > 100:
                results = optimizer.backtest_single(df, params)
                
                print("\nTEST RESULTS:")
                print(f"Total Return: ${results['total_return']:,.2f}")
                print(f"Total Return: {results['total_return_pct']:.2f}%")
                print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
                print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
                print(f"Win Rate: {results['win_rate']:.1f}%")
                print(f"Number of Trades: {results['n_trades']}")
                print(f"Average Return per Trade: {results['avg_return']:.2f}%")
                print(f"Final Equity: ${results['final_equity']:,.2f}")
            else:
                print(f"Error: Not enough data ({len(df)} rows)")
    
    main()

                            