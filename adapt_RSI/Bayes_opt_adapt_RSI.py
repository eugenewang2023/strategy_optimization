## BAyes_opt_adapt_RSI.py

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


DEFAULT_PARAMS = {
    "base_period": 14,
    "min_period": 5,
    "max_period": 30,
    "fast_period": 5,
    "slow_period": 14,
    "channel_length": 100,
    "channel_multi": 2.5,
    "atr_period": 14,
    "sl_multiplier": 2.0,
    "tp_multiplier": 3.0,
    "trail_multiplier": 1.5,
    "steepness": 8,
    "fast_steepness": 8,
    "slow_steepness": 8,
}


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

@dataclass
class StrategyParams:
    """All strategy parameters"""
    base_period: int = 5
    min_period: int = 3
    max_period: int = 40
    fast_period: int = 10
    slow_period: int = 40
    channel_length: int = 100
    channel_multi: float = 3.0
    atr_period: int = 7
    sl_multiplier: float = 4.0
    tp_multiplier: float = 6.0
    trail_multiplier: float = 2.5
    steepness: float = 10.0
    fast_steepness: float = 10.0
    slow_steepness: float = 10.0
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
            base_period=trial.suggest_int('base_period', 3, 20),
            min_period=trial.suggest_int('min_period', 2, 10),
            max_period=trial.suggest_int('max_period', 20, 60),
            fast_period=trial.suggest_int('fast_period', 5, 30),
            slow_period=trial.suggest_int('slow_period', 15, 60),
            channel_length=trial.suggest_int('channel_length', 50, 200),
            channel_multi=trial.suggest_float('channel_multi', 1.0, 5.0),
            atr_period=trial.suggest_int('atr_period', 5, 20),
            sl_multiplier=trial.suggest_float('sl_multiplier', 1.0, 8.0),
            tp_multiplier=trial.suggest_float('tp_multiplier', 2.0, 12.0),
            trail_multiplier=trial.suggest_float('trail_multiplier', 1.0, 5.0),
            steepness=trial.suggest_float('steepness', 5.0, 20.0),
            fast_steepness=trial.suggest_float('fast_steepness', 5.0, 20.0),
            slow_steepness=trial.suggest_float('slow_steepness', 5.0, 20.0),
            sigmoid_margin=trial.suggest_float('sigmoid_margin', 0.0, 0.3),
            median_length=trial.suggest_int('median_length', 5, 30),
            trim_percent=trial.suggest_float('trim_percent', 0.1, 0.4),
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
    
    def calculate_indicators_fast(self, df: pd.DataFrame, params: StrategyParams) -> pd.DataFrame:
        df = df.copy()
        
        # 1. Transform to Heikin Ashi
        ha_o, ha_h, ha_l, ha_c = calculate_heikin_ashi_numba(
            df['open'].values, df['high'].values, df['low'].values, df['close'].values
        )
        df['ha_close'], df['ha_high'], df['ha_low'] = ha_c, ha_h, ha_l

        # 2. Base RSI (Used for adaptation) - Calculated on HA Close
        df['base_rsi'] = rsi_numba(df['ha_close'].values, params.base_period)
        
        # 3. Adaptive Period Calculation
        scaled = (100 - df['base_rsi']) / 100
        df['adaptive_period'] = np.round(scaled * (params.max_period - params.min_period) + params.min_period)
        df['adaptive_period'] = df['adaptive_period'].clip(params.min_period, params.max_period).fillna(params.min_period).astype(int)
        
        # 4. Calculate all RSI possibilities for the adaptive lookup
        all_rsi = calculate_all_rsi_numba(df['ha_close'].values, params.min_period, params.max_period)
        
        # 5. Map adaptive RSI values
        adaptive_rsi = np.full(len(df), np.nan)
        periods = df['adaptive_period'].values
        for i in range(len(df)):
            p = periods[i]
            if params.min_period <= p <= params.max_period:
                adaptive_rsi[i] = all_rsi[i, p - params.min_period]
        df['adaptive_rsi'] = adaptive_rsi

        # 6. Smooth the Adaptive RSI (Sigmoid & EMA)
        def sigmoid_fast(x, steepness, margin=0):
            x_scaled = x / 100
            raw = 1 / (1 + np.exp(-steepness * (x_scaled - 0.5)))
            return (margin + (1 - 2 * margin) * raw) * 100

        df['slow_sigmoid_rsi'] = sigmoid_fast(df['adaptive_rsi'], params.slow_steepness)
        # Fill NaNs with the first valid value or a neutral 50 to help the EMA start
        first_valid = df['slow_sigmoid_rsi'].dropna().iloc[0] if not df['slow_sigmoid_rsi'].dropna().empty else 50
        df['slow_rsi'] = ema_numba(df['slow_sigmoid_rsi'].fillna(first_valid).values, params.slow_period)        
        # df['slow_rsi'] = ema_numba(df['slow_sigmoid_rsi'].values, params.slow_period)
        
        # 7. Volatility & Channels (Using HA High/Low/Close)
        df['mid_channel'] = sma_numba(df['ha_close'].values, params.channel_length)
        df['atr_val'] = atr_numba(df['ha_high'].values, df['ha_low'].values, df['ha_close'].values, params.atr_period)
        df['lower_channel'] = df['mid_channel'] - (df['atr_val'] * params.channel_multi)
        
        # 8. Trail Offset
        df['dist'] = np.maximum(df['ha_high'] - df['ha_low'], df['atr_val'])
        df['trail_offset'] = df['dist'] * params.trail_multiplier
        return df

    @staticmethod
    @jit(nopython=True, cache=True)
    def calculate_signals_vectorized(
        close, ha_close, ha_low, adaptive_rsi, slow_rsi, lower_channel, trail_offset, min_bars=10
    ):
        n = len(close)
        buy_signal = np.zeros(n, dtype=np.bool_)
        sell_signal = np.zeros(n, dtype=np.bool_)
        
        current_pos = 0
        trail_stop = 0.0
        
        # We start earlier (min_bars=10) to catch signals in small samples
        for i in range(min_bars, n):
            # Check for NaN to avoid logic errors
            if np.isnan(adaptive_rsi[i]) or np.isnan(slow_rsi[i]):
                continue
                
            # ENTRY: Adaptive RSI is above Slow RSI AND we are in an "up" state
            if current_pos == 0:
                # Replaced strict crossover with a 'greater than' check for the smoke test
                # This ensures if the trend is already up, we take it.
                if adaptive_rsi[i] > slow_rsi[i]: 
                    buy_signal[i] = True
                    current_pos = 1
                    trail_stop = ha_low[i] - trail_offset[i]
            
            # EXIT
            elif current_pos == 1:
                # 1. RSI Trend Reversal
                rsi_reversal = adaptive_rsi[i] < slow_rsi[i]
                # 2. Price Break (using a slightly looser stop for the smoke test)
                stop_hit = close[i] < (trail_stop * 0.98) 
                
                if rsi_reversal or stop_hit:
                    sell_signal[i] = True
                    current_pos = 0
                else:
                    # Update Trailing Stop
                    new_stop = ha_low[i] - trail_offset[i]
                    if new_stop > trail_stop:
                        trail_stop = new_stop
                        
        return buy_signal, sell_signal

    def backtest_single(self, df: pd.DataFrame, params: StrategyParams) -> Dict[str, Any]:
        """Run backtest on single file"""
        try:
            df = self.calculate_indicators_fast(df, params)
            close = df['close'].values
            buy_signal, sell_signal = self.calculate_signals_vectorized(
                df['close'].values, 
                df['ha_close'].values, 
                df['ha_low'].values, 
                df['adaptive_rsi'].values, 
                df['slow_rsi'].values, 
                df['lower_channel'].values, 
                df['trail_offset'].values
                )
            signal_count = int(np.sum(buy_signal))
            # print(f" Buy/Sell signals: {buy_signal.sum()}/{sell_signal.sum()}")
            
            # Portfolio simulation
            capital = params.initial_capital
            position = 0.0
            entry_price = 0.0
            
            equity_curve = np.zeros(len(df))
            returns = np.zeros(len(df))
            trades = []
            
            for i in range(len(df)):
                if buy_signal[i] and position == 0:
                    position_value = capital * params.position_size_pct
                    position = position_value / close[i]
                    entry_price = close[i]
                    
                    commission = position_value * params.commission_pct
                    capital -= commission
                    
                    trades.append({'entry_idx': i, 'entry_price': entry_price})
                
                elif sell_signal[i] and position > 0 and trades:
                    exit_value = position * close[i]
                    capital += exit_value
                    
                    commission = exit_value * params.commission_pct
                    capital -= commission
                    
                    trade = trades[-1]
                    trade_pnl = exit_value - (position * trade['entry_price'])
                    trades[-1]['pnl'] = trade_pnl
                    trades[-1]['return_pct'] = trade_pnl / (position * trade['entry_price']) * 100
                    
                    position = 0.0
                    entry_price = 0.0
                
                current_position_value = position * close[i] if position > 0 else 0.0
                equity = capital + current_position_value
                equity_curve[i] = equity
                
                if i > 0:
                    returns[i] = (equity_curve[i] - equity_curve[i-1]) / equity_curve[i-1]
            
            # Calculate metrics
            total_return = equity_curve[-1] - params.initial_capital
            total_return_pct = total_return / params.initial_capital * 100
            
            daily_returns = returns[returns != 0]
            if len(daily_returns) > 1:
                sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            cumulative_max = np.maximum.accumulate(equity_curve)
            drawdowns = (equity_curve - cumulative_max) / cumulative_max
            max_drawdown = np.min(drawdowns) * 100
            
            completed_trades = [t for t in trades if 'pnl' in t]
            n_trades = len(completed_trades)
            
            if n_trades > 0:
                winning_trades = [t for t in completed_trades if t['pnl'] > 0]
                win_rate = len(winning_trades) / n_trades * 100
                avg_return = np.mean([t['return_pct'] for t in completed_trades])
            else:
                win_rate = 0.0
                avg_return = 0.0
            
            return {
                'total_return': total_return,
                'total_return_pct': total_return_pct,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'n_trades': n_trades,
                'signal_count': signal_count,
                'win_rate': win_rate,
                'avg_return': avg_return,
                'final_equity': equity_curve[-1]
            }
            
        except Exception as e:
            logger.error(f"backtest_single error: {e}")
            return {
                'total_return': -float('inf'),
                'total_return_pct': -float('inf'),
                'sharpe_ratio': -float('inf'),
                'max_drawdown': 100.0,
                'n_trades': 0,
                'win_rate': 0.0,
                'avg_return': 0.0,
                'final_equity': params.initial_capital
            }
        
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

        # --- Parameters to optimize (subset only) ---
        param_ranges = {
            "base_period": (10, 20),
            "min_period": (3, 10),
            "max_period": (20, 40),
            "fast_period": (2, 10),
            "slow_period": (5, 20),
            "channel_length": (20, 60),
            "channel_multi": (1.5, 3.0),
            "atr_period": (10, 20),
            "sl_multiplier": (1.2, 3.0),
            "tp_multiplier": (1.5, 3.0),
            "trail_multiplier": (1.0, 2.0),
            "steepness": (5, 15),
            "fast_steepness": (5, 15),
            "slow_steepness": (5, 15)
        }

        # --- Safety check ---
        assert all(
            isinstance(v, tuple) and len(v) == 2
            for v in param_ranges.values()
        ), "param_ranges must contain (low, high) tuples"

        # --- Sample ONLY the ranged parameters ---
        for name, (low, high) in param_ranges.items():
            if isinstance(low, int) and isinstance(high, int):
                params[name] = trial.suggest_int(name, low, high)
            else:
                params[name] = trial.suggest_float(name, low, high)

        # --- Force known-working parameters for the first trial ---
        if trial.number == 0:
            params.update({
                "fast_period": 5,
                "slow_period": 14,
                "channel_multi": 1.8,
                "sl_multiplier": 1.5
            })

        # --- STEP 3: DEBUG FORCE for first few trials ---
        if trial.number < 5:
            params["channel_multi"] = 1.5  # slightly higher to increase trigger probability
            params["sl_multiplier"] = 1.5  # make stop loss wider

        # --- HARD CONSTRAINTS ---
        if params["fast_period"] >= params["slow_period"]:
            print("Skipping invalid fast/slow combo:", params)
            return None  # tells Optuna to ignore this trial
        if params["min_period"] >= params["base_period"]:
            return -10.0
        if params["max_period"] <= params["base_period"]:
            return -10.0

        try:
            # --- Run backtest ---
            result = self.run_backtest(
                files=files[:n_files],
                sample_size=sample_size,
                **params
            )

            print("Trial params:", params)
            trades = result.get("total_trades", 0)

            # --- Retry/backoff for first few trials if zero trades ---
            if trades == 0 and trial.number < 5:
                # slight nudge for first few trials
                params["channel_multi"] *= 1.5  # bigger nudge
                params["sl_multiplier"] *= 1.5
                result = self.run_backtest(files=files[:n_files], sample_size=sample_size, **params)
                trades = result.get("total_trades", 0)
                if trades == 0:
                    print("⚠️ Still no trades for params:", params)
                    return -10.0

            # --- Penalize zero-trade trials ---
            if trades == 0:
                print("⚠️ No trades for params:", params)
                return -10.0

            # --- Return objective metric (Sharpe) ---
            return result.get("sharpe", 0.0)

        except Exception as e:
            print(f"Backtest error: {e}")
            return -10.0


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

                            