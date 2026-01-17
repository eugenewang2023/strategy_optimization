#!/usr/bin/env python3
"""
Adjusted test for smaller datasets
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from Bayes_opt_adapt_RSI import BayesianRSIATROptimizer, StrategyParams
import pandas as pd

def main():
    print("="*80)
    print("ADJUSTED PARAMETERS TEST FOR SMALL DATASETS")
    print("="*80)
    
    optimizer = BayesianRSIATROptimizer()
    files = optimizer.find_parquet_files()
    
    if not files:
        print("No files found")
        return
    
    file_to_test = files[0]
    print(f"\nTesting: {file_to_test.name}")
    
    df = optimizer.load_data_chunk(file_to_test)
    print(f"Data: {len(df)} rows")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Test 1: Original parameters (will likely have 0 trades)
    print(f"\n{'='*80}")
    print("TEST 1: ORIGINAL PARAMETERS")
    print(f"{'='*80}")
    params_orig = StrategyParams()
    results_orig = optimizer.backtest_single(df, params_orig)
    
    print(f"Trades: {results_orig['n_trades']}")
    print(f"Return: {results_orig['total_return_pct']:.2f}%")
    
    # Test 2: Adjusted parameters for small datasets
    print(f"\n{'='*80}")
    print("TEST 2: ADJUSTED PARAMETERS FOR 250 ROWS")
    print(f"{'='*80}")
    
    params_adjusted = StrategyParams(
        base_period=5,
        min_period=3,
        max_period=20,  # Reduced from 40
        fast_period=5,  # Reduced from 10
        slow_period=10, # Reduced from 40
        channel_length=50,  # Reduced from 100
        channel_multi=2.5,
        atr_period=5,
        sl_multiplier=3.0,
        tp_multiplier=4.0,
        trail_multiplier=2.0,
        steepness=10.0,
        fast_steepness=10.0,
        slow_steepness=10.0,
        sigmoid_margin=0.0,
        median_length=10,
        trim_percent=0.2,
        use_replacement=True,
        initial_capital=100000.0,
        position_size_pct=0.1,
        commission_pct=0.0006
    )
    
    print("Adjusted Parameters:")
    adjusted_params = [
        ('channel_length', 50, 100),
        ('max_period', 20, 40),
        ('slow_period', 10, 40),
        ('fast_period', 5, 10)
    ]
    
    for name, adjusted, original in adjusted_params:
        print(f"  {name}: {adjusted} (was {original})")
    
    print("\nRunning backtest with adjusted parameters...")
    results_adj = optimizer.backtest_single(df, params_adjusted)
    
    print(f"\nResults:")
    print(f"  Trades: {results_adj['n_trades']}")
    print(f"  Return: {results_adj['total_return_pct']:.2f}%")
    print(f"  Sharpe: {results_adj['sharpe_ratio']:.3f}")
    print(f"  Max DD: {results_adj['max_drawdown']:.2f}%")
    print(f"  Win Rate: {results_adj['win_rate']:.1f}%")
    
    # Test 3: Even more aggressive parameters
    if results_adj['n_trades'] == 0:
        print(f"\n{'='*80}")
        print("TEST 3: ULTRA-AGGRESSIVE PARAMETERS")
        print(f"{'='*80}")
        
        params_aggressive = StrategyParams(
            base_period=3,
            min_period=2,
            max_period=10,
            fast_period=3,
            slow_period=6,
            channel_length=20,
            channel_multi=2.0,
            atr_period=5,
            sl_multiplier=2.0,
            tp_multiplier=3.0,
            trail_multiplier=1.5,
            position_size_pct=0.15
        )
        
        print("Ultra-aggressive parameters:")
        print(f"  channel_length: 20 (was 100)")
        print(f"  max_period: 10 (was 40)")
        print(f"  slow_period: 6 (was 40)")
        
        results_agg = optimizer.backtest_single(df, params_aggressive)
        
        print(f"\nResults:")
        print(f"  Trades: {results_agg['n_trades']}")
        print(f"  Return: {results_agg['total_return_pct']:.2f}%")
        print(f"  Sharpe: {results_agg['sharpe_ratio']:.3f}")
    
    # Show data statistics
    print(f"\n{'='*80}")
    print("DATA STATISTICS")
    print(f"{'='*80}")
    
    if len(df) > 0:
        print(f"Total bars: {len(df)}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"Average daily change: {df['close'].pct_change().mean()*100:.2f}%")
        print(f"Volatility: {df['close'].pct_change().std()*100:.2f}%")
        
        # Calculate how many bars are actually available for trading
        max_lookback = max(
            params_orig.channel_length,
            params_orig.max_period,
            params_orig.slow_period
        )
        tradable_bars = len(df) - max_lookback
        print(f"\nWith channel_length={params_orig.channel_length}:")
        print(f"  Need {max_lookback} bars for indicators")
        print(f"  Only {tradable_bars} bars available for trading")

if __name__ == "__main__":
    main()