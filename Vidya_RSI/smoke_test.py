from pathlib import Path
import Bayes_opt_adapt_RSI as bapt

def smoke_test_backtest(opt, backtest_params, strategy_params):
    """
    Run a single-file backtest for smoke testing.
    """
    files = backtest_params["files"]
    sample_size = backtest_params["sample_size"]
    n_files = 1  
    return opt.run_backtest(files=files[:n_files], sample_size=sample_size, **strategy_params)

def main():
    # 1️⃣ Find parquet files
    data_folder = "data"  
    files = list(Path(data_folder).glob("*.parquet"))

    if not files:
        print(f"No parquet files found in '{data_folder}'. Exiting.")
        return

    print(f"Found {len(files)} files. Using first file for smoke test: {files[0]}")

    # 2️⃣ Set parameters - Increased sample size slightly for better signal probability
    backtest_params = {
        "files": files[:1],
        "sample_size": 10000, 
    }

    # Strategy parameters: Adjusted to be more "trigger-happy" for Heikin Ashi
    strategy_params = {
        "base_period": 14,
        "min_period": 5,
        "max_period": 30,
        "fast_period": 5,
        "slow_period": 20,
        "channel_length": 50,
        "channel_multi": 2.0,
        "atr_period": 14,
        "sl_multiplier": 3.0,
        "tp_multiplier": 5.0,
        "trail_multiplier": 3.0,
        "steepness": 1.0,        # <--- Lowered significantly
        "fast_steepness": 1.0,   # <--- Lowered significantly
        "slow_steepness": 1.0,   # <--- Lowered significantly
    }

    # 3️⃣ Initialize optimizer instance
    opt = bapt.BayesianRSIATROptimizer()

    # 3a️⃣ Debug: Correctly unpack the new HA-based signal logic
    print("\n=== Debugging Indicator Calculation ===")
    df = opt.load_data_chunk(backtest_params["files"][0], sample_size=backtest_params["sample_size"])
    params = bapt.StrategyParams(**strategy_params)
    
    # Run indicators first
    df_with_inds = opt.calculate_indicators_fast(df, params)

    print(f"Adaptive RSI (first 10 valid): {df_with_inds['adaptive_rsi'].dropna().head(10).values}")
    print(f"Slow RSI (first 10 valid): {df_with_inds['slow_rsi'].dropna().head(10).values}")
    
    # Call the signal function directly for debug inspection
    # Matches the 2-value return: buy_signal, sell_signal
    buy_signals, sell_signals = opt.calculate_signals_vectorized(
        df_with_inds['close'].values,
        df_with_inds['ha_close'].values,
        df_with_inds['ha_low'].values,
        df_with_inds['adaptive_rsi'].values,
        df_with_inds['slow_rsi'].values,
        df_with_inds['lower_channel'].values,
        df_with_inds['trail_offset'].values
    )
    
    print(f"Total Buy Signals detected: {buy_signals.sum()}")
    print(f"Total Sell Signals detected: {sell_signals.sum()}")

    # 4️⃣ Run full backtest logic
    print("\nRunning smoke test backtest...")
    result = smoke_test_backtest(opt, backtest_params, strategy_params)

    # 5️⃣ Print results
    print("\n=== Smoke Test Results ===")
    if result.get("total_trades", 0) == 0:
        print("⚠️  No trades executed. Tips:")
        print("1. Check if 'adaptive_rsi' ever crosses 'slow_rsi' in your data.")
        print("2. Ensure 'channel_multi' isn't forcing an immediate exit.")
    else:
        print(f"Total Trades: {result.get('total_trades')}")
        print(f"Average Sharpe Ratio: {result.get('avg_sharpe'):.2f}")
        print(f"Average Return: {result.get('avg_return_pct'):.2f}%")
        print(f"Average Max Drawdown: {result.get('avg_max_dd'):.2f}%")
        print(f"Average Win Rate: {result.get('avg_win_rate'):.2f}%")

    print("\nSmoke test completed.")

if __name__ == "__main__":
    main()