1/25/2026
Key fixes:
Added Windows encoding support: The setup_windows_encoding() function handles UTF-8 encoding issues on Windows console
Replaced Unicode checkmarks: Changed ✓ and ✗ to [YES] and [NO] which work on all Windows consoles
Fixed the script completely: The Unicode error should no longer occur

sh run.sh
Loaded 7 valid optimization runs:
  - adapt_half_RSI_per_ticker_20260120_235614.csv
  - adapt_RSI_per_ticker_20260120_231930.csv
  - half_RSI_per_ticker_20260121_000047.csv
  - half_RSI_v0.2_per_ticker_20260125_184151.csv
  - vidya_per_ticker_20260121_meta.csv
  - vidya_RSI_per_ticker_20260121_001237.csv
  - vidya_RSI_per_ticker_next_open_20260125_210339.csv

==================================================
AGGREGATE RUN COMPARISON
==================================================
                             run_id  avg_total_return  median_total_return  avg_pf_eff  median_pf_eff  avg_maxdd  eligible_rate  avg_trades  num_tickers
      half_RSI_v0.2_20260125_184151            0.3324               0.2765         NaN            NaN    -0.1098            NaN      7.7959          147
           half_RSI_20260121_000047            0.2912               0.2087         NaN            NaN    -0.0817            NaN      6.3741          147
     adapt_half_RSI_20260120_235614            0.2574               0.1894         NaN            NaN    -0.0972            NaN      6.4286          147
                vidya_20260121_meta            0.2238               0.1814     21.3420         3.8897    -0.0544         1.0000      7.8163          147
          adapt_RSI_20260120_231930            0.2047               0.1799         NaN            NaN    -0.0891         0.9932     10.1565          147
          vidya_RSI_20260121_001237            0.1565               0.1557     46.2702         2.9006    -0.0732         1.0000      6.4898          147
vidya_RSI_next_open_20260125_210339               NaN                  NaN     57.7408         2.3334    -0.0648            NaN         NaN          147

==================================================
BEST RUN PER TICKER
==================================================
                        run_id  num_best_tickers
 half_RSI_v0.2_20260125_184151                42
adapt_half_RSI_20260120_235614                36
      half_RSI_20260121_000047                30
     adapt_RSI_20260120_231930                25
     vidya_RSI_20260121_001237                14

==================================================
PER-TICKER MATRIX
==================================================
Matrix shape: (147, 5)
Saved to per_ticker_return_matrix.csv

==================================================
DIAGNOSTICS
==================================================
Files WITH 'total_return' column (6):
  [YES] adapt_half_RSI_per_ticker_20260120_235614.csv
  [YES] adapt_RSI_per_ticker_20260120_231930.csv
  [YES] half_RSI_per_ticker_20260121_000047.csv
  [YES] half_RSI_v0.2_per_ticker_20260125_184151.csv
  [YES] vidya_per_ticker_20260121_meta.csv
  [YES] vidya_RSI_per_ticker_20260121_001237.csv

Files WITHOUT 'total_return' column (1):
  [NO]  vidya_RSI_per_ticker_next_open_20260125_210339.csv

Saved comparison outputs to: C:\Backup\_my\my_Github\strategy_optimization\compare_optimization\comparison_output

1/21/2026
run.sh
Loaded 5 optimization runs:
  - adapt_half_RSI_per_ticker_20260120_235614.csv
  - adapt_RSI_per_ticker_20260120_231930.csv
  - half_RSI_per_ticker_20260121_000047.csv
  - vidya_per_ticker_20260121_meta.csv
  - vidya_RSI_per_ticker_20260121_001237.csv

=== AGGREGATE RUN COMPARISON ===
         run_id  avg_total_return  median_total_return  avg_pf_eff  median_pf_eff  avg_maxdd  eligible_rate  avg_trades  num_tickers
20260121_000047            0.2912               0.2087         NaN            NaN    -0.0817            NaN      6.3741          147
20260120_235614            0.2574               0.1894         NaN            NaN    -0.0972            NaN      6.4286          147
  20260121_meta            0.2238               0.1814     21.3420         3.8897    -0.0544         1.0000      7.8163          147
20260120_231930            0.2047               0.1799         NaN            NaN    -0.0891         0.9932     10.1565          147
20260121_001237            0.1565               0.1557     46.2702         2.9006    -0.0732         1.0000      6.4898          147

=== BEST RUN PER TICKER (by total_return) ===
         run_id  num_best_tickers
20260120_235614                46
20260121_000047                44
20260120_231930                35
20260121_001237                22

Saved comparison outputs to: C:\Backup\_my\my_Github\strategy_optimization\compare_optimization\comparison_output
