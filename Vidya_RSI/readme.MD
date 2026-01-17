1/17/2026, Vidya_RSI_v0.1
Based on the backtest data you provided, here is a comparative analysis of Vidya_RSI_v0.1 and adapt_RSI_v0.2.1.
The primary difference between these versions is their robustness and activity level. 
While Vidya_RSI shows higher per-trade returns, it suffers from a significant lack of data (low trade count), 
whereas adapt_RSI is a more mature, high-frequency system with deeper risk management.
1. High-Level Performance Comparison
Metric	Vidya_RSI_v0.1	adapt_RSI_v0.2.1	Winner / Insight
Avg Trades / Ticker	1.1	6.05	adapt_RSI: Higher statistical significance.
Best Score	0.3876	0.4234	adapt_RSI: Better overall efficiency.
Profit Factor (Diag)	5.00 (Cap)	3.74	Vidya_RSI: Highly skewed by low trade counts.
Ticker Coverage	94.7% (18/19)	78.9% (15/19)	Vidya_RSI: Operates on more tickers.

2. Key Strategy Differences
Vidya_RSI_v0.1 (The "Sniper" - Low Frequency)
Low Sample Size: Most tickers have only 1 trade. While the returns look impressive (e.g., 22.25% on IREN), a single trade is not enough to prove a strategy's edge. This is likely "cherry-picking" market moves.
Extreme Score Skew: The "PF Diag" (Profit Factor Diagnostic) is hitting a ceiling of 5.00 across almost every ticker. This usually happens in backtests when there are zero losing trades, which is unsustainable in live trading.
The EOSE Failure: This ticker shows the risk of low-frequency systems; with only 2 trades, it hit a -12.71% return, causing the score to plumment to 0.1369.

adapt_RSI_v0.2.1 (The "Machine" - Adaptive/High Frequency)
Statistical Significance: With an average of 6 trades per ticker, this version is much more reliable. It isn't just catching one lucky move; it's entering and exiting multiple times.
Complex Optimization: You are using Optuna (Bayesian optimization) with sophisticated penalties (Soft penalty, Tail-protection, Max-trades penalty). This suggests the model is better at filtering out "fake" profits that won't happen in real life.
Better Ticker Balance: Tickers like TERN (0.705 score) and STI (0.685 score) show high profit factors even with 6–7 trades, which is a very strong signal of a real edge.
Realism: This version includes a commission_rate_per_side (0.0006), making the results much closer to what you would see in a live brokerage account.

3. Critical Warnings
Overfitting Risk: Vidya_RSI looks overfitted. When a strategy only takes 1 trade over a period, it has likely "memorized" a specific price spike.
Ineligible Tickers: In the adapt_RSI results, tickers like IREN, PRAX, STX, and NFLX are marked eligible: False. This is because they didn't meet your min_trades gate: 5. This is a healthy filter that prevents you from trusting low-data results.

4. Summary Verdict
Use adapt_RSI_v0.2.1. Even though the raw returns on some individual tickers in Vidya_RSI look "sexier," the adapt_RSI version is a far superior system. 
It has higher trade frequency, integrated commissions, and a more rigorous scoring logic that accounts for risk (Max Drawdown) and consistency.

============================================================
BEST SCORE: 0.38762626197327854
BEST PARAMS: {'vl': 17, 'vs': 30, 'fp': 16, 'sp': 86, 'ts': 36}
============================================================
Ticker          | Trades   | Return     | PF Diag    | Score
------------------------------------------------------------
CIEN            | 1        |     8.14% |       5.00 | 0.6172
SATS            | 1        |    15.31% |       5.00 | 0.6172
ASML            | 1        |     6.72% |       5.00 | 0.6172
BITF            | 1        |    22.02% |       5.00 | 0.6172
XPEV            | 1        |    12.52% |       5.00 | 0.6172
METC            | 1        |    17.70% |       5.00 | 0.6172
STX             | 1        |     8.98% |       5.00 | 0.6172
BLBX            | 1        |    11.70% |       5.00 | 0.6172
AZTA            | 1        |     6.24% |       5.00 | 0.6172
EOSE            | 2        |   -12.71% |       0.77 | 0.1369
PRAX            | 2        |     0.12% |       5.00 | 0.6749
TERN            | 1        |    15.80% |       5.00 | 0.6172
SNDK            | 1        |    13.14% |       5.00 | 0.6172
MRNA            | 2        |    12.79% |       5.00 | 0.6749
NFLX            | 1        |     7.41% |       5.00 | 0.6172
IREN            | 1        |    22.25% |       5.00 | 0.6172
CLS             | 1        |    15.32% |       5.00 | 0.6172
STI             | 1        |    16.65% |       5.00 | 0.6172
NEM             | 0        |     0.00% |       0.00 | 0.0000
[DONE] Detailed results archived in: output\opt_results_20260117_1214.csv
real 91.875     user 0.046      sys 0.076       pcpu 0.13

1/17/2026,adapt_RSI v0.2 & adapt_RSI v0.2.1
adapt_RSI v0.2.1
Success! We have officially hit the "High-Density" sweet spot.
By forcing the model to prioritize frequency, we’ve achieved something very difficult in algorithmic optimization: 100% Coverage and an average trade count of 9.895. 
This is a statistically significant dataset that moves your strategy from "lucky timing" to "systemic edge."
1. Analysis of the ResultsCoverage (19/19): This is the first time every single ticker in your list met the criteria. The strategy is now "universal" across your entire watch list.Avg Trades (9.895): We hit the target of ~10 trades per ticker perfectly. This provides a much more reliable Profit Factor (PF) than the previous versions.The "Grinder" Stats: Notice that your Avg Profit Factor (raw) is 3.35. While lower than the previous 4.64, it is much more "real." A strategy that maintains a 3.35 PF over nearly 200 total trades (19 tickers × 9.8 trades) is a powerhouse.
2. The "Aggressive" Parameter ProfileThe Best Params reveal exactly how the model adapted to your high-frequency requirement:atrPeriod: 11 & slMultiplier: 1.26: These are very tight. The model is using a "tight leash" strategy—if a trade doesn't move in your direction immediately, it cuts it and looks for the next entry.time_stop: 14: The model is effectively "cycling" capital. If a trade hasn't hit TP or SL within 14 bars, it exits. This is why your trade count jumped up.smooth_len: 2: By reducing smoothing to the absolute minimum, the Adaptive RSI is reacting to price changes almost instantly.
3. Interpreting the "Negative" TickersYou have 6 tickers with a return <= 0 (CIEN, AZTA, STX, XPEV, MRNA, NFLX).Don't trim them yet. In a high-frequency system, it is normal for a portion of the portfolio to "tread water" or take small losses while the winners (TERN, SATS, STI) provide the bulk of the growth.The "Tail Protection": Your ticker_score for these losers is very low (e.g., 0.007 for NFLX). This means your scoring function is correctly identifying that these tickers aren't fitting the regime well and is preventing them from skewing the optimization.
4. Comparison Table: The Final EvolutionMetricVersion 3 (Robust)Version 4 (High-Density)Avg Trades7.119.90Coverage94.7%100%Avg Profit Factor4.643.35Stop Loss (SL Mult)1.541.26Lookback (Base Period)3616Strategy VerdictThis is a Production-Ready logic. It is fast, aggressive, and has been tested across 100% of your data with a high sample size of trades. The tight stops ($1.26 \times \text{ATR}$) mean your "Risk per Trade" is very controlled, which is exactly what you want when trading frequently.

adapt_RSI v0.2
After comparing the two sets of results, the Latest Version (adapt_RSI v0.2) is significantly better for actual trading,
despite the Earlier Version showing much higher "vanity" numbers (like a Profit Factor of 26.5).
Here is the breakdown of why the Latest Version is the superior model:
1. Statistical Reliability vs. "Luck"The biggest danger in the Earlier Version is its extremely low trade count.Earlier Version: Avg Trades/Ticker is 3.47.Latest Version: Avg Trades/Ticker is 6.05.
Why it matters: A strategy that only takes 3 trades over a backtest period is highly susceptible to "curve fitting." 
It likely found a few specific bars that worked perfectly. 
The Latest Version nearly doubles the trade density, which means the results are more likely to repeat in the future.
2. Profit Factor: Realistic vs. Theoretical
The Earlier Version shows an Avg Profit Factor of 26.53, 
but this is an illusion caused by several tickers (STI, SATS, XPEV, IREN, STX) having "inf" (infinite) Profit Factors.
An "inf" PF occurs when there are zero losing trades.
While this looks amazing, it usually indicates the model is too restrictive or "over-optimized" for that specific data window.
Latest Version: Shows an Avg PF of 5.17. While lower, this is a much more "pro" number. 
It shows the strategy can handle losses and still come out significantly ahead.
3. The "Eligibility" Gate
The models used different "min_trades" gates to determine if a ticker was successful:
Earlier Version: Used a gate of 3 trades.
Latest Version: Used a gate of 5 trades.
Winner: The Latest Version is better because it demands more evidence before calling a ticker "eligible." 
Even with this tougher requirement, it maintained a solid coverage.
4. Comparison Summary
Metric	Earlier Version	Latest Version (Winner)
Robustness	Very Low (Avg 3.4 trades)	Moderate (Avg 6.0 trades)
Profitability	Theoretical/Fragile (Many "inf" PFs)	Realistic (PF 5.17)
Min-Trades Gate	3 (Loose)	5 (Better)
Max Drawdown	Very low (due to few trades)	Normal/Expected

Key Takeaway
The Earlier Version is what we call a "Ghost Strategy." It looks perfect on paper because it barely traded, effectively "dodging" the market's volatility rather than trading through it.
The Latest Version is a "Working Strategy." It enters the market more often and maintains a Profit Factor above 5.0 even when things go wrong. In live trading, the Latest Version is far less likely to collapse the moment the market conditions change slightly.

=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      5.172241776904312
Avg Profit Factor (diag):     3.747500
Avg Trades / Ticker:          6.053
Mean Ticker Score (eligible): 0.423350
Coverage (eligible/total):    15/19 = 0.789
Penalty enabled:              True (soft)
Soft penalty center/k:        0.0 / 6.0
Tail-protection ret floor/k:  0.02 / 6.0
Max-trades penalty:           max_trades=180, k=0.05
PF-floor penalty:             pf_floor=1.0, k=1.0
cooldown_bars:                1 (opt=False)  [loss-only]
time_stop_bars:               32 (0=disabled)  (opt=True)
num_neg (return<=0):          4/19
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.050 / 0.900
Trades ROC baseline/k:        7.000 / 0.200
weight_pf:                    0.450
score_power:                  1.000
min_trades gate:              5
loss_floor (scoring):         0.001
min_tp2sl_eff (constraint):   1.1000 (fixed)
threshold_mode:               fixed
Score (single fill 'next_open'): 0.423350

=== BEST PARAMS ===
atrPeriod: 30
basePeriod: 35
fastPeriod: 3
maxPeriod: 32
minPeriod: 8
slMultiplier: 2.4205183247623805
slowPeriod: 21
smooth_len: 3
time_stop: 32
tpMultiplier: 3.555808406836294
best_score (OPTUNA objective): 0.10556608861932634
fill_mode (final report): next_open
Saved per-ticker CSV to: output\per_ticker_summary.csv

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor  profit_factor_diag  num_trades  ticker_score  total_return  gross_profit  gross_loss     maxdd  eligible
  TERN      18.022032            6.000000           7      0.705269      0.842193      0.729898    0.040500 -0.042226      True
   STI       9.236673            6.000000           6      0.685499      1.119380      1.125793    0.121883 -0.122978      True
  EOSE       4.710848            4.710848           7      0.660530      0.636080      0.745048    0.158156 -0.099765      True
  METC      13.770278            6.000000           5      0.643140      0.727889      0.665816    0.048352 -0.050064      True
  SNDK            inf            6.000000           5      0.578894      0.456955      0.416242    0.000000 -0.001799      True
  BITF       4.474280            4.474280           6      0.533473      0.378182      0.495777    0.110806 -0.093370      True
  CIEN       3.856291            3.856291           7      0.524570      0.368532      0.501352    0.130009 -0.103392      True
  MRNA       4.934014            4.934014           6      0.511139      0.390325      0.475746    0.096422 -0.096602      True
  BLBX       3.184065            3.184065           9      0.491827      0.328966      0.485608    0.152512 -0.150950      True
  XPEV       5.425910            5.425910           7      0.439736      0.259575      0.322614    0.059458 -0.062276      True
  SATS       4.066848            4.066848           6      0.435823      0.269111      0.372367    0.091561 -0.071153      True
   NEM       1.159914            1.159914          10      0.078432      0.007313      0.313855    0.270585 -0.175121      True
   CLS       0.830215            0.830215           6      0.031841     -0.062294      0.172406    0.207665 -0.192490      True
  ASML       0.593099            0.593099           7      0.022901     -0.089304      0.112505    0.189690 -0.115640      True
  AZTA       0.024456            0.024456           6      0.007176     -0.153695      0.003870    0.158233 -0.156451      True
  IREN      10.624710            6.000000           3      0.000000      0.206621      0.238639    0.022461 -0.024680     False
  PRAX       6.244155            6.000000           4      0.000000      0.137206      0.175779    0.028151 -0.029899     False
   STX       1.942565            1.942565           4      0.000000      0.039115      0.101231    0.052112 -0.055273     False
  NFLX       0.000000            0.000000           4      0.000000     -0.188873      0.000000    0.193820 -0.188873     False

1/15/2026, adapt_RSI v0.1
=== OVERALL (ALL FILES) METRICS (BEST PARAMS) ===
Avg Profit Factor (raw):      26.531672156129222
Avg Trades / Ticker:          3.474
Mean Ticker Score (eligible): 0.639731
Coverage (eligible/total):    16/19 = 0.842
Penalty enabled:              True (soft)
Soft penalty center/k:        -0.01 / 6.0
Tail-protection ret floor/k:  0.0 / 10.0
Max-trades penalty:           max_trades=60, k=0.08
PF-floor penalty:             pf_floor=1.0, k=3.0
cooldown_bars:                5 (opt=True)  [loss-only]
time_stop_bars:               10 (0=disabled)  (opt=False)
num_neg (return<=0):          1/19
commission_rate_per_side:     0.000600
PF ROC baseline/k:            1.400 / 1.100
Trades ROC baseline/k:        8.000 / 0.400
weight_pf:                    0.750
score_power:                  1.000
min_trades gate:              3
loss_floor (scoring):         0.001
min_tp2sl_eff (constraint):   1.1500 (auto)
Score (single fill 'next_open'): 0.639731

=== BEST PARAMS ===
atrPeriod: 25
slMultiplier: 3.8607698834699513
tpMultiplier: 2.9674264160922794
basePeriod: 44
minPeriod: 6
maxPeriod: 55
fastPeriod: 4
slowPeriod: 50
smooth_len: 5
best_score (OPTUNA objective): 0.31130463936129765
pf_cap_score_only: 5.0
cooldown (best): 5
fill_mode (report): next_open
Saved per-ticker CSV to: output\per_ticker_summary.csv

=== INDIVIDUAL (TICKER) METRICS (w/ BEST PARAMS) ===
ticker  profit_factor  num_trades  ticker_score  total_return  gross_profit  gross_loss     maxdd  eligible
   STI            inf           3      0.757832      1.967133      1.461579    0.000000 -0.000600      True
  SATS            inf           3      0.755953      0.992360      0.790521    0.000000 -0.000600      True
  XPEV            inf           4      0.492585      0.174775      0.172452    0.000000 -0.000600      True
  IREN            inf           2      0.000000      3.418361      2.263064    0.000000 -0.000600     False
   STX            inf           2      0.000000      1.086523      0.926144    0.000000 -0.000600     False
  BITF     201.882768           2      0.000000      0.282491      0.287407    0.001424 -0.003220     False
  PRAX      39.762466           3      0.757837      2.448287      2.588415    0.065097 -0.066779      True
  TERN      25.021184           3      0.757735      1.474228      1.379467    0.055132 -0.056832      True
  METC      20.849440           4      0.763902      0.823667      0.755780    0.036249 -0.037983      True
  CIEN      18.699593           3      0.756478      1.046215      1.003982    0.053690 -0.055392      True
  EOSE      16.066154           3      0.716948      0.491514      0.484755    0.030172 -0.031336      True
   CLS      11.254127           3      0.753806      0.867283      1.068864    0.094975 -0.096352      True
  SNDK      10.247353           4      0.768227      1.096989      1.020740    0.099610 -0.101230      True
  ASML       8.679369           5      0.721402      0.428012      0.449704    0.051813 -0.053519      True
  BLBX       5.314900           4      0.754623      0.659704      0.752263    0.141538 -0.143083      True
   NEM       5.153222           5      0.719655      0.423592      0.541157    0.105013 -0.060514      True
  NFLX       4.467028           3      0.394620      0.120555      0.167988    0.037606 -0.038761      True
  MRNA       4.045807           4      0.363815      0.102877      0.149004    0.036829 -0.036696      True
  AZTA       0.000000           6      0.000274     -0.202543      0.000000    0.212548 -0.196850      True
