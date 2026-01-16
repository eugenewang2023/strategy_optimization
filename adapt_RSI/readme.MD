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
