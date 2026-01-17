#!/bin/bash

###=================================================================================
### Configuration for Bayes_opt_adapt_half_RSI.py
### Final Goal: Balance the high-ATR winners (STI) with the low-ATR winners (NFLX).
###=================================================================================

nTrails=2000
nFiles=200
fillOpt=next_open

# 1. Trade Volume (Higher Quality Gate)
# We move min_trades to 6 to ensure we aren't just seeing "lucky" streaks.
min_trades=6
trades_baseline=14
trades_k=0.7
max_trades=30
max_trades_k=0.15

# 2. Return & Drawdown Protection (Aggressive)
# We keep the high K-values to stay "allergic" to the NFLX-style -90% drawdowns.
ret_floor=0.02
ret_floor_k=20

penalty_center=0.05
penalty_k=25.0

# 3. Profit Factor Quality
pf_cap=8
pf_baseline=1.5
pf_k=1.5
pf_floor=1.1
pf_floor_k=12.0  # Even sharper cutoff to force the optimizer away from losers.

# 4. Scoring Weight
weight_pf=0.80   # Prioritize Profit Factor heavily to clean up the bottom 3 tickers.
score_power=1.0
coverage=0.7     # We want at least 14/19 tickers to be high-quality.

# 5. Execution & Risk
# We use the cooldown of 1 that your best params just discovered.
commission_per_side=0.0006
loss_floor=0.001
cooldown=1       
time_stop=6      

###=================================================================================
### Execution Command
###=================================================================================

# Adjusting tp2sl-min to 1.0 (1:1 ratio). 
# Your previous 1.4 was likely too restrictive for NFLX to exit profitably.

CMD="python Bayes_opt_adapt_half_RSI.py \
  --optimize \
  --trials $nTrails \
  --files $nFiles \
  --penalty \
  --penalty-ret-center $penalty_center \
  --penalty-ret-k $penalty_k \
  --min-trades $min_trades \
  --trades-baseline $trades_baseline \
  --trades-k $trades_k \
  --max-trades $max_trades \
  --max-trades-k $max_trades_k \
  --ret-floor $ret_floor \
  --ret-floor-k $ret_floor_k \
  --pf-cap $pf_cap \
  --pf-baseline $pf_baseline \
  --pf-k $pf_k \
  --pf-floor $pf_floor \
  --pf-floor-k $pf_floor_k \
  --weight-pf $weight_pf \
  --score-power $score_power \
  --commission_rate_per_side $commission_per_side \
  --loss_floor $loss_floor \
  --fill $fillOpt \
  --cooldown $cooldown \
  --time-stop $time_stop \
  --tp2sl-auto \
  --tp2sl-base 1.0 \
  --tp2sl-sr0 30 \
  --tp2sl-k 0.01 \
  --tp2sl-min 1.0 \
  --tp2sl-max 2.0 \
  --coverage-target $coverage \
  --opt-adaptive \
  --opt-cooldown"

echo "Executing: $CMD"
$CMD
