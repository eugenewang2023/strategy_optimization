#!/bin/bash
set -e

###=================================================================================
### ULTRA EASY MODE for Vidya_RSI.py
### Goal: get NON-ZERO hits fast (remove cliffs + reduce pruning)
### After you see hits, tighten step-by-step.
###=================================================================================

# Core Optimization Settings
nTrials=600
nFiles=120
fillOpt=next_open

# 1) Trade Volume (very easy eligibility)
min_trades=2
trades_baseline=6
trades_k=0.15
max_trades=120
max_trades_k=0.03

# 2) Return protection (remove cliffs for exploration)
# NOTE: keep penalty enabled but soft; remove return floor
ret_floor=0.0
ret_floor_k=4.0
penalty_center=-0.02
penalty_k=4.0

# 3) Profit Factor constraints (soft)
pf_cap=20.0
pf_baseline=1.05
pf_k=0.80
pf_floor=1.00
pf_floor_k=1.00

# 4) Scoring weight + coverage (make coverage NOT a cliff)
weight_pf=0.55
score_power=1.0
coverage=0.35
coverage_k=2.0

# 5) Execution & Risk
commission_per_side=0.0006
loss_floor=0.001
cooldown=1
time_stop=12

# 6) TP/SL constraint: DISABLE AUTO + make it permissive to avoid pruning
# min_tp2sl_eff is used as: prune if sl <= min_tp2sl * tp
# So set it LOW so trials survive.
min_tp2sl=0.60

CMD="python Vidya_RSI.py \
  --optimize \
  --trials $nTrials \
  --files $nFiles \
  --fill $fillOpt \
  --commission_rate_per_side $commission_per_side \
  --loss_floor $loss_floor \
  --cooldown $cooldown \
  --time-stop $time_stop \
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
  --coverage-target $coverage \
  --coverage-k $coverage_k \
  --min-tp2sl $min_tp2sl \
  --penalty \
  --penalty-ret-center $penalty_center \
  --penalty-ret-k $penalty_k \
  --opt-adaptive \
  --opt-fastslow \
  --opt-cooldown \
  --opt-time-stop"

echo "-------------------------------------------------------"
echo "Starting Optuna Study: Vidya_RSI.py (ULTRA EASY MODE)"
echo "Trials: $nTrials | Files: $nFiles | Fill: $fillOpt"
echo "min_trades=$min_trades | coverage_target=$coverage (k=$coverage_k)"
echo "ret_floor=$ret_floor | pf_floor=$pf_floor | pf_cap=$pf_cap"
echo "min_tp2sl=$min_tp2sl (tp2sl-auto disabled to reduce pruning)"
echo "cooldown=$cooldown | time_stop=$time_stop"
echo "-------------------------------------------------------"
echo "Executing: $CMD"
eval $CMD
