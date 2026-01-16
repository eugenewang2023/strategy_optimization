#!/bin/bash
set -euo pipefail

###=================================================================================
### Configuration for Bayes_opt_adapt_RSI.py
### PHASE 2 RECOVERY: Finding robust signals that survive 'next_open'
###=================================================================================

# Core run size
nTrails=2000
nFiles=200

# We use next_open to ensure results are tradable. 
# If results are still '0', we can temporarily go back to same_close.
fillOpt=next_open

# -------------------------
# 1) Trade Volume (Discovery Floor)
# - Lowered from 10 to 6 to find signals with longer RSI periods.
# -------------------------
min_trades=3
trades_baseline=8
trades_k=0.40
max_trades=60
max_trades_k=0.08

# -------------------------
# 2) Return & Penalty
# - Accept break-even or slightly negative results to guide Optuna.
# -------------------------
ret_floor=0.00
ret_floor_k=10

penalty_center=-0.01
penalty_k=6.0

# -------------------------
# 3) Profit Factor Quality
# - Lower floor to 1.0 to capture 'early' signal discovery.
# -------------------------
pf_cap=5.0
pf_baseline=1.4
pf_k=1.1
pf_floor=1.0
pf_floor_k=3.0

# -------------------------
# 4) Coverage & Weighting
# - Lowered coverage target to 60% to find where the strategy works best.
# -------------------------
weight_pf=0.75
score_power=1.0

coverage=0.85
coverage_k=7.0

# -------------------------
# 5) Risk & Execution
# -------------------------
commission_per_side=0.0006
loss_floor=0.001
cooldown=1
time_stop=10

# -------------------------
# 6) TP/SL Constraint (Phase 2 Strictness)
# - Forcing a healthy Risk/Reward profile.
# -------------------------
tp2sl_auto=1
tp2sl_base=1.2
tp2sl_sr0=30
tp2sl_k=0.01
tp2sl_min=1.1
tp2sl_max=2.5

# -------------------------
# 7) Optimization Toggles
# -------------------------
opt_adaptive=1
opt_cooldown=1

###=================================================================================
### Execution Command
###=================================================================================

CMD="python Bayes_opt_adapt_RSI.py \
  --trials $nTrails \
  --files $nFiles \
  --fill $fillOpt \
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
  --coverage-target $coverage \
  --coverage-k $coverage_k \
  --commission_rate_per_side $commission_per_side \
  --loss_floor $loss_floor \
  --cooldown $cooldown \
  --time-stop $time_stop"

if [[ "$tp2sl_auto" == "1" ]]; then
  CMD="$CMD --tp2sl-auto --tp2sl-base $tp2sl_base --tp2sl-sr0 $tp2sl_sr0 --tp2sl-k $tp2sl_k --tp2sl-min $tp2sl_min --tp2sl-max $tp2sl_max"
else
  CMD="$CMD --min-tp2sl 1.1"
fi

if [[ "$opt_adaptive" == "1" ]]; then
  CMD="$CMD --opt-adaptive"
fi

if [[ "$opt_cooldown" == "1" ]]; then
  CMD="$CMD --opt-cooldown"
fi

echo "================================================================================"
echo "Starting PHASE 2 DISCOVERY Run"
echo "Strategy: Adaptive RSI (Robustness Mode)"
echo "Target Fill: $fillOpt"
echo "================================================================================"
eval "$CMD"