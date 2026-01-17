#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- LOCKING IN THE SNIPER SETTINGS ---
seed=7
nTrials=1500      # Higher trials to fine-tune the 6/6 reactivity zone
nFiles=160
fillOpt="next_open"

# 1) Trade gating (REWARDING MULTI-TRADES)
min_trades=1           
trades_baseline=3.0    # Reward moving from 1 trade to 3 trades
trades_k=1.2           # STEEP reward for getting that 2nd or 3rd trade

# 2) PF & Quality (MAINTAINING THE 5.0 PF)
pf_baseline=1.15      
pf_k=2.5               
weight_pf=0.70         # QUALITY IS PARAMOUNT: 70% weight on PF/Returns

# 3) Signal Settings (The "Aggression" Sweet Spot)
threshold_fixed=0.012  # Slightly higher to ensure we only take high-conviction moves
vol_floor_mult_fixed=0.05 
time_stop=48           # Give the "Sniper" plenty of room to let the trend play out

CMD=(
  "python3" Vidya_RSI.py
  --optimize
  --seed "$seed"
  --trials "$nTrials"
  --files "$nFiles"
  --fill "$fillOpt"
  --data_dir "data"

  --min-trades "$min_trades"
  --trades-baseline "$trades_baseline"
  --trades-k "$trades_k"

  --pf-baseline "$pf_baseline"
  --pf-k "$pf_k"
  --weight-pf "$weight_pf"
  --score-power 1.5      # High contrast to separate 'good' from 'legendary'

  --threshold-fixed "$threshold_fixed"
  --vol-floor-mult-fixed "$vol_floor_mult_fixed"
  --opt-time-stop
  --min-tp2sl 0.8

  # Focus the search space around your successful discovery
  --opt-vidya           
  --opt-fastslow        
)

echo "-------------------------------------------------------"
echo "PHASE-7: FINAL SNIPER COMPOUNDING"
echo "Baseline: 3 trades | PF Weight: 70% | High Contrast Scoring"
echo "-------------------------------------------------------"

"${CMD[@]}"