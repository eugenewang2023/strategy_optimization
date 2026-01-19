#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

seed=7
nTrials=1500
nFiles=300
fillOpt="next_open"

# Trade gating
min_trades=1
trades_baseline=3.0
trades_k=1.2

# PF / scoring
pf_baseline=1.15
pf_k=2.5
weight_pf=0.70

# Signal
threshold_fixed=0.012
vol_floor_mult_fixed=0.05

# Objective / penalties
objective_mode="hybrid"
obj_penalty_mode="both"
zero_loss_target=0.05
zero_loss_k=12
cap_target=0.3
cap_k=6
min_glpt=0.003
min_glpt_k=12

pf_cap=4.0   # keep consistent with your run output (change if desired)

CMD=(
  python3 Vidya_RSI.py
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
  --score-power 1.5

  --threshold-fixed "$threshold_fixed"
  --vol-floor-mult-fixed "$vol_floor_mult_fixed"

  --pf-cap "$pf_cap"

  --objective-mode "$objective_mode"
  --obj-penalty-mode "$obj_penalty_mode"
  --zero-loss-target "$zero_loss_target"
  --zero-loss-k "$zero_loss_k"
  --cap-target "$cap_target"
  --cap-k "$cap_k"
  --min-glpt "$min_glpt"
  --min-glpt-k "$min_glpt_k"

  --opt-time-stop
  --min-tp2sl 0.8

  --opt-vidya
  --opt-fastslow
)

echo "-------------------------------------------------------"
echo "PHASE-7: FINAL SNIPER COMPOUNDING"
echo "-------------------------------------------------------"

"${CMD[@]}"
