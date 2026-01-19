#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use python3 if available; fall back to python (Git Bash + alias-safe)
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: python not found in PATH"; exit 2; }

seed=7
nTrials=1500
nFiles=300
fillOpt="next_open"

# -----------------------------
# Anti-degeneracy fixes
# -----------------------------
# Trade gating (raise this; 1-trade eligibility is PF-gaming heaven)
min_trades=8
trades_baseline=8.0
trades_k=0.5

# PF / scoring (reduce PF dominance and remove score exponent blow-ups)
pf_baseline=1.15
pf_k=2.5
weight_pf=0.55
score_power=1.0

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

pf_cap=4.0

# Optional: if your script supports it, increasing loss_floor helps PF degeneracy.
# Leave commented if Vidya_RSI.py doesn't have this flag.
loss_floor=0.01

CMD=(
  "$PYTHON_BIN" Vidya_RSI.py
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
  --score-power "$score_power"

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

# If Vidya_RSI.py supports --loss-floor, add it; otherwise skip.
if "$PYTHON_BIN" Vidya_RSI.py --help 2>/dev/null | grep -q -- "--loss-floor"; then
  CMD+=( --loss-floor "$loss_floor" )
fi

echo "-------------------------------------------------------"
echo "PHASE-7: FINAL SNIPER COMPOUNDING"
echo "-------------------------------------------------------"
echo "Python: $PYTHON_BIN"
echo "Cmd: ${CMD[*]}"
echo "-------------------------------------------------------"

"${CMD[@]}"
