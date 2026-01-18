#!/bin/bash
set -euo pipefail
set +x   # force-disable xtrace if inherited

###=================================================================================
### PHASE-4 (Two-Regime High-Density Aggression)
### Goal: Push trades to 10+ | Balanced Trend/Chop scoring | Forced Rotation
###=================================================================================

# --- run from script directory ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- python resolver ---
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
  else
    echo "ERROR: python/python3 not found in PATH" >&2
    exit 1
  fi
fi

# --- reproducibility ---
seed=7

nTrials=1500      # Sufficient trials to explore the sensitive search space
nFiles=300
fillOpt=next_open

# 1) Trade gating (AGGRESSIVE TARGETS)
min_trades=5
trades_baseline=10.0  # Crucial: Rewards peak at 10 trades per ticker
trades_k=0.4          # Punishes low frequency heavily
max_trades=250        # High ceiling for momentum runners
max_trades_k=0.03

# 2) Returns & penalties
ret_floor=0.01        # Low enough to capture micro-momentum
ret_floor_k=8.0       # Steep tail protection
penalty_center=0.0    
penalty_k=6.0

# 3) PF shaping
pf_cap=5.0            # Keeps Optuna focused on consistency over "lottery" wins
pf_baseline=1.02      
pf_k=1.2              
pf_floor=1.0
pf_floor_k=1.5

# 4) Scoring balance
weight_pf=0.40        # Favors trade count and stability over raw PF
score_power=1.1       
coverage_target=0.85  # Targets high participation across tickers
coverage_k=8.0

# 5) Execution (FORCED ROTATION)
commission_per_side=0.0006
loss_floor=0.001
cooldown=1            # FORCED: Immediate re-entry allowed
time_stop=15          # FORCED: Fast rotation (15-bar max hold)

# 6) Risk constraint
min_tp2sl=1.10

# 7) SIGNAL REGIME (LOCKED)
threshold_mode="fixed"
threshold_fixed=0.04
vol_floor_len=50
vol_floor_mult_fixed=0.55

# 8) Two-Regime Scoring Knobs
# Pivot point: 1.2 Return/DD ratio defines a "Trend"
trend_center=1.20
trend_k=4.0

CMD=(
  "$PYTHON_BIN" Bayes_opt_adapt_RSI.py
  --optimize
  --seed "$seed"
  --trials "$nTrials"
  --files "$nFiles"
  --fill "$fillOpt"

  --penalty
  --penalty-ret-center "$penalty_center"
  --penalty-ret-k "$penalty_k"

  --min-trades "$min_trades"
  --trades-baseline "$trades_baseline"
  --trades-k "$trades_k"
  --max-trades "$max_trades"
  --max-trades-k "$max_trades_k"

  --ret-floor "$ret_floor"
  --ret-floor-k "$ret_floor_k"

  --pf-cap "$pf_cap"
  --pf-baseline "$pf_baseline"
  --pf-k "$pf_k"
  --pf-floor "$pf_floor"
  --pf-floor-k "$pf_floor_k"

  --weight-pf "$weight_pf"
  --score-power "$score_power"

  --commission_rate_per_side "$commission_per_side"
  --loss_floor "$loss_floor"

  --cooldown "$cooldown"
  --time-stop "$time_stop"
  --opt-time-stop

  --min-tp2sl "$min_tp2sl"

  --coverage-target "$coverage_target"
  --coverage-k "$coverage_k"

  --threshold-mode "$threshold_mode"
  --threshold-fixed "$threshold_fixed"
  --vol-floor-len "$vol_floor_len"
  --vol-floor-mult-fixed "$vol_floor_mult_fixed"

  --trend-center "$trend_center"
  --trend-k "$trend_k"

  --opt-adaptive
  --opt-fastslow
)

echo "-------------------------------------------------------"
echo "PHASE-4: TWO-REGIME AGGRESSION (Target: 10+ trades)"
echo "Baseline: ${trades_baseline} | Cooldown: ${cooldown} | Time-Stop: ${time_stop}"
echo "Trend Pivot: ${trend_center} | Coverage Target: ${coverage_target}"
echo "-------------------------------------------------------"

# Execute
"${CMD[@]}"