#!/bin/bash
set -euo pipefail
set +x   # force-disable xtrace if inherited

###=================================================================================
### PHASE-4 (High-Density Aggression & Trade Expansion)
### Goal: Push trades from ~7 -> ~10+ | Tighten stops | Maximize rotation
###=================================================================================

# --- run from script directory (avoids path issues when launched elsewhere) ---
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

nTrials=1200      # Increased for tighter search space exploration
nFiles=160
fillOpt=next_open

# 1) Trade gating (AGGRESSIVE TARGETS)
min_trades=5
trades_baseline=10    # Peak score reward only at 10+ trades
trades_k=0.4          # Steeper penalty for low trade counts
max_trades=250        # Higher ceiling for high-velocity tickers
max_trades_k=0.03

# 2) Returns & penalties
ret_floor=0.01        # Lowered to allow smaller, frequent winners
ret_floor_k=8.0       # Sharper protection against "dust" returns
penalty_center=0.0    
penalty_k=6.0

# 3) PF shaping
pf_cap=5.0            # Cap vanity PFs to focus on frequency
pf_baseline=1.02      # Lowered baseline for high-frequency acceptance
pf_k=1.2              # Increased sensitivity to quality
pf_floor=1.0
pf_floor_k=1.5

# 4) Scoring balance
weight_pf=0.40        # Shifted weight toward trade count (trades now ~60%)
score_power=1.1       # Penalizes mediocre scores more aggressively
coverage_target=0.85  # Demands high eligibility across tickers
coverage_k=8.0

# 5) Execution (FORCED ROTATION)
commission_per_side=0.0006
loss_floor=0.001
cooldown=1            # Mandatory 1-bar cooldown for high frequency
time_stop=15          # Mandatory short time-stop to prevent "holding"

# 6) Risk constraint
min_tp2sl=1.10

# 7) SIGNAL REGIME
threshold_mode="fixed"
threshold_fixed=0.04
vol_floor_len=50
vol_floor_mult_fixed=0.55

# 8) Soft trend weighting knobs
trend_center=0.80
trend_k=3.0

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
echo "PHASE-4: HIGH-DENSITY AGGRESSION (Target: 10+ trades)"
echo "Target: 10 trades/ticker | Coverage >= 85% | Cooldown: ${cooldown}"
echo "Python: ${PYTHON_BIN} | Seed: ${seed} | Files: ${nFiles} | Trials: ${nTrials}"
echo "-------------------------------------------------------"

# Execute
"${CMD[@]}"