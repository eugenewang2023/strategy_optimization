#!/bin/bash
set -euo pipefail
set +x   # force-disable xtrace if inherited

###=================================================================================
### PHASE-3B (Recovery & Trade Expansion) + Trend-weighted scoring
### Goal: Recover coverage, lift trades from ~4 -> ~6
###=================================================================================

# --- run from script directory (avoids path issues when launched elsewhere) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# --- python resolver (aliases don't apply in non-interactive shells) ---
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

nTrials=1000
nFiles=160
fillOpt=next_open

# 1) Trade gating (RECOVERY)
min_trades=5
trades_baseline=7
trades_k=0.2
max_trades=180
max_trades_k=0.05

# 2) Returns & penalties
ret_floor=0.02
ret_floor_k=6.0
penalty_center=0.0     # avoid "-0.0" in logs
penalty_k=6.0

# 3) PF shaping
pf_cap=6.0
pf_baseline=1.05
pf_k=0.9
pf_floor=1.0
pf_floor_k=1.0

# 4) Scoring balance
weight_pf=0.45
score_power=1.0
coverage_target=0.60
coverage_k=6.0

# 5) Execution
commission_per_side=0.0006
loss_floor=0.001
cooldown=1
time_stop=40

# 6) Risk constraint
min_tp2sl=1.10

# 7) LOCKED SIGNAL REGIME
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
echo "Phase-3B: Coverage Recovery & Trade Expansion (+ Trend-weighted scoring)"
echo "Target: ~6 trades/ticker | Coverage >= 60% | min_trades=${min_trades} | pf_cap=${pf_cap}"
echo "Python: ${PYTHON_BIN} | Seed: ${seed} | Files: ${nFiles} | Trials: ${nTrials} | Fill: ${fillOpt}"
echo "-------------------------------------------------------"

# Optional: print the exact command only if requested
if [[ "${SHOW_CMD:-0}" == "1" ]]; then
  printf 'CMD:'; printf ' %q' "${CMD[@]}"; echo
fi

# Execute
"${CMD[@]}"
