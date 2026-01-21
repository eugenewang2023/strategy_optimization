#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use python3 if available; fall back to python
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: python not found in PATH"; exit 2; }

seed=7
nTrials=2000          # ↑ from 1500 – more exploration for high score_power
nFiles=300
fillOpt="next_open"

# ──────────────────────────────────────────────────────────────────────────────
# Common base parameters
# ──────────────────────────────────────────────────────────────────────────────
base_min_trades=1
base_trades_baseline=3.0
base_trades_k=1.2
base_pf_baseline=1.15
base_pf_k=2.5
base_weight_pf=0.70
base_score_power=1.5
base_threshold_fixed=0.012
base_vol_floor_mult_fixed=0.05
base_pf_cap=4.0
base_loss_floor=0.001
base_min_glpt=0.003
base_min_glpt_k=12

# ──────────────────────────────────────────────────────────────────────────────
# PHASE runner
# ──────────────────────────────────────────────────────────────────────────────
run_phase() {
    local phase_name="$1"
    local min_trades="$2"
    local weight_pf="$3"
    local score_power="$4"
    local ret_floor_center="$5"
    local ret_floor_k="$6"
    local pf_floor_k="$7"
    local min_glpt="$8"
    local min_glpt_k="$9"
    local extra_args=("${@:10}")

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "               PHASE: $phase_name"
    echo "═══════════════════════════════════════════════════════════════"
    echo "min_trades     = $min_trades"
    echo "weight_pf      = $weight_pf"
    echo "score_power    = $score_power"
    echo "ret_floor_center/k = $ret_floor_center / $ret_floor_k"
    echo "pf_floor_k     = $pf_floor_k"
    echo "min_glpt / k   = $min_glpt / $min_glpt_k"
    echo "extra args     = ${extra_args[*]}"
    echo ""

    local CMD=(
        "$PYTHON_BIN" Vidya_RSI.py
        --optimize
        --seed "$seed"
        --trials "$nTrials"
        --files "$nFiles"
        --fill "$fillOpt"
        --data_dir "data"
        --min-trades "$min_trades"
        --trades-baseline "$base_trades_baseline"
        --trades-k "$base_trades_k"
        --pf-baseline "$base_pf_baseline"
        --pf-k "$base_pf_k"
        --weight-pf "$weight_pf"
        --score-power "$score_power"
        --threshold-fixed "$base_threshold_fixed"
        --vol-floor-mult-fixed "$base_vol_floor_mult_fixed"
        --pf-cap "$base_pf_cap"
        --objective-mode "hybrid"
        --obj-penalty-mode "both"
        --min-glpt "$min_glpt"
        --min-glpt-k "$min_glpt_k"
        --opt-time-stop
        --min-tp2sl 0.8
        --opt-vidya
        --opt-fastslow
        # Tail / return floor penalty
        --penalty-ret-center "$ret_floor_center"
        --penalty-ret-k "$ret_floor_k"
        # PF floor penalty
        --pf-floor-k "$pf_floor_k"
        "${extra_args[@]}"
    )

    # Optional flags if supported
    if "$PYTHON_BIN" Vidya_RSI.py --help 2>/dev/null | grep -q -- "--loss-floor"; then
        CMD+=( --loss-floor "$base_loss_floor" )
    fi
    # Optional: soft coverage encouragement (if your script has it)
    if "$PYTHON_BIN" Vidya_RSI.py --help 2>/dev/null | grep -q -- "--coverage-target"; then
        CMD+=( --coverage-target 0.98 --coverage-k 8 )
    fi

    echo "Command:"
    printf '  %q' "${CMD[@]}"
    echo -e "\n"

    "${CMD[@]}"
}

# ──────────────────────────────────────────────────────────────────────────────
# MAIN PHASES – start with E + F recommended
# ──────────────────────────────────────────────────────────────────────────────
echo "Starting multi-phase optimization sweep..."
echo "Seed: $seed   |   Trials: $nTrials   |   Files: $nFiles"
echo ""

# E - Hybrid aggressive quality + min_trades safety (recommended next)
run_phase "E - Balanced Aggressive + Safety" \
    3           0.80        2.0       -0.105    2.4     7     0.003   13

# F - Edge focus (try to recover higher GL/trade)
run_phase "F - Higher Edge Target" \
    2           0.80        2.1       -0.11     2.4     7     0.0035  14

# Optional – uncomment if you want to test time-stop softness (if supported)
# run_phase "G - Softer Time-Stop Penalty" \
#     2           0.80        2.0       -0.11     2.4     7     0.003   13   \
#     --time-stop-penalty-center 20 --time-stop-penalty-k 3.0

# Uncomment others if you want to re-run or compare
# run_phase "B - Aggressive quality push v2" 1 0.82 2.2 -0.10 2.5 8 0.003 12
# run_phase "C - Min trades safety net" 4 0.75 1.8 -0.12 2.2 6 0.003 12

echo ""
echo "All phases completed."
echo "Check output/*.csv and logs for best_score, coverage, negatives count, median PF, etc."
