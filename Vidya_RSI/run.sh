#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────
# Configuration & Defaults
# ───────────────────────────────────────────────────────────────

SCRIPT_NAME="$(basename "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

# Default hardware/execution settings
SEED=7
TRIALS=1000
FILES=300
FILL_MODE="next_open"
DATA_DIR="data"
LOG_DIR="logs"
N_JOBS=4                  # Parallel trial execution
N_STARTUP_TRIALS=200      # Initial random exploration count

# Global Strategy Baselines
TRADES_BASELINE=5.0       # Targets ~5 trades per ticker for stability
COMMISSION=0.0006         # 0.06% per side (adjust to your broker)
REGIME_SLOPE_MIN=0.0
REGIME_PERSIST=3

DRY_RUN=false
SELECTED_PHASES=()

# ───────────────────────────────────────────────────────────────
# Usage / Help
# ───────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [phase_code ...]

Multi-phase Vidya_RSI optimization runner.

Options:
  --seed INT           Random seed (default: $SEED)
  --trials INT         Trials per phase (default: $TRIALS)
  --files INT          Number of data files (default: $FILES)
  --fill MODE          Fill mode: next_open | same_close (default: $FILL_MODE)
  --n-jobs INT         Parallel trials (default: $N_JOBS)
  --dry-run            Show commands without executing
  --help               Show this help message

Phases (runs all if none specified):
  A   Discovery: Escape the zero-plateau (wide search)
  B   Robustness: Filter for consistency and regime stability
  C   Quality: Tighten for high expectancy and low drawdown

Examples:
  $SCRIPT_NAME --trials 2000 A
  $SCRIPT_NAME B C
EOF
    exit "${1:-0}"
}

# ───────────────────────────────────────────────────────────────
# Parse Arguments
# ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)             SEED="$2"; shift 2 ;;
        --trials)           TRIALS="$2"; shift 2 ;;
        --files)            FILES="$2"; shift 2 ;;
        --fill)             FILL_MODE="$2"; shift 2 ;;
        --n-jobs)           N_JOBS="$2"; shift 2 ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --help|-h)          usage 0 ;;
        A|B|C)              SELECTED_PHASES+=("$1"); shift ;;
        *)                  echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Run all phases if none selected
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(A B C)

mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# Phase Definitions
# ───────────────────────────────────────────────────────────────
# Format: "min_tr weight_pf pwr ret_fl ret_k pf_k glpt glpt_k cov_t cov_k reg_ratio pf_cap loss_fl # Comment"
declare -A PHASE_PARAMS

# PHASE A: Discovery — Finds signal in the noise. Low requirements to prevent zero scores.
#PHASE_PARAMS["A"]="1 0.30 1.0 -0.50 0.5 0.0 0.000 1 0.10 1.0 1.5 8.0 0.0005 # PHASE A: Signal Discovery (Wide Search)"

# PHASE B: Robustness — Enforces trade density and regime filtering.
#PHASE_PARAMS["B"]="4 0.60 1.5 -0.15 1.5 5.0 0.001 8 0.35 2.5 3.0 5.0 0.0010 # PHASE B: Robustness & Regime Filtering"

# PHASE C: Refinement — Targets institutional-grade expectancy and tighter risk.
# The "Robustness Wall"
#PHASE_PARAMS["C"]="6 0.85 2.0 -0.05 3.0 10.0 0.002 15 0.45 4.0 3.5 3.5 0.0015 # PHASE C: High-Expectancy Refinement"

# PHASE D: "The Realist" 
# We drop min_trades to 3 and lower the GLPT target to find a middle ground.
PHASE_PARAMS["A"]="3 0.70 1.5 -0.10 2.0 5.0 0.0008 10 0.40 2.0 2.5 4.5 0.0010 # PHASE C: Sensitivity Calibration"
# ───────────────────────────────────────────────────────────────
# Runner Function
# ───────────────────────────────────────────────────────────────

run_phase() {
    local phase_code="$1"
    
    # Extract values from PHASE_PARAMS
    IFS=' ' read -r \
        min_trades weight_pf score_power \
        ret_floor ret_floor_k pf_floor_k \
        min_glpt min_glpt_k \
        coverage_target coverage_k \
        regime_ratio_val pf_cap_val loss_floor_val \
        comment <<< "${PHASE_PARAMS[$phase_code]}"
        
    local phase_name="${comment#*# }"
    local log_file="$LOG_DIR/phase_${phase_code}.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase_code : $phase_name"
    echo " Expectancy: $min_glpt | Coverage: $coverage_target"
    echo "═══════════════════════════════════════════════════════════════"

    local CMD=(
        "$PYTHON_BIN" Vidya_RSI.py
        --optimize
        --seed "$SEED"
        --trials "$TRIALS"
        --n-jobs "$N_JOBS"
        --n-startup-trials "$N_STARTUP_TRIALS"
        --files "$FILES"
        --fill "$FILL_MODE"
        --data_dir "$DATA_DIR"
        
        # Scoring Parameters
        --min-trades "$min_trades"
        --trades-baseline "$TRADES_BASELINE"
        --weight-pf "$weight_pf"
        --score-power "$score_power"
        --pf-cap "$pf_cap_val"
        --min-glpt "$min_glpt"
        --min-glpt-k "$min_glpt_k"
        
        # Penalty/Risk Gradients
        --loss_floor "$loss_floor_val"
        --ret-floor "$ret_floor"
        --ret-floor-k "$ret_floor_k"
        --pf-floor-k "$pf_floor_k"
        
        # Coverage & Regime Config
        --coverage-target "$coverage_target"
        --coverage-k "$coverage_k"
        --regime-slope-min "$REGIME_SLOPE_MIN"
        --regime-persist "$REGIME_PERSIST"
        
        # Search Range Limits
        --fast-min 8 --fast-max 25
        --slow-min 30 --slow-max 100
        --regime-ratio-min 1.5 --regime-ratio-max 5.0
        
        # Execution Toggles
        --opt-time-stop
        --opt-vidya
        --opt-fastslow
        --commission_rate_per_side "$COMMISSION"
        --objective-mode "hybrid"
    )

    if $DRY_RUN; then
        echo "DRY RUN COMMAND:"
        printf '  %q' "${CMD[@]}"
        echo ""
    else
        # Execute and pipe to both console and log
        "${CMD[@]}" 2>&1 | tee "$log_file"
    fi
}

# ───────────────────────────────────────────────────────────────
# Main Loop
# ───────────────────────────────────────────────────────────────

echo "Starting Vidya_RSI Optimization Sweep..."
echo "Seed: $SEED | Trials: $TRIALS | Jobs: $N_JOBS | Files: $FILES"

for phase in "${SELECTED_PHASES[@]}"; do
    if [[ -n "${PHASE_PARAMS[$phase]:-}" ]]; then
        run_phase "$phase"
    else
        echo "ERROR: Phase '$phase' is not defined."
        exit 1
    fi
done

echo ""
echo "Multi-phase optimization complete. Logs saved in $LOG_DIR."