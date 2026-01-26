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
TRIALS=300
FILES=300
FILL_MODE="next_open"
DATA_DIR="data"
LOG_DIR="logs"
N_JOBS=4                  # Parallel trial execution
N_STARTUP_TRIALS=200      # Initial random exploration count

# Enhanced Objective Settings (new feature)
USE_ENHANCED_OBJECTIVE=true  # Set to false to use original objective

# Global Strategy Baselines
TRADES_BASELINE=5.0       # Targets ~5 trades per ticker for stability
COMMISSION=0.0006         # 0.06% per side (adjust to your broker)
REGIME_SLOPE_MIN=0.0
REGIME_PERSIST=3
MAX_PENALTY_MULT=3.0      # Cap on total penalty multiplier (matches Python)

# Scoring baselines (using the correct argument names from Python script)
PF_BASELINE=1.5
PF_K=2.0                  # Added: PF logistic steepness
COV_K=10.0                # Added: Coverage logistic steepness  
STAB_K=10.0               # Added: Stability logistic steepness
ZL_K=10.0                 # Added: Zero-loss logistic steepness
GLPT_K=8.0                # Added: GLPT logistic steepness (first occurrence)

# These are defined in Python but might not be used in score_trial
GLPT_BASELINE=0.0         # Python has this but might not use it
COV_BASELINE=0.9          # Python default
STAB_BASELINE=0.9         # Python default  
ZL_BASELINE=0.0           # Python default

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
  --use-enhanced       Use enhanced objective function with returns/trades data
  --use-original       Use original objective function (default: enhanced)
  --dry-run            Show commands without executing
  --help               Show this help message

Phases (runs all if none specified):
  D   Institutional Stability
  E   High‑Expectancy Tightening
  F   Final Institutional

Enhanced Objective Features:
  - Uses daily returns and individual trades data
  - Better risk-adjusted return metrics
  - More robust coverage and stability calculations
  - Balanced composite scoring

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
        --use-enhanced)     USE_ENHANCED_OBJECTIVE=true; shift ;;
        --use-original)     USE_ENHANCED_OBJECTIVE=false; shift ;;
        --dry-run)          DRY_RUN=true; shift ;;
        --help|-h)          usage 0 ;;
        D|E|F)              SELECTED_PHASES+=("$1"); shift ;;
        *)                  echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Run all phases if none selected
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(D E F)

mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# Phase Definitions
# Format:
#   min_trades weight_pf score_power ret_floor ret_floor_k pf_floor_k min_glpt min_glpt_k coverage_target coverage_k pf_cap loss_floor
# ───────────────────────────────────────────────────────────────

declare -A PHASE_PARAMS
declare -A PHASE_NAMES

# Phase D — Institutional Stability
PHASE_PARAMS["D"]="4 0.75 1.75 -0.12 2.20 0.00140 0.00145 12 0.54 3.0 6.0 0.00105"
PHASE_NAMES["D"]="Institutional Stability"

# Phase E — High‑Expectancy Tightening
PHASE_PARAMS["E"]="5 0.80 1.90 -0.08 2.50 0.00155 0.00160 14 0.55 3.5 6.0 0.00105"
PHASE_NAMES["E"]="High‑Expectancy Tightening"

# Phase F — Final Institutional
PHASE_PARAMS["F"]="6 0.85 2.10 -0.05 3.00 0.00170 0.00180 16 0.55 3.0 5.0 0.00100"
PHASE_NAMES["F"]="Final Institutional"

# ───────────────────────────────────────────────────────────────
# Enhanced Objective Phase Adjustments
# These adjust parameters for the enhanced objective function
# ───────────────────────────────────────────────────────────────

declare -A ENHANCED_ADJUSTMENTS

# For enhanced objective, we can use slightly different parameters
ENHANCED_ADJUSTMENTS["min_trades"]="4"        # Keep at 4 for enhanced
ENHANCED_ADJUSTMENTS["weight_pf"]="0.70"      # Lower PF weight due to more metrics
ENHANCED_ADJUSTMENTS["score_power"]="1.70"    # Slightly less sharpening
ENHANCED_ADJUSTMENTS["min_glpt"]="0.0015"     # Slightly higher GLPT target
ENHANCED_ADJUSTMENTS["coverage_target"]="0.56" # Slightly higher coverage

# Function to get enhanced-adjusted value
get_enhanced_value() {
    local var_name="$1"
    local original_value="$2"
    
    if $USE_ENHANCED_OBJECTIVE && [[ -n "${ENHANCED_ADJUSTMENTS[$var_name]:-}" ]]; then
        echo "${ENHANCED_ADJUSTMENTS[$var_name]}"
    else
        echo "$original_value"
    fi
}

# ───────────────────────────────────────────────────────────────
# Runner Function
# ───────────────────────────────────────────────────────────────

run_phase() {
    local phase_code="$1"
    
    # Extract values from PHASE_PARAMS
    local params="${PHASE_PARAMS[$phase_code]}"
    IFS=' ' read -r \
        min_trades weight_pf score_power \
        ret_floor ret_floor_k pf_floor_k \
        min_glpt min_glpt_k \
        coverage_target coverage_k \
        pf_cap_val loss_floor_val <<< "$params"
    
    # Get enhanced-adjusted values if enabled
    if $USE_ENHANCED_OBJECTIVE; then
        min_trades=$(get_enhanced_value "min_trades" "$min_trades")
        weight_pf=$(get_enhanced_value "weight_pf" "$weight_pf")
        score_power=$(get_enhanced_value "score_power" "$score_power")
        min_glpt=$(get_enhanced_value "min_glpt" "$min_glpt")
        coverage_target=$(get_enhanced_value "coverage_target" "$coverage_target")
    fi
    
    local phase_name="${PHASE_NAMES[$phase_code]}"
    local log_file="$LOG_DIR/phase_${phase_code}.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase_code : $phase_name"
    echo " Objective: $($USE_ENHANCED_OBJECTIVE && echo "Enhanced" || echo "Original")"
    echo " min_trades: $min_trades | weight_pf: $weight_pf | score_power: $score_power"
    echo " ret_floor: $ret_floor | min_glpt: $min_glpt | coverage_target: $coverage_target"
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
        
        # Core Scoring Parameters
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
        --max-penalty-mult "$MAX_PENALTY_MULT"
        
        # Coverage & Regime Config
        --coverage-target "$coverage_target"
        --coverage-k "$coverage_k"
        --regime-slope-min "$REGIME_SLOPE_MIN"
        --regime-persist "$REGIME_PERSIST"
        
        # Search Range Limits
        --fast-min 8 --fast-max 25
        --slow-min 30 --slow-max 100
        --regime-ratio-min 1.5 --regime-ratio-max 5.0
        
        # Scoring Baselines - using CORRECT argument names from Python
        --pf-baseline "$PF_BASELINE"
        --pf-k "$PF_K"
        --glpt_k "$GLPT_K"
        --cov_k "$COV_K"
        --stab_k "$STAB_K"
        --zl_k "$ZL_K"
        
        # These might be needed for the scoring function
        --w_pf 0.25
        --w_glpt 0.25
        --w_cov 0.20
        --w_stab 0.20
        --w_zl 0.10
        
        # Execution Toggles
        --opt-time-stop
        --opt-vidya
        --opt-fastslow
        --commission_rate_per_side "$COMMISSION"
        --objective-mode "hybrid"
    )
    
    # Add enhanced objective flag if enabled
    if $USE_ENHANCED_OBJECTIVE; then
        CMD+=(--use-enhanced-objective)
    fi

    if $DRY_RUN; then
        echo "DRY RUN COMMAND:"
        printf '  %s\n' "${CMD[@]}"
        echo ""
        echo "Would run with parameters:"
        echo "  min_trades: $min_trades, weight_pf: $weight_pf, score_power: $score_power"
        echo "  ret_floor: $ret_floor, ret_floor_k: $ret_floor_k, pf_floor_k: $pf_floor_k"
        echo "  min_glpt: $min_glpt, min_glpt_k: $min_glpt_k"
        echo "  coverage_target: $coverage_target, coverage_k: $coverage_k"
        echo "  pf_cap: $pf_cap_val, loss_floor: $loss_floor_val"
        echo "  Enhanced objective: $USE_ENHANCED_OBJECTIVE"
    else
        echo "Running phase $phase_code..."
        echo "Command: ${CMD[*]}"
        echo ""
        
        # Run the command and capture output
        if ! "${CMD[@]}" 2>&1 | tee "$log_file"; then
            local exit_code=${PIPESTATUS[0]}
            echo ""
            echo "ERROR: Phase $phase_code failed with exit code $exit_code"
            echo "See log file: $log_file"
            return $exit_code
        fi
    fi
    echo ""
    echo "Done with phase $phase_code"
}

# ───────────────────────────────────────────────────────────────
# Main Loop
# ───────────────────────────────────────────────────────────────

echo "Starting Vidya_RSI Optimization Sweep..."
echo "Seed: $SEED | Trials: $TRIALS | Jobs: $N_JOBS | Files: $FILES"
echo "Selected phases: ${SELECTED_PHASES[*]}"
echo "Using Python: $PYTHON_BIN"
echo "Objective function: $($USE_ENHANCED_OBJECTIVE && echo "Enhanced" || echo "Original")"
echo ""

# Validate all selected phases exist
for phase in "${SELECTED_PHASES[@]}"; do
    if [[ -z "${PHASE_PARAMS[$phase]:-}" ]]; then
        echo "ERROR: Phase '$phase' is not defined."
        exit 1
    fi
done

# Run each phase
for phase in "${SELECTED_PHASES[@]}"; do
    if ! run_phase "$phase"; then
        echo "FATAL: Phase $phase failed. Stopping."
        exit 1
    fi
    
    # Add a small pause between phases
    if ! $DRY_RUN && [[ $phase != "${SELECTED_PHASES[-1]}" ]]; then
        echo ""
        echo "--- Pausing before next phase ---"
        sleep 3
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Multi-phase optimization complete."
echo "Logs saved in $LOG_DIR/"
echo "Objective used: $($USE_ENHANCED_OBJECTIVE && echo "Enhanced" || echo "Original")"
echo "═══════════════════════════════════════════════════════════════"

# Final recommendation based on objective used
if $USE_ENHANCED_OBJECTIVE; then
    echo ""
    echo "RECOMMENDATION: Run validation with enhanced objective:"
    echo "  python3 Vidya_RSI.py --use-enhanced-objective --files $FILES \\"
    echo "    --threshold-fixed <best_threshold> --vol-floor-mult-fixed <best_vol_floor> \\"
    echo "    --vidya-len-fixed <best_len> --vidya-smooth-fixed <best_smooth>"
else
    echo ""
    echo "RECOMMENDATION: Consider running with enhanced objective:"
    echo "  $SCRIPT_NAME --use-enhanced ${SELECTED_PHASES[@]}"
fi