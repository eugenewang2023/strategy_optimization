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
MAX_PENALTY_MULT=3.0      # Cap on total penalty multiplier (matches Python)

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
  D   Realist
  E   Quality Tightening
  F   Institutional Final

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
        A|B|C|D|E|F)        SELECTED_PHASES+=("$1"); shift ;;
        *)                  echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Run all phases if none selected
# [[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(A B C D E F)
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(F)

mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# Phase Definitions
# Format:
#   "min_tr weight_pf pwr ret_fl ret_k pf_k min_glpt min_glpt_k cov_t cov_k reg_ratio pf_cap loss_fl # Comment"
# ───────────────────────────────────────────────────────────────

declare -A PHASE_PARAMS

# PHASE A — Exploration / Zero‑Plateau Escape
# Loose floors, shallow steepness, generous coverage.
#PHASE_PARAMS["A"]="2 0.40 1.1 -0.40 1.0 0.0 0.0012 4 0.40 2.0 1.8 6.0 0.0010 # Phase A: Exploration"
#PHASE_PARAMS["A"]="2 0.40 1.10 -0.40 1.00 0.0008 0.0010 6 0.40 3.0 1.6 6.0 0.0010 # Phase A: Exploration"
PHASE_PARAMS["A"]="2 0.40 1.10 -0.40 1.00 0.0008 0.0010 6 0.40 3.0 6.0 0.0010 # A"

# PHASE B — Robustness Formation
# Slightly tighter returns, more PF weight, higher coverage.
#PHASE_PARAMS["B"]="3 0.55 1.4 -0.25 1.5 2.0 0.0013 6 0.45 2.5 1.9 6.0 0.0010 # Phase B: Robustness Formation"
#PHASE_PARAMS["B"]="3 0.55 1.30 -0.28 1.40 0.0010 0.00115 8 0.48 3.4 1.8 6.0 0.0010 # Phase B: Robustness"
PHASE_PARAMS["B"]="3 0.55 1.30 -0.28 1.40 0.0010 0.00115 8 0.48 3.4 6.0 0.0010 # B"

# PHASE C — Expectancy + PF Tightening
# Better floors, steeper GLPT and coverage.
#PHASE_PARAMS["C"]="4 0.70 1.7 -0.18 1.8 4.0 0.00145 8 0.50 3.0 2.1 6.0 0.00105 # Phase C: Quality Tightening"
#PHASE_PARAMS["C"]="4 0.68 1.55 -0.18 1.80 0.00125 0.00130 10 0.55 3.8 2.1 6.0 0.00105 # Phase C: Quality Tightening"
#PHASE_PARAMS["C"]="4 0.68 1.55 -0.18 1.80 0.00125 0.00130 10 0.55 3.8 6.0 0.00105"
PHASE_PARAMS["C"]="4 0.68 1.55 -0.18 1.80 0.00125 0.00130 10 0.55 3.8 6.0 0.00105 # C"

# PHASE D — Institutional Stability
# Focus on stability and DD, still moderate coverage.
#PHASE_PARAMS["D"]="4 0.78 1.9 -0.14 2.1 6.0 0.00155 10 0.55 3.5 2.3 6.0 0.00108 # Phase D: Stability Consolidation"
#PHASE_PARAMS["D"]="4 0.75 1.75 -0.12 2.20 0.00140 0.00145 12 0.60 4.1 2.3 6.0 0.00105 # Phase D: Stability"
#PHASE_PARAMS["D"]="4 0.75 1.75 -0.12 2.20 0.00140 0.00145 12 0.60 4.1 6.0 0.00105"
PHASE_PARAMS["D"]="4 0.75 1.75 -0.12 2.20 0.00140 0.00145 12 0.60 4.1 6.0 0.00105 # D"

# PHASE E — High‑Expectancy Tightening
# Stronger floors, higher GLPT and coverage.
#PHASE_PARAMS["E"]="5 0.80 2.0 -0.10 2.4 8.0 0.00160 12 0.58 4.0 2.6 6.0 0.00110 # Phase E: High-Expectancy Tightening"
#PHASE_PARAMS["E"]="5 0.80 1.90 -0.08 2.50 0.00155 0.00160 14 0.65 4.3 2.5 6.0 0.00105 # Phase E: Expectancy Tightening"
#PHASE_PARAMS["E"]="5 0.80 1.90 -0.08 2.50 0.00155 0.00160 14 0.65 4.3 6.0 0.00105"
PHASE_PARAMS["E"]="5 0.80 1.90 -0.08 2.50 0.00155 0.00160 14 0.65 4.3 6.0 0.00105 # E"

# PHASE F — Final Institutional‑Grade Filter
# Matches the Phase‑F suite above: GLPT ~0.0018, coverage 0.60, steeper k.
#PHASE_PARAMS["F"]="6 0.85 2.2 -0.08 2.8 9.0 0.00180 14 0.60 4.5 3.0 6.0 0.00100 # Phase F: Institutional Final"
#PHASE_PARAMS["F"]="6 0.85 2.2 -0.05 2.2 7.0 0.0018 12 0.60 3.2 1.5 6.0 0.0010 # Phase F: Institutional Final (Corrected)"
#PHASE_PARAMS["F"]="6 0.85 2.10 -0.05 3.00 0.00170 0.00180 16 0.60 4.0 3.0 6.0 0.00100 # Phase F: Institutional Final"
#PHASE_PARAMS["F"]="6 0.85 2.10 -0.05 3.00 0.00170 0.00180 16 0.60 4.0 6.0 0.00100"
PHASE_PARAMS["F"]="6 0.85 2.10 -0.05 3.00 0.00170 0.00180 16 0.60 4.0 6.0 0.00100 # F"

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
