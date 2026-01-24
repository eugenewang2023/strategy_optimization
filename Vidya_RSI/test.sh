#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────
# Configuration & defaults
# ───────────────────────────────────────────────────────────────

SCRIPT_NAME="$(basename "$0")"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: python not found"; exit 2; }

# Default values
SEED=7
TRIALS=10
FILES=20
FILL_MODE="next_open"
DATA_DIR="data"
LOG_DIR="logs"
N_JOBS=4                  # NEW: parallel trials
N_STARTUP_TRIALS=200      # NEW: more random exploration at start

# Base parameters (shared)
TRADES_BASELINE=3.0
TRADES_K=1.2
PF_BASELINE=1.15
PF_K=2.5
THRESHOLD_FIXED=0.005
VOL_FLOOR_MULT_FIXED=0.05
PF_CAP=4.0
LOSS_FLOOR=0.001
MIN_GLPT_BASE=0.0015
MIN_GLPT_K_BASE=8
REGIME_SLOPE_MIN=0.0
REGIME_PERSIST=3
COVERAGE_TARGET=0.20
COVERAGE_K=6

DRY_RUN=false
SELECTED_PHASES=()

# ───────────────────────────────────────────────────────────────
# Usage / Help
# ───────────────────────────────────────────────────────────────

usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [phaseA phaseB ...]

Multi-phase Vidya_RSI optimization runner

Options:
  --seed INT          Random seed (default: $SEED)
  --trials INT        Trials per phase (default: $TRIALS)
  --files INT         Number of data files (default: $FILES)
  --fill MODE         Fill mode: next_open | same_close (default: $FILL_MODE)
  --data-dir PATH     Data directory (default: $DATA_DIR)
  --log-dir PATH      Log directory (default: $LOG_DIR)
  --n-jobs INT        Parallel trials (default: $N_JOBS)
  --n-startup-trials INT  Number of random startup trials (default: $N_STARTUP_TRIALS)
  --dry-run           Show commands without executing
  --help              Show this help message

Phases (if none specified, runs both):
  A     Balanced Aggressive + Safety
  B     Higher Edge Target

Examples:
  $SCRIPT_NAME
  $SCRIPT_NAME --trials 2000 --n-jobs 6 A
  $SCRIPT_NAME --seed 42 --n-startup-trials 300 B
EOF
    exit "${1:-0}"
}

# ───────────────────────────────────────────────────────────────
# Parse arguments
# ───────────────────────────────────────────────────────────────

while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)            SEED="$2"; shift 2 ;;
        --trials)          TRIALS="$2"; shift 2 ;;
        --files)           FILES="$2"; shift 2 ;;
        --fill)            FILL_MODE="$2"; shift 2 ;;
        --data-dir)        DATA_DIR="$2"; shift 2 ;;
        --log-dir)         LOG_DIR="$2"; shift 2 ;;
        --n-jobs)          N_JOBS="$2"; shift 2 ;;
        --n-startup-trials) N_STARTUP_TRIALS="$2"; shift 2 ;;
        --dry-run)         DRY_RUN=true; shift ;;
        --help|-h)         usage 0 ;;
        A|B)               SELECTED_PHASES+=("$1"); shift ;;
        *)                 echo "Unknown option: $1"; usage 1 ;;
    esac
done

# If no phases selected → run both
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(A B)

# Create log dir
mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# Phase definitions
# ───────────────────────────────────────────────────────────────

declare -A PHASE_PARAMS

PHASE_PARAMS["A"]="3 0.80 2.0 -0.15 1.5 7 0.0015 8   # Balanced Aggressive + Safety"
PHASE_PARAMS["B"]="2 0.80 2.1 -0.11 2.4 7 0.0035 14  # Higher Edge Target"

# ───────────────────────────────────────────────────────────────
# Runner function
# ───────────────────────────────────────────────────────────────

run_phase() {
    local phase_code="$1"
    local phase_name phase_args

    IFS=' ' read -r min_trades weight_pf score_power ret_floor ret_floor_k pf_floor_k min_glpt min_glpt_k comment <<< "${PHASE_PARAMS[$phase_code]}"
    phase_name="${comment#* # }"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase_code : $phase_name"
    echo "═══════════════════════════════════════════════════════════════"
    echo "min_trades     = $min_trades"
    echo "weight_pf      = $weight_pf"
    echo "score_power    = $score_power"
    echo "ret_floor/k    = $ret_floor / $ret_floor_k"
    echo "pf_floor_k     = $pf_floor_k"
    echo "min_glpt / k   = $min_glpt / $min_glpt_k"
    echo ""

    local log_file="$LOG_DIR/phase_${phase_code}.log"

    local CMD=(
        "$PYTHON_BIN" Vidya_RSI.py
        --optimize
        --seed "$SEED"
        --trials "$TRIALS"
        --n-jobs "$N_JOBS"                # NEW
        --n-startup-trials "$N_STARTUP_TRIALS"  # NEW
        --files "$FILES"
        --fill "$FILL_MODE"
        --data_dir "$DATA_DIR"
        # Core scoring
        --min-trades "$min_trades"
        --trades-baseline "$TRADES_BASELINE"
        --trades-k "$TRADES_K"
        --pf-baseline "$PF_BASELINE"
        --pf-k "$PF_K"
        --weight-pf "$weight_pf"
        --score-power "$score_power"
        # Threshold & volatility
        --threshold-fixed "$THRESHOLD_FIXED"
        --vol-floor-mult-fixed "$VOL_FLOOR_MULT_FIXED"
        # PF cap & loss floor
        --pf-cap "$PF_CAP"
        --loss_floor "$LOSS_FLOOR"
        # Objective mode
        --objective-mode "hybrid"
        --obj-penalty-mode "both"
        # GLPT
        --min-glpt "$min_glpt"
        --min-glpt-k "$min_glpt_k"
        # Penalties
        --ret-floor "$ret_floor"
        --ret-floor-k "$ret_floor_k"
        --pf-floor-k "$pf_floor_k"
        # Regime filter
        --regime-slope-min "$REGIME_SLOPE_MIN"
        --regime-persist "$REGIME_PERSIST"
        # Coverage
        --coverage-target "$COVERAGE_TARGET"
        --coverage-k "$COVERAGE_K"
        # Optimization toggles
        --opt-time-stop
        --opt-vidya
        --opt-fastslow
        # Search ranges
        --fast-min 7
        --fast-max 14
        --slow-min 35
        --slow-max 70
        # Commission
        --commission_rate_per_side 0.0006
    )

    echo "Command:"
    printf '  %q' "${CMD[@]}"

    if $DRY_RUN; then
        echo "(dry-run mode — command not executed)"
        return
    fi

    ##"${CMD[@]}" > "$log_file" 2>&1
	"${CMD[@]}"                         # ← let it print live

    # ───────────────────────────────────────────────────────────
    # Extract best parameters from log
    # ───────────────────────────────────────────────────────────
    echo -e "\n"
    echo "Extracting best parameters from $log_file..."

    best_params=$(awk '
    BEGIN { best_value = -1; best_p = "" }
    /Trial [0-9]+ finished with value: [0-9.e+-]+ and parameters: \{/ {
        match($0, /Trial ([0-9]+)/, t); trial = t[1];
        match($0, /value: ([0-9.e+-]+)/, v); value = v[1] + 0;
        match($0, /parameters: (\{[^}]+\})/, p); params = p[1];
        if (value > best_value) {
            best_value = value;
            best_p = params;
        }
    }
    END { if (best_p != "") print best_p; else print "NONE" }
    ' "$log_file")

    if [[ "$best_params" != "NONE" && -n "$best_params" ]]; then
        echo "Best parameters: $best_params"
    else
        echo "Warning: Could not extract best parameters from log."
    fi
	
    echo "Log: $log_file"	
}

# ───────────────────────────────────────────────────────────────
# Main execution
# ───────────────────────────────────────────────────────────────

echo ""
echo "Starting multi-phase optimization sweep..."
echo "Seed: $SEED | Trials: $TRIALS | Startup trials: $N_STARTUP_TRIALS | Parallel jobs: $N_JOBS | Files: $FILES | Fill: $FILL_MODE"
echo "Phases: ${SELECTED_PHASES[*]}"
echo "Logs in: $LOG_DIR"
echo ""

for phase in "${SELECTED_PHASES[@]}"; do
    if [[ -z ${PHASE_PARAMS[$phase]+set} ]]; then
        echo "ERROR: Unknown phase '$phase'" >&2
        exit 1
    fi
    run_phase "$phase"
done

echo ""
echo "All requested phases completed."
echo "Check output/*.csv and $LOG_DIR/*.log for results."
echo ""