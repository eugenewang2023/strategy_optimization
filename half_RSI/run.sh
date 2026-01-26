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
N_JOBS=4
N_STARTUP_TRIALS=200

# Strategy / Scoring related defaults (adjusted for half_RSI.py)
TRADES_BASELINE=8.0
COMMISSION=0.0006
PF_BASELINE=1.8
PF_K=1.5
MIN_TRADES_PHASED=(2 3 4 5 6 8)

SELECTED_PHASES=()

# ───────────────────────────────────────────────────────────────
# Usage / Help
# ───────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [phase_code ...]
Multi-phase half_RSI optimization runner (adapted from Vidya_RSI version)

Options:
  --seed INT        Random seed (default: $SEED)
  --trials INT      Trials per phase (default: $TRIALS)
  --files INT       Number of data files (default: $FILES)
  --fill MODE       Fill mode: next_open | same_close (default: $FILL_MODE)
  --n-jobs INT      Parallel trials (default: $N_JOBS)
  --dry-run         Show commands without executing
  --help            Show this help

Phases (runs D,E,F if none specified):
  A     Wide / discovery phase
  B     Robustness formation
  C     Expectancy tightening
  D     Institutional stability
  E     High-expectancy tightening
  F     Final institutional tightening
EOF
    exit "${1:-0}"
}

# ───────────────────────────────────────────────────────────────
# Parse Arguments
# ───────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seed)          SEED="$2"; shift 2 ;;
        --trials)        TRIALS="$2"; shift 2 ;;
        --files)         FILES="$2"; shift 2 ;;
        --fill)          FILL_MODE="$2"; shift 2 ;;
        --n-jobs)        N_JOBS="$2"; shift 2 ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --help|-h)       usage 0 ;;
        A|B|C|D|E|F)     SELECTED_PHASES+=("$1"); shift ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Default phases if none selected
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(D E F)

mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# Phase Definitions
#     min_trades  weight_pf  score_power  ret_floor  ret_floor_k  pf_floor_k
# ───────────────────────────────────────────────────────────────
declare -A PHASE_PARAMS
declare -A PHASE_NAMES

PHASE_PARAMS["A"]="2 0.40 1.10 -0.40 1.00 0.0008"
PHASE_NAMES["A"]="Discovery: wide search"

PHASE_PARAMS["B"]="3 0.60 1.40 -0.22 1.60 0.00120"
PHASE_NAMES["B"]="Robustness formation"

PHASE_PARAMS["C"]="4 0.70 1.45 -0.22 1.95 0.00110"
PHASE_NAMES["C"]="Expectancy tightening"

PHASE_PARAMS["D"]="5 0.75 1.75 -0.12 2.20 0.00140"
PHASE_NAMES["D"]="Institutional stability"

PHASE_PARAMS["E"]="6 0.80 1.90 -0.08 2.50 0.00155"
PHASE_NAMES["E"]="High-expectancy tightening"

PHASE_PARAMS["F"]="8 0.85 2.10 -0.05 3.00 0.00170"
PHASE_NAMES["F"]="Final institutional"

# ───────────────────────────────────────────────────────────────
# Runner Function
# ───────────────────────────────────────────────────────────────
run_phase() {
    local phase_code="$1"
    local params="${PHASE_PARAMS[$phase_code]}"

    IFS=' ' read -r \
        min_trades weight_pf score_power \
        ret_floor ret_floor_k pf_floor_k <<< "$params"

    local phase_name="${PHASE_NAMES[$phase_code]}"
    local log_file="$LOG_DIR/phase_${phase_code}_half_RSI.log"

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase_code : $phase_name   (half_RSI)"
    echo " min_trades     : $min_trades"
    echo " weight-pf      : $weight_pf"
    echo " score-power    : $score_power"
    echo " ret-floor      : $ret_floor"
    echo "═══════════════════════════════════════════════════════════════"

    local CMD=(
        "$PYTHON_BIN" half_RSI.py
        --seed               "$SEED"
        --trials             "$TRIALS"
        --files              "$FILES"
        --fill               "$FILL_MODE"
        --data_dir           "$DATA_DIR"
        --output_dir         "output_half_RSI_phase_${phase_code}"

        # Core scoring parameters (the ones half_RSI.py actually accepts)
        --min-trades         "$min_trades"
        --trades-baseline    "$TRADES_BASELINE"
        --weight-pf          "$weight_pf"
        --score-power        "$score_power"
        --ret-floor          "$ret_floor"
        --ret-floor-k        "$ret_floor_k"
        --pf-floor-k         "$pf_floor_k"

        --commission_rate_per_side "$COMMISSION"
        --pf-baseline        "$PF_BASELINE"
        --pf-k               "$PF_K"

        # half_RSI hyperparameter ranges & optimization toggles
        --slow-window-min    14
        --slow-window-max    80
        --shift-min          0
        --shift-max          3
        --smooth-len-min     3
        --smooth-len-max     30

        --opt-slow-window
        --opt-shift
        --opt-smooth-len
        --opt-time-stop
        --opt-cooldown

        # Optional: tighten search if you want faster/more focused runs
        # --slow-window-min    20
        # --slow-window-max    50
        # --smooth-len-min     5
        # --smooth-len-max     20
    )
    
    if [[ "${DRY_RUN:-false}" == true ]]; then
        echo "DRY RUN COMMAND:"
        printf '  %s\n' "${CMD[@]}"
        echo ""
    else
        echo "Running phase $phase_code..."
        echo "→ Log: $log_file"
        echo ""

        if ! "${CMD[@]}" 2>&1 | tee "$log_file"; then
            local exit_code=${PIPESTATUS[0]}
            echo "ERROR: Phase $phase_code failed (exit $exit_code)"
            echo "See: $log_file"
            return $exit_code
        fi
    fi

    echo "Phase $phase_code completed."
    echo ""
}

# ───────────────────────────────────────────────────────────────
# Main Execution
# ───────────────────────────────────────────────────────────────
echo "Starting half_RSI Optimization Sweep..."
echo "Seed       : $SEED"
echo "Trials/phase: $TRIALS"
echo "Parallel jobs: $N_JOBS"
echo "Files       : $FILES"
echo "Phases      : ${SELECTED_PHASES[*]}"
echo "Python      : $PYTHON_BIN"
echo ""

# Validate phases
for phase in "${SELECTED_PHASES[@]}"; do
    if [[ -z "${PHASE_PARAMS[$phase]:-}" ]]; then
        echo "ERROR: Phase '$phase' not defined."
        exit 1
    fi
done

# Run selected phases
for phase in "${SELECTED_PHASES[@]}"; do
    if ! run_phase "$phase"; then
        echo "FATAL: Phase $phase failed → stopping."
        exit 1
    fi

    if [[ "$phase" != "${SELECTED_PHASES[-1]}" ]]; then
        echo "--- Pausing 3 seconds before next phase ---"
        sleep 3
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Multi-phase half_RSI optimization complete."
echo "Logs: $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════════"