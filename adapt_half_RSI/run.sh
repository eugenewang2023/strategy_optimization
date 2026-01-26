#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────
# Configuration & Defaults
# ───────────────────────────────────────────────────────────────
SCRIPT_NAME="$(basename "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

# Hardware/execution settings
SEED=42
TRIALS=300
FILES=300
FILL_MODE="next_open"
DATA_DIR="data"
LOG_DIR="logs"

# Strategy defaults
TRADES_BASELINE=6.0
COMMISSION=0.0006
PF_BASELINE=2.5
PF_K=2.0

SELECTED_PHASES=()

# ───────────────────────────────────────────────────────────────
# Usage / Help
# ───────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [phase_code ...]
Multi-phase adapt_half_RSI optimization runner

Options:
  --seed INT        Random seed (default: $SEED)
  --trials INT      Trials per phase (default: $TRIALS)
  --files INT       Number of data files (default: $FILES)
  --fill MODE       Fill mode: next_open | same_close (default: $FILL_MODE)
  --dry-run         Show commands without executing
  --help            Show this help

Phases:
  A     Test Phase - Minimal parameters
  B     Core Parameter Optimization
  C     Advanced Optimization
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
        --dry-run)       DRY_RUN=true; shift ;;
        --help|-h)       usage 0 ;;
        A|B|C)           SELECTED_PHASES+=("$1"); shift ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(A B C)
mkdir -p "$LOG_DIR"

# ───────────────────────────────────────────────────────────────
# PHASE DEFINITIONS - SIMPLIFIED
# ───────────────────────────────────────────────────────────────
declare -A PHASE_PARAMS
declare -A PHASE_NAMES

PHASE_PARAMS["A"]="3 0.30 1.05 -0.20 0.5 0.0005"
PHASE_NAMES["A"]="Test Phase - Minimal parameters"

PHASE_PARAMS["B"]="5 0.60 1.20 -0.10 1.0 0.0010"  
PHASE_NAMES["B"]="Core Parameter Optimization"

PHASE_PARAMS["C"]="6 0.70 1.30 -0.08 2.0 0.0020"
PHASE_NAMES["C"]="Advanced Optimization"

# ───────────────────────────────────────────────────────────────
# Get Script Parameters (ACTUAL TEST)
# Let's run a test to see what parameters work
# ───────────────────────────────────────────────────────────────
test_script_params() {
    echo "Testing script parameters..."
    
    # Test 1: Check if script runs with minimal params
    local test_cmd=(
        "$PYTHON_BIN" adapt_half_RSI.py
        --seed "$SEED"
        --trials 2
        --files 2
        --fill "$FILL_MODE"
        --data_dir "$DATA_DIR"
        --output_dir "test_params"
        --optimize
        --min-trades 2
        --trades-baseline "$TRADES_BASELINE"
        --weight-pf 0.3
        --score-power 1.05
        --ret-floor -0.2
        --ret-floor-k 0.5
        --pf-floor-k 0.0005
        --commission_rate_per_side "$COMMISSION"
        --pf-baseline "$PF_BASELINE"
        --pf-k "$PF_K"
    )
    
    echo "Test command: ${test_cmd[*]}"
    
    # Create test directory
    mkdir -p "test_params"
    
    # Run test
    if "${test_cmd[@]}" --dry-run 2>&1 | grep -q "error"; then
        echo "ERROR: Basic command failed"
        return 1
    else
        echo "SUCCESS: Basic command works"
        return 0
    fi
}

# ───────────────────────────────────────────────────────────────
# Parameter Search Space - USING ONLY VERIFIED PARAMETERS
# Based on the errors, we need to be very conservative
# ───────────────────────────────────────────────────────────────
get_search_space_params() {
    local phase="$1"
    
    case "$phase" in
        "A")
            # Phase A: Absolute minimal - only parameters that definitely work
            # From the help output, these definitely exist:
            echo "--adapt_k-fixed 0.2"
            echo "--atrPeriod-fixed 18"
            echo "--slMultiplier-fixed 4.0"
            echo "--tpMultiplier-fixed 3.0"
            echo "--base_slow_window-fixed 25"
            echo "--smooth_len-fixed 3"
            echo "--shift-fixed 1"
            ;;
        "B")
            # Phase B: Try optimization flags only
            # Use optimization flags instead of -min/-max
            echo "--opt-adaptive"
            echo "--opt-time-stop"
            echo "--opt-cooldown"
            # Keep some fixed values
            echo "--slMultiplier-fixed 4.0"
            echo "--tpMultiplier-fixed 3.0"
            echo "--base_slow_window-fixed 25"
            echo "--smooth_len-fixed 3"
            echo "--shift-fixed 1"
            ;;
        "C")
            # Phase C: Try with explicit ranges for SL/TP only
            # These definitely work based on help output
            echo "--slMultiplier-min 2.5 --slMultiplier-max 6.0"
            echo "--tpMultiplier-min 1.8 --tpMultiplier-max 4.5"
            # Fixed for others
            echo "--adapt_k-fixed 0.2"
            echo "--atrPeriod-fixed 18"
            echo "--base_slow_window-fixed 25"
            echo "--smooth_len-fixed 3"
            echo "--shift-fixed 1"
            # Optimization flags
            echo "--opt-adaptive"
            echo "--opt-time-stop"
            echo "--opt-cooldown"
            ;;
    esac
}

# ───────────────────────────────────────────────────────────────
# Build Command Array
# ───────────────────────────────────────────────────────────────
build_command() {
    local phase_code="$1"
    local params="$2"
    
    IFS=' ' read -r \
        min_trades weight_pf score_power \
        ret_floor ret_floor_k pf_floor_k <<< "$params"
    
    # Start building command array
    local cmd_array=()
    cmd_array+=("$PYTHON_BIN" "adapt_half_RSI.py")
    cmd_array+=("--seed" "$SEED")
    cmd_array+=("--trials" "$TRIALS")
    cmd_array+=("--files" "$FILES")
    cmd_array+=("--fill" "$FILL_MODE")
    cmd_array+=("--data_dir" "$DATA_DIR")
    cmd_array+=("--output_dir" "output/phase_${phase_code}")
    cmd_array+=("--optimize")
    
    # Core scoring - these definitely work
    cmd_array+=("--min-trades" "$min_trades")
    cmd_array+=("--trades-baseline" "$TRADES_BASELINE")
    cmd_array+=("--weight-pf" "$weight_pf")
    cmd_array+=("--score-power" "$score_power")
    cmd_array+=("--ret-floor" "$ret_floor")
    cmd_array+=("--ret-floor-k" "$ret_floor_k")
    cmd_array+=("--pf-floor-k" "$pf_floor_k")
    
    cmd_array+=("--commission_rate_per_side" "$COMMISSION")
    cmd_array+=("--pf-baseline" "$PF_BASELINE")
    cmd_array+=("--pf-k" "$PF_K")
    
    # Get search space parameters
    local search_lines
    search_lines=$(get_search_space_params "$phase_code")
    
    # Parse each line and add to command
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            # Split the line into arguments
            read -ra args <<< "$line"
            for arg in "${args[@]}"; do
                cmd_array+=("$arg")
            done
        fi
    done <<< "$search_lines"
    
    # Exit strategy - these should work
    cmd_array+=("--use_trailing_exit")
    cmd_array+=("--trail_mode" "trail_only")
    cmd_array+=("--close_on_sellSignal")
    
    # TP/SL constraints
    cmd_array+=("--tp2sl-auto")
    cmd_array+=("--tp2sl-base" "1.3")
    cmd_array+=("--tp2sl-sr0" "25.0")
    cmd_array+=("--tp2sl-k" "0.015")
    cmd_array+=("--tp2sl-min" "1.05")
    cmd_array+=("--tp2sl-max" "2.0")
    
    # Penalty settings
    cmd_array+=("--penalty")
    cmd_array+=("--loss-floor" "0.0005")
    cmd_array+=("--penalty-ret-center" "-0.015")
    cmd_array+=("--penalty-ret-k" "5.0")
    
    # Trade limits
    cmd_array+=("--max-trades" "25")
    cmd_array+=("--max-trades-k" "0.20")
    cmd_array+=("--pf-floor" "1.2")
    
    # Coverage control
    cmd_array+=("--coverage-target" "0.65")
    cmd_array+=("--coverage-k" "10.0")
    
    # Return as space-separated string
    printf '%s ' "${cmd_array[@]}"
    printf '\n'
}

# ───────────────────────────────────────────────────────────────
# Runner Function
# ───────────────────────────────────────────────────────────────
run_phase() {
    local phase_code="$1"
    local params="${PHASE_PARAMS[$phase_code]}"
    local phase_name="${PHASE_NAMES[$phase_code]}"
    local log_file="$LOG_DIR/phase_${phase_code}_adapt_half_RSI_$(date +%Y%m%d_%H%M%S).log"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase_code : $phase_name"
    echo "═══════════════════════════════════════════════════════════════"
    
    # Build the command
    local cmd
    cmd=$(build_command "$phase_code" "$params")
    
    if [[ "${DRY_RUN:-false}" == true ]]; then
        echo "DRY RUN COMMAND:"
        echo "$cmd"
        echo ""
        return 0
    fi
    
    echo "Running phase $phase_code..."
    echo "→ Log: $log_file"
    echo ""
    
    # Create output directory
    mkdir -p "output/phase_${phase_code}"
    
    # Execute command
    if ! eval "$cmd" 2>&1 | tee "$log_file"; then
        local exit_code=${PIPESTATUS[0]}
        echo "ERROR: Phase $phase_code failed (exit $exit_code)"
        echo "See: $log_file"
        return $exit_code
    fi
    
    echo ""
    echo "Phase $phase_code completed."
    
    # Show summary
    local result_file=$(find "output/phase_${phase_code}" -name "adapt_half_RSI_best_*.txt" -type f | sort -r | head -1)
    if [[ -n "$result_file" && -f "$result_file" ]]; then
        echo "Best result summary:"
        grep -E "^(Objective value|Mean ticker score|Avg PF|Avg trades)" "$result_file" | head -4
    fi
    echo ""
}

# ───────────────────────────────────────────────────────────────
# Main Execution
# ───────────────────────────────────────────────────────────────
echo "Starting adapt_half_RSI Optimization Sweep..."
echo "Configuration:"
echo "  Seed        : $SEED"
echo "  Trials/phase: $TRIALS"
echo "  Files       : $FILES"
echo "  Phases      : ${SELECTED_PHASES[*]}"
echo "  Python      : $PYTHON_BIN"
echo ""
echo "WARNING: Using conservative parameter approach"
echo ""

# First test the script
echo "Testing script compatibility..."
if ! test_script_params; then
    echo "ERROR: Script test failed. Cannot continue."
    exit 1
fi
echo ""

# Validate phases
for phase in "${SELECTED_PHASES[@]}"; do
    if [[ -z "${PHASE_PARAMS[$phase]:-}" ]]; then
        echo "ERROR: Phase '$phase' not defined."
        exit 1
    fi
done

# Create directories
mkdir -p "$LOG_DIR"
for phase in "${SELECTED_PHASES[@]}"; do
    mkdir -p "output/phase_${phase}"
done

# Run selected phases
for phase in "${SELECTED_PHASES[@]}"; do
    echo "────────────────────────────────────────────────────────────────"
    echo " STARTING PHASE $phase"
    echo "────────────────────────────────────────────────────────────────"
    
    if ! run_phase "$phase"; then
        echo "FATAL: Phase $phase failed → stopping."
        exit 1
    fi

    if [[ "$phase" != "${SELECTED_PHASES[-1]}" ]]; then
        echo "--- Pausing 3 seconds ---"
        sleep 3
    fi
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "Optimization complete!"
echo "Results in: output/phase_[A-C]/"
echo "Logs in: $LOG_DIR/"
echo "═══════════════════════════════════════════════════════════════"

# Clean up test directory
rm -rf "test_params" 2>/dev/null || true