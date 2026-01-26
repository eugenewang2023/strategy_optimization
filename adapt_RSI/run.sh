#!/usr/bin/env bash
set -euo pipefail

# ───────────────────────────────────────────────────────────────
# Configuration & Defaults
# ───────────────────────────────────────────────────────────────
SCRIPT_NAME="$(basename "$0")"
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"

# Default execution settings
SEED=42
TRIALS=300
FILES=300
FILL_MODE="same_close"
DATA_DIR="data"
LOG_DIR="logs"
OUTPUT_BASE_DIR="output"

# Strategy defaults (from your best results)
COMMISSION=0.0006

SELECTED_PHASES=()
USE_PREVIOUS_RESULTS=true  # Each phase loads best params from previous phase

# ───────────────────────────────────────────────────────────────
# Usage / Help
# ───────────────────────────────────────────────────────────────
usage() {
    cat <<EOF
Usage: $SCRIPT_NAME [OPTIONS] [phase_code ...]
Progressive Adaptive RSI optimization - Each phase builds on previous results

Options:
  --seed INT          Random seed (default: $SEED)
  --trials INT        Trials per phase (default: $TRIALS)
  --files INT         Number of data files (default: $FILES)
  --fill MODE         Fill mode: next_open | same_close (default: $FILL_MODE)
  --no-progressive    Don't use previous phase results (start fresh each phase)
  --dry-run           Show commands without executing
  --help              Show this help

Phases (runs A-F if none specified):
  A     Wide Exploration - Broad parameter ranges
  B     Refinement - Narrow ranges based on A
  C     Optimization - Focus on best performers
  D     Tightening - Reduce to robust parameters
  E     Final Tuning - Small adjustments
  F     Validation - Test final parameters

Progressive Mode: Each phase uses the best parameters from the previous phase
as starting points, with progressively narrower search ranges.
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
        --no-progressive) USE_PREVIOUS_RESULTS=false; shift ;;
        --dry-run)       DRY_RUN=true; shift ;;
        --help|-h)       usage 0 ;;
        A|B|C|D|E|F)     SELECTED_PHASES+=("$1"); shift ;;
        *) echo "Unknown option: $1"; usage 1 ;;
    esac
done

# Default phases if none selected
[[ ${#SELECTED_PHASES[@]} -eq 0 ]] && SELECTED_PHASES=(A B C D E F)

mkdir -p "$LOG_DIR" "$OUTPUT_BASE_DIR"

# ───────────────────────────────────────────────────────────────
# Phase Definitions - Progressive Search Ranges
# Each phase narrows the search based on previous results
# ───────────────────────────────────────────────────────────────
declare -A PHASE_NAMES
declare -A PHASE_FLAGS  # Store optimization flags for each phase

PHASE_NAMES["A"]="Wide Exploration - Broad parameter discovery"
PHASE_NAMES["B"]="Refinement - Narrow based on A results"
PHASE_NAMES["C"]="Optimization - Focus on best performers"
PHASE_NAMES["D"]="Tightening - Reduce to robust parameters"
PHASE_NAMES["E"]="Final Tuning - Small adjustments"
PHASE_NAMES["F"]="Validation - Test final parameters"

# Phase A: Initial wide exploration (all parameters optimized)
PHASE_FLAGS["A"]="
--optimize
--opt-adaptive
--opt-fastslow
--opt-cooldown
--opt-time-stop
--atrPeriod-fixed 25
--slMultiplier-fixed 3.0
--tpMultiplier-fixed 3.0
--basePeriod-fixed 20
--minPeriod-fixed 5
--maxPeriod-fixed 35
--fastPeriod-fixed 4
--slowPeriod-fixed 50
--smooth_len-fixed 5
--shift-fixed 0
--threshold-fixed 0.5
--threshold-floor 0.1
--threshold-std-mult 0.5
--cooldown 1
--time-stop 0
"

# Phase B: Narrower exploration (only some parameters optimized)
PHASE_FLAGS["B"]="
--optimize
--opt-adaptive
--opt-fastslow
--opt-cooldown
--opt-time-stop
--threshold-floor 0.05
--threshold-std-mult 0.3
--cooldown 1
--time-stop 5
"

# Phase C: Focus on key parameters
PHASE_FLAGS["C"]="
--optimize
--opt-adaptive
--opt-fastslow
--cooldown 3
--time-stop 10
--threshold-floor 0.03
--threshold-std-mult 0.2
"

# Phase D: Tight tuning
PHASE_FLAGS["D"]="
--optimize
--opt-fastslow
--cooldown 3
--time-stop 12
--threshold-floor 0.04
--threshold-std-mult 0.15
"

# Phase E: Very tight tuning
PHASE_FLAGS["E"]="
--optimize
--cooldown 4
--time-stop 12
--threshold-floor 0.045
--threshold-std-mult 0.1
"

# Phase F: Validation with fixed best parameters (no optimization)
PHASE_FLAGS["F"]="
--report-only
--atrPeriod-fixed 14
--slMultiplier-fixed 1.85
--tpMultiplier-fixed 3.65
--basePeriod-fixed 21
--minPeriod-fixed 5
--maxPeriod-fixed 20
--fastPeriod-fixed 5
--slowPeriod-fixed 26
--smooth_len-fixed 3
--shift-fixed 3
--threshold-fixed 0.055
--threshold-floor 0.05
--threshold-std-mult 0.05
--cooldown 3
--time-stop 12
"

# ───────────────────────────────────────────────────────────────
# Load Best Parameters from Previous Phase
# ───────────────────────────────────────────────────────────────
load_previous_params() {
    local prev_phase="$1"
    local param_file=""
    
    # Find the latest best result file from previous phase
    param_file=$(find "$OUTPUT_BASE_DIR/phase_${prev_phase}" -name "adapt_RSI_best_*.txt" -type f 2>/dev/null | sort -r | head -1)
    
    if [[ -n "$param_file" && -f "$param_file" ]]; then
        echo "Loading best parameters from: $param_file"
        
        # Extract parameters from the result file
        local params=""
        params=$(grep -A 20 "Best parameters:" "$param_file" | tail -n +2 | grep -v "^$" | while read -r line; do
            key=$(echo "$line" | awk -F ':' '{print $1}' | xargs)
            value=$(echo "$line" | awk -F ':' '{print $2}' | xargs)
            # Convert parameter names to what adapt_RSI.py expects
            case "$key" in
                "adapt_k") echo "--threshold-std-mult $value" ;;
                "atrPeriod") echo "--atrPeriod-fixed $value" ;;
                "base_slow_window") echo "--slowPeriod-fixed $value" ;;
                "cooldown") echo "--cooldown $value" ;;
                "shift") echo "--shift-fixed $value" ;;
                "slMultiplier") echo "--slMultiplier-fixed $value" ;;
                "smooth_len") echo "--smooth_len-fixed $value" ;;
                "time_stop") echo "--time-stop $value" ;;
                "tpMultiplier") echo "--tpMultiplier-fixed $value" ;;
                *) echo "--${key}-fixed $value" ;;
            esac
        done)
        
        echo "$params"
        return 0
    else
        echo "No previous results found for phase $prev_phase"
        return 1
    fi
}

# ───────────────────────────────────────────────────────────────
# Build Command for Phase
# ───────────────────────────────────────────────────────────────
build_command() {
    local phase="$1"
    local prev_phase=""
    
    # Determine previous phase
    case "$phase" in
        "A") prev_phase="" ;;
        "B") prev_phase="A" ;;
        "C") prev_phase="B" ;;
        "D") prev_phase="C" ;;
        "E") prev_phase="D" ;;
        "F") prev_phase="E" ;;
    esac
    
    local base_flags="${PHASE_FLAGS[$phase]}"
    
    # If using progressive mode and we have a previous phase, load its best params
    if [[ "$USE_PREVIOUS_RESULTS" == true && -n "$prev_phase" && "$phase" != "F" ]]; then
        local previous_params
        if previous_params=$(load_previous_params "$prev_phase"); then
            echo "Using progressive optimization with previous phase $prev_phase results"
            
            # For phases B-E, use previous params as fixed values
            local fixed_params=""
            while IFS= read -r param; do
                if [[ -n "$param" ]]; then
                    # Extract parameter name
                    if [[ "$param" =~ ^--([^-]+)- ]]; then
                        param_name="${BASH_REMATCH[1]}"
                        
                        # Check if this parameter should remain optimized in current phase
                        case "$phase" in
                            "B"|"C"|"D"|"E")
                                # For these phases, some parameters remain optimized
                                case "$param_name" in
                                    "atrPeriod"|"slMultiplier"|"tpMultiplier"|"basePeriod"|"minPeriod"|"maxPeriod"|"fastPeriod"|"slowPeriod"|"smooth_len"|"shift")
                                        # These remain optimized, don't fix them
                                        continue
                                        ;;
                                    *)
                                        # Keep as fixed
                                        fixed_params+="$param"$'\n'
                                        ;;
                                esac
                                ;;
                            *)
                                # Keep all as fixed
                                fixed_params+="$param"$'\n'
                                ;;
                        esac
                    fi
                fi
            done <<< "$previous_params"
            
            echo "$base_flags"
            echo "$fixed_params"
            return
        fi
    fi
    
    # Not using progressive mode or no previous results
    echo "$base_flags"
}

# ───────────────────────────────────────────────────────────────
# Run Phase
# ───────────────────────────────────────────────────────────────
run_phase() {
    local phase="$1"
    local phase_name="${PHASE_NAMES[$phase]}"
    local log_file="$LOG_DIR/phase_${phase}_adapt_RSI_$(date +%Y%m%d_%H%M%S).log"
    local output_dir="$OUTPUT_BASE_DIR/phase_${phase}"
    
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PHASE $phase : $phase_name"
    echo "═══════════════════════════════════════════════════════════════"
    
    # Build the command
    local phase_flags
    phase_flags=$(build_command "$phase")
    
    # Build command array
    local CMD=(
        "$PYTHON_BIN" adapt_RSI.py
        --seed "$SEED"
        --trials "$TRIALS"
        --files "$FILES"
        --fill "$FILL_MODE"
        --data_dir "$DATA_DIR"
        --output_dir "$output_dir"
        --commission_rate_per_side "$COMMISSION"
        
        # Core scoring parameters (FIXED values)
        --min-trades 2
        --trades-baseline 6.0
        --weight-pf 0.6
        --score-power 1.1
        --ret-floor -0.15
        --ret-floor-k 1.0
        --pf-floor-k 0.001
        --pf-baseline 2.0
        --pf-k 2.0
        
        # Exit strategy
        --use_trailing_exit "True"
        --trail_mode "trail_only"
        --close_on_sellSignal "True"
        
        # TP/SL constraints
        --tp2sl-auto
        --tp2sl-base 1.3
        --tp2sl-sr0 25.0
        --tp2sl-k 0.015
        --tp2sl-min 1.05
        --tp2sl-max 2.0
        
        # Penalty settings
        --penalty
        --loss_floor 0.0005
        --penalty-ret-center -0.015
        --penalty-ret-k 5.0
        
        # Trade limits
        --max-trades 25
        --max-trades-k 0.20
        --pf-floor 1.2
        
        # Coverage control
        --coverage-target 0.65
        --coverage-k 10.0
        
        # Fixed parameters that don't change
        --threshold-mode "dynamic"
        --vol-floor-mult-fixed 1.0
        --vol-floor-len 100
        --trend-center 0.80
        --trend-k 3.0
    )
    
    # Add phase-specific flags
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            # Split the line into arguments
            read -ra args <<< "$line"
            for arg in "${args[@]}"; do
                CMD+=("$arg")
            done
        fi
    done <<< "$phase_flags"
    
    if [[ "${DRY_RUN:-false}" == true ]]; then
        echo "DRY RUN COMMAND:"
        echo "${CMD[*]}"
        echo ""
        echo "Phase flags for phase $phase:"
        echo "$phase_flags"
        echo ""
        return 0
    fi
    
    echo "Running phase $phase..."
    echo "→ Output: $output_dir"
    echo "→ Log: $log_file"
    echo ""
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Execute command
    echo "Executing command..."
    echo ""
    
    if ! "${CMD[@]}" 2>&1 | tee "$log_file"; then
        local exit_code=${PIPESTATUS[0]}
        echo ""
        echo "ERROR: Phase $phase failed (exit $exit_code)"
        echo "See log: $log_file"
        return $exit_code
    fi
    
    echo ""
    echo "Phase $phase completed successfully."
    
    # Show summary of best result
    local result_file=""
    result_file=$(find "$output_dir" -name "adapt_RSI_best_*.txt" -type f | sort -r | head -1)
    
    if [[ -n "$result_file" && -f "$result_file" ]]; then
        echo ""
        echo "=== BEST RESULT SUMMARY (Phase $phase) ==="
        echo "File: $(basename "$result_file")"
        grep -E "^(Objective value|Mean ticker score|Avg PF \(raw\)|Avg trades/ticker|Coverage|Negative returns)" "$result_file" | head -10
        echo ""
    fi
    
    # Extract objective value for comparison
    if [[ -f "$result_file" ]]; then
        local objective_value
        objective_value=$(grep "Objective value" "$result_file" | awk '{print $3}')
        echo "Objective value for phase $phase: $objective_value"
    fi
    
    echo ""
}

# ───────────────────────────────────────────────────────────────
# Main Execution
# ───────────────────────────────────────────────────────────────
main() {
    echo ""
    echo "┌──────────────────────────────────────────────────────────────┐"
    echo "│  Progressive Adaptive RSI Optimization                      │"
    echo "│  Each phase builds on the previous phase's best results     │"
    echo "└──────────────────────────────────────────────────────────────┘"
    echo ""
    
    echo "Configuration:"
    echo "  Seed:              $SEED"
    echo "  Trials/phase:      $TRIALS"
    echo "  Files:             $FILES"
    echo "  Phases:            ${SELECTED_PHASES[*]}"
    echo "  Progressive mode:  $USE_PREVIOUS_RESULTS"
    echo "  Python:            $PYTHON_BIN"
    echo "  Data dir:          $DATA_DIR"
    echo ""
    
    # Validate phases
    for phase in "${SELECTED_PHASES[@]}"; do
        if [[ -z "${PHASE_NAMES[$phase]:-}" ]]; then
            echo "ERROR: Phase '$phase' not defined."
            exit 1
        fi
    done
    
    # Run phases in sequence
    for phase in "${SELECTED_PHASES[@]}"; do
        echo "────────────────────────────────────────────────────────────────"
        echo " STARTING PHASE $phase: ${PHASE_NAMES[$phase]}"
        echo "────────────────────────────────────────────────────────────────"
        
        if ! run_phase "$phase"; then
            echo "FATAL: Phase $phase failed → stopping."
            exit 1
        fi
        
        # Pause between phases (except after last one)
        if [[ "$phase" != "${SELECTED_PHASES[-1]}" ]]; then
            echo "--- Pausing 5 seconds before next phase ---"
            sleep 5
        fi
    done
    
    # Final summary
    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo " PROGRESSIVE OPTIMIZATION COMPLETE!"
    echo "═══════════════════════════════════════════════════════════════"
    echo ""
    
    # Show results from all phases
    echo "Results by phase:"
    echo "┌───────┬─────────────────────────────┬──────────────────────┐"
    echo "│ Phase │ Objective Value            │ Coverage             │"
    echo "├───────┼─────────────────────────────┼──────────────────────┤"
    
    for phase in "${SELECTED_PHASES[@]}"; do
        local result_file=""
        result_file=$(find "$OUTPUT_BASE_DIR/phase_${phase}" -name "adapt_RSI_best_*.txt" -type f 2>/dev/null | sort -r | head -1)
        
        if [[ -n "$result_file" && -f "$result_file" ]]; then
            local objective_value coverage
            objective_value=$(grep "Objective value" "$result_file" | awk '{print $3}')
            coverage=$(grep "Coverage" "$result_file" | awk '{print $2}')
            
            printf "│ %-5s │ %-27s │ %-20s │\n" "$phase" "$objective_value" "$coverage"
        else
            printf "│ %-5s │ %-27s │ %-20s │\n" "$phase" "No result" "N/A"
        fi
    done
    
    echo "└───────┴─────────────────────────────┴──────────────────────┘"
    echo ""
    echo "Output directories:"
    for phase in "${SELECTED_PHASES[@]}"; do
        echo "  Phase $phase: $OUTPUT_BASE_DIR/phase_${phase}/"
    done
    echo ""
    echo "Logs: $LOG_DIR/"
    echo ""
}

# Run main function
main