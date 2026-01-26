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
FILL_MODE="next_open"
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
declare -A PHASE_SEARCH_RANGES

PHASE_NAMES["A"]="Wide Exploration - Broad parameter discovery"
PHASE_NAMES["B"]="Refinement - Narrow based on A results"
PHASE_NAMES["C"]="Optimization - Focus on best performers"
PHASE_NAMES["D"]="Tightening - Reduce to robust parameters"
PHASE_NAMES["E"]="Final Tuning - Small adjustments"
PHASE_NAMES["F"]="Validation - Test final parameters"

# Phase A: Initial wide ranges (based on typical RSI parameters)
PHASE_SEARCH_RANGES["A"]="
--basePeriod-min 10 --basePeriod-max 30
--minPeriod-min 2 --minPeriod-max 8
--maxPeriod-min 10 --maxPeriod-max 40
--fastPeriod-min 2 --fastPeriod-max 10
--slowPeriod-min 10 --slowPeriod-max 60
--smooth_len-min 1 --smooth_len-max 8
--shift-min 0 --shift-max 5
--threshold_floor-min 0.01 --threshold_floor-max 0.15
--threshold_std_mult-min 0.01 --threshold_std_mult-max 1.0
--atrPeriod-min 5 --atrPeriod-max 25
--slMultiplier-min 1.0 --slMultiplier-max 3.0
--tpMultiplier-min 1.5 --tpMultiplier-max 6.0
--cooldown-min 0 --cooldown-max 10
--time_stop-min 2 --time_stop-max 30
"

# Phase B: Narrower ranges (focus on areas that worked in A)
PHASE_SEARCH_RANGES["B"]="
--basePeriod-min 15 --basePeriod-max 25
--minPeriod-min 3 --minPeriod-max 7
--maxPeriod-min 12 --maxPeriod-max 30
--fastPeriod-min 3 --fastPeriod-max 8
--slowPeriod-min 15 --slowPeriod-max 45
--smooth_len-min 2 --smooth_len-max 6
--shift-min 1 --shift-max 4
--threshold_floor-min 0.02 --threshold_floor-max 0.10
--threshold_std_mult-min 0.02 --threshold_std_mult-max 0.5
--atrPeriod-min 8 --atrPeriod-max 20
--slMultiplier-min 1.2 --slMultiplier-max 2.5
--tpMultiplier-min 2.0 --tpMultiplier-max 5.0
--cooldown-min 1 --cooldown-max 8
--time_stop-min 5 --time_stop-max 20
"

# Phase C: Even narrower (focus on best performers from B)
PHASE_SEARCH_RANGES["C"]="
--basePeriod-min 18 --basePeriod-max 22
--minPeriod-min 4 --minPeriod-max 6
--maxPeriod-min 15 --maxPeriod-max 25
--fastPeriod-min 4 --fastPeriod-max 7
--slowPeriod-min 20 --slowPeriod-max 35
--smooth_len-min 3 --smooth_len-max 5
--shift-min 2 --shift-max 4
--threshold_floor-min 0.03 --threshold_floor-max 0.08
--threshold_std_mult-min 0.03 --threshold_std_mult-max 0.3
--atrPeriod-min 10 --atrPeriod-max 18
--slMultiplier-min 1.5 --slMultiplier-max 2.2
--tpMultiplier-min 2.5 --tpMultiplier-max 4.5
--cooldown-min 2 --cooldown-max 6
--time_stop-min 8 --time_stop-max 15
"

# Phase D: Tight ranges around likely optimal values
PHASE_SEARCH_RANGES["D"]="
--basePeriod-min 19 --basePeriod-max 21
--minPeriod-min 4 --minPeriod-max 5
--maxPeriod-min 18 --maxPeriod-max 22
--fastPeriod-min 5 --fastPeriod-max 6
--slowPeriod-min 25 --slowPeriod-max 30
--smooth_len-min 3 --smooth_len-max 4
--shift-min 2 --shift-max 3
--threshold_floor-min 0.04 --threshold_floor-max 0.07
--threshold_std_mult-min 0.04 --threshold_std_mult-max 0.2
--atrPeriod-min 12 --atrPeriod-max 16
--slMultiplier-min 1.7 --slMultiplier-max 2.0
--tpMultiplier-min 3.0 --tpMultiplier-max 4.0
--cooldown-min 3 --cooldown-max 5
--time_stop-min 10 --time_stop-max 12
"

# Phase E: Very tight ranges for final tuning
PHASE_SEARCH_RANGES["E"]="
--basePeriod-min 20 --basePeriod-max 21
--minPeriod-min 4 --minPeriod-max 5
--maxPeriod-min 19 --maxPeriod-max 21
--fastPeriod-min 5 --fastPeriod-max 5
--slowPeriod-min 26 --slowPeriod-max 28
--smooth_len-min 3 --smooth_len-max 3
--shift-min 3 --shift-max 3
--threshold_floor-min 0.05 --threshold_floor-max 0.06
--threshold_std_mult-min 0.05 --threshold_std_mult-max 0.1
--atrPeriod-min 13 --atrPeriod-max 15
--slMultiplier-min 1.8 --slMultiplier-max 1.9
--tpMultiplier-min 3.5 --tpMultiplier-max 3.8
--cooldown-min 3 --cooldown-max 4
--time_stop-min 11 --time_stop-max 12
"

# Phase F: Validation with fixed best parameters (no optimization)
PHASE_SEARCH_RANGES["F"]="
--basePeriod-fixed 21
--minPeriod-fixed 5
--maxPeriod-fixed 20
--fastPeriod-fixed 5
--slowPeriod-fixed 26
--smooth_len-fixed 3
--shift-fixed 3
--threshold_floor-fixed 0.055
--threshold_std_mult-fixed 0.05
--atrPeriod-fixed 14
--slMultiplier-fixed 1.85
--tpMultiplier-fixed 3.65
--cooldown-fixed 3
--time_stop-fixed 12
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
            echo "--${key}-fixed $value"
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
    
    local search_params="${PHASE_SEARCH_RANGES[$phase]}"
    
    # If using progressive mode and we have a previous phase, load its best params
    if [[ "$USE_PREVIOUS_RESULTS" == true && -n "$prev_phase" ]]; then
        local previous_params
        if previous_params=$(load_previous_params "$prev_phase"); then
            echo "Using progressive optimization with previous phase $prev_phase results"
            
            # For Phase F, use only fixed values from previous phase
            if [[ "$phase" == "F" ]]; then
                echo "$previous_params"
            else
                # For phases B-E, convert fixed params to search ranges
                local converted_params=""
                while IFS= read -r param; do
                    if [[ -n "$param" ]] && [[ "$param" == *"-fixed"* ]]; then
                        # Extract parameter name and value
                        param_name=$(echo "$param" | sed 's/--\(.*\)-fixed.*/\1/')
                        value=$(echo "$param" | awk '{print $2}')
                        
                        # Define range based on phase
                        case "$phase" in
                            "B")
                                case "$param_name" in
                                    "basePeriod"|"minPeriod"|"maxPeriod"|"fastPeriod"|"slowPeriod"|"smooth_len"|"shift"|"atrPeriod"|"cooldown"|"time_stop")
                                        range=3
                                        min=$(( $(printf "%.0f" "$value") - range ))
                                        max=$(( $(printf "%.0f" "$value") + range ))
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                    "slMultiplier"|"tpMultiplier"|"threshold_floor"|"threshold_std_mult")
                                        percent=0.3
                                        min=$(echo "$value * (1 - $percent)" | bc -l)
                                        max=$(echo "$value * (1 + $percent)" | bc -l)
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                esac
                                ;;
                            "C")
                                case "$param_name" in
                                    "basePeriod"|"minPeriod"|"maxPeriod"|"fastPeriod"|"slowPeriod"|"smooth_len"|"shift"|"atrPeriod"|"cooldown"|"time_stop")
                                        range=2
                                        min=$(( $(printf "%.0f" "$value") - range ))
                                        max=$(( $(printf "%.0f" "$value") + range ))
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                    "slMultiplier"|"tpMultiplier"|"threshold_floor"|"threshold_std_mult")
                                        percent=0.2
                                        min=$(echo "$value * (1 - $percent)" | bc -l)
                                        max=$(echo "$value * (1 + $percent)" | bc -l)
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                esac
                                ;;
                            "D")
                                case "$param_name" in
                                    "basePeriod"|"minPeriod"|"maxPeriod"|"fastPeriod"|"slowPeriod"|"smooth_len"|"shift"|"atrPeriod"|"cooldown"|"time_stop")
                                        range=1
                                        min=$(( $(printf "%.0f" "$value") - range ))
                                        max=$(( $(printf "%.0f" "$value") + range ))
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                    "slMultiplier"|"tpMultiplier"|"threshold_floor"|"threshold_std_mult")
                                        percent=0.1
                                        min=$(echo "$value * (1 - $percent)" | bc -l)
                                        max=$(echo "$value * (1 + $percent)" | bc -l)
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                esac
                                ;;
                            "E")
                                case "$param_name" in
                                    "basePeriod"|"minPeriod"|"maxPeriod"|"fastPeriod"|"slowPeriod"|"smooth_len"|"shift"|"atrPeriod"|"cooldown"|"time_stop")
                                        range=0
                                        min=$value
                                        max=$value
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                    "slMultiplier"|"tpMultiplier"|"threshold_floor"|"threshold_std_mult")
                                        percent=0.05
                                        min=$(echo "$value * (1 - $percent)" | bc -l)
                                        max=$(echo "$value * (1 + $percent)" | bc -l)
                                        echo "--${param_name}-min $min --${param_name}-max $max"
                                        ;;
                                esac
                                ;;
                        esac
                    fi
                done <<< "$previous_params"
            fi
        else
            # Fall back to default search ranges if no previous results
            echo "$search_params"
        fi
    else
        # Not using progressive mode, use predefined search ranges
        echo "$search_params"
    fi
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
    local search_params
    search_params=$(build_command "$phase")
    
    # Parse whether this is an optimization phase or validation phase
    local optimize_flag="--optimize"
    if [[ "$phase" == "F" ]]; then
        optimize_flag=""
        echo "Phase $phase: Validation (no optimization)"
    else
        echo "Phase $phase: Optimization"
    fi
    
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
        
        # Core scoring parameters
        --min-trades 2
        --trades-baseline 6.0
        --weight-pf 0.6
        --score-power 1.1
        --ret-floor -0.15
        --ret-floor-k 1.0
        --pf-floor-k 0.001
        --pf-baseline 2.0
        --pf-k 2.0
        
        # Exit strategy (FIXED: added True argument)
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
        --loss-floor 0.0005
        --penalty-ret-center -0.015
        --penalty-ret-k 5.0
        
        # Trade limits
        --max-trades 25
        --max-trades-k 0.20
        --pf-floor 1.2
        
        # Coverage control
        --coverage-target 0.65
        --coverage-k 10.0
    )
    
    # Add optimize flag if needed
    if [[ -n "$optimize_flag" ]]; then
        CMD+=("$optimize_flag")
    fi
    
    # Add search parameters
    while IFS= read -r line; do
        if [[ -n "$line" ]]; then
            # Split the line into arguments
            read -ra args <<< "$line"
            for arg in "${args[@]}"; do
                CMD+=("$arg")
            done
        fi
    done <<< "$search_params"
    
    if [[ "${DRY_RUN:-false}" == true ]]; then
        echo "DRY RUN COMMAND:"
        echo "${CMD[*]}"
        echo ""
        echo "Search parameters for phase $phase:"
        echo "$search_params"
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