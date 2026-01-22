#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Use python3 if available; fall back to python
PYTHON_BIN="${PYTHON_BIN:-python3}"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || PYTHON_BIN="python"
command -v "$PYTHON_BIN" >/dev/null 2>&1 || { echo "ERROR: python not found in PATH"; exit 2; }

seed=7
nTrials=2000
nFiles=300
fillOpt="next_open"

echo ""
echo "Starting multi-phase optimization sweep..."
echo "Seed: $seed   |   Trials: $nTrials   |   Files: $nFiles"
echo ""

# ───────────────────────────────────────────────────────────────
# BASE PARAMETERS (shared across phases)
# ───────────────────────────────────────────────────────────────
base_trades_baseline=3.0
base_trades_k=1.2
base_pf_baseline=1.15
base_pf_k=2.5
base_threshold_fixed=0.012
base_vol_floor_mult_fixed=0.05
base_pf_cap=4.0
base_loss_floor=0.001

# GLPT
base_min_glpt=0.003
base_min_glpt_k=12

# Regime filter tuning (adaptive engine)
base_regime_slope_min=0.0
base_regime_persist=3

# Coverage penalty (asymmetric)
coverage_target=0.98
coverage_k=8

# ───────────────────────────────────────────────────────────────
# PHASE RUNNER
# ───────────────────────────────────────────────────────────────
run_phase() {
    local phase_name="$1"
    local min_trades="$2"
    local weight_pf="$3"
    local score_power="$4"
    local ret_floor="$5"
    local ret_floor_k="$6"
    local pf_floor_k="$7"
    local min_glpt="$8"
    local min_glpt_k="$9"
    shift 9
    local extra_args=("$@")

    echo ""
    echo "═══════════════════════════════════════════════════════════════"
    echo "               PHASE: $phase_name"
    echo "═══════════════════════════════════════════════════════════════"
    echo "min_trades     = $min_trades"
    echo "weight_pf      = $weight_pf"
    echo "score_power    = $score_power"
    echo "ret_floor/k    = $ret_floor / $ret_floor_k"
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

        # Core scoring
        --min-trades "$min_trades"
        --trades-baseline "$base_trades_baseline"
        --trades-k "$base_trades_k"
        --pf-baseline "$base_pf_baseline"
        --pf-k "$base_pf_k"
        --weight-pf "$weight_pf"
        --score-power "$score_power"

        # Threshold & volatility
        --threshold-fixed "$base_threshold_fixed"
        --vol-floor-mult-fixed "$base_vol_floor_mult_fixed"

        # PF cap
        --pf-cap "$base_pf_cap"

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

        # Regime filter (adaptive)
        --regime-slope-min "$base_regime_slope_min"
        --regime-persist "$base_regime_persist"

        # Coverage penalty (asymmetric)
        --coverage-target "$coverage_target"
        --coverage-k "$coverage_k"

        # Optimization toggles
        --opt-time-stop
        --opt-vidya
        --opt-fastslow

        # Commission & loss floor
        --commission_rate_per_side 0.0006
        --loss_floor "$base_loss_floor"
    )

    CMD+=( "${extra_args[@]}" )

    echo "Command:"
    printf '  %q' "${CMD[@]}"
    echo -e "\n"

    "${CMD[@]}"
}

# ───────────────────────────────────────────────────────────────
# PHASES (A + B recommended)
# ───────────────────────────────────────────────────────────────

# A — Balanced Aggressive + Safety
run_phase "A - Balanced Aggressive + Safety" \
    3       0.80    2.0     -0.105   2.4     7     0.003   13

# B — Higher Edge Target
run_phase "B - Higher Edge Target" \
    2       0.80    2.1     -0.11    2.4     7     0.0035  14

echo ""
echo "All phases completed."
echo "Check output/*.csv and logs for objective_score, coverage, stability_score, PF_diag, etc."
