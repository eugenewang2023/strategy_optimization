#!/usr/bin/env bash
set -euo pipefail

# =========================
# run_report.sh
# Default report run for Bayes_opt_adapt_RSI.py
# (script defaults are already set to your fixed params)
# =========================

PYTHON=${PYTHON:-python3}
SCRIPT=${SCRIPT:-Bayes_opt_adapt_RSI.py}

DATA_DIR=${DATA_DIR:-data}
OUT_DIR=${OUT_DIR:-output}
FILES=${FILES:-200}
SEED=${SEED:-7}

# Optional: choose fill for single-fill report: same_close or next_open
FILL=${FILL:-same_close}

# Optional: set to 1 to report both fills (same_close + next_open)
REPORT_BOTH_FILLS=${REPORT_BOTH_FILLS:-1}

# Optional: scoring knobs (leave as defaults if you want)
COMMISSION_PER_SIDE=${COMMISSION_PER_SIDE:-0.0006}

# -------------------------
# Build command
# -------------------------
cmd=(
  "$PYTHON" "$SCRIPT"
  --data_dir "$DATA_DIR"
  --output_dir "$OUT_DIR"
  --files "$FILES"
  --seed "$SEED"
  --commission_rate_per_side "$COMMISSION_PER_SIDE"
)

if [[ "$REPORT_BOTH_FILLS" == "1" ]]; then
  cmd+=( --report-both-fills )
else
  cmd+=( --fill "$FILL" )
fi

echo "Running:"
printf ' %q' "${cmd[@]}"
echo
echo

"${cmd[@]}"
