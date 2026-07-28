#!/bin/bash
# /*****************************************************************************/
# * File: run_synth.sh
# * Author: Olavo Alves Barros Silva
# * Contact: olavo.barros@ufv.com
# * Date: 2026-07-24
# * Description: Runs Yosys's coarse area/cell-count check (no techmap/abc -
# * see src/hw/cnn/synth.ys's header for why) for every dataset that already
# * has a generated results/<dataset>/hw/synth.ys (written by
# * export_cnn_weights.py <Dataset>). Safe to run concurrently with run.sh:
# * it only READS artifacts run.sh/export_cnn_weights.py already produced,
# * and writes into results/<dataset>/hw/area_report.json, a file run.sh
# * never touches. Datasets not processed yet by export_cnn_weights.py are
# * skipped with a note (not an error) - just re-run this script later.
# *
# * Usage:
# *   ./run_synth.sh                  # every dataset with a synth.ys ready
# *   ./run_synth.sh Wafer FordA      # only these datasets
# /*****************************************************************************/

RED='\033[0;31m'
GREEN='\033[0;32m'
GOLD='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
RESET='\033[0m'

echo_error()   { echo -e "${RED}${BOLD}[ERROR]${RESET} $1"; }
echo_success() { echo -e "${GREEN}${BOLD}[SUCCESS]${RESET} $1"; }
echo_info()    { echo -e "${BLUE}${BOLD}[INFO]${RESET} $1"; }
echo_warning() { echo -e "${GOLD}${BOLD}[WARNING]${RESET} $1"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HW_DIR="$REPO_ROOT/src/hw/cnn"

if [[ -z "${VIRTUAL_ENV:-}" && -d "$REPO_ROOT/.venv" ]]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

if [[ $# -gt 0 ]]; then
    DATASETS=("$@")
else
    mapfile -t DATASETS < <(python -c "
import json
with open('$REPO_ROOT/config.json') as f:
    print('\n'.join(json.load(f)['datasets']))
")
fi

echo_info "Datasets to check: ${DATASETS[*]}"

DONE=()
SKIPPED=()
FAILED=()

for DATASET in "${DATASETS[@]}"; do
    SYNTH_YS="$REPO_ROOT/results/$DATASET/hw/synth.ys"
    if [[ ! -f "$SYNTH_YS" ]]; then
        echo_warning "[$DATASET] SKIP: $SYNTH_YS not found yet - export_cnn_weights.py hasn't run for it. Try again later."
        SKIPPED+=("$DATASET")
        continue
    fi

    HW_DIR_REL="$(python -c "import os; print(os.path.relpath('$REPO_ROOT/results/$DATASET/hw', '$HW_DIR'))")"

    echo_info "[$DATASET] Running Yosys coarse area check..."
    if ! (cd "$HW_DIR" && yosys "$HW_DIR_REL/synth.ys"); then
        echo_error "[$DATASET] Yosys failed."
        FAILED+=("$DATASET")
        continue
    fi

    echo_success "[$DATASET] area_report.json written to results/$DATASET/hw/"
    DONE+=("$DATASET")
done

echo ""
echo "=============================================="
echo_info "Yosys: ${#DONE[@]} done, ${#SKIPPED[@]} skipped (not ready), ${#FAILED[@]} failed."
if [[ ${#FAILED[@]} -gt 0 ]]; then
    echo_error "Failed: ${FAILED[*]}"
fi
if [[ ${#SKIPPED[@]} -gt 0 ]]; then
    echo_warning "Skipped (re-run later): ${SKIPPED[*]}"
fi

if [[ ${#DONE[@]} -gt 0 ]]; then
    echo_info "Collecting area summary across all available datasets..."
    (cd "$HW_DIR" && python collect_area_summary.py)
fi
