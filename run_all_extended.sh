#!/usr/bin/env bash
# =============================================================================
# PEEF — Master Extended Experiment Runner
# =============================================================================
# Runs all 18 experiment configs in sequence (12 extended + 6 new shot counts),
# then evaluates, analyses, and regenerates plots.
#
# USAGE:
#   cd "/Users/vk25/Documents/Claude/Projects/Prompt Engineering for AI Language Models"
#   bash run_all_extended.sh
#
# TIME ESTIMATE: ~4-8 hours depending on API rate limits.
# Leave this running overnight or in a tmux/screen session.
#
# SAFE TO RE-RUN: The cache in .cache/completed.txt deduplicates everything.
# If the script is interrupted, re-running it will resume from where it left off.
# =============================================================================

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Colours
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"; }
ok()  { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $1"; }
hdr() { echo -e "\n${YELLOW}══════════════════════════════════════════${NC}"; echo -e "${YELLOW} $1${NC}"; echo -e "${YELLOW}══════════════════════════════════════════${NC}"; }

# Check Python and .env
if [ ! -f ".env" ]; then
    echo "ERROR: .env file not found. Create one with ANTHROPIC_API_KEY=sk-..." >&2
    exit 1
fi

# -----------------------------------------------------------------
# Phase 1 — Extended sample sizes (12 existing configs)
# Cache will skip the original 200; only new samples are called.
# -----------------------------------------------------------------
hdr "Phase 1 of 3 — Extended sample sizes"

QA_CONFIGS=(
    "05_code/configs/qa_zero_shot.yaml"
    "05_code/configs/qa_few_shot_4shot.yaml"
    "05_code/configs/qa_cot.yaml"
    "05_code/configs/qa_zero_shot_cot.yaml"
)
SUM_CONFIGS=(
    "05_code/configs/summarisation_zero_shot.yaml"
    "05_code/configs/summarisation_few_shot_4shot.yaml"
    "05_code/configs/summarisation_cot.yaml"
    "05_code/configs/summarisation_zero_shot_cot.yaml"
)
RES_CONFIGS=(
    "05_code/configs/reasoning_zero_shot.yaml"
    "05_code/configs/reasoning_few_shot_4shot.yaml"
    "05_code/configs/reasoning_cot.yaml"
    "05_code/configs/reasoning_zero_shot_cot.yaml"
)

for cfg in "${QA_CONFIGS[@]}" "${SUM_CONFIGS[@]}" "${RES_CONFIGS[@]}"; do
    log "Running $(basename $cfg) ..."
    python run.py --config "$cfg"
    ok "Done: $(basename $cfg)"
    sleep 2   # brief pause between experiments
done

# -----------------------------------------------------------------
# Phase 2 — New shot-count configs (6 new experiments)
# All samples are fresh; no cache hits.
# -----------------------------------------------------------------
hdr "Phase 2 of 3 — New 2-shot and 8-shot experiments"

NEW_CONFIGS=(
    "05_code/configs/qa_few_shot_2shot.yaml"
    "05_code/configs/qa_few_shot_8shot.yaml"
    "05_code/configs/summarisation_few_shot_4shot_corrected.yaml"
    "05_code/configs/summarisation_few_shot_8shot.yaml"
    "05_code/configs/reasoning_few_shot_2shot.yaml"
    "05_code/configs/reasoning_few_shot_8shot.yaml"
)

for cfg in "${NEW_CONFIGS[@]}"; do
    log "Running $(basename $cfg) ..."
    python run.py --config "$cfg"
    ok "Done: $(basename $cfg)"
    sleep 2
done

# -----------------------------------------------------------------
# Phase 3 — Evaluate, Analyse, Plot
# -----------------------------------------------------------------
hdr "Phase 3 of 3 — Evaluate, analyse, and plot"

log "Scoring all results..."
python evaluate.py
ok "Evaluation complete"

log "Running statistical analysis..."
python analyse.py
ok "Analysis complete"

log "Regenerating figures..."
python plot_results.py
ok "Plots saved"

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║   ALL DONE — Full extended dataset complete!     ║${NC}"
echo -e "${GREEN}║   Check 03_results/ for scores and figures.      ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
echo "Next step: update the dissertation with the new results."
echo "Open Cowork and ask Claude to update the dissertation."
