#!/usr/bin/env bash
set -euo pipefail

THREADS="${THREADS:-data/processed/threads.jsonl}"
SPLITS="${SPLITS:-data/processed/splits_loeo.json}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"

if (( $# > 0 )); then
    MINUTES=("$@")
else
    MINUTES=(10 30 60 120 240)
fi

EVENTS=()
while IFS= read -r EVENT; do
    EVENTS+=("$EVENT")
done < <(
    python -c 'import json, sys; print(*sorted(json.load(open(sys.argv[1], encoding="utf-8"))), sep="\n")' "$SPLITS"
)

if (( ${#EVENTS[@]} == 0 )); then
    echo "No folds found in $SPLITS" >&2
    exit 1
fi

for MINS in "${MINUTES[@]}"; do
    for EVENT in "${EVENTS[@]}"; do
        echo "Running early detection: ${MINS} minutes | fold: ${EVENT}"

        python src/train_gnn.py \
            --threads "$THREADS" \
            --splits "$SPLITS" \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "$RESULTS_ROOT/early_${MINS}/gnn/$EVENT"

        python src/train_fusion.py \
            --threads "$THREADS" \
            --splits "$SPLITS" \
            --fold "$EVENT" \
            --early_minutes "$MINS" \
            --out_dir "$RESULTS_ROOT/early_${MINS}/fuse_t/$EVENT" \
            --init_text_ckpt "$RESULTS_ROOT/text/$EVENT/best.pt" \
            --init_gnn_ckpt "$RESULTS_ROOT/early_${MINS}/gnn/$EVENT/best.pt"
    done
done
