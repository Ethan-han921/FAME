#!/bin/bash
# Run Stage 1 only (Entity Matching) on full datasets
# Each dataset runs 1 time

set -e

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

for DATASET in nyc semi_heter semi_rel; do
    echo "=============================================="
    echo "Dataset: $DATASET | Stage 1 Only"
    echo "=============================================="

    python pipeline.py \
        --dataset "$DATASET" \
        --mode two_stage \
        --stage1-only \
        --log-prefix LLM \
        --data-file "data/Test_dataset/transoformd_dataset/${DATASET}_combined.json"

    echo ""
done

echo "=============================================="
echo "All 3 datasets completed! (Stage 1 only)"
echo "=============================================="
