#!/bin/bash
# Run Stage 2 only (Attribute Matching) on positive pairs
# Each dataset runs 3 times
# Output logs saved to: logs/gold_{dataset}_two_stage/

set -e

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

# Run 3 times for each dataset
for DATASET in nyc semi_heter semi_rel; do
    for RUN in 1 2 3; do
        echo "=============================================="
        echo "Dataset: $DATASET | Run: $RUN/3"
        echo "=============================================="

        python pipeline.py \
            --dataset "$DATASET" \
            --model gpt-5.6-luna \
            --mode two_stage \
            --stage2-only \
            --log-prefix gold \
            --data-file "data/Test_dataset/transoformd_dataset/${DATASET}_test_positive.json"

        echo ""
    done
done

echo "=============================================="
echo "All 9 runs completed! (3 datasets x 3 runs)"
echo "Logs saved to: logs/gold_*_two_stage/"
echo "=============================================="
