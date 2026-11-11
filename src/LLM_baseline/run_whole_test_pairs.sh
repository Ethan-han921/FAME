#!/bin/bash
# LLM Whole mode on test_pairs.json
# 2 datasets × 3 runs = 6 experiments

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

DATA_ROOT="/Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset"

for run in 1 2 3; do
    echo "========== SEMI_REL WHOLE (test_pairs) RUN $run / 3 =========="
    python pipeline.py --dataset semi_rel \
        --data-file "$DATA_ROOT/semi_rel_test_pairs.json" \
        --mode whole \
        --model gpt-5.6-luna \
        --log-prefix whole_test_pairs

    echo "========== SEMI_HETER WHOLE (test_pairs) RUN $run / 3 =========="
    python pipeline.py --dataset semi_heter \
        --data-file "$DATA_ROOT/semi_heter_test_pairs.json" \
        --mode whole \
        --model gpt-5.6-luna \
        --log-prefix whole_test_pairs
done

echo "========== ALL WHOLE (test_pairs) DONE =========="
