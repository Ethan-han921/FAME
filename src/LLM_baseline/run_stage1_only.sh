#!/bin/bash
cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

for run in 1 2 3; do
    echo "========== SEMI_REL STAGE1 ONLY RUN $run / 3 =========="
    python pipeline.py --dataset semi_rel --data-file /Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/semi_rel_test_pairs.json --mode two_stage --stage1-only --model gpt-5.6-luna --log-prefix stage1

    echo "========== SEMI_HETER STAGE1 ONLY RUN $run / 3 =========="
    python pipeline.py --dataset semi_heter --data-file /Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/semi_heter_test_pairs.json --mode two_stage --stage1-only --model gpt-5.6-luna --log-prefix stage1
done

echo "========== ALL STAGE1 DONE =========="
