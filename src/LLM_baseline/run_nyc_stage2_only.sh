#!/bin/bash
cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

for run in 1 2 3; do
    echo "========== NYC STAGE2-ONLY RUN $run / 3 =========="
    python pipeline.py --dataset nyc --data-file /Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/nyc_label1.json --mode two_stage --stage2-only --model gpt-5.6-luna --log-prefix gold
done
echo "NYC stage2-only done!"
