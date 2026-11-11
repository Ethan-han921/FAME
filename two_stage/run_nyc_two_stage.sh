#!/bin/bash
cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

for run in 1 2 3; do
    echo "========== NYC TWO_STAGE RUN $run / 3 =========="
    python pipeline.py --dataset nyc --data-file /Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/nyc_combined.json --mode two_stage --model gpt-5.6-luna
done
echo "NYC two_stage done!"
