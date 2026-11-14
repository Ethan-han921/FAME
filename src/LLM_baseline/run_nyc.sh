#!/bin/bash
cd "$(dirname "$0")"

for run in 1 2 3; do
    echo "========== NYC RUN $run / 3 =========="
    python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna
    python pipeline.py --dataset nyc --mode whole --model gpt-5.6-luna
    python pipeline.py --dataset nyc --mode evidence --model gpt-5.6-luna
done
echo "NYC done!"
