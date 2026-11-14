#!/bin/bash
cd "$(dirname "$0")"

for run in 1 2 3; do
    echo "========== NYC TWO_STAGE RUN $run / 3 =========="
    python pipeline.py --dataset nyc --mode two_stage --model gpt-5.6-luna
done
echo "NYC two_stage done!"
