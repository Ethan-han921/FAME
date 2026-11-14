#!/bin/bash
cd "$(dirname "$0")"

for run in 1 2 3; do
    echo "========== SEMI_REL RUN $run / 3 =========="
    python pipeline.py --dataset semi_rel --mode two_stage --model gpt-5.6-luna
    python pipeline.py --dataset semi_rel --mode whole --model gpt-5.6-luna
    python pipeline.py --dataset semi_rel --mode evidence --model gpt-5.6-luna
done
echo "SEMI_REL done!"
