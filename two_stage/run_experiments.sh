#!/bin/bash

# Run experiments: 2 modes × 3 datasets × 2 runs = 12 total runs

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

echo "========================================"
echo "Starting experiments: $(date)"
echo "========================================"
echo ""

for run in 1 ; do
  echo "############################################"
  echo "# RUN $run"
  echo "############################################"
  echo ""

  for mode in two_stage whole; do
    for dataset in nyc semi_rel semi_heter; do
      echo "----------------------------------------"
      echo "Mode: $mode | Dataset: $dataset | Run: $run"
      echo "Started: $(date '+%Y-%m-%d %H:%M:%S')"
      echo "----------------------------------------"

      python pipeline.py --dataset $dataset --mode $mode

      echo ""
      echo "Finished: $(date '+%Y-%m-%d %H:%M:%S')"
      echo ""
    done
  done
done

echo "========================================"
echo "All experiments completed: $(date)"
echo "========================================"
