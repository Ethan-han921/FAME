#!/bin/bash
# Experiment Group 2: Stage 2 Few-shot with Oracle EM (label1.json - GT positive pairs)
# 3 datasets × 3 runs = 9 experiments

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

# Oracle EM data paths (label1.json = GT positive pairs)
NYC_LABEL1="/Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/nyc_label1.json"
SEMI_REL_LABEL1="/Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/semi_rel_label1.json"
SEMI_HETER_LABEL1="/Users/yixuan/Documents/GitHub/machamp-main/two_stage/data/Test_dataset/transoformd_dataset/semi_heter_label1.json"

echo "=============================================="
echo "Stage 2 Few-shot with Oracle EM (9 runs)"
echo "=============================================="

# NYC (3 runs)
echo "[1/9] NYC - Run 1"
python pipeline.py --dataset nyc --stage2-only --data-file "$NYC_LABEL1" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run1

echo "[2/9] NYC - Run 2"
python pipeline.py --dataset nyc --stage2-only --data-file "$NYC_LABEL1" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run2

echo "[3/9] NYC - Run 3"
python pipeline.py --dataset nyc --stage2-only --data-file "$NYC_LABEL1" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run3

# Semi-Rel (3 runs)
echo "[4/9] Semi-Rel - Run 1"
python pipeline.py --dataset semi_rel --stage2-only --data-file "$SEMI_REL_LABEL1" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run1

echo "[5/9] Semi-Rel - Run 2"
python pipeline.py --dataset semi_rel --stage2-only --data-file "$SEMI_REL_LABEL1" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run2

echo "[6/9] Semi-Rel - Run 3"
python pipeline.py --dataset semi_rel --stage2-only --data-file "$SEMI_REL_LABEL1" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run3

# Semi-Heter (3 runs)
echo "[7/9] Semi-Heter - Run 1"
python pipeline.py --dataset semi_heter --stage2-only --data-file "$SEMI_HETER_LABEL1" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run1

echo "[8/9] Semi-Heter - Run 2"
python pipeline.py --dataset semi_heter --stage2-only --data-file "$SEMI_HETER_LABEL1" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run2

echo "[9/9] Semi-Heter - Run 3"
python pipeline.py --dataset semi_heter --stage2-only --data-file "$SEMI_HETER_LABEL1" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix oracle_em_few_shot_5_run3

echo "=============================================="
echo "All 9 experiments completed!"
echo "=============================================="
