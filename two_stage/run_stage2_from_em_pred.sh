#!/bin/bash
# Experiment Group 1: Stage 2 Few-shot using EM predictions (matched_pairs.json from Stage 1)
# 3 datasets × 3 logs = 9 experiments

cd /Users/yixuan/Documents/GitHub/machamp-main/two_stage

LOG_ROOT="/Users/yixuan/Documents/GitHub/machamp-main/two_stage/logs"

echo "=============================================="
echo "Stage 2 Few-shot from EM Predictions (9 runs)"
echo "=============================================="

# NYC (3 logs)
echo "[1/9] NYC - Run 1"
python pipeline.py --dataset nyc --stage2-only \
    --data-file "$LOG_ROOT/nyc_two_stage/nyc_gpt-5.6-luna_20261026_222759/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run1

echo "[2/9] NYC - Run 2"
python pipeline.py --dataset nyc --stage2-only \
    --data-file "$LOG_ROOT/nyc_two_stage/nyc_gpt-5.6-luna_20261026_230112/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run2

echo "[3/9] NYC - Run 3"
python pipeline.py --dataset nyc --stage2-only \
    --data-file "$LOG_ROOT/nyc_two_stage/nyc_gpt-5.6-luna_20261026_233420/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_nyc.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run3

# Semi-Rel (3 logs)
echo "[4/9] Semi-Rel - Run 1"
python pipeline.py --dataset semi_rel --stage2-only \
    --data-file "$LOG_ROOT/semi_rel_two_stage/semi_rel_gpt-5.6-luna_20261027_200023/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run1

echo "[5/9] Semi-Rel - Run 2"
python pipeline.py --dataset semi_rel --stage2-only \
    --data-file "$LOG_ROOT/semi_rel_two_stage/semi_rel_gpt-5.6-luna_20261027_211925/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run2

echo "[6/9] Semi-Rel - Run 3"
python pipeline.py --dataset semi_rel --stage2-only \
    --data-file "$LOG_ROOT/semi_rel_two_stage/semi_rel_gpt-5.6-luna_20261027_223452/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_rel.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run3

# Semi-Heter (3 logs)
echo "[7/9] Semi-Heter - Run 1"
python pipeline.py --dataset semi_heter --stage2-only \
    --data-file "$LOG_ROOT/semi_heter_two_stage/semi_heter_gpt-5.6-luna_20261026_222816/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run1

echo "[8/9] Semi-Heter - Run 2"
python pipeline.py --dataset semi_heter --stage2-only \
    --data-file "$LOG_ROOT/semi_heter_two_stage/semi_heter_gpt-5.6-luna_20261026_234028/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run2

echo "[9/9] Semi-Heter - Run 3"
python pipeline.py --dataset semi_heter --stage2-only \
    --data-file "$LOG_ROOT/semi_heter_two_stage/semi_heter_gpt-5.6-luna_20261027_004940/stage1_em/matched_pairs.json" \
    --few-shot-file ./data/few_shot_semi_heter.json --few-shot-num 5 --log-prefix em_pred_few_shot_5_run3

echo "=============================================="
echo "All 9 experiments completed!"
echo "=============================================="
