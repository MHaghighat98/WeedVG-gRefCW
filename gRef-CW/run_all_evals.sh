#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_WEEDVG="gRef-CW/eval_weedvg.py"
EVAL_BASELINES="gRef-CW/eval_baselines.py"
MODEL_ROOT="Weed-VG"

GREFS_JSON="data/grefs(unc).json"
INSTANCES_JSON="data/instances.json"
IMAGES_ROOT="data/images"

CONFIG_T="$MODEL_ROOT/configs/GroundingDINO_SwinT_OGC.py"
CONFIG_B="$MODEL_ROOT/configs/GroundingDINO_SwinB_cfg.py"

CKP_T="$MODEL_ROOT/weights/groundingdino_swint_ogc.pth"
CKP_B="$MODEL_ROOT/weights/groundingdino_swinb_cogcoor.pth"

CKP_OURS="$MODEL_ROOT/checkpoints/stage_two.pth"

MDETR_ROOT="mdetr"

COMMON_VG="--grefs-json $GREFS_JSON --instances-json $INSTANCES_JSON \
  --images-root $IMAGES_ROOT --iou-thresh 0.5 --nacc-iou-thresh 0.5 \
  --top-k 5 --batch-size 4 --max-proposals 900 \
  --text-threshold 0.95 --box-threshold 0.01 \
  --grounding-prompt plant\ or\ vegetation"

COMMON_BL="--grefs-json $GREFS_JSON --instances-json $INSTANCES_JSON \
  --images-root $IMAGES_ROOT --iou-thresh 0.5 --top-k 5 --batch-size 4"

mkdir -p eval_results

echo "======== [1/10] GDino-T (SwinT)  TEST ========"
python -u "$EVAL_WEEDVG" --vanilla-gdino \
  --config-file "$CONFIG_T" --detector-checkpoint "$CKP_T" \
  $COMMON_VG --splits test --model-label "GDino-T" \
  --output-dir eval_results/gdino_t_test
echo "DONE"

echo "======== [2/10] GDino-B (SwinB)  TEST ========"
python -u "$EVAL_WEEDVG" --vanilla-gdino \
  --config-file "$CONFIG_B" --detector-checkpoint "$CKP_B" \
  $COMMON_VG --splits test --model-label "GDino-B" \
  --output-dir eval_results/gdino_b_test
echo "DONE"

echo "======== [3/10] Weed-VG  TEST ========"
python -u "$EVAL_WEEDVG" \
  --config-file "$CONFIG_B" \
  --base-detector-checkpoint "$CKP_B" \
  --detector-checkpoint "$CKP_OURS" \
  --projector-checkpoint "$CKP_OURS" \
  --decoder-layers 1 \
  --detect-with-generic-prompt \
  $COMMON_VG --splits test --model-label "Weed-VG" \
  --output-dir eval_results/weedvg_test
echo "DONE"

echo "======== [4/10] MDETR  TEST ========"
python -u "$EVAL_BASELINES" --model mdetr --split test \
  $COMMON_BL --mdetr-root "$MDETR_ROOT" \
  --output-dir eval_results/mdetr_test
echo "DONE"

echo "======== [5/10] SAM3  TEST ========"
python -u "$EVAL_BASELINES" --model sam3 --split test \
  $COMMON_BL --output-dir eval_results/sam3_test
echo "DONE"

echo "======== [6/10] GDino-T (SwinT)  VAL ========"
python -u "$EVAL_WEEDVG" --vanilla-gdino \
  --config-file "$CONFIG_T" --detector-checkpoint "$CKP_T" \
  $COMMON_VG --splits val --model-label "GDino-T" \
  --output-dir eval_results/gdino_t_val
echo "DONE"

echo "======== [7/10] GDino-B (SwinB)  VAL ========"
python -u "$EVAL_WEEDVG" --vanilla-gdino \
  --config-file "$CONFIG_B" --detector-checkpoint "$CKP_B" \
  $COMMON_VG --splits val --model-label "GDino-B" \
  --output-dir eval_results/gdino_b_val
echo "DONE"

echo "======== [8/10] Weed-VG  VAL ========"
python -u "$EVAL_WEEDVG" \
  --config-file "$CONFIG_B" \
  --base-detector-checkpoint "$CKP_B" \
  --detector-checkpoint "$CKP_OURS" \
  --projector-checkpoint "$CKP_OURS" \
  --decoder-layers 1 \
  --detect-with-generic-prompt \
  $COMMON_VG --splits val --model-label "Weed-VG" \
  --output-dir eval_results/weedvg_val
echo "DONE"

echo "======== [9/10] MDETR  VAL ========"
python -u "$EVAL_BASELINES" --model mdetr --split val \
  $COMMON_BL --mdetr-root "$MDETR_ROOT" \
  --output-dir eval_results/mdetr_val
echo "DONE"

echo "======== [10/10] SAM3  VAL ========"
python -u "$EVAL_BASELINES" --model sam3 --split val \
  $COMMON_BL --output-dir eval_results/sam3_val
echo "DONE"

echo ""
echo "========== ALL 10 EVALUATIONS COMPLETE =========="
echo "Results in eval_results/*/metrics.json"
