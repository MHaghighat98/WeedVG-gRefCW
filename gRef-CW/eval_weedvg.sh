#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

EVAL_WEEDVG="gRef-CW/eval_weedvg.py"
MODEL_ROOT="Weed-VG"

GREFS_JSON="data/grefs(unc).json"
INSTANCES_JSON="data/instances.json"
IMAGES_ROOT="data/images"
CONFIG_FILE="$MODEL_ROOT/configs/GroundingDINO_SwinB_cfg.py"
CHECKPOINT="$MODEL_ROOT/checkpoints/stage_two.pth"
BASE_CHECKPOINT="$MODEL_ROOT/weights/groundingdino_swinb_cogcoor.pth"
OUTPUT_DIR="eval_results/weedvg"

SPLITS="val,test"
TEXT_THRESHOLD=0.95
BOX_THRESHOLD=0.01
BATCH_SIZE=4
MAX_PROPOSALS=900
DECODER_LAYERS=1

python "$EVAL_WEEDVG" \
  --config-file "$CONFIG_FILE" \
  --base-detector-checkpoint "$BASE_CHECKPOINT" \
  --detector-checkpoint "$CHECKPOINT" \
  --projector-checkpoint "$CHECKPOINT" \
  --grefs-json "$GREFS_JSON" \
  --instances-json "$INSTANCES_JSON" \
  --images-root "$IMAGES_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --splits "$SPLITS" \
  --decoder-layers "$DECODER_LAYERS" \
  --detect-with-generic-prompt \
  --text-threshold "$TEXT_THRESHOLD" \
  --box-threshold "$BOX_THRESHOLD" \
  --batch-size "$BATCH_SIZE" \
  --max-proposals "$MAX_PROPOSALS" \
  --grounding-prompt "plant or vegetation" \
  --iou-thresh 0.5 \
  --top-k 5
