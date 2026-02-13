#!/usr/bin/env bash
set -euo pipefail
BASE_MODEL=${1:-hassaku_xl_illustrious.safetensors}
LORA=${2:-}
PROMPT=${3:-"test image of a cat"}
OUT="/home/rocketegg/clawd/dashboard/assets/test_ui_${BASE_MODEL//[^a-zA-Z0-9]/_}_$(date +%s).png"

/home/rocketegg/workspace/pytorch_cuda/.venv/bin/python3 /home/rocketegg/clawd/dashboard/gen_creative.py "$LORA" "$PROMPT" "$OUT" "$BASE_MODEL"

if [[ -f "$OUT" ]]; then
  echo "OK: generated $OUT"
  ls -lh "$OUT"
else
  echo "FAIL: no output file generated"
  exit 1
fi
