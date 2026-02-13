#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/rocketegg/clawd"

printf "\n[TEST] ComfyUI pipeline (headless)\n"
"${ROOT_DIR}/scripts/test_comfyui_pipeline.sh"

printf "\nAll tests passed.\n"
