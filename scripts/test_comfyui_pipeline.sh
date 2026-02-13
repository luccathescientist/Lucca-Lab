#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/home/rocketegg/clawd"
COMFY_DIR="${ROOT_DIR}/ComfyUI"
VENV_PY="${ROOT_DIR}/.venv/bin/python3"
PORT=8188
BASE_MODEL="hassaku_xl_illustrious.safetensors"
PROMPT="test image of a cat"
OUTPUT_PATH="${ROOT_DIR}/dashboard/assets/test_pipeline_$(date +%s).png"

if [[ ! -x "${VENV_PY}" ]]; then
  echo "[ERROR] Missing venv python at ${VENV_PY}"
  exit 1
fi

if [[ ! -d "${COMFY_DIR}" ]]; then
  echo "[ERROR] Missing ComfyUI at ${COMFY_DIR}"
  exit 1
fi

cleanup() {
  if [[ -n "${COMFY_PID:-}" ]]; then
    kill "${COMFY_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Start ComfyUI headless
"${VENV_PY}" "${COMFY_DIR}/main.py" --listen 0.0.0.0 --port ${PORT} >/tmp/comfyui_test.log 2>&1 &
COMFY_PID=$!

# Wait for ComfyUI
for i in {1..30}; do
  if curl -s "http://127.0.0.1:${PORT}/system_stats" >/dev/null 2>&1; then
    break
  fi
  sleep 1
  if ! kill -0 "${COMFY_PID}" >/dev/null 2>&1; then
    echo "[ERROR] ComfyUI process exited early."
    tail -n 50 /tmp/comfyui_test.log || true
    exit 1
  fi
  if [[ $i -eq 30 ]]; then
    echo "[ERROR] ComfyUI did not become ready."
    tail -n 50 /tmp/comfyui_test.log || true
    exit 1
  fi
done

# Run pipeline via gen_creative
"${VENV_PY}" "${ROOT_DIR}/dashboard/gen_creative.py" "" "${PROMPT}" "${OUTPUT_PATH}" "${BASE_MODEL}"

if [[ -f "${OUTPUT_PATH}" ]]; then
  echo "[OK] Generated ${OUTPUT_PATH}"
  ls -lh "${OUTPUT_PATH}"
else
  echo "[ERROR] Output not created."
  exit 1
fi
