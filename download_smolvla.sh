#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
REPO_ROOT="$(dirname "${SCRIPT_PATH}")"
MODEL_REPO="HuggingFaceVLA/smolvla_libero"
DEFAULT_OUT_DIR="${REPO_ROOT}/models/smolvla_libero"
OUT_DIR="${1:-${DEFAULT_OUT_DIR}}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"

HF_CMD=()
if [[ -x "${REPO_ROOT}/.venv/bin/hf" ]]; then
  HF_CMD=("${REPO_ROOT}/.venv/bin/hf")
elif command -v hf >/dev/null 2>&1; then
  HF_CMD=("hf")
elif command -v uv >/dev/null 2>&1; then
  HF_CMD=("uv" "run" "--no-sync" "hf")
else
  echo "Error: hf command not found in .venv or system PATH, and uv is unavailable." >&2
  exit 1
fi

if [[ -f "${OUT_DIR}/config.json" && "${FORCE_DOWNLOAD}" != "1" ]]; then
  echo "[download_smolvla] Found existing checkpoint at ${OUT_DIR}, skip download."
  echo "[download_smolvla] Set FORCE_DOWNLOAD=1 to download again."
  exit 0
fi

mkdir -p "${OUT_DIR}"

echo "[download_smolvla] Downloading ${MODEL_REPO} to ${OUT_DIR} ..."
"${HF_CMD[@]}" download "${MODEL_REPO}" --repo-type model --local-dir "${OUT_DIR}"
echo "[download_smolvla] Done."
