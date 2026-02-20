#!/usr/bin/env bash
set -euo pipefail

# Generate answers with and without context to study hallucination behavior.
# Environment overrides:
#   MODEL (required), BASE_URL (required), API_KEY (optional, default EMPTY)
#   TEMP, TOP_P, MAX_TOKENS, N_SAMPLES, CONCURRENCY, MAX_RECORDS, INPUT, OUT_ROOT
#   FORCE_CONFIDENT=1 to force confident answers in no-context mode.

: "${MODEL:?set MODEL env, e.g. openai/gpt-oss-120b}"
: "${BASE_URL:?set BASE_URL env, e.g. http://localhost:8000/v1}"
API_KEY="${API_KEY:-EMPTY}"

TEMP="${TEMP:-0.9}"
TOP_P="${TOP_P:-}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
N_SAMPLES="${N_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-32}"
MAX_RECORDS="${MAX_RECORDS:-0}"
INPUT="${INPUT:-data/wiki/en/questions_long.jsonl}"
OUT_ROOT="${OUT_ROOT:-data/wiki/en/hallucinations}"

MODEL_SLUG="$(echo "${MODEL}" | tr '/ ' '__')"
OUT_DIR="${OUT_ROOT}/${MODEL_SLUG}"
LOG_DIR="logs/hallucinations"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

common_args=(
  --input "${INPUT}"
  --base-url "${BASE_URL}"
  --api-key "${API_KEY}"
  --model "${MODEL}"
  --temperature "${TEMP}"
  --max-tokens "${MAX_TOKENS}"
  --n-samples "${N_SAMPLES}"
  --concurrency "${CONCURRENCY}"
  --max-records "${MAX_RECORDS}"
)
if [[ -n "${TOP_P}" ]]; then
  common_args+=(--top-p "${TOP_P}")
fi

run_mode() {
  local mode="$1"          # no_context | with_context
  local label="$2"         # human label

  local out="${OUT_DIR}/answers_${mode}.jsonl"
  local err="${OUT_DIR}/answers_${mode}.errors.jsonl"

  echo "[$(date -Is)] start ${label} mode=${mode} -> ${out}"

  python3 hallucination_answerer.py \
    "${common_args[@]}" \
    --prompt-mode "${mode}" \
    --out "${out}" \
    --errors-out "${err}" \
    ${FORCE_CONFIDENT:+--force-confident}

  python3 - "$mode" "$out" "$err" <<'PY'
import sys
from pathlib import Path
mode, out, err = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
def cnt(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0
print(f"mode={mode} out={cnt(out)} errors={cnt(err)}")
PY

  echo "[$(date -Is)] done ${label}"
}

if [[ "${SKIP_NO_CONTEXT:-0}" != "1" ]]; then
  run_mode "no_context" "hallucination (no context)"
else
  echo "Skipping no_context phase (SKIP_NO_CONTEXT=1)"
fi

run_mode "with_context" "grounded (with context)"

echo "[$(date -Is)] all runs completed. Outputs under ${OUT_DIR}"
