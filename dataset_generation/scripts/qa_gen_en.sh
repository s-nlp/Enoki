#!/usr/bin/env bash
set -euo pipefail

# Generate long-form questions from mined English paragraphs.

export OPENAI_API_KEY="EMPTY"
export OPENAI_BASE_URL="http://99dgx-16:8128/v1/"

LANG="en"
MODEL="${MODEL:-openai/gpt-oss-120b}"
TEMP="${TEMP:-1}"

INPUT_PARAS="data/wiki/${LANG}/paragraphs.sampled.jsonl"
OUT_QA="data/wiki/${LANG}/questions_long.jsonl"
OUT_ERR="data/wiki/${LANG}/questions_long.errors.jsonl"
OUT_RAW="${OUT_QA}.raw.jsonl"
LOG_FILE="logs/qa_gen_fast_${LANG}.log"

CONTEXT_PARAS=1
STRIDE=1
QUESTIONS_PER_CONTEXT=3
GENERATIONS=1
MAX_CONTEXT_CHARS=5500
MAX_OUTPUT_TOKENS=16000

WORKERS=12
MAX_IN_FLIGHT=50

echo "[$(date -Is)] start qa_gen_fast lang=${LANG} model=${MODEL}"

python qa_gen_fast.py \
  --input "${INPUT_PARAS}" \
  --out "${OUT_QA}" \
  --errors-out "${OUT_ERR}" \
  --raw-out "${OUT_RAW}" \
  --model "${MODEL}" \
  --temperature "${TEMP}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS}" \
  --context-paragraphs "${CONTEXT_PARAS}" \
  --stride "${STRIDE}" \
  --questions-per-context "${QUESTIONS_PER_CONTEXT}" \
  --generations "${GENERATIONS}" \
  --max-context-chars "${MAX_CONTEXT_CHARS}" \
  --language-mode auto \
  --workers "${WORKERS}" \
  --max-in-flight "${MAX_IN_FLIGHT}" \
  --log "${LOG_FILE}" \
  --log-level INFO

echo "[$(date -Is)] finished qa_gen_fast; quick counts"

python - <<PY
from pathlib import Path

paths = {
    "out": Path("${OUT_QA}"),
    "errors": Path("${OUT_ERR}"),
    "raw": Path("${OUT_RAW}"),
}
for name, p in paths.items():
    n = sum(1 for _ in p.open()) if p.exists() else 0
    print(f"{name}: {n}")
PY

echo "[$(date -Is)] done."
