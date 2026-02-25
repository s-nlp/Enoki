#!/usr/bin/env bash
set -euo pipefail

# Fast Wikipedia metadata mining for English only.
# Tuned for diversity (random seeds) and later PV/topic balancing in downstream filters.

LANG="en"
N_PAGES=${N_PAGES:-80000}            # total pages to mine; override via env
PV_DAYS=90
PV_AGENT="user"
MAX_CATEGORIES=100
MW_CONCURRENCY=4
PV_CONCURRENCY=40
OUTLINKS_SAMPLE=0                    # keep 0 for speed; set >0 to also sample outlinks
USER_AGENT="WikiMetaMiner/0.2 (contact: you@example.com)"  # replace with real contact

OUT_DIR="data/wiki/${LANG}"
LOG_DIR="logs/wiki_miner"
mkdir -p "${OUT_DIR}" "${LOG_DIR}"

OUT_JSONL="${OUT_DIR}/meta_en.raw.jsonl"
ERR_JSONL="${OUT_DIR}/meta_en.errors.jsonl"
LOG_FILE="${LOG_DIR}/wiki_miner_${LANG}.log"

echo "[$(date -Is)] start wiki_miner lang=${LANG} n_pages=${N_PAGES} out=${OUT_JSONL}"

python wiki_miner.py \
    --lang "${LANG}" \
    --n-pages "${N_PAGES}" \
    --seed-mode random \
    --pv-days "${PV_DAYS}" \
    --pv-agent "${PV_AGENT}" \
    --max-categories "${MAX_CATEGORIES}" \
    --mw-concurrency "${MW_CONCURRENCY}" \
    --pv-concurrency "${PV_CONCURRENCY}" \
    --outlinks-sample "${OUTLINKS_SAMPLE}" \
    --user-agent "${USER_AGENT}" \
    --out "${OUT_JSONL}" \
    --errors-out "${ERR_JSONL}" \
    --log "${LOG_FILE}" \
    --log-level INFO

echo "[$(date -Is)] finished mining; quick stats follow"

python - <<PY
import json
from pathlib import Path

out_path = Path("${OUT_JSONL}")
err_path = Path("${ERR_JSONL}")

def count_lines(path: Path) -> int:
    return sum(1 for _ in path.open()) if path.exists() else 0

def pct(values, p):
    if not values:
        return 0
    idx = int((len(values) - 1) * p / 100)
    return values[idx]

kept = count_lines(out_path)
errs = count_lines(err_path)
print(f"records_kept={kept} errors={errs}")

pv = []
if out_path.exists():
    with out_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            v = obj.get("pv_total")
            if isinstance(v, (int, float)):
                pv.append(int(v))

if pv:
    pv_sorted = sorted(pv)
    print("pv_total quantiles:",
          {k: pct(pv_sorted, k) for k in (0, 25, 50, 75, 90, 99)})
else:
    print("pv_total quantiles: no data")
PY

echo "[$(date -Is)] done. Consider downsampling by pv buckets for balance."
