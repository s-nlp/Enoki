#!/usr/bin/env bash
set -euo pipefail

# Paragraph extraction for English Wikipedia meta mined by wiki_mine_en.sh

LANG="en"
USER_AGENT="WikiParagraphMiner/0.2 (contact: you@example.com)"  # replace with real contact

# META_IN="data/wiki/${LANG}/meta_en.raw.jsonl"
# OUT_PARAS="data/wiki/${LANG}/paragraphs.jsonl"
# OUT_ERR="data/wiki/${LANG}/paragraphs.errors.jsonl"

META_IN="data/wiki/en/meta_en.sampled20k.jsonl"
OUT_PARAS="data/wiki/en/paragraphs.sampled.jsonl"
OUT_ERR="data/wiki/en/paragraphs.sampled.errors.jsonl"
LOG_FILE="logs/paragraph_miner_${LANG}.log"


MIN_PAR_CHARS=300          # keep reasonably detailed paragraphs (not too aggressive)
MAX_PAR_CHARS=2500         # clamp very long blobs by sentence split
MAX_PAR_PER_PAGE=32
MAX_CHARS_EXTRACT=200000
BATCH_SIZE=12
CONCURRENCY=6

echo "[$(date -Is)] start paragraph_miner lang=${LANG} meta=${META_IN}"

python paragraph_miner.py \
  --lang "${LANG}" \
  --in-meta "${META_IN}" \
  --out "${OUT_PARAS}" \
  --errors-out "${OUT_ERR}" \
  --extract-mode full \
  --max-chars "${MAX_CHARS_EXTRACT}" \
  --min-paragraph-chars "${MIN_PAR_CHARS}" \
  --max-paragraph-chars "${MAX_PAR_CHARS}" \
  --max-paragraphs-per-page "${MAX_PAR_PER_PAGE}" \
  --concurrency "${CONCURRENCY}" \
  --batch-size "${BATCH_SIZE}" \
  --drop-disambiguation \
  --user-agent "${USER_AGENT}" \
  --log "${LOG_FILE}" \
  --log-level INFO

echo "[$(date -Is)] finished paragraph_miner; quick stats"

python - <<PY
import json
from pathlib import Path
from statistics import median

paras_path = Path("${OUT_PARAS}")
errs_path = Path("${OUT_ERR}")

def count_lines(p: Path) -> int:
    return sum(1 for _ in p.open()) if p.exists() else 0

def quantiles(xs, ps=(0, 25, 50, 75, 90, 99)):
    if not xs:
        return {}
    xs = sorted(xs)
    out = {}
    for p in ps:
        idx = int((len(xs) - 1) * p / 100)
        out[p] = xs[idx]
    return out

n_paras = count_lines(paras_path)
n_errs = count_lines(errs_path)
print(f"paras={n_paras} errors={n_errs}")

lengths = []
if paras_path.exists():
    with paras_path.open() as f:
        for line in f:
            try:
                obj = json.loads(line)
            except Exception:
                continue
            l = obj.get("para_char_len")
            if isinstance(l, int):
                lengths.append(l)

if lengths:
    qs = quantiles(lengths)
    print("para_char_len quantiles:", qs)
else:
    print("para_char_len quantiles: no data")
PY

echo "[$(date -Is)] done."
