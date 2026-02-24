# VeriScore (VeriScore-like baseline)

VeriScore-like factuality evaluation for **FactBench** and **FELM** using **offline** evidence only.

**Evidence sources**
- FactBench: `auto_evidence`, `auto_evidence_url`, `human_evidence` across the whole sample.
- FELM: `ref_text` (per example).

## Pipeline
1. Load samples and sentences, skip `sentence_factuality_label == NA`.
2. Build evidence passages (offline only).
3. Claim extraction.
4. Retrieval per claim using BM25 over passages (fallback to first `topk` if needed).
5. Verification using binary VeriScore prompt.
6. Aggregate per sentence.

**Claim sources**
- `llm` (default): extract claims with the template.
- `sentence`: use the sentence itself as a single claim.
- `dataset`: use `sentences[*].claims` when present (FactBench only).

**Verification labels**
- `Supported` or `Unsupported` (parsed into `supported` / `not_supported`).

## Assets
Provide either:
- `--veriscore_assets_dir` (auto-fills standard paths), or
- explicit `--extraction_template`, `--verification_instruction_binary`, `--fewshot_jsonl`.

Default asset filenames expected inside `--veriscore_assets_dir`:
- `prompt/non_qa_template.txt`
- `prompt/verification_instruction_binary.txt`
- `data/demos/few_shot_examples.jsonl`

## Outputs
`--out_dir` contains `metrics.json` and `segments_with_veriscore.jsonl`.

## Installation
```bash
python3.11 -m venv venv
pip install -r requirements.txt
```

## Run
### FactBench
```bash
python veriscore_run.py factbench \
  --data_jsonl ../data/factcheck-GPT-benchmark.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_factbench \
  --veriscore_assets_dir ./assets/veriscore \
  --claims_source llm \
  --max_claims 4 \
  --empty_claims_policy supported
```

### FELM
```bash
python veriscore_run.py felm \
  --felm_dir ../data/felm_with_ref_text \
  --subset wk \
  --split test \
  --model Qwen/Qwen2.5-7B-Instruct \
  --out_dir ./out/veriscore/qwen25_felm \
  --veriscore_assets_dir ./assets/veriscore \
  --skip_no_context \
  --empty_claims_policy supported
```
