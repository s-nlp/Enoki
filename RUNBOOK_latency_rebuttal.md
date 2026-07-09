# Rebuttal experiment runbook: latency vs. single-step LLM detector

Answers the reviewer's comment: *"the overhead of the three-stage pipeline (Extraction + Verification + Mapping) is not justified"* — on **span-level RAGTruth**, comparing a single-step few-shot LLM baseline against **Enoki-LLM**, **Enoki-Rule**, and **Enoki-Encoder**, all on **Qwen/Qwen3.6-35B-A3B**.

## 0. What I found / what I changed

The repo you gave me (`results_anah` branch) only has **Enoki-Rule** extraction (`fact_extractor/`) plus NLI verifiers (ModernBERT / AlignScore / Qwen). **Enoki-LLM** and **Enoki-Encoder** extraction backends don't exist there — they only exist in the full implementation on `origin/main` (your local `main` ref was stale; the real `origin/main` has them, merged in from an `enoki-v2` branch). That branch, however, had *dropped* the Qwen LLM verifier when it rewrote `nli/`.

I merged the two on a new branch **`rebuttal-latency`** (based on `results_anah` + `origin/main`), restored the Qwen verifier, and added two new scripts. Everything is packed into a git bundle sitting in your project folder:

```
enoki-rebuttal-latency.bundle
```

**Pull it into your real repo (run this on your machine, not in this sandbox):**

```bash
cd /path/to/Enoki
git fetch enoki-rebuttal-latency.bundle rebuttal-latency:rebuttal-latency
git checkout rebuttal-latency
git log --oneline -6   # sanity check
```

This branch contains, on top of your `results_anah` work:

1. **Merge of `origin/main`** — brings in `fact_extractor/enoki_llm_extractor.py` (Enoki-LLM / CycleOIE), `fact_extractor/enoki_encoder_extractor.py` + `model/` (Enoki-Encoder / IGL architecture + trainer), `fact_extractor/enoki_rules/` (the 35-rule library described in Appendix H), `factextractor_backend.py` (LLM-extraction runner with built-in FLOPs/latency tracking), and the new `enoki_cli.py` with `--extractor-method {enoki_rules,cycleoie,enoki_encoder}`.
2. **Restored Qwen LLM verifier** — `nli/llm_nli.py` + `nli/alignscore_nli.py` brought back, wired into `get_nli_checker()` as `qwen_06b` / `qwen_4b` / `qwen_8b` / `alignscore` / a new generic `llm` method (pass any HF repo id).
3. **`singlestep_ragtruth_baseline.py`** (new) — the single-step few-shot baseline.
4. **`measure_pipeline_latency.py`** (new) — reproduces the paper's Table 10/11 methodology (avg claims / extract_time / verify_time / total_time **per sentence**) for any of the three Enoki extraction backends.
5. `requirements.txt` fixed (was missing `openai`, `lightning`, `scipy`, all needed by the code that got merged in).

I verified: no merge conflicts, no leftover conflict markers, every tracked `.py` file compiles (`py_compile`). I could not run anything end-to-end here — no GPU in this sandbox.

## 1. Decisions I made (flag these in the rebuttal text)

| Question | Decision | Why |
|---|---|---|
| Enoki-LLM extraction | Use the real CycleOIE-incremental prompt (Appendix K), already implemented in `factextractor_backend.py` | Matches the paper's actual design; needed so the "extraction is the bottleneck" story (Table 10 obs. #2) actually reproduces |
| Enoki-Encoder | Train it — the IGL architecture, trainer, and label data (`data/enoki_encoder_train/{train,val}_labels`) all exist on the merged branch, no need to build from scratch | Turned out to be much less work than expected once merged |
| FLOPs for Qwen3.6-35B-A3B (MoE, 35B total / ~3B active) | Use **active params (~3e9)** in `--model-params` / `--model-params` flags, not 35e9 | Matches actual compute; note this explicitly in the paper text so it doesn't look like you're hiding behind MoE math |
| Single-step baseline prompt | **Updated 2026-07-09**: using the exact "ZS RAGTruth Prompt" text you supplied — zero-shot, single user turn (no system message, no few-shot examples), output is `{"hallucination list": [...]}` | Supersedes the earlier placeholder few-shot reconstruction (see git history) now that the real published prompt text is available |

**Two bugs found and fixed while wiring the real prompt in** (2026-07-09, both on `rebuttal-latency`):

1. `evaluation/dataset_loaders.py::load_ragtruth_dataset` never surfaced the question at all — the HF dataset column is named `query` (confirmed against `baselines/ragtruth_utils.py`'s docstring + a sample file), but the loader's returned dict had no `question`/`query` key whatsoever. `singlestep_ragtruth_baseline.py` was doing `row.get("question", "")`, which silently returned `""` for all 900 rows — the `{question}` slot in every single prompt sent so far was empty. Fixed by adding `"question": row.get("query", "")` to the loader. This also affects `enoki_cli.py` and `measure_pipeline_latency.py` since they import the same loader (worth double-checking whether Enoki's own extraction path needs the question anywhere — it currently doesn't appear to use it).
2. `singlestep_ragtruth_baseline.py`'s output parser (`parse_span_lines`) expected one bare span per line + a `NONE` sentinel, which matched the old placeholder prompt's output format, not the real prompt's `{"hallucination list": [...]}` JSON format. Replaced with `parse_hallucination_json`, which handles clean JSON, markdown-fenced JSON, extra prose around the dict, and a regex fallback for slightly malformed output (incl. a repair pass for raw unescaped newlines inside string values, which otherwise silently zero out every span for that example).

**If you already had a run in flight using the old code**: kill it and delete/rename `predictions/singlestep_ragtruth.jsonl` before restarting. The script resumes by skipping ids already present in the output file — if you don't clear it, rows scored under the old (wrong-prompt, no-question, line-based-parsing) code will silently stay mixed in with rows scored under the fixed code, and you won't be able to tell them apart in the final numbers.

## 2. Environment setup (on your GPU machine)

```bash
cd /path/to/Enoki
git checkout rebuttal-latency
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

## 3. Serve Qwen3.6-35B-A3B via vLLM (no container)

```bash
# terminal 1 — leave running
vllm serve Qwen/Qwen3.6-35B-A3B \
    --port 8000 \
    --tensor-parallel-size <N_GPUS>   # size to your box

# terminal 2
export OPENAI_API_KEY=dummy
export OPENAI_BASE_URL=http://moderation-ingress-controller.moderation.k8s.moderation-xs:18089/v1
```
<!-- http://localhost:8000/v1 -->

This single server backs **both** the single-step baseline and Enoki-LLM's extraction step (both use the OpenAI-compatible `factextractor_backend.py` client plumbing), so the served model is identical across the two comparisons.

The Qwen NLI verifier (`nli/llm_nli.py`, used for `--nli-method llm` / `qwen_8b` etc.) loads vLLM **in-process** instead (`vllm.LLM(...)`), so it does **not** go through the server above — don't run it in the same process as anything else holding GPU memory; run verification passes separately from the server, or size `--gpu-memory-utilization` down on both sides if you need them concurrently.

## 4. Baseline: single-step few-shot LLM

```bash
python singlestep_ragtruth_baseline.py \
    --model Qwen/Qwen3.6-35B-A3B \
    --model-params 3e9 \
    --output predictions/singlestep_ragtruth.jsonl \
    --workers 8
```

Prints Span Coverage F1 + avg latency/FLOPs per example at the end (also re-runnable standalone with `--score-only`).

## 5. Enoki-LLM (CycleOIE extraction on Qwen)

**Step 1 — extract triplets** (hits the vLLM server from step 3; records per-sentence extract time/tokens/FLOPs):

```bash
python enoki_cli.py extract-triplets \
    --dataset ragtruth \
    --output data/pre_extracted/ragtruth_test.jsonl \
    --model Qwen/Qwen3.6-35B-A3B \
    --prompt incremental \
    --model-params 3e9 \
    --save-sentence-metrics \
    --workers 8
```

**Step 2 — evaluate accuracy** (span-level, RAGTruth):

```bash
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method cycleoie \
    --pre-extracted-facts-file data/pre_extracted/ragtruth_test.jsonl \
    --method modernbert
```

Swap `--method modernbert` → `--method llm` (with `VLLM_MODEL=Qwen/Qwen3.6-35B-A3B` exported, or edit `nli/utils.py`'s `get_nli_checker` call site once to pass `model=`) if you also want the **Qwen verifier** instead of ModernBERT for this row — the paper's main tables always verify with ModernBERT, so keep both if you want an ablation like Appendix D (Table 8).

**Step 3 — latency** (verify-stage timing measured live; extract-stage timing read from step 1's saved metrics):

```bash
python measure_pipeline_latency.py \
    --extractor-method cycleoie \
    --pre-extracted-facts-file data/pre_extracted/ragtruth_test.jsonl \
    --nli-method modernbert \
    --n-samples 200 \
    --output predictions/latency_enoki_llm.csv
```

## 6. Enoki-Rule

```bash
# accuracy
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method enoki_rules \
    --method modernbert

# latency
python measure_pipeline_latency.py \
    --extractor-method enoki_rules \
    --nli-method modernbert \
    --n-samples 200 \
    --output predictions/latency_enoki_rule.csv
```

No GPU needed for extraction here (pure spaCy + rules); verification still needs a GPU for ModernBERT.

## 7. Enoki-Encoder (train, then evaluate)

**Train** (label data already included in the merged branch, ~516k/26k lines under `data/enoki_encoder_train/`):

```bash
python enoki_cli.py train encoder \
    --train data/enoki_encoder_train/train_labels \
    --dev   data/enoki_encoder_train/val_labels \
    --model answerdotai/ModernBERT-large \
    --epochs 15 --batch-size 32 \
    --out checkpoints/
```

(Note: the README says `data/enoki-encoder_train` with a hyphen — the actual directory on disk uses an underscore, `data/enoki_encoder_train`. Use the underscore version.)

**Evaluate + measure latency** against the resulting `checkpoints/best.ckpt` (or whatever the checkpoint callback names it — check `checkpoints/` after training):

```bash
python enoki_cli.py evaluate span \
    --dataset ragtruth \
    --extractor-method enoki_encoder \
    --checkpoint checkpoints/best.ckpt \
    --method modernbert

python measure_pipeline_latency.py \
    --extractor-method enoki_encoder \
    --checkpoint checkpoints/best.ckpt \
    --nli-method modernbert \
    --n-samples 200 \
    --output predictions/latency_enoki_encoder.csv
```

Training on ~516k label lines will take a while — if you want to sanity-check the pipeline first, add `--epochs 1` and a truncated label file.

## 8. Final comparison table

Once all four CSV/JSONL outputs exist, pull the numbers together:

| Method | Span Coverage F1 | Avg total latency / sentence | Avg FLOPs / sentence |
|---|---|---|---|
| Single-step (Qwen3.6-35B-A3B, few-shot) | from `singlestep_ragtruth_baseline.py` stdout (per-example, not per-sentence — see note below) | — | — |
| Enoki-LLM (Qwen3.6-35B-A3B) | from step 5.2 | from `latency_enoki_llm.csv` | from same file |
| Enoki-Rule | from step 6 | from `latency_enoki_rule.csv` | ~0 (rule-based, no LLM in extraction) |
| Enoki-Encoder | from step 7 | from `latency_enoki_encoder.csv` | ~0 (395M-param encoder, no LLM) |

**Important asymmetry to call out explicitly in the rebuttal text:** the single-step baseline makes **one call per RAGTruth response** (its natural granularity), while Enoki and `measure_pipeline_latency.py` report **per sentence**. To compare them on equal footing, either (a) divide the single-step baseline's per-example latency by that example's sentence count (use `factextractor_backend.split_sentences_with_spans` for a consistent count), or (b) sum Enoki's per-sentence latency across all sentences in an example and compare per-example totals. Pick one and state it — don't silently mix granularities, that's exactly the kind of thing a reviewer will catch.

## 9. Expected story (per your hypothesis)

- Enoki-LLM: likely *worse* latency/FLOPs than the single-step baseline, since it makes one LLM call **per sentence** for extraction plus (optionally) a second LLM call for verification, vs. the baseline's one call per whole response. This is fine to show and matches paper Table 10 obs. #2 ("Enoki-LLM shifts most cost into extraction").
- Enoki-Encoder: should land far below both LLM-based methods on latency/FLOPs (395M-param single forward pass per fact, no autoregressive generation) while staying close to Enoki-LLM on F1 — that's the "distillation paid off" result.
- Enoki-Rule: fastest of all (no learned model in the extraction stage at all).

If Enoki-LLM comes out *faster* than the single-step baseline in your run, that's still a valid and interesting result — just report what you measure.
