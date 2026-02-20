# Description
This script implements a VeriScore-like factuality evaluation pipeline on FactBench without any web retrieval.
Instead, it uses the evidence already stored in the dataset and performs a lightweight local passage selection per claim.

## FactBench
Inputs

Dataset: [factcheck-GPT-benchmark.jsonl](https://github.com/yuxiaw/Factcheck-GPT/blob/main/factcheck-GPT-benchmark.jsonl)

Each sample contains:
`prompt` – the question/query.

`sentences` – a dict like `{"sentence1": {...}, "sentence2": {...}, ...}` where each sentence item typically includes:
  
  `text` and/or `decontext`

  `sentence_factuality_label` – True / False / "NA"

evidence fields:

  `auto_evidence`

  `auto_evidence_url`

  `human_evidence`

VeriScore assets folder (`--veriscore_assets_dir`), comes from original VeriScore [repo](https://github.com/Yixiao-Song/VeriScore/tree/main/veriscore):

Required files:
* `prompt/extraction_qa_template.txt` – extraction prompt template
* `prompt/verification_instruction_trinary.txt` – verification instruction template
* `data/demos/few_shot_examples.jsonl` – few-shot demo examples



### Pipeline

1) Load sample + sentences
* Read `prompt` and `sentences`.
* Use `decontext` if available, else `text`.
* Skip `sentences` where `sentence_factuality_label == "NA"`.

2) Build evidence from dataset (**OFFLINE**)
* Collect evidence from all sentences: `auto_evidence`, `auto_evidence_url`, `human_evidence`.
* Flatten nested structures => one big evidence text.
* Split into passages (max_chars, max_passages).

3) Claim extraction (**VeriScore extraction prompt**)
* For each sentence from `sentences`, build VeriScore context snippet: 3 previous + current + 1 next (plus lead sentence if >5 total).
* Run `extraction_qa_template.txt` to produce either:
  * No verifiable claim. => FAIL `no_verifiable_claim` class, or 
  * a bullet list of claims under `Facts`: => parse into claim list.

4) Retrieval per claim (**OFFLINE**)
* For each extracted claim, select `top-k` passages by simple lexical overlap (fallback: first k).
* Format selected passages as `"Search result i"` blocks.

5) Verification (**VeriScore trinary verifier prompt**)
* For each claim, run verifier using:
  * few-shot prefix from `verification_instruction_trinary.txt` + `few_shot_examples.jsonl`
    * suffix: `Claim: ... + Search results ... + Your decision:`
* Parse output label: `Supported / Contradicted / Inconclusive`.
* If nothing parsed => FAIL (`unparsed_verification`).

6) Aggregate claim labels (sentence level)
* If any claim is Contradicted or Inconclusive => `sentence = not_supported`
* Else if any claim is Supported => `sentence = supported`
* Else => `ir`

7) Metrics (only for `gold != NA`)
* ALL metrics: failures are forced wrong (FAIL always counted as error).
* EVALUABLE metrics: only sentences with parsed verification labels.
* ROC-AUC: `risk = 1 - (#supported_claims / #parsed_claims)`.



## AUC calculate
Let's define:
* take all parsed claim labels for the sentence
* compute `p_supported = (# supported) / (# parsed claims)`
* define `risk_not_supported = 1 - p_supported`

So:
* if all claims supported => `risk = 0`
* if half supported, half not => `risk = 0.5`
* if none supported => `risk = 1`

That's continuous score is used for AUC, it depends on:
* how many claims got extracted
* how strict the verifier is (inconclusive counts as not-supported)
* parsing success


# RUN
Create a venv:

`python3.11 -m venv venv`

Install dependencies:

`pip install -r requirements.txt`


```bash
python veriscore_factbench_llama.py \
  --data_jsonl ../factcheck-GPT-benchmark.jsonl \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --extraction_template ./assets/veriscore/prompt/non_qa_template.txt \
  --verification_instruction_binary ./assets/veriscore/prompt/verification_instruction_binary_no_links.txt \
  --fewshot_jsonl ./assets/veriscore/data/demos/few_shot_examples.jsonl \
  --alpaca_template ./assets/veriscore/prompt/verification_alpaca_template.txt \
  --gpu_memory_utilization 0.5 \
  --out_dir ./veriscore_llama \
  --max_chars 1200 \
  --max_tokens_extract 1000 \
  --max_tokens_verify 1000 \
  --max_claims 3 \
  --claims_source llm \
  --max_items 2 # for tests
```


## FELM

This script implements a VeriScore-like factuality evaluation pipeline on **FELM**, using **only offline evidence** from the dataset (`ref_text`).  
There is **no web retrieval**. Instead, we run claim-conditioned local passage selection over `ref_text`, then verify each claim with the VeriScore trinary verifier prompt.

## Inputs

Dataset directory: `--felm_dir` (e.g., `./felm_with_ref_text`)

Each FELM example contains:
- `prompt` — question / instruction
- `segmented_response` — list of sentences/segments (sentence-level)
- `labels` — list of booleans aligned with `segmented_response`
  - `True` = supported
  - `False` = not_supported
- `ref_text` — offline evidence text (may be long; may be empty)
- `ref` — wikipedia URL list

## Run

```bash
python veriscore_felm_llama.py \
  --felm_dir ./felm_with_ref_text \
  --subset wk \
  --split test \
  --model VityaVitalich/Llama3.1-8b-instruct \
  --out_dir ./out_veriscore_felm_llama/wk \
  --veriscore_assets_dir ./assets/veriscore \
  --extraction_template ./assets/veriscore/prompt/non_qa_template.txt \
  --verification_instruction_binary ./assets/veriscore/prompt/verification_instruction_binary_no_links.txt \
  --fewshot_jsonl ./assets/veriscore/data/demos/few_shot_examples.jsonl \
  --alpaca_template ./assets/veriscore/prompt/verification_alpaca_template.txt \
  --claims_source llm \
  --skip_no_context
```