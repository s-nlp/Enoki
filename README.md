# Enoki: Multi-Level Hallucination Detection

[Paper: *Enoki: Efficient Multi-Level Hallucination Detection*](https://huggingface.co/papers/2609.00581)

![Enoki OpenIE](assets/enoki-openie-banner.png)

Enoki is an end-to-end pipeline for detecting hallucinations in generated
text. It extracts relational facts, verifies them against evidence, and maps
unsupported facts back to the corresponding text spans.

### From an unsupported fact to the words behind it

Example adapted from Figure 1 of the paper (illustration, not a recorded model run):

**Context:** Lucien Tesnière was born in Mont-Saint-Aignan, France, in 1893.

**Answer:** Tesnière was born in **Montpellier**, in 1893.

| Extracted fact | Evidence check | Highlight |
| --- | --- | --- |
| Tesnière · was born · in 1893 | Supported | — |
| Tesnière · was born · in Montpellier | Unsupported | **Montpellier** |

The supported year stays intact. Each localized result includes the full fact
used for verification and character offsets into the original answer.

Choose one of three fact-extraction backends for the same detection pipeline:

| Backend | Best for | Additional setup |
| --- | --- | --- |
| **Enoki-Encoder** | Fast local neural extraction | [`s-nlp/enoki-openie-encoder`](https://huggingface.co/s-nlp/enoki-openie-encoder) or a local exported model |
| **Enoki-Rules** | Deterministic, local extraction | `en_core_web_trf` spaCy model |
| **Enoki-LLM** | Flexible extraction through an OpenAI-compatible API | API credentials |

## Quick start

Install [Poetry 2.2+](https://python-poetry.org/docs/#installation), then clone
the repository and install all dependencies:

```bash
git clone https://github.com/s-nlp/Enoki.git
cd Enoki
poetry install --all-extras
poetry run enoki --help
```

## Detect hallucinations in your own text

### Enoki Encoder

Give Enoki an evidence `context` and a generated `answer`. It extracts facts
from the answer and verifies each one against the context.

```python
from enoki import EnokiPipeline


context = "Apple acquired Beats Electronics in 2014 for $3 billion."
answer = "Apple acquired Beats Electronics in 2015 for $3 billion."

enoki = EnokiPipeline(model="s-nlp/enoki-openie-encoder")
report = enoki.detect(context=context, answer=answer, return_stats=True)
print(report["results"])
print(report["stats"])
```

```text
print(report["results"])
[{"span": "2015", "start": 36, "end": 40,
  "fact": {"subject": "Apple", "predicate": "acquired in",
           "object": "Beats Electronics 2015"},
  "probability": 0.9625091552734375}]

print(report["stats"])
{"sentences_total": 1, "sentences_with_checked_facts": 1,
 "sentences_without_checked_facts": 0, "facts_extracted": 4,
 "facts_checked": 4, "facts_skipped": 0, "facts_supported": 3,
 "facts_unsupported": 1, "facts_unlocalized": 0,
 "facts_localized_unsupported": 1, "facts_suppressed_by_base": 0,
 "threshold": 0.5, "status": "checked"}
```

### Enoki Rules

Enoki-Rules is a deterministic, training-free OpenIE backend with 35
dependency-parse rules over spaCy. Its rule library is refined through an
agent-assisted, automatically validated loop; inference itself runs only the
resulting lightweight heuristics.

Download the spaCy model once:

```bash
poetry run python -m spacy download en_core_web_trf
```

```python
enoki = EnokiPipeline(method="rules")
print(enoki.detect(context=context, answer=answer))
```

```text
[{"span": "2015", "start": 36, "end": 40,
  "fact": {"subject": "Apple",
           "predicate": "acquired Beats Electronics in", "object": "2015"},
  "probability": 0.9926090240478516}]
```

PyTorch may print a `FutureWarning` about `torch.jit.script`; it does not affect
the result.

### Enoki LLM

Enoki-LLM uses a CycleOIE-style prompting method. Its default is the
[incremental prompt](fact_extractor/prompts/incremental.txt), which adds
instructions for text-anchored fact decomposition so finer unsupported spans
can be verified separately. The baseline [original prompt](fact_extractor/prompts/original.txt)
is also available.

Enoki calls the configured OpenAI-compatible API while extracting facts:

```bash
export OPENAI_API_KEY="..."
# export OPENAI_BASE_URL="https://your-endpoint.example/v1"  # optional
```

```python
enoki = EnokiPipeline(method="llm", model="gpt-5")
print(enoki.detect(context=context, answer=answer))
```

## Train Enoki-Encoder

Training remains available through the same CLI:

```bash
poetry run enoki train encoder \
  --train data/enoki_encoder_train/train_labels \
  --dev data/enoki_encoder_train/val_labels \
  --model answerdotai/ModernBERT-large \
  --epochs 15 \
  --batch-size 32 \
  --out models/
```

Training writes a portable model to `models/enoki-encoder`; pass that directory
to `EnokiPipeline(method="encoder", model=...)` or
`enoki evaluate ... --encoder-model ...`. Use `--checkpoint` only to resume a
training run. The training CLI reads OIE4-style label files.

## Validate on benchmarks

### Entity-level results: HalluEntity

Table 2 of the paper evaluates localized semantic units on HalluEntity.
Enoki-LLM reaches **55.09 AUPRC**, **+15.32 points** over MinIE, the strongest
external baseline by AUPRC in that table. All values below are percentages;
higher is better. These are published paper results, not a new evaluation of
this checkout.

| Method | Extractor | AUROC | AUPRC |
| --- | --- | ---: | ---: |
| **Enoki-LLM** | GPT-OSS-120B | **79.70** | **55.09** |
| Enoki-Rules | Rule-based | 76.41 | 46.81 |
| Enoki-Encoder | ModernBERT-large | 70.47 | 44.25 |
| OpenIE MinIE | MinIE | 67.14 | 39.77 |
| ZS RAGTruth Prompt | GPT-5.2 | 77.63 | 36.63 |
| lettucedetect | ModernBERT-large | 68.61 | 35.70 |

### Span-level results

Span Coverage F1 (%) from [Table 3 of the paper](https://arxiv.org/abs/2609.00581);
higher is better, and bold marks the best displayed result for each benchmark.
Selected strong baselines from the same table are included for context.

| Method | Extractor | MuSHROOM | RAGTruth | PsiloQA |
| --- | --- | ---: | ---: | ---: |
| **Enoki-LLM** | GPT-OSS-120B | **52.07** | 37.32 | **71.15** |
| **Enoki-Rules** | Rule-based | 49.18 | 27.87 | 65.73 |
| **Enoki-Encoder** | ModernBERT-large | 46.96 | 34.84 | 65.51 |
| FT on RAGTruth | Qwen3-8B | 4.33 | **42.20** | 23.81 |
| haldetect | ModernBERT-base-32k | 11.43 | 41.54 | 27.70 |
| ZS RAGTruth Prompt | GPT-5.2 | 5.67 | 35.97 | 39.17 |

### Run evaluations

Download the parser with `poetry run python -m spacy download en_core_web_trf`.

Start with span-level evaluation. Enoki supports all three span benchmarks:

```bash
poetry run enoki evaluate span --dataset psiloqa --extractor-method enoki-encoder
poetry run enoki evaluate span --dataset mushroom --extractor-method enoki-encoder
poetry run enoki evaluate span --dataset ragtruth --extractor-method enoki-encoder
```

Run entity-level evaluation on HalluEntity:

```bash
poetry run enoki evaluate entity --dataset halluentity --extractor-method enoki-encoder
```

For sentence-level evaluation, for example use FactCheckBench:

```bash
poetry run enoki evaluate sentence --dataset factcheckbench --extractor-method enoki-encoder
```

Additional reproducibility tools live in `scripts/benchmarks/`.

## EnokiQA

[EnokiQA](https://huggingface.co/datasets/s-nlp/EnokiQA)
is a long-form QA benchmark with claim-level verification labels aligned to
span-level localization. It contains 3,990 labeled examples across seven
generator models and 19,594 unlabeled question-answer-context triples.

## Repository map

```text
enoki/                 Public Python inference API and CLI
fact_extractor/        Encoder, LLM, and rules extraction implementations
model/                 Enoki-Encoder training model and data code
evaluation/            Sentence, span, and entity benchmark runners
nli/                   Verification backends
data/                  Included benchmark samples and encoder labels
scripts/               Benchmark and inference utilities
baselines/             External baseline integrations and result utilities
```

For source checkouts, the same CLI is available through `python -m enoki`.

## Citation

```bibtex
@misc{rykov2026enokiefficientmultilevelhallucination,
      title={Enoki: Efficient Multi-Level Hallucination Detection},
      author={Elisei Rykov and Timur Ionov and Nikolay Ivanov and Maksim Savkin and Maksim Makarenko and Alexander Panchenko and Vasily Konovalov and Julia Belikova},
      year={2026},
      eprint={2609.00581},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2609.00581},
}
```
