# Enoki: Multi-Level Hallucination Detection

![Enoki OpenIE](assets/enoki-openie-banner.png)

Enoki is an end-to-end pipeline for detecting hallucinations in generated
text. It extracts relational facts, verifies them against evidence, and maps
unsupported facts back to the corresponding text spans.

Choose one of three fact-extraction backends for the same detection pipeline:

| Backend | Best for | Additional setup |
| --- | --- | --- |
| **Enoki-Encoder** | Fast local neural extraction | [`s-nlp/enoki-openie-encoder`](https://huggingface.co/s-nlp/enoki-openie-encoder) or a local exported model |
| **Enoki-Rules** | Deterministic, local extraction | `en_core_web_trf` spaCy model |
| **Enoki-LLM** | Flexible extraction through an OpenAI-compatible API | API credentials |

## Quick start

Clone the repository and install all Enoki components:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Enoki-Encoder is the default and uses the published Hugging Face model. Enoki-Rules
needs the spaCy model; Enoki-LLM needs an OpenAI-compatible API key.

## Detect hallucinations in your own text

Give Enoki an evidence `context` and a generated `answer`. It extracts facts
from the answer and verifies each one against the context.

```python
from enoki import EnokiPipeline


context = "Apple acquired Beats Electronics in 2014 for $3 billion."
answer = "Apple acquired Beats Electronics in 2015 for $3 billion."

enoki = EnokiPipeline()
print(enoki.detect(context=context, answer=answer))
# [{"span": "2015", "fact": {"subject": "Apple", "predicate": "acquired in",
#   "object": "2015"}, "probability": 0.97}]
```

Each result has plain-text `span`, character offsets, a structured `fact`
triplet, and NLI `probability`. By default, Enoki returns only facts with
`probability > 0.5`; pass `return_all=True` to inspect every scored fact.
Choose a backend by changing only pipeline construction:

### Enoki-Encoder

Enoki-Encoder is a trainable, non-generative OpenIE extractor. It uses
Iterative Grid Labeling (IGL) with a ModernBERT-large encoder; Hungarian
matching makes supervision permutation-invariant across unordered incremental
fact rows. It is the fast local neural option when an LLM is unnecessary.

Use the published Hugging Face model (the default), another Hugging Face ID,
or a local model directory exported by `enoki train encoder`:

```python
EnokiPipeline().detect(context=context, answer=answer)
EnokiPipeline(model="s-nlp/enoki-openie-encoder").detect(
    context=context, answer=answer
)
EnokiPipeline(model="models/enoki-encoder").detect(
    context=context, answer=answer
)
```

### Enoki-Rules

Enoki-Rules is a deterministic, training-free OpenIE backend with 35
dependency-parse rules over spaCy. Its rule library is refined through an
agent-assisted, automatically validated loop; inference itself runs only the
resulting lightweight heuristics.

Install the spaCy model once:

```bash
python -m spacy download en_core_web_trf
```

```python
enoki = EnokiPipeline(method="rules")
enoki.detect(context=context, answer=answer)
```

### Enoki-LLM

Enoki-LLM uses a CycleOIE-style prompting method. The prompt is extended with
instructions for incremental, text-anchored fact decomposition, so finer
unsupported spans can be verified separately.

Enoki calls the configured OpenAI-compatible API while extracting facts:

```bash
export OPENAI_API_KEY="..."
# export OPENAI_BASE_URL="https://your-endpoint.example/v1"  # optional
```

```python
enoki = EnokiPipeline(method="llm", model="gpt-4o")
enoki.detect(context=context, answer=answer)
```

By default detection uses the local ModernBERT NLI verifier. Pass
`nli_method="alignscore"` (or another supported verifier) to `detect` to
change it.

## Extract facts separately

You can also use Enoki as an OpenIE extractor without evidence verification or
hallucination scoring:

```bash
enoki extract \
  --method encoder \
  --text "Apple acquired Beats Electronics in 2014."
```

Change `--method` to `rules` or `llm` to select another backend. All methods
emit the same JSON fields: `text`, `triples`, `subject`, `predicate`, `object`,
and `confidence`. Pass `--input sentences.txt` for one input per line,
`--output triples.json` to save the result, or pipe text through stdin.

### Python API

```python
from enoki import EnokiPipeline

pipeline = EnokiPipeline()
results = pipeline.extract(
    "Apple acquired Beats Electronics for $3 billion in 2014."
)
print(results)
```

Example output:

```python
[
    {
        "text": "Apple acquired Beats Electronics for $3 billion in 2014.",
        "triples": [
            {
                "subject": "Apple",
                "predicate": "acquired",
                "object": "Beats Electronics",
                "confidence": 0.969,
            },
            {
                "subject": "Apple",
                "predicate": "acquired for",
                "object": "$3 billion",
                "confidence": 0.934,
            },
        ],
    }
]
```

Change `method` to `"llm"` or `"rules"`; the result schema remains the same.
`confidence` is the extraction confidence, not a hallucination probability.
Enoki-LLM returns `None` because its generated triples do not have a calibrated
extraction score.

## Train Enoki-Encoder

Training remains available through the same CLI:

```bash
enoki train encoder \
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

```bash
enoki evaluate sentence \
  --dataset factcheckbench \
  --extractor-method enoki-rules

enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-encoder \
  --encoder-model s-nlp/enoki-openie-encoder
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
