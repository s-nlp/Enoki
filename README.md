# Enoki: Multi-Level Hallucination Detection

![Enoki OpenIE](assets/enoki-openie-banner.png)

Enoki is an end-to-end pipeline for detecting hallucinations in generated
text. It extracts relational facts, verifies them against evidence, and maps
unsupported facts back to the corresponding text spans.

Choose one of three fact-extraction backends for the same detection pipeline:

| Backend | Best for | Additional setup |
| --- | --- | --- |
| **Enoki-Rules** | Deterministic, local extraction | `en_core_web_trf` spaCy model |
| **Enoki-LLM** | Flexible extraction through an OpenAI-compatible API | API credentials |
| **Enoki-Encoder** | Fast local neural extraction | Trained encoder checkpoint |

## Quick start

Clone the repository and install all Enoki components:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m spacy download en_core_web_trf
```

The spaCy model is needed only by Enoki-Rules. To use Enoki-LLM, also set an
OpenAI-compatible API key; Enoki-Encoder requires a trained checkpoint.

## Detect hallucinations

The `evaluate` commands run the complete workflow: extract facts from an
answer, verify them against its evidence, and report hallucination predictions.
The extractor is the only interchangeable part.

### Enoki-Rules

```bash
enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-rules
```

### Enoki-LLM

Enoki calls the configured OpenAI-compatible API while evaluating:

```bash
export OPENAI_API_KEY="..."
# export OPENAI_BASE_URL="https://your-endpoint.example/v1"  # optional

enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-llm \
  --llm-model gpt-4o
```

### Enoki-Encoder

Pass the path to a trained Enoki-Encoder checkpoint:

```bash
enoki evaluate span \
  --dataset ragtruth \
  --extractor-method enoki-encoder \
  --checkpoint checkpoints/best.ckpt
```

Use `enoki evaluate --help` and the level-specific `--help` commands to see
supported datasets, NLI backends, caching, and output options.

## Extract facts separately

You can also use Enoki as an OpenIE extractor without evidence verification or
hallucination scoring:

```bash
enoki extract \
  --method rules \
  --text "Apple acquired Beats Electronics in 2014."
```

Change `--method` to `encoder` or `llm` to select another backend. All methods
emit the same JSON fields: `text`, `triples`, `subject`, `predicate`, `object`,
and `confidence`. Pass `--input sentences.txt` for one input per line,
`--output triples.json` to save the result, or pipe text through stdin.

### Python API

```python
from enoki import EnokiPipeline

pipeline = EnokiPipeline(method="encoder")
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
  --out checkpoints/
```

Fine-tune from an existing checkpoint with `--checkpoint`. The training CLI
reads OIE4-style label files. Run `enoki train encoder --help` for all training
and resume options. Local Hugging Face export directories and publication
tooling are intentionally excluded from Git.

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
  --checkpoint checkpoints/best.ckpt
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
