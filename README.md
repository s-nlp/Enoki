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
the repository and install encoder extraction + local NLI inference:

```bash
git clone https://github.com/s-nlp/Enoki.git
cd Enoki
poetry install
poetry run enoki --help
```

Enoki-Encoder is the default and uses the published Hugging Face model. Enoki-Rules
needs `poetry install -E rules` and the spaCy model; Enoki-LLM needs
`poetry install -E llm` and an OpenAI-compatible API key.

Training dependencies are separate: `poetry install -E train`.
Benchmark runners use `poetry install -E eval`. Extras can be combined, e.g.
`poetry install --all-extras`. The default install does not include
Lightning, benchmark datasets, FastCoref, Stanford OpenIE, or the metric package.
Local Lightning model exports also require `-E train`; the published encoder does not.

`pyproject.toml` is the single dependency manifest; `poetry.lock` records the
resolved versions. Run scripts with `poetry run python your_script.py`.
Poetry manages the virtual environment. For an existing environment, standard
`pip install -e .` and extras such as `pip install -e '.[train]'` still work.
The independent baseline environments under `baselines/` retain their own
upstream dependency snapshots; they are not installed with Enoki.

## Detect hallucinations in your own text

Give Enoki an evidence `context` and a generated `answer`. It extracts facts
from the answer and verifies each one against the context.

```python
from enoki import EnokiPipeline


context = "Apple acquired Beats Electronics in 2014 for $3 billion."
answer = "Apple acquired Beats Electronics in 2015 for $3 billion."

enoki = EnokiPipeline()
report = enoki.detect(context=context, answer=answer, return_stats=True)
print(report["results"])
print(report["stats"])
```

Each result has plain-text `span`, character offsets, a structured `fact`
triplet, and NLI `probability`. By default, Enoki returns only facts with
`probability > 0.5`; set `threshold=...` to change the cutoff. The score is
`neutral + contradiction`: lack of support in the supplied context, not a
calibrated probability of real-world falsity.

`return_stats=True` returns `results` and `stats`; without it, `detect()` keeps
returning a list. Statistics include sentences with/without checked facts,
extracted/checked/skipped facts, supported/unsupported facts, unlocalized facts,
and unsupported facts suppressed because a simpler base fact already failed.
`status` is `no_facts`, `partial`, or `checked`. **`checked` means the extracted
facts were checked, not that the extractor found every claim or the answer is correct.**
An empty result with `no_facts` is not a clean bill of health.

`return_all=True` includes supported and suppressed facts for inspection. A fact
without a unique localization has `span`, `start`, and `end` set to `None`.
One fact can yield multiple results when its selected tokens are discontiguous;
statistics count facts, not these output rows.

Both `extract()` and `detect()` split documents into sentences, preserving original
whitespace and global character offsets. Encoder/rules projections use native
word labels/spaCy offsets. Public detection and span evaluation replay share
the same token projection function, including when the threshold changes. Nested facts are verified in full; an unsupported
refinement is attributed to its added tokens only when its simpler base is
supported. Descendants of a failing base are suppressed in the default output.
LLM text is aligned within its source sentence; ambiguous mentions are not guessed.
The encoder rejects a sentence exceeding its token budget (128 for the published
model, including special tokens) with an actionable error instead of truncating it.

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
poetry install -E rules
poetry run python -m spacy download en_core_web_trf
```

```python
enoki = EnokiPipeline(method="rules")
enoki.detect(context=context, answer=answer)
```

### Enoki-LLM

Enoki-LLM uses a CycleOIE-style prompting method. Its default is the
[incremental prompt](fact_extractor/prompts/incremental.txt), which adds
instructions for text-anchored fact decomposition so finer unsupported spans
can be verified separately. The baseline [original prompt](fact_extractor/prompts/original.txt)
is also available.

Enoki calls the configured OpenAI-compatible API while extracting facts:

```bash
poetry install -E llm
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
poetry run enoki extract \
  --method encoder \
  --text "Apple acquired Beats Electronics in 2014."
```

Change `--method` to `rules` or `llm` to select another backend. All methods
emit the same JSON fields: `text`, `triples`, `subject`, `predicate`, `object`,
and `confidence`. Triples also include `spans` (per-role lists of half-open
character offsets), `sentence_start`, and `sentence_end`. Offsets always refer to
the original input; implicit predicate words may have no source tokens. Pass `--input sentences.txt` for one input per line,
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

Example output (core fields only; illustrative):

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
poetry install -E train
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

Install benchmark dependencies with `poetry install -E eval` and the parser
with `poetry run python -m spacy download en_core_web_trf`.

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
